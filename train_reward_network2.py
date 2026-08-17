import os
import glob
import argparse
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # 学習曲線をPNG保存するだけなので非対話バックエンドを使う。
# ※既定の qtagg のままだと、DISPLAYはあるがQtのxcbプラグインを初期化できない環境で
#   plt.figure() が Python例外ではなく SIGABRT でプロセスごと落ちる。本スクリプトは
#   plot_learning_curve() を model.save() より前に呼ぶため、学習結果が保存されずに失われる。
import matplotlib.pyplot as plt

import reward_features as rf  # 特徴量エンジニアリングの共有モジュール（学習・推論で一致）
import histogram_loss as hl   # HL-Gaussianの共有モジュール（学習・推論でビン定義を一致させる）

# ▼▼▼ 重み付けの効果検証時に結果がぶれないよう乱数シードを固定 ▼▼▼
np.random.seed(42)
tf.random.set_seed(42)


def custom_accuracy(y_true, y_pred):
    # LLMの評価値と予測値のズレが 0.15 以内であれば「正解(1)」とする
    tolerance = 0.15
    correct_predictions = tf.cast(tf.abs(y_true - y_pred) <= tolerance, tf.float32)
    return tf.reduce_mean(correct_predictions)


def load_and_preprocess_data(csv_dir):
    """CSVを読み込み、状態特徴量行列(X_state)・モードone-hot・報酬(y)・状態列名を返す。

    特徴量エンジニアリングは reward_features（共有モジュール）に一本化。
    評価NNの入力は [正規化した状態特徴量 | mode one-hot(MODE_DIM)] で、
    mode one-hot は正規化後に付加する（main側で結合）。
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")

    df_list, skipped_files = [], []
    for file in csv_files:
        temp_df = pd.read_csv(file, encoding='utf-8-sig')
        if 'required_speed' not in temp_df.columns:
            skipped_files.append(os.path.basename(file))
            continue
        df_list.append(temp_df)
    if skipped_files:
        print(f"[警告] 'required_speed'列が無い旧形式のため除外: {len(skipped_files)}件: {skipped_files}")
    if not df_list:
        raise ValueError(f"'{csv_dir}' 内に 'required_speed' 列を含む有効なCSVがありません。")

    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")

    # mode列（無ければnormal扱い＝旧データ後方互換）
    if 'mode' not in df.columns:
        df['mode'] = 'normal'
    df['mode'] = df['mode'].fillna('normal').replace('', 'normal')
    print("モード分布:", {c: int((df['mode'] == c).sum()) for c in rf.MODE_CLASSES if (df['mode'] == c).any()})

    # 状態特徴量（共有モジュール）
    X_state, state_cols = rf.build_state_matrix(df)
    # モードone-hot（教師強制: LLMが付与したmodeラベルを使用）
    mode_onehot = np.stack([rf.mode_to_onehot(m) for m in df['mode']]).astype(np.float32)
    y = df['reward'].astype(np.float32).values.reshape(-1, 1)

    return X_state, mode_onehot, y, state_cols


def build_trunk(input_dim, out_units, out_activation):
    """回帰器・ゲートで共通の幹（128-64-32）。出力層だけ差し替える。"""
    l2_reg = l2(1e-4)
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(32, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.1),
        Dense(out_units, activation=out_activation),
    ])


def build_model(input_dim):
    """旧来のスカラー回帰器（Huber）。--head scalar のときに使う。"""
    model = build_trunk(input_dim, 1, 'linear')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0), metrics=['mae', custom_accuracy])
    return model


def build_hl_gauss_model(input_dim, centers):
    """HL-Gaussian の回帰器（既定）。出力はビンのsoftmaxで、予測はビン中心の期待値。

    詳細と採用理由は histogram_loss.py のdocstring、および
    docs/報酬NN出力ヘッド比較レポート.html を参照。
    """
    model = build_trunk(input_dim, len(centers), 'softmax')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[hl.make_expected_mae(centers)])
    return model


def build_gate_model(input_dim):
    """2段階化（ハードルモデル）の1段目：「reward=0.0か否か」を判定する二値分類器。"""
    l2_reg = l2(1e-4)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(32, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.1),
        Dense(1, activation='sigmoid'),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def compute_bin_sample_weights(y):
    """0.1刻みのラベル区間ごとの出現頻度の逆数（sqrt緩和）を重みとして算出する。
    件数の少ない区間が損失関数上で過小評価されるのを防ぐ。重み平均が1.0になるよう正規化。"""
    bin_idx = np.clip(np.round(y.flatten() * 10).astype(np.int32), 0, 10)
    bin_counts = np.bincount(bin_idx, minlength=11).astype(np.float32)
    bin_counts[bin_counts == 0] = 1.0
    inv_freq = 1.0 / np.sqrt(bin_counts)
    sample_weight = inv_freq[bin_idx]
    sample_weight = sample_weight / sample_weight.mean()
    return sample_weight.astype(np.float32), bin_idx


def plot_learning_curve(history, out_path='learning_curve_direct.png', head='hl_gauss'):
    # ヘッドによって損失名とMAE指標名が変わる（HL-Gaussianは交差エントロピー＋expected_mae）
    loss_name = 'Cross-Entropy (HL-Gaussian)' if head == 'hl_gauss' else 'Huber'
    mae_key = 'expected_mae' if head == 'hl_gauss' else 'mae'
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label=f'Train Loss ({loss_name})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({loss_name})')
    plt.title(f'Model Loss ({loss_name})'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history[mae_key], label='Train MAE')
    plt.plot(history.history[f'val_{mae_key}'], label='Validation MAE')
    plt.title('Model Mean Absolute Error (MAE)'); plt.ylabel('MAE'); plt.xlabel('Epoch'); plt.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(out_path)


# 「reward=0.0」とみなすラベルの閾値（浮動小数の丸め誤差対策）
ZERO_THRESHOLD = 0.05

# 停車完了フェーズ行の損失重み係数（少数派＋回帰平滑化で頂点が潰れるため優先度を上げる）
STOP_PHASE_WEIGHT = 3.0
STOP_PHASE_COL = 'phase_駅停車完了（速度0km/h）'

# 停車完了フェーズの1.0（完璧な停止）ラベルの損失重み係数（巡航の1.0には適用しない）
CEILING_LABEL_WEIGHT = 2.5
CEILING_LABEL_THRESHOLD = 0.95


def main(csv_dir='train_reward_csv_direct',
         model_path='direct_reward_model2.h5',
         gate_path='direct_reward_gate2.h5',
         scaler_path='direct_reward_scaler2.pkl',
         manifest_path='direct_reward_manifest.json',
         plot_path='learning_curve_direct.png',
         epochs=500,
         head='hl_gauss',
         bins=hl.DEFAULT_BINS,
         guard_bins=hl.DEFAULT_GUARD,
         sigma_ratio=hl.DEFAULT_SIGMA_RATIO):
    X_state, mode_onehot, y, state_cols = load_and_preprocess_data(csv_dir)
    state_dim = X_state.shape[1]
    print(f"状態特徴量次元数: {state_dim}, mode one-hot次元: {rf.MODE_DIM}, 評価NN入力次元: {state_dim + rf.MODE_DIM}")

    # 0.1刻みのラベル区間で層化。状態・モード・報酬を同じインデックスで分割する。
    _, bin_idx_all = compute_bin_sample_weights(y)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=bin_idx_all)
    Xs_tr, Xs_te = X_state[idx_tr], X_state[idx_te]
    mo_tr, mo_te = mode_onehot[idx_tr], mode_onehot[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]
    print(f"学習データ数: {len(idx_tr)}, テストデータ数: {len(idx_te)}")

    # スケーラは状態特徴量のみでfit（mode one-hotは正規化後に付加）
    scaler = StandardScaler()
    Xs_tr_scaled = scaler.fit_transform(Xs_tr)
    Xs_te_scaled = scaler.transform(Xs_te)
    X_train_scaled = np.hstack([Xs_tr_scaled, mo_tr]).astype(np.float32)
    X_test_scaled = np.hstack([Xs_te_scaled, mo_te]).astype(np.float32)

    # ※回帰器のコールバックはヘッド確定後に作る（監視する指標名がヘッドで変わるため）

    # --- 停車完了フェーズ行の識別（サンプル重み用・未正規化の状態特徴量で判定）---
    stop_col_idx = state_cols.index(STOP_PHASE_COL)
    train_is_stop = Xs_tr[:, stop_col_idx] >= 0.5
    print(f"[サンプル重み] 停車完了フェーズ行: 学習{int(train_is_stop.sum())}件に重み×{STOP_PHASE_WEIGHT}を適用")

    # --- 1段目: ゲート分類器（全データで reward=0.0 か否か）---
    y_train_zero = (y_train.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    y_test_zero = (y_test.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    print(f"[ゲート分類器] 0.0ラベル: {int(y_train_zero.sum())}件 / 非0.0: {int(len(y_train_zero) - y_train_zero.sum())}件")

    gate_sample_weight = np.where(train_is_stop, STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    gate_sample_weight = gate_sample_weight / gate_sample_weight.mean()

    gate_model = build_gate_model(X_train_scaled.shape[1])
    gate_early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    gate_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    print("ゲート分類器の学習を開始します...")
    gate_model.fit(X_train_scaled, y_train_zero, sample_weight=gate_sample_weight,
                   validation_data=(X_test_scaled, y_test_zero), epochs=epochs, batch_size=64,
                   callbacks=[gate_early_stop, gate_reduce_lr], verbose=1)

    # --- 2段目: 非0.0のみの回帰器 ---
    nonzero_train = y_train.flatten() > ZERO_THRESHOLD
    nonzero_test = y_test.flatten() > ZERO_THRESHOLD
    X_train_nz = X_train_scaled[nonzero_train]
    y_train_nz = y_train[nonzero_train]
    X_test_nz = X_test_scaled[nonzero_test]
    y_test_nz = y_test[nonzero_test]
    print(f"\n[回帰器] 学習データ数（非0.0のみ）: {X_train_nz.shape[0]}, テストデータ数: {X_test_nz.shape[0]}")

    train_sample_weight_nz, _ = compute_bin_sample_weights(y_train_nz)
    train_sample_weight_nz = train_sample_weight_nz * np.where(
        train_is_stop[nonzero_train], STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    y_train_nz_flat = y_train_nz.flatten()
    ceiling_mask = (y_train_nz_flat >= CEILING_LABEL_THRESHOLD) & train_is_stop[nonzero_train]
    train_sample_weight_nz = train_sample_weight_nz * np.where(ceiling_mask, CEILING_LABEL_WEIGHT, 1.0).astype(np.float32)
    train_sample_weight_nz = train_sample_weight_nz / train_sample_weight_nz.mean()

    # --- 回帰器の構築（ヘッドの選択）---
    if head == 'hl_gauss':
        centers, width = hl.make_centers(bins, guard_bins)
        sigma = sigma_ratio * width
        print(f"\n[回帰器] HL-Gaussian ヘッド: ビン{len(centers)}個 "
              f"[{centers[0]:.3f}, {centers[-1]:.3f}] 幅={width:.4f} σ={sigma:.4f} 予備ビン={guard_bins}")
        # 教師を「ラベル中心の打ち切り正規分布」に変換する（学習前に一度だけ計算）
        target_train = hl.gaussian_bin_targets(y_train_nz.flatten(), centers, width, sigma)
        target_test = hl.gaussian_bin_targets(y_test_nz.flatten(), centers, width, sigma)
        model = build_hl_gauss_model(X_train_nz.shape[1], centers)
        monitor = 'val_expected_mae'
    else:
        print("\n[回帰器] スカラー回帰ヘッド（Huber）")
        centers = None
        target_train, target_test = y_train_nz, y_test_nz
        model = build_model(X_train_nz.shape[1])
        monitor = 'val_loss'
    # EarlyStopping/ReduceLROnPlateau の監視対象をヘッドに合わせる
    early_stop = EarlyStopping(monitor=monitor, patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    print("回帰器の学習を開始します...")
    history = model.fit(X_train_nz, target_train, sample_weight=train_sample_weight_nz,
                        validation_data=(X_test_nz, target_test), epochs=epochs, batch_size=64,
                        callbacks=[early_stop, reduce_lr], verbose=1)

    plot_learning_curve(history, plot_path, head)

    # --- 合成モデルとしての評価（推論時と同じ合成ロジック）---
    gate_prob = gate_model.predict(X_test_scaled, verbose=0).flatten()
    raw_out = model.predict(X_test_scaled, verbose=0)
    if head == 'hl_gauss':
        # 推論時と同一: reward = (1 - ゲート確率) * ビン中心の期待値（閾値・clip・roundなし）
        reg_pred = hl.expectation(raw_out, centers)
        combined = (1.0 - gate_prob) * reg_pred
    else:
        reg_pred = raw_out.flatten()
        combined = np.where(gate_prob >= 0.5, 0.0, np.clip(reg_pred, 0.1, 1.0))
    y_test_flat = y_test.flatten()
    mae = np.abs(combined - y_test_flat).mean()
    print(f"\n[合成モデル評価] テストMAE: {mae:.4f}")
    zero_mask = y_test_flat <= ZERO_THRESHOLD
    if zero_mask.any():
        leak = (combined[zero_mask] > 0.25).mean()
        print(f"[合成モデル評価] ラベル0.0のうち予測0.25超（報酬リーク）: {leak:.1%} (平均予測 {combined[zero_mask].mean():.3f})")
    if (~zero_mask).any():
        mae_nz = np.abs(combined[~zero_mask] - y_test_flat[~zero_mask]).mean()
        print(f"[合成モデル評価] 非0.0ラベルのMAE: {mae_nz:.4f}")

    # 回帰器の生出力の値域。スカラー回帰では値域外が多く、clipが破綻を隠していた
    # （2026-08-17 実測でテスト行の26.7%）。HL-Gaussianなら原理的に値域内に収まる。
    oor = ((reg_pred < 0.1) | (reg_pred > 1.0)).mean()
    print(f"[合成モデル評価] 回帰器の生出力: [{reg_pred.min():+.3f}, {reg_pred.max():+.3f}] 値域外={oor:.1%}")

    model.save(model_path)
    gate_model.save(gate_path)
    joblib.dump(scaler, scaler_path)
    manifest = {'state_feature_cols': state_cols,
                'mode_classes_onehot': rf.MODE_CLASSES,
                'state_dim': state_dim,
                'input_dim': state_dim + rf.MODE_DIM}
    # 推論側（direct_reward_predictor2.py）はこのブロックを読んでヘッドを判別する
    manifest['regressor_head'] = (hl.head_manifest(bins, guard_bins, sigma_ratio)
                                  if head == 'hl_gauss' else hl.scalar_head_manifest())
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"回帰器('{model_path}')・ゲート('{gate_path}')・スケーラー('{scaler_path}')・マニフェスト('{manifest_path}')を保存しました。")
    print(f"→ 回帰器ヘッド: {manifest['regressor_head']['head']} / "
          f"合成: {manifest['regressor_head']['composition']}")


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description='報酬予測NN（回帰）の学習。既定はHL-Gaussianヘッド（推奨構成）。',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', choices=['hl_gauss', 'scalar'], default='hl_gauss',
                    help='回帰器の出力ヘッド。hl_gauss=ビンsoftmax＋ガウス目標の交差エントロピー（既定・推奨）／'
                         'scalar=従来のHuber回帰（旧構成の再現用）')
    ap.add_argument('--bins', type=int, default=hl.DEFAULT_BINS,
                    help=f'HL-Gaussianの核ビン数（既定{hl.DEFAULT_BINS}＝幅0.05）')
    ap.add_argument('--guard-bins', type=int, default=hl.DEFAULT_GUARD,
                    help='ラベル範囲の外側に足す予備ビン数。0にすると端のラベルで教師分布の期待値がずれる（既定1）')
    ap.add_argument('--sigma-ratio', type=float, default=hl.DEFAULT_SIGMA_RATIO,
                    help=f'σ ÷ ビン幅（既定{hl.DEFAULT_SIGMA_RATIO}）')
    ap.add_argument('--csv-dir', default='train_reward_csv_direct')
    ap.add_argument('--epochs', type=int, default=500)
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(csv_dir=args.csv_dir, epochs=args.epochs, head=args.head,
         bins=args.bins, guard_bins=args.guard_bins, sigma_ratio=args.sigma_ratio)
