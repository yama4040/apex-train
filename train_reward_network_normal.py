"""通常運転モードのみの評価NN学習スクリプト（比較実験用・2026-08-11）。

8月中間報告の指摘に対応するアブレーション。現行系（train_reward_network2.py）との違いは
**学習データのモードの扱いだけ**で、モデル構造・損失関数・サンプル重み・分割方法は
train_reward_network2 から import してそのまま使う（＝公平な比較のため実装差を作らない）。

現行系との違い:
  1. 学習に使う行を mode ラベルで絞る（既定: normal のみ）。
  2. モードone-hot は全行 normal（[1,0,0,0]）に固定して入力する。
  3. 保存先を *_normal.* にして、現行系のモデルを一切上書きしない。

使い方:
    python train_reward_network_normal.py
    python train_reward_network_normal.py --modes normal,anti_mid_stop
        └ 駅間停車防止モードの行も残す（遅延回復モードだけを外した比較をしたい場合）。
          ただしモード入力自体は normal 固定のままなので、
          「先行列車対応の教師データはあるが、モードの切り替えは行わない」系になる。
    python train_reward_network_normal.py --epochs 300

※ apex2 系のファイル（train_reward_network2.py 等）は一切変更していない。
"""
import argparse
import json

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import reward_features as rf
# モデル構造・損失・重み付けは現行系と完全に共有する（比較の公平性）
import train_reward_network2 as T


def main(csv_dir='train_reward_csv_direct',
         keep_modes=('normal',),
         model_path='direct_reward_model_normal.h5',
         gate_path='direct_reward_gate_normal.h5',
         scaler_path='direct_reward_scaler_normal.pkl',
         manifest_path='direct_reward_manifest_normal.json',
         plot_path='learning_curve_direct_normal.png',
         epochs=500):
    X_state, mode_onehot, y, state_cols = T.load_and_preprocess_data(csv_dir)

    # --- 1. モードラベルで行を絞る -------------------------------------------
    keep_idx = np.zeros(len(y), dtype=bool)
    for m in keep_modes:
        col = rf.MODE_CLASSES.index(m)
        keep_idx |= (mode_onehot[:, col] > 0.5)
    dropped = int((~keep_idx).sum())
    print(f"[通常運転モードのみ] 使用するモード: {list(keep_modes)}")
    print(f"  採用 {int(keep_idx.sum())}行 / 除外 {dropped}行（他モードのラベル行）")

    X_state = X_state[keep_idx]
    y = y[keep_idx]

    # --- 2. モードone-hotは全行 normal 固定 ----------------------------------
    normal_oh = rf.mode_to_onehot('normal').astype(np.float32)
    mode_onehot = np.tile(normal_oh, (len(y), 1))
    print(f"  モードone-hotは全行 normal に固定: {normal_oh.tolist()}")

    # --- 3. 以降の学習手順は現行系と同一 -------------------------------------
    state_dim = X_state.shape[1]
    print(f"状態特徴量次元数: {state_dim}, mode one-hot次元: {rf.MODE_DIM}, 評価NN入力次元: {state_dim + rf.MODE_DIM}")

    _, bin_idx_all = T.compute_bin_sample_weights(y)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=bin_idx_all)
    Xs_tr, Xs_te = X_state[idx_tr], X_state[idx_te]
    mo_tr, mo_te = mode_onehot[idx_tr], mode_onehot[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]
    print(f"学習データ数: {len(idx_tr)}, テストデータ数: {len(idx_te)}")

    scaler = StandardScaler()
    Xs_tr_scaled = scaler.fit_transform(Xs_tr)
    Xs_te_scaled = scaler.transform(Xs_te)
    X_train_scaled = np.hstack([Xs_tr_scaled, mo_tr]).astype(np.float32)
    X_test_scaled = np.hstack([Xs_te_scaled, mo_te]).astype(np.float32)

    import tensorflow as tf  # noqa: F401  (Kerasコールバックのため)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    stop_col_idx = state_cols.index(T.STOP_PHASE_COL)
    train_is_stop = Xs_tr[:, stop_col_idx] >= 0.5
    print(f"[サンプル重み] 停車完了フェーズ行: 学習{int(train_is_stop.sum())}件に重み×{T.STOP_PHASE_WEIGHT}を適用")

    # --- ゲート分類器（reward=0.0 か否か）---
    y_train_zero = (y_train.flatten() <= T.ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    y_test_zero = (y_test.flatten() <= T.ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    print(f"[ゲート分類器] 0.0ラベル: {int(y_train_zero.sum())}件 / 非0.0: {int(len(y_train_zero) - y_train_zero.sum())}件")

    gate_sample_weight = np.where(train_is_stop, T.STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    gate_sample_weight = gate_sample_weight / gate_sample_weight.mean()

    gate_model = T.build_gate_model(X_train_scaled.shape[1])
    print("ゲート分類器の学習を開始します...")
    gate_model.fit(X_train_scaled, y_train_zero, sample_weight=gate_sample_weight,
                   validation_data=(X_test_scaled, y_test_zero), epochs=epochs, batch_size=64,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)],
                   verbose=1)

    # --- 非0.0のみの回帰器 ---
    nonzero_train = y_train.flatten() > T.ZERO_THRESHOLD
    nonzero_test = y_test.flatten() > T.ZERO_THRESHOLD
    X_train_nz, y_train_nz = X_train_scaled[nonzero_train], y_train[nonzero_train]
    X_test_nz, y_test_nz = X_test_scaled[nonzero_test], y_test[nonzero_test]
    print(f"\n[回帰器] 学習データ数（非0.0のみ）: {X_train_nz.shape[0]}, テストデータ数: {X_test_nz.shape[0]}")

    w_nz, _ = T.compute_bin_sample_weights(y_train_nz)
    ceiling_mask = y_train_nz.flatten() >= T.CEILING_LABEL_THRESHOLD
    w_nz = w_nz * np.where(ceiling_mask, T.CEILING_LABEL_WEIGHT, 1.0).astype(np.float32)
    w_nz = w_nz / w_nz.mean()

    model = T.build_model(X_train_nz.shape[1])
    print("回帰器の学習を開始します...")
    history = model.fit(X_train_nz, y_train_nz, sample_weight=w_nz,
                        validation_data=(X_test_nz, y_test_nz), epochs=epochs, batch_size=64,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)],
                        verbose=1)

    T.plot_learning_curve(history, plot_path)

    # --- 合成モデルとしての評価（推論時と同じ合成ロジック）---
    gate_prob = gate_model.predict(X_test_scaled, verbose=0).flatten()
    reg_pred = model.predict(X_test_scaled, verbose=0).flatten()
    combined = np.where(gate_prob >= 0.5, 0.0, np.clip(reg_pred, 0.1, 1.0))
    y_test_flat = y_test.flatten()
    print(f"\n[合成モデル評価] テストMAE: {np.abs(combined - y_test_flat).mean():.4f}")
    zero_mask = y_test_flat <= T.ZERO_THRESHOLD
    if zero_mask.any():
        print(f"[合成モデル評価] ラベル0.0のうち予測0.25超（報酬リーク）: {(combined[zero_mask] > 0.25).mean():.1%} "
              f"(平均予測 {combined[zero_mask].mean():.3f})")
    if (~zero_mask).any():
        print(f"[合成モデル評価] 非0.0ラベルのMAE: {np.abs(combined[~zero_mask] - y_test_flat[~zero_mask]).mean():.4f}")

    model.save(model_path)
    gate_model.save(gate_path)
    joblib.dump(scaler, scaler_path)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'state_feature_cols': state_cols,
                   'mode_classes': list(rf.MODE_CLASSES),
                   'variant': 'normal_only',
                   'keep_modes': list(keep_modes)}, f, ensure_ascii=False, indent=2)
    print(f"\n回帰器('{model_path}')・ゲート('{gate_path}')・スケーラー('{scaler_path}')・"
          f"マニフェスト('{manifest_path}')を保存しました。")
    print("※ 現行系（direct_reward_model2.h5 等）は変更していません。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="通常運転モードのみの評価NNを学習する（比較実験用）")
    ap.add_argument("--csv-dir", default="train_reward_csv_direct")
    ap.add_argument("--modes", default="normal",
                    help="学習に使うmodeラベル（カンマ区切り）。既定 normal のみ")
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()
    main(csv_dir=args.csv_dir,
         keep_modes=tuple(m.strip() for m in args.modes.split(",") if m.strip()),
         epochs=args.epochs)
