# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の報酬予測NN（蒸留）の学習。

既存 `train_reward_network2.py` は**一切変更しない**。成果物のファイル名も分離してあるため、
既存の `direct_reward_model2.h5` / `direct_reward_scaler2.pkl` / `direct_reward_manifest.json` を
上書きすることはない。

構成は既存の推奨構成を踏襲する。
  * 2段階（ハードルモデル）: ゲート分類器（reward=0.0か否か）＋ 非0.0のみの回帰器
  * 回帰器は **HL-Gaussian ヘッド**（ビンsoftmax＋打ち切り正規分布の教師＋交差エントロピー）
  * ビン定義は `histogram_loss.py` を学習・推論で共有する

複数駅間版で追加した点
  * **駅停車中（is_dwelling）の行にサンプル重みを与える。** 停車中のサンプルは走行中に比べて
    圧倒的に少なく、そのまま学習すると発車判断が平均に引き寄せられて潰れる
    （ラベル不均衡は蒸留NNの校正を直接壊す）。
  * モードone-hotは5次元（hold_at_station を含む）。

使い方:
    python train_reward_network_ymulti.py
    python train_reward_network_ymulti.py --head scalar        # 旧構成（Huber回帰）の再現
    python train_reward_network_ymulti.py --csv-dir 別のディレクトリ
"""
import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
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
matplotlib.use("Agg")   # PNG保存のみ。既定バックエンドだとQt初期化失敗でプロセスごと落ちる環境がある
import matplotlib.pyplot as plt

import config_ymulti as CFG
import reward_features_ymulti as rf
import histogram_loss as hl

np.random.seed(42)
tf.random.set_seed(42)

ZERO_THRESHOLD = 0.05             # 「reward=0.0」とみなすラベルの閾値（丸め誤差対策）
STOP_PHASE_WEIGHT = 3.0           # 停車完了フェーズ行の損失重み（少数派＋平滑化で頂点が潰れるため）
STOP_PHASE_COL = "phase_駅停車完了（速度0km/h）"
DWELL_WEIGHT = 4.0                # 駅停車中（発車判断）行の損失重み
DWELL_COL = "is_dwelling"
CEILING_LABEL_WEIGHT = 2.5        # 停車完了フェーズの満点ラベル（完璧な停止）の重み
CEILING_LABEL_THRESHOLD = 0.95


def expected_mae_metric(centers):
    return hl.make_expected_mae(centers)


def custom_accuracy(y_true, y_pred):
    """LLMの評価値とのズレが0.15以内なら正解とみなす指標（スカラー回帰用）"""
    return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= 0.15, tf.float32))


def load_data(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"ディレクトリ '{csv_dir}' にCSVがありません。"
            f"evaluate_csv_with_llm_ymulti.py の出力を配置してください。")
    print(f"{len(files)} 個のCSVを読み込みます...")
    dfs = []
    for p in files:
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "reward" not in d.columns:
            print(f"  [除外] reward列が無い: {os.path.basename(p)}")
            continue
        d = d[d["reward"].notna() & (d["reward"].astype(str) != "")]
        if len(d):
            dfs.append(d)
    if not dfs:
        raise ValueError(f"'{csv_dir}' に評価済み（reward列あり）のCSVがありません。")
    df = pd.concat(dfs, ignore_index=True)
    print(f"合計データ数: {len(df)} 行")

    if "mode" not in df.columns:
        df["mode"] = "normal"
    df["mode"] = df["mode"].fillna("normal").replace("", "normal")
    dist = {c: int((df["mode"] == c).sum()) for c in rf.MODE_CLASSES if (df["mode"] == c).any()}
    print("モード分布:", dist)
    if "is_dwelling" in df.columns:
        n_dwell = int((df["is_dwelling"].astype(float) >= 0.5).sum())
        print(f"駅停車中の行: {n_dwell} 行（{n_dwell/len(df)*100:.1f}%）")

    X_state, state_cols = rf.build_state_matrix(df)
    mode_onehot = np.stack([rf.mode_to_onehot(m) for m in df["mode"]]).astype(np.float32)
    y = df["reward"].astype(np.float32).values.reshape(-1, 1)
    return X_state, mode_onehot, y, state_cols


def build_trunk(input_dim, units, activation):
    reg = l2(1e-4)
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=reg), BatchNormalization(), Activation("relu"), Dropout(0.3),
        Dense(64, kernel_regularizer=reg), BatchNormalization(), Activation("relu"), Dropout(0.2),
        Dense(32, kernel_regularizer=reg), BatchNormalization(), Activation("relu"), Dropout(0.1),
        Dense(units, activation=activation),
    ])


def build_regressor(input_dim, head, centers):
    if head == "hl_gauss":
        m = build_trunk(input_dim, len(centers), "softmax")
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=[expected_mae_metric(centers)])
    else:
        m = build_trunk(input_dim, 1, "linear")
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss=Huber(delta=1.0), metrics=["mae", custom_accuracy])
    return m


def build_gate(input_dim):
    m = build_trunk(input_dim, 1, "sigmoid")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def bin_sample_weights(y):
    """0.1刻みのラベル区間の出現頻度の逆数（sqrt緩和）。少数区間が損失で埋もれるのを防ぐ。"""
    idx = np.clip(np.round(y.flatten() * 10).astype(np.int32), 0, 10)
    counts = np.bincount(idx, minlength=11).astype(np.float32)
    counts[counts == 0] = 1.0
    w = (1.0 / np.sqrt(counts))[idx]
    return (w / w.mean()).astype(np.float32), idx


def plot_curve(history, path, head):
    loss_name = "Cross-Entropy (HL-Gaussian)" if head == "hl_gauss" else "Huber"
    mae_key = "expected_mae" if head == "hl_gauss" else "mae"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(f"Loss ({loss_name})"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history[mae_key], label="Train")
    plt.plot(history.history[f"val_{mae_key}"], label="Validation")
    plt.title("Mean Absolute Error"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(path)
    plt.close()


def main(csv_dir=CFG.TRAIN_CSV_DIR, epochs=500, head="hl_gauss",
         bins=hl.DEFAULT_BINS, guard_bins=hl.DEFAULT_GUARD, sigma_ratio=hl.DEFAULT_SIGMA_RATIO,
         plot_path="learning_curve_ymulti.png"):
    X_state, mode_oh, y, state_cols = load_data(csv_dir)
    state_dim = X_state.shape[1]
    print(f"状態特徴量 {state_dim} 次元 + モード {rf.MODE_DIM} 次元 → 入力 {state_dim + rf.MODE_DIM} 次元")

    _, bin_idx = bin_sample_weights(y)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=bin_idx)
    Xs_tr, Xs_te = X_state[idx_tr], X_state[idx_te]
    mo_tr, mo_te = mode_oh[idx_tr], mode_oh[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"学習 {len(idx_tr)} 行 / テスト {len(idx_te)} 行")

    scaler = StandardScaler()
    Xtr = np.hstack([scaler.fit_transform(Xs_tr), mo_tr]).astype(np.float32)
    Xte = np.hstack([scaler.transform(Xs_te), mo_te]).astype(np.float32)

    stop_col = state_cols.index(STOP_PHASE_COL)
    dwell_col = state_cols.index(DWELL_COL)
    is_stop = Xs_tr[:, stop_col] >= 0.5
    is_dwell = Xs_tr[:, dwell_col] >= 0.5
    print(f"[サンプル重み] 停車完了フェーズ {int(is_stop.sum())} 行 ×{STOP_PHASE_WEIGHT} / "
          f"駅停車中 {int(is_dwell.sum())} 行 ×{DWELL_WEIGHT}")

    base_w = np.ones(len(Xs_tr), dtype=np.float32)
    base_w = np.where(is_stop, base_w * STOP_PHASE_WEIGHT, base_w)
    base_w = np.where(is_dwell, base_w * DWELL_WEIGHT, base_w)

    # --- 1段目: ゲート分類器 ---
    ytr_zero = (y_tr.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    yte_zero = (y_te.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    print(f"[ゲート] 0.0ラベル {int(ytr_zero.sum())} 行 / 非0.0 {int(len(ytr_zero)-ytr_zero.sum())} 行")
    gate = build_gate(Xtr.shape[1])
    gw = base_w / base_w.mean()
    gate.fit(Xtr, ytr_zero, sample_weight=gw, validation_data=(Xte, yte_zero),
             epochs=epochs, batch_size=64, verbose=1,
             callbacks=[EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)])

    # --- 2段目: 非0.0のみの回帰器 ---
    nz_tr = y_tr.flatten() > ZERO_THRESHOLD
    nz_te = y_te.flatten() > ZERO_THRESHOLD
    Xtr_nz, ytr_nz = Xtr[nz_tr], y_tr[nz_tr]
    Xte_nz, yte_nz = Xte[nz_te], y_te[nz_te]
    print(f"\n[回帰器] 学習 {len(Xtr_nz)} 行 / テスト {len(Xte_nz)} 行（非0.0のみ）")

    w_nz, _ = bin_sample_weights(ytr_nz)
    w_nz = w_nz * base_w[nz_tr]
    ceiling = (ytr_nz.flatten() >= CEILING_LABEL_THRESHOLD) & is_stop[nz_tr]
    w_nz = w_nz * np.where(ceiling, CEILING_LABEL_WEIGHT, 1.0).astype(np.float32)
    w_nz = (w_nz / w_nz.mean()).astype(np.float32)

    if head == "hl_gauss":
        centers, width = hl.make_centers(bins, guard_bins)
        sigma = sigma_ratio * width
        print(f"[回帰器] HL-Gaussian: ビン{len(centers)}個 "
              f"[{centers[0]:.3f}, {centers[-1]:.3f}] 幅={width:.4f} σ={sigma:.4f}")
        t_tr = hl.gaussian_bin_targets(ytr_nz.flatten(), centers, width, sigma)
        t_te = hl.gaussian_bin_targets(yte_nz.flatten(), centers, width, sigma)
        monitor = "val_expected_mae"
    else:
        print("[回帰器] スカラー回帰ヘッド（Huber）")
        centers, t_tr, t_te, monitor = None, ytr_nz, yte_nz, "val_loss"

    model = build_regressor(Xtr_nz.shape[1], head, centers)
    hist = model.fit(Xtr_nz, t_tr, sample_weight=w_nz, validation_data=(Xte_nz, t_te),
                     epochs=epochs, batch_size=64, verbose=1,
                     callbacks=[EarlyStopping(monitor=monitor, patience=30, restore_best_weights=True),
                                ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-6)])
    plot_curve(hist, plot_path, head)

    # --- 合成モデルの評価（推論時と同一の合成ロジック）---
    gate_prob = gate.predict(Xte, verbose=0).flatten()
    raw = model.predict(Xte, verbose=0)
    if head == "hl_gauss":
        reg = hl.expectation(raw, centers)
        combined = (1.0 - gate_prob) * reg
    else:
        reg = raw.flatten()
        combined = np.where(gate_prob >= 0.5, 0.0, np.clip(reg, 0.0, 1.0))
    yf = y_te.flatten()
    print(f"\n[合成モデル] テストMAE: {np.abs(combined - yf).mean():.4f}")
    zm = yf <= ZERO_THRESHOLD
    if zm.any():
        print(f"[合成モデル] ラベル0.0のうち予測0.25超（報酬リーク）: "
              f"{(combined[zm] > 0.25).mean():.1%}（平均予測 {combined[zm].mean():.3f}）")
    if (~zm).any():
        print(f"[合成モデル] 非0.0ラベルのMAE: {np.abs(combined[~zm] - yf[~zm]).mean():.4f}")
    # 駅停車中の行だけのMAE（発車判断が学習できているかの確認）
    dwell_te = Xs_te[:, dwell_col] >= 0.5
    if dwell_te.any():
        print(f"[合成モデル] 駅停車中の行のMAE: {np.abs(combined[dwell_te] - yf[dwell_te]).mean():.4f}"
              f"（{int(dwell_te.sum())} 行）")

    model.save(CFG.REWARD_MODEL_PATH)
    gate.save(CFG.REWARD_GATE_PATH)
    joblib.dump(scaler, CFG.REWARD_SCALER_PATH)
    manifest = {
        "state_feature_cols": state_cols,
        "mode_classes_onehot": rf.MODE_CLASSES,
        "state_dim": state_dim,
        "input_dim": state_dim + rf.MODE_DIM,
        "regressor_head": (hl.head_manifest(bins, guard_bins, sigma_ratio)
                           if head == "hl_gauss" else hl.scalar_head_manifest()),
    }
    with open(CFG.REWARD_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {CFG.REWARD_MODEL_PATH} / {CFG.REWARD_GATE_PATH} / "
          f"{CFG.REWARD_SCALER_PATH} / {CFG.REWARD_MANIFEST_PATH}")
    print(f"  → 回帰器ヘッド: {manifest['regressor_head']['head']} / "
          f"合成: {manifest['regressor_head']['composition']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="複数駅間版の報酬予測NN（蒸留）を学習する")
    ap.add_argument("--head", choices=["hl_gauss", "scalar"], default="hl_gauss")
    ap.add_argument("--bins", type=int, default=hl.DEFAULT_BINS)
    ap.add_argument("--guard-bins", type=int, default=hl.DEFAULT_GUARD)
    ap.add_argument("--sigma-ratio", type=float, default=hl.DEFAULT_SIGMA_RATIO)
    ap.add_argument("--csv-dir", default=CFG.TRAIN_CSV_DIR)
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()
    main(csv_dir=a.csv_dir, epochs=a.epochs, head=a.head,
         bins=a.bins, guard_bins=a.guard_bins, sigma_ratio=a.sigma_ratio)
