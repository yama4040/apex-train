# -*- coding: utf-8 -*-
"""
train_reward_network2.py のネットワーク構成探索スクリプト

隠れ層の深さ（1～3層）× ユニット数（256～16、2の累乗）×
構成パターン（全層同一 / ピラミッド型・半減）を総当たりで学習し、
val_loss / val_MAE を比較して最良の構成を探す。

- 前処理・train/test分割（stratify, random_state=42）は全構成で共通
- 損失関数・最適化・正則化は train_reward_network2.py と同一条件
- 各構成を3シードで学習し平均±標準偏差で評価（乱数の影響を抑制）
- 出力層は回帰問題のため恒等関数（linear）で固定
"""
import os
import json
import time
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

# 前処理・重み計算は本体スクリプトのものをそのまま使う（条件を完全一致させる）
from train_reward_network2 import (
    load_and_preprocess_data,
    compute_bin_sample_weights,
    custom_accuracy,
)

# ===== 探索設定 =====
UNITS_LIST = [256, 128, 64, 32, 16]  # 第一層のユニット数
DEPTHS = [1, 2, 3]                   # 隠れ層の深さ
SCHEMES = ['uniform', 'pyramid']     # 全層同一 / 半減ピラミッド
SEEDS = [42, 123, 2024]              # 3シード平均
EPOCHS = 300
ES_PATIENCE = 20
LR_PATIENCE = 8
RESULT_CSV = 'architecture_search_results.csv'


def make_layer_units(first_units, depth, scheme):
    """構成パターンに応じた各層のユニット数リストを返す"""
    if scheme == 'uniform':
        return [first_units] * depth
    # pyramid: 半減（例: 128 -> [128, 64, 32]）
    return [max(first_units // (2 ** i), 1) for i in range(depth)]


def build_model(input_dim, layer_units):
    """train_reward_network2.build_model と同一方針で任意構成のモデルを構築"""
    l2_reg = l2(1e-4)
    # Dropout率は現行モデル（0.3, 0.2, 0.1）に合わせ、層が進むごとに0.1ずつ減衰
    dropout_rates = [max(0.3 - 0.1 * i, 0.1) for i in range(len(layer_units))]

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units, rate in zip(layer_units, dropout_rates):
        model.add(Dense(units, kernel_regularizer=l2_reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(rate))
    # 回帰問題のため出力層は恒等関数
    model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0),
                  metrics=['mae', custom_accuracy])
    return model


def main():
    # ===== データ準備（全構成で共通） =====
    X, y, feature_cols = load_and_preprocess_data('train_reward_csv_direct')
    print(f"入力特徴量次元数: {X.shape[1]}")

    _, bin_idx_all = compute_bin_sample_weights(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bin_idx_all
    )
    train_sample_weight, _ = compute_bin_sample_weights(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")

    # ===== 構成リスト生成（1層はuniform/pyramidが同一なので重複排除） =====
    configs = []
    seen = set()
    for scheme in SCHEMES:
        for depth in DEPTHS:
            for units in UNITS_LIST:
                layer_units = make_layer_units(units, depth, scheme)
                key = tuple(layer_units)
                if key in seen:
                    continue
                seen.add(key)
                configs.append({'scheme': scheme, 'depth': depth,
                                'first_units': units, 'layer_units': layer_units})

    total_runs = len(configs) * len(SEEDS)
    print(f"検証構成数: {len(configs)}, 総学習回数: {total_runs}")

    results = []
    run_count = 0
    t_start = time.time()

    for cfg in configs:
        seed_records = []
        for seed in SEEDS:
            run_count += 1
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.keras.backend.clear_session()

            model = build_model(X_train_scaled.shape[1], cfg['layer_units'])
            n_params = model.count_params()

            early_stop = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE,
                                       restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=LR_PATIENCE, min_lr=1e-6, verbose=0)

            history = model.fit(
                X_train_scaled, y_train,
                sample_weight=train_sample_weight,
                validation_data=(X_test_scaled, y_test),
                epochs=EPOCHS,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            # restore_best_weights=True なので best weights で評価
            eval_res = model.evaluate(X_test_scaled, y_test, verbose=0)
            val_loss, val_mae, val_acc = eval_res[0], eval_res[1], eval_res[2]
            best_epoch = int(np.argmin(history.history['val_loss'])) + 1
            n_epochs = len(history.history['val_loss'])

            seed_records.append({
                'seed': seed, 'val_loss': val_loss, 'val_mae': val_mae,
                'val_acc': val_acc, 'best_epoch': best_epoch, 'epochs': n_epochs,
            })
            elapsed = time.time() - t_start
            print(f"[{run_count}/{total_runs}] {cfg['layer_units']} seed={seed} "
                  f"val_loss={val_loss:.5f} val_mae={val_mae:.5f} "
                  f"acc={val_acc:.3f} ep={best_epoch}/{n_epochs} "
                  f"({elapsed/60:.1f}分経過)", flush=True)

        losses = [r['val_loss'] for r in seed_records]
        maes = [r['val_mae'] for r in seed_records]
        accs = [r['val_acc'] for r in seed_records]
        results.append({
            'scheme': cfg['scheme'],
            'depth': cfg['depth'],
            'first_units': cfg['first_units'],
            'layer_units': json.dumps(cfg['layer_units']),
            'n_params': n_params,
            'val_loss_mean': np.mean(losses),
            'val_loss_std': np.std(losses),
            'val_mae_mean': np.mean(maes),
            'val_mae_std': np.std(maes),
            'val_acc_mean': np.mean(accs),
            'val_acc_std': np.std(accs),
            'best_epoch_mean': np.mean([r['best_epoch'] for r in seed_records]),
        })
        # 途中経過を逐次保存（中断しても結果が残るように）
        pd.DataFrame(results).to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')

    df = pd.DataFrame(results).sort_values('val_mae_mean')
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')

    print("\n===== 結果（val_MAE平均の昇順・上位10件） =====")
    print(df.head(10).to_string(index=False))
    best = df.iloc[0]
    print(f"\n最良構成: {best['layer_units']} ({best['scheme']}) "
          f"val_loss={best['val_loss_mean']:.5f}±{best['val_loss_std']:.5f} "
          f"val_mae={best['val_mae_mean']:.5f}±{best['val_mae_std']:.5f}")
    print(f"結果を '{RESULT_CSV}' に保存しました。")
    print(f"総所要時間: {(time.time() - t_start)/60:.1f}分")


if __name__ == "__main__":
    main()
