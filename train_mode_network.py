"""モードNN（運転モード分類器）の学習（二重蒸留の1段目・2026-07-25）

LLMが付与した mode ラベル（normal / delay_recovery / anti_mid_stop）を、状態特徴量から
再現する3クラス分類器を蒸留する。RL実行時はこのモードNNがモードを供給し、その結果を
評価NNへ入力する（学習時のLLMモードラベル→評価NN教師入力と同一経路）。

- 入力: reward_features.STATE_FEATURE_COLS（評価NNと共通の状態特徴量）
- 出力: 3クラスsoftmax（MODE_CLASSES_ACTIVE）
- クラス不均衡（normalが約94%）のため balanced クラス重みを適用
- 出力: mode_model.h5 / mode_scaler.pkl / mode_manifest.json
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib
matplotlib.use('Agg')  # 学習曲線をPNG保存するだけなので非対話バックエンドを使う（詳細は train_reward_network2.py の同箇所）。
import matplotlib.pyplot as plt

import reward_features as rf

np.random.seed(42)
tf.random.set_seed(42)

ACTIVE = rf.MODE_CLASSES_ACTIVE  # ["normal","delay_recovery","anti_mid_stop"]


def load_data(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"'{csv_dir}' にCSVがありません。")
    dfs, skipped = [], []
    for f in files:
        d = pd.read_csv(f, encoding='utf-8-sig')
        if 'required_speed' not in d.columns or 'mode' not in d.columns:
            skipped.append(os.path.basename(f))
            continue
        dfs.append(d)
    if skipped:
        print(f"[警告] required_speed/mode列が無いため除外: {len(skipped)}件")
    df = pd.concat(dfs, ignore_index=True)
    # 有効な3クラスのみ（spacing等の想定外ラベルは除外）
    df = df[df['mode'].isin(ACTIVE)].reset_index(drop=True)
    print(f"合計データ数: {len(df)}行")
    print("モード分布:", {c: int((df['mode'] == c).sum()) for c in ACTIVE})

    X, cols = rf.build_state_matrix(df)
    y = df['mode'].map(ACTIVE.index).values.astype(np.int64)
    return X, y, cols


def plot_mode_learning_curve(history, out_path='learning_curve_mode.png'):
    """モード分類器（3クラス）の学習曲線: loss と accuracy。"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Mode Classifier Loss (sparse categorical CE)')
    plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Mode Classifier Accuracy')
    plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"[保存] 学習曲線: {out_path}")


def build_model(input_dim, n_classes):
    l2_reg = l2(1e-4)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(32, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.1),
        Dense(n_classes, activation='softmax'),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(csv_dir='train_reward_csv_direct',
         model_path='mode_model.h5', scaler_path='mode_scaler.pkl',
         manifest_path='mode_manifest.json', epochs=300):
    X, y, cols = load_data(csv_dir)
    print(f"入力特徴量次元数: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"学習{len(X_tr)} / テスト{len(X_te)}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # クラス不均衡対策: balanced クラス重み
    # 学習データに1件も存在しないクラス（例: delay_recovery が0件）があると、
    # compute_class_weight の返す辞書にそのキーが欠落する。Kerasのfitは class_weight に
    # 0..len(ACTIVE)-1 の全キーを要求するため、欠損クラスは重み1.0で補完する。
    classes = np.unique(y_tr)
    cw = compute_class_weight('balanced', classes=classes, y=y_tr)
    class_weight = {i: 1.0 for i in range(len(ACTIVE))}
    class_weight.update({int(c): float(w) for c, w in zip(classes, cw)})
    print("クラス重み:", {ACTIVE[k]: round(v, 2) for k, v in class_weight.items()})

    model = build_model(X_tr_s.shape[1], len(ACTIVE))
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    history = model.fit(X_tr_s, y_tr, validation_data=(X_te_s, y_te), epochs=epochs, batch_size=64,
                        class_weight=class_weight, callbacks=[es, rl], verbose=1)
    plot_mode_learning_curve(history, 'learning_curve_mode.png')

    # 評価: 混同行列（モード判定の弱い境界を特定する診断）
    y_pred = np.argmax(model.predict(X_te_s, verbose=0), axis=1)
    print("\n=== 混同行列 (行=正解, 列=予測) ===")
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(ACTIVE))))
    print("labels:", ACTIVE)
    print(cm)
    print("\n=== classification_report ===")
    print(classification_report(y_te, y_pred, labels=list(range(len(ACTIVE))),
                                target_names=ACTIVE, zero_division=0))

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'state_feature_cols': cols, 'mode_classes_active': ACTIVE,
                   'mode_classes_onehot': rf.MODE_CLASSES}, f, ensure_ascii=False, indent=2)
    print(f"\nモデル('{model_path}')・スケーラー('{scaler_path}')・マニフェスト('{manifest_path}')を保存しました。")


if __name__ == "__main__":
    main()
