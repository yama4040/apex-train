import os
import glob
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def custom_accuracy(y_true, y_pred):
    """
    分類問題用のカスタムAccuracy。
    予測クラス（確率が最大のインデックス）と正解クラスのズレが
    ±1クラス（元の0.1スケールで±0.1）以内であれば正解(1)とする。
    """
    true_class = tf.cast(tf.squeeze(y_true), tf.float32)
    pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    
    tolerance = 1.5  # インデックスの差が1.5以下（実質±1クラス以内）
    correct_predictions = tf.cast(tf.abs(true_class - pred_class) <= tolerance, tf.float32)
    return tf.reduce_mean(correct_predictions)

def extract_limit_info(text):
    text = str(text)
    if "この先制限速度なし" in text:
        return 0.0, 0.0, 0.0
    match = re.search(r'(\d+)m先に制限速度(\d+)km/h', text)
    if match:
        return 1.0, float(match.group(1)), float(match.group(2))
    return 0.0, 0.0, 0.0

def extract_gradient_info(text):
    text = str(text)
    if "この先目立った勾配なし" in text:
        return 0.0, 0.0, 0.0
    match = re.search(r'(\d+)m先に(上り|下り)勾配(\d+\.?\d*)‰あり', text)
    if match:
        dist = float(match.group(1))
        direction = match.group(2)
        val = float(match.group(3))
        if direction == '下り':
            val = -val
        return 1.0, dist, val
    return 0.0, 0.0, 0.0

def extract_forward_info(text):
    text = str(text)
    if "先行列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0 
    
    match = re.search(r'前方\s*([\d\.]+)\s*m\s*先を\s*([\d\.]+)\s*km/h', text)
    if match:
        return 1.0, float(match.group(1)), float(match.group(2))
    return 0.0, 5000.0, 0.0

def extract_backward_info(text):
    text = str(text)
    if "後続列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0 
    
    match = re.search(r'後方\s*([\d\.]+)\s*m\s*後ろを\s*([\d\.]+)\s*km/h', text)
    if match:
        return 1.0, float(match.group(1)), float(match.group(2))
    return 0.0, 5000.0, 0.0

def load_and_preprocess_data(csv_dir):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    
    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")
    
    columns = ["time", "train_id", "phase", "current_notch", "holding_time", 
               "prev_notch", "prev_notch_duration", 
               "speed_limit", "signal_speed", "current_speed", 
               "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delay", "current_gradient", 
               "next_limit_info", "next_gradient_info", "forward_info", "backward_info", "reward", "reason"]
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file, names=columns, skiprows=1)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")
    
    # DQNの3つの行動空間[力行, 惰行, ブレーキ]に集約
    df['prev_notch'] = df['prev_notch'].replace('なし（または停止）', 'ブレーキ（減速）')
    df['current_notch'] = df['current_notch'].replace('停止・その他', 'ブレーキ（減速）中')
    
    # =====================================================================
    # ▼▼▼ 派生特徴量の作成 ▼▼▼
    # =====================================================================
    
    df['margin_speed'] = df['speed_limit'] - df['current_speed']
    df['margin_stop_dist'] = df['dist_to_next_station'] - df['req_stop_dist']
    df['margin_signal_speed'] = df['signal_speed'] - df['current_speed']
    
    df['safe_time'] = df['time_to_next_station'].clip(lower=1.0)
    df['required_speed_mps'] = df['dist_to_next_station'] / df['safe_time']
    df['required_cruise_speed'] = (df['required_speed_mps'] * 3.6) * 1.3
    
    hunting_condition = (df['holding_time'] < 5.0) & (df['prev_notch_duration'] < 5.0) & (df['current_notch'] != df['prev_notch'])
    df['hunting_score'] = np.where(hunting_condition, np.maximum(0.0, 5.0 - df['holding_time']) / 5.0, 0.0).astype(np.float32)
    
    # =====================================================================
    
    phase_categories = [
        "駅出発直後の加速フェーズ（20秒以内）", 
        "巡航フェーズ（駅間走行中）", 
        "制限速度区間に接近中（500m以内に制限区間在り）", 
        "次駅への減速フェーズ（駅手前400m以内）",
        "駅停車完了（速度0km/h）"
    ]
    notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
    prev_notch_categories = ["惰行", "力行（加速）", "ブレーキ（減速）"]
    
    df['phase'] = pd.Categorical(df['phase'], categories=phase_categories)
    df['current_notch'] = pd.Categorical(df['current_notch'], categories=notch_categories)
    df['prev_notch'] = pd.Categorical(df['prev_notch'], categories=prev_notch_categories)
    
    df = pd.get_dummies(df, columns=['phase', 'current_notch', 'prev_notch'], dummy_na=False)
    
    df['hold_coast'] = df['holding_time'] * df['current_notch_惰行中']
    df['hold_accel'] = df['holding_time'] * df['current_notch_力行（加速）中']
    df['hold_decel'] = df['holding_time'] * df['current_notch_ブレーキ（減速）中']
    
    df['next_limit_flag'], df['next_limit_dist'], df['next_limit_speed'] = zip(*df['next_limit_info'].apply(extract_limit_info))
    df['next_gradient_flag'], df['next_gradient_dist'], df['next_gradient_val'] = zip(*df['next_gradient_info'].apply(extract_gradient_info))
    
    forward_features = df['forward_info'].apply(extract_forward_info).apply(pd.Series)
    forward_features.columns = ['f_exist', 'f_distance', 'f_speed']
    backward_features = df['backward_info'].apply(extract_backward_info).apply(pd.Series)
    backward_features.columns = ['b_exist', 'b_distance', 'b_speed']
    df = pd.concat([df, forward_features, backward_features], axis=1)
    
    df['f_relative_speed'] = df['current_speed'] - df['f_speed']
    df['b_relative_speed'] = df['b_speed'] - df['current_speed']
    
    df['f_distance'] = df['f_distance'].clip(upper=2000.0)
    df['b_distance'] = df['b_distance'].clip(upper=2000.0)
    
    feature_cols = [
        'hold_coast', 'hold_accel', 'hold_decel', 
        'prev_notch_duration',
        'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
        'margin_speed', 'margin_signal_speed', 'margin_stop_dist', 'required_speed_mps', 'required_cruise_speed', 'hunting_score',
        'f_relative_speed', 'b_relative_speed',
        'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
        'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
        'f_exist', 'f_distance', 'f_speed',
        'b_exist', 'b_distance', 'b_speed'
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('current_notch_') or (col.startswith('prev_notch_') and col != 'prev_notch_duration')]
    
    X = df[feature_cols].values.astype(np.float32)
    
    # ▼▼▼ 【変更】0.0~1.0の報酬を、0~10の整数クラスインデックスに変換 ▼▼▼
    # np.round で安全に丸めた後、整数型(int32)にキャストします
    y_continuous = df['reward'].values.astype(np.float32)
    y_class = np.round(y_continuous * 10).astype(np.int32)
    y = y_class.reshape(-1, 1)
    
    return X, y, feature_cols

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # ▼▼▼ 【変更】1次元出力(linear)から11次元出力(softmax)へ ▼▼▼
        Dense(11, activation='softmax')
    ])
    
    # ▼▼▼ 【変更】損失関数を交差エントロピー誤差に変更 ▼▼▼
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy', custom_accuracy]
    )
    return model

def plot_learning_curve(history):
    plt.figure(figsize=(12, 5))
    
    # Lossのプロット
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Crossentropy)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    # 精度(custom_accuracy)のプロット
    plt.subplot(1, 2, 2)
    plt.plot(history.history['custom_accuracy'], label='Train Custom Acc')
    plt.plot(history.history['val_custom_accuracy'], label='Validation Custom Acc')
    plt.title('Model Accuracy (Tolerance: ±0.1)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('learning_curve_classification.png') 
    plt.show() 

def main():
    csv_dir = 'train_reward_csv_direct'
    
    X, y, feature_cols = load_and_preprocess_data(csv_dir)
    print(f"入力特徴量次元数: {X.shape[1]}")
    print(f"クラスラベルの分布:\n{pd.Series(y.flatten()).value_counts().sort_index()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = build_model(X_train_scaled.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    print("学習を開始します...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=500,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    plot_learning_curve(history)
    
    # 保存名も分類モデルであることがわかるように変更
    model.save('classification_reward_model.h5')
    joblib.dump(scaler, 'classification_reward_scaler.pkl')
    print("モデル('classification_reward_model.h5')とスケーラー('classification_reward_scaler.pkl')を保存しました。")

if __name__ == "__main__":
    main()