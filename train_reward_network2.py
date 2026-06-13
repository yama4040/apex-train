import os
import glob
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def custom_accuracy(y_true, y_pred):
    # LLMの評価値と予測値のズレが 0.15 以内であれば「正解(1)」とする
    tolerance = 0.15
    correct_predictions = tf.cast(tf.abs(y_true - y_pred) <= tolerance, tf.float32)
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
        # [存在フラグ, 距離, 速度]
        # ※いない場合は安全な遠方(5000m)として扱うことでNNの混乱を防ぐ
        return 0.0, 5000.0, 0.0 
    
    # 例: "前方 360.5m 先を 50.0km/h で走行中" から数値を抽出
    match = re.search(r'前方\s*([\d\.]+)\s*m\s*先を\s*([\d\.]+)\s*km/h', text)
    if match:
        return 1.0, float(match.group(1)), float(match.group(2))
    return 0.0, 5000.0, 0.0

def extract_backward_info(text):
    text = str(text)
    if "後続列車なし" in text or text == "nan":
        # 後続がいない場合も安全な遠方(5000m)として扱う
        return 0.0, 5000.0, 0.0 
    
    # 例: "後方 1000.0m 後ろを 70.0km/h で走行中" から数値を抽出
    match = re.search(r'後方\s*([\d\.]+)\s*m\s*後ろを\s*([\d\.]+)\s*km/h', text)
    if match:
        return 1.0, float(match.group(1)), float(match.group(2))
    return 0.0, 5000.0, 0.0

def load_and_preprocess_data(csv_dir):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    
    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")
    
    # 1. カラムに holding_time と time_to_next_station を両方含む完全な構成
    columns = ["time", "train_id", "phase", "current_notch", "holding_time", 
               "prev_notch", "prev_notch_duration", 
               "speed_limit", "current_speed", 
               "dist_to_next_station", "time_to_next_station", "delay", "current_gradient", 
               "next_limit_info", "next_gradient_info", "forward_info", "backward_info", "reward", "reason"]
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file, names=columns, skiprows=1)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")
    
    phase_categories = [
        "駅出発直後の加速フェーズ（20秒以内）", 
        "巡航フェーズ（駅間走行中）", 
        "制限速度区間に接近中（500m以内に制限区間在り）", 
        "次駅への減速フェーズ（駅手前400m以内）"
    ]
    notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
    
    # ▼▼▼ 追加: 直前のノッチのカテゴリ ▼▼▼
    prev_notch_categories = ["なし（または停止）", "力行（加速）", "惰行", "ブレーキ（減速）"]
    # ▲▲▲ 追加 ▲▲▲
    
    df['phase'] = pd.Categorical(df['phase'], categories=phase_categories)
    df['current_notch'] = pd.Categorical(df['current_notch'], categories=notch_categories)
    df['prev_notch'] = pd.Categorical(df['prev_notch'], categories=prev_notch_categories) # 追加
    
    # ▼ 変更: prev_notch もダミー変数化に含める
    df = pd.get_dummies(df, columns=['phase', 'current_notch', 'prev_notch'], dummy_na=False)
    
    # 2. 保持時間を「惰行」「加速」「減速」の3つの独立した特徴量に分離する（交差特徴量）
    df['hold_coast'] = df['holding_time'] * df['current_notch_惰行中']
    df['hold_accel'] = df['holding_time'] * df['current_notch_力行（加速）中']
    df['hold_decel'] = df['holding_time'] * df['current_notch_ブレーキ（減速）中']
    
    df['next_limit_flag'], df['next_limit_dist'], df['next_limit_speed'] = zip(*df['next_limit_info'].apply(extract_limit_info))
    df['next_gradient_flag'], df['next_gradient_dist'], df['next_gradient_val'] = zip(*df['next_gradient_info'].apply(extract_gradient_info))
    
    # 先行列車情報の変換
    forward_features = df['forward_info'].apply(extract_forward_info).apply(pd.Series)
    forward_features.columns = ['f_exist', 'f_distance', 'f_speed']
    
    # 後続列車情報の変換
    backward_features = df['backward_info'].apply(extract_backward_info).apply(pd.Series)
    backward_features.columns = ['b_exist', 'b_distance', 'b_speed']
    
    # ▼▼▼ 追加：抽出した特徴量を元の df に横方向に結合する ▼▼▼
    df = pd.concat([df, forward_features, backward_features], axis=1)
    
    # 3. 最終的な特徴量ベクトル（22次元）の構築
    feature_cols = [
        'hold_coast', 'hold_accel', 'hold_decel', 
        'prev_notch_duration',  # ←【追加】直前の保持時間
        'speed_limit', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'delay', 'current_gradient',
        'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
        'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
        'f_exist', 'f_distance', 'f_speed',
        'b_exist', 'b_distance', 'b_speed'
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('current_notch_') or (col.startswith('prev_notch_') and col != 'prev_notch_duration')]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['reward'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y, feature_cols

def build_model(input_dim):
    # 4. L2正則化を用いた、汎化性能重視の軽量ネットワーク設計
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        #Dropout(0.1),
        Dense(1, activation='sigmoid') # 0.0 ~ 1.0 の評価値を出力
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_learning_curve(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('learning_curve_direct.png') 
    plt.show() 

def main():
    csv_dir = 'train_reward_csv_direct'
    
    X, y, feature_cols = load_and_preprocess_data(csv_dir)
    print(f"入力特徴量次元数: {X.shape[1]}") # 22次元と出力されれば成功です
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = build_model(X_train_scaled.shape[1])
    
    # 5. 無駄な過学習を自動で停止し、最もLossが低かった瞬間の重みを採用する
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    print("学習を開始します...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    plot_learning_curve(history)
    
    model.save('direct_reward_model.h5')
    joblib.dump(scaler, 'direct_reward_scaler.pkl')
    print("モデル('direct_reward_model.h5')とスケーラー('direct_reward_scaler.pkl')を保存しました。")

if __name__ == "__main__":
    main()