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
from tensorflow.keras.losses import Huber  # <--- 【追加】Huber Loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def custom_accuracy(y_true, y_pred):
    # LLMの評価値と予測値のズレが 0.15 以内であれば「正解(1)」とする
    # (※推論値はLinearで0以下や1以上になる可能性もあるため、ここで計算上考慮されます)
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
               "speed_limit", "current_speed", 
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
    # ▼▼▼ 【重要】ダミー変数化の前に、文字列を比較して計算する派生特徴量 ▼▼▼
    # =====================================================================
    
    # 1. 速度と距離の余裕度
    df['margin_speed'] = df['speed_limit'] - df['current_speed']
    df['margin_stop_dist'] = df['dist_to_next_station'] - df['req_stop_dist']
    
    # 2. 定時到着に必要な平均速度と、LLMが考慮する「上乗せ分(20km/h)」を加味した要求巡航速度
    df['required_speed_mps'] = df['dist_to_next_station'] / (df['time_to_next_station'] + 1e-3)
    df['required_cruise_speed'] = (df['required_speed_mps'] * 3.6) + 20.0
    
    # 3. ノコギリ運転のスコア化 (連続値化)
    # 現在と直前のノッチが異なり、かつ継続時間が共に5秒未満の場合にペナルティスコア(最大1.0)をつける
    hunting_condition = (df['holding_time'] < 5.0) & (df['prev_notch_duration'] < 5.0) & (df['current_notch'] != df['prev_notch'])
    # 5秒からどれだけ短いかを割合にする (例: holding_timeが0.5秒なら (5-0.5)/5 = 0.9スコア)
    df['hunting_score'] = np.where(hunting_condition, np.maximum(0.0, 5.0 - df['holding_time']) / 5.0, 0.0).astype(np.float32)
    
    # =====================================================================
    
    # カテゴリ化とダミー変数化
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
    
    # 先行・後続情報
    forward_features = df['forward_info'].apply(extract_forward_info).apply(pd.Series)
    forward_features.columns = ['f_exist', 'f_distance', 'f_speed']
    backward_features = df['backward_info'].apply(extract_backward_info).apply(pd.Series)
    backward_features.columns = ['b_exist', 'b_distance', 'b_speed']
    df = pd.concat([df, forward_features, backward_features], axis=1)
    
    # =====================================================================
    # ▼▼▼ 【重要】先行列車の衝突判定用 特徴量 (TTC) ▼▼▼
    # =====================================================================
    
    # 相対速度 (km/h) ※プラスなら接近している
    df['f_relative_speed'] = df['current_speed'] - df['f_speed']
    # 衝突までの余裕時間 TTC (Time To Collision) [秒]
    # 接近していない(相対速度がゼロ以下)場合は安全とみなし、最大の5000秒とする
    df['f_ttc'] = np.where(df['f_relative_speed'] > 0, 
                           df['f_distance'] / (df['f_relative_speed'] / 3.6 + 1e-3), 
                           5000.0)
    
    # 後続列車（任意）
    df['b_relative_speed'] = df['b_speed'] - df['current_speed']
    df['b_ttc'] = np.where(df['b_relative_speed'] > 0, 
                           df['b_distance'] / (df['b_relative_speed'] / 3.6 + 1e-3), 
                           5000.0)
    
    # =====================================================================
    # ▼▼▼ 異常値・極端な距離情報のクリッピング（NNのスケーリング崩れ防止） ▼▼▼
    # =====================================================================
    df['dist_to_next_station'] = df['dist_to_next_station'].clip(upper=2000.0)
    df['req_stop_dist'] = df['req_stop_dist'].clip(upper=2000.0)
    df['f_distance'] = df['f_distance'].clip(upper=2000.0)
    df['f_ttc'] = df['f_ttc'].clip(upper=5000.0)
    df['b_distance'] = df['b_distance'].clip(upper=2000.0)
    df['b_ttc'] = df['b_ttc'].clip(upper=5000.0)
    
    # 【追加】新しい特徴量ベクトル
    feature_cols = [
        'hold_coast', 'hold_accel', 'hold_decel', 
        'prev_notch_duration',
        'speed_limit', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
        # --- 今回追加した派生特徴量 ---
        'margin_speed', 'margin_stop_dist', 'required_speed_mps', 'required_cruise_speed', 'hunting_score',
        'f_relative_speed', 'f_ttc', 'b_relative_speed', 'b_ttc',
        # ------------------------------
        'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
        'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
        'f_exist', 'f_distance', 'f_speed',
        'b_exist', 'b_distance', 'b_speed'
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('current_notch_') or (col.startswith('prev_notch_') and col != 'prev_notch_duration')]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['reward'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y, feature_cols

def build_model(input_dim):
    # ▼▼▼ 【完全刷新】表現力と安定性を高めたモデル設計 ▼▼▼
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),      # 表現力を大幅に強化
        Dropout(0.2),                       # 過学習防止
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')       # sigmoidの偏り問題を排除し、linearに変更
    ])
    
    # LLMのノイズ（理不尽な評価ブレ）に強い Huber Loss を採用
    model.compile(optimizer='adam', loss=Huber(delta=0.1), metrics=['mae', custom_accuracy])
    return model

def plot_learning_curve(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (Huber)')
    plt.plot(history.history['val_loss'], label='Validation Loss (Huber)')
    plt.title('Model Loss (Huber)')
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
    print(f"入力特徴量次元数: {X.shape[1]}")
    
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
        batch_size=64, # バッチサイズを少し大きくして勾配を安定させる
        callbacks=[early_stop],
        verbose=1
    )
    
    plot_learning_curve(history)
    
    model.save('direct_reward_model.h5')
    joblib.dump(scaler, 'direct_reward_scaler.pkl')
    print("モデル('direct_reward_model.h5')とスケーラー('direct_reward_scaler.pkl')を保存しました。")

if __name__ == "__main__":
    main()