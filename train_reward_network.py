import os
import glob
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

import re

EPOCHS = 500
BATCH_SIZE = 32

def extract_limit_info(text):
    text = str(text)
    # 制限速度がない場合: (フラグ, 距離, 速度) = (0.0, 0.0, 0.0)
    if "この先制限速度なし" in text:
        return 0.0, 0.0, 0.0
    
    match = re.search(r'(\d+)m先に制限速度(\d+)km/h', text)
    if match:
        # 制限速度がある場合: (フラグ, 距離, 速度) = (1.0, 抽出距離, 抽出速度)
        return 1.0, float(match.group(1)), float(match.group(2))
    
    return 0.0, 0.0, 0.0

def extract_gradient_info(text):
    text = str(text)
    # 勾配がない場合: (フラグ, 距離, 勾配値) = (0.0, 0.0, 0.0)
    if "この先目立った勾配なし" in text:
        return 0.0, 0.0, 0.0
    
    match = re.search(r'(\d+)m先に(上り|下り)勾配(\d+\.?\d*)‰あり', text)
    if match:
        dist = float(match.group(1))
        direction = match.group(2)
        val = float(match.group(3))
        if direction == '下り':
            val = -val
        # 勾配がある場合: (フラグ, 距離, 勾配値) = (1.0, 抽出距離, 抽出勾配)
        return 1.0, dist, val
        
    return 0.0, 0.0, 0.0

def load_and_preprocess_data(csv_dir):
    # 指定ディレクトリ内の全CSVファイルを取得
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    
    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")
    
    columns = ["time", "train_id", "phase", "notch", "speed_limit", "current_speed", 
               "dist_to_next_station", "delay", "current_gradient", "next_limit_info", 
               "next_gradient_info", "w_surv", "w_conf", "w_comp", "reason"]
    
    df_list = []
    for file in csv_files:
        # 1行目はヘッダとしてスキップ
        temp_df = pd.read_csv(file, names=columns, skiprows=1)
        df_list.append(temp_df)
    
    # 複数ファイルのデータを結合
    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")
    
    # カテゴリ変数のOne-Hotエンコーディング
    phase_categories = [
        "駅出発直後の加速フェーズ（駅発車20秒以内）", 
        "巡航フェーズ（駅間走行中）", 
        "制限速度接近フェーズ（500m以内に制限速度があり、かつ現在速度が制限速度を超えている）", 
        "次駅への減速フェーズ（駅手前400m以内）"
    ]
    notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
    
    df['phase'] = pd.Categorical(df['phase'], categories=phase_categories)
    df['notch'] = pd.Categorical(df['notch'], categories=notch_categories)
    df = pd.get_dummies(df, columns=['phase', 'notch'], dummy_na=False)
    
    
    # テキスト情報からの数値とフラグの抽出
    df['next_limit_flag'], df['next_limit_dist'], df['next_limit_speed'] = zip(*df['next_limit_info'].apply(extract_limit_info))
    df['next_gradient_flag'], df['next_gradient_dist'], df['next_gradient_val'] = zip(*df['next_gradient_info'].apply(extract_gradient_info))
    
    # 入力特徴量 (X) の選択にフラグを追加
    feature_cols = [
        'speed_limit', 'current_speed', 'dist_to_next_station', 'delay', 'current_gradient',
        'next_limit_flag', 'next_limit_dist', 'next_limit_speed', # フラグを追加
        'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val' # フラグを追加
    ] + [col for col in df.columns if 'phase_' in col or 'notch_' in col]
    
    X = df[feature_cols].values.astype(np.float32)
    # LLMが出力した正解ラベル
    y = df[['w_surv', 'w_conf', 'w_comp']].values.astype(np.float32)
    
    return X, y, feature_cols

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.1), # 過学習防止
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(3, activation='sigmoid') # 重みは0.0~1.0なのでsigmoid
    ])
    # 損失関数は平均二乗誤差(MSE)、評価指標に平均絶対誤差(MAE)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_learning_curve(history):
    # 学習曲線の描画
    plt.figure(figsize=(12, 5))
    
    # Lossのプロット
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    # MAEのプロット
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('learning_curve.png') # 画像として保存
    plt.show() # 画面に表示

def main():
    csv_dir = 'train_reward_csv'
    
    # 1. 複数CSVの読み込みと前処理
    X, y, feature_cols = load_and_preprocess_data(csv_dir)
    print(f"入力特徴量次元数: {X.shape[1]}")
    
    # 2. データをランダムに8:2で分割 (学習用:テスト用)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")
    
    # 3. スケーリング (特徴量の標準化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. モデルの構築
    model = build_model(X_train_scaled.shape[1])
    
    # 5. モデルの学習
    print("学習を開始します...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=EPOCHS,     #エポック数は適宜調整
        batch_size=BATCH_SIZE, #バッチサイズは適宜調整
        verbose=1
    )
    
    # 6. 学習曲線の表示と保存
    plot_learning_curve(history)
    
    # 7. モデルとスケーラーの保存
    model.save('reward_weight_model.h5')
    joblib.dump(scaler, 'reward_weight_scaler.pkl')
    print("モデル('reward_weight_model.h5')とスケーラー('reward_weight_scaler.pkl')を保存しました。")

if __name__ == "__main__":
    # GPUを無効化する場合は以下のコメントアウトを外す (Apex-Train側の環境に合わせる場合)
    # tf.config.set_visible_devices([], "GPU")
    main()