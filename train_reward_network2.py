import os
import glob
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber  # <--- 【追加】Huber Loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# ▼▼▼ 【修正】重み付けの効果検証時に結果がぶれないよう乱数シードを固定 ▼▼▼
np.random.seed(42)
tf.random.set_seed(42)

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

    # ▼▼▼ 【改修】required_speed（必要速度）列の存在を前提に、実ファイルのヘッダをそのまま使用する ▼▼▼
    # 固定の列名リストで位置決め読み込みをすると、required_speed列の有無でファイルごとに
    # 列がずれて破損するため、各CSV自身のヘッダから列名を読み取る方式に変更している。
    df_list = []
    skipped_files = []
    for file in csv_files:
        temp_df = pd.read_csv(file, encoding='utf-8-sig')
        if 'required_speed' not in temp_df.columns:
            skipped_files.append(os.path.basename(file))
            continue
        df_list.append(temp_df)

    if skipped_files:
        print(f"[警告] 'required_speed'列が存在しない旧形式のため除外したファイル ({len(skipped_files)}件): {skipped_files}")
        print("       最新のapex.py / evaluate_csv_with_llm.pyでCSVを再生成してください。")

    if not df_list:
        raise ValueError(
            f"'{csv_dir}' 内に 'required_speed' 列を含む有効なCSVがありません。"
            "train_reward_csv_direct内のデータは旧形式（required_speed列なし）のため、"
            "最新のapex.pyで学習データを再収集してから実行してください。"
        )

    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")
    
    # DQNの3つの行動空間[力行, 惰行, ブレーキ]に集約
    df['prev_notch'] = df['prev_notch'].replace('なし（または停止）', 'ブレーキ（減速）')
    df['current_notch'] = df['current_notch'].replace('停止・その他', 'ブレーキ（減速）中')
    
    # =====================================================================
    # ▼▼▼ 派生特徴量の作成 ▼▼▼
    # =====================================================================
    
    # 1. 速度と距離の余裕度
    df['margin_speed'] = df['speed_limit'] - df['current_speed']
    df['margin_stop_dist'] = df['dist_to_next_station'] - df['req_stop_dist']
    
    # ▼▼▼ 【改修2】現在速度とCBTC信号現示のマージンを追加 ▼▼▼
    df['margin_signal_speed'] = df['signal_speed'] - df['current_speed']
    
    # 2. 必要速度（CSVに保存済みの物理シミュレーションベースの値をそのまま使用）
    #    加速にかかる時間・惰行による自然減速・ブレーキ特性を考慮した値であり、
    #    単純な平均速度×1.3という近似（旧required_cruise_speed）はもう使用しない。
    # 現在速度と必要速度との差。正なら必要速度に届いておらず力行継続が必要、
    # 負なら必要速度を超えており惰行への移行余地があることを示す。
    df['speed_margin_to_required'] = df['current_speed'] - df['required_speed']

    # 3. ノコギリ運転のスコア化 (連続値化)
    # 閾値はLLM評価プロンプトのノコギリ判定と同じ7秒。
    # direct_reward_predictor2.py（推論側）の特徴量エンジニアリングと完全に一致させること。
    hunting_condition = (df['holding_time'] < 7.0) & (df['prev_notch_duration'] < 7.0) & (df['current_notch'] != df['prev_notch'])
    df['hunting_score'] = np.where(hunting_condition, np.maximum(0.0, 7.0 - df['holding_time']) / 7.0, 0.0).astype(np.float32)
    
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
    
    # 相対速度 (km/h) ※プラスなら接近している
    df['f_relative_speed'] = df['current_speed'] - df['f_speed']
    df['b_relative_speed'] = df['b_speed'] - df['current_speed']
    
    # ▼▼▼ 【改修4】 f_ttc, b_ttc の算出を削除 ▼▼▼
    
    # 異常値・極端な距離情報のクリッピング（NNのスケーリング崩れ防止）
    df['dist_to_next_station'] = df['dist_to_next_station']
    df['req_stop_dist'] = df['req_stop_dist']
    df['f_distance'] = df['f_distance'].clip(upper=2000.0)
    df['b_distance'] = df['b_distance'].clip(upper=2000.0)
    
    # ▼▼▼ 【改修】学習特徴量の見直し ▼▼▼
    feature_cols = [
        'hold_coast', 'hold_accel', 'hold_decel', 
        'prev_notch_duration',
        'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient', # signal_speed追加
        # 派生特徴量
        'margin_speed', 'margin_signal_speed', 'margin_stop_dist', 'required_speed', 'speed_margin_to_required', 'hunting_score', # margin_signal_speed追加
        'f_relative_speed', 'b_relative_speed', # f_ttc, b_ttcを削除
        'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
        'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
        'f_exist', 'f_distance', 'f_speed',
        'b_exist', 'b_distance', 'b_speed'
    ] + [col for col in df.columns if col.startswith('phase_') or col.startswith('current_notch_') or (col.startswith('prev_notch_') and col != 'prev_notch_duration')]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['reward'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y, feature_cols

def build_model(input_dim):
    # ▼▼▼ 【修正】出力が0~1の範囲から大きく外れる(-1.0付近に張り付く等)不安定な学習を防ぐため、
    #      BatchNormalization・L2正則化・段階的Dropoutで各層の出力レンジを安定させる。
    #      回帰問題のため出力層はsigmoidにせず線形(linear)のまま維持し、
    #      代わりに学習の安定化側で0~1レンジへの収束を促す。
    l2_reg = l2(1e-4)
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, kernel_regularizer=l2_reg),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(64, kernel_regularizer=l2_reg),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Dense(32, kernel_regularizer=l2_reg),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.1),

        Dense(1, activation='linear')
    ])

    # ▼▼▼ 【修正】勾配クリッピングを追加し、外れ値的な入力による重みの急激な発散を抑制 ▼▼▼
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0), metrics=['mae', custom_accuracy])
    return model

def compute_bin_sample_weights(y):
    """
    0.1刻みのラベル区間ごとの出現頻度の逆数を重みとして算出する。
    件数の少ない区間（例: 1.0付近）が損失関数上で過小評価されるのを防ぐ。
    1.0帯の精度を優先するため、緩和(sqrt)や上限クリップは行わず
    単純な逆頻度重み付けとする（0.9帯等が多少過小評価される副作用は許容）。
    重みの平均が1.0になるよう正規化し、学習率など他のハイパーパラメータへの
    影響を最小限に抑える。
    """
    bin_idx = np.clip(np.round(y.flatten() * 10).astype(np.int32), 0, 10)
    bin_counts = np.bincount(bin_idx, minlength=11).astype(np.float32)
    bin_counts[bin_counts == 0] = 1.0  # ゼロ割り防止

    inv_freq = 1.0 / bin_counts
    sample_weight = inv_freq[bin_idx]
    sample_weight = sample_weight / sample_weight.mean()
    return sample_weight.astype(np.float32), bin_idx


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

    # ▼▼▼ 【修正】0.1刻みのラベル区間で層化(stratify)し、
    #      1.0付近など件数の少ない区間が検証データに偏らないようにする ▼▼▼
    _, bin_idx_all = compute_bin_sample_weights(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bin_idx_all
    )
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")

    # ▼▼▼ 【修正】少数派ラベル(1.0付近など)の損失への寄与を高める重み ▼▼▼
    train_sample_weight, _ = compute_bin_sample_weights(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    # ▼▼▼ 【修正】val_lossが停滞したら学習率を下げ、局所解での発散/停滞を防ぐ ▼▼▼
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    print("学習を開始します...")
    history = model.fit(
        X_train_scaled, y_train,
        sample_weight=train_sample_weight,
        validation_data=(X_test_scaled, y_test),
        epochs=500,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    plot_learning_curve(history)
    
    model.save('direct_reward_model2.h5')
    joblib.dump(scaler, 'direct_reward_scaler2.pkl')
    print("モデル('direct_reward_model2.h5')とスケーラー('direct_reward_scaler2.pkl')を保存しました。")

if __name__ == "__main__":
    main()