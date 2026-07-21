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

    # ▼▼▼ 【改修 2026-07-17】停止境界のクリップ済み特徴量 ▼▼▼
    # margin_stop_dist は全データでσ≈775mのスケールになるため、StandardScaler正規化後は
    # 「±5mの停止境界」が0.0064σにしかならず、NNから原理的に見えない
    # （ブレーキ判断マップ・停車完了がΔに対しフラットになる根本原因だった）。
    # 判断に効くレンジだけを切り出したクリップ済み特徴を追加し、境界を可視化する。
    df['margin_stop_dist_clip'] = df['margin_stop_dist'].clip(-30.0, 30.0)   # ブレーキ開始判断レンジ
    df['dist_to_station_clip'] = df['dist_to_next_station'].clip(-20.0, 20.0)  # 停止誤差レンジ
    # 【2026-07-18】停車完了の1m段差（±1m→1.0 / 1〜3m→0.8）用の細クリップ特徴。
    # ±20mクリップではσ≈10mとなり1mの差が0.1σで埋もれる（0.2mでも1.5mでも0.8のフラット化を実測）。
    # ±3mクリップならσ≈2mで1m=約0.5σとなり、境界が通常の学習で見えるようになる。
    df['dist_to_station_clip3'] = df['dist_to_next_station'].clip(-3.0, 3.0)

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

    # 【2026-07-20】保持時間系を30秒でクリップ（DQN観測 environment2.normalized_state と統一）。
    # クリップしないと長い惰行/ブレーキ後（例: 55秒）が hold_* で+12σ、prev_notch_duration で+2.8σの
    # 外れ値になり、ゲート分類器がその領域で誤発火して正当なブレーキに0.0を出す事故が起きた
    # （14500_0.csvで実測。回帰器は0.96と正しいのにゲートのハード閾値で1.0→0.0に反転）。
    # ノコギリ判定（7秒閾値）はhunting_scoreが別途担うため、30秒超の情報損失はない。
    holding_clip = df['holding_time'].clip(upper=30.0)
    df['prev_notch_duration'] = df['prev_notch_duration'].clip(upper=30.0)
    df['hold_coast'] = holding_clip * df['current_notch_惰行中']
    df['hold_accel'] = holding_clip * df['current_notch_力行（加速）中']
    df['hold_decel'] = holding_clip * df['current_notch_ブレーキ（減速）中']
    
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
    # ※並び順は direct_reward_predictor2.py / analyze_reward_nn_vs_llm.py と完全に一致させること
    feature_cols = [
        'hold_coast', 'hold_accel', 'hold_decel',
        'prev_notch_duration',
        'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient', # signal_speed追加
        # 派生特徴量
        'margin_speed', 'margin_signal_speed', 'margin_stop_dist',
        'margin_stop_dist_clip', 'dist_to_station_clip',  # 【2026-07-17追加】停止境界のクリップ済み特徴
        'dist_to_station_clip3',  # 【2026-07-18追加】1m段差用の細クリップ特徴
        'required_speed', 'speed_margin_to_required', 'hunting_score', # margin_signal_speed追加
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

def build_gate_model(input_dim):
    """
    2段階化（ハードルモデル）の1段目：「reward=0.0か否か」を判定する二値分類器。
    回帰NN単体では0.0の山（多数派）と中間値の間で滑らかに補間してしまい、
    本来0.0の状況に0.1〜0.3を漏らす（DQNの搾取の燃料になる）ため、
    ルール的な0.0判定はシャープな境界を引ける分類器に分離する。
    出力は「reward=0.0である確率」（sigmoid）。
    """
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

        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def compute_bin_sample_weights(y):
    """
    0.1刻みのラベル区間ごとの出現頻度の逆数（sqrt緩和）を重みとして算出する。
    件数の少ない区間（例: 0.7付近）が損失関数上で過小評価されるのを防ぐ。

    【改修】旧実装の単純な逆頻度重みでは、多数派の0.0ビン（3,441件）と少数派の
    0.7ビン（22件）の重み比が約156倍にもなり、0.0ラベルのサンプルの誤差がほぼ
    罰されなくなっていた。その結果「本来0.0を出すべき状況で0.1〜0.3を出す」較正崩れが
    生じ、DQNがその漏れ報酬を搾取していた（駅手前ホバリング）。
    1/sqrt(件数) に緩和することで重み比を10倍強程度に抑え、少数ビンへの配慮と
    多数派の予測精度を両立させる。
    重みの平均が1.0になるよう正規化し、学習率など他のハイパーパラメータへの
    影響を最小限に抑える。
    """
    bin_idx = np.clip(np.round(y.flatten() * 10).astype(np.int32), 0, 10)
    bin_counts = np.bincount(bin_idx, minlength=11).astype(np.float32)
    bin_counts[bin_counts == 0] = 1.0  # ゼロ割り防止

    inv_freq = 1.0 / np.sqrt(bin_counts)
    sample_weight = inv_freq[bin_idx]
    sample_weight = sample_weight / sample_weight.mean()
    return sample_weight.astype(np.float32), bin_idx


def plot_learning_curve(history, out_path='learning_curve_direct.png'):
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
    plt.savefig(out_path)
    plt.show()

# 「reward=0.0」とみなすラベルの閾値（浮動小数の丸め誤差対策）
ZERO_THRESHOLD = 0.05

# 【2026-07-19】停車完了フェーズ行の損失重み係数。
# 停車完了は全体の約9%しかなく、回帰の平滑化で頂点が潰れる（教師平均0.957に対し
# 出力0.9止まり・1m段差の位置ずれ）ため、損失への寄与を引き上げて優先的に合わせさせる。
# clip3特徴で境界は「見える」状態になっているので、あとは優先度の問題という位置づけ。
# ゲート分類器にも適用し、停車完了での誤0.0判定（良い停車への外れ値）も抑える。
STOP_PHASE_WEIGHT = 3.0
STOP_PHASE_COL = 'phase_駅停車完了（速度0km/h）'

# 【2026-07-20】停車完了フェーズの1.0（完璧な停止）ラベルの損失重み係数。
# 回帰器は非0.0行のみで学習するため、最多数の1.0ビン（約9,600行）がビン重み(1/√件数)で
# 最小になり、停車完了の1.0が2.6%まで圧縮＝±1m停車の終端報酬勾配が消えていた。
# 【2026-07-21 重要修正】当初は全1.0ラベルに適用したが、1.0ラベルの大半は巡航フェーズであり、
# 巡航のNN出力が0.75→0.98に過度に上昇（ゼロ中心化後+0.48/秒）した結果、報酬スプレッドが消えて
# 生存バイアスが復活しQ値が発散した（run 20260720223422のcyc12900で崩壊）。
# そのため適用対象を「停車完了フェーズの1.0のみ」に限定し、巡航の報酬スプレッドを保つ。
CEILING_LABEL_WEIGHT = 2.5
CEILING_LABEL_THRESHOLD = 0.95

def main(csv_dir='train_reward_csv_direct',
         model_path='direct_reward_model2.h5',
         gate_path='direct_reward_gate2.h5',
         scaler_path='direct_reward_scaler2.pkl',
         plot_path='learning_curve_direct.png',
         epochs=500):
    X, y, feature_cols = load_and_preprocess_data(csv_dir)
    print(f"入力特徴量次元数: {X.shape[1]}")

    # ▼▼▼ 【修正】0.1刻みのラベル区間で層化(stratify)し、
    #      1.0付近など件数の少ない区間が検証データに偏らないようにする ▼▼▼
    _, bin_idx_all = compute_bin_sample_weights(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bin_idx_all
    )
    print(f"学習データ数: {X_train.shape[0]}, テストデータ数: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    # ▼▼▼ 【修正】val_lossが停滞したら学習率を下げ、局所解での発散/停滞を防ぐ ▼▼▼
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    # =====================================================================
    # ▼▼▼ 【改修】2段階化（ハードルモデル） ▼▼▼
    # 1段目（ゲート分類器）: 全データで「reward=0.0か否か」を学習
    # 2段目（回帰器）      : 非0.0のデータのみで「0.0でないなら何点か」を学習
    # 推論時はゲートが0.0と判定したら0.0、そうでなければ回帰器の出力を採用する
    # （direct_reward_predictor2.py側の合成ロジックと対応）。
    # =====================================================================

    # --- 停車完了フェーズ行の識別（サンプル重み用） ---
    stop_col_idx = feature_cols.index(STOP_PHASE_COL)
    train_is_stop = X_train[:, stop_col_idx] >= 0.5
    print(f"\n[サンプル重み] 停車完了フェーズ行: 学習{int(train_is_stop.sum())}件に重み×{STOP_PHASE_WEIGHT}を適用")

    # --- 1段目: ゲート分類器 ---
    y_train_zero = (y_train.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    y_test_zero = (y_test.flatten() <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    print(f"[ゲート分類器] 0.0ラベル: {int(y_train_zero.sum())}件 / 非0.0: {int(len(y_train_zero) - y_train_zero.sum())}件")

    # 停車完了行の判定ミス（良い停車への誤0.0）を抑えるため、ゲートにも同じ重みを適用
    gate_sample_weight = np.where(train_is_stop, STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    gate_sample_weight = gate_sample_weight / gate_sample_weight.mean()

    gate_model = build_gate_model(X_train_scaled.shape[1])
    gate_early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    gate_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    print("ゲート分類器の学習を開始します...")
    gate_model.fit(
        X_train_scaled, y_train_zero,
        sample_weight=gate_sample_weight,
        validation_data=(X_test_scaled, y_test_zero),
        epochs=epochs,
        batch_size=64,
        callbacks=[gate_early_stop, gate_reduce_lr],
        verbose=1
    )

    # --- 2段目: 非0.0のみの回帰器 ---
    nonzero_train = y_train.flatten() > ZERO_THRESHOLD
    nonzero_test = y_test.flatten() > ZERO_THRESHOLD
    X_train_nz = X_train_scaled[nonzero_train]
    y_train_nz = y_train[nonzero_train]
    X_test_nz = X_test_scaled[nonzero_test]
    y_test_nz = y_test[nonzero_test]
    print(f"\n[回帰器] 学習データ数（非0.0のみ）: {X_train_nz.shape[0]}, テストデータ数: {X_test_nz.shape[0]}")

    # ▼▼▼ 少数派ラベル(0.7付近など)の損失への寄与を高める重み（sqrt緩和済み） ▼▼▼
    train_sample_weight_nz, _ = compute_bin_sample_weights(y_train_nz)
    # 停車完了フェーズ行の重みを引き上げ（ビン重みとの積。平均1.0に再正規化して学習率への影響を防ぐ）
    train_sample_weight_nz = train_sample_weight_nz * np.where(train_is_stop[nonzero_train], STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    # 停車完了フェーズの1.0（完璧な停止）ラベルの重みを引き上げ、頂点が0.9に潰れる問題を是正。
    # 【2026-07-21】巡航の1.0まで押し上げるとQ値が発散するため、停車完了フェーズに限定する。
    y_train_nz_flat = y_train_nz.flatten()
    ceiling_mask = (y_train_nz_flat >= CEILING_LABEL_THRESHOLD) & train_is_stop[nonzero_train]
    train_sample_weight_nz = train_sample_weight_nz * np.where(
        ceiling_mask, CEILING_LABEL_WEIGHT, 1.0).astype(np.float32)
    train_sample_weight_nz = train_sample_weight_nz / train_sample_weight_nz.mean()

    model = build_model(X_train_nz.shape[1])
    print("回帰器の学習を開始します...")
    history = model.fit(
        X_train_nz, y_train_nz,
        sample_weight=train_sample_weight_nz,
        validation_data=(X_test_nz, y_test_nz),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    plot_learning_curve(history, plot_path)

    # =====================================================================
    # ▼▼▼ 合成モデルとしての評価（推論時と同じ合成ロジックで検証） ▼▼▼
    # =====================================================================
    gate_prob = gate_model.predict(X_test_scaled, verbose=0).flatten()
    reg_pred = model.predict(X_test_scaled, verbose=0).flatten()
    combined = np.where(gate_prob >= 0.5, 0.0, np.clip(reg_pred, 0.1, 1.0))
    y_test_flat = y_test.flatten()

    mae = np.abs(combined - y_test_flat).mean()
    print(f"\n[合成モデル評価] テストMAE: {mae:.4f}")
    zero_mask = y_test_flat <= ZERO_THRESHOLD
    if zero_mask.any():
        leak = (combined[zero_mask] > 0.25).mean()
        print(f"[合成モデル評価] ラベル0.0のうち予測0.25超（報酬リーク）: {leak:.1%} "
              f"(平均予測 {combined[zero_mask].mean():.3f})")
    if (~zero_mask).any():
        mae_nz = np.abs(combined[~zero_mask] - y_test_flat[~zero_mask]).mean()
        print(f"[合成モデル評価] 非0.0ラベルのMAE: {mae_nz:.4f}")

    model.save(model_path)
    gate_model.save(gate_path)
    joblib.dump(scaler, scaler_path)
    print(f"回帰器('{model_path}')・ゲート分類器('{gate_path}')・スケーラー('{scaler_path}')を保存しました。")

if __name__ == "__main__":
    main()