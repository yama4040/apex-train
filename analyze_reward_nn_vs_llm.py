"""
LLMラベル分布と報酬予測NN（新回帰/旧回帰/分類）の出力分布を比較・可視化するスクリプト。

実装内容①:
    - NN出力分布とLLMラベル分布のヒストグラム比較
    - LLMラベルが0.0の入力に対して、NNが実際にどのような値を出力しているかの分布

実装内容②:
    - 学習データ（入力）に対するLLM出力分布とNN出力分布の比較（0.0〜1.0を0.1刻みで集計）
      回帰モデルは apex.py / apex2.py と同様に 0〜1 にクリップしたうえで四捨五入する

本プロジェクトには3系統のNNが存在する。
    - 旧回帰NN（train_reward_network.py → direct_reward_model.h5、apex.pyが使用）:
      必要速度を平均速度×1.3の近似（required_speed_mps / required_cruise_speed）で算出
    - 新回帰NN（train_reward_network2.py → direct_reward_model2.h5、apex2.pyが使用）:
      required_speed（CSVに保存済みの物理シミュレーションベースの必要速度。required_speed.py参照）を使用
    - 分類NN（train_reward_network3.py → classification_reward_model.h5、apex3.pyが使用）:
      旧回帰NNと同じ特徴量セット（required_speed_mps / required_cruise_speed）を使用

旧回帰NNと分類NNは特徴量セットが同一なので同じ特徴量行列（X_cls）で評価できるが、
新回帰NNだけはrequired_speed列を必要とするため、required_speed列を持たない旧形式CSVの
行は新回帰NNの評価からのみ除外する。

実装内容⑤:
    - 次駅減速フェーズ／駅停車完了フェーズについて、evaluate_csv_with_llm.pyのプロンプトに
      明記された閾値ルール（delta_stopや停止誤差など）をLLMを介さずに機械的に再計算し、
      「本来どの評価帯（大幅減点/部分減点/高評価）になるべきか」をリファレンスとして求める。
      これとLLMラベル・NN出力をそれぞれ突き合わせることで、
      「LLMがルールに反した判定（ハルシネーション）をしている疑い」と
      「NNがルールを学習しきれていない疑い」を切り分けられるようにする。
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# =====================================================================
# 特徴量抽出（train_reward_network.py / train_reward_network2.py / train_reward_network3.py と共通）
# =====================================================================

CSV_DIR = "train_reward_csv_direct"

OLD_REGRESSION_MODEL_PATH = "direct_reward_model.h5"
OLD_REGRESSION_SCALER_PATH = "direct_reward_scaler.pkl"

NEW_REGRESSION_MODEL_PATH = "direct_reward_model2.h5"
NEW_REGRESSION_SCALER_PATH = "direct_reward_scaler2.pkl"

CLASSIFICATION_MODEL_PATH = "classification_reward_model.h5"
CLASSIFICATION_SCALER_PATH = "classification_reward_scaler.pkl"

OUTPUT_DIR = "."


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


def load_raw_data(csv_dir):
    """
    train_reward_csv_direct 内の全CSVをそれぞれの実ヘッダのまま読み込む。

    required_speed列の有無でファイルごとに列数が異なるため、固定の列名リストで
    位置決め読み込みをすると列がずれて破損する。各CSV自身のヘッダから列名を
    読み取る方式にしている（train_reward_network2.pyと同じ理由）。
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")

    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")

    df_list = [pd.read_csv(file, encoding='utf-8-sig') for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")

    df['prev_notch'] = df['prev_notch'].replace('なし（または停止）', 'ブレーキ（減速）')
    df['current_notch'] = df['current_notch'].replace('停止・その他', 'ブレーキ（減速）中')

    df['margin_speed'] = df['speed_limit'] - df['current_speed']
    df['margin_stop_dist'] = df['dist_to_next_station'] - df['req_stop_dist']
    df['margin_signal_speed'] = df['signal_speed'] - df['current_speed']

    # 分類NN（train_reward_network3.py）向け：旧来の平均速度×1.3近似（変更なし）
    df['safe_time'] = df['time_to_next_station'].clip(lower=1.0)
    df['required_speed_mps'] = df['dist_to_next_station'] / df['safe_time']
    df['required_cruise_speed'] = (df['required_speed_mps'] * 3.6) * 1.3

    # 回帰NN（train_reward_network2.py）向け：CSVに保存済みのrequired_speedを使用
    # 旧形式（required_speed列なし）のファイルではNaNになるため、回帰側の評価では別途除外する
    if 'required_speed' in df.columns:
        df['speed_margin_to_required'] = df['current_speed'] - df['required_speed']

    hunting_condition = (df['holding_time'] < 5.0) & (df['prev_notch_duration'] < 5.0) & (df['current_notch'] != df['prev_notch'])
    df['hunting_score'] = np.where(hunting_condition, np.maximum(0.0, 5.0 - df['holding_time']) / 5.0, 0.0).astype(np.float32)

    # 誤差分析（フェーズ・特徴量別集計）用に、ダミー変数化で失われる前のラベルを別列として保持しておく
    # ※ '_dummy_cols' が 'phase_' / 'current_notch_' 始まりの列をNN入力として拾うため、
    #   衝突しない列名（group_接頭辞）にしている
    df['group_phase'] = df['phase'].astype(str)
    df['group_notch'] = df['current_notch'].astype(str)
    df['group_delay'] = np.where(df['delay'] <= 0.0, "遅延なし", np.where(df['delay'] <= 10.0, "遅延10秒以内", "遅延10秒超"))
    df['group_hunting'] = np.where(df['hunting_score'] > 0.0, "ノコギリ疑いあり", "なし")

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

    return df


CLASSIFICATION_FEATURE_COLS = [
    'hold_coast', 'hold_accel', 'hold_decel',
    'prev_notch_duration',
    'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
    'margin_speed', 'margin_signal_speed', 'margin_stop_dist', 'required_speed_mps', 'required_cruise_speed', 'hunting_score',
    'f_relative_speed', 'b_relative_speed',
    'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
    'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
    'f_exist', 'f_distance', 'f_speed',
    'b_exist', 'b_distance', 'b_speed'
]

REGRESSION_FEATURE_COLS = [
    'hold_coast', 'hold_accel', 'hold_decel',
    'prev_notch_duration',
    'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station', 'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
    'margin_speed', 'margin_signal_speed', 'margin_stop_dist', 'required_speed', 'speed_margin_to_required', 'hunting_score',
    'f_relative_speed', 'b_relative_speed',
    'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
    'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
    'f_exist', 'f_distance', 'f_speed',
    'b_exist', 'b_distance', 'b_speed'
]


def _dummy_cols(df):
    return [col for col in df.columns if col.startswith('phase_') or col.startswith('current_notch_') or (col.startswith('prev_notch_') and col != 'prev_notch_duration')]


def build_classification_matrix(df):
    """
    分類NN（train_reward_network3.py）および旧回帰NN（train_reward_network.py）と
    完全一致する特徴量行列を構築する（両者は特徴量セットが同一）。
    required_speed列の有無に関わらず全行を評価対象にできる。

    誤差分析用に、Xの各行と対応するdf（group_*列やreasonを含む）も併せて返す。
    """
    feature_cols = CLASSIFICATION_FEATURE_COLS + _dummy_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    llm_reward = df['reward'].values.astype(np.float32)
    return X, feature_cols, llm_reward, df.reset_index(drop=True)


def build_new_regression_matrix(df):
    """
    新回帰NN（train_reward_network2.py）と完全一致する特徴量行列を構築する。
    required_speed列を持たない旧形式CSVの行はNaNになるため除外する。
    """
    if 'required_speed' not in df.columns:
        print("[警告] 全CSVに'required_speed'列が存在しないため、新回帰NN（train_reward_network2.py）の評価はスキップします。")
        print("       最新のapex2.pyで学習データを再収集してください。")
        return None, None, None, None

    valid = df['required_speed'].notna()
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"[警告] 'required_speed'列がない旧形式データ {n_dropped}行を新回帰NNの評価から除外します。")

    df_reg = df[valid]
    if len(df_reg) == 0:
        print("[警告] 'required_speed'列を持つ有効な行が1件もないため、新回帰NNの評価はスキップします。")
        return None, None, None, None

    feature_cols = REGRESSION_FEATURE_COLS + _dummy_cols(df_reg)
    X = df_reg[feature_cols].values.astype(np.float32)
    llm_reward = df_reg['reward'].values.astype(np.float32)
    return X, feature_cols, llm_reward, df_reg.reset_index(drop=True)


# =====================================================================
# NN推論
# =====================================================================

def load_model_and_scaler(model_path, scaler_path):
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"[警告] '{model_path}' または '{scaler_path}' が見つからないため、このモデルの評価はスキップします。")
        return None, None
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_regression(model, scaler, X):
    """回帰NNの出力。apex2.py同様に0〜1へクリップしたうえで0.1刻みに四捨五入する"""
    X_scaled = scaler.transform(X)
    raw = model.predict(X_scaled, verbose=0).flatten()
    clipped = np.clip(raw, 0.0, 1.0)
    rounded = np.round(clipped * 10.0) / 10.0
    return raw, rounded


def predict_classification(model, scaler, X):
    """分類NNの出力。argmaxクラス(0〜10)を0.1刻みの値に変換する"""
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)
    class_idx = np.argmax(probs, axis=1)
    rounded = class_idx.astype(np.float32) * 0.1
    return probs, rounded


# =====================================================================
# 可視化: 実装内容②（全学習データに対するLLM分布とNN分布の比較）
# =====================================================================

def plot_full_distribution_comparison(llm_reward, nn_series, save_path):
    """nn_series: [(label, rounded_values, color), ...] 任意個のNN出力系列と比較する"""
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)

    def counts_on_grid(values):
        counts = []
        for v in grid:
            counts.append(np.sum(np.isclose(values, v, atol=1e-6)))
        return np.array(counts)

    llm_counts = counts_on_grid(llm_reward)

    series = [("LLM Label", llm_counts, 'tab:gray')]
    for label, values, color in nn_series:
        series.append((label, counts_on_grid(values), color))

    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(len(grid))

    plt.figure(figsize=(14, 6))
    for i, (label, counts, color) in enumerate(series):
        offset = (i - (n_series - 1) / 2.0) * bar_width
        plt.bar(x + offset, counts, width=bar_width, label=label, color=color, edgecolor='black', alpha=0.85)

    plt.xticks(x, [f"{v:.1f}" for v in grid])
    plt.xlabel("Reward Value (0.1 step)")
    plt.ylabel("Count")
    plt.title("Training Data: LLM Label Distribution vs NN Output Distribution")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[保存] {save_path}")
    plt.close()

    # コンソールにも件数の表を出力
    print("\n=== 学習データ全体の分布 (0.1刻み) ===")
    header = " 値  | LLM"
    for label, _, _ in series[1:]:
        header += f" | {label}"
    print(header)
    for i, v in enumerate(grid):
        row = f"{v:4.1f} | {llm_counts[i]:5d}"
        for _, counts, _ in series[1:]:
            row += f" | {counts[i]:6d}"
        print(row)
    print("=======================================\n")


# =====================================================================
# 可視化: 実装内容①（LLMラベルが0.0の入力に対するNN出力の実態）
# =====================================================================

def plot_zero_label_output(llm_reward, continuous_series, discrete_series, save_path):
    """
    continuous_series: [(label, raw_values, rounded_values, color), ...] 回帰系（連続値）のNN出力
    discrete_series:   [(label, rounded_values, color), ...] 分類系（離散値）のNN出力
    """
    mask = np.isclose(llm_reward, 0.0, atol=1e-6)
    n_zero = int(np.sum(mask))
    print(f"LLMラベルが0.0のデータ数: {n_zero} / {len(llm_reward)}")

    if n_zero == 0:
        print("[警告] LLMラベルが0.0のデータが存在しないため、このグラフは生成しません。")
        return

    n_plots = len(continuous_series) + len(discrete_series)
    if n_plots == 0:
        print("[警告] 評価可能なNNモデルが存在しないため、このグラフは生成しません。")
        return

    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
    plt.figure(figsize=(7 * n_plots, 6))
    plot_idx = 1

    for label, raw_values, rounded_values, color in continuous_series:
        plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1
        raw_zero = raw_values[mask]
        plt.hist(raw_zero, bins=30, color=color, edgecolor='black', alpha=0.8)
        plt.axvline(0.0, color='red', linestyle='--', label='LLM Label (0.0)')
        plt.title(f"{label} Output when LLM Label = 0.0 (n={n_zero})\nRaw output (before clipping)")
        plt.xlabel(f"{label} Output")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # クリップ・四捨五入後の分布も表示
        rounded_zero = rounded_values[mask]
        rounded_counts = [np.sum(np.isclose(rounded_zero, v, atol=1e-6)) for v in grid]
        print(f"\n--- {label}: LLMラベル0.0時の出力(クリップ&四捨五入後)の内訳 ---")
        for v, c in zip(grid, rounded_counts):
            print(f"  {v:.1f}: {c}件")

    for label, rounded_values, color in discrete_series:
        plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1
        values_zero = rounded_values[mask]
        counts = [np.sum(np.isclose(values_zero, v, atol=1e-6)) for v in grid]
        plt.bar([f"{v:.1f}" for v in grid], counts, color=color, edgecolor='black', alpha=0.85)
        plt.title(f"{label} Output when LLM Label = 0.0 (n={n_zero})")
        plt.xlabel(f"{label} Output (argmax class x 0.1)")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        print(f"\n--- {label}: LLMラベル0.0時の出力の内訳 ---")
        for v, c in zip(grid, counts):
            print(f"  {v:.1f}: {c}件")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[保存] {save_path}")
    plt.close()


# =====================================================================
# 可視化: 実装内容③（LLM出力 × NN出力の件数ヒートマップ）
# =====================================================================

def plot_confusion_heatmap(llm_reward, nn_rounded, label, save_path):
    """
    横軸=LLMラベル、縦軸=NN出力（いずれも0.0〜1.0を0.1刻み）の件数ヒートマップ。
    NNの出力がLLMラベルに対してどの値域に散らばっているかを一望できる。
    対角線上に集中していれば一致度が高く、対角線から外れて分布していれば
    NNがLLMのラベルとズレた値を出力していることを示す。
    """
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
    n = len(grid)

    def to_bin_idx(values):
        return np.clip(np.round(values * 10).astype(np.int32), 0, 10)

    llm_idx = to_bin_idx(llm_reward)
    nn_idx = to_bin_idx(nn_rounded)

    counts = np.zeros((n, n), dtype=np.int64)
    for li, ni in zip(llm_idx, nn_idx):
        counts[ni, li] += 1  # 行=NN出力、列=LLMラベル

    fig, ax = plt.subplots(figsize=(9, 8))
    # 件数（量）のエンコーディングなので単色濃淡（sequential, light→dark）を使う。レインボー配色は使わない。
    im = ax.imshow(counts, cmap='Blues', origin='lower', aspect='auto')

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"{v:.1f}" for v in grid])
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"{v:.1f}" for v in grid])
    ax.set_xlabel("LLM Label")
    ax.set_ylabel(f"{label} Output")
    ax.set_title(f"{label} vs LLM Label: Count Heatmap (n={len(llm_reward)})")

    # セルごとに件数を注記。背景が濃い場合は白文字にして視認性を確保する
    vmax = counts.max() if counts.max() > 0 else 1
    for ni in range(n):
        for li in range(n):
            v = counts[ni, li]
            if v == 0:
                continue
            text_color = 'white' if v > vmax * 0.5 else 'black'
            ax.text(li, ni, str(v), ha='center', va='center', color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[保存] {save_path}")
    plt.close()


# =====================================================================
# 実装内容④: フェーズ・特徴量別の誤差集計 ＆ 誤差の大きい行の抽出
# =====================================================================

# 誤差をグルーピングして集計する対象の列（load_raw_data で作成した group_* 列）
ERROR_GROUP_COLS = [
    ("group_phase", "走行フェーズ"),
    ("group_notch", "現在のノッチ"),
    ("group_delay", "遅延状況"),
    ("group_hunting", "ノコギリ運転疑い"),
]

# 誤差の大きい行を書き出す際に含める列（存在するものだけを実際に出力する）
MISMATCH_EXPORT_COLS = [
    "time", "train_id", "group_phase", "group_notch", "group_delay", "group_hunting",
    "current_speed", "speed_limit", "signal_speed", "dist_to_next_station", "time_to_next_station",
    "delay", "holding_time", "prev_notch_duration", "hunting_score",
    "llm_reward", "nn_output", "abs_error", "reason"
]


def analyze_error_by_group(df_subset, llm_reward, nn_rounded, label):
    """
    NN出力とLLMラベルの絶対誤差を、フェーズ・ノッチ・遅延状況・ノコギリ運転疑いの
    各カテゴリ別に集計し、誤差が大きいシチュエーションを定量的に洗い出す。
    """
    analysis_df = df_subset.copy()
    analysis_df['llm_reward'] = llm_reward
    analysis_df['nn_output'] = nn_rounded
    analysis_df['abs_error'] = np.abs(nn_rounded - llm_reward)

    print(f"\n=== {label}: 全体の平均絶対誤差(MAE) = {analysis_df['abs_error'].mean():.4f} (n={len(analysis_df)}) ===")

    for col, jp_name in ERROR_GROUP_COLS:
        if col not in analysis_df.columns:
            continue
        grouped = (
            analysis_df.groupby(col, observed=True)['abs_error']
            .agg(平均絶対誤差='mean', 件数='count')
            .sort_values('平均絶対誤差', ascending=False)
        )
        print(f"\n--- {label}: 「{jp_name}」別の平均絶対誤差（誤差が大きい順） ---")
        print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    return analysis_df


def export_worst_mismatches(analysis_df, label, save_path, top_n=50):
    """誤差(|NN出力 - LLMラベル|)が大きい行を上位top_n件CSVに書き出す（reason列があれば含める）。"""
    cols = [c for c in MISMATCH_EXPORT_COLS if c in analysis_df.columns]
    worst = analysis_df.sort_values('abs_error', ascending=False).head(top_n)
    worst[cols].to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] {label}: 誤差が大きい上位{min(top_n, len(worst))}行を '{save_path}' に書き出しました。")
    print("       LLMの判定根拠（reason列）と見比べて、NNが見落としているルールを確認してください。")


# =====================================================================
# 実装内容⑤: 次駅減速フェーズ／駅停車完了フェーズのルールベース検証
#   evaluate_csv_with_llm.py のプロンプトに明記された閾値ルールをLLMを介さず
#   機械的に再計算し、「LLMのハルシネーション疑い」と「NNの学習不足疑い」を切り分ける。
# =====================================================================

RULE_CHECK_EXPORT_COLS = [
    "time", "train_id", "group_phase", "group_notch", "current_speed", "signal_speed", "speed_limit",
    "dist_to_next_station", "req_stop_dist", "delta_stop", "delay", "f_exist", "f_distance",
    "rule_bucket", "rule_reason", "llm_bucket", "nn_bucket", "llm_reward", "nn_output", "reason"
]


def compute_rule_based_expected(df):
    """
    次駅減速フェーズ／駅停車完了フェーズについて、evaluate_csv_with_llm.pyのプロンプトに
    明記された閾値ルール（delta_stop = dist_to_next_station - req_stop_dist、停止誤差など）から、
    LLMを介さず「本来どの評価帯になるべきか」を機械的に再計算する。

    戻り値の rule_bucket は 'zero'（大幅減点=0.0が妥当）/ 'mid'（部分減点が妥当）/
    'high'（高評価が妥当）/ None（このルールでは判定できない＝他の評価基準に委ねる）のいずれか。
    ノコギリ運転など「合理的理由の有無」を要する主観的なルールは対象外（機械的に判定不能なため）。
    """
    df = df.copy()
    delta_stop = df['dist_to_next_station'] - df['req_stop_dist']
    df['delta_stop'] = delta_stop

    is_braking = df['group_notch'] == 'ブレーキ（減速）中'
    is_accel = df['group_notch'] == '力行（加速）中'
    is_coast = df['group_notch'] == '惰行中'
    has_forward = df['f_exist'] > 0.5
    forward_close = has_forward & (df['f_distance'] < 600.0)

    bucket = pd.Series([None] * len(df), index=df.index, dtype=object)
    reason = pd.Series([""] * len(df), index=df.index, dtype=object)

    decel_mask = df['group_phase'] == '次駅への減速フェーズ（駅手前400m以内）'
    stop_mask = df['group_phase'] == '駅停車完了（速度0km/h）'

    def set_bucket(mask, value, why):
        bucket[mask] = value
        reason[mask] = why

    # ---- 駅停車完了フェーズ ----
    stop_dist_abs = df['dist_to_next_station'].abs()
    set_bucket(stop_mask & (stop_dist_abs <= 1.0), 'high', '停止誤差1m以内')
    set_bucket(stop_mask & (stop_dist_abs > 1.0) & (stop_dist_abs <= 10.0), 'mid', '停止誤差1〜10mの範囲（段階的減点）')
    set_bucket(stop_mask & (stop_dist_abs > 10.0) & (df['signal_speed'] > 0.0), 'zero', '停止誤差10m超（signal_speed>0）')
    set_bucket(stop_mask & (stop_dist_abs > 10.0) & (df['signal_speed'] <= 0.0), 'high', '停止誤差10m超だがsignal_speed=0（先行列車衝突回避）')

    # ---- 次駅減速フェーズ：先行列車が近くにいる場合（600m未満）を優先評価 ----
    set_bucket(
        decel_mask & forward_close & ((df['current_speed'] - df['signal_speed']).abs() <= 2.0) & is_braking,
        'high', '先行列車接近中、信号速度に整合してブレーキ中'
    )
    set_bucket(
        decel_mask & forward_close & (df['current_speed'] > df['signal_speed']) & ~is_braking,
        'zero', '先行列車接近中、信号超過かつブレーキなし（信号無視）'
    )
    set_bucket(
        decel_mask & forward_close & (delta_stop > 0) & is_accel & (df['current_speed'] <= df['signal_speed'] + 2.0),
        'high', '先行列車解放に伴う加速（適切）'
    )

    # ---- 次駅減速フェーズ：先行列車がいない／十分離れている場合 ----
    far_or_none = decel_mask & ~forward_close
    set_bucket(far_or_none & is_braking & (delta_stop.abs() <= 2.0), 'high', 'ブレーキ中、|delta_stop|<=2m')
    set_bucket(far_or_none & is_braking & (delta_stop.abs() > 2.0) & (delta_stop.abs() <= 5.0), 'mid', 'ブレーキ中、2m<|delta_stop|<=5m')
    set_bucket(far_or_none & is_braking & (delta_stop.abs() > 5.0), 'zero', 'ブレーキ中、|delta_stop|>5m')

    set_bucket(far_or_none & ~is_braking & (delta_stop >= 0.0), 'high', 'ブレーキなし、delta_stop>=0m')
    set_bucket(far_or_none & ~is_braking & (delta_stop < 0.0) & (delta_stop >= -2.0), 'mid', 'ブレーキなし、-2m<=delta_stop<0m')
    set_bucket(far_or_none & ~is_braking & (delta_stop < -2.0), 'zero', 'ブレーキなし、delta_stop<-2m（オーバーランリスク）')
    # 例外: 先行列車なし・遅延ありで低速惰行（遅延助長）は上の「delta_stop>=0m→high」を上書きする
    set_bucket(
        far_or_none & is_coast & (df['delay'] > 0.0) & (df['current_speed'] < 20.0) & (delta_stop >= 0.0),
        'zero', '先行列車なし・遅延ありで低速惰行（遅延助長）'
    )

    df['rule_bucket'] = bucket
    df['rule_reason'] = reason
    return df


def _value_to_bucket(values):
    """reward値(0.0〜1.0)を 'zero'（<=0.05）/ 'high'（>=0.7）/ 'mid'（それ以外）の3帯に分類する"""
    values = np.asarray(values, dtype=np.float32)
    return np.select([values <= 0.05, values >= 0.7], ['zero', 'high'], default='mid')


def analyze_rule_based_decel_check(df_subset, llm_reward, nn_rounded, label, save_dir=OUTPUT_DIR):
    """
    次駅減速フェーズ／駅停車完了フェーズについて、ルールベースの期待値とLLM/NNの出力を突き合わせる。
    - ルールと不一致のLLMラベル → LLMがルールに反した判定（ハルシネーション）をしている疑い
    - LLMはルールと一致しているのにNNだけ不一致 → NNがルールを学習しきれていない疑い
    """
    df_rule = compute_rule_based_expected(df_subset)
    df_rule['llm_reward'] = llm_reward
    df_rule['nn_output'] = nn_rounded
    df_rule['llm_bucket'] = _value_to_bucket(llm_reward)
    df_rule['nn_bucket'] = _value_to_bucket(nn_rounded)

    target = df_rule[df_rule['rule_bucket'].notna()].copy()
    print(f"\n=== {label}: ルール判定可能な行数 = {len(target)} / {len(df_rule)}（次駅減速・駅停車完了フェーズのみ対象） ===")

    if len(target) == 0:
        print("[警告] ルール判定可能な行がないため、この分析はスキップします。")
        return

    llm_match = target['llm_bucket'] == target['rule_bucket']
    nn_match = target['nn_bucket'] == target['rule_bucket']
    print(f"LLMラベルがルールと一致: {llm_match.mean() * 100:.1f}% ({llm_match.sum()}/{len(target)})")
    print(f"NN出力がルールと一致:   {nn_match.mean() * 100:.1f}% ({nn_match.sum()}/{len(target)})")

    cols = [c for c in RULE_CHECK_EXPORT_COLS if c in target.columns]

    llm_suspect = target[~llm_match]
    print(f"\n--- LLMがルールに反した判定をしている疑いのある行: {len(llm_suspect)}件 ---")
    if len(llm_suspect) > 0:
        print(llm_suspect['rule_bucket'].value_counts().rename('ルール上の正解帯（実際はズレ）').to_string())
        path = os.path.join(save_dir, f"rule_check_llm_suspect_{label}.csv")
        llm_suspect[cols].to_csv(path, index=False, encoding='utf-8-sig')
        print(f"[保存] {path}")

    nn_suspect = target[llm_match & ~nn_match]
    print(f"\n--- LLMは正しいがNNがルールに反した出力をしている疑いのある行: {len(nn_suspect)}件 ---")
    if len(nn_suspect) > 0:
        print(nn_suspect['rule_bucket'].value_counts().rename('ルール上の正解帯（NNがズレ）').to_string())
        path = os.path.join(save_dir, f"rule_check_nn_suspect_{label}.csv")
        nn_suspect[cols].to_csv(path, index=False, encoding='utf-8-sig')
        print(f"[保存] {path}")


def main():
    df = load_raw_data(CSV_DIR)

    # 旧回帰NN・分類NNは特徴量セットが同一のため同じ行列（X_cls、全行）を共有する
    X_cls, cls_feature_cols, llm_reward_cls, df_cls = build_classification_matrix(df)
    print(f"旧回帰NN/分類NN向け入力特徴量次元数: {X_cls.shape[1]} (全{len(llm_reward_cls)}行)")

    # 新回帰NNはrequired_speed列が必要なため、対応する行のみの別行列になる
    X_new_reg, new_reg_feature_cols, llm_reward_new_reg, df_new_reg = build_new_regression_matrix(df)
    if X_new_reg is not None:
        print(f"新回帰NN向け入力特徴量次元数: {X_new_reg.shape[1]} (全{len(llm_reward_new_reg)}行)")

    old_reg_model, old_reg_scaler = load_model_and_scaler(OLD_REGRESSION_MODEL_PATH, OLD_REGRESSION_SCALER_PATH)
    new_reg_model, new_reg_scaler = load_model_and_scaler(NEW_REGRESSION_MODEL_PATH, NEW_REGRESSION_SCALER_PATH)
    cls_model, cls_scaler = load_model_and_scaler(CLASSIFICATION_MODEL_PATH, CLASSIFICATION_SCALER_PATH)

    raw_old_reg, rounded_old_reg = (None, None)
    if old_reg_model is not None:
        print("旧回帰NN（train_reward_network.py）で推論中...")
        raw_old_reg, rounded_old_reg = predict_regression(old_reg_model, old_reg_scaler, X_cls)

    raw_new_reg, rounded_new_reg = (None, None)
    if new_reg_model is not None and X_new_reg is not None:
        print("新回帰NN（train_reward_network2.py）で推論中...")
        raw_new_reg, rounded_new_reg = predict_regression(new_reg_model, new_reg_scaler, X_new_reg)

    rounded_cls = None
    if cls_model is not None:
        print("分類NN（train_reward_network3.py）で推論中...")
        _, rounded_cls = predict_classification(cls_model, cls_scaler, X_cls)

    # 旧回帰NN・分類NNは常にX_cls（全行）を共有するため同じLLMラベル分布と対比できる。
    # 新回帰NNは、required_speed列を全CSVが持つ場合のみ同じ行数になり同じグラフにまとめられる。
    # 行数が異なる場合（required_speed列を持たない旧形式CSVが混在する場合）は別グラフにする。
    new_reg_matches_full = rounded_new_reg is not None and len(llm_reward_new_reg) == len(llm_reward_cls)

    full_continuous = []
    full_discrete = []
    if rounded_old_reg is not None:
        full_continuous.append(("Old Regression NN", raw_old_reg, rounded_old_reg, 'tab:blue'))
    if new_reg_matches_full:
        full_continuous.append(("New Regression NN", raw_new_reg, rounded_new_reg, 'tab:green'))
    if rounded_cls is not None:
        full_discrete.append(("Classification NN", rounded_cls, 'tab:orange'))

    if full_continuous or full_discrete:
        plot_full_distribution_comparison(
            llm_reward_cls,
            [(label, rounded, color) for label, _, rounded, color in full_continuous] +
            [(label, rounded, color) for label, rounded, color in full_discrete],
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_distribution_all.png")
        )
        plot_zero_label_output(
            llm_reward_cls, full_continuous, full_discrete,
            save_path=os.path.join(OUTPUT_DIR, "nn_output_for_llm_zero_label.png")
        )

    # 新回帰NNが旧回帰NN/分類NNと行数不一致の場合は、新回帰NN単独のグラフを別途出力する
    if rounded_new_reg is not None and not new_reg_matches_full:
        plot_full_distribution_comparison(
            llm_reward_new_reg,
            [("New Regression NN", rounded_new_reg, 'tab:green')],
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_distribution_new_regression.png")
        )
        plot_zero_label_output(
            llm_reward_new_reg,
            [("New Regression NN", raw_new_reg, rounded_new_reg, 'tab:green')], [],
            save_path=os.path.join(OUTPUT_DIR, "nn_output_for_llm_zero_label_new_regression.png")
        )

    # 実装内容③: LLM出力 × NN出力の件数ヒートマップ（モデルごとに、対応するLLMラベルと組で作成）
    if rounded_old_reg is not None:
        plot_confusion_heatmap(
            llm_reward_cls, rounded_old_reg, "Old Regression NN",
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_heatmap_old_regression.png")
        )
    if rounded_new_reg is not None:
        plot_confusion_heatmap(
            llm_reward_new_reg, rounded_new_reg, "New Regression NN",
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_heatmap_new_regression.png")
        )
    if rounded_cls is not None:
        plot_confusion_heatmap(
            llm_reward_cls, rounded_cls, "Classification NN",
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_heatmap_classification.png")
        )

    # 実装内容④: フェーズ・特徴量別の誤差集計 ＆ 誤差の大きい行の抽出
    if rounded_old_reg is not None:
        analysis_df = analyze_error_by_group(df_cls, llm_reward_cls, rounded_old_reg, "Old Regression NN")
        export_worst_mismatches(
            analysis_df, "Old Regression NN",
            save_path=os.path.join(OUTPUT_DIR, "mismatch_old_regression.csv")
        )
    if rounded_new_reg is not None:
        analysis_df = analyze_error_by_group(df_new_reg, llm_reward_new_reg, rounded_new_reg, "New Regression NN")
        export_worst_mismatches(
            analysis_df, "New Regression NN",
            save_path=os.path.join(OUTPUT_DIR, "mismatch_new_regression.csv")
        )
    if rounded_cls is not None:
        analysis_df = analyze_error_by_group(df_cls, llm_reward_cls, rounded_cls, "Classification NN")
        export_worst_mismatches(
            analysis_df, "Classification NN",
            save_path=os.path.join(OUTPUT_DIR, "mismatch_classification.csv")
        )

    # 実装内容⑤: 次駅減速フェーズ／駅停車完了フェーズのルールベース検証
    #   （LLMのハルシネーション疑い と NNの学習不足疑い の切り分け）
    if rounded_old_reg is not None:
        analyze_rule_based_decel_check(df_cls, llm_reward_cls, rounded_old_reg, "OldRegression")
    if rounded_new_reg is not None:
        analyze_rule_based_decel_check(df_new_reg, llm_reward_new_reg, rounded_new_reg, "NewRegression")
    if rounded_cls is not None:
        analyze_rule_based_decel_check(df_cls, llm_reward_cls, rounded_cls, "Classification")


if __name__ == "__main__":
    main()
