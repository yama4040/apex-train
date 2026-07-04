"""
LLMラベル分布と報酬予測NN（回帰/分類）の出力分布を比較・可視化するスクリプト。

実装内容①:
    - NN出力分布とLLMラベル分布のヒストグラム比較
    - LLMラベルが0.0の入力に対して、NNが実際にどのような値を出力しているかの分布

実装内容②:
    - train_reward_csv_direct 内の全学習データ（入力）に対する
      LLM出力分布とNN出力分布の比較（0.0〜1.0を0.1刻みで集計）
      回帰モデルは apex2.py と同様に 0〜1 にクリップしたうえで四捨五入する

特徴量の前処理は train_reward_network2.py（回帰）/ train_reward_network3.py（分類）と
完全に一致させている（direct_reward_predictor.py / direct_reward_predictor3.py の
前処理ロジックとも同一）。

【注意】回帰NN（train_reward_network2.py）は required_speed（CSVに保存済みの
物理シミュレーションベースの必要速度。required_speed.py参照）を特徴量として使うが、
分類NN（train_reward_network3.py）は旧来の平均速度×1.3近似（required_speed_mps /
required_cruise_speed）のままであるため、特徴量セットが両モデルで異なる。
そのため本スクリプトでは回帰用・分類用で別々の特徴量行列を構築する。
required_speed列を持たない旧形式CSVの行は回帰側の評価からのみ除外する。
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
# 特徴量抽出（train_reward_network2.py / train_reward_network3.py と共通）
# =====================================================================

CSV_DIR = "train_reward_csv_direct"

REGRESSION_MODEL_PATH = "direct_reward_model.h5"
REGRESSION_SCALER_PATH = "direct_reward_scaler.pkl"

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
    """分類NN（train_reward_network3.py）と完全一致する特徴量行列を構築する"""
    feature_cols = CLASSIFICATION_FEATURE_COLS + _dummy_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    llm_reward = df['reward'].values.astype(np.float32)
    return X, feature_cols, llm_reward


def build_regression_matrix(df):
    """
    回帰NN（train_reward_network2.py）と完全一致する特徴量行列を構築する。
    required_speed列を持たない旧形式CSVの行はNaNになるため除外する。
    """
    if 'required_speed' not in df.columns:
        print("[警告] 全CSVに'required_speed'列が存在しないため、回帰NNの評価はスキップします。")
        print("       最新のapex2.pyで学習データを再収集してください。")
        return None, None, None

    valid = df['required_speed'].notna()
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"[警告] 'required_speed'列がない旧形式データ {n_dropped}行を回帰NNの評価から除外します。")

    df_reg = df[valid]
    if len(df_reg) == 0:
        print("[警告] 'required_speed'列を持つ有効な行が1件もないため、回帰NNの評価はスキップします。")
        return None, None, None

    feature_cols = REGRESSION_FEATURE_COLS + _dummy_cols(df_reg)
    X = df_reg[feature_cols].values.astype(np.float32)
    llm_reward = df_reg['reward'].values.astype(np.float32)
    return X, feature_cols, llm_reward


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

def plot_full_distribution_comparison(llm_reward, rounded_reg, rounded_cls, save_path):
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)

    def counts_on_grid(values):
        counts = []
        for v in grid:
            counts.append(np.sum(np.isclose(values, v, atol=1e-6)))
        return np.array(counts)

    llm_counts = counts_on_grid(llm_reward)

    series = [("LLM Label", llm_counts, 'tab:gray')]
    if rounded_reg is not None:
        series.append(("Regression NN (clipped & rounded)", counts_on_grid(rounded_reg), 'tab:blue'))
    if rounded_cls is not None:
        series.append(("Classification NN (argmax)", counts_on_grid(rounded_cls), 'tab:orange'))

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
    if rounded_reg is not None:
        header += " | 回帰NN"
    if rounded_cls is not None:
        header += " | 分類NN"
    print(header)
    for i, v in enumerate(grid):
        row = f"{v:4.1f} | {llm_counts[i]:5d}"
        if rounded_reg is not None:
            row += f" | {counts_on_grid(rounded_reg)[i]:6d}"
        if rounded_cls is not None:
            row += f" | {counts_on_grid(rounded_cls)[i]:6d}"
        print(row)
    print("=======================================\n")


# =====================================================================
# 可視化: 実装内容①（LLMラベルが0.0の入力に対するNN出力の実態）
# =====================================================================

def plot_zero_label_output(llm_reward, raw_reg, rounded_reg, rounded_cls, save_path):
    mask = np.isclose(llm_reward, 0.0, atol=1e-6)
    n_zero = int(np.sum(mask))
    print(f"LLMラベルが0.0のデータ数: {n_zero} / {len(llm_reward)}")

    if n_zero == 0:
        print("[警告] LLMラベルが0.0のデータが存在しないため、このグラフは生成しません。")
        return

    n_plots = int(raw_reg is not None) + int(rounded_cls is not None)
    if n_plots == 0:
        print("[警告] 評価可能なNNモデルが存在しないため、このグラフは生成しません。")
        return

    plt.figure(figsize=(7 * n_plots, 6))
    plot_idx = 1

    if raw_reg is not None:
        plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1
        raw_zero = raw_reg[mask]
        plt.hist(raw_zero, bins=30, color='tab:blue', edgecolor='black', alpha=0.8)
        plt.axvline(0.0, color='red', linestyle='--', label='LLM Label (0.0)')
        plt.title(f"Regression NN Output when LLM Label = 0.0 (n={n_zero})\nRaw output (before clipping)")
        plt.xlabel("Regression NN Output")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # クリップ・四捨五入後の分布も表示
        rounded_zero = rounded_reg[mask]
        grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
        rounded_counts = [np.sum(np.isclose(rounded_zero, v, atol=1e-6)) for v in grid]
        print("\n--- 回帰NN: LLMラベル0.0時の出力(クリップ&四捨五入後)の内訳 ---")
        for v, c in zip(grid, rounded_counts):
            print(f"  {v:.1f}: {c}件")

    if rounded_cls is not None:
        plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1
        cls_zero = rounded_cls[mask]
        grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
        counts = [np.sum(np.isclose(cls_zero, v, atol=1e-6)) for v in grid]
        plt.bar([f"{v:.1f}" for v in grid], counts, color='tab:orange', edgecolor='black', alpha=0.85)
        plt.title(f"Classification NN Output when LLM Label = 0.0 (n={n_zero})")
        plt.xlabel("Classification NN Output (argmax class x 0.1)")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        print("\n--- 分類NN: LLMラベル0.0時の出力の内訳 ---")
        for v, c in zip(grid, counts):
            print(f"  {v:.1f}: {c}件")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[保存] {save_path}")
    plt.close()


def main():
    df = load_raw_data(CSV_DIR)

    X_cls, cls_feature_cols, llm_reward_cls = build_classification_matrix(df)
    print(f"分類NN向け入力特徴量次元数: {X_cls.shape[1]} (全{len(llm_reward_cls)}行)")

    X_reg, reg_feature_cols, llm_reward_reg = build_regression_matrix(df)
    if X_reg is not None:
        print(f"回帰NN向け入力特徴量次元数: {X_reg.shape[1]} (全{len(llm_reward_reg)}行)")

    reg_model, reg_scaler = load_model_and_scaler(REGRESSION_MODEL_PATH, REGRESSION_SCALER_PATH)
    cls_model, cls_scaler = load_model_and_scaler(CLASSIFICATION_MODEL_PATH, CLASSIFICATION_SCALER_PATH)

    raw_reg, rounded_reg = (None, None)
    if reg_model is not None and X_reg is not None:
        print("回帰NNで推論中...")
        raw_reg, rounded_reg = predict_regression(reg_model, reg_scaler, X_reg)

    rounded_cls = None
    if cls_model is not None:
        print("分類NNで推論中...")
        _, rounded_cls = predict_classification(cls_model, cls_scaler, X_cls)

    # 回帰NNと分類NNで評価対象の行数が一致する場合（required_speed列を全CSVが持つ場合）は
    # 同一のLLMラベル分布と対比できるため1枚のグラフにまとめる。
    # 行数が異なる場合（required_speed列を持たない旧形式CSVが混在する場合）は
    # 対応するLLMラベル分布が別物になるため、モデルごとに別々のグラフを出力する。
    same_dataset = rounded_reg is not None and len(llm_reward_reg) == len(llm_reward_cls)

    if same_dataset:
        plot_full_distribution_comparison(
            llm_reward_cls, rounded_reg, rounded_cls,
            save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_distribution_all.png")
        )
        plot_zero_label_output(
            llm_reward_cls, raw_reg, rounded_reg, rounded_cls,
            save_path=os.path.join(OUTPUT_DIR, "nn_output_for_llm_zero_label.png")
        )
    else:
        if rounded_cls is not None:
            plot_full_distribution_comparison(
                llm_reward_cls, None, rounded_cls,
                save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_distribution_classification.png")
            )
            plot_zero_label_output(
                llm_reward_cls, None, None, rounded_cls,
                save_path=os.path.join(OUTPUT_DIR, "nn_output_for_llm_zero_label_classification.png")
            )
        if rounded_reg is not None:
            plot_full_distribution_comparison(
                llm_reward_reg, rounded_reg, None,
                save_path=os.path.join(OUTPUT_DIR, "nn_vs_llm_distribution_regression.png")
            )
            plot_zero_label_output(
                llm_reward_reg, raw_reg, rounded_reg, None,
                save_path=os.path.join(OUTPUT_DIR, "nn_output_for_llm_zero_label_regression.png")
            )


if __name__ == "__main__":
    main()
