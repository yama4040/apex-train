# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の状態スキーマと特徴量エンジニアリング。

既存 `reward_features.py` は**一切変更しない**（apex2.py 系がそのまま使い続ける）。
本モジュールは以下の4者が共有する**唯一の正準スキーマ**である。

    environment_ymulti.py      … RL実行時に生の状態辞書を作る
    generate_eval_csv_ymulti.py… LLM評価用の走行ログCSVを作る（列名＝RAW_COLS）
    prompt_ymulti.py           … 生の状態辞書からLLMプロンプト本文を組み立てる
    train_reward_network_ymulti.py / direct_reward_predictor_ymulti.py … NNの入出力

既存版との主な違い
  * **駅停車中の状態量**（is_dwelling / dwell_elapsed / dwell_over_std / 発車可否）を持つ
  * **複数駅間の通算ダイヤ**（total_remaining_time / total_delay / stations_remaining）を持つ
  * **標準運転曲線 v_std との速度差**を持つ（区間ごとの基準運転からの逸脱量）
  * **惰行到達可能性**を持つ（白兎→蚕桑の +11.4‰ が 1km 続くため、惰行の可否が本質的）
  * 運転モードに **hold_at_station**（駅停車中の発車判断）を追加
  * 後続列車の列（b_*）は**枠だけ確保**して既定値（後続なし）で埋める。
    後続列車を導入するときに列を足さずに済ませ、モデル世代の断絶を避けるため。

【重要】既存の `評価用csv/` `train_reward_csv_direct/` とは**絶対に混ぜない**。
3ノッチである点は同じだが、列構成・モード定義・プロンプト世代が異なる。
"""
import re

import numpy as np

import config_ymulti as CFG

MODE_CLASSES = CFG.MODE_CLASSES
MODE_CLASSES_ACTIVE = CFG.MODE_CLASSES_ACTIVE
MODE_DIM = CFG.MODE_DIM


def onehot_index(mode_str):
    m = (mode_str or "normal").strip()
    return MODE_CLASSES.index(m) if m in MODE_CLASSES else 0


def mode_to_onehot(mode_str):
    v = np.zeros(MODE_DIM, dtype=np.float32)
    v[onehot_index(mode_str)] = 1.0
    return v


# =============================================================================
# 生の状態辞書の正準スキーマ（＝LLM評価用CSVの列）
# =============================================================================
RAW_COLS = [
    # --- 識別・時刻 ---
    "run_id", "time", "section", "position",
    # --- フェーズ・ノッチ ---
    "phase", "current_notch", "holding_time", "prev_notch", "prev_notch_duration",
    # --- 速度 ---
    "current_speed", "speed_limit", "signal_speed",
    "required_speed", "target_speed_no_stop", "v_std", "v_std_deviation",
    # --- 次駅まで ---
    "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delta_stop",
    # --- 勾配 ---
    "current_gradient", "coast_accel", "power_accel",
    "next_limit_info", "next_gradient_info",
    # --- 惰行到達可能性（上り勾配での失速判定） ---
    "coast_reachable", "coast_arrival_speed",
    # --- 遅延・通算ダイヤ ---
    "delay", "total_delay", "stations_remaining",
    "total_remaining_distance", "total_remaining_time",
    # --- 駅停車（発車判断） ---
    "is_dwelling", "dwell_elapsed", "dwell_min", "dwell_max", "dwell_over_std",
    "time_to_stop_limit",
    # --- 先行列車 ---
    "forward_info", "forward_train_delay", "standard_headway",
    "forward_clear_remaining_time", "forward_observed_delay",
    "forward_dwell_elapsed", "forward_departed_next",
    # --- 後続列車（枠のみ・本フェーズでは常に「後続列車なし」） ---
    "backward_info",
    # --- LLMの出力 ---
    "mode", "reward", "reason",
]

# LLMが埋める列（生成側は空にしておく）
LLM_OUTPUT_COLS = ["mode", "reward", "reason"]


# =============================================================================
# テキスト情報のパース
# =============================================================================
def extract_limit_info(text):
    text = str(text)
    if "この先制限速度なし" in text:
        return 0.0, 0.0, 0.0
    m = re.search(r"(\d+)m先に制限速度(\d+)km/h", text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    return 0.0, 0.0, 0.0


def extract_gradient_info(text):
    text = str(text)
    if "この先目立った勾配なし" in text:
        return 0.0, 0.0, 0.0
    m = re.search(r"(\d+)m先に(上り|下り)勾配(\d+\.?\d*)‰あり", text)
    if m:
        val = float(m.group(3))
        if m.group(2) == "下り":
            val = -val
        return 1.0, float(m.group(1)), val
    m2 = re.search(r"(\d+)m先で(?:上り|下り)勾配\d+\.?\d*‰が終わり平坦になる", text)
    if m2:
        return 1.0, float(m2.group(1)), 0.0
    return 0.0, 0.0, 0.0


def extract_forward_info(text):
    text = str(text)
    if "先行列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0
    m = re.search(r"前方\s*([\d\.]+)\s*m\s*先を\s*([\d\.]+)\s*km/h", text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    m = re.search(r"前方\s*([\d\.]+)\s*m\s*先.*停車中", text)
    if m:
        return 1.0, float(m.group(1)), 0.0
    return 0.0, 5000.0, 0.0


def extract_backward_info(text):
    text = str(text)
    if "後続列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0
    m = re.search(r"後方\s*([\d\.]+)\s*m\s*後ろを\s*([\d\.]+)\s*km/h", text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    return 0.0, 5000.0, 0.0


def _f(raw, key, default=0.0):
    v = raw.get(key, default)
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if (isinstance(f, float) and f != f) else f    # NaN を既定値へ


def _s(raw, key):
    v = raw.get(key, "")
    return "" if v is None else str(v)


# =============================================================================
# フェーズ・ノッチの語彙（プロンプト・CSV・NNで共有）
# =============================================================================
PHASES = [
    "駅停車中（発車判断）",
    "駅出発直後の加速フェーズ（20秒以内）",
    "巡航フェーズ（駅間走行中）",
    "制限速度区間に接近中（500m以内に制限区間在り）",
    "次駅への減速フェーズ（駅手前400m以内）",
    "駅停車完了（速度0km/h）",
]
NOTCH_CURRENT = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
NOTCH_PREV = ["惰行", "力行（加速）", "ブレーキ（減速）"]
# 停車中はノッチを「発車 / 待機」と読み替える（3ノッチのうち惰行は禁止）
DWELL_NOTCH = {"力行（加速）中": "発車", "ブレーキ（減速）中": "待機（停止保持）"}


# =============================================================================
# 状態特徴量の正準な並び順
# =============================================================================
# ※この順序を変えると学習済みモデルと不整合になる。追加は必ず末尾に行うこと。
STATE_FEATURE_COLS = [
    # 直前操作
    "hold_coast", "hold_accel", "hold_decel", "prev_notch_duration",
    # 速度
    "speed_limit", "signal_speed", "current_speed",
    "margin_speed", "margin_signal_speed",
    "required_speed", "speed_margin_to_required",
    "target_speed_no_stop", "speed_margin_to_target",
    "v_std", "v_std_deviation",
    # 次駅まで
    "dist_to_next_station", "time_to_next_station", "req_stop_dist",
    "margin_stop_dist", "margin_stop_dist_clip",
    "dist_to_station_clip", "dist_to_station_clip3",
    # 勾配・惰行
    "current_gradient", "coast_accel", "power_accel",
    "coast_reachable", "coast_arrival_speed",
    "next_limit_flag", "next_limit_dist", "next_limit_speed",
    "next_gradient_flag", "next_gradient_dist", "next_gradient_val",
    # 遅延・通算ダイヤ
    "delay", "total_delay", "stations_remaining",
    "total_remaining_distance", "total_remaining_time",
    # 駅停車（発車判断）
    "is_dwelling", "dwell_elapsed", "dwell_over_std",
    "dwell_departable", "dwell_forced",
    "time_to_stop_limit", "clear_minus_reach",
    # 先行列車
    "f_exist", "f_distance", "f_speed", "f_relative_speed",
    "forward_clear_remaining_time", "forward_observed_delay", "forward_dwell_elapsed",
    "forward_train_delay", "forward_departed_flag", "standard_headway",
    # 後続列車（枠のみ・既定は後続なし）
    "b_exist", "b_distance", "b_speed", "b_relative_speed",
    # ノコギリ検出
    "hunting_score",
    # フェーズのone-hot
    "phase_駅停車中（発車判断）",
    "phase_駅出発直後の加速フェーズ（20秒以内）",
    "phase_巡航フェーズ（駅間走行中）",
    "phase_制限速度区間に接近中（500m以内に制限区間在り）",
    "phase_次駅への減速フェーズ（駅手前400m以内）",
    "phase_駅停車完了（速度0km/h）",
    # ノッチのone-hot
    "current_notch_惰行中",
    "current_notch_力行（加速）中",
    "current_notch_ブレーキ（減速）中",
    "prev_notch_惰行",
    "prev_notch_力行（加速）",
    "prev_notch_ブレーキ（減速）",
]
STATE_DIM = len(STATE_FEATURE_COLS)


def engineer_features(raw):
    """生の状態辞書 raw → named特徴量辞書（STATE_FEATURE_COLS の各キーを含む）。"""
    current_notch = _s(raw, "current_notch")
    prev_notch = _s(raw, "prev_notch")
    if prev_notch in ("なし（または停止）", ""):
        prev_notch = "ブレーキ（減速）"
    phase = _s(raw, "phase")

    is_coast = 1.0 if current_notch == "惰行中" else 0.0
    is_accel = 1.0 if current_notch == "力行（加速）中" else 0.0
    is_decel = 1.0 if current_notch == "ブレーキ（減速）中" else 0.0

    speed_limit = _f(raw, "speed_limit")
    signal_speed = _f(raw, "signal_speed")
    speed = _f(raw, "current_speed")
    required_speed = _f(raw, "required_speed")
    target_ns = _f(raw, "target_speed_no_stop", required_speed)
    v_std = _f(raw, "v_std")
    dist_next = _f(raw, "dist_to_next_station")
    time_next = _f(raw, "time_to_next_station")
    req_stop = _f(raw, "req_stop_dist")
    holding_time = _f(raw, "holding_time")
    prev_dur = _f(raw, "prev_notch_duration")

    margin_stop_dist = dist_next - req_stop
    is_hunting = (holding_time < 7.0) and (prev_dur < 7.0) and (current_notch != prev_notch)
    hunting_score = max(0.0, 7.0 - holding_time) / 7.0 if is_hunting else 0.0

    nl_flag, nl_dist, nl_speed = extract_limit_info(_s(raw, "next_limit_info"))
    ng_flag, ng_dist, ng_val = extract_gradient_info(_s(raw, "next_gradient_info"))
    f_exist, f_dist, f_speed = extract_forward_info(_s(raw, "forward_info"))
    b_exist, b_dist, b_speed = extract_backward_info(_s(raw, "backward_info"))

    dwell_elapsed = _f(raw, "dwell_elapsed")
    is_dwelling = 1.0 if _f(raw, "is_dwelling") >= 0.5 else 0.0
    dwell_min = _f(raw, "dwell_min", CFG.DWELL_MIN)
    dwell_max = _f(raw, "dwell_max", CFG.DWELL_MAX)
    clear_remaining = _f(raw, "forward_clear_remaining_time")
    reach = _f(raw, "time_to_stop_limit")

    return {
        "hold_coast": min(holding_time, 30.0) * is_coast,
        "hold_accel": min(holding_time, 30.0) * is_accel,
        "hold_decel": min(holding_time, 30.0) * is_decel,
        "prev_notch_duration": min(prev_dur, 30.0),

        "speed_limit": speed_limit,
        "signal_speed": signal_speed,
        "current_speed": speed,
        "margin_speed": speed_limit - speed,
        "margin_signal_speed": signal_speed - speed,
        "required_speed": required_speed,
        "speed_margin_to_required": speed - required_speed,
        "target_speed_no_stop": target_ns,
        "speed_margin_to_target": speed - target_ns,
        "v_std": v_std,
        "v_std_deviation": _f(raw, "v_std_deviation", speed - v_std),

        "dist_to_next_station": min(dist_next, 2500.0),
        "time_to_next_station": time_next,
        "req_stop_dist": min(req_stop, 2500.0),
        "margin_stop_dist": margin_stop_dist,
        "margin_stop_dist_clip": min(max(margin_stop_dist, -30.0), 30.0),
        "dist_to_station_clip": min(max(dist_next, -20.0), 20.0),
        "dist_to_station_clip3": min(max(dist_next, -3.0), 3.0),

        "current_gradient": _f(raw, "current_gradient"),
        "coast_accel": _f(raw, "coast_accel"),
        "power_accel": _f(raw, "power_accel"),
        "coast_reachable": 1.0 if _f(raw, "coast_reachable") >= 0.5 else 0.0,
        "coast_arrival_speed": _f(raw, "coast_arrival_speed"),
        "next_limit_flag": nl_flag, "next_limit_dist": nl_dist, "next_limit_speed": nl_speed,
        "next_gradient_flag": ng_flag, "next_gradient_dist": ng_dist, "next_gradient_val": ng_val,

        "delay": _f(raw, "delay"),
        "total_delay": _f(raw, "total_delay"),
        "stations_remaining": _f(raw, "stations_remaining"),
        "total_remaining_distance": min(_f(raw, "total_remaining_distance"), 4000.0),
        "total_remaining_time": _f(raw, "total_remaining_time"),

        "is_dwelling": is_dwelling,
        "dwell_elapsed": min(dwell_elapsed, dwell_max),
        # 標準停車を何秒超えたか（負にはしない）。ユーザー指定の減点カーブの主軸。
        "dwell_over_std": max(0.0, dwell_elapsed - CFG.STD_DWELL),
        # 発車できる状態か（最低停車時間を満たしたか）
        "dwell_departable": 1.0 if (is_dwelling >= 0.5 and dwell_elapsed >= dwell_min) else 0.0,
        # 強制発車の域か
        "dwell_forced": 1.0 if (is_dwelling >= 0.5 and dwell_elapsed >= dwell_max) else 0.0,
        # 今発車したとき、先行が塞ぐ停止限界に到達するまでの秒数
        "time_to_stop_limit": reach,
        # 「先行クリア残時間 − 停止限界到達時間」。正なら今発車すると機外停車する見込み。
        # 発車判断の中核指標。
        "clear_minus_reach": clear_remaining - reach,

        "f_exist": f_exist,
        "f_distance": min(f_dist, 2500.0),
        "f_speed": f_speed,
        "f_relative_speed": speed - f_speed,
        "forward_clear_remaining_time": clear_remaining,
        "forward_observed_delay": _f(raw, "forward_observed_delay"),
        "forward_dwell_elapsed": _f(raw, "forward_dwell_elapsed"),
        "forward_train_delay": _f(raw, "forward_train_delay"),
        "forward_departed_flag": 1.0 if _s(raw, "forward_departed_next") == "発車済み" else 0.0,
        "standard_headway": _f(raw, "standard_headway"),

        "b_exist": b_exist,
        "b_distance": min(b_dist, 2500.0),
        "b_speed": b_speed,
        "b_relative_speed": b_speed - speed,

        "hunting_score": hunting_score,

        "phase_駅停車中（発車判断）": 1.0 if phase == "駅停車中（発車判断）" else 0.0,
        "phase_駅出発直後の加速フェーズ（20秒以内）": 1.0 if phase == "駅出発直後の加速フェーズ（20秒以内）" else 0.0,
        "phase_巡航フェーズ（駅間走行中）": 1.0 if phase == "巡航フェーズ（駅間走行中）" else 0.0,
        "phase_制限速度区間に接近中（500m以内に制限区間在り）": 1.0 if phase == "制限速度区間に接近中（500m以内に制限区間在り）" else 0.0,
        "phase_次駅への減速フェーズ（駅手前400m以内）": 1.0 if phase == "次駅への減速フェーズ（駅手前400m以内）" else 0.0,
        "phase_駅停車完了（速度0km/h）": 1.0 if phase == "駅停車完了（速度0km/h）" else 0.0,

        "current_notch_惰行中": is_coast,
        "current_notch_力行（加速）中": is_accel,
        "current_notch_ブレーキ（減速）中": is_decel,
        "prev_notch_惰行": 1.0 if prev_notch == "惰行" else 0.0,
        "prev_notch_力行（加速）": 1.0 if prev_notch == "力行（加速）" else 0.0,
        "prev_notch_ブレーキ（減速）": 1.0 if prev_notch == "ブレーキ（減速）" else 0.0,
    }


def state_vector(raw):
    """生の状態辞書 → STATE_FEATURE_COLS 順の1次元 np.array（float32）。推論用。"""
    feats = engineer_features(raw)
    return np.array([feats[c] for c in STATE_FEATURE_COLS], dtype=np.float32)


def build_state_matrix(df):
    """pandas DataFrame → (X[N, STATE_DIM], STATE_FEATURE_COLS)。学習用。"""
    import pandas as pd
    feats = df.apply(lambda row: engineer_features(row.to_dict()), axis=1)
    feat_df = pd.DataFrame(list(feats))
    X = feat_df[STATE_FEATURE_COLS].values.astype(np.float32)
    return X, list(STATE_FEATURE_COLS)


# =============================================================================
# 運転モードの判定（ルール・プロンプトと実行時で完全一致させる）
# =============================================================================
# 遅延回復の判定閾値[km/h]（既存 direct_reward_predictor2 と同じ考え方）
DELAY_RECOVERY_PIN_TOL = 0.5
CBTC_MARGIN_FOR_RECOVERY = 5.0
# 駅間停車防止の判定: 機外停車回避の加速上限が定時必要速度より明確に低い＝先行が塞いでいる
ANTI_MID_STOP_TOL = 1.0


def decide_mode(raw):
    """生の状態辞書 → 運転モード文字列。

    **ルールで決定的に決める**。既存版はモードNNのargmaxを併用していたが、条件が成立し続ける
    区間でも normal ↔ delay_recovery が反転する問題が実測されている
    （`direct_reward_predictor2._infer_mode` のコメント参照）。本系統ではモードNNを持たず、
    プロンプトに書くモード定義とこの関数を1対1で対応させる。

    優先順位: 駅停車中 ＞ 駅間停車防止 ＞ 遅延回復 ＞ 通常運転
    """
    if _f(raw, "is_dwelling") >= 0.5:
        return "hold_at_station"

    required = _f(raw, "required_speed")
    target_ns = _f(raw, "target_speed_no_stop", required)
    clear_remaining = _f(raw, "forward_clear_remaining_time")
    limit = _f(raw, "speed_limit")
    signal = _f(raw, "signal_speed")

    # 駅間停車防止: 先行が塞いでいて、機外停車を避けるには定時より遅く走る必要がある
    if clear_remaining > 0.0 and target_ns < required - ANTI_MID_STOP_TOL:
        return "anti_mid_stop"
    # 先行に迫ってCBTC現示が大きく下がっている場合も駅間停車防止
    f_exist, _f_dist, _f_speed = extract_forward_info(_s(raw, "forward_info"))
    if f_exist >= 0.5 and limit > 0.0 and signal < limit - 10.0:
        return "anti_mid_stop"

    # 遅延回復: 定時到達に制限速度びたづきの走行が必要で、かつ先行に抑えられていない
    pinned = (limit > 0.0 and limit - required <= DELAY_RECOVERY_PIN_TOL)
    cbtc_free = (signal >= limit - CBTC_MARGIN_FOR_RECOVERY)
    if pinned and cbtc_free:
        return "delay_recovery"
    return "normal"


if __name__ == "__main__":
    print(f"状態特徴量: {STATE_DIM} 次元")
    print(f"モードone-hot: {MODE_DIM} 次元（{MODE_CLASSES}）")
    print(f"NN入力次元: {STATE_DIM + MODE_DIM}")
    print(f"LLM評価用CSVの列: {len(RAW_COLS)} 列")
    for i, c in enumerate(STATE_FEATURE_COLS):
        print(f"  {i:3d} {c}")
