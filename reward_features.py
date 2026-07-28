"""報酬予測NN・モードNNの特徴量エンジニアリング（共有モジュール・2026-07-25）

学習側（train_reward_network2.py / train_mode_network.py）と推論側
（direct_reward_predictor2.py / mode_predictor.py / environment2.py）が
この1モジュールを共有することで、特徴量の定義・並び順の二重管理による不一致を防ぐ
（設計メモ §3⑥「マニフェスト方式」の本命）。

- engineer_features(raw): 生の状態辞書 → named特徴量辞書（状態特徴量のみ・mode one-hotは含まない）
- STATE_FEATURE_COLS: 状態特徴量の正準な並び順（モードNN・評価NNの共通入力）
- MODE_CLASSES / mode_to_onehot / onehot_index: 運転モードのone-hot（評価NN入力・モードNN出力）

先行スナップショット列（target_speed_no_stop 等）を状態特徴量に追加済み。
先行なし行は既定値（target=required, clear=0 等）で整合する。
"""
import re
import numpy as np


# ---- 運転モード（フェーズ1は3クラス学習、one-hotは将来の間隔調整用に4次元確保）----
MODE_CLASSES = ["normal", "delay_recovery", "anti_mid_stop", "spacing"]
MODE_DIM = len(MODE_CLASSES)
# 学習・分類で実際に使う3クラス（間隔調整=spacingはフェーズ2まで学習対象外）
MODE_CLASSES_ACTIVE = ["normal", "delay_recovery", "anti_mid_stop"]


def onehot_index(mode_str):
    """モード文字列 → MODE_CLASSES上のインデックス。不明はnormal(0)扱い。"""
    m = (mode_str or "normal").strip()
    return MODE_CLASSES.index(m) if m in MODE_CLASSES else 0


def mode_to_onehot(mode_str):
    """モード文字列 → MODE_DIM次元のone-hot（float32）。"""
    v = np.zeros(MODE_DIM, dtype=np.float32)
    v[onehot_index(mode_str)] = 1.0
    return v


# ---- テキスト情報のパース（学習・推論で共有）----
def extract_limit_info(text):
    text = str(text)
    if "この先制限速度なし" in text:
        return 0.0, 0.0, 0.0
    m = re.search(r'(\d+)m先に制限速度(\d+)km/h', text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    return 0.0, 0.0, 0.0


def extract_gradient_info(text):
    text = str(text)
    if "この先目立った勾配なし" in text:
        return 0.0, 0.0, 0.0
    m = re.search(r'(\d+)m先に(上り|下り)勾配(\d+\.?\d*)‰あり', text)
    if m:
        dist = float(m.group(1))
        val = float(m.group(3))
        if m.group(2) == '下り':
            val = -val
        return 1.0, dist, val
    return 0.0, 0.0, 0.0


def extract_forward_info(text):
    text = str(text)
    if "先行列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0
    m = re.search(r'前方\s*([\d\.]+)\s*m\s*先を\s*([\d\.]+)\s*km/h', text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    return 0.0, 5000.0, 0.0


def extract_backward_info(text):
    text = str(text)
    if "後続列車なし" in text or text == "nan":
        return 0.0, 5000.0, 0.0
    m = re.search(r'後方\s*([\d\.]+)\s*m\s*後ろを\s*([\d\.]+)\s*km/h', text)
    if m:
        return 1.0, float(m.group(1)), float(m.group(2))
    return 0.0, 5000.0, 0.0


def _f(raw, key, default=0.0):
    v = raw.get(key, default)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(raw, key):
    v = raw.get(key, "")
    return "" if v is None else str(v)


# 状態特徴量の正準な並び順（モードNN・評価NNの共通入力）。
# ※この順序を変えると学習済みモデルと不整合になる。追加は末尾に行うこと。
STATE_FEATURE_COLS = [
    'hold_coast', 'hold_accel', 'hold_decel',
    'prev_notch_duration',
    'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station',
    'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
    'margin_speed', 'margin_signal_speed', 'margin_stop_dist',
    'margin_stop_dist_clip', 'dist_to_station_clip', 'dist_to_station_clip3',
    'required_speed', 'speed_margin_to_required', 'hunting_score',
    'f_relative_speed', 'b_relative_speed',
    'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
    'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
    'f_exist', 'f_distance', 'f_speed',
    'b_exist', 'b_distance', 'b_speed',
    # ▼ 先行スナップショット（駅間停車防止モード対応・2026-07-25追加）
    'target_speed_no_stop', 'speed_margin_to_target', 'forward_clear_remaining_time',
    'forward_train_delay', 'forward_departed_flag', 'standard_headway',
    # ▼ 先行の次駅での観測遅延（標準30秒超の停車・因果的・2026-07-28追加。設計メモ §13）
    'forward_observed_delay',
    # ▼ フェーズ・ノッチのone-hot
    'phase_駅出発直後の加速フェーズ（20秒以内）',
    'phase_巡航フェーズ（駅間走行中）',
    'phase_制限速度区間に接近中（500m以内に制限区間在り）',
    'phase_次駅への減速フェーズ（駅手前400m以内）',
    'phase_駅停車完了（速度0km/h）',
    'current_notch_惰行中',
    'current_notch_力行（加速）中',
    'current_notch_ブレーキ（減速）中',
    'prev_notch_惰行',
    'prev_notch_力行（加速）',
    'prev_notch_ブレーキ（減速）',
]


def engineer_features(raw):
    """生の状態辞書 raw → named特徴量辞書（STATE_FEATURE_COLSの各キーを含む）。

    raw に必要なキー（数値/文字列）:
      current_notch, prev_notch, phase, holding_time, prev_notch_duration,
      speed_limit, signal_speed, current_speed, dist_to_next_station,
      time_to_next_station, req_stop_dist, delay, current_gradient,
      next_limit_info, next_gradient_info, forward_info, backward_info,
      required_speed,
      target_speed_no_stop, forward_clear_remaining_time, forward_departed_next,
      forward_train_delay, standard_headway
    先行スナップショット列が欠損する場合は既定値（先行なし相当）で扱う。
    """
    current_notch = _s(raw, 'current_notch')
    prev_notch = _s(raw, 'prev_notch')
    if prev_notch == 'なし（または停止）':
        prev_notch = 'ブレーキ（減速）'
    phase = _s(raw, 'phase')

    is_coast = 1.0 if current_notch == "惰行中" else 0.0
    is_accel = 1.0 if current_notch == "力行（加速）中" else 0.0
    is_decel = 1.0 if current_notch == "ブレーキ（減速）中" else 0.0

    speed_limit = _f(raw, 'speed_limit')
    signal_speed = _f(raw, 'signal_speed')
    current_speed = _f(raw, 'current_speed')
    dist_next = _f(raw, 'dist_to_next_station')
    time_next = _f(raw, 'time_to_next_station')
    req_stop = _f(raw, 'req_stop_dist')
    delay = _f(raw, 'delay')
    gradient = _f(raw, 'current_gradient')
    required_speed = _f(raw, 'required_speed')
    holding_time = _f(raw, 'holding_time')
    prev_dur = _f(raw, 'prev_notch_duration')

    margin_speed = speed_limit - current_speed
    margin_signal_speed = signal_speed - current_speed
    margin_stop_dist = dist_next - req_stop
    margin_stop_dist_clip = min(max(margin_stop_dist, -30.0), 30.0)
    dist_to_station_clip = min(max(dist_next, -20.0), 20.0)
    dist_to_station_clip3 = min(max(dist_next, -3.0), 3.0)
    speed_margin_to_required = current_speed - required_speed

    is_hunting = (holding_time < 7.0) and (prev_dur < 7.0) and (current_notch != prev_notch)
    hunting_score = max(0.0, 7.0 - holding_time) / 7.0 if is_hunting else 0.0

    holding_clip = min(holding_time, 30.0)
    prev_dur_clip = min(prev_dur, 30.0)

    nl_flag, nl_dist, nl_speed = extract_limit_info(_s(raw, 'next_limit_info'))
    ng_flag, ng_dist, ng_val = extract_gradient_info(_s(raw, 'next_gradient_info'))
    f_exist, f_dist, f_speed = extract_forward_info(_s(raw, 'forward_info'))
    b_exist, b_dist, b_speed = extract_backward_info(_s(raw, 'backward_info'))

    dist_next_c = min(dist_next, 2000.0)
    req_stop_c = min(req_stop, 2000.0)
    f_dist = min(f_dist, 2000.0)
    b_dist = min(b_dist, 2000.0)
    f_rel_speed = current_speed - f_speed
    b_rel_speed = b_speed - current_speed

    # ▼ 先行スナップショット（駅間停車防止モード対応）
    target_ns = _f(raw, 'target_speed_no_stop', required_speed)  # 欠損時はrequired（先行なし相当）
    speed_margin_to_target = current_speed - target_ns
    clear_remaining = _f(raw, 'forward_clear_remaining_time', 0.0)
    fwd_delay = _f(raw, 'forward_train_delay', 0.0)
    departed = _s(raw, 'forward_departed_next')
    forward_departed_flag = 1.0 if departed == "発車済み" else 0.0
    headway = _f(raw, 'standard_headway', 0.0)
    observed_delay = _f(raw, 'forward_observed_delay', 0.0)  # 先行の次駅観測遅延[秒]（因果的）

    return {
        'hold_coast': holding_clip * is_coast,
        'hold_accel': holding_clip * is_accel,
        'hold_decel': holding_clip * is_decel,
        'prev_notch_duration': prev_dur_clip,
        'speed_limit': speed_limit,
        'signal_speed': signal_speed,
        'current_speed': current_speed,
        'dist_to_next_station': dist_next_c,
        'time_to_next_station': time_next,
        'req_stop_dist': req_stop_c,
        'delay': delay,
        'current_gradient': gradient,
        'margin_speed': margin_speed,
        'margin_signal_speed': margin_signal_speed,
        'margin_stop_dist': margin_stop_dist,
        'margin_stop_dist_clip': margin_stop_dist_clip,
        'dist_to_station_clip': dist_to_station_clip,
        'dist_to_station_clip3': dist_to_station_clip3,
        'required_speed': required_speed,
        'speed_margin_to_required': speed_margin_to_required,
        'hunting_score': hunting_score,
        'f_relative_speed': f_rel_speed,
        'b_relative_speed': b_rel_speed,
        'next_limit_flag': nl_flag, 'next_limit_dist': nl_dist, 'next_limit_speed': nl_speed,
        'next_gradient_flag': ng_flag, 'next_gradient_dist': ng_dist, 'next_gradient_val': ng_val,
        'f_exist': f_exist, 'f_distance': f_dist, 'f_speed': f_speed,
        'b_exist': b_exist, 'b_distance': b_dist, 'b_speed': b_speed,
        'target_speed_no_stop': target_ns,
        'speed_margin_to_target': speed_margin_to_target,
        'forward_clear_remaining_time': clear_remaining,
        'forward_train_delay': fwd_delay,
        'forward_departed_flag': forward_departed_flag,
        'standard_headway': headway,
        'forward_observed_delay': observed_delay,
        'phase_駅出発直後の加速フェーズ（20秒以内）': 1.0 if phase == "駅出発直後の加速フェーズ（20秒以内）" else 0.0,
        'phase_巡航フェーズ（駅間走行中）': 1.0 if phase == "巡航フェーズ（駅間走行中）" else 0.0,
        'phase_制限速度区間に接近中（500m以内に制限区間在り）': 1.0 if phase == "制限速度区間に接近中（500m以内に制限区間在り）" else 0.0,
        'phase_次駅への減速フェーズ（駅手前400m以内）': 1.0 if phase == "次駅への減速フェーズ（駅手前400m以内）" else 0.0,
        'phase_駅停車完了（速度0km/h）': 1.0 if phase == "駅停車完了（速度0km/h）" else 0.0,
        'current_notch_惰行中': is_coast,
        'current_notch_力行（加速）中': is_accel,
        'current_notch_ブレーキ（減速）中': is_decel,
        'prev_notch_惰行': 1.0 if prev_notch == "惰行" else 0.0,
        'prev_notch_力行（加速）': 1.0 if prev_notch == "力行（加速）" else 0.0,
        'prev_notch_ブレーキ（減速）': 1.0 if prev_notch == "ブレーキ（減速）" else 0.0,
    }


def state_vector(raw):
    """生の状態辞書 → STATE_FEATURE_COLS順の1次元np.array（float32）。推論用。"""
    feats = engineer_features(raw)
    return np.array([feats[c] for c in STATE_FEATURE_COLS], dtype=np.float32)


def build_state_matrix(df):
    """pandas DataFrame → (X_state[N,len(STATE_FEATURE_COLS)], STATE_FEATURE_COLS)。学習用。"""
    import pandas as pd
    feats = df.apply(lambda row: engineer_features(row.to_dict()), axis=1)
    feat_df = pd.DataFrame(list(feats))
    X = feat_df[STATE_FEATURE_COLS].values.astype(np.float32)
    return X, list(STATE_FEATURE_COLS)
