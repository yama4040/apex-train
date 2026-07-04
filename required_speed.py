from typing import Tuple

# ==========================================
# 必要速度（巡航速度）算出ロジック
#    apex系スクリプト実行時（train.py / environment2.py / environment3.py）と同一の
#    物理パラメータを用い、「加速にかかる時間」「惰行による自然減速」
#    「ブレーキにかかる時間・距離」を考慮したシミュレーションにより、
#    定時運行に必要な巡航速度を算出する。
#    一定速度で走り続けることを仮定する単純な平均速度法とは異なり、
#    巡航速度到達後は必ず惰行によって速度が低下する前提でモデル化している。
#
#    evaluate_csv_with_llm.py（LLM評価用プロンプト生成）と
#    environment2.py / apex2.py（回帰NNの学習・推論）の両方から参照され、
#    必要速度の算出方法を完全に一致させるための共通モジュールである。
# ==========================================

# train.py（apex系スクリプト実行時に使用される物理モデル）と同一の定数
FACTOR_OF_INERTIA = 28.34467
WEIGHT_CORRECTION = 1.0     # apex系スクリプト実行時のデフォルト重量補正（Environment.resetの既定値）
BRAKE_DECEL_MS2 = 2.5 / 3.6  # apex系スクリプトがreq_stop_dist算出に用いる簡易ブレーキ減速度[m/s^2]
SIM_DT = 0.25                # シミュレーション刻み幅[s]
SIM_MAX_TIME = 400.0         # シミュレーションの安全上限[s]

def tractive_force(speed_kmh: float) -> float:
    """引張力[kg/t]（train.pyのtractive_forceと同一の特性曲線）"""
    if 0 <= speed_kmh < 42:
        return -1.489 * speed_kmh + 92.408
    elif 42 <= speed_kmh < 68:
        return -0.4 * speed_kmh + 46.68
    else:
        return -0.0963 * speed_kmh + 26.0284

def travel_resistance(speed_kmh: float) -> float:
    """走行抵抗[kg/t]（train.pyのtravel_resistanceと同一）"""
    return 2.39 + 0.0224 * speed_kmh + 0.00062 * (speed_kmh ** 2)

def brake_stop_distance_m(speed_kmh: float) -> float:
    """
    現在速度からブレーキを開始し停止するまでに要する距離[m]。
    apex系スクリプトがreq_stop_distの算出に用いている簡易減速モデル（減速度2.5km/h/s相当）と同一のもの。
    """
    v_ms = max(0.0, speed_kmh / 3.6)
    return (v_ms ** 2) / (2 * BRAKE_DECEL_MS2)

def simulate_trip_time(v0_kmh: float, target_cruise_kmh: float, distance_m: float, grade_resistance: float = 0.0) -> Tuple[float, float]:
    """
    現在速度v0から
      1. 力行でtarget_cruise_kmhまで加速
      2. 到達後は惰行に切り替え、走行抵抗により自然に速度が低下
      3. 残り距離がブレーキ停止距離まで縮まった時点でブレーキに切り替え、0km/hまで減速
    という運転曲線をシミュレートし、distance_mを走破するのに要する合計時間[s]と走行距離[m]を返す。
    """
    speed = v0_kmh
    t = 0.0
    dist = 0.0
    phase = "accel" if speed < target_cruise_kmh else "coast"

    steps = int(SIM_MAX_TIME / SIM_DT)
    for _ in range(steps):
        remaining = distance_m - dist
        if remaining <= 0.0:
            break
        if phase == "brake" and speed <= 0.0:
            break

        if phase == "accel" and speed >= target_cruise_kmh:
            phase = "coast"
        if phase == "coast" and brake_stop_distance_m(speed) >= remaining:
            phase = "brake"

        if phase == "accel":
            accel = ((tractive_force(speed) - travel_resistance(speed) - grade_resistance) * WEIGHT_CORRECTION) / FACTOR_OF_INERTIA
            if accel <= 0.0:
                # これ以上加速できない速度域に達した場合は惰行へ移行
                phase = "coast"
                accel = (-(travel_resistance(speed) + grade_resistance) * WEIGHT_CORRECTION) / FACTOR_OF_INERTIA
        elif phase == "coast":
            accel = (-(travel_resistance(speed) + grade_resistance) * WEIGHT_CORRECTION) / FACTOR_OF_INERTIA
        else:  # brake
            accel = -BRAKE_DECEL_MS2 * 3.6  # m/s^2 -> km/h/s換算

        speed = max(0.0, speed + accel * SIM_DT)
        dist += (speed / 3.6) * SIM_DT
        t += SIM_DT

    return t, dist

def calculate_required_speed(current_speed: float, dist_to_next_station: float, time_to_next_station: float,
                              speed_limit: float, current_gradient: float = 0.0) -> float:
    """
    定時運行に必要な巡航速度[km/h]を算出する。

    「駅までの残り距離」と「残り時間」から平均速度を求めてその1.3〜1.4倍とする簡易近似ではなく、
    実際の加速特性（tractive_force）・惰行時の自然減速（travel_resistance）・
    ブレーキ特性（apex系スクリプトのreq_stop_distモデル）を反映した走行シミュレーションにより、
    「この速度まで力行し、その後惰行に切り替えれば定時運行できる」という巡航速度を二分探索で求める。
    現在速度のまま惰行に切り替えても定時運行可能な場合は、それ以上の加速は不要であるため
    current_speedをそのまま返す（＝直ちに惰行へ移行すべきことを意味する）。
    """
    if dist_to_next_station <= 0.0 or speed_limit <= 0.0:
        return 0.0
    if time_to_next_station <= 0.0:
        # 既に定刻を過ぎている場合は制限速度までの加速を要求する
        return speed_limit

    # 現在速度のまま追加加速せず惰行→ブレーキに移行した場合の到達時間
    t_now, d_now = simulate_trip_time(current_speed, current_speed, dist_to_next_station, current_gradient)
    if d_now >= dist_to_next_station - 1.0 and t_now <= time_to_next_station:
        return current_speed

    lo, hi = current_speed, speed_limit
    for _ in range(24):
        mid = (lo + hi) / 2.0
        t_sim, d_sim = simulate_trip_time(current_speed, mid, dist_to_next_station, current_gradient)
        if d_sim < dist_to_next_station - 1.0:
            # 駅に到達しきれない＝所要時間を過大評価させ、より高い速度を探索させる
            t_sim = SIM_MAX_TIME
        if t_sim > time_to_next_station:
            lo = mid
        else:
            hi = mid

    return min(hi, speed_limit)
