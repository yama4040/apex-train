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
SIM_DT = 0.25                # シミュレーション刻み幅[s]
SIM_MAX_TIME = 400.0         # シミュレーションの安全上限[s]

# 減速ノッチによるブレーキ減速度[km/h/s]
# 制動時 dv/dt = -BRAKE_NOTCH_DECEL_KMHS - (Rr+Rg+Rc)/28.34467（修論 式(3.2)と同じ構造）
# 標準的な常用ブレーキ相当の 2.5 km/h/s（train.pyのDECELERATEと同一の値であること）
BRAKE_NOTCH_DECEL_KMHS = 2.5
BRAKE_TABLE_DV = 0.25        # ブレーキ停止距離テーブルの速度刻み[km/h]
ARRIVAL_TOL_M = 5.0          # シミュレーション上の到達判定の許容誤差[m]
                             # （オイラー積分の離散化誤差により数m手前で停止扱いになる場合があるため）

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

def _brake_decel_kmh_s(speed_kmh: float, grade_resistance: float) -> float:
    """
    減速ノッチ使用時の実効減速度[km/h/s]（正の値）。
    train.pyのstep()と同一の運動方程式（修論 式(3.2)/(4.1)と同じ構造）：
      dv/dt = -BRAKE_NOTCH_DECEL_KMHS * WC - (Rr * WC + Rg) / FACTOR_OF_INERTIA  （WC = 1/kw）
    ※曲線抵抗は情報が無いため省略（影響は小さい）
    """
    decel = (BRAKE_NOTCH_DECEL_KMHS * WEIGHT_CORRECTION
             + (travel_resistance(speed_kmh) * WEIGHT_CORRECTION + grade_resistance) / FACTOR_OF_INERTIA)
    # 急な下り勾配でも発散しないよう最低限の減速度を保証する
    return max(decel, 0.05)

def brake_stop_distance_m(speed_kmh: float, grade_resistance: float = 0.0) -> float:
    """
    現在速度からブレーキを開始し停止するまでに要する距離[m]。
    train.pyの実ダイナミクス（減速ノッチ2.5km/h/s＋走行抵抗＋勾配抵抗）と一致するモデル。
    """
    v = max(0.0, speed_kmh)
    dist = 0.0
    while v > 1e-9:
        step = min(BRAKE_TABLE_DV, v)
        v_mid = v - step / 2.0
        dt = step / _brake_decel_kmh_s(v_mid, grade_resistance)
        dist += (v_mid / 3.6) * dt
        v -= step
    return dist

def _build_brake_table(v_max_kmh: float, grade_resistance: float) -> list:
    """
    0からv_max_kmhまでBRAKE_TABLE_DV刻みの停止距離テーブル[m]を1回の積分で構築する。
    simulate_trip_time内で毎ステップ停止距離を参照するための高速化用。
    """
    table = [0.0]
    v = 0.0
    while v < v_max_kmh:
        v_mid = v + BRAKE_TABLE_DV / 2.0
        dt = BRAKE_TABLE_DV / _brake_decel_kmh_s(v_mid, grade_resistance)
        table.append(table[-1] + (v_mid / 3.6) * dt)
        v += BRAKE_TABLE_DV
    return table

def _lookup_brake_dist(table: list, speed_kmh: float) -> float:
    """テーブルの線形補間により停止距離[m]を返す"""
    if speed_kmh <= 0.0:
        return 0.0
    idx = speed_kmh / BRAKE_TABLE_DV
    lo = int(idx)
    if lo + 1 >= len(table):
        return table[-1]
    frac = idx - lo
    return table[lo] * (1.0 - frac) + table[lo + 1] * frac

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

    # 停止距離テーブルを事前構築（下り勾配での惰行加速により目標速度を多少超える余地を持たせる）
    brake_table = _build_brake_table(max(v0_kmh, target_cruise_kmh) + 15.0, grade_resistance)

    steps = int(SIM_MAX_TIME / SIM_DT)
    for _ in range(steps):
        remaining = distance_m - dist
        if remaining <= 0.0:
            break
        if phase == "brake" and speed <= 0.0:
            break

        if phase == "accel" and speed >= target_cruise_kmh:
            phase = "coast"
        # 加速中でも停止距離が残距離に達したら即ブレーキへ移行する（短距離での過走を防止）
        if phase != "brake" and _lookup_brake_dist(brake_table, speed) >= remaining:
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
            accel = -_brake_decel_kmh_s(speed, grade_resistance)  # train.pyと同一のブレーキ特性

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
    ブレーキ特性（train.pyの減速ノッチと同一の実効減速度モデル）を反映した走行シミュレーションにより、
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
    if d_now >= dist_to_next_station - ARRIVAL_TOL_M and t_now <= time_to_next_station:
        return current_speed

    lo, hi = current_speed, speed_limit
    for _ in range(24):
        mid = (lo + hi) / 2.0
        t_sim, d_sim = simulate_trip_time(current_speed, mid, dist_to_next_station, current_gradient)
        if d_sim < dist_to_next_station - ARRIVAL_TOL_M:
            # 駅に到達しきれない＝所要時間を過大評価させ、より高い速度を探索させる
            t_sim = SIM_MAX_TIME
        if t_sim > time_to_next_station:
            lo = mid
        else:
            hi = mid

    return min(hi, speed_limit)


# ==========================================
# 機外停車（駅間停車）回避のための加速上限速度
#   駅間停車防止モードで「先行列車に追いつかず機外停車せずに進める上限速度」を算出する。
#   calculate_required_speed の制約を「定刻までに駅到着」から
#   「先行がクリアするまで駅に着かない（先行に追いつかない）」へ差し替えたもの。
#   実効所要時間 = max(定刻残り時間, 先行クリア残時間 + 安全マージン) を渡すだけで、
#   同じ走行シミュレーション＋二分探索から加速上限が得られる。
#   （詳細は docs_先行列車対応_設計メモ.md §5「駅間停車防止モードの加速上限評価」）
# ==========================================

# 機外停車回避の安全マージン[s]。先行がクリアした瞬間ちょうどに駅到着するのではなく、
# 少し余裕を持って（先行が十分に前方へ離れてから）到着させるための加算時間。
# ※スモークテストで調整予定の暫定値（設計メモ §5「要決定の残パラメータ」）。
NO_STOP_SAFETY_MARGIN_S = 15.0


def calculate_no_stop_target_speed(current_speed: float, dist_to_next_station: float,
                                   time_to_next_station: float, forward_clear_remaining_time: float,
                                   speed_limit: float, current_gradient: float = 0.0,
                                   safety_margin: float = NO_STOP_SAFETY_MARGIN_S) -> float:
    """
    機外停車（駅間停車）を避けつつ進める「加速上限（目標巡航速度）」[km/h]を算出する。

    駅間停車防止モードにおける評価基準値。現在速度をこの値付近まで（超えない範囲で）
    上げるのは適切、明確に超える力行は「先行に追いついて機外停車を招く過剰加速」として減点、
    という上限として用いる。

    Args:
        forward_clear_remaining_time: 先行列車が自列車の次駅を発車するまでの残り秒数。
            駅標準停車時間は全列車30秒として算出する（呼び出し側で先行軌道から計算）。
            先行がいない／既にクリアした場合は0以下を渡せばよく、その場合は
            calculate_required_speed（定刻ベース）と同値になる。

    実効所要時間 = max(定刻残り時間, 先行クリア残時間 + 安全マージン)。
    先行が長く塞ぐほど実効所要時間が伸び、上限速度は下がる（低速で惰行して進めば機外停車しない）。
    先行がクリアすると自動的に通常の required_speed と一致し、解放後の再加速が適切になる。

    【重要】上限速度は「現在速度に依存しない状況ベースの値」として算出する。
    calculate_required_speed は「現在速度のまま惰行で間に合えば現在速度を返す」仕様で早着を許容するため、
    列車が加速するほど上限も上がってしまい過剰加速を検知できない。そこで機外停車回避では
    「その速度から惰行すると実効所要時間ちょうどに次駅へ到着する速度」を二分探索で求める。
    現在速度がこの上限を超えていれば、惰行しても早着＝先行に追いつき機外停車、を意味する。
    """
    # 先行がクリア済み／先行なし（先行クリア残時間が0以下）の場合は塞ぎがないため、
    # 安全マージンに関わらず通常の required_speed に委ねる（駅接近で残り時間が小さいと
    # マージンにより target が不当に下がるのを防ぐ）。
    if forward_clear_remaining_time <= 0.0:
        return calculate_required_speed(current_speed, dist_to_next_station, time_to_next_station,
                                        speed_limit, current_gradient)

    effective_time = max(time_to_next_station, forward_clear_remaining_time + safety_margin)

    # 実効所要時間が定刻残り時間と同じ（先行が定刻より早くクリア）の場合も通常の required_speed に委ねる。
    if effective_time <= time_to_next_station + 1e-6:
        return calculate_required_speed(current_speed, dist_to_next_station, time_to_next_station,
                                        speed_limit, current_gradient)
    if dist_to_next_station <= 0.0 or speed_limit <= 0.0:
        return 0.0

    # 「速度vから惰行→ブレーキ停止」でちょうど effective_time に次駅到着する v を二分探索する。
    # 惰行の走行時間は v が高いほど短い（＝早着）。v が高すぎれば早着（機外停車リスク）、
    # 低すぎれば遅着。current_speed には依存しない。
    lo, hi = 1.0, speed_limit
    for _ in range(24):
        mid = (lo + hi) / 2.0
        t_sim, d_sim = simulate_trip_time(mid, mid, dist_to_next_station, current_gradient)
        if d_sim < dist_to_next_station - ARRIVAL_TOL_M:
            # 惰行だけでは次駅に到達しきれない＝所要時間過大とみなし、より高い速度を探索
            t_sim = SIM_MAX_TIME
        if t_sim > effective_time:
            lo = mid   # 遅着 → もっと速く
        else:
            hi = mid   # 早着 → もっと遅く
    return min(hi, speed_limit)
