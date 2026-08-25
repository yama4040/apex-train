# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の目標速度・制動距離・CBTC現示の算出。

既存 `required_speed.py` は**一切変更しない**（apex2.py 系がそのまま使い続ける）。
本モジュールが新たに担うのは以下。

  1. **勾配プロファイルを積分する**。既存は「現在地点の勾配が駅まで一定」と仮定するが、
     白兎→蚕桑は +11.4‰ の上りが 1km 続いたあと −2.3‰ に反転するため、この仮定では
     惰行の減速量を大きく誤る（惰行減速度が −0.55 → −0.03 km/h/s と 18 倍変わる）。
  2. **駅の制動は逆積分した制動曲線**（`brake_curve_ymulti`）を使う。停止位置が固定なので
     キャッシュでき、数cm精度が得られる。
  3. **列車長を織り込んだCBTC現示**。停止限界は「先行の最後尾から50m手前」＝
     先行の先頭から `列車長20m + 50m = 70m` 手前（既存 environment2 は50mしか引いていない）。
  4. **通算ダイヤ**（駅停車30秒を挟む複数駅間）に基づく残り時間・遅延。
  5. **駅停車中の発車判断**に使う指標（先行クリア残時間・今発車したときの機外停車予測）。

物理定数・引張力・走行抵抗は `required_speed.py`（＝`train.py` と一致）から取り込む。
"""
import math
from bisect import bisect_left

import config_ymulti as CFG
import required_speed as _rs
from brake_curve_ymulti import get_lookup, get_brake_curve

# 既存と同一の物理定数（train.py 由来）
FACTOR_OF_INERTIA = _rs.FACTOR_OF_INERTIA
WEIGHT_CORRECTION = _rs.WEIGHT_CORRECTION
BRAKE_NOTCH_DECEL_KMHS = _rs.BRAKE_NOTCH_DECEL_KMHS
tractive_force = _rs.tractive_force
travel_resistance = _rs.travel_resistance

# 走行シミュレーションの刻み[s]。**目標速度は計画のための推定値**であり物理の再現ではないので、
# 0.25→0.5 にしても所要時間の推定差は1秒未満（実測）。目標速度は毎ステップ2回呼ぶため、
# ここの刻みと二分探索の回数がRL1ステップのコストをほぼ決める。
SIM_DT = 0.5
BISECT_ITERS = 14      # 二分探索の回数。70km/h ÷ 2^14 = 0.004 km/h の分解能で十分
SIM_MAX_TIME = 600.0   # 安全上限[s]
ARRIVAL_TOL_M = 5.0    # 到達判定の許容誤差[m]

# 機外停車回避の安全マージン[s]（既存 required_speed と同じ意味）
NO_STOP_SAFETY_MARGIN_S = 15.0

_BRAKE_TABLE_CACHE = {}


# =============================================================================
# 路線プロファイル
# =============================================================================
def resistance_at(position):
    """位置 position[km] の勾配抵抗＋曲線抵抗[kg/t]（track.py と同一の参照規則）"""
    lk = get_lookup()
    return lk.grade(position) + lk.curve(position)


def grade_at(position):
    """位置 position[km] の勾配[‰]（＝勾配抵抗[kg/t]）"""
    return get_lookup().grade(position)


def speed_limit_at(position):
    """位置 position[km] の制限速度[km/h]"""
    return get_lookup().speed_limit(position)


def coast_accel(position, speed):
    """位置・速度における惰行時の加速度[km/h/s]（負なら減速・正なら下り勾配で加速）"""
    return (-(travel_resistance(speed) * WEIGHT_CORRECTION + resistance_at(position))
            / FACTOR_OF_INERTIA)


def power_accel(position, speed):
    """位置・速度における力行時の加速度[km/h/s]"""
    return (((tractive_force(speed) - travel_resistance(speed)) * WEIGHT_CORRECTION
             - resistance_at(position)) / FACTOR_OF_INERTIA)


def brake_accel(position, speed):
    """位置・速度における制動時の加速度[km/h/s]（負）"""
    return (coast_accel(position, speed) - BRAKE_NOTCH_DECEL_KMHS * WEIGHT_CORRECTION)


# =============================================================================
# 制動距離
# =============================================================================
def station_stop_distance_m(speed, stop_position):
    """駅（固定位置）にちょうど停止するための制動距離[m]。逆積分した制動曲線を使うため数cm精度。"""
    return get_brake_curve(stop_position).stop_distance(speed) * 1000.0


def moving_stop_distance_m(speed, position):
    """任意地点から制動を開始したときの停止距離[m]（先行列車など**停止点が動く**対象向け）。

    停止点が毎ステップ変わるものに逆積分の制動曲線は作れないため、
    現在地点の勾配・曲線が一定という近似を使う（既存 environment2.cbtc_signal_speed と同じ扱い）。
    誤差は本区間で最大6m程度。停止限界には別途 70m の余裕があるため実害はない。
    """
    res = round(resistance_at(position), 3)
    if res not in _BRAKE_TABLE_CACHE:
        _BRAKE_TABLE_CACHE[res] = _rs._build_brake_table(90.0, res)
    return _rs._lookup_brake_dist(_BRAKE_TABLE_CACHE[res], speed)


def cbtc_signal_speed(position, forward_position, base_limit):
    """CBTC指示速度[km/h]。

    停止限界は「先行の最後尾から50m手前に自列車の先頭」＝
    先行の先頭から `列車長20m + 停止限界50m = 70m` 手前（`config_ymulti.CBTC_HEAD_MARGIN_KM`）。
    先行がいない場合は路線の制限速度をそのまま返す。
    """
    if forward_position is None:
        return base_limit
    target_distance_m = (forward_position - position - CFG.CBTC_HEAD_MARGIN_KM) * 1000.0
    if target_distance_m <= 0.0:
        return 0.0
    lo, hi, best = 0.0, base_limit, 0.0
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if moving_stop_distance_m(mid, position) <= target_distance_m:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


# =============================================================================
# 走行シミュレーション（プロファイル対応）
# =============================================================================
def simulate_trip(v0, v_cruise, start_position, stop_position, dt=SIM_DT,
                  max_time=SIM_MAX_TIME):
    """「v_cruise まで力行 → 惰行 → 制動曲線に当たったら制動」で駅まで走る所要時間を求める。

    勾配・曲線は**位置ごとに参照**する（既存 required_speed の一定勾配仮定を置き換える）。
    戻り値: (所要時間[s], 到達距離[m], 駅に到達できたか)
    """
    bc = get_brake_curve(stop_position)
    v = max(0.0, v0)
    x = start_position
    t = 0.0
    phase = "accel" if v < v_cruise else "coast"
    steps = int(max_time / dt)
    for _ in range(steps):
        if x >= stop_position:
            return t, (x - start_position) * 1000.0, True
        if phase == "brake" and v <= 0.0:
            return t, (x - start_position) * 1000.0, True
        # 制動曲線に当たったら制動へ
        if phase != "brake" and v >= bc.speed_at(x):
            phase = "brake"
        if phase == "accel" and v >= v_cruise:
            phase = "coast"

        if phase == "accel":
            a = power_accel(x, v)
            if a <= 0.0:
                phase = "coast"
                a = coast_accel(x, v)
        elif phase == "coast":
            a = coast_accel(x, v)
        else:
            a = brake_accel(x, v)
            if a >= 0.0:
                a = -0.05

        v_new = max(0.0, v + a * dt)
        x += ((v + v_new) / 2.0 / 3600.0) * dt
        v = v_new
        t += dt
        if phase != "brake" and v <= 0.0:
            # 惰行・力行中に失速＝駅間停車（到達できない）
            return t, (x - start_position) * 1000.0, False
    return t, (x - start_position) * 1000.0, False


def coast_probe(position, speed, stop_position):
    """今の速度から**惰行だけ**で駅の制動開始点に到達できるかを1回の積分で調べる。

    戻り値: (到達できるか[bool], 制動開始点に達する速度[km/h]。失速なら0.0)

    白兎→蚕桑は +11.4‰ が 1km 続くため、惰行に入る速度が低すぎると駅間停車する。
    「惰行してよいか」の判断に使う。到達可否と到達速度は同じ積分から得られるので1関数にまとめる
    （別々に呼ぶと同じ積分を2回回すことになる）。
    """
    bc = get_brake_curve(stop_position)
    v, x = max(0.0, speed), position
    for _ in range(int(SIM_MAX_TIME / SIM_DT)):
        if x >= stop_position or v >= bc.speed_at(x):
            return True, v
        a = coast_accel(x, v)
        v_new = max(0.0, v + a * SIM_DT)
        x += ((v + v_new) / 2.0 / 3600.0) * SIM_DT
        v = v_new
        if v <= 0.5:
            return False, 0.0
    return False, v


def coast_reachable(position, speed, stop_position):
    """惰行だけで駅の制動開始点に到達できるか（`coast_probe` の薄いラッパ）"""
    return coast_probe(position, speed, stop_position)[0]


def coast_arrival_speed(position, speed, stop_position):
    """惰行を続けたときに制動開始点に達する速度[km/h]（`coast_probe` の薄いラッパ）"""
    return coast_probe(position, speed, stop_position)[1]


# =============================================================================
# 目標速度
# =============================================================================
def calculate_required_speed(current_speed, position, stop_position, time_to_station,
                             speed_limit):
    """定時運行に必要な巡航速度[km/h]。

    「この速度まで力行し、その後は惰行に切り替えれば定刻に着く」速度を二分探索で求める。
    現在速度のまま惰行に切り替えても間に合うなら現在速度を返す（＝直ちに惰行してよい）。
    """
    dist_m = (stop_position - position) * 1000.0
    if dist_m <= 0.0 or speed_limit <= 0.0:
        return 0.0
    if time_to_station <= 0.0:
        return speed_limit

    t_now, d_now, ok_now = simulate_trip(current_speed, current_speed, position, stop_position)
    if ok_now and t_now <= time_to_station:
        return current_speed

    lo, hi = current_speed, speed_limit
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2.0
        t_sim, _d, ok = simulate_trip(current_speed, mid, position, stop_position)
        if not ok:
            t_sim = SIM_MAX_TIME
        if t_sim > time_to_station:
            lo = mid
        else:
            hi = mid
    return min(hi, speed_limit)


def calculate_no_stop_target_speed(current_speed, position, stop_position, time_to_station,
                                   forward_clear_remaining_time, speed_limit,
                                   safety_margin=NO_STOP_SAFETY_MARGIN_S):
    """機外停車（駅間停車）を避けつつ進める加速上限[km/h]。

    実効所要時間 = max(定刻残り時間, 先行クリア残時間 + 安全マージン)。
    先行が長く塞ぐほど実効所要時間が伸び、上限速度は下がる（低速で惰行すれば機外停車しない）。
    **現在速度に依存しない状況ベースの値**として求める（現在速度に依存させると
    加速するほど上限も上がって過剰加速を検知できなくなるため。既存 required_speed と同じ設計）。
    """
    if forward_clear_remaining_time <= 0.0:
        return calculate_required_speed(current_speed, position, stop_position,
                                        time_to_station, speed_limit)
    effective_time = max(time_to_station, forward_clear_remaining_time + safety_margin)
    if effective_time <= time_to_station + 1e-6:
        return calculate_required_speed(current_speed, position, stop_position,
                                        time_to_station, speed_limit)
    dist_m = (stop_position - position) * 1000.0
    if dist_m <= 0.0 or speed_limit <= 0.0:
        return 0.0

    lo, hi = 1.0, speed_limit
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2.0
        t_sim, _d, ok = simulate_trip(mid, mid, position, stop_position)
        if not ok:
            t_sim = SIM_MAX_TIME
        if t_sim > effective_time:
            lo = mid     # 遅着 → もっと速く
        else:
            hi = mid     # 早着 → もっと遅く
    return min(hi, speed_limit)


# =============================================================================
# 駅停車中の発車判断に使う指標
# =============================================================================
def time_to_stop_limit(position, stop_position, forward_position_at_stop=None,
                       speed_limit=70.0, v_cruise=None):
    """今この位置から発車したとき、**先行が次駅に停車している場合の停止限界**
    （次駅の `CBTC_HEAD_MARGIN_KM` 手前）に到達するまでの秒数[s]。

    「今発車すると機外停車するか」の判定に使う。
    先行が停止限界より前にいなければ意味を持たないので、呼び出し側で状況を判断すること。
    """
    limit_pos = stop_position - CFG.CBTC_HEAD_MARGIN_KM
    if limit_pos <= position:
        return 0.0
    v_cruise = v_cruise if v_cruise is not None else speed_limit
    v, x, t = 0.0, position, 0.0
    for _ in range(int(SIM_MAX_TIME / SIM_DT)):
        if x >= limit_pos:
            return t
        a = power_accel(x, v) if v < v_cruise else coast_accel(x, v)
        v_new = max(0.0, v + a * SIM_DT)
        x += ((v + v_new) / 2.0 / 3600.0) * SIM_DT
        v = v_new
        t += SIM_DT
    return t


if __name__ == "__main__":
    import codecs
    import time as _time
    import pandas as pd
    with codecs.open("./input/Station.csv", "r", "utf-8", "ignore") as f:
        st = pd.read_csv(f)
    pos = [float(st["position"][i]) for i in CFG.STATION_INDICES]

    print("=== 区間ごとの目標速度・惰行到達可能性 ===")
    for k in range(len(CFG.RUNNING_TIMES)):
        a, b = pos[k], pos[k + 1]
        rt = CFG.RUNNING_TIMES[k]
        name = f"{CFG.STATION_NAMES_JA[CFG.STATION_INDICES[k]]}→{CFG.STATION_NAMES_JA[CFG.STATION_INDICES[k+1]]}"
        print(f"\n--- 区間{k} {name}  {(b-a)*1000:.0f} m / 標準 {rt:.0f}s ---")
        print("  位置[km]  勾配‰  惰行a   力行a   制動距離m  required(残り時間別)")
        n = 6
        for i in range(n):
            x = a + (b - a) * i / (n - 1) * 0.92
            v = 50.0
            req_full = calculate_required_speed(v, x, b, rt * (1 - i / (n - 1) * 0.92), 70.0)
            req_late = calculate_required_speed(v, x, b, rt * 0.5 * (1 - i / (n - 1) * 0.92), 70.0)
            print(f"  {x:8.4f} {grade_at(x):+6.1f} {coast_accel(x,v):+7.3f} "
                  f"{power_accel(x,v):+7.3f} {station_stop_distance_m(v,b):9.1f}  "
                  f"定時={req_full:5.1f} 遅延={req_late:5.1f}")
        # 惰行到達可能性
        print("  惰行のみで駅に届く最低速度（区間発車直後の位置から）:")
        for x_off in (0.2, 0.5, 0.8):
            x = a + (b - a) * x_off
            lo, hi = 0.0, 70.0
            for _ in range(18):
                mid = (lo + hi) / 2
                if coast_reachable(x, mid, b):
                    hi = mid
                else:
                    lo = mid
            print(f"    {x_off*100:.0f}%地点（{x:.4f} km・残り{(b-x)*1000:.0f}m）: {hi:.1f} km/h 以上")

    print("\n=== CBTC現示（先行の先頭から70m手前が停止限界）===")
    for d in (0.05, 0.07, 0.1, 0.2, 0.3, 0.5):
        print(f"  先行まで {d*1000:.0f} m → {cbtc_signal_speed(pos[0], pos[0]+d, 70.0):5.1f} km/h")

    t0 = _time.perf_counter()
    for _ in range(100):
        calculate_required_speed(45.0, pos[1] + 0.3, pos[2], 80.0, 70.0)
    print(f"\ncalculate_required_speed: {(_time.perf_counter()-t0)/100*1000:.2f} ms/回")
