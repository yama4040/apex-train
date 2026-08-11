# -*- coding: utf-8 -*-
"""
数理モデルによる標準運転曲線（省エネルギー・定時運行）生成スクリプト

■ 目的
    DQN（apex2.py）が学習する「通常運転モード（先行列車なし・自列車遅延なし）」の
    テストケースと同一の駅間・同一の物理モデルに対して、
    「定時（標準運転時間）を満たしつつ力行エネルギーが最小となる運転曲線」を
    数値計算（最適スイッチング点の探索）で求め、比較用の基準（標準運転曲線）として出力する。

■ 物理モデルの前提（すべて既存実装と一致させている）
    - 運動方程式・引張力・走行抵抗・ブレーキ減速度は train.py の Train.step と同一
      （積分刻み 0.01 秒、力行/惰行/制動の3ノッチのみ）。
    - 勾配抵抗・曲線抵抗・制限速度は track.py と同一の参照規則（区間の境界の扱いまで一致）。
    - ノッチ判断の周期は environment2.Environment.time_step と同一
      （駅手前100m以内で 1.0 秒 → 0.1 秒）。DQNと同じ操作粒度で比較できるようにするため。
      ただし「惰行開始点」と「制動開始点」の2つのスイッチング点のみ、
      最適制御の設計変数として 0.01 秒（≒数cm）刻みで配置する。
      制動開始が1秒粗いと最大十数mの停止位置誤差になり、基準曲線として使えないため。

■ 探索する運転パターン（最適列車制御の標準形）
        力行 → 定速保持（力行と惰行のバンバン制御） → 惰行 → 制動
    設計変数は
        V_hold : 定速保持速度[km/h]
        x_coast: 惰行開始位置[km]
    の2つ。x_coast は「到着時刻＝標準運転時間」となるよう二分探索で決定し（等式制約）、
    V_hold は総力行エネルギーが最小となるものをグリッド探索で選ぶ。
    V_hold を制限速度に固定すれば「力行→惰行→制動」の3ノッチ運転（--strategy pcb）になり、
    既存の input/sr_11.csv と同じ形の運転曲線が得られる。
    制動開始点は「駅にちょうど停止する制動曲線」を駅から逆方向に積分して事前に求めるため、
    勾配の変化（白兎駅手前の上り9.2‰など）も正確に織り込まれる。

■ 出力
    - <出力先>/<名前>.png        : 標準運転曲線（apex2.py が出力する運転曲線PNGと同一書式。
                                   dpi200・10×10インチ・駅線・制限速度の階段線・モード別配色）。
    - <出力先>/<名前>_detail.png : 勾配を含めた標準運転曲線。ノッチ別に色分けし、
                                   到着時刻・停止位置誤差・力行エネルギー等の指標を併記する。
    - <出力先>/<名前>.csv        : apex2.py の Tester と同一スキーマ（52列）の走行ログ。
                                   drive_monitor.py の「新形式」としてそのまま再生・比較できる。
                                   （併せて drive_monitor.py 用の <名前>_meta.json も出力する）
    - （任意）--sr-out 指定時     : input/sr_*.csv と同じ形式（position,time,speed,action）の1秒刻みCSV。

■ 使い方
    python generate_standard_curve.py
    python generate_standard_curve.py --strategy pcb                    # 力行→惰行→制動のみ
    python generate_standard_curve.py --compare data/<run>/0_0.csv      # DQNの走行ログと比較
    python drive_monitor.py standard_curve/standard_curve_11.csv data/<run>/0_0.csv  # 2本重ねて再生
"""

import argparse
import codecs
import csv
import json
import math
import os
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from actions import Actions  # noqa: E402
from track import Track  # noqa: E402
from train import Train  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 物理定数（train.py と同一。実体との齟齬は verify_physics_constants() で検出する）
# =============================================================================
SUB_DT = 0.01              # 積分刻み[s]（train.py の time_step_base）
FACTOR_OF_INERTIA = 28.34467
DECELERATE = -2.5          # 制動ノッチの減速度[km/h/s]
TRAIN_WEIGHT = 28.0        # 列車重量[t]
GRAVITY = 9.80665          # [N/kgf]
DEFAULT_DEPARTURE_INDEX = 11   # 羽前成田（apex2.py の Tester が env.reset(11, ...) 固定で使う駅）

# ノッチの表示色（運転曲線の色分け用）
NOTCH_COLOR = {
    Actions.acceleration: "#d62728",   # 力行
    Actions.coasting: "#2ca02c",       # 惰行
    Actions.deceleration: "#1f77b4",   # 制動
}
NOTCH_LABEL_JA = {
    Actions.acceleration: "力行",
    Actions.coasting: "惰行",
    Actions.deceleration: "制動",
}
NOTCH_LABEL_EN = {
    Actions.acceleration: "Power",
    Actions.coasting: "Coast",
    Actions.deceleration: "Brake",
}

# 運転モード→運転曲線の色（apex2.py と同一。DQN出力のPNGと同じ書式で描くために複製している。
# apex2.py を import すると ray / tensorflow まで読み込まれてしまうため、定義をここに写している）
MODE_COLORS = {"normal": "red", "delay_recovery": "green", "anti_mid_stop": "orange", "spacing": "purple"}
MODE_LABELS = {"normal": "Normal", "delay_recovery": "DelayRecovery",
               "anti_mid_stop": "AntiMidStop", "spacing": "Spacing"}


def plot_curve_by_mode(ax_plot, x, y, modes):
    """運転曲線をモード別に色分けして描画する（apex2.py の同名関数と同一の実装）。"""
    n = min(len(x), len(y), len(modes))
    if n == 0:
        return

    def norm(k):
        return k if k in MODE_COLORS else "normal"

    seen = set()
    i = 0
    while i < n - 1:
        m = norm(modes[i])
        j = i
        while j < n - 1 and norm(modes[j]) == m:
            j += 1
        lbl = None
        if m not in seen:
            seen.add(m)
            lbl = f"Own Train ({MODE_LABELS.get(m, m)})"
        ax_plot(x[i:j + 1], y[i:j + 1], color=MODE_COLORS[m], lw=1.3, label=lbl)
        i = j


def verify_physics_constants() -> None:
    """train.py の Train が持つ実際の値と本スクリプトの定数が一致するか検証する。
    train.py 側が変更された場合に、黙って食い違ったまま基準曲線を出すことを防ぐ。"""
    t = Train(0.0)
    mismatches = []
    if abs(t.time_step_base - SUB_DT) > 1e-12:
        mismatches.append(f"time_step_base: train.py={t.time_step_base} / 本script={SUB_DT}")
    if abs(t.FACTOR_OF_INERTIA - FACTOR_OF_INERTIA) > 1e-9:
        mismatches.append(f"FACTOR_OF_INERTIA: train.py={t.FACTOR_OF_INERTIA} / 本script={FACTOR_OF_INERTIA}")
    if abs(t.DECELERATE - DECELERATE) > 1e-9:
        mismatches.append(f"DECELERATE: train.py={t.DECELERATE} / 本script={DECELERATE}")
    if abs(t.TRAIN_WEIGHT - TRAIN_WEIGHT) > 1e-9:
        mismatches.append(f"TRAIN_WEIGHT: train.py={t.TRAIN_WEIGHT} / 本script={TRAIN_WEIGHT}")
    if mismatches:
        raise RuntimeError("train.py と物理定数が一致しません:\n  " + "\n  ".join(mismatches))


def tractive_force(speed_kmh: float) -> float:
    """引張力[kg/t]（train.py の Train.tractive_force と同一）"""
    if speed_kmh < 42.0:
        return -1.489 * speed_kmh + 92.408
    elif speed_kmh < 68.0:
        return -0.4 * speed_kmh + 46.68
    return -0.0963 * speed_kmh + 26.0284


def travel_resistance(speed_kmh: float) -> float:
    """走行抵抗[kg/t]（train.py の Train.travel_resistance と同一）"""
    return 2.39 + 0.0224 * speed_kmh + 0.00062 * (speed_kmh ** 2)


# =============================================================================
# 路線データの高速参照（track.py と同一の区間判定を二分探索で行う）
# =============================================================================
class TrackLookup:
    """勾配抵抗・曲線抵抗・制限速度を track.Track と同じ規則で返す（内部は二分探索）。"""

    def __init__(self, track: Track):
        self._grade_starts = [g["start"] for g in track.grade]
        self._grade_vals = [g["grade"] for g in track.grade]
        self._curve_starts = [c["start"] for c in track.curve]
        self._curve_vals = [c["curve"] for c in track.curve]
        self._sec_starts = [s["start"] for s in track.sections]
        self._sec_limits = [s["speed_limit"] for s in track.sections]

    def grade(self, position: float) -> float:
        # Track.get_grade_resistance は「starts[i] <= pos <= starts[i+1] を満たす最初の i」を返す。
        # 境界にちょうど乗った場合は手前側の区間を採用する点まで含めて bisect_left で一致する。
        i = bisect_left(self._grade_starts, position) - 1
        if i < 0 or i >= len(self._grade_starts) - 1:
            return 0.0
        return self._grade_vals[i]

    def curve(self, position: float) -> float:
        i = bisect_left(self._curve_starts, position) - 1
        if i < 0 or i >= len(self._curve_starts) - 1:
            return 0.0
        return self._curve_vals[i]

    def speed_limit(self, position: float) -> float:
        # Track.get_section_id は「pos < starts[i+1] を満たす最初の i」＝直近の区間先頭を返す。
        i = bisect_right(self._sec_starts, position) - 1
        if i < 0:
            i = 0
        if i >= len(self._sec_limits):
            i = len(self._sec_limits) - 1
        return self._sec_limits[i]

    def min_speed_limit(self, start: float, end: float) -> float:
        """区間[start, end]内の最小制限速度[km/h]"""
        limits = [self.speed_limit(start)]
        for s, lim in zip(self._sec_starts, self._sec_limits):
            if start < s <= end:
                limits.append(lim)
        return min(limits)

    def limit_sections(self, start: float, end: float) -> List[dict]:
        """[start, end] を制限速度ごとに区切ったリスト（meta.json / 作図用）"""
        out = []
        pos = start
        while pos < end - 1e-12:
            lim = self.speed_limit(pos)
            nxt = end
            for s in self._sec_starts:
                if pos < s < end:
                    nxt = min(nxt, s)
                    break
            out.append({"start": pos, "distance": nxt - pos, "speed_limit": lim})
            pos = nxt
        return out


# =============================================================================
# 走行シミュレーション
# =============================================================================
@dataclass
class Row:
    """1制御周期分の記録（各値は周期の開始時点＝そのノッチを選んだ瞬間の状態）"""
    t: float
    position: float
    speed: float
    action: int
    dt: float


@dataclass
class RunResult:
    ok: bool                       # 駅に停止できたか（駅間停車・オーバーランは False）
    reason: str = ""
    hold_used: bool = False        # 定速保持（バンバン制御）が実際に発生したか
    time: float = math.inf         # 到着時刻[s]
    stop_position: float = 0.0     # 停止位置[km]
    stop_error_m: float = 0.0      # 停止位置誤差[m]（+が過走）
    energy_j: float = 0.0          # 力行仕事[J]（車輪周・回生なし）
    max_speed: float = 0.0
    brake_speed: float = 0.0       # 制動開始速度[km/h]
    coast_position: float = 0.0    # 惰行開始位置[km]
    brake_position: float = 0.0    # 制動開始位置[km]
    notch_changes: int = 0
    power_time: float = 0.0
    coast_time: float = 0.0
    brake_time: float = 0.0
    rows: List[Row] = field(default_factory=list)

    @property
    def energy_kwh(self) -> float:
        return self.energy_j / 3.6e6

    def notch_segments(self) -> List[dict]:
        """同一ノッチが続く区間のリスト（運転パターンの要約用）"""
        segs = []
        for r in self.rows:
            if segs and segs[-1]["action"] == r.action:
                segs[-1]["t_end"] = r.t
                segs[-1]["x_end"] = r.position
                segs[-1]["v_end"] = r.speed
            else:
                segs.append({"action": r.action, "t_start": r.t, "t_end": r.t,
                             "x_start": r.position, "x_end": r.position,
                             "v_start": r.speed, "v_end": r.speed})
        return segs

    def pattern_text(self, jp: bool = True) -> str:
        """「力行(0-51s) → 惰行(51-172s) → 制動(172-180s)」のような運転パターン文字列"""
        labels = NOTCH_LABEL_JA if jp else NOTCH_LABEL_EN
        return " → ".join(
            f"{labels[Actions(s['action'])]}({s['t_start']:.0f}-{s['t_end']:.0f}s, "
            f"{s['v_start']:.0f}→{s['v_end']:.0f}km/h)" for s in self.notch_segments())


class StandardCurveSolver:
    """駅間の標準運転曲線（省エネ・定時）を数値計算で求めるソルバ"""

    def __init__(self, departure_index: int = DEFAULT_DEPARTURE_INDEX,
                 target_time: Optional[float] = None,
                 base_time_step: float = 1.0,
                 hold_band: float = 2.0,
                 weight_correction: float = 1.0,
                 max_time: float = 400.0):
        self.track = Track()
        self.lookup = TrackLookup(self.track)
        stations = self._read_stations()
        self.departure_index = departure_index
        self.departure_station = stations[departure_index]
        self.arrival_station = stations[departure_index + 1]
        self.start_position = float(self.departure_station["position"])
        self.station_position = float(self.arrival_station["position"])
        self.distance_km = self.station_position - self.start_position
        # 標準運転時間は Station.csv の出発駅の rt（environment2.remaining_time と同じ定義）
        self.target_time = float(self.departure_station["running_time"]) if target_time is None else float(target_time)
        self.base_time_step = base_time_step
        self.hold_band = hold_band
        self.wc = weight_correction
        self.max_time = max_time
        self.speed_cap = self.lookup.min_speed_limit(self.start_position, self.station_position)
        self._build_brake_curve()

    def _read_stations(self) -> List[dict]:
        with codecs.open(os.path.join(BASE_DIR, "input", "Station.csv"), "r", "utf-8", "ignore") as f:
            df = pd.read_csv(f)
        return [{"name": str(df["name"][i]), "position": float(df["position"][i]),
                 "running_time": float(df["rt"][i])} for i in range(len(df))]

    # ---------------------------------------------------------------- 制動曲線
    def _build_brake_curve(self, v_max: float = 90.0, grid_km: float = 1e-5) -> None:
        """到着駅にちょうど停止する制動曲線 v_brake(x) を駅から逆向きに積分して求める。

        train.py の前進更新式をそのまま逆に解くことで、勾配・曲線・走行抵抗の位置変化を
        すべて織り込んだ「これ以上は待てない制動開始点」が数cm精度で得られる。
        参照コストを一定にするため、最後に位置一定間隔（既定1cm）のテーブルへ載せ替える。
        """
        xs = [self.station_position]
        vs = [0.0]
        x = self.station_position
        v = 0.0
        while v < v_max and x > self.start_position - 0.5:
            rr = travel_resistance(v)
            rg = self.lookup.grade(x)
            rc = self.lookup.curve(x)
            a = ((0.0 - rr) * self.wc - (rg + rc)) / FACTOR_OF_INERTIA + DECELERATE * self.wc
            if a >= 0.0:
                a = -0.0001
            v_prev = v - a * SUB_DT
            x_prev = x - (v_prev / 3600.0) * SUB_DT - (a / 3600.0) * (SUB_DT ** 2)
            x, v = x_prev, v_prev
            xs.append(x)
            vs.append(v)
        xs = np.asarray(xs[::-1])   # 位置の昇順に並べ替え
        vs = np.asarray(vs[::-1])
        self._brake_x0 = float(xs[0])
        self._brake_dx = grid_km
        n = int((self.station_position - self._brake_x0) / grid_km) + 2
        grid = self._brake_x0 + grid_km * np.arange(n)
        self._brake_table = np.interp(grid, xs, vs).tolist()

    def brake_curve_speed(self, position: float) -> float:
        """位置 position における「駅にちょうど停止するための速度」[km/h]。
        これを上回っていれば直ちに制動が必要。制動曲線の範囲外（駅から遠い）は無限大扱い。"""
        i = int((position - self._brake_x0) / self._brake_dx)
        if i < 0:
            return math.inf
        if i >= len(self._brake_table):
            return 0.0
        return self._brake_table[i]

    # -------------------------------------------------------------- 走行の再現
    def simulate(self, v_hold: float, x_coast: float, sub_dt: float = SUB_DT,
                 record: bool = False) -> RunResult:
        """(定速保持速度, 惰行開始位置) を与えて1回の駅間走行を再現する。

        ノッチの判断は environment2 と同じ制御周期（駅手前100m以内は0.1秒）で行い、
        「惰行開始点に到達」「制動曲線に到達」の2イベントのみ周期の途中でも切り替える
        （＝最適制御のスイッチング点は sub_dt 刻みで配置される）。
        """
        wc = self.wc
        lk = self.lookup
        station = self.station_position
        pos = self.start_position
        v = 0.0
        t = 0.0
        energy = 0.0
        braking = False
        hold_coast = False      # 定速保持のヒステリシスで惰行側にいるか
        max_speed = 0.0
        brake_speed = 0.0
        brake_position = 0.0
        power_t = coast_t = brake_t = 0.0
        notch_changes = 0
        prev_action = None
        rows: List[Row] = []
        result = RunResult(ok=False)

        while True:
            if t > self.max_time:
                result.reason = "時間超過（駅に到達しない）"
                return result

            # --- 制御周期（environment2.Environment.time_step と同一規則） ---
            dt_rule = self.base_time_step if pos < station - 0.1 else self.base_time_step * 0.1

            # --- ノッチの決定 ---
            if braking or v >= self.brake_curve_speed(pos):
                if not braking:
                    braking = True
                    brake_speed = v
                    brake_position = pos
                action = Actions.deceleration
            elif pos >= x_coast:
                action = Actions.coasting
            else:
                limit = min(v_hold, lk.speed_limit(pos))
                if v >= limit:
                    if not hold_coast:
                        # 惰行開始点より手前で保持速度に達した＝定速保持が実際に働いた
                        result.hold_used = True
                    hold_coast = True
                elif v <= limit - self.hold_band:
                    hold_coast = False
                action = Actions.coasting if hold_coast else Actions.acceleration

            if prev_action is not None and action != prev_action:
                notch_changes += 1
            prev_action = action

            row_t, row_pos, row_v = t, pos, v

            # --- サブステップ積分（train.py の Train.step と同一の更新式） ---
            elapsed = 0.0
            stopped = False
            while elapsed < dt_rule - 1e-9:
                force = tractive_force(v) if action == Actions.acceleration else 0.0
                rr = travel_resistance(v)
                rg = lk.grade(pos)
                rc = lk.curve(pos)
                accel = (((force - rr) * wc) - (rg + rc)) / FACTOR_OF_INERTIA
                if action == Actions.deceleration:
                    accel += DECELERATE * wc
                if action == Actions.acceleration:
                    # 力行仕事[J] = 引張力[kgf/t] × 重量[t] × g × 速度[m/s] × 時間[s]
                    energy += force * TRAIN_WEIGHT * GRAVITY * (v / 3.6) * sub_dt
                if v + accel * sub_dt >= 0.0:
                    pos += (v / 3600.0) * sub_dt + (accel / 3600.0) * (sub_dt ** 2)
                    v += accel * sub_dt
                else:
                    v = 0.0
                elapsed += sub_dt
                t += sub_dt
                if v > max_speed:
                    max_speed = v

                if v <= 0.0:
                    stopped = True
                    break
                if pos >= station and not braking:
                    # 制動に入る前に駅を通過＝オーバーラン。制動中は停止まで積分を続ける
                    # （駅位置を跨いだ瞬間に周期を刻むと、停止直前に0.01秒の行が並んでしまうため）
                    break
                # スイッチング点に到達したら周期の途中でも打ち切って判断し直す
                if not braking and v >= self.brake_curve_speed(pos):
                    break
                if action == Actions.acceleration and pos >= x_coast:
                    break

            if action == Actions.acceleration:
                power_t += elapsed
            elif action == Actions.coasting:
                coast_t += elapsed
            else:
                brake_t += elapsed

            if record:
                rows.append(Row(row_t, row_pos, row_v, int(action), elapsed))

            if stopped:
                if not braking:
                    result.reason = "駅間停車（惰行中に停止）"
                    return result
                break
            if pos > station + 0.005 and v > 0.0:
                result.reason = "オーバーラン（駅の許容範囲を超えて走行）"
                return result

        if record:
            # 停止した状態も1行として残す（モニターで到着まで再生できるようにするため）
            rows.append(Row(t, pos, v, int(Actions.deceleration), 0.0))

        result.ok = True
        result.time = t
        result.stop_position = pos
        result.stop_error_m = (pos - station) * 1000.0
        result.energy_j = energy
        result.max_speed = max_speed
        result.brake_speed = brake_speed
        result.brake_position = brake_position
        result.coast_position = min(max(x_coast, self.start_position), brake_position)
        result.notch_changes = notch_changes
        result.power_time = power_t
        result.coast_time = coast_t
        result.brake_time = brake_t
        result.rows = rows
        return result

    # ------------------------------------------------------------------ 最適化
    def solve_coast_point(self, v_hold: float, sub_dt: float = SUB_DT,
                          iterations: int = 34, record: bool = False) -> Optional[RunResult]:
        """定速保持速度 v_hold のもとで「到着時刻＝標準運転時間」となる惰行開始位置を二分探索する。

        惰行開始位置を後ろへずらすほど力行が長く到着が早まるため、到着時刻は惰行開始位置の
        単調減少関数になる（定速保持のバンバン制御による数十msの揺らぎは無視できる）。
        時間内に到着できない v_hold（低すぎる保持速度）では None を返す。
        """
        hi = self.station_position           # 制動開始まで力行/定速保持＝この v_hold での最短時間
        best = self.simulate(v_hold, hi, sub_dt)
        if not best.ok or best.time > self.target_time:
            return None
        lo = self.start_position             # 発車直後から惰行＝到達不能または最も遅い
        for _ in range(iterations):
            if hi - lo < 1e-6:               # 1mm 未満まで詰めたら終了
                break
            mid = (lo + hi) / 2.0
            r = self.simulate(v_hold, mid, sub_dt)
            if (not r.ok) or r.time > self.target_time:
                lo = mid                     # 遅すぎる／到達しない → もっと力行する
            else:
                hi = mid
                best = r
        if record:
            best = self.simulate(v_hold, hi, sub_dt, record=True)
        return best

    def optimize(self, sub_dt_search: float = 0.05, sub_dt_final: float = SUB_DT,
                 coarse_step: float = 2.0, fine_step: float = 0.25,
                 verbose: bool = True) -> (RunResult, float, List[tuple]):
        """定速保持速度をグリッド探索し、力行エネルギー最小の運転曲線を求める。

        戻り値: (最適解, 最適な定速保持速度, [(保持速度, 結果), ...]（粗探索の一覧）)
        """
        candidates = []
        v = 10.0
        coarse = []
        while v <= self.speed_cap + 1e-9:
            coarse.append(round(v, 3))
            v += coarse_step
        if abs(coarse[-1] - self.speed_cap) > 1e-9:
            coarse.append(self.speed_cap)

        for vh in coarse:
            r = self.solve_coast_point(vh, sub_dt_search)
            if r is not None:
                candidates.append((vh, r))
        if not candidates:
            raise RuntimeError(
                f"標準運転時間 {self.target_time:.0f} 秒で到達できる運転曲線が見つかりません"
                f"（制限速度 {self.speed_cap:.0f}km/h で最短時間より短い時分が要求されています）")

        if verbose:
            print("[粗探索] 定速保持速度ごとの力行エネルギー")
            for vh, r in candidates:
                print(f"    V_hold={vh:5.1f} km/h  到着={r.time:6.2f}s  "
                      f"惰行開始={r.coast_position:8.4f}km  E={r.energy_kwh:6.3f} kWh  "
                      f"切替={r.notch_changes:3d}回")

        best_vh = min(candidates, key=lambda kv: kv[1].energy_j)[0]

        # 粗探索の最良点の周辺を細かく再探索する
        fine = []
        vv = max(10.0, best_vh - coarse_step)
        while vv <= min(self.speed_cap, best_vh + coarse_step) + 1e-9:
            fine.append(round(vv, 3))
            vv += fine_step
        fine_results = []
        for vh in fine:
            r = self.solve_coast_point(vh, sub_dt_search)
            if r is not None:
                fine_results.append((vh, r))
        if fine_results:
            best_vh = min(fine_results, key=lambda kv: kv[1].energy_j)[0]

        # 最終解のみ train.py と同じ 0.01 秒刻みで作り直す（探索は粗い刻みで高速化していたため）
        best = self.solve_coast_point(best_vh, sub_dt_final, record=True)
        if best is None:
            # 刻みを細かくした結果わずかに間に合わなくなった場合は保持速度を上げて解き直す
            for vh in [v for v in fine if v > best_vh] + [self.speed_cap]:
                best = self.solve_coast_point(vh, sub_dt_final, record=True)
                if best is not None:
                    best_vh = vh
                    break
        if best is None:
            raise RuntimeError("最終刻みでの解の再構成に失敗しました")
        return best, best_vh, candidates

    def solve_fixed(self, v_hold: float, sub_dt_final: float = SUB_DT) -> RunResult:
        """定速保持速度を固定して解く（--strategy pcb は制限速度に固定＝力行→惰行→制動）。"""
        r = self.solve_coast_point(v_hold, sub_dt_final, record=True)
        if r is None:
            raise RuntimeError(f"V_hold={v_hold:.1f}km/h では標準運転時間 {self.target_time:.0f} 秒に間に合いません")
        return r


# =============================================================================
# 出力（CSV / meta.json / PNG）
# =============================================================================
def build_environment(departure_index: int, base_time_step: float):
    """観測ベクトル（30次元）を DQN と完全に同じ式で得るための Environment を用意する。
    報酬予測NN（TensorFlow）は不要なので読み込ませない。"""
    import environment2
    environment2.DirectRewardPredictor = None   # コンストラクタでのモデル読み込みを止める
    env = environment2.Environment(time_step=base_time_step, load_reward_predictor=False)
    env.reset(departure_index, 0.0)
    return env


CSV_HEADER = [
    # 1. 生の観測値（raw_state: 8次元）
    "raw_speed", "raw_stat_dist", "raw_rem_time", "raw_hold_time",
    "raw_pre_act", "raw_stat_dist_2", "raw_fw_dist", "raw_cbtc_signal",
    # 2. ネットワーク入力値（normalized_state: 30次元）
    "norm_speed", "norm_stat_dist_wide", "norm_stat_dist_zoom", "norm_rem_time", "norm_hold_time",
    "norm_pre_act_c", "norm_pre_act_a", "norm_pre_act_d", "norm_fw_dist",
    "norm_cbtc_signal", "norm_speed_limit", "norm_req_stop_dist", "norm_margin_stop_dist",
    "phase_accel", "phase_cruise", "phase_limit", "phase_decel", "phase_stop",
    "norm_fw_speed",
    "norm_gradient", "norm_next_grade_dist", "norm_next_grade_val",
    "norm_next_limit_dist", "norm_next_limit_val", "norm_prev_notch_duration",
    "norm_target_no_stop", "mode_normal", "mode_delay_recovery", "mode_anti_mid_stop", "mode_spacing",
    # 3. ネットワークの出力と報酬情報（本スクリプトではNNを使わないため0埋め）
    "Q_coast", "Q_accel", "Q_decel", "step_reward", "llm_reward",
    # 4. 運転曲線モニター用の生値（9列）
    "time", "position", "speed_limit", "fw_position", "fw_speed", "mode", "action",
    "gradient", "fw_dwell_elapsed",
]


def write_log_csv(solver: StandardCurveSolver, result: RunResult, path: str) -> None:
    """apex2.py の Tester と同一スキーマの走行ログCSVを書き出す（drive_monitor.py の新形式）。

    Q値・報酬列はニューラルネットを使っていないため0で埋める。
    観測ベクトルは environment2.Environment に状態を流し込んで生成するので、
    DQNが同じ状況で見る値と完全に一致する。
    """
    env = build_environment(solver.departure_index, solver.base_time_step)
    # environment2.reset 直後と同じ初期値から、step() 末尾と同じ規則で更新していく
    pre_action = Actions.deceleration
    holding_time = 30.0
    prev_notch = None
    prev_notch_duration = 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in result.rows:
            action = Actions(row.action)
            env.train.set_states(row.speed, row.position)
            env.t = row.t
            env.pre_action = pre_action
            env.holding_time = holding_time
            env.prev_notch = prev_notch
            env.prev_notch_duration = prev_notch_duration
            env.current_mode = "normal"

            raw = env.raw_state
            norm = env.normalized_state
            gradient = solver.lookup.grade(row.position)
            monitor = [row.t, row.position, env.current_speed_limit, "", "", "normal",
                       int(action), gradient, 0.0]
            writer.writerow([*raw, *norm, 0.0, 0.0, 0.0, 0.0, 0.0, *monitor])

            if pre_action == action:
                holding_time += row.dt
            else:
                prev_notch = pre_action
                prev_notch_duration = holding_time
                holding_time = row.dt
            pre_action = action


def write_meta_json(solver: StandardCurveSolver, result: RunResult, v_hold: float,
                    path: str, desc: str) -> None:
    """drive_monitor.py 用のメタ情報。標準運転曲線固有の指標も併記しておく。"""
    meta = {
        "desc": desc,
        "case_index": 0,
        "ego_delay": 0.0,
        "forward_delay": 0.0,
        "forward_dwell": None,
        "headway": None,
        "f_train_csv": None,
        "has_forward_train": False,
        "departure_station": {"name": solver.departure_station["name"],
                              "position": solver.start_position},
        "arrival_station": {"name": solver.arrival_station["name"],
                            "position": solver.station_position},
        "standard_running_time": solver.target_time,
        "base_time_step": solver.base_time_step,
        "speed_limit_sections": [
            {"start": round(s["start"], 6), "distance": round(s["distance"], 6),
             "speed_limit": float(s["speed_limit"])}
            for s in solver.lookup.limit_sections(solver.start_position, solver.station_position)
        ],
        # ▼本スクリプト固有（drive_monitor.py は未知のキーを無視する）
        "standard_curve": {
            "source": "generate_standard_curve.py",
            "hold_speed_kmh": round(v_hold, 3),
            "coast_position_km": round(result.coast_position, 6),
            "brake_position_km": round(result.brake_position, 6),
            "brake_entry_speed_kmh": round(result.brake_speed, 3),
            "max_speed_kmh": round(result.max_speed, 3),
            "arrival_time_s": round(result.time, 3),
            "stop_error_m": round(result.stop_error_m, 4),
            "traction_energy_kwh": round(result.energy_kwh, 5),
            "notch_changes": result.notch_changes,
            "power_time_s": round(result.power_time, 2),
            "coast_time_s": round(result.coast_time, 2),
            "brake_time_s": round(result.brake_time, 2),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_sr_csv(solver: StandardCurveSolver, result: RunResult, path: str) -> None:
    """input/sr_*.csv と同じ形式（position,time,speed,action）で1秒刻みに出力する。
    environment2 の位置基準の遅延計算（_scheduled_time_at）に使える標準走行曲線。"""
    t = np.array([r.t for r in result.rows])
    pos = np.array([r.position for r in result.rows])
    spd = np.array([r.speed for r in result.rows])
    act = [Actions(r.action) for r in result.rows]
    grid = np.arange(0.0, math.floor(result.time) + 1.0, 1.0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "time", "speed", "action"])
        for g in grid:
            i = int(np.searchsorted(t, g, side="right") - 1)
            i = max(0, min(i, len(t) - 1))
            # actionは input/sr_*.csv・input/f_train_*.csv と同じ "Actions.xxx" 表記で書く
            w.writerow([np.interp(g, t, pos), int(g), np.interp(g, t, spd),
                        f"Actions.{act[i].name}"])


# ----------------------------------------------------------------- 図の描画
_JP_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
]


def setup_japanese_font() -> bool:
    """日本語フォントを matplotlib に登録する。見つからなければ False（英語ラベルにする）。"""
    for path in _JP_FONT_FILES:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
        return True
    return False


def _plot_by_notch(ax, xs, ys, actions, jp: bool, lw=1.8, alpha=1.0, label_prefix=""):
    """ノッチごとに色を変えて折れ線を描く（凡例は種類ごとに1つだけ付ける）"""
    labels = NOTCH_LABEL_JA if jp else NOTCH_LABEL_EN
    used = set()
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or actions[i] != actions[start]:
            a = Actions(actions[start])
            seg = slice(start, min(i + 1, len(xs)))
            lbl = None
            if a not in used:
                used.add(a)
                lbl = label_prefix + labels[a]
            ax.plot(xs[seg], ys[seg], color=NOTCH_COLOR[a], lw=lw, alpha=alpha, label=lbl)
            start = i


def plot_dqn_style(solver: StandardCurveSolver, result: RunResult, path: str) -> None:
    """apex2.py の Tester が出力する運転曲線PNGと同一書式で保存する。

    dpi=200・10×10インチ、駅の縦線と地面の横線（黒・lw3）、区間ごとの制限速度の階段線、
    運転モード別の配色（本スクリプトは通常運転モードのみなので赤一色）、
    軸ラベル・凡例位置まで apex2.py の描画に合わせてある（DQNの出力と直接並べられるようにするため）。
    """
    pos = [r.position for r in result.rows]
    spd = [r.speed for r in result.rows]
    modes = ["normal"] * len(result.rows)

    plt.figure(dpi=200, figsize=(10, 10))
    plt.xlabel("Position[km]")
    plt.ylabel("Speed[km/h]")
    plt.plot([solver.start_position, solver.station_position], [0, 0], "k-", lw=3)
    plt.plot([solver.start_position, solver.start_position], [0, 100], "k-", lw=3)
    plt.plot([solver.station_position, solver.station_position], [0, 100], "k-", lw=3)

    sections = solver.lookup.limit_sections(solver.start_position, solver.station_position)
    for i, s in enumerate(sections):
        plt.plot([s["start"], s["start"] + s["distance"]],
                 [s["speed_limit"], s["speed_limit"]], "k-", lw=1)
        if i > 0:
            plt.plot([s["start"], s["start"]],
                     [s["speed_limit"], sections[i - 1]["speed_limit"]], "k-", lw=1)

    plot_curve_by_mode(plt.plot, pos, spd, modes)
    plt.legend(loc="upper right")
    plt.savefig(path)
    plt.close("all")


def plot_detail(solver: StandardCurveSolver, result: RunResult, v_hold: float,
                path: str, jp: bool, compare: Optional[dict] = None) -> None:
    """勾配を含めた標準運転曲線。ノッチ別の配色とスイッチング点、各種指標を併記する。"""
    pos = np.array([r.position for r in result.rows])
    spd = np.array([r.speed for r in result.rows])
    act = [r.action for r in result.rows]

    T = lambda ja, en: ja if jp else en  # noqa: E731

    fig = plt.figure(figsize=(12.0, 8.0), dpi=200)
    ax = fig.add_axes([0.075, 0.30, 0.855, 0.60])

    # --- 勾配（背景の帯。上り＝赤系、下り＝青系） ---
    ax_g = ax.twinx()
    gp = np.linspace(solver.start_position, solver.station_position, 1200)
    gv = np.array([solver.lookup.grade(p) for p in gp])
    ax_g.fill_between(gp, 0, gv, where=gv >= 0, color="#d62728", alpha=0.13, lw=0)
    ax_g.fill_between(gp, 0, gv, where=gv < 0, color="#1f77b4", alpha=0.13, lw=0)
    ax_g.plot(gp, gv, color="0.45", lw=0.8)
    ax_g.axhline(0.0, color="0.65", lw=0.6)
    ax_g.set_ylabel(T("勾配 [‰]（上り+／下り−）", "Gradient [permil]"))
    # 勾配の帯が運転曲線と重ならないよう、右軸の目盛を図の下部1/4に押し込む
    ax_g.set_ylim(-20, 100)
    ax_g.set_yticks([-20, -10, 0, 10, 20])
    ax_g.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # --- 制限速度・駅・運転曲線 ---
    for s in solver.lookup.limit_sections(solver.start_position, solver.station_position):
        ax.plot([s["start"], s["start"] + s["distance"]],
                [s["speed_limit"], s["speed_limit"]], color="0.3", lw=1.2, ls="--")
    ax.plot([], [], color="0.3", lw=1.2, ls="--", label=T("路線制限速度", "Speed limit"))
    _plot_by_notch(ax, pos, spd, act, jp, lw=2.0)
    if compare is not None:
        ax.plot(compare["position"], compare["speed"], color="0.5", lw=1.3, ls="-.",
                label=compare["label"])
    ax.axvline(solver.start_position, color="k", lw=2.5)
    ax.axvline(solver.station_position, color="k", lw=2.5)
    if result.coast_position < result.brake_position - 1e-6:
        i = min(int(np.searchsorted(pos, result.coast_position)), len(spd) - 1)
        ax.plot(result.coast_position, spd[i], "o", color="#2ca02c", ms=9,
                label=T("惰行開始点", "Coast onset"))
    ax.plot(result.brake_position, result.brake_speed, "v", color="#1f77b4", ms=10,
            label=T("制動開始点", "Brake onset"))

    ax.set_xlabel(T("位置 [km]", "Position [km]"))
    ax.set_ylabel(T("速度 [km/h]", "Speed [km/h]"))
    ax.set_xlim(solver.start_position - 0.02, solver.station_position + 0.02)
    ax.set_ylim(0, max(solver.speed_cap, result.max_speed) * 1.18)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=6, fontsize=9)

    dep_name = solver.departure_station["name"]
    arr_name = solver.arrival_station["name"]
    fig.suptitle(T(f"数理モデルによる標準運転曲線（勾配つき）　{dep_name} → {arr_name}"
                   f"　{solver.distance_km*1000:.0f}m／標準運転時間 {solver.target_time:.0f}秒",
                   f"Optimal standard driving curve with gradient  {dep_name} -> {arr_name}"),
                 fontsize=14, y=0.965)

    # --- 指標（図の下部にまとめて表示） ---
    hold_txt = (T(f"定速保持 {v_hold:.2f} km/h", f"hold {v_hold:.2f} km/h") if result.hold_used
                else T(f"定速保持なし（力行上限 {v_hold:.2f} km/h に未到達）",
                       f"no holding phase (cap {v_hold:.2f} km/h not reached)"))
    if jp:
        lines = [
            f"運転パターン　　： {result.pattern_text(jp)}",
            f"　　　　　　　　　 ［{hold_txt}］",
            f"到着時刻　　　　： {result.time:.2f} 秒　（標準運転時間 {solver.target_time:.0f} 秒との差 "
            f"{result.time - solver.target_time:+.2f} 秒）",
            f"停止位置誤差　　： {result.stop_error_m:+.3f} m　／　最高速度 {result.max_speed:.2f} km/h"
            f"　／　制動開始速度 {result.brake_speed:.2f} km/h",
            f"力行エネルギー　： {result.energy_kwh:.3f} kWh　({result.energy_kwh / solver.distance_km:.3f} kWh/km"
            f"・回生なし・車輪周)　／　ノッチ切替 {result.notch_changes} 回",
            f"力行／惰行／制動： {result.power_time:.1f} ／ {result.coast_time:.1f} ／ {result.brake_time:.1f} 秒"
            f"　／　惰行開始 {result.coast_position:.4f} km・制動開始 {result.brake_position:.4f} km",
        ]
    else:
        lines = [
            f"pattern      : {result.pattern_text(jp)}  [{hold_txt}]",
            f"arrival      : {result.time:.2f} s ({result.time - solver.target_time:+.2f} s vs "
            f"{solver.target_time:.0f} s)",
            f"stop error   : {result.stop_error_m:+.3f} m / vmax {result.max_speed:.2f} km/h / "
            f"brake entry {result.brake_speed:.2f} km/h",
            f"energy       : {result.energy_kwh:.3f} kWh ({result.energy_kwh / solver.distance_km:.3f} kWh/km) / "
            f"{result.notch_changes} notch changes",
            f"power/coast/brake: {result.power_time:.1f} / {result.coast_time:.1f} / {result.brake_time:.1f} s",
        ]
    if compare is not None:
        lines.append(compare["info"])
    fig.text(0.075, 0.215, "\n".join(lines), ha="left", va="top", fontsize=10.5,
             linespacing=1.6)

    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# DQN走行ログとの比較
# =============================================================================
def load_comparison(csv_path: str, solver: StandardCurveSolver, jp: bool) -> dict:
    """DQNの走行ログ（apex2.py Tester の出力CSV）を読み、同じ指標を算出する。"""
    with codecs.open(csv_path, "r", "utf-8", "ignore") as f:
        df = pd.read_csv(f)
    if "time" in df.columns and "position" in df.columns:
        t = df["time"].to_numpy(dtype=float)
        pos = df["position"].to_numpy(dtype=float)
        action = pd.to_numeric(df["action"], errors="coerce").fillna(0).to_numpy(dtype=int)
    else:
        # 旧形式: 位置は駅残距離から復元し、時刻は environment2 の time_step 規則で再構成する
        dist = df["raw_stat_dist"].to_numpy(dtype=float)
        pos = solver.station_position - dist
        dt = np.where(dist > 0.1, solver.base_time_step, solver.base_time_step * 0.1)
        t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
        pre_act = df["raw_pre_act"].to_numpy(dtype=float)
        action = np.concatenate([pre_act[1:], pre_act[-1:]]).astype(int)
    speed = df["raw_speed"].to_numpy(dtype=float)

    dt = np.diff(t, append=t[-1])
    energy = 0.0
    for v, a, d in zip(speed, action, dt):
        if a == int(Actions.acceleration):
            energy += tractive_force(v) * TRAIN_WEIGHT * GRAVITY * (v / 3.6) * d
    changes = int(np.sum(action[1:] != action[:-1]))
    stop_error = (pos[-1] - solver.station_position) * 1000.0
    label = os.path.splitext(os.path.basename(csv_path))[0]
    if jp:
        info = (f"[比較] {label}：到着 {t[-1]:.2f} 秒／停止位置誤差 {stop_error:+.3f} m／"
                f"最高速度 {speed.max():.2f} km/h／力行エネルギー {energy/3.6e6:.3f} kWh／"
                f"ノッチ切替 {changes} 回")
    else:
        info = (f"[compare] {label}: arrival {t[-1]:.2f}s / stop error {stop_error:+.3f} m / "
                f"vmax {speed.max():.2f} km/h / energy {energy/3.6e6:.3f} kWh / {changes} changes")
    return {"label": label, "time": t, "position": pos, "speed": speed, "action": action,
            "energy_kwh": energy / 3.6e6, "notch_changes": changes,
            "arrival_time": float(t[-1]), "stop_error_m": stop_error,
            "max_speed": float(speed.max()), "info": info}


# =============================================================================
# メイン
# =============================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="数理モデルによる標準運転曲線（省エネ・定時）の生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--departure-index", type=int, default=DEFAULT_DEPARTURE_INDEX,
                        help="出発駅のindex（input/Station.csvの行番号。11=羽前成田→白兎）")
    parser.add_argument("--target-time", type=float, default=None,
                        help="標準運転時間[秒]（既定はStation.csvの出発駅のrt）")
    parser.add_argument("--strategy", choices=["auto", "pcb"], default="auto",
                        help="auto=力行→定速保持→惰行→制動のエネルギー最小解、"
                             "pcb=制限速度まで力行→惰行→制動の3ノッチ運転")
    parser.add_argument("--hold-band", type=float, default=2.0,
                        help="定速保持のヒステリシス幅[km/h]（小さいほどノッチ切替が増える）")
    parser.add_argument("--base-time-step", type=float, default=1.0,
                        help="ノッチ判断の基本周期[秒]（environment2と同じく駅手前100mで1/10になる）")
    parser.add_argument("--out-dir", default="standard_curve", help="出力先ディレクトリ")
    parser.add_argument("--name", default=None, help="出力ファイル名（既定: standard_curve_<出発駅index>）")
    parser.add_argument("--compare", default=None,
                        help="比較するDQN走行ログCSV（data/<run>/<file>_0.csv など）")
    parser.add_argument("--sr-out", default=None,
                        help="input/sr_*.csv と同じ形式の標準走行曲線も書き出す場合の出力パス")
    args = parser.parse_args(argv)

    verify_physics_constants()
    jp = setup_japanese_font()

    solver = StandardCurveSolver(departure_index=args.departure_index,
                                 target_time=args.target_time,
                                 base_time_step=args.base_time_step,
                                 hold_band=args.hold_band)
    print(f"■ 対象区間: {solver.departure_station['name']} ({solver.start_position:.3f} km) → "
          f"{solver.arrival_station['name']} ({solver.station_position:.3f} km)  "
          f"距離 {solver.distance_km*1000:.0f} m")
    print(f"■ 標準運転時間: {solver.target_time:.0f} 秒（表定速度 "
          f"{solver.distance_km / (solver.target_time/3600.0):.1f} km/h）"
          f" ／ 区間内の最小制限速度: {solver.speed_cap:.0f} km/h")

    if args.strategy == "pcb":
        v_hold = solver.speed_cap
        result = solver.solve_fixed(v_hold)
        print(f"[方式] 力行→惰行→制動（定速保持なし・制限速度 {v_hold:.0f} km/h まで力行可）")
    else:
        result, v_hold, _ = solver.optimize()
        if result.hold_used:
            print(f"[方式] 力行→定速保持→惰行→制動（エネルギー最小の定速保持速度 {v_hold:.2f} km/h）")
        else:
            print(f"[方式] 力行→惰行→制動（力行上限 {v_hold:.2f} km/h には到達せず定速保持は発生しない＝"
                  f"この区間では惰行を長く取る運転が最小エネルギー）")

    print("── 最適運転曲線 ─────────────────────────────")
    print(f"  運転パターン    : {result.pattern_text()}")
    print(f"  到着時刻        : {result.time:.2f} 秒（標準運転時間との差 {result.time - solver.target_time:+.2f} 秒）")
    print(f"  停止位置誤差    : {result.stop_error_m:+.3f} m")
    print(f"  最高速度        : {result.max_speed:.2f} km/h")
    print(f"  惰行開始位置    : {result.coast_position:.4f} km（駅まで "
          f"{(solver.station_position - result.coast_position)*1000:.0f} m）")
    print(f"  制動開始位置    : {result.brake_position:.4f} km（駅まで "
          f"{(solver.station_position - result.brake_position)*1000:.0f} m）／ "
          f"制動開始速度 {result.brake_speed:.2f} km/h")
    print(f"  力行エネルギー  : {result.energy_kwh:.3f} kWh "
          f"({result.energy_kwh / solver.distance_km:.3f} kWh/km, 回生なし・車輪周)")
    print(f"  ノッチ切替回数  : {result.notch_changes} 回")
    print(f"  力行/惰行/制動  : {result.power_time:.1f} / {result.coast_time:.1f} / {result.brake_time:.1f} 秒")

    compare = None
    if args.compare:
        compare = load_comparison(args.compare, solver, jp)
        print("── DQN走行ログとの比較 ───────────────────────")
        print(f"  対象ログ        : {args.compare}")
        print(f"  到着時刻        : {compare['arrival_time']:.2f} 秒  "
              f"(標準運転曲線 {result.time:.2f} 秒)")
        print(f"  停止位置誤差    : {compare['stop_error_m']:+.3f} m  "
              f"(標準運転曲線 {result.stop_error_m:+.3f} m)")
        print(f"  最高速度        : {compare['max_speed']:.2f} km/h  "
              f"(標準運転曲線 {result.max_speed:.2f} km/h)")
        ratio = compare["energy_kwh"] / result.energy_kwh if result.energy_kwh > 0 else float("nan")
        print(f"  力行エネルギー  : {compare['energy_kwh']:.3f} kWh  "
              f"(標準運転曲線 {result.energy_kwh:.3f} kWh, 比 {ratio:.2f} 倍)")
        print(f"  ノッチ切替回数  : {compare['notch_changes']} 回  "
              f"(標準運転曲線 {result.notch_changes} 回)")

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    name = args.name or f"standard_curve_{args.departure_index}"
    csv_path = os.path.join(out_dir, f"{name}.csv")
    meta_path = os.path.join(out_dir, f"{name}_meta.json")
    png_path = os.path.join(out_dir, f"{name}.png")
    detail_png_path = os.path.join(out_dir, f"{name}_detail.png")

    desc = (f"数理モデル標準運転曲線（{'力行→惰行→制動' if args.strategy == 'pcb' else '省エネ最適'}・"
            f"定時{solver.target_time:.0f}秒）")
    write_log_csv(solver, result, csv_path)
    write_meta_json(solver, result, v_hold, meta_path, desc)
    plot_dqn_style(solver, result, png_path)
    plot_detail(solver, result, v_hold, detail_png_path, jp, compare)
    if args.sr_out:
        write_sr_csv(solver, result, args.sr_out)

    print("── 出力 ───────────────────────────────────")
    print(f"  運転曲線PNG     : {png_path}（apex2.pyの出力と同一書式）")
    print(f"  勾配つき運転曲線: {detail_png_path}（指標つき）")
    print(f"  走行ログCSV     : {csv_path}")
    print(f"  （モニター用メタ: {meta_path}）")
    if args.sr_out:
        print(f"  標準走行曲線    : {args.sr_out}")
    print(f"  モニターで比較: python drive_monitor.py {os.path.relpath(csv_path, BASE_DIR)}"
          + (f" {args.compare}" if args.compare else ""))
    if not jp:
        print("  ※日本語フォントが見つからないため、図のラベルは英語で描画しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
