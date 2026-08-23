# -*- coding: utf-8 -*-
"""
複数駅間版の目標速度算出（設計: docs_複数駅間最適化_計画.md §7）

既存 `required_speed.py` は**一切変更しない**（計画書 §9.1）。

既存との決定的な違い —— **勾配とATC現示を「現在地点の値1つ」で駅まで一定と仮定しない**。
羽前成田線は制限70km/h固定・勾配−6.7〜+12.5‰と狭く、スカラー近似で足りていた。
東西線は現示45〜75km/h・勾配−16.3〜+29.7‰で、惰行減速度が **+0.29 〜 −1.34 km/h/s と符号ごと反転**する。
現在地点の勾配で駅まで外挿すると惰行減速度を最大3.1倍過小評価する（§7.1）。
本モジュールはすべて**前方プロファイルを積分**して算出する。

提供する目標速度（§7.3 の三層構造）

    v_ceiling(x)  = min( ATC現示の先読みパターン , 先行列車によるCBTC現示 )   ← 超えたら不可
    v_target(x)   = モード別
        normal          → v_std(x)                  標準運転曲線（定時＋省エネの最適解）
        delay_recovery  → v_ceiling(x) の直下の維持帯（勾配・乗り心地T_minを考慮）
        anti_mid_stop   → target_speed_no_stop(x)   先行クリア時間から算出（勾配・現示を考慮）
        spacing         → target_speed_spacing(x)   前後の車間を均す速度
    維持帯 band_lower..band_upper と使用ノッチ対  ← 乗り心地（T_min秒）から幅を決める
"""
import os
import csv as csvmod
from bisect import bisect_right

import numpy as np

import line_config as LC
from actions_multi import FROM_CODE
from train_multi import Vehicle, get_track

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STD_DIR = os.path.join(BASE_DIR, "standard_curve_multi")


class SpeedProfile:
    """1駅間ぶんの目標速度算出。区間に入るとき（reset / 区間切替）に1度だけ構築する。

    位置 x はすべて**内部座標[km]**（進行方向に増加・起点駅が0）。
    """

    def __init__(self, line_name="tozai", section=0, atc_mode=None, t_min=5.0,
                 dx_km=0.0005, load_std_curve=True):
        self.track = get_track(line_name)
        self.vehicle = Vehicle(self.track.cfg["vehicle"])
        self.line_name, self.section = line_name, section
        st = self.track.stations
        self.dep, self.arr = st[section], st[section + 1]
        self.x0, self.x1 = self.dep["position"], self.arr["position"]
        self.target_time = float(self.dep["running_time"])
        self.atc_mode = atc_mode or self.track.cfg.get("atc_mode", "anticipatory")
        self.t_min = float(t_min)
        self.train_len_km = self.vehicle.train_length_km
        self._dx = dx_km
        self._build_atc_ceiling()
        self._build_station_brake_curve()
        self._build_grid()
        # 区間内の最大ATC現示。目標速度の探索上限に使う。
        # 局所現示（東陽町発車直後の45km/h・13.3m）を上限にすると、
        # そこで必要巡航速度が45に頭打ちになり「必要速度＝現示」となって
        # 遅延回復モードが誤発動する（実測で確認）。
        self._section_cap = float(max(self._grid_ceil))
        self._v_std = None
        if load_std_curve:
            self._load_std_curve()

    # =====================================================================
    # 1. ATC現示の先読みパターン（§7.4）
    # =====================================================================
    def _build_atc_ceiling(self):
        """現示低下点から後ろ向きに制動曲線を逆積分し、各地点の上限速度を作る。

        (a)予見型: 低下点に到達した時点で新現示以下になっているよう手前から下げる。
        (b)追従型: 低下点で制動を開始し、制動曲線に沿って下がる途中を許容する。
        """
        x_end = self.x1 + 0.05
        n = int((x_end - self.x0) / self._dx) + 2
        xs = self.x0 + self._dx * np.arange(n)
        c = np.array([self.track.atc_limit(x) for x in xs], dtype=float)
        seg_m = self._dx * 1000.0
        if self.atc_mode == "anticipatory":
            for i in range(n - 2, -1, -1):
                v = c[i + 1]
                steps = max(1, int(seg_m / 0.5))
                for _ in range(steps):
                    v += self._brake_decel(v, xs[i]) * ((seg_m / steps) / max(v / 3.6, 0.1))
                c[i] = min(c[i], v)
        else:  # pattern（追従型）
            for k in range(1, len(self.track.limit_starts)):
                s, L = self.track.limit_starts[k], self.track.limit_vals[k]
                Lp = self.track.limit_vals[k - 1]
                if Lp <= L or s < self.x0 or s > x_end:
                    continue
                i0 = int((s - self.x0) / self._dx)
                v = Lp
                for i in range(max(i0, 0), n):
                    if v <= L:
                        break
                    c[i] = max(c[i], min(v, Lp))
                    v -= self._brake_decel(v, xs[i]) * (seg_m / max(v / 3.6, 0.1))
        self._atc_ceiling = c

    def _build_grid(self, step_km=0.005):
        """保持時間の積分に使う位置グリッド（5m刻み）。毎回 bisect を叩かないための事前計算。"""
        n = int((self.x1 - self.x0) / step_km) + 2
        self._grid_step = step_km
        self._grid_x = self.x0 + step_km * np.arange(n)
        self._grid_ceil = np.array([self.atc_pattern(x) for x in self._grid_x])
        self._grid_grade = np.array([self.track.grade(x) for x in self._grid_x])

    def _hold_time(self, V, x_from, x_to):
        """速度 V を保って x_from→x_to を走る所要時間[s]。ATC現示で頭打ちにする。"""
        if x_to <= x_from:
            return 0.0
        if V <= 0.5:
            # 【重要】0km/hで保持しても距離は縮まらない。ここで0を返すと
            # 「停止したまま到着できる」ことになり、発車時（速度0）の
            # required_speed / target_speed_no_stop が 0 になってしまう。
            return float("inf")
        i0 = max(0, int((x_from - self.x0) / self._grid_step))
        i1 = min(len(self._grid_x) - 1, int(np.ceil((x_to - self.x0) / self._grid_step)))
        if i1 <= i0:
            return (x_to - x_from) * 1000.0 / (min(V, self._grid_ceil[i0]) / 3.6)
        vv = np.minimum(V, self._grid_ceil[i0:i1])
        vv = np.maximum(vv, 1.0)
        return float(np.sum(self._grid_step * 1000.0 / (vv / 3.6)))

    def _power_trajectory(self, v0, x, sub_dt=0.5, v_cap=None):
        """現在地点から全力行したときの (時刻, 位置, 速度) 軌跡。

        目標速度の二分探索では候補Vごとに加速をやり直す必要はない。
        **1回だけ積分して軌跡を持ち、候補Vごとに「Vに達する点」を引く**ことで
        二分探索の内側から積分を追い出す（高速化）。
        """
        cap = v_cap if v_cap is not None else self.atc_pattern(x)
        ts, xs, vs = [0.0], [x], [max(0.0, v0)]
        t, xx, vv = 0.0, x, max(0.0, v0)
        while t < 300.0 and xx < self.x1:
            c = min(cap, self.atc_pattern(xx))
            if vv >= c - 0.05:
                break
            a = self.notch_accel("P1", vv, xx)
            if a <= 1e-6:
                break
            nv = vv + a * sub_dt
            xx += (vv / 3600.0) * sub_dt + (a / 3600.0) * (sub_dt ** 2)
            vv = nv
            t += sub_dt
            ts.append(t); xs.append(xx); vs.append(vv)
        return np.asarray(ts), np.asarray(xs), np.asarray(vs)

    def _time_to_station(self, traj, V):
        """軌跡 traj で V まで加速し、V を保って走り、制動して停止するまでの総時間[s]。
        到達不能なら None。"""
        ts, xs, vs = traj
        if V <= 0.5:
            return None                      # 停止したままでは到着できない
        if V <= vs[0]:
            t_acc, x_acc = 0.0, xs[0]
        elif V > vs[-1] + 1e-6:
            return None                      # その速度まで加速できない
        else:
            j = int(np.searchsorted(vs, V))
            j = min(j, len(vs) - 1)
            t_acc, x_acc = float(ts[j]), float(xs[j])
        x_b = self.brake_start_x(V)
        if x_b < x_acc:                      # 加速中に制動開始点を過ぎる＝Vが高すぎる
            return None
        return t_acc + self._hold_time(V, x_acc, x_b) + self.brake_time(V)

    def _brake_decel(self, v, x):
        """制動ノッチB1使用時の実効減速度[km/h/s]（正の値）。走行抵抗・勾配・曲線を含む。"""
        a = self.vehicle.accel(FROM_CODE["B1"], max(v, 0.0),
                               self.track.grade(x), self.track.curve(x))
        return max(-a, 0.05)

    def atc_pattern(self, x):
        """位置 x での線路条件によるATC現示（先読み込み）[km/h]"""
        i = int((x - self.x0) / self._dx)
        i = max(0, min(i, len(self._atc_ceiling) - 1))
        return float(self._atc_ceiling[i])

    def atc_now(self, x):
        """位置 x での**生の**ATC現示（先読みなし）[km/h]"""
        return self.track.atc_limit(x)

    def next_limit_drop(self, x):
        """次の現示低下点までの距離[m]と低下後の現示[km/h]。無ければ (None, None)"""
        d, v = self.track.next_limit_drop(x, ahead_km=1.0)
        return (None, None) if d is None else (d * 1000.0, v)

    # =====================================================================
    # 2. 制動（プロファイル対応）
    # =====================================================================
    def _build_station_brake_curve(self, v_max=95.0, grid_km=1e-5):
        """到着駅にちょうど停止する制動曲線を駅から逆向きに積分する。

        速度だけでなく**残り制動時間**も持たせる。目標速度の算出で
        「Vから制動して停止するまでの時間」をテーブル参照で得るため（高速化）。
        """
        xs, vs, ts = [self.x1], [0.0], [0.0]
        x, v, t = self.x1, 0.0, 0.0
        while v < v_max and x > self.x0 - 0.5:
            a = self.vehicle.accel(FROM_CODE["B1"], v,
                                   self.track.grade(x), self.track.curve(x))
            if a >= 0.0:
                a = -0.0001
            v_prev = v - a * LC.SUB_DT
            x_prev = x - (v_prev / 3600.0) * LC.SUB_DT - (a / 3600.0) * (LC.SUB_DT ** 2)
            x, v, t = x_prev, v_prev, t + LC.SUB_DT
            xs.append(x); vs.append(v); ts.append(t)
        xs = np.asarray(xs[::-1]); vs = np.asarray(vs[::-1]); ts = np.asarray(ts[::-1])
        self._bx0, self._bdx = float(xs[0]), grid_km
        n = int((self.x1 - self._bx0) / grid_km) + 2
        grid = self._bx0 + grid_km * np.arange(n)
        self._btab = np.interp(grid, xs, vs)
        self._bttab = np.interp(grid, xs, ts)          # その位置から停止するまでの時間[s]
        # V → 制動開始位置 の逆引き（_btab は位置に対して単調減少）
        self._brake_v_desc = self._btab[::-1]
        self._brake_x_desc = grid[::-1]

    def station_brake_speed(self, x):
        """位置 x で「駅にちょうど停止するための速度」[km/h]。これを超えたら即制動が必要。"""
        i = int((x - self._bx0) / self._bdx)
        if i < 0:
            return float("inf")
        if i >= len(self._btab):
            return 0.0
        return float(self._btab[i])

    def brake_start_x(self, v):
        """速度 v で制動を開始すれば駅にちょうど停止できる位置[km]"""
        if v <= 0.0:
            return self.x1
        i = int(np.searchsorted(self._brake_v_desc, v))
        i = max(0, min(i, len(self._brake_x_desc) - 1))
        return float(self._brake_x_desc[i])

    def brake_time(self, v):
        """速度 v から制動して停止するまでの時間[s]"""
        if v <= 0.0:
            return 0.0
        i = int(np.searchsorted(self._brake_v_desc, v))
        i = max(0, min(i, len(self._bttab) - 1))
        return float(self._bttab[::-1][i])

    def stop_distance(self, v, x, dv=0.25):
        """位置 x で速度 v から制動して停止するまでの距離[m]（**前方プロファイルを積分**）。

        既存 `brake_stop_distance_m` は勾配スカラー1つを一定と仮定するが、
        東西線では同じ70km/hでも勾配により 181.8〜321.0 m と1.77倍の開きがある（§6.5）。
        """
        vv, d = max(0.0, v), 0.0
        xx = x
        while vv > 1e-9:
            step = min(dv, vv)
            vm = vv - step / 2.0
            dec = self._brake_decel(vm, xx)
            dt = step / dec
            seg = (vm / 3.6) * dt
            d += seg
            xx += seg / 1000.0
            vv -= step
        return d

    def max_speed_to_stop_at(self, x, x_stop, ds_m=1.0):
        """位置 x_stop までに停止できる最大速度[km/h]（x_stop から後ろ向きに制動曲線を積分）"""
        if x_stop <= x:
            return 0.0
        v, xx = 0.0, x_stop
        step_km = ds_m / 1000.0
        guard = 0
        while xx > x and guard < 200000:
            dec = self._brake_decel(v, xx)
            # v dv = a ds  →  v_prev = sqrt(v² + 2·a·ds)   （a, v はm/s系へ換算）
            v_ms = v / 3.6
            v_ms = (v_ms ** 2 + 2.0 * (dec / 3.6) * ds_m) ** 0.5
            v = v_ms * 3.6
            xx -= step_km
            guard += 1
            if v > 200.0:
                break
        return v

    # =====================================================================
    # 3. 先行列車によるCBTC現示（§6.6）
    # =====================================================================
    def cbtc_only(self, x, fw_head_x=None):
        """**先行列車のみ**によるATC現示[km/h]（線路条件は含まない）。

        停止限界 = 先行の**最後尾**から50m手前に自列車の**先頭**が来る点
                 = 先行の先頭位置 − 列車長 − 50m
        先行がいなければ無限大扱い。
        """
        if fw_head_x is None:
            return float("inf")
        x_stop = fw_head_x - self.train_len_km - LC.CBTC_STOP_LIMIT_KM
        if x_stop <= x:
            return 0.0
        return self.max_speed_to_stop_at(x, x_stop)

    def signal_speed(self, x, fw_head_x=None):
        """**実際のATC現示**[km/h] = min(生の線路条件の現示, 先行列車によるCBTC現示)。

        【重要】これは「超えたら違反」というハードな上限であり、即reward0.0の判定に使う。
        先読み（予見型パターン）は含めない。
        """
        return min(self.atc_now(x), self.cbtc_only(x, fw_head_x))

    def v_ceiling(self, x, fw_head_x=None):
        """運転の**目標上限**[km/h] = min(ATC現示の先読みパターン, 先行列車によるCBTC現示)。

        【重要】`signal_speed` とは別物である。先読みパターンは現示低下点の手前で
        あらかじめ下がるため、**v_ceiling < signal_speed** となる局面がある。
        これは「この先で現示が下がるので今のうちに減速せよ」という運転上の目標であって、
        超えたこと自体はATC違反ではない（＝即reward0.0にしてはならない）。
        """
        return min(self.atc_pattern(x), self.cbtc_only(x, fw_head_x))

    def cbtc_speed(self, x, fw_head_x=None):
        """後方互換のための別名（= v_ceiling）"""
        return self.v_ceiling(x, fw_head_x)

    # =====================================================================
    # 4. ノッチの物理量と維持帯（§7.4.3(5)）
    # =====================================================================
    def notch_accel(self, code, v, x):
        return self.vehicle.accel(FROM_CODE[code], v, self.track.grade(x), self.track.curve(x))

    def coast_accel(self, v, x):
        """惰行時の加速度[km/h/s]。**正なら惰行では減速できない**（急な下り勾配）"""
        return self.notch_accel("C", v, x)

    def power_accel(self, v, x):
        return self.notch_accel("P1", v, x)

    def hold_notches(self, v, x, look_ahead_m=50.0):
        """速度を保つためのノッチ (up, dn, hold, a_hold)。

        勾配は**前方 look_ahead_m の距離加重平均**で評価する。瞬時値を使うと
        勾配急変点で帯が飛んでノッチが振動する（§14.4(実装の落とし穴)）。
        """
        g = self.mean_grade(x, look_ahead_m)
        c = self.track.curve(x)
        acc = {k: self.vehicle.accel(FROM_CODE[k], v, g, c) for k in LC.NOTCH_ORDER}
        hold = min(LC.NOTCH_ORDER, key=lambda k: abs(acc[k]))
        pos = [k for k in LC.NOTCH_ORDER if acc[k] > 1e-6]
        neg = [k for k in LC.NOTCH_ORDER if acc[k] < -1e-6]
        up = pos[-1] if pos else "P1"       # NOTCH_ORDER は強い順なので末尾が最弱
        dn = neg[0] if neg else "B1"
        return up, dn, hold, acc[hold]

    def mean_grade(self, x, look_ahead_m=50.0):
        """前方 look_ahead_m の距離加重平均勾配[‰]"""
        n = max(1, int(look_ahead_m / 5.0))
        return float(np.mean([self.track.grade(x + i * 0.005) for i in range(n)]))

    def band(self, v, x, v_top_limit, time_step=1.0):
        """維持帯 (band_upper, band_lower, (up, dn), 定速ノッチ) を返す。

        band_upper = v_top_limit − M_up      M_up: 1ステップ力行しても超過しない余裕
        band_lower = band_upper − W          W = T_min × max(|a_up|, |a_dn|)

        惰行で減速できない下り勾配（a_dn が惰行でない）でも、up/dn は
        `hold_notches` が勾配ごとに選ぶので同じ式で扱える。
        """
        g = self.mean_grade(x)
        up, dn, hold, a_hold = self.hold_notches(v, x)
        a_up = abs(self.vehicle.accel(FROM_CODE[up], v, g))
        a_dn = abs(self.vehicle.accel(FROM_CODE[dn], v, g))
        m_up = max(2.0, self.power_accel(v, x) * time_step * 1.5)
        top = v_top_limit - m_up
        if abs(a_hold) < 0.05:
            w = 0.6                       # 真の定速ノッチがある（±35‰のB2/P2）
        else:
            w = self.t_min * max(a_up, a_dn)
        return top, max(0.0, top - w), (up, dn), hold

    def notch_jump(self, code_a, code_b, v, x):
        """ノッチ変更による加速度のジャンプ幅[km/h/s]（乗り心地。§4.7.4）"""
        if code_a is None or code_b is None:
            return 0.0
        return abs(self.notch_accel(code_b, v, x) - self.notch_accel(code_a, v, x))

    # =====================================================================
    # 5. 走行シミュレーション（プロファイル対応・§7.2）
    # =====================================================================
    def simulate(self, v0, x_start, x_end, cruise_v=None, sub_dt=0.25, max_time=600.0,
                 respect_ceiling=True):
        """x_start から x_end まで
            「cruise_v まで力行 → 到達後は保持（勾配に応じたノッチ）→ 制動曲線に当たったら制動」
        を**実際の勾配・現示プロファイルを積分して**再現する。

        cruise_v=None なら「現在速度のまま保持」＝加速しない。
        戻り値 (所要時間[s], 到達距離[m], 終端速度[km/h])
        """
        x, v, t = x_start, max(0.0, v0), 0.0
        target = cruise_v if cruise_v is not None else v0
        while t < max_time:
            if x >= x_end:
                break
            ceil_v = self.atc_pattern(x) if respect_ceiling else 1e9
            tgt = min(target, ceil_v)
            # 駅にちょうど止まる制動曲線に当たったら制動
            if v >= self.station_brake_speed(x) - 1e-9:
                code = "B1"
            elif v < tgt - 0.3:
                code = "P1"
            else:
                up, dn, hold, a_hold = self.hold_notches(v, x)
                if abs(a_hold) < 0.05:
                    code = hold
                elif v > tgt:
                    code = dn
                else:
                    code = up
            a = self.notch_accel(code, v, x)
            nv = max(0.0, v + a * sub_dt)
            x += (v / 3600.0) * sub_dt + (a / 3600.0) * (sub_dt ** 2)
            v = nv
            t += sub_dt
            if v <= 1e-6 and code != "B1":
                break
        return t, (x - x_start) * 1000.0, v

    # =====================================================================
    # 6. 目標速度
    # =====================================================================
    def required_speed(self, v, x, time_left, iters=16):
        """定時運行に必要な巡航速度[km/h]（§7.3）。

        「この速度まで力行し、以降は保持して走れば定刻に着く」速度を二分探索する。
        現在速度のまま保持して間に合うなら現在速度を返す（＝これ以上の加速は不要）。
        軌跡は1度だけ積分し、候補ごとにテーブル参照で評価する。
        """
        cap = self.atc_pattern(x)
        if x >= self.x1 or cap <= 0.0:
            return 0.0
        if time_left <= 0.0:
            return cap                                # 既に定刻超過 → 現示まで加速を要求
        traj = self._power_trajectory(v, x)
        t_now = self._time_to_station(traj, min(v, cap))
        if t_now is not None and t_now <= time_left:
            return min(v, cap)                        # 現在速度のままで間に合う＝加速不要（現示で頭打ち）
        lo, hi = max(v, 1.0), cap
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            t_s = self._time_to_station(traj, mid)
            if t_s is None or t_s > time_left:
                lo = mid
            else:
                hi = mid
        return min(hi, cap)

    def target_speed_no_stop(self, v, x, time_left, forward_clear_time,
                             safety_margin=15.0, iters=16):
        """機外停車を避けつつ進める上限速度[km/h]（§7.3）。

        先行列車が自列車の次駅を発車するまで（forward_clear_time 秒）は駅に着けない。
        「その速度を保って走ると、実効所要時間ちょうどに次駅へ到着する速度」を求める。
        先行が長く塞ぐほど実効所要時間が伸び、上限速度は下がる。

        **現在速度に依存しない状況ベースの値**として算出する（既存 required_speed.py と同じ思想）。
        現在速度に依存させると、加速するほど上限も上がって過剰加速を検知できない。
        """
        if forward_clear_time <= 0.0:
            return self.required_speed(v, x, time_left)
        eff = max(time_left, forward_clear_time + safety_margin)
        if eff <= time_left + 1e-6:
            return self.required_speed(v, x, time_left)
        return self._speed_for_time(x, eff, iters)

    def schedule_speed(self, x, time_left, iters=16):
        """定刻に着くために**必要な巡航速度**[km/h]（現在速度に依存しない状況ベースの値）。

        `required_speed` は「現在速度のまま間に合うなら現在速度を返す」仕様のため、
        過速している列車では現在速度をそのまま返してしまい、
        **モード判定（遅延回復か否か）に使うと過速を遅延と取り違える**。
        モード判定にはこちらを使うこと。

        **駅手前の制動域では None を返す。** そこでは所要時間が制動曲線で決まり、
        巡航速度の選択では変えられない。「必要巡航速度」という概念が成立しないうえ、
        二分探索が上限に張り付いて遅延回復モードを誤発動させる（実測で確認）。
        加速による遅延回復が不可能な局面なので、モード判定から除外するのが正しい。
        """
        # 次駅減速フェーズ（駅手前400m以内）は対象外。ここでは所要時間が制動曲線で決まり、
        # 巡航速度の選択では変えられない。加速による遅延回復も不可能なので、
        # 「必要巡航速度」を返すとモード判定を誤らせるだけである。
        if (self.x1 - x) * 1000.0 <= 400.0:
            return None
        if time_left <= 0.0:
            return self._section_cap
        return self._speed_for_time(x, time_left, iters)

    def _speed_for_time(self, x, eff_time, iters=16):
        """「その速度を保って走ると eff_time ちょうどに次駅へ着く」速度[km/h]。
        現在速度に依存しない状況ベースの値。

        探索上限は**区間内の最大現示**を使う（局所現示ではない）。
        走行中の現示制約は `_hold_time` が位置ごとに `min(V, 現示)` で掛けるので、
        上限を上げても物理的な整合は保たれる。
        """
        cap = self._section_cap
        if x >= self.x1 or cap <= 0.0:
            return 0.0
        lo, hi = 1.0, cap
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            x_b = self.brake_start_x(mid)
            t_s = self._hold_time(mid, x, x_b) + self.brake_time(mid) if x_b >= x else None
            if t_s is None or t_s > eff_time:
                lo = mid                    # 遅着 → もっと速く
            else:
                hi = mid                    # 早着 → もっと遅く
        return min(hi, cap)

    def target_speed_spacing(self, v, x, time_left, fw_head_x, bw_head_x, iters=16):
        """前後の車間を均す目標速度[km/h]。

        前が近い（詰まっている）ほど遅く、後ろが近い（詰まらせている）ほど速くする。
        """
        if fw_head_x is None or bw_head_x is None:
            return self.required_speed(v, x, time_left)
        d_fw = max(0.0, (fw_head_x - self.train_len_km - x) * 1000.0)
        d_bw = max(0.0, (x - self.train_len_km - bw_head_x) * 1000.0)
        if d_fw + d_bw < 1.0:
            return self.required_speed(v, x, time_left)
        # 車間の偏り（−1: 前に詰まる 〜 +1: 後ろが近い）を所要時間の補正に写す
        bias = (d_fw - d_bw) / (d_fw + d_bw)
        eff = max(1.0, time_left * (1.0 - 0.25 * bias))
        return self._speed_for_time(x, eff, iters)

    def coast_reachable(self, v, x):
        """今から惰行を続けたとき、制動開始点まで到達できるか（§7.5）。

        「勾配で何km/h落ちたら加速」という閾値は作らない。前方プロファイルを積分して
        「惰行のままでは届かない／定刻に間に合わない」と判定した時点が再加速のタイミングになる。
        戻り値 (到達可否, 到達時の余裕[m], 惰行での所要時間[s])
        """
        xx, vv, t = x, v, 0.0
        dt = 0.25
        while t < 600.0:
            if vv >= self.station_brake_speed(xx) - 1e-9:
                return True, (xx - x) * 1000.0, t
            if xx >= self.x1:
                return True, (xx - x) * 1000.0, t
            a = self.notch_accel("C", vv, xx)
            nv = max(0.0, vv + a * dt)
            xx += (vv / 3600.0) * dt + (a / 3600.0) * (dt ** 2)
            vv = nv
            t += dt
            if vv <= 0.5:
                return False, (xx - x) * 1000.0, t     # 失速＝駅間停車
        return False, (xx - x) * 1000.0, t

    # =====================================================================
    # 7. 標準運転曲線 v_std(x)（§7.6）
    # =====================================================================
    def _load_std_curve(self):
        base = f"{self.line_name}_{self.section}_{self.dep['name']}_{self.arr['name']}.csv"
        path = os.path.join(STD_DIR, base)
        if not os.path.exists(path):
            return
        xs, vs = [], []
        with open(path, encoding="utf-8-sig") as f:
            for row in csvmod.DictReader(f):
                xs.append(float(row["position_km"])); vs.append(float(row["speed"]))
        if len(xs) < 2:
            return
        order = np.argsort(xs)
        self._std_x = np.asarray(xs)[order]
        self._std_v = np.asarray(vs)[order]
        self._v_std = True

    def v_std(self, x):
        """標準運転曲線の位置 x での速度[km/h]。曲線が無ければ None。"""
        if not self._v_std:
            return None
        return float(np.interp(x, self._std_x, self._std_v))

    # =====================================================================
    # 8. モードをまとめた目標速度
    # =====================================================================
    def targets(self, v, x, time_left, mode="normal", fw_head_x=None, bw_head_x=None,
                forward_clear_time=0.0, time_step=1.0):
        """プロンプト・観測へ渡す目標速度一式を返す（`prompt_multi.FEATURE_KEYS` に対応）"""
        ceil = self.v_ceiling(x, fw_head_x)          # 先読みを含む運転目標
        req = self.required_speed(v, x, time_left)
        nostop = self.target_speed_no_stop(v, x, time_left, forward_clear_time)
        spacing = (self.target_speed_spacing(v, x, time_left, fw_head_x, bw_head_x)
                   if (fw_head_x is not None and bw_head_x is not None) else None)
        std = self.v_std(x)
        if mode == "normal" and std is not None:
            tgt = min(std, ceil)
        elif mode == "delay_recovery":
            tgt = ceil
        elif mode == "anti_mid_stop":
            tgt = min(nostop, ceil)
        elif mode == "spacing" and spacing is not None:
            tgt = min(spacing, ceil)
        else:
            tgt = min(req, ceil)
        top, bot, pair, hold = self.band(v, x, ceil, time_step)
        drop_d, drop_v = self.next_limit_drop(x)
        return dict(
            v_ceiling=ceil, v_target=tgt,
            band_upper=top, band_lower=bot,
            band_notch_pair=f"{LC.NOTCH_LABEL_JA[pair[0]]} ↔ {LC.NOTCH_LABEL_JA[pair[1]]}",
            hold_notch=LC.NOTCH_LABEL_JA[hold],
            required_speed=req, schedule_speed=self.schedule_speed(x, time_left),
            target_speed_no_stop=nostop, target_speed_spacing=spacing,
            v_std=std, v_std_deviation=(v - std) if std is not None else None,
            atc_now=self.atc_now(x), signal_speed=self.signal_speed(x, fw_head_x),
            section_cap=self._section_cap,
            coast_accel=self.coast_accel(v, x), power_accel=self.power_accel(v, x),
            req_stop_dist=self.stop_distance(v, x),
            limit_drop_ahead=(f"{drop_d:.0f}m先でATC現示が{self.atc_now(x):.0f}km/hから{drop_v:.0f}km/hに低下する"
                              if drop_d is not None else "前方に現示の低下なし"),
        )


# =========================================================================
# 単体確認
# =========================================================================
def _demo():
    print("=" * 96)
    print("目標速度算出の単体確認（東京メトロ東西線・15000系10両）")
    print("=" * 96)
    for sec in (0, 1):
        sp = SpeedProfile("tozai", sec)
        L = (sp.x1 - sp.x0) * 1000.0
        print(f"\n### [{sec}] {sp.dep['name']} → {sp.arr['name']}  {L:.1f} m / 標準 {sp.target_time:.0f} s")

        print("\n-- 制動距離が勾配で大きく変わること（既存はスカラー1つで近似していた） --")
        print(f"{'位置':>8}{'勾配':>9}{'70km/hからの制動距離':>22}")
        for d in (100, 300, 500, 700, 900):
            if d > L:
                continue
            x = sp.x0 + d / 1000.0
            print(f"{d:6.0f}m{sp.track.grade(x):+8.1f}‰{sp.stop_distance(70.0, x):18.1f} m")

        print("\n-- 惰行時の加速度と維持帯（遅延回復モードの基準） --")
        print(f"{'位置':>8}{'勾配':>9}{'惰行a':>9}{'天井':>8}{'維持帯':>16}{'使用ノッチ対':>22}")
        for d in (100, 300, 500, 700, 900):
            if d > L:
                continue
            x = sp.x0 + d / 1000.0
            ceil = sp.v_ceiling(x)
            top, bot, pair, hold = sp.band(60.0, x, ceil)
            print(f"{d:6.0f}m{sp.track.grade(x):+8.1f}‰{sp.coast_accel(60.0, x):+8.3f}{ceil:8.1f}"
                  f"{bot:7.1f}〜{top:5.1f}"
                  f"{LC.NOTCH_LABEL_JA[pair[0]] + ' ↔ ' + LC.NOTCH_LABEL_JA[pair[1]]:>20}")


if __name__ == "__main__":
    _demo()
