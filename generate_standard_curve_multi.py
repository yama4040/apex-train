# -*- coding: utf-8 -*-
"""
複数駅間版の標準運転曲線ジェネレータ（設計: docs_複数駅間最適化_計画.md §6.4・§7）

既存 generate_standard_curve.py（山形28t・3ノッチ・単一区間）は**一切変更しない**。
本スクリプトは以下を新規に備える。

  * 東京メトロ15000系（10両）の物理（train_multi.Vehicle）
  * **5ノッチ**（P1/P2/C/B2/B1）。定速保持を中間ノッチで行える（§4.7）
  * **ATC現示の先読みパターン**（予見型 v_ceiling）に沿った運転（§7.4）
  * 進行方向に増加する内部座標（下りB線に対応。§6.2）

運転パターン: 力行 → 定速保持 → 惰行 → 制動
  設計変数は 定速保持速度 V_hold と 惰行開始位置 x_coast。
  x_coast は「到着時刻 = 標準運転時間」となるよう二分探索し、
  V_hold は力行仕事が最小となるものをグリッド探索する。

使い方:
    python generate_standard_curve_multi.py                   # 全駅間
    python generate_standard_curve_multi.py --section 0       # 東陽町→木場のみ
    python generate_standard_curve_multi.py --line yamagata --section 11
"""
import os
import sys
import json
import math
import argparse
import csv as csvmod
from bisect import bisect_right

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import line_config as LC
from actions_multi import ActionsMulti, FROM_CODE, CODE
from train_multi import Vehicle, get_track

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "standard_curve_multi")
SUB_DT = LC.SUB_DT

_JP_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
]


def setup_japanese_font():
    for p in _JP_FONTS:
        if not os.path.exists(p):
            continue
        try:
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return True
    return False


# =============================================================================
# ソルバ
# =============================================================================
class StandardCurveMulti:
    def __init__(self, line_name="tozai", section=0, target_time=None,
                 hold_band=None, atc_mode=None, t_min=5.0):
        self.track = get_track(line_name)
        self.vehicle = Vehicle(self.track.cfg["vehicle"])
        self.line_name = line_name
        self.section = section
        st = self.track.stations
        if section + 1 >= len(st):
            raise ValueError(f"section={section} は範囲外（駅は{len(st)}）")
        self.dep, self.arr = st[section], st[section + 1]
        self.x0, self.x1 = self.dep["position"], self.arr["position"]
        self.distance_km = self.x1 - self.x0
        self.target_time = float(self.dep["running_time"]) if target_time is None else float(target_time)
        self.atc_mode = atc_mode or self.track.cfg.get("atc_mode", "anticipatory")
        # 定速保持の帯（§7.4.3(5)）。固定幅にすると、勾配によって相の継続時間が
        # 数百ms〜数十秒まで変動し、ノッチ切替が多発して乗り心地とエネルギーを損なう。
        # **両相が t_min 秒以上continueする最小幅**を、その地点の勾配・速度から都度算出する。
        self.t_min = float(t_min)
        self.hold_band = None if hold_band is None else float(hold_band)   # 明示指定時は固定幅
        self._build_ceiling()
        self._build_brake_curve()
        # V_hold グリッドの上限。区間内の**最大**現示を使う。
        # 最小にすると、出発直後の短い低現示（東陽町の45km/h・13.3m）に引きずられて
        # 区間全体の保持速度が不当に抑えられてしまう。
        # 局所的な現示の制約は simulate() 内で min(v_hold, ceil_v) として別途かかる。
        self.speed_cap = max(self.ceiling(x) for x in
                             np.arange(self.x0, self.x1, 0.005))

    # ------------------------------------------------------------ ATC現示天井
    def _build_ceiling(self, dx=0.0005):
        """(a)予見型: 現示低下点から後ろ向きに制動曲線を逆積分し、各地点の上限速度を作る（§7.4）"""
        x_end = self.x1 + 0.05
        n = int((x_end - self.x0) / dx) + 2
        xs = self.x0 + dx * np.arange(n)
        c = np.array([self.track.atc_limit(x) for x in xs])
        if self.atc_mode == "anticipatory":
            for i in range(n - 2, -1, -1):
                v = c[i + 1]
                # 位置 xs[i] から dx 進んで v になる速度（制動の逆積分）
                steps = max(1, int(dx * 1000.0 / 0.5))
                for _ in range(steps):
                    dec = (-self.vehicle.DECELERATE
                           + (self.vehicle.travel_resistance(v)
                              + self.track.grade(xs[i]) + self.track.curve(xs[i]))
                           / LC.FACTOR_OF_INERTIA)
                    dec = max(dec, 0.05)
                    v += dec * ((dx * 1000.0 / steps) / max(v / 3.6, 0.1))
                c[i] = min(c[i], v)
        self._ceil_x0, self._ceil_dx, self._ceil = float(xs[0]), dx, c

    def ceiling(self, x):
        i = int((x - self._ceil_x0) / self._ceil_dx)
        i = max(0, min(i, len(self._ceil) - 1))
        return float(self._ceil[i])

    # -------------------------------------------------------------- 制動曲線
    def _build_brake_curve(self, v_max=95.0, grid_km=1e-5):
        """到着駅にちょうど停止する制動曲線を駅から逆向きに積分する"""
        xs, vs = [self.x1], [0.0]
        x, v = self.x1, 0.0
        while v < v_max and x > self.x0 - 0.5:
            a = self.vehicle.accel(ActionsMulti.braking, v,
                                   self.track.grade(x), self.track.curve(x))
            if a >= 0.0:
                a = -0.0001
            v_prev = v - a * SUB_DT
            x_prev = x - (v_prev / 3600.0) * SUB_DT - (a / 3600.0) * (SUB_DT ** 2)
            x, v = x_prev, v_prev
            xs.append(x); vs.append(v)
        xs = np.asarray(xs[::-1]); vs = np.asarray(vs[::-1])
        self._bx0 = float(xs[0]); self._bdx = grid_km
        n = int((self.x1 - self._bx0) / grid_km) + 2
        grid = self._bx0 + grid_km * np.arange(n)
        self._btab = np.interp(grid, xs, vs)

    def brake_speed(self, x):
        i = int((x - self._bx0) / self._bdx)
        if i < 0:
            return math.inf
        if i >= len(self._btab):
            return 0.0
        return float(self._btab[i])

    # ------------------------------------------------------------ 走行の再現
    def _hold_notches(self, v, g):
        """定速保持に使うノッチを返す (up, dn, hold, a_hold)。

        up   : 速度を上げられる最も緩やかなノッチ（加速度が正）
        dn   : 速度を下げられる最も緩やかなノッチ（加速度が負）
        hold : 加速度の絶対値が最小のノッチ。|a_hold| がほぼ0なら**真の定速保持**ができる
               （−35‰でのB2、+35‰でのP2がこれに当たる。§4.7.1）
        """
        acc = {c: self.vehicle.accel(FROM_CODE[c], v, g) for c in LC.NOTCH_ORDER}
        hold = min(LC.NOTCH_ORDER, key=lambda c: abs(acc[c]))
        pos = [c for c in LC.NOTCH_ORDER if acc[c] > 1e-6]
        neg = [c for c in LC.NOTCH_ORDER if acc[c] < -1e-6]
        up = pos[-1] if pos else "P1"     # NOTCH_ORDER は強い順なので末尾が最弱
        dn = neg[0] if neg else "B1"
        return up, dn, hold, acc[hold]

    def band_width(self, v, g):
        """維持帯の幅[km/h]。両相が t_min 秒以上continueする最小幅（§7.4.3(5)）。"""
        if self.hold_band is not None:
            return self.hold_band
        up, dn, hold, a_hold = self._hold_notches(v, g)
        if abs(a_hold) < 0.05:
            return 0.6            # 真の定速ノッチがある（±35‰のB2/P2）→ 帯は最小限でよい
        a_up = abs(self.vehicle.accel(FROM_CODE[up], v, g))
        a_dn = abs(self.vehicle.accel(FROM_CODE[dn], v, g))
        return self.t_min * max(a_up, a_dn)

    def simulate(self, v_hold, x_coast, sub_dt=SUB_DT, record=False):
        x, v, t = self.x0, 0.0, 0.0
        notch = "P1"
        stick = False        # 天井へ張り付くための制動ラッチ（ヒステリシス）
        rows = [] if record else None
        energy = 0.0          # 力行仕事 [J/t]
        e_brake = 0.0         # 制動で捨てた仕事 [J/t]（ノッチ由来の減速分のみ）
        changes = 0
        phase = "accel"
        max_t = 600.0
        while t < max_t:
            g = self.track.grade(x)
            cur = self.track.curve(x)
            ceil_v = self.ceiling(x)
            # --- ノッチ決定 ---
            if v >= self.brake_speed(x) - 1e-9:
                phase = "brake"; nn = "B1"
            elif x >= x_coast:
                phase = "coast"
                # 【重要】惰行で加速する下り勾配（coast_accel > 0）でも、**天井に達するまでは惰行**する。
                # 天井に達したときだけ最も弱い制動で張り付く（§7.4.3(7)）。
                # ここで常に制動すると、下り勾配で重力の助けを捨てて延々と制動し続けることになる。
                ca = self.vehicle.accel(ActionsMulti.coasting, v, g, cur)
                # 【重要】天井への張り付きには**ヒステリシス**が要る。
                # 「天井に触れたら制動／離れたら惰行」を素で書くと、弱い制動でも1ステップで
                # 天井を割るため 0.1 秒周期でノッチが振動する（実測でノッチ切替87回）。
                # 一度制動に入ったら hold_band ぶん下がるまで保持する。
                if ca > 0 and v >= ceil_v - 0.05:
                    stick = True
                if stick and v <= ceil_v - self.band_width(v, g):
                    stick = False
                nn = self._hold_notches(v, g)[1] if stick else "C"
            else:
                if v >= min(v_hold, ceil_v) - 1e-9:
                    phase = "hold"
                if phase == "hold":
                    up, dn, hold, a_hold = self._hold_notches(v, g)
                    # 帯の上端は天井のわずかに下に置く。天井と重ねると帯が潰れて
                    # ノッチが高速で往復する（実測でノッチ切替124回）。
                    top = min(v_hold, ceil_v - 0.2)
                    bot = top - self.band_width(v, g)
                    if v > ceil_v - 0.05 or v > top:
                        nn = dn
                    elif v < bot:
                        nn = up
                    elif abs(a_hold) < 0.05:
                        nn = hold          # 真の定速保持ノッチがある（±35‰でのB2/P2）
                    else:
                        nn = notch if notch in (up, dn, hold) else up
                else:
                    nn = "P1" if v < ceil_v - 0.3 else self._hold_notches(v, g)[1]
            if nn != notch:
                changes += 1
                notch = nn
            act = FROM_CODE[notch]
            a = self.vehicle.accel(act, v, g, cur)
            f = (self.vehicle.tractive_force(v) if act == ActionsMulti.power
                 else self.vehicle.tractive_force_p2(v) if act == ActionsMulti.grade_power else 0.0)
            if record:
                rows.append((t, x, v, notch, g, ceil_v, self.track.atc_limit(x)))
            # 力行仕事 [J/t] = F[kgf/t] × g × 走行距離[m]
            dx_m = (v / 3.6) * sub_dt
            energy += f * LC.GRAVITY * dx_m
            fb = (-self.vehicle.DECELERATE * LC.FACTOR_OF_INERTIA if notch == "B1"
                  else self.vehicle.brake_decel_b2(v) * LC.FACTOR_OF_INERTIA if notch == "B2" else 0.0)
            e_brake += fb * LC.GRAVITY * dx_m
            nv = v + a * sub_dt
            if nv < 0:
                nv = 0.0
            x += (v / 3600.0) * sub_dt + (a / 3600.0) * (sub_dt ** 2)
            v = nv
            t += sub_dt
            if x >= self.x1:
                break
            if v <= 1e-6 and t > 1.0 and phase != "brake":
                return dict(ok=False, time=t, stop_err=(self.x1 - x) * 1000.0,
                        energy=energy, e_brake=e_brake, changes=changes, rows=rows,
                        coast=x_coast, v_hold=v_hold)
            if v <= 1e-6 and phase == "brake":
                break
        ok = abs(x - self.x1) * 1000.0 < 30.0
        return dict(ok=ok, time=t, x_end=x, v_end=v,
                    stop_err=(x - self.x1) * 1000.0, energy=energy, e_brake=e_brake,
                    changes=changes, rows=rows, coast=x_coast, v_hold=v_hold)

    # ------------------------------------------------------------------ 最適化
    def solve_coast(self, v_hold, sub_dt=SUB_DT, iters=34, record=False):
        hi = self.x1
        best = self.simulate(v_hold, hi, sub_dt)
        if not best["ok"] or best["time"] > self.target_time:
            return None
        lo = self.x0
        for _ in range(iters):
            if hi - lo < 1e-6:
                break
            mid = (lo + hi) / 2.0
            r = self.simulate(v_hold, mid, sub_dt)
            if (not r["ok"]) or r["time"] > self.target_time:
                lo = mid
            else:
                hi = mid; best = r
        return self.simulate(v_hold, hi, sub_dt, record=True) if record else best

    def _pick(self, cands, tol=0.02):
        """力行仕事が最小値の tol 以内の候補のうち、**ノッチ切替が最少**のものを選ぶ。

        エネルギーだけで選ぶと、ほぼ同じエネルギーでも切替が数倍多い解が採られてしまう
        （実測: 木場→門前仲町でエネルギー差0.1%・切替 4回 vs 20回）。
        長い力行・惰行を保ち、出力を何度も切り替えない運転を選好する。
        """
        emin = min(r["energy"] for _, r in cands)
        near = [(vh, r) for vh, r in cands if r["energy"] <= emin * (1.0 + tol)]
        return min(near, key=lambda kv: (kv[1]["changes"], kv[1]["energy"]))[0]

    def optimize(self, coarse=2.0, fine=0.25, verbose=True):
        cands = []
        v = 10.0
        grid = []
        while v <= self.speed_cap + 1e-9:
            grid.append(round(v, 3)); v += coarse
        if abs(grid[-1] - self.speed_cap) > 1e-9:
            grid.append(round(self.speed_cap, 3))
        for vh in grid:
            r = self.solve_coast(vh, 0.05)
            if r is not None:
                cands.append((vh, r))
        if not cands:
            raise RuntimeError(
                f"標準運転時間 {self.target_time:.0f} 秒で到達できる運転曲線が見つかりません"
                f"（現示上限 {self.speed_cap:.0f} km/h）")
        if verbose:
            print("  [粗探索] V_hold ごとの力行仕事")
            for vh, r in cands:
                print(f"    V_hold={vh:5.1f} km/h  到着={r['time']:6.2f}s  "
                      f"惰行開始={(r['coast']-self.x0)*1000:7.1f}m  力行={r['energy']/1000:7.1f} kJ/t  "
                      f"制動={r['e_brake']/1000:7.1f} kJ/t  切替={r['changes']:3d}回")
        best_vh = self._pick(cands)
        fr = []
        vv = max(10.0, best_vh - coarse)
        while vv <= min(self.speed_cap, best_vh + coarse) + 1e-9:
            r = self.solve_coast(round(vv, 3), 0.05)
            if r is not None:
                fr.append((round(vv, 3), r))
            vv += fine
        pool = fr if fr else cands
        # 【重要】探索は 0.05 秒刻みで行っているため、切替回数が最終解（0.01秒刻み）と一致しない。
        # 力行仕事が最小値の tol 以内の候補を**最終刻みで解き直して**から選ぶ。
        emin = min(r["energy"] for _, r in pool)
        near = [vh for vh, r in pool if r["energy"] <= emin * (1.0 + 0.02)]
        near = sorted(set(near))[:8]
        finals = []
        for vh in near:
            r = self.solve_coast(vh, SUB_DT, record=True)
            if r is not None and r["ok"]:
                finals.append((vh, r))
        if verbose and finals:
            print("  [最終刻みでの再評価] 力行仕事が最小の2%以内の候補")
            for vh, r in finals:
                print(f"    V_hold={vh:6.2f} km/h  到着={r['time']:6.2f}s  力行={r['energy']/1000:7.1f} kJ/t  "
                      f"制動={r['e_brake']/1000:7.1f} kJ/t  切替={r['changes']:3d}回")
        if finals:
            best_vh, best = min(finals, key=lambda kv: (kv[1]["changes"], kv[1]["energy"]))
            return best, best_vh
        best = self.solve_coast(best_vh, SUB_DT, record=True)
        return best, best_vh


# =============================================================================
# 出力
# =============================================================================
def write_csv(sol, res, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csvmod.writer(f)
        w.writerow(["time", "position_km", "kilometrage", "speed", "notch",
                    "gradient", "atc_ceiling", "atc_now"])
        for (t, x, v, n, g, cv, an) in res["rows"][::10]:   # 0.1秒間隔に間引く
            w.writerow([f"{t:.2f}", f"{x:.6f}", f"{sol.track.to_kilometrage(x):.6f}",
                        f"{v:.4f}", n, f"{g:.2f}", f"{cv:.2f}", f"{an:.1f}"])


def plot_curve(sol, res, v_hold, path, jp):
    L = LC.NOTCH_LABEL_JA if jp else LC.NOTCH_LABEL_EN
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1], sharex=True)
    xs = np.array([r[1] for r in res["rows"]]) - sol.x0
    vs = np.array([r[2] for r in res["rows"]])
    ns = [r[3] for r in res["rows"]]

    # --- 勾配の帯 ---
    for s in sol.track.grade_sections(sol.x0, sol.x1):
        g = s["grade"]
        col = "#ffdddd" if g > 0 else ("#ddddff" if g < 0 else "#f2f2f2")
        ax.axvspan((s["start"] - sol.x0) * 1000, (s["end"] - sol.x0) * 1000,
                   color=col, zorder=0)
        if abs(g) >= 3.0 and (s["end"] - s["start"]) * 1000 > 60:
            ax.text(((s["start"] + s["end"]) / 2 - sol.x0) * 1000, 4.5,
                    f"{g:+.0f}‰", ha="center", fontsize=8, color="#666", zorder=1)
        ax2.axvspan((s["start"] - sol.x0) * 1000, (s["end"] - sol.x0) * 1000, color=col, zorder=0)

    # --- ATC現示（生の階段線と先読みパターン） ---
    step_x, step_y = [], []
    for s in sol.track.limit_sections(sol.x0, sol.x1):
        step_x += [(s["start"] - sol.x0) * 1000, (s["start"] + s["distance"] - sol.x0) * 1000]
        step_y += [s["speed_limit"], s["speed_limit"]]
    ax.plot(step_x, step_y, "k--", lw=1.2, alpha=0.7,
            label="ATC現示" if jp else "ATC indication", zorder=3)
    cx = np.arange(sol.x0, sol.x1, 0.001)
    ax.plot((cx - sol.x0) * 1000, [sol.ceiling(x) for x in cx], color="#888", lw=1.0, ls=":",
            label="先読みパターン(v_ceiling)" if jp else "Anticipatory ceiling", zorder=3)

    # --- 制動曲線 ---
    bx = np.arange(max(sol.x0, sol._bx0), sol.x1, 0.001)
    ax.plot((bx - sol.x0) * 1000, [sol.brake_speed(x) for x in bx], color="#bbb", lw=1.0, ls="-.",
            label="制動曲線" if jp else "Brake curve", zorder=2)

    # --- 運転曲線（ノッチ別） ---
    seen = set()
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[i]:
            j += 1
        lbl = None
        if ns[i] not in seen:
            seen.add(ns[i]); lbl = L[ns[i]]
        ax.plot(xs[i:j + 2] * 1000, vs[i:j + 2], color=LC.NOTCH_COLOR[ns[i]],
                lw=2.4, label=lbl, zorder=5)
        i = j + 1

    ax.axvline((res["coast"] - sol.x0) * 1000, color="#999", lw=1.0, ls="--", zorder=4)
    ax.text((res["coast"] - sol.x0) * 1000, 0.6,
            " 惰行開始" if jp else " coast", fontsize=9, color="#444",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    ax.set_ylabel("速度 [km/h]" if jp else "Speed [km/h]")
    ax.set_ylim(0, max(80, vs.max() * 1.15))
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9, ncol=2)
    title = (f"{sol.dep['name']} → {sol.arr['name']}  "
             f"{sol.distance_km*1000:.1f} m / 標準 {sol.target_time:.0f} s"
             f"   [{sol.vehicle.name}]")
    ax.set_title(title if jp else title.replace("標準", "std "))

    # --- 下段: 勾配プロファイル ---
    gx = np.arange(sol.x0, sol.x1, 0.001)
    ax2.plot((gx - sol.x0) * 1000, [sol.track.grade(x) for x in gx], color="#333", lw=1.4)
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_ylabel("勾配 [‰]" if jp else "Grade [permil]")
    ax2.set_xlabel("出発駅からの距離 [m]" if jp else "Distance from departure [m]")
    ax2.grid(alpha=0.3)

    info = (f"到着 {res['time']:.2f} s（標準 {sol.target_time:.0f} s）／停止位置誤差 {res['stop_err']*-1:+.3f} m\n"
            f"V_hold {v_hold:.2f} km/h ／最高速度 {vs.max():.1f} km/h ／"
            f"惰行開始 {(res['coast']-sol.x0)*1000:.1f} m\n"
            f"力行仕事 {res['energy']/1000:.1f} kJ/t ／制動仕事 {res['e_brake']/1000:.1f} kJ/t ／ノッチ切替 {res['changes']} 回")
    fig.text(0.012, 0.012, info, fontsize=9, va="bottom",
             bbox=dict(fc="white", ec="#ccc", alpha=0.9))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="tozai")
    ap.add_argument("--section", type=int, default=None, help="駅間index（省略で全駅間）")
    ap.add_argument("--hold-band", type=float, default=None)
    ap.add_argument("--atc-mode", default=None, choices=["anticipatory", "pattern"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    jp = setup_japanese_font()
    os.makedirs(OUT_DIR, exist_ok=True)
    tr = get_track(a.line)
    sections = [a.section] if a.section is not None else list(range(len(tr.stations) - 1))

    summary = []
    for sec in sections:
        sol = StandardCurveMulti(a.line, sec, hold_band=a.hold_band, atc_mode=a.atc_mode)
        print(f"\n=== [{sec}] {sol.dep['name']} → {sol.arr['name']} "
              f"{sol.distance_km*1000:.1f} m / 標準 {sol.target_time:.0f} s "
              f"/ 現示上限 {sol.speed_cap:.0f} km/h ===")
        try:
            res, vh = sol.optimize(verbose=not a.quiet)
        except RuntimeError as e:
            print(f"  ✗ {e}")
            summary.append((sec, sol, None, None))
            continue
        print(f"  → 到着 {res['time']:.2f}s / 停止位置誤差 {-res['stop_err']:+.3f} m "
              f"/ V_hold {vh:.2f} km/h / 最高速 {max(r[2] for r in res['rows']):.1f} km/h")
        print(f"     惰行開始 {(res['coast']-sol.x0)*1000:.1f} m / 力行仕事 {res['energy']/1000:.1f} kJ/t "
              f"/ 制動仕事 {res['e_brake']/1000:.1f} kJ/t / ノッチ切替 {res['changes']} 回")
        used = {}
        for r in res["rows"]:
            used[r[3]] = used.get(r[3], 0) + SUB_DT
        print("     ノッチ使用時間: " + " / ".join(
            f"{LC.NOTCH_LABEL_JA[c]} {used.get(c,0):.1f}s" for c in LC.NOTCH_ORDER if used.get(c, 0) > 0.05))
        base = f"{a.line}_{sec}_{sol.dep['name']}_{sol.arr['name']}"
        png = os.path.join(OUT_DIR, base + ".png")
        plot_curve(sol, res, vh, png, jp)
        write_csv(sol, res, os.path.join(OUT_DIR, base + ".csv"))
        with open(os.path.join(OUT_DIR, base + "_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"line": a.line, "section": sec,
                       "departure": sol.dep["name"], "arrival": sol.arr["name"],
                       "distance_m": sol.distance_km * 1000, "target_time": sol.target_time,
                       "arrival_time": res["time"], "stop_error_m": -res["stop_err"],
                       "v_hold": vh, "v_max": max(r[2] for r in res["rows"]),
                       "coast_start_m": (res["coast"] - sol.x0) * 1000,
                       "energy_kJ_per_t": res["energy"] / 1000,
                       "brake_kJ_per_t": res["e_brake"] / 1000, "notch_changes": res["changes"],
                       "vehicle": sol.vehicle.name, "atc_mode": sol.atc_mode},
                      f, ensure_ascii=False, indent=2)
        print(f"     出力: {os.path.relpath(png, BASE_DIR)}")
        summary.append((sec, sol, res, vh))

    print("\n================ まとめ ================")
    print(f"{'区間':<28}{'標準':>7}{'到着':>9}{'誤差':>9}{'V_hold':>9}{'最高速':>8}{'力行仕事':>11}{'制動仕事':>11}{'切替':>6}")
    for sec, sol, res, vh in summary:
        nm = f"[{sec}] {sol.dep['name']}→{sol.arr['name']}"
        if res is None:
            print(f"{nm:<28}{sol.target_time:6.0f}s{'  —  ':>9}{'  —  ':>9}{'  —  ':>9}{'  —  ':>8}{'  —  ':>11}{'  — ':>6}")
            continue
        print(f"{nm:<28}{sol.target_time:6.0f}s{res['time']:8.2f}s{-res['stop_err']:+8.3f}m"
              f"{vh:8.2f}{max(r[2] for r in res['rows']):7.1f}{res['energy']/1000:10.1f}kJ/t"
              f"{res['e_brake']/1000:10.1f}kJ/t{res['changes']:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
