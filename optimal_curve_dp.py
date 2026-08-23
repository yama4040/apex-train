# -*- coding: utf-8 -*-
"""
動的計画法による**エネルギー最小運転曲線**（generate_standard_curve_multi.py の検証用）

`generate_standard_curve_multi.py` は「力行 → 定速保持 → 惰行 → 制動」という運転パターンを
仮定し、設計変数を (V_hold, x_coast) の2つに絞って探索する。
この仮定が最適解を取り逃がしていないかを確認するため、**運転パターンを一切仮定せず**
位置×速度の格子上で動的計画法（DP）を解く。

    最小化   E_net + λ·T + μ·(ノッチ切替回数)
    λ を二分探索して T = 標準運転時間 に一致させる（時間制約のラグランジュ緩和）

E_net の定義（15000系は**回生ブレーキ付き**である。PDF「CS-ATC連動電気指令式電空併用ブレーキ（回生付）」）
    E_net = E_power − η · E_brake      η: 回生効率（既定0.0＝回生なしで評価）
    E_power: 力行仕事 [J/t]   E_brake: 制動で捨てた仕事 [J/t]

μ を正にすると「長い力行・惰行を使い、ノッチ切替の少ない運転」を選好する。
"""
import argparse
import numpy as np

import line_config as LC
from actions_multi import FROM_CODE
from generate_standard_curve_multi import StandardCurveMulti

NOTCHES = LC.NOTCH_ORDER
BIG = 1e15


def solve_dp(sol, lam, dx_m=5.0, dv=0.25, v_max=80.0, mu=0.0, eta=0.0):
    """E_net + λT + μ(切替) を最小化する後ろ向きDP。状態は (位置, 速度, 直前ノッチ)。"""
    L = sol.distance_km * 1000.0
    n = int(round(L / dx_m))
    dx_km = dx_m / 1000.0
    vs = np.arange(0.0, v_max + dv, dv)
    nv, nk = len(vs), len(NOTCHES)
    veh = sol.vehicle

    ceil = np.array([sol.ceiling(sol.x0 + i * dx_km) for i in range(n + 1)])
    grade = np.array([sol.track.grade(sol.x0 + (i + 0.5) * dx_km) for i in range(n)])
    curve = np.array([sol.track.curve(sol.x0 + (i + 0.5) * dx_km) for i in range(n)])

    # J[k][v] : 直前ノッチ k・速度 v で位置 i にいるときの最小コスト
    J = np.full((nk, nv), BIG)
    J[:, 0] = 0.0                                  # 終端: 駅で速度0
    policy = np.zeros((n, nk, nv), dtype=np.int8)

    for i in range(n - 1, -1, -1):
        g, c = grade[i], curve[i]
        newJ = np.full((nk, nv), BIG)
        for k, code in enumerate(NOTCHES):
            act = FROM_CODE[code]
            a = np.array([veh.accel(act, v, g, c) for v in vs]) / 3.6      # m/s²
            v_ms = vs / 3.6
            sq = v_ms ** 2 + 2.0 * a * dx_m
            reach = sq > 1e-9                       # このセルを走り切れるか
            vn = np.where(reach, np.sqrt(np.maximum(sq, 0.0)) * 3.6, 0.0)
            v_avg = (vs + vn) / 2.0
            dt = np.where(v_avg > 1e-6, dx_m / (v_avg / 3.6), BIG)
            # セル内で停止してしまう場合: 停止までの時間を使い、駅までの残距離に比例した罰則を科す。
            # 最終セル（i = n-1）では残距離0なので罰則なし＝これが駅での停止に相当する。
            with np.errstate(divide="ignore", invalid="ignore"):
                dt_stop = np.where(a < -1e-9, v_ms / np.abs(a), BIG)
            stop_pen = 1e7 * (n - 1 - i)
            dt = np.where(reach, dt, dt_stop)
            if code == "P1":
                f = np.array([veh.tractive_force(v) for v in vs])
            elif code == "P2":
                f = np.array([veh.tractive_force_p2(v) for v in vs])
            else:
                f = np.zeros(nv)
            e_pow = f * LC.GRAVITY * dx_m
            if code == "B1":
                fb = np.full(nv, -veh.DECELERATE * LC.FACTOR_OF_INERTIA)
            elif code == "B2":
                fb = np.array([veh.brake_decel_b2(v) for v in vs]) * LC.FACTOR_OF_INERTIA
            else:
                fb = np.zeros(nv)
            e_brk = fb * LC.GRAVITY * dx_m
            imm = e_pow - eta * e_brk + lam * dt + np.where(reach, 0.0, stop_pen)
            idx = np.clip(np.rint(vn / dv).astype(int), 0, nv - 1)
            future = J[k, idx]                      # 今のノッチが次ステップの「直前ノッチ」になる
            tot = imm + future
            bad = (vn > ceil[i + 1] + 1e-9) | (vs > ceil[i] + 1e-9) | (future >= BIG)
            tot = np.where(bad, BIG, tot)
            for prev in range(nk):
                cand = tot + (mu if prev != k else 0.0)
                upd = cand < newJ[prev]
                newJ[prev] = np.where(upd, cand, newJ[prev])
                policy[i, prev] = np.where(upd, k, policy[i, prev])
        J = newJ
    return J, policy, vs, dv


def rollout(sol, policy, vs, dv, dx_m=5.0, eta=0.0):
    """方策を前向きに再生する（0.01秒刻みの物理で厳密に）"""
    n = policy.shape[0]
    dx_km = dx_m / 1000.0
    x, v, t = sol.x0, 0.0, 0.0
    rows = []
    e_pow = e_brk = 0.0
    changes = 0
    prev_k = 0
    nv = len(vs)
    for i in range(n):
        vi = int(np.clip(round(v / dv), 0, nv - 1))
        k = int(policy[i, prev_k, vi])
        code = NOTCHES[k]
        if k != prev_k:
            changes += 1
        prev_k = k
        act = FROM_CODE[code]
        x_target = sol.x0 + (i + 1) * dx_km
        guard = 0
        while x < x_target and guard < 200000:
            g = sol.track.grade(x); c = sol.track.curve(x)
            a = sol.vehicle.accel(act, v, g, c)
            f = (sol.vehicle.tractive_force(v) if code == "P1"
                 else sol.vehicle.tractive_force_p2(v) if code == "P2" else 0.0)
            fb = (-sol.vehicle.DECELERATE * LC.FACTOR_OF_INERTIA if code == "B1"
                  else sol.vehicle.brake_decel_b2(v) * LC.FACTOR_OF_INERTIA if code == "B2" else 0.0)
            rows.append((t, x, v, code, g, sol.ceiling(x), sol.track.atc_limit(x)))
            dxm = (v / 3.6) * LC.SUB_DT
            e_pow += f * LC.GRAVITY * dxm
            e_brk += fb * LC.GRAVITY * dxm
            nv_ = max(0.0, v + a * LC.SUB_DT)
            x += (v / 3600.0) * LC.SUB_DT + (a / 3600.0) * (LC.SUB_DT ** 2)
            v = nv_
            t += LC.SUB_DT
            guard += 1
            if v <= 1e-9 and a <= 0:
                break
        if v <= 1e-9 and i > n * 0.5:
            break
    return dict(time=t, energy=e_pow, e_brake=e_brk, e_net=e_pow - eta * e_brk,
                changes=changes, rows=rows, stop_err=(x - sol.x1) * 1000.0,
                coast=sol.x0, v_hold=float("nan"))


def optimize(sol, dx_m=5.0, dv=0.25, mu=0.0, eta=0.0, verbose=True):
    lo, hi, best = 0.0, 50000.0, None
    for it in range(20):
        lam = (lo + hi) / 2.0
        _, pol, vs, dvv = solve_dp(sol, lam, dx_m, dv, mu=mu, eta=eta)
        r = rollout(sol, pol, vs, dvv, dx_m, eta=eta)
        if verbose and (it % 4 == 0 or it >= 17):
            print(f"    λ={lam:9.1f} → {r['time']:7.2f}s  力行 {r['energy']/1000:7.1f} kJ/t  "
                  f"制動 {r['e_brake']/1000:7.1f} kJ/t  切替 {r['changes']:3d}回")
        if r["time"] > sol.target_time:
            lo = lam
        else:
            hi = lam; best = r
        if best is not None and abs(best["time"] - sol.target_time) < 0.2:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="tozai")
    ap.add_argument("--section", type=int, default=2)
    ap.add_argument("--dx", type=float, default=5.0)
    ap.add_argument("--dv", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=0.0, help="ノッチ切替1回あたりのコスト[J/t相当]")
    ap.add_argument("--eta", type=float, default=0.0, help="回生効率（0=回生なし）")
    a = ap.parse_args()
    sol = StandardCurveMulti(a.line, a.section)
    print(f"=== [{a.section}] {sol.dep['name']} → {sol.arr['name']} "
          f"{sol.distance_km*1000:.1f} m / 標準 {sol.target_time:.0f} s ===")
    print(f"格子 {a.dx:.0f} m × {a.dv} km/h / 切替コスト μ={a.mu} / 回生効率 η={a.eta}")
    r = optimize(sol, a.dx, a.dv, a.mu, a.eta)
    if r is None:
        print("  解なし"); return
    print(f"\n[DP最適] 所要 {r['time']:.2f}s / 停止位置誤差 {-r['stop_err']:+.2f} m")
    print(f"  力行仕事 {r['energy']/1000:.1f} kJ/t / 制動仕事 {r['e_brake']/1000:.1f} kJ/t "
          f"/ ノッチ切替 {r['changes']}回")
    used = {}
    for row in r["rows"]:
        used[row[3]] = used.get(row[3], 0) + LC.SUB_DT
    print("  ノッチ使用時間: " + " / ".join(
        f"{LC.NOTCH_LABEL_JA[c]} {used.get(c,0):.1f}s" for c in NOTCHES if used.get(c, 0) > 0.05))


if __name__ == "__main__":
    main()
