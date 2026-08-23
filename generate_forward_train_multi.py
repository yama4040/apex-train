# -*- coding: utf-8 -*-
"""
複数駅間版の先行／後続列車の走行パターン生成（設計: docs_複数駅間最適化_計画.md §8.1）

既存 `generate_forward_train.py`（羽前成田→白兎・単一区間・3ノッチ）は**一切変更しない**。

方式は既存と同じ「**惰行ポイント方式**」を複数駅間へ延長したもの。
    出発 → 惰行ポイント V[km/h] まで力行 → 惰行 → 駅に向かって制動 → 停車 → 再出発
先行列車も自列車と同じ省エネ運転をしている前提とする。

【テストケース（2ケース）】
  normal … 通常運転。V は標準運転曲線の**惰行開始速度**（`generate_standard_curve_multi.py` の解）
           東陽町発 48.7 km/h ／ 木場発 74.0 km/h
  slow   … 低速運転。東陽町発 40 km/h ／ 木場発 55 km/h

【登り勾配での再力行】
  登り勾配区間で速度が **35 km/h** を下回ったら再力行し、V まで戻してから惰行に復帰する。
  これが無いと +27〜29.7‰ のランプで失速して駅間停車になる。

【ATC現示の順守】
  下り勾配（−16.3‰）では惰行でも加速するため、現示に近づいたら**勾配ブレーキ**で頭を抑える。
"""
import os
import csv as csvmod
import argparse

import line_config as LC
from actions_multi import FROM_CODE, CODE
from required_speed_multi import SpeedProfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "input", "f_train_multi")

# 惰行ポイント[km/h]（駅間ごと）
COAST_POINTS = {
    # 標準運転曲線の惰行開始速度（generate_standard_curve_multi.py の解より）
    "normal": [48.7, 74.0],
    # 低速運転（ユーザー指定）
    "slow": [40.0, 55.0],
}
REACCEL_MIN_SPEED = 35.0     # 登り勾配でこの速度を下回ったら再力行[km/h]
UPHILL_THRESHOLD = 1.0       # 「登り勾配」とみなす勾配[‰]
RECORD_DT = 1.0              # CSVの記録間隔[s]（環境側は線形補間して参照する）


def _run_section(sp, v_coast, t0, x0_override=None, sub_dt=LC.SUB_DT):
    """1駅間ぶんを走行して (時刻, 位置, 速度, ノッチ) の列を返す。"""
    x = sp.x0 if x0_override is None else x0_override
    v, t = 0.0, t0
    out = []
    coasting = False          # 惰行ポイントに到達済み
    reaccel = False           # 登り勾配での再力行中
    guard = 0
    while guard < 200000:
        guard += 1
        g = sp.track.grade(x)
        ceil = sp.atc_pattern(x)
        # --- ノッチ決定 ---
        if v >= sp.station_brake_speed(x) - 1e-9:
            code = "B1"                                   # 駅に向かって制動
        elif not coasting:
            code = "P1" if v < min(v_coast, ceil) - 0.2 else "C"
            if v >= min(v_coast, ceil) - 0.2:
                coasting = True
        else:
            # 登り勾配で失速しかけたら再力行し、V まで戻す
            if reaccel:
                if v >= min(v_coast, ceil) - 0.2:
                    reaccel = False
                    code = "C"
                else:
                    code = "P1"
            elif g > UPHILL_THRESHOLD and v < REACCEL_MIN_SPEED:
                reaccel = True
                code = "P1"
            elif v > ceil - 0.3 and sp.coast_accel(v, x) > 0:
                code = "B2"                               # 下り勾配で現示に迫る → 弱い制動
            else:
                code = "C"
        a = sp.notch_accel(code, v, x)
        nv = max(0.0, v + a * sub_dt)
        x += (v / 3600.0) * sub_dt + (a / 3600.0) * (sub_dt ** 2)
        v = nv
        t += sub_dt
        out.append((t, x, v, code))
        if x >= sp.x1 or (v <= 1e-6 and code == "B1"):
            break
        if v <= 1e-6 and t - t0 > 2.0:
            break                                          # 失速（駅間停車）
    return out, t, x, v


def generate(line="tozai", pattern="normal", dwell_b=30.0, dwell_c=30.0,
             stations=None, verbose=True):
    """複数駅間を通しで走る先行列車の走行パターンを生成する。"""
    from train_multi import get_track
    tr = get_track(line)
    stations = stations if stations is not None else tr.cfg["default_stations"]
    n_sec = len(stations) - 1
    coast = COAST_POINTS[pattern]
    dwells = [dwell_b, dwell_c]

    rows = []          # (time, x_internal, speed, notch)
    t = 0.0
    for si in range(n_sec):
        sp = SpeedProfile(line, stations[si], load_std_curve=False)
        v_c = coast[si] if si < len(coast) else coast[-1]
        seg, t, x_end, v_end = _run_section(sp, v_c, t)
        rows.extend(seg)
        err = (x_end - sp.x1) * 1000.0
        if verbose:
            used = {}
            for (_, _, _, c) in seg:
                used[c] = used.get(c, 0) + LC.SUB_DT
            vmax = max(r[2] for r in seg)
            print(f"    [{si}] {sp.dep['name']}→{sp.arr['name']}  惰行ポイント {v_c:.1f} km/h"
                  f"  所要 {seg[-1][0]-seg[0][0]+LC.SUB_DT:.1f}s（標準 {sp.target_time:.0f}s）"
                  f"  最高速 {vmax:.1f}  停止位置誤差 {-err:+.2f} m")
            print("        ノッチ: " + " / ".join(
                f"{LC.NOTCH_LABEL_JA[c]} {used.get(c,0):.1f}s" for c in LC.NOTCH_ORDER if used.get(c, 0) > 0.05))
            if abs(err) > 1.0:
                print(f"        ⚠ 停止位置誤差が大きい（{-err:+.2f} m）")
        # 駅停車（最終駅も停車時間ぶん記録して、後続の判定に使えるようにする）
        d = dwells[si] if si < len(dwells) else 30.0
        n = int(round(d / LC.SUB_DT))
        for _ in range(n):
            t += LC.SUB_DT
            rows.append((t, sp.x1, 0.0, "B1"))
    return rows


def write_csv(rows, path, line="tozai"):
    from train_multi import get_track
    tr = get_track(line)
    step = int(round(RECORD_DT / LC.SUB_DT))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        w.writerow(["time", "position", "kilometrage", "speed", "action"])
        for i in range(0, len(rows), step):
            t, x, v, c = rows[i]
            w.writerow([f"{t:.0f}", f"{x:.6f}", f"{tr.to_kilometrage(x):.6f}",
                        f"{v:.4f}", c])
        t, x, v, c = rows[-1]
        w.writerow([f"{t:.0f}", f"{x:.6f}", f"{tr.to_kilometrage(x):.6f}", f"{v:.4f}", c])


def path_for(line, pattern, dwell_b, dwell_c):
    return os.path.join(OUT_DIR, f"{line}_{pattern}_stopB{int(dwell_b)}_stopC{int(dwell_c)}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="tozai")
    ap.add_argument("--dwell-b", type=float, nargs="+", default=[30.0, 45.0])
    ap.add_argument("--dwell-c", type=float, nargs="+",
                    default=[30.0, 60.0, 90.0, 120.0, 180.0, 240.0])
    ap.add_argument("--patterns", nargs="+", default=["normal", "slow"])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 92)
    print("先行／後続列車の走行パターン生成（惰行ポイント方式・複数駅間）")
    print(f"  登り勾配（>{UPHILL_THRESHOLD:.0f}‰）で {REACCEL_MIN_SPEED:.0f} km/h を下回ったら V まで再力行")
    print("=" * 92)
    n = 0
    for pat in a.patterns:
        print(f"\n### パターン '{pat}'  惰行ポイント {COAST_POINTS[pat]} km/h")
        first = True
        for db in a.dwell_b:
            for dc in a.dwell_c:
                rows = generate(a.line, pat, db, dc, verbose=first)
                first = False
                write_csv(rows, path_for(a.line, pat, db, dc), a.line)
                n += 1
        print(f"    → {len(a.dwell_b) * len(a.dwell_c)} ファイル出力")
    print(f"\n合計 {n} ファイルを {os.path.relpath(OUT_DIR, BASE_DIR)}/ に出力しました。")


if __name__ == "__main__":
    main()
