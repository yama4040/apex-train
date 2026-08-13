# 運転曲線の停止部分をTASC（停止位置制御）の制動パターンで上書きするスクリプト（2026-08-13）
#
# 【背景・設計メモ §30】
# 実車では停止位置制御をTASC（自動列車停止位置制御装置）が担うため、DQNに停止まで学習させる
# 必要はない。しかしTASCを学習ループに入れると、TASC作動中は3つの行動がすべて同じ結果になり、
# Q学習のmax演算子が過大評価バイアスを累積してQ値が膨張し、方策が無差別化して破綻した
# （実測: 本来0のQ値が211まで膨張／行動間の差がQ値の0.36%）。
#
# そこで学習ループにはTASCを入れず（environment2.TASC_ENABLED=False）、
# **学習後の運転曲線に対してのみ**TASCの制動を後処理で適用する。
#
# 【処理内容】
#   1. Testerが出力した走行ログCSV（apex2.pyの新形式）を読む。
#   2. DQNが最後に減速を始めた地点を検出する。
#   3. その手前の状態から再シミュレーションし、
#        ・まだ制動パターンに達していなければ、直前のノッチを延長してパターンまで走らせる
#        ・パターンに達したらTASCの制動（パターン追従）で所定位置に停止させる
#   4. 上書き後の走行曲線をCSVとPNGに出力する。
#
# 制動パターンと物理モデルは environment2 / train.py と同一のものを使う（実勾配で逆積分）。
#
# 使い方:
#     python apply_tasc_to_runcurve.py data/<run>/<cycle>_<ci>.csv
#     python apply_tasc_to_runcurve.py "data/<run>/21750_*.csv"      # まとめて処理
#     python apply_tasc_to_runcurve.py <csv> --outdir tasc_curves    # 出力先を指定

import argparse
import bisect
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")   # PNG保存のみ。既定のqtaggだとSIGABRTで落ちる環境があるため。
import matplotlib.pyplot as plt

import environment2 as e2
from actions import Actions
from train import Train

TASC_SUB_DT = 0.01          # TASC作動中の制御周期[s]（train.pyの積分刻みと同じ下限）
DEFAULT_OUTDIR = "tasc_curves"


def build_pattern(env):
    """停止点から実勾配で逆積分した制動パターン（距離[m]→速度[km/h]）を返す。"""
    return env._build_tasc_pattern(env.arrival_station["position"])


def pattern_speed(dists, speeds, d_m):
    """停止点まで d_m [m] の地点における制動パターン速度[km/h]。"""
    if d_m <= 0.0:
        return 0.0
    i = bisect.bisect_left(dists, d_m)
    if i >= len(dists):
        return speeds[-1]
    if i == 0:
        return speeds[0]
    span = dists[i] - dists[i - 1]
    if span <= 0:
        return speeds[i]
    r = (d_m - dists[i - 1]) / span
    return speeds[i - 1] + r * (speeds[i] - speeds[i - 1])


def load_run(path):
    """Tester出力CSVから (時刻, 位置[km], 速度[km/h], 行動) の系列を読む。

    apex2.pyの新形式（末尾に time/position/speed_limit/... の9列を持つ）を前提とする。
    旧形式（列を持たない）は raw_speed / raw_stat_dist から復元する。
    """
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"空のCSVです: {path}")

    def fnum(r, k, d=0.0):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return d

    if "position" in rows[0] and "time" in rows[0]:
        ts = [fnum(r, "time") for r in rows]
        pos = [fnum(r, "position") for r in rows]
        spd = [fnum(r, "speed") if "speed" in rows[0] else fnum(r, "raw_speed") for r in rows]
        act = [int(fnum(r, "action", 0)) for r in rows] if "action" in rows[0] else [0] * len(rows)
    else:
        # 旧形式: 駅までの残距離(km)と速度から復元する。時刻は time_step 規則で再構成。
        spd = [fnum(r, "raw_speed") for r in rows]
        rem = [fnum(r, "raw_stat_dist") for r in rows]
        pos, ts, act = [], [], [0] * len(rows)
        t = 0.0
        for i, r in enumerate(rows):
            ts.append(t)
            t += 0.1 if rem[i] <= 0.1 else 1.0
        pos = rem   # 呼び出し側で駅位置を足して絶対位置にする
    return rows, ts, pos, spd, act


def find_splice_index(spd, pos_km, station_km, dists, speeds, min_speed=3.0):
    """TASCへ引き継ぐインデックスと、その理由を返す。

    優先順位:
      1. **DQNの軌跡が制動パターンに最初に到達した点**。実車のTASCが作動する瞬間そのもので、
         引き継ぎが遅れる心配がない。DQNが高速のまま駅へ接近した場合はここで捕まる。
      2. パターンに一度も到達しない（早めに減速しすぎた）場合は、最後の減速を始めた点。
         そこから直前のノッチを延長してパターンまで走らせる。

    戻り値: (index, reason)
    """
    n = len(spd)
    for i in range(n):
        d = (station_km - pos_km[i]) * 1000.0
        if d <= 0:
            break
        if spd[i] >= pattern_speed(dists, speeds, d):
            # 【重要】記録は巡航中1秒刻みのため、この点は既にパターンを最大19m行き過ぎている
            # 可能性がある（69km/hで1秒＝19.2m）。制動は最大減速なので行き過ぎは回復できない。
            # そこで1つ手前（まだパターンの下）から再現し、0.1秒刻みの先読み判定に任せる。
            return max(i - 1, 0), "pattern"

    # パターンに達していない → 最終減速の開始点まで遡る
    i = n - 1
    while i > 0 and spd[i] < min_speed:      # 末尾の停車部分を飛ばす
        i -= 1
    peak = i
    while peak > 0 and spd[peak - 1] >= spd[peak]:   # 減速が始まった点まで戻る
        peak -= 1
    return peak, "extend"


def simulate_tasc_tail(env, start_pos_km, start_speed, hold_action, dists, speeds,
                       t0=0.0, coarse_dt=0.1):
    """start から「パターンに達するまで hold_action を継続 → TASC制動で停止」を再現する。

    戻り値: (時刻リスト, 位置[km]リスト, 速度リスト, 行動リスト, TASC作動時の速度)
    """
    station = env.arrival_station["position"]
    tr = Train(station, start_pos_km, start_speed, 1.0)
    ts, pos, spd, act = [], [], [], []
    t = t0
    engaged = False
    engage_speed = None
    guard = 0
    while tr.speed > 0.001 and guard < 200000:
        guard += 1
        d = (station - tr.position) * 1000.0
        if not engaged:
            # 次ステップで通り過ぎる分を先読みして判定する（遅れると制動では回復できない）
            d_check = d - (tr.speed / 3.6) * coarse_dt
            if tr.speed >= pattern_speed(dists, speeds, max(d_check, 0.0)):
                engaged = True
                engage_speed = tr.speed
        if engaged:
            # TASC: パターンを上回れば制動、下回れば惰行。制御周期は TASC_SUB_DT。
            a = Actions.deceleration if tr.speed >= pattern_speed(dists, speeds, d) else Actions.coasting
            ts.append(t); pos.append(tr.position); spd.append(tr.speed); act.append(int(a))
            tr.step(a, TASC_SUB_DT)
            t += TASC_SUB_DT
        else:
            a = Actions(hold_action)
            ts.append(t); pos.append(tr.position); spd.append(tr.speed); act.append(int(a))
            tr.step(a, coarse_dt)
            t += coarse_dt
            if d <= 0:      # パターンに達しないまま駅を通過した場合の保険
                engaged = True
    ts.append(t); pos.append(tr.position); spd.append(tr.speed); act.append(int(Actions.deceleration))
    return ts, pos, spd, act, engage_speed


def process(path, outdir, departure_index=11):
    env = e2.Environment(load_reward_predictor=False)
    env.reset(departure_index, 0.0, 1.0)
    station = env.arrival_station["position"]
    dists, speeds = build_pattern(env)

    rows, ts, pos_raw, spd, act = load_run(path)
    # 旧形式は「駅までの残距離[km]」なので絶対位置へ直す
    pos = pos_raw if max(pos_raw) > 1.0 else [station - p for p in pos_raw]

    b, reason = find_splice_index(spd, pos, station, dists, speeds)
    hold = act[b - 1] if b > 0 else int(Actions.coasting)
    # 直前が制動なら惰行に読み替える（「制動を始める直前の行動」を延長する意図のため）
    if hold == int(Actions.deceleration):
        hold = int(Actions.coasting)

    tail_t, tail_pos, tail_spd, tail_act, eng_v = simulate_tasc_tail(
        env, pos[b], spd[b], hold, dists, speeds, t0=ts[b])

    new_t = ts[:b] + tail_t
    new_pos = pos[:b] + tail_pos
    new_spd = spd[:b] + tail_spd
    new_act = act[:b] + tail_act
    err = (new_pos[-1] - station) * 1000.0

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_csv = os.path.join(outdir, f"{base}_tasc.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "position", "speed", "action", "source"])
        for i in range(len(new_t)):
            w.writerow([round(new_t[i], 3), round(new_pos[i], 6), round(new_spd[i], 4),
                        new_act[i], "DQN" if i < b else "TASC"])

    # --- 運転曲線（DQN区間とTASC区間を色分け）---
    plt.figure(figsize=(10, 10), dpi=200)
    plt.plot([p for p in pos[:b + 1]], spd[:b + 1], color="red", lw=1.3, label="DQN")
    plt.plot(tail_pos, tail_spd, color="blue", lw=1.3, label="TASC (stop control)")
    plt.axvline(station, color="black", lw=2)
    plt.axhline(env.current_speed_limit, color="black", lw=0.8)
    plt.xlabel("Position[km]"); plt.ylabel("Speed[km/h]")
    plt.title(f"{base}  stop error {err:+.3f} m"
              + (f" / TASC from {eng_v:.1f} km/h" if eng_v else ""))
    plt.legend()
    out_png = os.path.join(outdir, f"{base}_tasc.png")
    plt.savefig(out_png); plt.close()

    tag = "パターン到達点" if reason == "pattern" else "減速開始点から延長"
    print(f"  {base}: DQN {b}歩({tag}) → TASC上書き / 停止誤差 {err:+.3f}m"
          + (f" / 作動時 {eng_v:.1f}km/h" if eng_v else ""))
    return err


def main():
    ap = argparse.ArgumentParser(
        description="運転曲線の停止部分をTASCの制動パターンで上書きする（学習には影響しない後処理）")
    ap.add_argument("csv", nargs="+", help="Tester出力の走行ログCSV（ワイルドカード可）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"出力先（既定 {DEFAULT_OUTDIR}/）")
    ap.add_argument("--departure-index", type=int, default=11, help="出発駅インデックス（既定11）")
    args = ap.parse_args()

    paths = []
    for pat in args.csv:
        paths.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not paths:
        raise SystemExit("対象CSVが見つかりません。")

    print(f"{len(paths)}件を処理します（出力: {args.outdir}/）")
    errs = []
    for p in paths:
        try:
            errs.append(process(p, args.outdir, args.departure_index))
        except Exception as e:      # 1件失敗しても残りを続ける
            print(f"  [失敗] {os.path.basename(p)}: {type(e).__name__}: {e}")
    if errs:
        print(f"\n停止誤差: 平均 {sum(abs(e) for e in errs)/len(errs):.3f}m / "
              f"最大 {max(abs(e) for e in errs):.3f}m / ±0.2m以内 {sum(1 for e in errs if abs(e)<=0.2)}/{len(errs)}件")


if __name__ == "__main__":
    main()


# =====================================================================================
# apex2.py の Tester から呼ぶための組み込みAPI
# =====================================================================================
def overwrite_trajectory(env, times, positions, speeds, actions, modes):
    """走行軌跡の停止部分をTASCの制動パターンで上書きする（学習には影響しない後処理）。

    apex2.py の Tester が記録した系列を受け取り、TASC適用後の系列を返す。
    引き継ぎ規則は本スクリプト単体実行時と同一:
      1. 制動パターンに最初に到達した点で引き継ぐ（ただし記録は巡航中1秒刻みで
         最大19m行き過ぎている可能性があるため1つ手前から再現する）
      2. パターンに達しない場合は最終減速の開始点まで遡り、直前のノッチを延長する

    戻り値: (times, positions, speeds, actions, modes, info)
      info = {"splice": 引き継ぎindex, "reason": "pattern"/"extend",
              "engage_speed": 作動時速度 or None, "stop_error_m": 停止位置誤差[m]}
    """
    station = env.arrival_station["position"]
    dists, speeds_pat = env._build_tasc_pattern(station)

    b, reason = find_splice_index(speeds, positions, station, dists, speeds_pat)
    hold = actions[b - 1] if b > 0 else int(Actions.coasting)
    if hold == int(Actions.deceleration):
        # 「制動を始める直前の行動」を延長する意図なので、制動なら惰行に読み替える
        hold = int(Actions.coasting)

    t_tail, p_tail, v_tail, a_tail, eng_v = simulate_tasc_tail(
        env, positions[b], speeds[b], hold, dists, speeds_pat, t0=times[b])

    # モードは引き継ぎ時点のものをTASC区間にも適用する（配色を運転曲線と揃えるため）
    mode_at_splice = modes[b] if b < len(modes) else (modes[-1] if modes else "normal")
    new_modes = list(modes[:b]) + [mode_at_splice] * len(t_tail)

    info = {"splice": b, "reason": reason, "engage_speed": eng_v,
            "stop_error_m": (p_tail[-1] - station) * 1000.0}
    return (list(times[:b]) + t_tail, list(positions[:b]) + p_tail,
            list(speeds[:b]) + v_tail, list(actions[:b]) + a_tail, new_modes, info)


def forward_train_at(env, t):
    """時刻 t における先行列車の (位置[km], 速度[km/h])。先行なしは None。

    environment2.step() と同一の補間規則（記録軌道の線形補間）を用いる。
    TASC上書きで時間軸が変わるため、先行列車も新しい時刻で引き直す必要がある。
    """
    ctr = getattr(env, "fowerd_train_controls", None)
    if not ctr or getattr(env, "fowerd_train_time_offset", None) is None:
        return None
    f_tau = t + env.fowerd_train_time_offset
    idx = int(f_tau)
    if idx >= len(ctr) - 1:
        return ctr[-1]["position"], ctr[-1]["speed"]
    frac = f_tau - idx
    p = ctr[idx]["position"] + (ctr[idx + 1]["position"] - ctr[idx]["position"]) * frac
    v = ctr[idx]["speed"] + (ctr[idx + 1]["speed"] - ctr[idx]["speed"]) * frac
    return p, v
