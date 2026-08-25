# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の先行列車 走行パターンCSVを生成する。

既存 `generate_forward_train.py`（単一区間・白兎で1回だけ停車）は**一切変更しない**。
本スクリプトは以下を新たに担う。

  * **停車駅が2つ**（白兎 D_B 秒 / 蚕桑 D_C 秒）。自列車が白兎に停車中に発車判断を行うため、
    **蚕桑での長時間停車**（急病人救護など＝120秒・180秒）を表現できる必要がある。
  * 運転は自列車と同じ省エネ運転（惰行ポイント方式）。
      出発 → 惰行ポイント V[km/h] まで力行 → 惰行 → 駅に向かって制動 → 停車 → 再出発
  * 上り勾配で惰行だけでは制動開始点に届かない場合は V まで再力行する
    （白兎手前は 6.1→9.2‰、白兎→蚕桑は +11.4‰ が 1km 続くため、これが無いと駅間停車する）。

出力: `input/f_train_ymulti/coast{V}_b{D_B}_c{D_C}.csv`（time, position, speed, action）
先行の出発遅延はCSVでは表現しない（全パターン t=0 出発）。遅延は
`出発間隔 headway = 標準出発間隔120秒 − 先行遅延` に換算して環境へ渡す（既存と同じheadway換算モデル）。

使い方:
    python generate_forward_train_ymulti.py                 # 全312種
    python generate_forward_train_ymulti.py --jobs 8        # 並列生成
    python generate_forward_train_ymulti.py --test-only     # 検証用（V=65,50）のみ
"""
import os
import csv
import argparse
import codecs
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import config_ymulti as CFG
from train import Train
from actions import Actions
from brake_curve_ymulti import get_brake_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, CFG.F_TRAIN_DIR)

# ノッチ判断の周期[s]。train.py の積分刻み（0.01秒）と揃えることで制動開始点を0.01秒精度で置ける。
CONTROL_DT = 0.01
# 惰行から再力行に移るヒステリシス幅[km/h]
REACCEL_BAND = 3.0
# 惰行のみで制動開始点に到達できるかを調べる試行シミュレーションの刻み[s]と失速判定速度[km/h]
PROBE_DT = 1.0
PROBE_STALL_SPEED = 15.0
# 制動開始点の監視を始める駅までの距離[km]
BRAKE_WATCH_DIST = 0.5


def _station_positions():
    """input/Station.csv から対象3駅の位置[km]を読む（Station.csv は読むだけで変更しない）"""
    with codecs.open(os.path.join(BASE_DIR, "input/Station.csv"), "r", "utf-8", "ignore") as f:
        st = pd.read_csv(f)
    return [float(st["position"][i]) for i in CFG.STATION_INDICES]


def _action_str(action):
    if action == Actions.acceleration:
        return "Actions.acceleration"
    if action == Actions.deceleration:
        return "Actions.deceleration"
    return "Actions.coasting"


def _coast_reaches_brake_point(position, speed, stop_pos, target):
    """現在の位置・速度から惰行のみで停車駅の制動開始点に到達できるか。

    到達できる（＝このまま惰行して駅に停止できる）なら True、
    途中で PROBE_STALL_SPEED を下回る（＝駅間停車しそう）なら False。
    """
    bc = get_brake_curve(stop_pos)
    sim = Train(target, position=position, speed=speed)
    for _ in range(int(600 / PROBE_DT)):
        dist = stop_pos - sim.position
        if dist <= 0.0:
            return True
        if sim.speed >= bc.speed_at(sim.position):
            return True
        if sim.speed < PROBE_STALL_SPEED:
            return False
        sim.step(Actions.coasting, PROBE_DT)
    return False


def generate_pattern_csv(path, coast_speed, stops, start_pos, target,
                         total_seconds=None):
    """惰行ポイント方式の先行列車CSVを生成する。

    Args:
        stops: [(停車位置[km], 停車時間[s]), ...] を進行方向順に並べたもの。
        target: Train に与える遠方の目標駅位置[km]（CSVの記録範囲を走り切らせるため）。
    """
    total_seconds = total_seconds or CFG.F_TOTAL_SECONDS
    train = Train(target, position=start_pos, speed=0.0)
    ticks_per_sec = int(round(1.0 / CONTROL_DT))

    stop_queue = list(stops)
    stop_pos, dwell_time = (stop_queue.pop(0) if stop_queue else (None, 0.0))

    phase = "power"          # power / coast / brake / dwell
    coast_latched = False    # 惰行のみで制動開始点に届くと判定済み（＝もう再力行しない）
    dwell_elapsed = 0.0
    probe_fail_pos = None
    # 停車駅ごとの制動曲線（駅から逆積分。勾配・曲線の位置変化を織り込む）。
    # train.Train.req_stop_dist の「制動開始点の勾配・曲線が停止まで一定」という近似では
    # 蚕桑手前の R=400m 曲線を過大評価し 5.75m 過走した実績があるため、必ず制動曲線を使う。
    brake_curves = {round(p, 6): get_brake_curve(p) for p, _ in stops}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "position", "speed", "action"])

        for k in range(total_seconds * ticks_per_sec):
            if phase == "dwell":
                # ① 駅での停車（制動を込めたまま停車時間を消化する）
                action = Actions.deceleration
                dwell_elapsed += CONTROL_DT
                if dwell_elapsed >= dwell_time:
                    # 再出発。次の停車駅があればそれを監視対象にする
                    stop_pos, dwell_time = (stop_queue.pop(0) if stop_queue else (None, 0.0))
                    phase = "power"
                    coast_latched = False
                    probe_fail_pos = None
            elif phase == "brake":
                # ② 駅に向かって制動中
                action = Actions.deceleration
                if train.speed <= 0.0:
                    phase = "dwell"
                    dwell_elapsed = 0.0
                    # 停止位置の残差（数cm）を吸収して駅位置に正確に据える
                    if stop_pos is not None and abs(train.position - stop_pos) < 0.005:
                        train.set_states(0.0, stop_pos)
            else:
                dist = (stop_pos - train.position) if stop_pos is not None else float("inf")
                # ③ 制動開始点（制動曲線）に達したか。停車駅がまだ前方にある場合のみ監視する
                if stop_pos is not None and dist <= BRAKE_WATCH_DIST:
                    if train.speed >= brake_curves[round(stop_pos, 6)].speed_at(train.position):
                        phase = "brake"

                if phase == "brake":
                    action = Actions.deceleration
                elif phase == "power":
                    # ④ 惰行ポイントまで力行
                    action = Actions.acceleration
                    if train.speed >= coast_speed:
                        phase = "coast"
                        action = Actions.coasting
                else:
                    # ⑤ 惰行。駅間停車しそうなら惰行ポイントまで再力行する
                    action = Actions.coasting
                    if train.speed <= coast_speed - REACCEL_BAND:
                        if stop_pos is None:
                            # 前方に停車駅が無い（＝対象区間を走り終えた）ので V 付近を保持する
                            phase = "power"
                        elif not coast_latched:
                            # 「50m以上進んでから」再判定する（同じ結論の再計算を省く）
                            if probe_fail_pos is None or train.position - probe_fail_pos >= 0.05:
                                if _coast_reaches_brake_point(train.position, train.speed,
                                                              stop_pos, target):
                                    coast_latched = True     # 最後の惰行に入る
                                else:
                                    probe_fail_pos = train.position
                                    phase = "power"
                            else:
                                phase = "power"

            if k % ticks_per_sec == 0:
                writer.writerow([k // ticks_per_sec, round(train.position, 6),
                                 round(train.speed, 2), _action_str(action)])

            train.step(action, CONTROL_DT)


def _job(args):
    """1パターン分の生成（並列実行のワーカ）"""
    coast_speed, dwell_b, dwell_c, pos_b, pos_c, start_pos = args
    path = os.path.join(BASE_DIR, CFG.f_train_csv(coast_speed, dwell_b, dwell_c))
    generate_pattern_csv(path, float(coast_speed),
                         [(pos_b, float(dwell_b)), (pos_c, float(dwell_c))],
                         start_pos, CFG.F_TARGET_POSITION)
    return verify(path, pos_b, pos_c, dwell_b, dwell_c)


def verify(path, pos_b, pos_c, dwell_b, dwell_c):
    """生成したCSVを検証し、停止位置誤差と各駅の到着/発車時刻を返す。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["time"]), float(r["position"]), float(r["speed"])))
    info = {"path": os.path.relpath(path, BASE_DIR)}
    for tag, pos, dwell in (("b", pos_b, dwell_b), ("c", pos_c, dwell_c)):
        arrive = depart = None
        best_err = None
        stopped = False
        for t, p, v in rows:
            if v < 0.5 and abs(p - pos) < 0.05:
                if arrive is None:
                    arrive = t
                    best_err = (p - pos) * 1000.0
                stopped = True
            elif stopped and v >= 0.5 and p >= pos - 0.05 and depart is None:
                depart = t
        info[f"{tag}_arrive"] = arrive
        info[f"{tag}_depart"] = depart
        info[f"{tag}_err_m"] = best_err
        info[f"{tag}_dwell"] = (depart - arrive) if (arrive is not None and depart is not None) else None
    # 駅間停車（対象区間内で駅以外の場所に停止していないか）
    mid_stop = False
    for t, p, v in rows:
        if v < 0.5 and p > pos_b + 0.06 and p < pos_c - 0.06:
            mid_stop = True
            break
    info["mid_stop"] = mid_stop
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="複数駅間版の先行列車パターンCSVを生成する")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="並列プロセス数")
    ap.add_argument("--test-only", action="store_true",
                    help="検証用の惰行ポイント（65, 50）のみ生成する")
    a = ap.parse_args(argv)

    pos_a, pos_b, pos_c = _station_positions()
    speeds = CFG.F_COAST_SPEEDS_TEST if a.test_only else sorted(
        set(CFG.F_COAST_SPEEDS_TRAIN) | set(CFG.F_COAST_SPEEDS_TEST))

    jobs = [(v, b, c, pos_b, pos_c, pos_a)
            for v in speeds for b in CFG.F_DWELL_B for c in CFG.F_DWELL_C]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== 先行列車パターン（惰行ポイント方式・停車2駅）の生成 ===")
    print(f"  出発 {CFG.STATION_NAMES_JA[CFG.STATION_INDICES[0]]} {pos_a} km")
    print(f"  停車 {CFG.STATION_NAMES_JA[CFG.STATION_INDICES[1]]} {pos_b} km（{CFG.F_DWELL_B} 秒）")
    print(f"  停車 {CFG.STATION_NAMES_JA[CFG.STATION_INDICES[2]]} {pos_c} km（{CFG.F_DWELL_C} 秒）")
    print(f"  惰行ポイント {speeds[0]}〜{speeds[-1]} km/h / 全 {len(jobs)} 種 / 並列 {a.jobs}")

    results = []
    if a.jobs <= 1:
        for j in jobs:
            results.append(_job(j))
            print(f"  生成: {results[-1]['path']}")
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(_job, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if n % 20 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} 完了")

    # ---- 検証結果のサマリ ----
    bad = [r for r in results if r["mid_stop"] or r["b_arrive"] is None or r["c_arrive"] is None]
    errs = [abs(r[f"{t}_err_m"]) for r in results for t in ("b", "c") if r[f"{t}_err_m"] is not None]
    print(f"\n=== 完了: {len(results)} 件を {CFG.F_TRAIN_DIR}/ に出力 ===")
    if errs:
        print(f"  停止位置誤差: 最大 {max(errs):.3f} m / 平均 {sum(errs)/len(errs):.3f} m")
    if bad:
        print(f"  [警告] 駅間停車または停車失敗が {len(bad)} 件あります:")
        for r in bad[:10]:
            print(f"    {r['path']} mid_stop={r['mid_stop']} "
                  f"b_arrive={r['b_arrive']} c_arrive={r['c_arrive']}")
    else:
        print("  駅間停車・停車失敗: なし")
    # 代表パターンの時刻表
    print("\n  代表パターンの到着/発車時刻[s]（先行の出発を0とする）")
    for r in sorted(results, key=lambda x: x["path"])[:0] or []:
        pass
    for v in (65, 50):
        for c in CFG.F_DWELL_C:
            key = CFG.f_train_csv(v, 30, c)
            m = [r for r in results if r["path"].replace("\\", "/") == key]
            if m:
                r = m[0]
                print(f"    V={v} b30 c{c:<3}: 白兎着 {r['b_arrive']:>4} 発 {r['b_depart']:>4} / "
                      f"蚕桑着 {r['c_arrive']:>4} 発 {str(r['c_depart']):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
