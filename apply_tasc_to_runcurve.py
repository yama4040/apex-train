# 運転曲線の停止部分をTASC（停止位置制御）の制動パターンで上書きする後処理スクリプト
#   初版 2026-08-13 ／ バッチ化・不具合修正 2026-08-18
#
# 【背景・設計メモ §30〜§33】
# 実車では停止位置制御をTASC（自動列車停止位置制御装置）が担うため、DQNに停止まで学習させる
# 必要はない。しかしTASCを学習ループに入れると、TASC作動中は3つの行動がすべて同じ結果になり、
# Q学習のmax演算子が過大評価バイアスを累積してQ値が膨張し、方策が無差別化して破綻した
# （実測: 本来0のQ値が211まで膨張／行動間の差がQ値の0.36%）。
#
# そこで学習ループにはTASCを入れず（environment2.TASC_ENABLED=False）、
# **学習後の走行ログに対してのみ**TASCの制動を後処理で適用する。
# apex2.py の実行中に出力していたが、学習時間が延びるため後処理スクリプトへ分離した（§33）。
#
# 【使い方】
#   # 学習フォルダとcycle番号を指定して、そのcycleの全テストケースを一括処理する
#   python apply_tasc_to_runcurve.py data/20260818015134 12950
#   python apply_tasc_to_runcurve.py data/20260818015134            # cycle省略=最新cycle
#   python apply_tasc_to_runcurve.py data/20260818015134 12950 --cases 0 3 14
#   python apply_tasc_to_runcurve.py --csv data/20260818015134/12950_4.csv   # 単一ファイル
#
# 出力先は `<run>/TASC制御/`（`<cycle>_<ci>_tasc.png` と `<cycle>_<ci>_tasc.csv`）。
# PNGはapex2.pyのTesterが出す運転曲線と同一書式（モード別配色・先行列車の破線・駅線・制限速度の階段線）。

import argparse
import bisect
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")   # PNG保存のみ。既定のqtaggだとSIGABRTで落ちる環境があるため。
import matplotlib.pyplot as plt

import environment2 as e2
from actions import Actions
from train import Train
from runcurve_plot import curve_background_params, draw_curve_background, plot_curve_by_mode

TASC_SUB_DT = 0.01          # TASC作動中の制御周期[s]（train.pyの積分刻みと同じ下限）
HOLD_DT = 0.1               # 引き継ぎ後～パターン到達までの制御周期[s]
DEPARTURE_STATION_INDEX = 11   # 羽前成田（apex2.pyのTesterと同じ）
SPEED_LIMIT_MARGIN = 0.1    # 制限速度・信号現示に対する力行の停止余裕[km/h]
STALL_SPEED = 5.0           # これを下回ったら駅間停車を避けるため力行に切り替える[km/h]


# =====================================================================================
# 制動パターン
# =====================================================================================
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


# =====================================================================================
# 引き継ぎ点の決定
# =====================================================================================
def find_splice_index(spd, pos_km, station_km, dists, speeds, actions=None, min_speed=3.0):
    """TASCへ引き継ぐインデックスと、その理由を返す。

    優先順位:
      1. **DQNの軌跡が制動パターンに最初に到達した点**（reason="pattern"）。
         実車のTASCが作動する瞬間そのもので、引き継ぎが遅れる心配がない。
      2. パターンに一度も到達しない（早めに減速しすぎた）場合は、
         **最後の制動を開始した点**（reason="extend"）。その直前のノッチを延長して
         パターンまで走らせる（ユーザ要件「ブレーキをかける直前の行動を延長させる」）。

    【2026-08-18 修正】(2)は以前「速度が単調減少しなくなる点まで遡る」実装だったが、
    これだと惰行区間をまるごと遡ってしまい、駅の700〜850m手前（速度のピーク）が
    引き継ぎ点になっていた。そこから直前ノッチ（力行）を延長するため、
    ・DQNが実際に走った惰行区間が丸ごと消える
    ・不要な再加速が30秒以上続き、制限速度70km/hを最大5.4km/h超過する
    という不具合が出ていた（run 20260818015134 cycle12950 の実測。ci=2,3で超過）。
    制動の開始点だけを遡るよう修正した。

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

    # --- パターンに達していない → 最後の制動の開始点まで遡る ---
    i = n - 1
    while i > 0 and spd[i] < min_speed:      # 末尾の停車部分を飛ばす
        i -= 1
    if actions is not None:
        j = i
        while j > 0 and int(actions[j - 1]) == int(Actions.deceleration):
            j -= 1
        if j > 0:
            return j, "extend"
    # 行動列が無い場合のフォールバック（旧形式ログ）: 速度が減り始めた点まで遡る
    peak = i
    while peak > 0 and spd[peak - 1] >= spd[peak]:
        peak -= 1
    return peak, "extend"


# =====================================================================================
# TASC区間の再シミュレーション
# =====================================================================================
def _signal_speed(env, t, position, speed):
    """その時刻・位置における許容速度[km/h]（路線の制限速度とCBTC現示の小さい方）。

    envに状態を流し込んで environment2 のプロパティをそのまま読む（定義のずれを避けるため）。
    """
    env.t = t
    env.train.set_states(speed, position)
    fwx = forward_train_at(env, t)
    if fwx is not None and env.fowerd_train is not None:
        env.fowerd_train.set_states(fwx[1], fwx[0])
    return min(env.current_speed_limit, env.cbtc_signal_speed)


def simulate_tasc_tail(env, start_pos_km, start_speed, hold_action, dists, speeds,
                       t0=0.0, coarse_dt=HOLD_DT):
    """start から「パターンに達するまで hold_action を継続 → TASC制動で停止」を再現する。

    延長中は次の2つの安全弁を掛ける（2026-08-18 追加）。
      ・**速度超過の防止**: 力行は「路線の制限速度」と「CBTC現示」の低い方を超えない範囲でのみ許す。
        超えそうなら惰行に落とす（environment2.forbidden_action と同じ考え方）。
        DQNの走行中は環境側がこれを禁止していたのに、後処理側には無かったため超過していた。
      ・**駅間停車の防止**: 惰行を続けると停まってしまう場合（上り勾配で失速）は力行に切り替える。

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
            allowed = _signal_speed(env, t, tr.position, tr.speed)
            if a == Actions.acceleration and tr.speed >= allowed - SPEED_LIMIT_MARGIN:
                a = Actions.coasting          # 速度超過の防止
            elif a != Actions.acceleration and tr.speed < STALL_SPEED and tr.speed < allowed - SPEED_LIMIT_MARGIN:
                a = Actions.acceleration      # 駅間停車の防止
            ts.append(t); pos.append(tr.position); spd.append(tr.speed); act.append(int(a))
            tr.step(a, coarse_dt)
            t += coarse_dt
            if d <= 0:      # パターンに達しないまま駅を通過した場合の保険
                engaged = True
    ts.append(t); pos.append(tr.position); spd.append(tr.speed); act.append(int(Actions.deceleration))
    return ts, pos, spd, act, engage_speed


def overwrite_trajectory(env, times, positions, speeds, actions, modes):
    """走行軌跡の停止部分をTASCの制動パターンで上書きする。

    戻り値: (times, positions, speeds, actions, modes, info)
      info = {"splice": 引き継ぎindex, "reason": "pattern"/"extend",
              "engage_speed": 作動時速度 or None, "stop_error_m": 停止位置誤差[m],
              "hold_action": 延長したノッチ}
    """
    station = env.arrival_station["position"]
    dists, speeds_pat = build_pattern(env)

    b, reason = find_splice_index(speeds, positions, station, dists, speeds_pat, actions)
    hold = int(actions[b - 1]) if b > 0 else int(Actions.coasting)
    if hold == int(Actions.deceleration):
        # 「制動を始める直前の行動」を延長する意図なので、制動なら惰行に読み替える
        hold = int(Actions.coasting)

    t_tail, p_tail, v_tail, a_tail, eng_v = simulate_tasc_tail(
        env, positions[b], speeds[b], hold, dists, speeds_pat, t0=times[b])

    # モードは引き継ぎ時点のものをTASC区間にも適用する（配色を運転曲線と揃えるため）
    mode_at_splice = modes[b] if b < len(modes) else (modes[-1] if modes else "normal")
    new_modes = list(modes[:b]) + [mode_at_splice] * len(t_tail)

    info = {"splice": b, "reason": reason, "engage_speed": eng_v, "hold_action": hold,
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


# =====================================================================================
# 出力（apex2.py の Tester と同一書式）
# =====================================================================================
def save_tasc_outputs(env, dir_name, file_name, ci, header, csv_rows,
                      times, positions, speeds, actions, modes, curve_bg):
    """停止部分をTASCの制動パターンで上書きした運転曲線PNGとCSVログを「TASC制御」フォルダへ保存する。

    apex2.py の Tester からも同じ引数で呼べるようにしてある（撤去済みだが復活可能）。
    戻り値は overwrite_trajectory の info（失敗時は None）。
    """
    try:
        tasc_dir = os.path.join(dir_name, "TASC制御")
        os.makedirs(tasc_dir, exist_ok=True)

        n = min(len(times), len(positions), len(speeds), len(actions), len(modes), len(csv_rows))
        if n < 3:
            return None
        t2, p2, v2, a2, m2, info = overwrite_trajectory(
            env, times[:n], positions[:n], speeds[:n], actions[:n], modes[:n])
        b = info["splice"]

        # --- 運転曲線PNG（Testerの本編と同一書式）---
        draw_curve_background(curve_bg)
        plot_curve_by_mode(plt.plot, p2, v2, m2)
        # 先行列車は時間軸が変わるため新しい時刻で引き直す
        fw = [forward_train_at(env, t) for t in t2]
        fw = [x for x in fw if x is not None]
        if len(fw) > 0:
            plt.plot([x[0] for x in fw], [x[1] for x in fw], "b--", label="Forward Train")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(tasc_dir, f"{file_name}_{ci}_tasc.png"))
        plt.close('all')

        # --- 上書き後のCSVログ（本編と同一スキーマ + source列）---
        # TASC区間の観測値は「envに状態を流し込んで env.raw_state を読む」方式で作る。
        # 手計算で埋めると本編と定義がずれるおそれがあるため（残り時間・CBTC現示・保持時間など）、
        # 本編とまったく同じプロパティ経由で算出する。
        col = {name: i for i, name in enumerate(header)}
        # 引き継ぎ点でのノッチ保持状況を本編のCSV行から引き継ぐ（TASC区間もenvと同じ規則で更新する）
        def _num(row, name, default=0.0):
            try:
                return float(row[col[name]])
            except (KeyError, TypeError, ValueError):
                return default
        hold_time = _num(csv_rows[b], "raw_hold_time")
        pre_act = int(_num(csv_rows[b], "raw_pre_act", int(Actions.coasting)))
        prev_notch_duration = getattr(env, "prev_notch_duration", 0.0)
        prev_notch = getattr(env, "prev_notch", Actions.coasting)

        with open(os.path.join(tasc_dir, f"{file_name}_{ci}_tasc.csv"), "w", newline="") as f_t:
            w = csv.writer(f_t)
            w.writerow([*header, "source"])
            for i in range(b):
                w.writerow([*csv_rows[i], "DQN"])
            for i in range(b, len(t2)):
                if i > b:
                    # env.step と同じ規則でノッチ保持時間を進める（1つ前の行で実行した行動の分）
                    dt = t2[i] - t2[i - 1]
                    if int(a2[i - 1]) == pre_act:
                        hold_time += dt
                    else:
                        prev_notch, prev_notch_duration = Actions(pre_act), hold_time
                        hold_time = dt
                        pre_act = int(a2[i - 1])

                # envへ現在の状態を流し込む（本編の各プロパティがそのまま使えるようにする）
                env.t = t2[i]
                env.train.set_states(v2[i], p2[i])
                env.holding_time = hold_time
                env.pre_action = Actions(pre_act)
                env.prev_notch = prev_notch
                env.prev_notch_duration = prev_notch_duration
                fwx = forward_train_at(env, t2[i])
                if fwx is not None and env.fowerd_train is not None:
                    env.fowerd_train.set_states(fwx[1], fwx[0])

                raw = env.raw_state
                gradient = env.train.front_grades[0]["grade"] if len(env.train.front_grades) > 0 else 0.0

                # NN入力・Q値・報酬はTASC区間では使わないため空欄のままにする
                row = [""] * len(header)
                def put(name, val):
                    if name in col:
                        row[col[name]] = val
                # 生の観測8列（本編と同一の定義・同一の順序）
                for name, val in zip(("raw_speed", "raw_stat_dist", "raw_rem_time", "raw_hold_time",
                                      "raw_pre_act", "raw_stat_dist_2", "raw_fw_dist", "raw_cbtc_signal"), raw):
                    put(name, int(val) if name == "raw_pre_act" else round(float(val), 6))
                # モニター用9列
                put("time", round(t2[i], 3))
                put("position", round(p2[i], 6))
                put("speed_limit", env.current_speed_limit)
                put("fw_position", round(fwx[0], 6) if fwx else "")
                put("fw_speed", round(fwx[1], 4) if fwx else "")
                put("mode", m2[i])
                put("action", int(a2[i]))
                put("gradient", gradient)
                put("fw_dwell_elapsed", round(env.forward_dwell_elapsed, 3))
                w.writerow([*row, "TASC"])
        return info
    except Exception as ex:
        print(f"[TASC] 上書き出力に失敗しました (ci={ci}): {ex}")
        return None


# =====================================================================================
# バッチ処理（学習フォルダ＋cycle番号を指定して全テストケースを処理）
# =====================================================================================
ACTION_NAMES = {0: "惰行", 1: "力行", 2: "制動"}


def load_case(csv_path):
    """Tester出力CSV（新形式）から (header, rows, times, positions, speeds, actions, modes) を読む。"""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [r for r in rd if r]
    col = {name: i for i, name in enumerate(header)}
    need = ("time", "position", "raw_speed", "action", "mode")
    missing = [k for k in need if k not in col]
    if missing:
        raise ValueError(f"新形式のログではありません（列 {missing} が無い）: {csv_path}")
    times = [float(r[col["time"]]) for r in rows]
    positions = [float(r[col["position"]]) for r in rows]
    speeds = [float(r[col["raw_speed"]]) for r in rows]
    actions = [int(float(r[col["action"]])) for r in rows]
    modes = [r[col["mode"]] for r in rows]
    return header, rows, times, positions, speeds, actions, modes


def build_env_from_meta(meta):
    """meta.json のテストケース条件を再現した Environment を返す（reset直後の状態）。"""
    env = e2.Environment(load_reward_predictor=False)
    if meta and meta.get("has_forward_train"):
        env.reset(DEPARTURE_STATION_INDEX, float(meta.get("ego_delay", 0.0)), 1.0,
                  fowerd_train_time_offset=meta.get("headway"),
                  fowerd_train_controls=meta.get("f_train_csv"),
                  forward_train_delay=meta.get("forward_delay"))
    else:
        env.reset(DEPARTURE_STATION_INDEX, float(meta.get("ego_delay", 0.0)) if meta else 0.0, 1.0)
    return env


def process_case(csv_path, run_dir):
    """1テストケースを処理して情報を表示する。戻り値は info（失敗時None）。"""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    m = re.match(r"^(.*)_(\d+)$", base)
    file_name, ci = (m.group(1), int(m.group(2))) if m else (base, 0)

    meta_path = os.path.join(run_dir, f"{base}_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    header, rows, times, positions, speeds, actions, modes = load_case(csv_path)
    env = build_env_from_meta(meta)
    curve_bg = curve_background_params(env)

    info = save_tasc_outputs(env, run_dir + os.sep, file_name, ci, header, rows,
                             times, positions, speeds, actions, modes, curve_bg)
    if info is None:
        return None
    eng = info["engage_speed"]
    print(f"  {base:>14s} : 引き継ぎ={info['reason']:7s} 残{(env.arrival_station['position'] - positions[info['splice']]) * 1000:7.1f}m "
          f"延長ノッチ={ACTION_NAMES.get(info['hold_action'], '?')} "
          f"作動速度={'-' if eng is None else f'{eng:5.1f}'}km/h "
          f"停止位置誤差={info['stop_error_m'] * 100:+6.1f}cm  {meta.get('desc', '')}")
    return info


def latest_cycle(run_dir):
    """runフォルダ内で最も大きい cycle 番号（テストケースCSVが揃っているもの）を返す。"""
    cycles = set()
    for p in glob.glob(os.path.join(run_dir, "*_*.csv")):
        m = re.match(r"^(\d+)_(\d+)$", os.path.splitext(os.path.basename(p))[0])
        if m:
            cycles.add(int(m.group(1)))
    if not cycles:
        raise SystemExit(f"テストケースCSVが見つかりません: {run_dir}")
    return max(cycles)


def main():
    ap = argparse.ArgumentParser(
        description="DQNの走行ログの停止部分をTASCの制動で上書きし、運転曲線PNGとCSVを出力する")
    ap.add_argument("run", nargs="?", help="学習フォルダ（例: data/20260818015134）")
    ap.add_argument("cycle", nargs="?", help="cycle番号（省略時はそのrunの最新cycle）")
    ap.add_argument("--cases", nargs="*", type=int, default=None,
                    help="処理するテストケース番号（既定: そのcycleの全ケース）")
    ap.add_argument("--csv", default=None, help="単一のログCSVだけを処理する")
    args = ap.parse_args()

    if args.csv:
        run_dir = os.path.dirname(os.path.abspath(args.csv))
        print(f"[TASC] 単一ファイル: {args.csv}")
        process_case(args.csv, run_dir)
        return

    if not args.run:
        ap.error("学習フォルダを指定してください（または --csv）")
    run_dir = args.run.rstrip("/\\")
    cycle = int(args.cycle) if args.cycle else latest_cycle(run_dir)

    paths = sorted(glob.glob(os.path.join(run_dir, f"{cycle}_*.csv")),
                   key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1)))
    if args.cases is not None:
        paths = [p for p in paths
                 if int(re.search(r"_(\d+)\.csv$", p).group(1)) in args.cases]
    if not paths:
        raise SystemExit(f"cycle {cycle} のテストケースCSVが見つかりません: {run_dir}")

    print(f"[TASC] {run_dir} cycle={cycle} テストケース{len(paths)}件を処理します")
    errs = []
    for p in paths:
        info = process_case(p, run_dir)
        if info is not None:
            errs.append(abs(info["stop_error_m"]))
    if errs:
        print(f"[TASC] 完了: {len(errs)}件 停止位置誤差 平均{sum(errs) / len(errs) * 100:.1f}cm "
              f"最大{max(errs) * 100:.1f}cm → {os.path.join(run_dir, 'TASC制御')}")


if __name__ == "__main__":
    main()
