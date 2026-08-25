# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田 → 白兎 → 蚕桑）の標準運転曲線ジェネレータ。

既存 `generate_standard_curve.py` の `StandardCurveSolver` を**読み取り専用でimportして再利用**する
（物理モデル・制御周期・制動曲線の逆積分がすべて既存と同一になるため、独自に書き直すより安全）。
本スクリプトが新たに担うのは以下。

  1. **2区間を通しで解き、通算ダイヤ（駅停車30秒を挟む）に載せる**
  2. `config_ymulti.RUNNING_TIMES` の標準運転時間を使う
     （白兎→蚕桑は Station.csv の rt=180秒ではなく **130秒**。input/Station.csv は書き換えない）
  3. 環境・プロンプトから毎ステップ引ける **v_std(区間, 位置) のルックアップ表**を出力する
     （`standard_curve_ymulti/v_std_<駅index>.csv`）。DQNの観測特徴量「標準運転曲線との速度差」に使う。
  4. 2区間を1枚にまとめた運転曲線PNG・時刻-位置ダイアグラムPNG

使い方:
    python standard_curve_ymulti.py                 # 全区間を解いて standard_curve_ymulti/ へ出力
    python standard_curve_ymulti.py --section 0     # 羽前成田→白兎のみ
    python standard_curve_ymulti.py --force         # キャッシュを無視して解き直す
"""
import os
import csv
import json
import argparse
from bisect import bisect_right

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_ymulti as CFG
from actions import Actions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, CFG.STANDARD_CURVE_DIR)

NOTCH_COLOR = {Actions.acceleration: "#d62728", Actions.coasting: "#2ca02c",
               Actions.deceleration: "#1f77b4"}
NOTCH_LABEL = {Actions.acceleration: "力行", Actions.coasting: "惰行",
               Actions.deceleration: "制動"}


def setup_japanese_font():
    """日本語フォントを設定する（見つからなければ英語表記にフォールバック）"""
    import generate_standard_curve as gsc
    return gsc.setup_japanese_font()


# =============================================================================
# 解く
# =============================================================================
def solve_section(section, verbose=True):
    """区間 section（0=羽前成田→白兎, 1=白兎→蚕桑）の標準運転曲線を解く。

    戻り値: (RunResult, V_hold, StandardCurveSolver)
    """
    import generate_standard_curve as gsc
    dep_index = CFG.STATION_INDICES[section]
    target_time = CFG.RUNNING_TIMES[section]
    solver = gsc.StandardCurveSolver(departure_index=dep_index, target_time=target_time)
    result, v_hold, _ = solver.optimize(verbose=False)
    if verbose:
        name = f"{CFG.STATION_NAMES_JA[dep_index]}→{CFG.STATION_NAMES_JA[CFG.STATION_INDICES[section+1]]}"
        print(f"[区間{section}] {name} {solver.distance_km*1000:.0f}m / 標準運転時間 {target_time:.0f}s")
        print(f"    到着 {result.time:.2f}s / 停止位置誤差 {result.stop_error_m:+.3f} m / "
              f"定速保持速度 {v_hold:.2f} km/h / 最高 {result.max_speed:.1f} km/h")
        print(f"    力行仕事 {result.energy_kwh:.3f} kWh / ノッチ切替 {result.notch_changes} 回")
        print(f"    {result.pattern_text()}")
    return result, v_hold, solver


# =============================================================================
# v_std ルックアップ表
# =============================================================================
def write_v_std_table(section, result, solver):
    """位置[km] → 標準運転曲線の速度[km/h]・経過時刻[s] の表を書き出す。

    環境（environment_ymulti.py）が毎ステップ参照するため、位置について単調増加に整理し、
    1m 刻みへ再サンプリングしておく（bisect で O(log n) 参照できる）。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    dep_index = CFG.STATION_INDICES[section]
    x0 = solver.start_position
    x1 = solver.station_position

    # rows は制御周期ごと。位置が単調増加になるよう整理する
    pts = [(r.position, r.speed, r.t) for r in result.rows]
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    ts = [p[2] for p in pts]

    path = os.path.join(OUT_DIR, f"v_std_{dep_index}.csv")
    n = int(round((x1 - x0) * 1000.0)) + 1     # 1m刻み
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "v_std", "t_std"])
        for i in range(n):
            x = x0 + i / 1000.0
            j = bisect_right(xs, x) - 1
            if j < 0:
                v, t = vs[0], ts[0]
            elif j >= len(xs) - 1:
                v, t = vs[-1], ts[-1]
            else:
                span = xs[j + 1] - xs[j]
                r = 0.0 if span <= 0 else (x - xs[j]) / span
                v = vs[j] + r * (vs[j + 1] - vs[j])
                t = ts[j] + r * (ts[j + 1] - ts[j])
            w.writerow([f"{x:.6f}", f"{v:.4f}", f"{t:.3f}"])
    return path


class VStdTable:
    """標準運転曲線の速度・通過時刻を位置から引く（環境・プロンプト・可視化から使う）。

    表が無い場合は `standard_curve_ymulti.py` を実行するよう促す例外を出す
    （黙って0を返すと「標準からの逸脱量」が常に現在速度そのものになり、学習が静かに壊れるため）。
    """

    def __init__(self):
        self.sections = []
        for section in range(len(CFG.RUNNING_TIMES)):
            dep_index = CFG.STATION_INDICES[section]
            path = os.path.join(OUT_DIR, f"v_std_{dep_index}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"標準運転曲線の表が見つかりません: {path}\n"
                    f"  先に `python standard_curve_ymulti.py` を実行してください。")
            xs, vs, ts = [], [], []
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    xs.append(float(row["position"]))
                    vs.append(float(row["v_std"]))
                    ts.append(float(row["t_std"]))
            self.sections.append({"x": xs, "v": vs, "t": ts})

    def _lookup(self, section, position, key):
        s = self.sections[section]
        xs = s["x"]
        arr = s[key]
        if position <= xs[0]:
            return arr[0]
        if position >= xs[-1]:
            return arr[-1]
        i = bisect_right(xs, position) - 1
        span = xs[i + 1] - xs[i]
        r = 0.0 if span <= 0 else (position - xs[i]) / span
        return arr[i] + r * (arr[i + 1] - arr[i])

    def v_std(self, section, position):
        """区間 section の位置 position[km] における標準運転曲線の速度[km/h]"""
        return self._lookup(section, position, "v")

    def t_std(self, section, position):
        """区間 section の位置 position[km] における標準運転曲線の経過時刻[s]（区間発車からの秒数）"""
        return self._lookup(section, position, "t")


# =============================================================================
# 出力
# =============================================================================
def write_section_csv(section, result, solver):
    """区間の走行ログCSV（time, position, speed, action）"""
    dep_index = CFG.STATION_INDICES[section]
    path = os.path.join(OUT_DIR, f"standard_curve_{dep_index}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "position", "speed", "action"])
        for r in result.rows:
            w.writerow([f"{r.t:.3f}", f"{r.position:.6f}", f"{r.speed:.4f}", str(Actions(r.action))])
    return path


def write_meta(results):
    """全区間をまとめた meta.json"""
    arr = CFG.scheduled_arrival_times()
    dep = CFG.scheduled_departure_times()
    meta = {
        "line": "山形鉄道フラワー長井線（羽前成田→白兎→蚕桑）",
        "notches": 3,
        "station_indices": CFG.STATION_INDICES,
        "station_names": [CFG.STATION_NAMES_JA[i] for i in CFG.STATION_INDICES],
        "running_times": CFG.RUNNING_TIMES,
        "running_times_station_csv": CFG.RUNNING_TIMES_CSV,
        "std_dwell": CFG.STD_DWELL,
        "scheduled_arrival": arr,
        "scheduled_departure": dep,
        "total_scheduled_time": CFG.total_scheduled_time(),
        "sections": [],
    }
    for section, (res, vh, solver) in enumerate(results):
        meta["sections"].append({
            "section": section,
            "departure_index": CFG.STATION_INDICES[section],
            "from": CFG.STATION_NAMES_JA[CFG.STATION_INDICES[section]],
            "to": CFG.STATION_NAMES_JA[CFG.STATION_INDICES[section + 1]],
            "start_position": solver.start_position,
            "end_position": solver.station_position,
            "distance_m": solver.distance_km * 1000.0,
            "target_time": CFG.RUNNING_TIMES[section],
            "arrival_time": res.time,
            "stop_error_m": res.stop_error_m,
            "v_hold": vh,
            "max_speed": res.max_speed,
            "energy_kwh": res.energy_kwh,
            "notch_changes": res.notch_changes,
            "coast_position": res.coast_position,
            "brake_position": res.brake_position,
            "brake_speed": res.brake_speed,
            "pattern": res.pattern_text(),
        })
    path = os.path.join(OUT_DIR, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def plot_all(results, jp):
    """2区間を1枚にまとめた運転曲線PNGと、時刻-位置ダイアグラムPNG"""
    L = (lambda ja, en: ja if jp else en)
    # ---- 運転曲線（位置-速度） ----
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for section, (res, vh, solver) in enumerate(results):
        xs = [r.position for r in res.rows]
        vs = [r.speed for r in res.rows]
        acts = [Actions(r.action) for r in res.rows]
        seen = set()
        i = 0
        while i < len(xs) - 1:
            a = acts[i]
            j = i
            while j < len(xs) - 1 and acts[j] == a:
                j += 1
            lbl = None
            if a not in seen:
                seen.add(a)
                lbl = L(NOTCH_LABEL[a], a.name)
            ax.plot(xs[i:j + 1], vs[i:j + 1], color=NOTCH_COLOR[a], lw=1.6, label=lbl)
            i = j
        # 制限速度の階段線
        for sec in solver.lookup.limit_sections(solver.start_position, solver.station_position):
            ax.plot([sec["start"], sec["start"] + sec["distance"]],
                    [sec["speed_limit"], sec["speed_limit"]], "k-", lw=1)
    for k, idx in enumerate(CFG.STATION_INDICES):
        pos = results[k][2].start_position if k < len(results) else results[-1][2].station_position
        ax.axvline(pos, color="k", lw=2.5)
        ax.text(pos, 74, CFG.STATION_NAMES_JA[idx] if jp else str(idx),
                ha="center", va="bottom", fontsize=11)
    ax.set_xlabel(L("位置 [km]", "Position [km]"))
    ax.set_ylabel(L("速度 [km/h]", "Speed [km/h]"))
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(L("標準運転曲線（羽前成田→白兎→蚕桑・3ノッチ・省エネ最適）",
                   "Standard run curve (3 notches)"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "standard_curve_all.png")
    fig.savefig(p1)
    plt.close(fig)

    # ---- ダイアグラム（時刻-位置。停車30秒を挟んで通算） ----
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    t_off = 0.0
    tx, ty = [], []
    for section, (res, vh, solver) in enumerate(results):
        for r in res.rows:
            tx.append(t_off + r.t)
            ty.append(r.position)
        t_off += res.time
        if section < len(results) - 1:
            tx.append(t_off + CFG.STD_DWELL)
            ty.append(solver.station_position)
            t_off += CFG.STD_DWELL
    ax.plot(tx, ty, color="#d62728", lw=1.8, label=L("標準運転曲線", "Standard"))
    for k, idx in enumerate(CFG.STATION_INDICES):
        pos = results[k][2].start_position if k < len(results) else results[-1][2].station_position
        ax.axhline(pos, color="k", lw=1, ls="--", alpha=0.6)
        ax.text(0, pos, CFG.STATION_NAMES_JA[idx] if jp else str(idx),
                ha="left", va="bottom", fontsize=10)
    ax.set_xlabel(L("時刻 [s]（羽前成田発車を0とする）", "Time [s]"))
    ax.set_ylabel(L("位置 [km]", "Position [km]"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(L(f"標準ダイヤ（通算 {CFG.total_scheduled_time():.0f} 秒）",
                   f"Schedule ({CFG.total_scheduled_time():.0f}s)"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "standard_diagram.png")
    fig.savefig(p2)
    plt.close(fig)
    return p1, p2


def main(argv=None):
    ap = argparse.ArgumentParser(description="複数駅間版（羽前成田→白兎→蚕桑）の標準運転曲線を生成する")
    ap.add_argument("--section", type=int, default=None, help="区間index（0=羽前成田→白兎, 1=白兎→蚕桑）。省略で全区間")
    ap.add_argument("--no-plot", action="store_true", help="PNGを出力しない")
    a = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    jp = setup_japanese_font()
    sections = [a.section] if a.section is not None else list(range(len(CFG.RUNNING_TIMES)))

    results = []
    for section in sections:
        res, vh, solver = solve_section(section)
        write_section_csv(section, res, solver)
        p = write_v_std_table(section, res, solver)
        print(f"    → v_std表: {p}")
        results.append((res, vh, solver))

    if a.section is None:
        print(f"    → meta: {write_meta(results)}")
        if not a.no_plot:
            p1, p2 = plot_all(results, jp)
            print(f"    → PNG: {p1}")
            print(f"    → PNG: {p2}")
        total = sum(r[0].time for r in results) + CFG.STD_DWELL * (len(results) - 1)
        print(f"\n通算所要（標準停車{CFG.STD_DWELL:.0f}秒込み）: {total:.1f} 秒 "
              f"（累積標準ダイヤ {CFG.total_scheduled_time():.0f} 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
