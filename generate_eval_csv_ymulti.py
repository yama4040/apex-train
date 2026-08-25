# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）のLLM評価用 走行ログCSV生成。

既存 `評価用csv/` は**一切変更しない**。出力先は `評価用csv_Yamagata/` で、
既存データセットとは**絶対に混ぜない**（プロンプト世代・列構成・モード定義が異なる。
プロンプト世代の混在は過去にラベル矛盾の実害を出している）。

【設計上の要点】走行ログは `environment_ymulti.EnvironmentYMulti` を**実際に走らせて**作る。
生の状態辞書（`env.last_raw_state`）をそのまま書き出すので、
**LLMが評価する状態と、RL実行時に報酬NNが見る状態が構造的に一致する**。
別実装で作ると特徴量の定義がずれ、蒸留したNNが実行時に別物を見ることになる。

方策は「正例（望ましい運転）」と「負例（典型的な失敗）」を明示的に定義する。
LLMに評価させる以上、良い操作と悪い操作の両方が十分な数だけデータに含まれている必要がある。

使い方:
    python generate_eval_csv_ymulti.py                  # 既定の全方策×シナリオ
    python generate_eval_csv_ymulti.py --rows 3000      # 目標行数を指定
    python generate_eval_csv_ymulti.py --list           # 方策とシナリオの一覧
"""
import os
import csv
import json
import random
import argparse

import config_ymulti as CFG
import reward_features_ymulti as rf
import required_speed_ymulti as rsm
from brake_curve_ymulti import get_brake_curve
from environment_ymulti import EnvironmentYMulti
from actions import Actions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, CFG.EVAL_CSV_DIR)

_META = json.load(open(os.path.join(BASE_DIR, CFG.STANDARD_CURVE_DIR, "meta.json"), encoding="utf-8"))
V_HOLD = [s["v_hold"] for s in _META["sections"]]
X_COAST = [s["coast_position"] for s in _META["sections"]]

ACC, CST, BRK = int(Actions.acceleration), int(Actions.coasting), int(Actions.deceleration)


# =============================================================================
# 走行方策
# =============================================================================
def _brake_needed(env):
    return env.speed >= get_brake_curve(env.arrival_station["position"]).speed_at(env.position)


def _signal_guard(env, margin=2.0):
    return env.speed > env.cbtc_signal_speed - margin


def _restart(env):
    """信号開通後・停止後の再起動が必要か（ちんたら運転の回避）"""
    return env.speed < 5.0 and env.cbtc_signal_speed > 5.0 and env.station_remaining_distance > 0.03


def p_standard(env, prm):
    """【正例】標準運転曲線の再現。力行 → 惰行 → 制動。"""
    if _brake_needed(env):
        return BRK
    if _signal_guard(env):
        return BRK
    if _restart(env):
        return ACC
    if env.position < X_COAST[env.section] and env.speed < V_HOLD[env.section]:
        return ACC
    return CST


def p_coast_shift(env, prm):
    """【正例】惰行開始点を前後にずらした運転（標準の近傍・許容範囲の揺らぎ）。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if _restart(env):
        return ACC
    shift = prm.get("coast_shift", 0.0)
    if env.position < X_COAST[env.section] + shift and env.speed < V_HOLD[env.section]:
        return ACC
    if not rsm.coast_reachable(env.position, env.speed, env.arrival_station["position"]):
        return ACC          # 上り勾配で失速しそうなら力行で立て直す（駅間停車の回避）
    return CST


def p_anti_mid_stop(env, prm):
    """【正例】先行に塞がれている局面で target_speed_no_stop に沿って早めに惰行する。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if _restart(env):
        return ACC
    raw = getattr(env, "last_raw_state", None)
    target = raw["target_speed_no_stop"] if raw else V_HOLD[env.section]
    if env.speed < target - 3.0:
        return ACC
    if env.speed > target + 1.0:
        return CST
    return CST


def p_delay_recovery(env, prm):
    """【正例】遅延している局面で制限直下まで力行して回復する。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if env.speed < env.current_speed_limit - 2.0:
        return ACC
    return CST


def p_early_coast(env, prm):
    """【負例】早すぎる惰行。上り勾配で失速し駅間停車に至る。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if env.speed < prm.get("coast_at", 35.0) and env.t - env.section_start_t < 25.0:
        return ACC
    return CST


def p_over_power(env, prm):
    """【負例】過剰力行。制限直下まで加速し続け、駅手前で強い制動になる。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if env.speed < env.current_speed_limit - 1.0:
        return ACC
    return CST


def p_useless_brake(env, prm):
    """【負例】理由のない制動を挟む運転。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    if _restart(env):
        return ACC
    if env.position < X_COAST[env.section] and env.speed < V_HOLD[env.section]:
        # 一定間隔で意味のない制動を入れる
        return BRK if int(env.t) % 17 in (0, 1, 2) else ACC
    return CST


def p_sawtooth(env, prm):
    """【負例】ノコギリ運転。数秒ごとにノッチを反転させる。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    period = prm.get("period", 3)
    return ACC if (int(env.t) // period) % 2 == 0 else BRK


def p_late_brake(env, prm):
    """【負例】制動開始が遅い（オーバーラン方向）。"""
    d = env.station_remaining_distance * 1000.0
    req = rsm.station_stop_distance_m(env.speed, env.arrival_station["position"])
    if d - req <= prm.get("late_m", -12.0):
        return BRK
    if _signal_guard(env):
        return BRK
    if env.position < X_COAST[env.section] and env.speed < V_HOLD[env.section]:
        return ACC
    return CST


def p_early_brake(env, prm):
    """【負例】制動開始が早すぎる（駅手前で止まる方向）。"""
    if _signal_guard(env):
        return BRK
    if env.station_remaining_distance * 1000.0 <= prm.get("early_m", 500.0):
        return BRK
    if env.position < X_COAST[env.section] and env.speed < V_HOLD[env.section]:
        return ACC
    return CST


def p_creep(env, prm):
    """【負例】ちんたら運転。極低速で時間を稼ぐ。"""
    if _brake_needed(env) or _signal_guard(env):
        return BRK
    return ACC if env.speed < prm.get("creep_v", 12.0) else CST


RUNNING_POLICIES = {
    "std":            (p_standard, "【正例】標準運転曲線の再現"),
    "coast_early":    (p_coast_shift, "【正例】惰行開始点をやや手前に"),
    "coast_late":     (p_coast_shift, "【正例】惰行開始点をやや奥に"),
    "anti_mid_stop":  (p_anti_mid_stop, "【正例】先行の塞ぎに合わせて早めに惰行"),
    "delay_recovery": (p_delay_recovery, "【正例】遅延回復の力行"),
    "early_coast":    (p_early_coast, "【負例】早すぎる惰行（上り勾配で失速）"),
    "over_power":     (p_over_power, "【負例】過剰力行"),
    "useless_brake":  (p_useless_brake, "【負例】無駄な制動"),
    "sawtooth":       (p_sawtooth, "【負例】ノコギリ運転"),
    "late_brake":     (p_late_brake, "【負例】制動開始が遅い"),
    "early_brake":    (p_early_brake, "【負例】制動開始が早すぎる"),
    "creep":          (p_creep, "【負例】ちんたら運転"),
}

# =============================================================================
# 停車中の発車判断の方策
# =============================================================================
def dwell_action(env, kind):
    """駅停車中の行動。kind ごとに発車タイミングを変える。"""
    e = env.dwell_elapsed + CFG.DWELL_TIME_STEP
    if kind == "on_time":                 # 標準停車で発車
        return ACC if e >= CFG.STD_DWELL else BRK
    if kind == "wait_for_forward":        # 先行がクリアするまで待って発車（塞ぎ時の正例）
        raw = getattr(env, "last_raw_state", None)
        clear = raw["forward_clear_remaining_time"] if raw else 0.0
        reach = raw["time_to_stop_limit"] if raw else 0.0
        if e < CFG.STD_DWELL:
            return BRK
        return BRK if clear > reach else ACC
    if kind == "hold_60":                 # 理由の有無によらず60秒待つ
        return ACC if e >= 60.0 else BRK
    if kind == "hold_120":
        return ACC if e >= 120.0 else BRK
    if kind == "hold_200":
        return ACC if e >= 200.0 else BRK
    return ACC if e >= CFG.STD_DWELL else BRK


DWELL_KINDS = ["on_time", "wait_for_forward", "hold_60", "hold_120", "hold_200"]


# =============================================================================
# シナリオ
# =============================================================================
def build_scenarios():
    """(名前, reset引数) のリスト。先行の塞ぎ・自列車遅延を網羅する。"""
    sc = [("solo", {}),
          ("solo_delay20", {"delay": 20.0}),
          ("solo_delay60", {"delay": 60.0})]
    for hw in (120.0, 90.0, 60.0):
        for coast in (65, 50):
            for b, c in ((30, 30), (30, 120), (30, 180), (60, 60), (45, 120)):
                sc.append((f"hw{int(hw)}_v{coast}_b{b}_c{c}",
                           {"f_train_csv": CFG.f_train_csv(coast, b, c), "headway": hw}))
    return sc


# =============================================================================
# 走行の実行
# =============================================================================
def run_episode(env, policy, prm, dwell_kind, reset_kw, max_steps=9000):
    """1エピソードを走らせ、各ステップの生状態辞書を集めて返す。"""
    env.reset(**reset_kw)
    rows = []
    done = False
    n = 0
    while not done and n < max_steps:
        a = dwell_action(env, dwell_kind) if env.is_dwelling else policy(env, prm)
        forb = env.forbidden_action
        if forb[a]:
            allowed = [i for i, f in enumerate(forb) if not f]
            # 停車中は待機(制動)を優先、走行中は惰行を優先して代替する
            a = BRK if (env.is_dwelling and BRK in allowed) else allowed[0]
        _s, _r, done = env.step(a)
        raw = dict(env.last_raw_state)
        rows.append(raw)
        n += 1
    return rows, {"goal": env.goal_reached, "failed": env.failed,
                  "reason": env.fail_reason, "t": env.t,
                  "dwell": env.dwell_log[0]["dwell"] if env.dwell_log else None}


# 1エピソードから残す駅停車中の行数（等間隔で抽出）。
# 停車は1秒刻みなので標準30秒でも30行、最大停車なら300行になり、そのまま入れると
# データセットの大半が停車行になってしまう。逆に減らしすぎると発車判断が学習できない。
DWELL_KEEP_MIN = 2
DWELL_KEEP_MAX = 12
DWELL_KEEP_DIV = 12      # 停車行数 ÷ この値 が目安（30秒→2行 / 300秒→12行）


def subsample(rows, keep_running, rng):
    """走行中・停車中の行をそれぞれ等間隔で間引く。

    ラベル不均衡は蒸留NNの校正を直接壊すので、走行と停車の比率をここで作り込む
    （計画書 §5.2 フェーズ3の注意点）。停車行は走行行の3割程度を目安にする。
    """
    out = []
    run_rows = [r for r in rows if r["is_dwelling"] < 0.5]
    dwell_rows = [r for r in rows if r["is_dwelling"] >= 0.5]
    if run_rows:
        step = max(1, int(len(run_rows) / max(1, keep_running)))
        out.extend(run_rows[::step])
    if dwell_rows:
        keep = min(DWELL_KEEP_MAX, max(DWELL_KEEP_MIN, len(dwell_rows) // DWELL_KEEP_DIV))
        step = max(1, int(len(dwell_rows) / keep))
        out.extend(dwell_rows[::step])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="複数駅間版のLLM評価用 走行ログCSVを生成する")
    ap.add_argument("--rows", type=int, default=4000, help="目標行数の目安")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--list", action="store_true", help="方策とシナリオの一覧を表示して終了")
    a = ap.parse_args(argv)

    scenarios = build_scenarios()
    if a.list:
        print("=== 走行方策 ===")
        for k, (_fn, desc) in RUNNING_POLICIES.items():
            print(f"  {k:<16} {desc}")
        print("\n=== 停車中の発車判断 ===")
        for k in DWELL_KINDS:
            print(f"  {k}")
        print(f"\n=== シナリオ（{len(scenarios)}種）===")
        for n, _ in scenarios:
            print(f"  {n}")
        return 0

    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    env = EnvironmentYMulti(load_reward_predictor=False, reward_mode="rule")

    # 方策×シナリオの組み合わせを作る。正例は全シナリオ、負例は代表シナリオに絞る。
    jobs = []
    solo = [s for s in scenarios if s[0].startswith("solo")]
    blocked = [s for s in scenarios if not s[0].startswith("solo")]
    rep = blocked[::3]        # 負例用の代表シナリオ

    for name, (fn, _d) in RUNNING_POLICIES.items():
        prm = {}
        if name == "coast_early":
            prm = {"coast_shift": -0.15}
        elif name == "coast_late":
            prm = {"coast_shift": +0.10}
        target = scenarios if name in ("std", "anti_mid_stop") else (solo + rep)
        for sname, kw in target:
            # 停車中の方策: 正例走行には塞ぎに応じた待機、負例には定時発車を割り当てる
            kinds = (["on_time", "wait_for_forward"] if name in ("std", "anti_mid_stop")
                     else ["on_time"])
            for kind in kinds:
                jobs.append((name, fn, prm, kind, sname, kw))

    # 停車判断そのものを評価させるための追加ジョブ（標準走行 × 各待機パターン）
    for sname, kw in scenarios:
        for kind in ("hold_60", "hold_120", "hold_200"):
            jobs.append(("std", RUNNING_POLICIES["std"][0], {}, kind, sname, kw))

    keep_running = max(6, int(a.rows * 0.72 / max(1, len(jobs))))
    print(f"=== LLM評価用CSVの生成 ===")
    print(f"  方策 {len(RUNNING_POLICIES)} 種 × シナリオ {len(scenarios)} 種 → {len(jobs)} エピソード")
    print(f"  1エピソードあたり 走行 {keep_running} 行 / 停車中 {DWELL_KEEP_MIN}〜{DWELL_KEEP_MAX} 行 を目安に間引き")

    all_rows = []
    stats = {}
    for i, (pname, fn, prm, kind, sname, kw) in enumerate(jobs, 1):
        rows, info = run_episode(env, fn, prm, kind, kw)
        run_id = f"{pname}__{kind}__{sname}"
        for r in rows:
            r["run_id"] = run_id
        picked = subsample(rows, keep_running, rng)
        all_rows.extend(picked)
        stats[pname] = stats.get(pname, 0) + len(picked)
        if i % 25 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} エピソード  累計 {len(all_rows)} 行")

    # ---- 書き出し（方策ごとに1ファイル）----
    by_policy = {}
    for r in all_rows:
        by_policy.setdefault(r["run_id"].split("__")[0], []).append(r)

    for pname, rows in sorted(by_policy.items()):
        path = os.path.join(OUT_DIR, f"ymulti_{pname}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rf.RAW_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = {k: r.get(k, "") for k in rf.RAW_COLS}
                for k in rf.LLM_OUTPUT_COLS:
                    out[k] = ""            # mode / reward / reason はLLMが埋める
                for k, v in out.items():
                    if isinstance(v, float):
                        out[k] = f"{v:.4f}"
                w.writerow(out)
        print(f"  → {path}  {len(rows)} 行")

    n_dwell = sum(1 for r in all_rows if float(r["is_dwelling"]) >= 0.5)
    print(f"\n合計 {len(all_rows)} 行（うち駅停車中 {n_dwell} 行 = {n_dwell/len(all_rows)*100:.1f}%）")
    print(f"  方策別: " + " / ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
