# -*- coding: utf-8 -*-
"""
複数駅間版のLLM評価用 走行ログCSV 生成（設計: docs_複数駅間最適化_計画.md §14.3f）

`prompt_multi.py` が要求する特徴量（FEATURE_KEYS）を全て埋めた走行ログを出力する。
このCSVを `evaluate_csv_with_llm_multi.py` がLLMへ送り、mode/reward/reason を付けて
`評価済ログ_Tozai/` へ、さらに `train_reward_csv_direct_Tozai/` へ集約する。

既存の `評価用csv/` 系（羽前成田線・3ノッチ・旧プロンプト）とは**別系統**であり、
混在させてはならない（プロンプト世代の混在はラベル矛盾を生む。§14.3f(2)）。

【今回の対象】通常運転モード（先行列車なし・後続なし・遅延なし）
  正例: 標準運転曲線に沿う運転とその周辺
  負例: 早すぎる惰行／過剰力行／無駄な制動／ノコギリ／制動開始の遅早／ちんたら運転／下り勾配での力行
"""
import os
import csv as csvmod
import argparse

import line_config as LC
from actions_multi import FROM_CODE
from required_speed_multi import SpeedProfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "評価用csv_Tozai")

COLUMNS = [
    # --- 追跡用（プロンプトには渡さない） ---
    "time", "run_id", "section", "position_km",
    # --- 操作 ---
    "phase", "current_notch", "holding_time", "prev_notch", "prev_notch_duration", "notch_jump",
    # --- 速度と基準 ---
    "current_speed", "atc_now", "signal_speed", "v_ceiling", "v_target",
    "band_upper", "band_lower", "band_notch_pair", "hold_notch",
    "v_std", "v_std_deviation", "schedule_speed", "section_cap", "required_speed",
    "target_speed_no_stop", "target_speed_spacing",
    # --- 停止位置 ---
    "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delta_stop",
    # --- 勾配 ---
    "current_gradient", "coast_accel", "power_accel", "next_gradient_info", "limit_drop_ahead",
    # --- 運行 ---
    "delay", "total_delay", "stations_remaining", "total_remaining_time",
    # --- 先行・後続 ---
    "forward_info", "forward_train_delay", "forward_departed_next", "standard_headway",
    "forward_clear_remaining_time", "forward_observed_delay",
    "backward_info", "backward_distance", "backward_speed", "backward_delay",
    # --- 停車 ---
    "is_dwelling", "dwell_elapsed", "dwell_min", "dwell_max",
    # --- LLMが埋める ---
    "mode", "reward", "reason",
]

POWER_SIDE = {"P1", "P2"}
BRAKE_SIDE = {"B1", "B2"}


def _phase(sp, x, v, t_since_dep, dist_m):
    if v <= 0.5 and dist_m <= 10.0:
        return "駅停車完了（速度0km/h）"
    if t_since_dep <= 20.0:
        return "駅出発直後の加速フェーズ（20秒以内）"
    if dist_m <= 400.0:
        return "次駅への減速フェーズ（駅手前400m以内）"
    d, nv = sp.next_limit_drop(x)
    if d is not None and d <= 500.0:
        return "現示低下点への接近フェーズ（500m以内に現示の低下あり）"
    return "巡航フェーズ（駅間走行中）"


def _grad_info(sp, x):
    g0 = sp.track.grade(x)
    for i in range(1, 201):
        xx = x + i * 0.005
        if xx > sp.x1:
            break
        g = sp.track.grade(xx)
        if abs(g - g0) > 0.1:
            return f"{i*5}m先で勾配が{g0:+.1f}‰から{g:+.1f}‰に変わる"
    return f"当面{g0:+.1f}‰が続く"


def run_policy(sp, policy, run_id, section, log_dt=1.0, near_dt=0.5, max_time=400.0):
    """policy(ctx) -> ノッチ記号 を与えて1駅間を走らせ、ログ行を返す。"""
    x, v, t = sp.x0, 0.0, 0.0
    notch, hold, prev_notch, prev_dur = "P1", 0.0, None, 0.0
    rows, next_log = [], 0.0
    while t < max_time:
        dist_m = (sp.x1 - x) * 1000.0
        ctx = dict(sp=sp, x=x, v=v, t=t, dist_m=dist_m, notch=notch, hold=hold)
        nn = policy(ctx)
        if nn != notch:
            prev_notch, prev_dur = notch, hold
            jump = sp.notch_jump(notch, nn, v, x)
            notch, hold = nn, 0.0
        else:
            jump = 0.0
        hold += LC.SUB_DT
        # --- 記録 ---
        if t >= next_log - 1e-9:
            tg = sp.targets(v, x, sp.target_time - t, mode="normal")
            rows.append({
                "time": round(t, 2), "run_id": run_id, "section": section,
                "position_km": round(x, 6),
                "phase": _phase(sp, x, v, t, dist_m),
                "current_notch": LC.NOTCH_LABEL_JA[notch], "holding_time": round(hold, 1),
                "prev_notch": LC.NOTCH_LABEL_JA[prev_notch] if prev_notch else "なし（または停止）",
                "prev_notch_duration": round(prev_dur, 1),
                "notch_jump": round(sp.notch_jump(prev_notch, notch, v, x), 2),
                "current_speed": round(v, 2),
                "atc_now": round(tg["atc_now"], 1), "signal_speed": round(tg["signal_speed"], 1),
                "v_ceiling": round(tg["v_ceiling"], 1), "v_target": round(tg["v_target"], 1),
                "band_upper": round(tg["band_upper"], 1), "band_lower": round(tg["band_lower"], 1),
                "band_notch_pair": tg["band_notch_pair"], "hold_notch": tg["hold_notch"],
                "v_std": round(tg["v_std"], 1) if tg["v_std"] is not None else "",
                "v_std_deviation": round(tg["v_std_deviation"], 1) if tg["v_std_deviation"] is not None else "",
                "schedule_speed": (round(tg["schedule_speed"], 1)
                                   if tg["schedule_speed"] is not None else ""),
                "section_cap": round(tg["section_cap"], 1),
                "required_speed": round(tg["required_speed"], 1),
                "target_speed_no_stop": round(tg["target_speed_no_stop"], 1),
                "target_speed_spacing": "",
                "dist_to_next_station": round(dist_m, 1),
                "time_to_next_station": round(max(0.0, sp.target_time - t), 1),
                "req_stop_dist": round(tg["req_stop_dist"], 2),
                "delta_stop": round(dist_m - tg["req_stop_dist"], 2),
                "current_gradient": round(sp.track.grade(x), 1),
                "coast_accel": round(tg["coast_accel"], 3),
                "power_accel": round(tg["power_accel"], 3),
                "next_gradient_info": _grad_info(sp, x),
                "limit_drop_ahead": tg["limit_drop_ahead"],
                "delay": round(max(0.0, t - sp.target_time), 1),
                "total_delay": round(max(0.0, t - sp.target_time), 1),
                "stations_remaining": 1, "total_remaining_time": round(max(0.0, sp.target_time - t), 1),
                "forward_info": "先行列車なし", "forward_train_delay": 0,
                "forward_departed_next": "", "standard_headway": 0,
                "forward_clear_remaining_time": 0, "forward_observed_delay": 0,
                "backward_info": "後続列車なし", "backward_distance": "該当なし",
                "backward_speed": "該当なし", "backward_delay": 0,
                "is_dwelling": 0, "dwell_elapsed": 0, "dwell_min": 30, "dwell_max": 240,
                "mode": "", "reward": "", "reason": "",
            })
            next_log = t + (near_dt if dist_m <= 150.0 else log_dt)
        a = sp.notch_accel(notch, v, x)
        nv = max(0.0, v + a * LC.SUB_DT)
        x += (v / 3600.0) * LC.SUB_DT + (a / 3600.0) * (LC.SUB_DT ** 2)
        v = nv
        t += LC.SUB_DT
        if x >= sp.x1 + 0.010:
            break
        if v <= 1e-6 and t > 2.0:
            break
    return rows


# =========================================================================
# 運転方策
# =========================================================================
def _brake_if_needed(ctx):
    return ctx["v"] >= ctx["sp"].station_brake_speed(ctx["x"]) - 1e-9


def p_standard(v_coast, ceil_margin=0.3):
    """正例: 力行 → 惰行ポイントで惰行 → 天井に迫れば弱制動 → 制動曲線で停止"""
    st = {"coasting": False}
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        ceil = sp.v_ceiling(x)
        if not st["coasting"]:
            if v >= min(v_coast, ceil) - ceil_margin:
                st["coasting"] = True
            else:
                return "P1"
        if v > ceil - 0.3 and sp.coast_accel(v, x) > 0:
            return "B2"
        if sp.track.grade(x) > 1.0 and v < 35.0:
            return "P2"
        return "C"
    return f


def p_track_vstd():
    """正例: 標準運転曲線 v_std を追従する（帯運転）"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        tgt = sp.v_std(x)
        if tgt is None:
            return "C"
        # 発車直後は v_std 自体がまだ低い（曲線も加速中）。追従だけだと動き出せないので、
        # 立ち上がりは素直に力行する。
        if v < 10.0 or v < tgt - 2.0:
            return "P1"
        up, dn, hold, a_hold = sp.hold_notches(v, x)
        if v > tgt + 2.0:
            return dn
        return hold if abs(a_hold) < 0.05 else (up if v < tgt else dn)
    return f


def p_coast_too_early(x_coast_m):
    """負例: 早すぎる惰行（速度が乗り切らず遅延）"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        return "C" if (x - sp.x0) * 1000.0 >= x_coast_m else "P1"
    return f


def p_over_power():
    """負例: 現示一杯まで力行し続ける（惰行を使わない＝省エネ性が悪い）"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        return "P1" if v < sp.v_ceiling(x) - 0.3 else "C"
    return f


def p_useless_brake(x_from_m, x_to_m, v_coast=60.0):
    """負例: 巡航中に理由なく制動して速度を捨て、その後再力行する"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        d = (x - sp.x0) * 1000.0
        if _brake_if_needed(c):
            return "B1"
        if x_from_m <= d < x_to_m:
            return "B1"
        return "P1" if v < v_coast - 0.3 else "C"
    return f


def p_sawtooth(period=3.0, v_coast=60.0):
    """負例: 短周期で力行⇔惰行を反転させるノコギリ運転"""
    def f(c):
        sp, x, v, t = c["sp"], c["x"], c["v"], c["t"]
        if _brake_if_needed(c):
            return "B1"
        if v < 25.0:
            return "P1"
        return "P1" if int(t / period) % 2 == 0 else "C"
    return f


def p_brake_late(offset_m):
    """負例: 制動開始が遅い（オーバーラン方向）"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if (sp.x1 - x) * 1000.0 <= sp.stop_distance(v, x) - offset_m:
            return "B1"
        return "P1" if v < 60.0 - 0.3 else "C"
    return f


def p_brake_early(offset_m):
    """負例: 制動開始が早い（大幅に手前で停止）"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if (sp.x1 - x) * 1000.0 <= sp.stop_distance(v, x) + offset_m:
            return "B1"
        return "P1" if v < 60.0 - 0.3 else "C"
    return f


def p_crawl(v_cap):
    """負例: 低速で這うちんたら運転"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        return "P1" if v < v_cap - 0.5 else "C"
    return f


def p_power_on_downgrade():
    """負例: 下り勾配（惰行でも加速する）で力行し続ける＝勾配を活用していない"""
    def f(c):
        sp, x, v = c["sp"], c["x"], c["v"]
        if _brake_if_needed(c):
            return "B1"
        ceil = sp.v_ceiling(x)
        if v > ceil - 0.3:
            return "B2"
        return "P1"
    return f


def build(kind):
    """(名前, 区間, 方策) のリストを返す"""
    if kind == "good":
        out = []
        for sec, vc in ((0, 48.7), (1, 74.0)):
            out.append((f"std_s{sec}", sec, p_standard(vc)))
            out.append((f"vstd_s{sec}", sec, p_track_vstd()))
            for dv in (-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0):
                out.append((f"std_s{sec}_v{dv:+.1f}", sec, p_standard(vc + dv)))
        return out
    out = []
    for sec in (0, 1):
        base = 200 if sec == 0 else 300
        out += [
            (f"early_coast_s{sec}_a", sec, p_coast_too_early(base * 0.25)),
            (f"early_coast_s{sec}_b", sec, p_coast_too_early(base * 0.5)),
            (f"over_power_s{sec}", sec, p_over_power()),
            (f"useless_brake_s{sec}", sec, p_useless_brake(base + 100, base + 200)),
            (f"sawtooth_s{sec}_3s", sec, p_sawtooth(3.0)),
            (f"sawtooth_s{sec}_2s", sec, p_sawtooth(2.0)),
            (f"brake_late_s{sec}", sec, p_brake_late(25.0)),
            (f"brake_early_s{sec}", sec, p_brake_early(60.0)),
            (f"crawl_s{sec}_30", sec, p_crawl(30.0)),
        ]
    out.append(("power_on_downgrade_s0", 0, p_power_on_downgrade()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="tozai")
    ap.add_argument("--max-rows", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    profiles = {s: SpeedProfile(a.line, s) for s in (0, 1)}
    for kind, fname in (("good", "tozai_normal_good.csv"), ("bad", "tozai_normal_bad.csv")):
        rows, used = [], []
        for name, sec, pol in build(kind):
            if len(rows) >= a.max_rows:
                break
            r = run_policy(profiles[sec], pol, name, sec)
            if len(rows) + len(r) > a.max_rows:
                continue          # 走行を途中で切らない（切ると不自然な行が混ざる）
            rows += r
            used.append((name, sec, len(r), r[-1]["time"] if r else 0,
                         max((x["current_speed"] for x in r), default=0),
                         r[-1]["dist_to_next_station"] if r else 0))
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csvmod.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== {fname}  {len(rows)} 行 ===")
        print(f"{'run_id':<26}{'区間':>4}{'行数':>6}{'所要':>8}{'最高速':>8}{'停止位置':>10}")
        for nm, sec, n, t, vmax, dd in used:
            print(f"{nm:<26}{sec:>4}{n:>6}{t:>7.1f}s{vmax:>7.1f}{-dd:>9.1f}m")
    print(f"\n出力先: {os.path.relpath(OUT_DIR, BASE_DIR)}/")


if __name__ == "__main__":
    main()
