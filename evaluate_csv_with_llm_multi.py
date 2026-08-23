# -*- coding: utf-8 -*-
"""
複数駅間版のLLM評価ランナー（設計: docs_複数駅間最適化_計画.md §14.3f）

既存 `evaluate_csv_with_llm.py`（羽前成田線・3ノッチ・旧プロンプト）は**一切変更しない**。
プロンプト本文は `prompt_multi.py` に分離してあり、本スクリプトは
「CSVを読む → プロンプトを組む → LLMへ送る → 応答を検証して書き戻す」だけを行う。

既存ランナーから流用できない点（§14.3f(1)）:
  * `checks` のキーが `speed_limit`/`cbtc` → `atc`/`gradient` に変わった
  * `mode` が3種 → 5種（`spacing` / `hold_at_station` 追加）
  * 数値列が47キーに増えた

    入力  評価用csv_Tozai/*.csv
    出力  評価済ログ_Tozai/llm_evaluated_dataset_multi.csv     （mode/reward/reason/checks 付き）
          評価済ログ_Tozai/llm_evaluation_failed_rows.csv       （評価に失敗した行の退避先）

使い方:
    python evaluate_csv_with_llm_multi.py --dry-run --limit 3   # APIを呼ばずプロンプトだけ確認
    python evaluate_csv_with_llm_multi.py --limit 20            # 20行だけ試す
    python evaluate_csv_with_llm_multi.py --workers 6           # 本番（並列6）
    python evaluate_csv_with_llm_multi.py --resume              # 中断した続きから
"""
import os
import glob
import csv
import json
import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

import prompt_multi as PM

load_dotenv()

IN_DIR = "評価用csv_Tozai"
OUT_DIR = "評価済ログ_Tozai"
OUT_NAME = "llm_evaluated_dataset_multi.csv"
FAIL_NAME = "llm_evaluation_failed_rows.csv"

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# LLMが返すべき checks のキー（プロンプトの Step 構成と一致させること）
CHECK_KEYS = ["immediate_zero_rule", "atc", "stop_position", "gradient",
              "phase", "sawtooth", "train_interval", "punctuality_energy"]

# CSVの文字列 → float に直す列（FEATURE_KEYS のうち数値のもの）
NUMERIC = {
    "holding_time", "prev_notch_duration", "notch_jump", "current_speed",
    "atc_now", "signal_speed", "v_ceiling", "v_target", "band_upper", "band_lower",
    "v_std", "v_std_deviation", "schedule_speed", "section_cap", "required_speed",
    "target_speed_no_stop", "target_speed_spacing",
    "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delta_stop",
    "current_gradient", "coast_accel", "power_accel",
    "delay", "total_delay", "total_remaining_time",
    "forward_train_delay", "standard_headway", "forward_clear_remaining_time",
    "forward_observed_delay", "backward_delay",
    "dwell_elapsed", "dwell_min", "dwell_max",
}

_write_lock = threading.Lock()


# =========================================================================
# 1. 行 → プロンプト
# =========================================================================
def row_to_features(row):
    """CSVの1行を `prompt_multi.build_prompt()` に渡せる形へ変換する。"""
    f = dict(row)
    for k in NUMERIC:
        v = f.get(k)
        f[k] = float(v) if v not in (None, "") else None
    f["is_dwelling"] = bool(int(row.get("is_dwelling", 0) or 0))
    f["stations_remaining"] = int(row.get("stations_remaining", 0) or 0)
    f["forward_departed_next"] = row.get("forward_departed_next") or None
    missing = [k for k in PM.FEATURE_KEYS if k not in f]
    if missing:
        raise KeyError(f"CSVに必要な列がありません: {missing}")
    return f


# =========================================================================
# 2. LLM呼び出し
# =========================================================================
def _call_once(prompt_text):
    from openai import OpenAI
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    if not api_key or not base_url:
        raise RuntimeError(".env に LLM_API_KEY / LLM_API_URL が設定されていません")
    client = OpenAI(api_key=api_key, base_url=base_url)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "あなたは列車の自動運転制御を評価するエキスパートです。必ず指示されたJSONフォーマットのみを出力してください。"},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.0,
        timeout=120.0,
    )
    return r.choices[0].message.content


def _parse_json(text):
    t = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", t))     # 末尾カンマを除去して再試行


def _validate(d):
    """応答の妥当性を検証する。問題があれば理由の文字列を返す（正常なら None）。"""
    mode = d.get("mode")
    if mode not in PM.MODES:
        return f"mode が不正: {mode!r}（許可: {PM.MODES}）"
    try:
        rw = float(d.get("reward"))
    except (TypeError, ValueError):
        return f"reward が数値でない: {d.get('reward')!r}"
    if not (0.0 <= rw <= 1.0):
        return f"reward が範囲外: {rw}"
    ch = d.get("checks")
    if not isinstance(ch, dict):
        return "checks が辞書でない"
    lack = [k for k in CHECK_KEYS if k not in ch]
    if lack:
        return f"checks のキー不足: {lack}"
    # 即0.0ルールがNGなら reward は 0.0 でなければならない（プロンプトの明示ルール）
    if str(ch.get("immediate_zero_rule", "")).upper().startswith("NG") and rw != 0.0:
        return f"immediate_zero_rule=NG なのに reward={rw}（0.0であるべき）"
    return None


def evaluate_row(row, max_retries=3, dry_run=False):
    """1行を評価して (mode, reward, reason, checks, error) を返す。"""
    prompt = PM.build_prompt(row_to_features(row))
    if dry_run:
        return None, None, "", "", None, prompt
    last = "不明なエラー"
    for att in range(max_retries):
        try:
            txt = _call_once(prompt)
        except Exception as e:
            last = f"通信エラー: {e}"
            if att < max_retries - 1:
                time.sleep(3 * (att + 1))
            continue
        try:
            d = _parse_json(txt)
        except Exception as e:
            last = f"JSON解析エラー: {e} / 応答冒頭: {txt[:120]!r}"
            if att < max_retries - 1:
                time.sleep(2.0)
            continue
        bad = _validate(d)
        if bad:
            last = f"検証エラー: {bad}"
            if att < max_retries - 1:
                time.sleep(2.0)
            continue
        return (d["mode"], float(d["reward"]), d.get("reason", ""),
                json.dumps(d["checks"], ensure_ascii=False), None, prompt)
    return None, None, f"評価失敗: {last}", "", last, prompt


# =========================================================================
# 3. CSV処理
# =========================================================================
def load_rows(in_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(in_dir, "*.csv"))):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_src"] = os.path.basename(path)
                out.append(r)
    return out


def row_key(r):
    return (r["_src"], r.get("run_id", ""), r.get("time", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=IN_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4, help="並列数（APIはI/O待ちなので効く）")
    ap.add_argument("--limit", type=int, default=None, help="先頭N行だけ処理（試験用）")
    ap.add_argument("--sample", type=int, default=None,
                    help="全体からN行を等間隔で抽出して処理（試験用。先頭N行だと発車直後に偏るため）")
    ap.add_argument("--resume", action="store_true", help="既存の出力に無い行だけ処理する")
    ap.add_argument("--dry-run", action="store_true", help="APIを呼ばずプロンプトだけ確認する")
    a = ap.parse_args()

    rows = load_rows(a.in_dir)
    if not rows:
        print(f"'{a.in_dir}' にCSVがありません。"); return
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, OUT_NAME)
    fail_path = os.path.join(a.out_dir, FAIL_NAME)
    cols = [c for c in rows[0] if c != "_src"]
    headers = ["source_file"] + [c for c in cols if c not in ("mode", "reward", "reason")] \
              + ["mode", "reward", "reason", "checks"]

    done = set()
    if a.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["source_file"], r.get("run_id", ""), r.get("time", "")))
        print(f"[resume] 既に評価済み: {len(done)} 行")
    todo = [r for r in rows if row_key(r) not in done]
    if a.sample and a.sample < len(todo):
        step = len(todo) / a.sample
        todo = [todo[int(i * step)] for i in range(a.sample)]
    elif a.limit:
        todo = todo[:a.limit]

    if a.dry_run:
        print(f"[dry-run] 対象 {len(todo)} 行。APIは呼びません。\n")
        for r in todo[:3]:
            _, _, _, _, _, p = evaluate_row(r, dry_run=True)
            print("=" * 90)
            print(f"{r['_src']} / run_id={r.get('run_id')} / t={r.get('time')}s "
                  f"/ {r.get('current_notch')} / {r.get('current_speed')} km/h")
            print(f"プロンプト長 {len(p)} 文字")
            print(p[p.index("# 現在の走行状況と運転操作"):][:1400])
        print("=" * 90)
        print(f"\n全 {len(todo)} 行でプロンプトが組めることを確認するには --dry-run --limit 0 は使わず、"
              f"下の検証を見てください。")
        ng = 0
        for r in todo:
            try:
                PM.build_prompt(row_to_features(r))
            except Exception as e:
                ng += 1
                if ng <= 5:
                    print(f"  ✗ {row_key(r)}: {e}")
        print(f"プロンプト生成: 成功 {len(todo)-ng} / 失敗 {ng}")
        return

    new_out = not (a.resume and os.path.exists(out_path))
    if new_out:
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(headers)
        with open(fail_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(headers)

    print(f"対象 {len(todo)} 行 / 並列 {a.workers} / モデル {MODEL}")
    t0 = time.time()
    stat = {"ok": 0, "ng": 0}
    modes = {}

    def work(r):
        mode, rw, reason, checks, err, _ = evaluate_row(r)
        base = [r["_src"]] + [r[c] for c in cols if c not in ("mode", "reward", "reason")]
        out_row = base + [mode or "", "" if rw is None else rw, reason, checks]
        with _write_lock:
            path = fail_path if err else out_path
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(out_row)
            if err:
                stat["ng"] += 1
            else:
                stat["ok"] += 1
                modes[mode] = modes.get(mode, 0) + 1
            n = stat["ok"] + stat["ng"]
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(f"  {n}/{len(todo)}  成功{stat['ok']} 失敗{stat['ng']}  "
                      f"{rate:.2f} 行/秒  残り約 {eta/60:.1f} 分")
        return err

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(as_completed([ex.submit(work, r) for r in todo]))

    el = time.time() - t0
    print(f"\n完了: 成功 {stat['ok']} / 失敗 {stat['ng']} / 所要 {el/60:.1f} 分")
    if modes:
        print("  モード分布: " + " / ".join(f"{k}={v}" for k, v in sorted(modes.items())))
    print(f"  出力: {out_path}")
    if stat["ng"]:
        print(f"  失敗行: {fail_path}（原因を確認して --resume で再実行できます）")


if __name__ == "__main__":
    main()
