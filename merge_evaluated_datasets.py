# 2026-07-14の評価結果を1つのデータセットに結合するスクリプト
# 結合対象:
#   1. 評価済ログ/llm_evaluated_dataset_backup_20260713.csv （7/13 11:04開始プロセスの 〜23:58 までの出力）
#   2. 評価済ログ/llm_evaluated_dataset.csv                 （同プロセスの 00:03以降 の出力。冒頭に別プロセスの数行が混在）
#   3. 評価済ログ/synthetic_20260714/llm_evaluated_dataset.csv （synthetic 494行の評価結果）
# 同一状態が二重評価されている可能性があるため、特徴量列で重複排除する（最初の1件を採用）。
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(BASE, "評価済ログ/llm_evaluated_dataset_backup_20260713.csv"),
    os.path.join(BASE, "評価済ログ/llm_evaluated_dataset.csv"),
    os.path.join(BASE, "評価済ログ/synthetic_20260714/llm_evaluated_dataset.csv"),
]
OUT = os.path.join(BASE, "train_reward_csv_direct/llm_eval_merged_20260714.csv")

dfs = []
for src in SOURCES:
    if not os.path.exists(src):
        print(f"[スキップ] {src} が見つかりません")
        continue
    d = pd.read_csv(src, encoding="utf-8-sig")
    d = d[pd.to_numeric(d["reward"], errors="coerce").notna()]  # 評価失敗行を除外
    print(f"{os.path.basename(src)}: {len(d)}行")
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
# 特徴量列（reward/reason以外）で重複排除
feature_cols = [c for c in df.columns if c not in ("reward", "reason")]
before = len(df)
df = df.drop_duplicates(subset=feature_cols, keep="first")
print(f"重複排除: {before} → {len(df)}行")
print("ラベル分布:", df["reward"].round(1).value_counts().sort_index().to_dict())

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"保存完了: {OUT}")
print("このあと train_reward_network2.py を実行すればNNが再学習されます（MPLBACKEND=Agg推奨）。")
