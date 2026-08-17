低速ブレーキ厳格化プロンプト（2026-07-21）に伴う矛盾ラベルの退避
==================================================
退避行数合計: 1191行

【矛盾の内容】低速域(v<15km/h)のブレーキで、旧±5m一律基準の高評価が
新・速度依存基準（低速はΔ2〜4m→0.4-0.6, Δ>4m→0.2-0.3, Δ<-1m→0.0）と衝突する行。

【この退避だけを元に戻す方法】
各 backup__<ファイル名> を ../<ファイル名> へ上書きコピーすれば、
この退避を行う直前の状態に完全に戻ります。
例: cp backup__llm_evaluated_dataset.csv ../llm_evaluated_dataset.csv

【conflict_rows__<ファイル名>】退避した矛盾行のみ（LLM評価フォーマット）。
新プロンプトで再評価すれば正しいラベルで回収できます。

対象ファイル:
  llm_eval_data_20260710_005435.csv: 3行
  llm_eval_data_20260710_173627.csv: 4行
  llm_eval_data_20260712_215835.csv: 3行
  llm_eval_data_20260713_105043.csv: 3行
  llm_eval_data_20260713_235311.csv: 8行
  llm_eval_data_20260714_220814.csv: 54行
  llm_eval_merged_20260714.csv: 14行
  llm_evaluated_dataset (11).csv: 47行
  llm_evaluated_dataset (14).csv: 12行
  llm_evaluated_dataset (17).csv: 135行
  llm_evaluated_dataset (20).csv: 48行
  llm_evaluated_dataset (21).csv: 163行
  llm_evaluated_dataset (23).csv: 235行
  llm_evaluated_dataset (24).csv: 186行
  llm_evaluated_dataset (3).csv: 5行
  llm_evaluated_dataset (4).csv: 8行
  llm_evaluated_dataset (5).csv: 19行
  llm_evaluated_dataset (6).csv: 1行
  llm_evaluated_dataset(14).csv: 36行
  llm_evaluated_dataset(18).csv: 207行
