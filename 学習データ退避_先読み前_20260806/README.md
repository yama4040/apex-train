# 退避理由（2026-08-06）

target_speed_no_stop / forward_clear_remaining_time の**先読み化（設計メモ §15・2026-07-31）より前**に
生成・LLM評価された先行列車ありデータ。現行環境とは特徴量の意味が異なるため学習から除外した。

## 判別根拠（100%明確に分離できた）
「先行の次駅での観測遅延 > 0」の行における forward_clear_remaining_time:
- 旧実装: clear = 0 に潰れる（想定発車時刻＝現在時刻になるため）
- 新実装: clear >= 15 を確保（OBSERVED_DWELL_LOOKAHEAD_S）

退避した6ファイルは該当行すべてが clear=0、残した5ファイル((45)(46)(49)(50)(30))は
すべて clear>=10 で、中間の混在ファイルは無かった。

## 代替データ
同じシナリオは先読み化後に再生成・再評価済み:
- 先行あり(先読み): llm_evaluated_dataset (45)(46), (30)
- CBTC遵守:        llm_evaluated_dataset (49)(50), (32)
