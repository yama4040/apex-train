## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## プロジェクト概要
- 別のシミュレータを用いて走行パターンを集めLLMにノッチ操作の評価を行わせデータセットを作成
- データセットを用いてLLMを蒸留したニューラルネットワークを作成
- 作成したニューラルネットワークを報酬関数としてApex-Trainに組み込む
- train_reward_network2.pyは出力値を回帰問題としている
- train_reward_network3.pyは出力値を分類問題としている
- 回帰問題のNNを適用しているのがapex2.py，分類問題のNNを適用しているのがapex3.pyである
- apex2.pyを基本的に使用して研究を行っていく

## 実行環境
- uvで管理されたPython 3.11の仮想環境（`.venv/`）を使用する
- 依存パッケージは`requirements.txt`を参照（TensorFlow 2.15 / Ray 2.54 / pandas / scikit-learn / openai 等）
- LLM API（evaluate_csv_with_llm.py, test_prompt_speed.py）を使う場合は`.env`に`LLM_API_URL`・`LLM_API_KEY`の設定が必要

## 研究パイプラインとデータフロー
本プロジェクトは大きく3段階のパイプラインで構成される。

1. **LLMによるデータセット作成**
   `評価用csv/`（シミュレータの走行ログ）→ `evaluate_csv_with_llm.py` がLLM APIにノッチ操作を評価させる → `評価済ログ/llm_evaluated_dataset.csv` を出力
2. **報酬予測NNの蒸留学習**
   `train_reward_csv_direct/`内の評価済みデータを`train_reward_network2.py`（回帰）/ `train_reward_network3.py`（分類）が読み込み学習 → `direct_reward_model*.h5` ＋ `*_scaler.pkl` を生成
3. **Apex DQNへの組み込み**
   `apex2.py`が`environment2.py`経由で学習済みNNをロードし、報酬関数として利用しながらRay分散でDQN学習を行う（学習結果は`data/`以下に出力）

NNは3系統存在し、対応関係は以下の通り（詳細は`analyze_reward_nn_vs_llm.py`冒頭のコメント参照）。

| 系統 | 学習スクリプト | モデル | 予測器 | 環境 | Apexスクリプト |
|---|---|---|---|---|---|
| 旧回帰 | train_reward_network.py | direct_reward_model.h5 | direct_reward_predictor.py | environment.py | apex.py |
| 回帰（現行） | train_reward_network2.py | direct_reward_model2.h5 | direct_reward_predictor2.py | environment2.py | apex2.py（★基本使用） |
| 分類 | train_reward_network3.py | classification_reward_model.h5 | direct_reward_predictor3.py | environment3.py | apex3.py |

## ディレクトリ構成
- `input/` — シミュレーション条件の設定CSV（駅位置・速度制限・曲線・勾配・ダイヤ・遅延パターン等）。`track.py`や`environment*.py`が参照する固定データ
- `data/` — `apex*.py`実行時の学習ログ・重み（`*.weights.h5`）・走行ダイアグラム等の出力先
- `評価用csv/` — LLM評価前のシミュレータ走行ログ（`evaluate_csv_with_llm.py`の入力）
- `評価済ログ/` — LLM評価済みデータセット（`evaluate_csv_with_llm.py`の出力）
- `train_reward_csv_direct/` — 報酬予測NNの学習に実際に使用するCSV置き場（`train_reward_network*.py`が直接読み込む）
- `csv_direct_plas/`, `dataset（0～1.0）/` — LLM評価済みデータの中間・派生データ置き場
- `comp/` — `evaluate_result.py`で比較対象とする個別走行ログCSV置き場
- `*.h5` / `*.pkl`（リポジトリ直下） — 学習済み報酬予測NNの重みとスケーラ
- `apex_def.py` / `environment_def.py` — 先行研究で使用していた実装。現行のどのスクリプトからも参照されていないため**書き換え禁止**

## 主要スクリプトの役割

### 強化学習（Apex DQN）本体
- `apex.py` / `apex2.py` / `apex3.py` — Rayを用いた分散Apex DQN学習のエントリポイント（Actor/Learner/Testerで構成）。それぞれ旧回帰NN／回帰NN／分類NNを報酬関数として使用
- `environment.py` / `environment2.py` / `environment3.py` — 列車制御タスクの環境（`Environment`クラス）。状態の正規化、報酬計算、対応する`direct_reward_predictor*.py`の呼び出しを担当
- `model.py` — `QNetwork`（Dense 5層のQ関数モデル）
- `train.py` — 列車の運動モデル（`Train`クラス）。加速・減速・惰行時の物理シミュレーションを行う共通ロジック
- `track.py` — 路線データ（速度制限・曲線・勾配・ダイヤ）の読み込み
- `actions.py` — 行動定義（`coasting`＝惰行／`acceleration`＝加速／`deceleration`＝減速）
- `segment_tree.py` — 優先度付き経験再生用の`SumTree`実装
- `required_speed.py` — 必要速度（巡航速度）・ブレーキ停止距離の算出ロジック。`evaluate_csv_with_llm.py`（LLM評価プロンプト生成）と`environment2.py`/`environment3.py`（NN学習・推論）の両方から参照され、算出方法を一致させるための共通モジュール
- `generate_forward_train.py` — 先行列車用の走行制御パターンCSV（`input/f_train_*.csv`）を生成するスクリプト

### LLMによるデータセット作成
- `evaluate_csv_with_llm.py` — `評価用csv/`内の走行ログをLLM APIに送り、ノッチ操作の評価（報酬値・理由）を取得して`評価済ログ/`に出力
- `test_prompt_speed.py` — LLM APIへのプロンプト送信・応答時間計測用スクリプト

### 報酬予測NN（蒸留）の学習・評価
- `train_reward_network.py` / `train_reward_network2.py` / `train_reward_network3.py` — LLM評価済みデータセットから報酬予測NNを学習（それぞれ旧回帰／回帰／分類）
- `direct_reward_predictor.py` / `direct_reward_predictor2.py` / `direct_reward_predictor3.py` — 学習済みNNをロードして推論する予測器クラス（各`environment*.py`から利用される）
- `reward_predictor.py` — `RewardWeightPredictor`。環境要素ごとの重み付けを予測する旧方式の予測器（`environment.py`のみが使用）
- `analyze_reward_nn_vs_llm.py` — LLMラベル分布と3系統のNN出力分布を比較・可視化し、LLM／NNそれぞれの評価の妥当性を検証する
- `check_reward_distribution.py` — `train_reward_csv_direct/`内データの報酬分布を可視化
- `evaluate_result.py` — 個別走行ログCSV（`comp/`）に対する報酬の比較・検証

## 実行コマンド例
```bash
# 回帰NNを報酬関数として使うApex DQN学習（本研究で基本的に使用）
python apex2.py

# 回帰NNの学習（train_reward_csv_direct/のデータを使用）
python train_reward_network2.py

# LLMによるデータセット評価（.envにLLM_API_URL/LLM_API_KEYが必要）
python evaluate_csv_with_llm.py

# NN出力とLLMラベルの分布比較
python analyze_reward_nn_vs_llm.py
```
