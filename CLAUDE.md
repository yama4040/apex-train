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
- `input/f_train/` — 先行列車の走行パターンCSV（惰行ポイント方式・`coast{V}_stop{D}.csv`）。`generate_forward_train.py`が生成し、`apex2.py`のActor（学習）／Tester（検証）が読み込む
- `data/` — `apex*.py`実行時の学習ログ・重み（`*.weights.h5`）・走行ダイアグラム等の出力先。`apex2.py`は各runの直下に`TASC制御/`を作り、テストケースごとに停止部分をTASCの制動で上書きした運転曲線PNG（本編と同一書式）と上書き後のCSVログを併せて出力する（学習ループにはTASCを入れない後処理。詳細は`docs_先行列車対応_設計メモ.md` §30・§31）
- `評価用csv/` — LLM評価前のシミュレータ走行ログ（`evaluate_csv_with_llm.py`の入力）
- `評価済ログ/` — LLM評価済みデータセット（`evaluate_csv_with_llm.py`の出力）
- `train_reward_csv_direct/` — 報酬予測NNの学習に実際に使用するCSV置き場（`train_reward_network*.py`が直接読み込む）
- `csv_direct_plas/`, `dataset（0～1.0）/` — LLM評価済みデータの中間・派生データ置き場
- `comp/` — `evaluate_result.py`で比較対象とする個別走行ログCSV置き場
- `standard_curve/` — `generate_standard_curve.py`が出力する数理モデル標準運転曲線（走行ログCSV・meta.json・運転曲線PNG）
- `*.h5` / `*.pkl`（リポジトリ直下） — 学習済み報酬予測NNの重みとスケーラ
- `apex_def.py` / `environment_def.py` — 先行研究で使用していた実装。現行のどのスクリプトからも参照されていないため**書き換え禁止**

## 設計上の重要事項（time_step と報酬・割引のスケーリング）
`environment2.py`の`time_step`は駅手前100m以内で1.0秒→0.1秒に短縮される（`time_step`プロパティ）。停止精度に必要な精密なノッチ制御を可能にするためだが、実時間あたりの整合性を保つため以下の2箇所でステップ幅に比例したスケーリングを行っている。

- **報酬**（`environment2.py`）— NN出力は「1秒あたりの評価値」とみなし、`reward = (llm_reward - 0.5) * (time_step / base_time_step)` でスケール（ゼロ中心化済み）。スケールしないと駅手前で報酬密度が10倍になり「駅手前に留まり続けて報酬を稼ぐ」搾取行動が最適化されてしまう（[[hover-exploit-and-reward-calibration]]の実測問題への対処）。
- **割引率**（`apex2.py`のActor.gamma）— `gamma**(env.time_step/self.time_step)` で実時間あたりの割引を揃える。

**残る副作用（未解決）:** 実時間あたりの報酬・割引は揃うが、TD更新（ブートストラップ）の回数は駅手前だけ10倍に増える。vanilla maxのDQNではブートストラップ連鎖が長いほど過大評価バイアスが蓄積しやすく、駅前区間でのQ値過大評価やリプレイバッファ内での駅前遷移の過剰代表の一因となる。対策案としてDouble DQN化・ターゲット同期間隔の調整・駅手前のみ行動選択粒度を1秒に維持（Nステップ分を1遷移に集約）などが挙がっている。学習具合は`analyze_qnet_coverage.py`で駅直前領域の行動間ギャップとして追跡できる。

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
- `generate_forward_train.py` — 先行列車用の走行パターンCSV（`input/f_train/coast{V}_stop{D}.csv`）を生成するスクリプト。先行列車も自列車と同じ省エネ運転（**惰行ポイント方式**: 出発→惰行ポイントV[km/h]まで力行→惰行→駅に向かって制動→次駅停車→再出発）をしているものとして扱う。V=65が`generate_standard_curve.py`の標準運転曲線と一致し（白兎に181秒で到着）、V=50は標準より遅い運転。この駅間は白兎手前が上り勾配のため惰行のみで駅に届くのはV≒62km/h以上に限られ、V<65では「惰行を続けても制動開始点に届かない」と判定した時点でVまで再加速して駅間停車を避ける（届くと判定した後は再加速しないので、最後の「Vで惰行して駅に向かって減速」フェーズはVによらず必ず現れる）。停止位置誤差は全78パターンで0.00m。先行の出発遅延はCSVでは表現せず、`apex2.py`側で出発間隔（headway）に換算して与える。旧形式（定速走行・`input/f_train_*.csv`、`apex.py`/`apex3.py`が参照）は`--legacy`で再生成できる。詳細は`docs_先行列車対応_設計メモ.md` §32

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

### 強化学習（QNetwork）の診断・可視化
- `analyze_qnet_coverage.py` — 学習済み`QNetwork`（`model.py`／25次元入力・3行動出力）の「Qテーブルの埋まり具合」に相当する診断ツール。表形式Q学習ではなく関数近似のため文字通りのテーブルは無いが、「速度 × 駅までの距離」を格子状にスイープした人工状態を`data/<run>/*.weights.h5`にロードした重みで一括推論し、①max Q（過大評価・発散のチェック）②貪欲方策マップ（惰行/力行/ブレーキ）③行動間ギャップ（≒0の領域＝行動を区別できていない未学習に近い領域）の3面ヒートマップを`qnet_analysis/`へ出力する。グリッド2軸以外の23次元は`environment2.py`の`normalized_state`と同一の正規化式で「定時運行・先行列車なし・平坦・制限70km/h」のシナリオ値を埋める（時刻依存の加速フェーズ・路線依存の制限接近フェーズはグリッド再現不可のため対象外）。`--overlay-csv`でTester出力CSV（`comp/`・`data/`配下）の実走行訪問状態を白点で重ね描き、`--pre-action`で直前ノッチのシナリオを変更できる。runごとに実行し駅直前領域の行動間ギャップが育つか（テーブルが埋まるか）を追跡する用途。

### 数理モデルによる標準運転曲線（DQNの比較基準）
- `generate_standard_curve.py` — 通常運転モードのテストケース（先行なし・自列車遅延なし）と同一の駅間・同一の物理モデルに対し、**定時（標準運転時間180秒）を満たしつつ力行エネルギーが最小となる運転曲線**を数値計算で求めるスクリプト。DQNの学習結果を「理論上の最適運転」と比較するための基準曲線を作る。

  **モデルの一致**: 運動方程式・引張力・走行抵抗・ブレーキ減速度は`train.py`の`Train.step`と同一（0.01秒積分・3ノッチのみ）、勾配/曲線/制限速度の参照規則は`track.py`と同一（区間境界の扱いまで一致）、ノッチ判断の周期は`environment2.py`の`time_step`と同一（駅手前100mで1.0→0.1秒）。物理定数は起動時に`train.py`の実値と突き合わせ、食い違えば例外を出す。生成した行動系列を`train.py`の`Train`でそのまま再生すると停止位置が完全一致することを確認済み。

  **探索する運転パターン**: 最適列車制御の標準形「力行 → 定速保持（力行と惰行のバンバン制御） → 惰行 → 制動」。設計変数は定速保持速度`V_hold`と惰行開始位置`x_coast`の2つで、`x_coast`は到着時刻＝標準運転時間となるよう二分探索（等式制約）、`V_hold`は力行エネルギー最小となるものをグリッド探索で決める。制動開始点は「駅にちょうど停止する制動曲線」を駅から逆向きに積分して事前に求めるため、白兎駅手前の上り勾配（6.1→9.2‰）まで正確に織り込まれ、停止位置誤差は数cmに収まる。惰行開始点・制動開始点のみ0.01秒刻みで配置し、それ以外のノッチ判断はDQNと同じ周期で行う（制動開始が1秒粗いと十数mの停止誤差になり基準曲線にならないため）。

  **出力**（`standard_curve/`）:
  - `<名前>.png` — 標準運転曲線。`apex2.py`のTesterが出力する運転曲線PNGと**同一書式**（dpi200・10×10インチ・駅の黒線・制限速度の階段線・モード別配色・凡例位置）なので、DQNの出力PNGとそのまま並べて比較できる。
  - `<名前>_detail.png` — 勾配を含めた標準運転曲線。ノッチ別（力行/惰行/制動）に色分けし、惰行開始点・制動開始点を示したうえで、運転パターン・到着時刻・停止位置誤差・最高速度・力行エネルギー・ノッチ切替回数などの指標を図中に併記する。
  - `<名前>.csv` — `apex2.py`のTesterと同一スキーマの52列。`drive_monitor.py`の**新形式**としてそのまま再生・比較できる（観測30次元は`environment2.Environment`に状態を流し込んで生成しているためDQNが見る値と完全一致。Q値・報酬列はNNを使わないため0埋め）。モニター用の`<名前>_meta.json`も併せて出力する。

  `--compare`にDQNの走行ログCSVを渡すと、到着時刻・停止位置誤差・力行エネルギー・ノッチ切替回数を並べて表示し、`_detail.png`にも重ね描きする（新形式・旧形式の両方に対応）。`--sr-out`を付ければ`input/sr_*.csv`と同じ形式の標準走行曲線も書き出せる。

### 走行結果の可視化（運転曲線モニター）
- `drive_monitor.py` — テストケースのログCSVを**時間軸に沿って再生する**デスクトップアプリ（PyQt5 + matplotlib、黒背景テーマ）。運転曲線（位置-速度）・ダイヤグラム（時刻-位置）・列車の動きの模式図・自列車実況（モード／ノッチ／現在速度／信号現示／勾配／先行距離／先行停車経過／駅残距離／残り時間）を同時にリアルタイム描画する。開始・停止・リセット、倍速指定（0.1〜50×）、シークバーを備え、CSVはGUIのファイルダイアログで選択する。**2本まで重ねられる**ため「通常運転のみ vs 遅延回復モードあり」のような運転曲線比較に使える（1本目=実線、2本目=破線、模式図では平行な2本の線路として表示）。描画はblitによる差分描画で約20fps、再生速度は実経過時間ベースなのでフレーム落ちしても倍速指定どおりの速さを保つ。

  **表示上の約束**
  - 列車の模式図は**実スケール**（列車長20m・CBTC停止限界50mの帯を自列車前方に表示）。見た目重視の大きな箱で描くと車間が数百mあっても衝突しているように見えるため。位置は先頭位置とみなし車体は後方へ伸ばす。
  - 実況の**残り時間**は`raw_rem_time`ではなく「標準運転時間（出発駅のrt）− 経過時刻」。`environment2.remaining_time`は先行列車がいる場合「先行の位置から引いた標準運転時間 − 経過時刻」となり、先行が進まない間は値が止まる／増えるため時計として読めない（先行なしの場合は両者が完全一致する）。
  - 路線制限速度は信号現示（CBTC指示速度）とほぼ同一のため実況には出さず、運転曲線の背景の階段線としてのみ表示する。

  **入力するログの形式**（`load_run()`が自動判別）
  - *新形式*（2026-08以降の`apex2.py`が出力）: `data/<run>/<file>_<ci>.csv`（末尾に`time`/`position`/`speed_limit`/`fw_position`/`fw_speed`/`mode`/`action`/`gradient`/`fw_dwell_elapsed`の9列）＋`data/<run>/<file>_<ci>_meta.json`（テストケース説明・自列車/先行の遅延・先行の駅停車時間・出発間隔・駅名/位置・標準運転時間・制限速度プロファイル）。タイトルの「先行遅延○秒，駅停車時間○秒」はmeta.json由来。
  - *旧形式*（上記の列を持たない過去のログ）: `raw_*`列とモードone-hotから復元する。時刻・制限速度・勾配・先行列車速度は同runの`data/<run>/LLM評価用/<file>_<ci>_llm.csv`（raw CSVと行数が完全一致する）から取得し、無い場合は`environment2.py`の`time_step`規則（駅手前100m以内で0.1秒、それ以外1.0秒）で時刻を再構成する。絶対位置は`input/Station.csv`のindex 11/12（羽前成田→白兎、`env.reset(11, ...)`固定に対応）から復元する。先行停車経過は旧形式では標準30秒を超過中しか復元できない（LLM評価用CSVが持つ`forward_observed_delay`は超過分のみのため）。復元経路はステータスバーに明示される。

  **注意**: 新形式の9列とmeta.jsonは`apex2.py`の`Tester.test_play`のみが出力する（`apex.py`/`apex3.py`は未対応＝旧形式として読まれる）。9列は既存43列の**後ろに追記**しているため、列名で参照する既存の解析スクリプト（`analyze_qnet_coverage.py --overlay-csv`等）には影響しない。先行の停車経過時間は`environment2.Environment.forward_dwell_elapsed`（モニター表示専用に追加したプロパティ。制御・報酬側からは未参照）で取得する。

## 実行コマンド例
```bash
# 回帰NNを報酬関数として使うApex DQN学習（本研究で基本的に使用）
python apex2.py

# 先行列車の走行パターンCSVの生成（惰行ポイント40〜65km/h × 次駅停車[30,45,60]秒の78種）
python generate_forward_train.py
python generate_forward_train.py --legacy   # 旧形式（定速走行）のinput/f_train_*.csvを再生成

# 回帰NNの学習（train_reward_csv_direct/のデータを使用）
python train_reward_network2.py

# LLMによるデータセット評価（.envにLLM_API_URL/LLM_API_KEYが必要）
python evaluate_csv_with_llm.py

# NN出力とLLMラベルの分布比較
python analyze_reward_nn_vs_llm.py

# QNetworkの学習具合（Qテーブルの埋まり具合）を可視化（data/以下の最新重みを自動選択）
python analyze_qnet_coverage.py --overlay-csv comp/12100_0.csv

# 運転曲線モニター（テストケースのログを再生するGUIアプリ）
python drive_monitor.py                                        # GUIでCSVを選択
python drive_monitor.py data/<run>/0_13.csv                    # 起動時に読み込む
python drive_monitor.py data/<run>/0_0.csv data/<run>/0_13.csv  # 2本を重ねて比較

# 数理モデルによる標準運転曲線（省エネ・定時180秒）の生成とDQNとの比較
python generate_standard_curve.py                                   # standard_curve/ にCSV・PNGを出力
python generate_standard_curve.py --strategy pcb                    # 力行→惰行→制動の3ノッチ運転に固定
python generate_standard_curve.py --compare data/<run>/0_0.csv      # DQNログと指標比較＋重ね描き
python drive_monitor.py standard_curve/standard_curve_11.csv data/<run>/0_0.csv  # モニターで並べて再生
```
※ WSL等でQtのxcbプラグインが読み込めない環境では自動的にwaylandへフォールバックする。
それでも起動しない場合は `sudo apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0` を実行する。
## 先行列車・後続列車がある場合に実現したいこと
### 先行列車のみいる場合
- 先行列車が遅延している場合でも駅間に停車することなく次の駅まで走行する。（駅間に停車しない速度で惰行する方策を獲得する）
- 先行列車に接近しすぎてしまった場合、CBTCの停止限界距離までに停止することができる。
- 信号により停止した後、信号が開通すれば駅に向かって加速をし、駅停車に向けて減速できる。

### 先行・後続列車どちらもいる場合
- 先行列車が遅れている場合、先行列車と後続列車の列車間隔が一定となるように速度を調整する（早めに惰行をするなど）
- 後続列車が遅れている場合、先行列車と後続列車の列車間隔が一定となるように速度を調整する（早めに惰行するなど）

### 共通の目標
- 列車間隔を保つため早めに惰行をした場合、遅延することはや無負えないが、その遅延量を最小限とする必要がある。
- 駅間停車は極力避ける方策を獲得したい。
- 単一列車時と同様、無駄な加減速やノッチ切り替えはせず、乗り心地の良い省エネルギーな運転方策を獲得する必要がある。

### その他備考
- 各列車の駅停車時間は30秒とする。
- CBTCの停止限界距離は50mとする。（先行列車の最後尾と自列車の先頭部分の距離）
- 各列車は1両編成であり、列車長は20mとする。
- その他設定項目はtrain.pyを参照すること。