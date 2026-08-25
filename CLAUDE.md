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
- **進行中の拡張**: 単一駅間から**複数駅間**の最適化へ拡張中（フェーズ0に着手済み）。対象は東京メトロ東西線の**東陽町→木場→門前仲町**（3駅・2駅間）で、先行列車に加えて**後続列車**も扱う。設計・実装手順・研究手順は`docs_複数駅間最適化_計画.md`にまとめてある（既存の単一区間スクリプトは残し、`*_multi.py`として新規作成する方針）

### 複数駅間最適化（`*_multi` 系・新規／既存には非干渉）
東京メトロ東西線の3駅（東陽町→木場→門前仲町）へ拡張するための新規スクリプト群。
**既存の単一区間手法（`apex2.py`・`evaluate_csv_with_llm.py`のプロンプト・学習済みモデル7ファイル）は
一切変更せず、いつでも実行可能な状態に保つ**（詳細と変更禁止リストは`docs_複数駅間最適化_計画.md` §9.1）。
- `line_config.py` — 路線メタデータ＋車両パラメータ。東西線（下りB線・CS-ATC現示・標準運転間隔140秒・標準停車30秒）と東京メトロ15000系（10両200m・走行抵抗A/B/C・引張力3領域・制動2.5 km/h/s・勾配補償35‰）
- `track_multi.py` — 路線データローダ。**進行方向に増加する内部座標へ変換**（下りB線でキロ程が減少するため。勾配符号は保持）、勾配フィルタ緩和（範囲外は例外。既存`track.py`は無言で0.0にする）、`curve.csv`欠損許容、ATC現示は先頭位置で判定、現示低下点の先読み。単体実行で路線諸元を一覧表示できる
- `actions_multi.py` — **5行動**の定義（P1力行／P2勾配力行／C惰行／B2勾配ブレーキ／B1制動）。P2は+35‰、B2は−35‰でちょうど定速になる
- `train_multi.py` — 5ノッチの運動モデル。走行抵抗・制動・引張力を**1箇所に集約**（既存`train.py`は4箇所に重複）。P2の引張力クリップと起動時アサーション付き。単体実行で5ノッチ×勾配の加速度表を出力
- `generate_standard_curve_multi.py` — **標準運転曲線ジェネレータ**。ATC現示の先読み天井・制動曲線の逆積分・V_holdグリッド探索・惰行開始点の二分探索。出力は`standard_curve_multi/`（PNG・CSV・meta.json）。東西線の全3駅間で停止位置誤差ミリメートル級で生成できることを確認済み
- `evaluate_csv_with_llm_multi.py` — **複数駅間版のLLM評価ランナー**。プロンプト本文は`prompt_multi.py`に分離してあり、本体はCSV読み込み・プロンプト組み立て・API呼び出し・応答検証・書き戻しを担う。`--dry-run`（APIを呼ばず全行のプロンプト生成を検証）／`--workers N`（並列）／`--resume`（中断再開）／`--limit N`（試験）。応答は mode が5種のいずれか・reward が0〜1・checks の8キーが揃う・immediate_zero_ruleがNGならreward=0.0、を満たさなければリトライする。**既存の`evaluate_csv_with_llm.py`とはchecksのキーもmodeの種類も違うため流用不可**
- `generate_eval_csv_multi.py` — **LLM評価用の走行ログCSV生成**。`prompt_multi.py`が要求する特徴量を全て埋めた走行ログを`評価用csv_Tozai/`へ出力する。正例（標準運転曲線の再現・v_std追従・惰行ポイント変動）と負例（早すぎる惰行・過剰力行・無駄な制動・ノコギリ・制動開始の遅早・ちんたら運転・下り勾配での力行）を方策として定義してある
- `generate_forward_train_multi.py` — 先行／後続列車の走行パターン生成（複数駅間の惰行ポイント方式）。テストケースは2種で、`normal`（標準運転曲線の惰行開始速度＝東陽町発48.7／木場発74.0 km/h）と`slow`（東陽町発40／木場発55 km/h）。**登り勾配で35km/hを下回ったらVまで再力行**する（このルールが無いと+27〜29.7‰のランプで失速し駅の481〜578m手前で駅間停車する）。出力は`input/f_train_multi/`（駅停車時間の組合せで24ファイル）
- `required_speed_multi.py` — **目標速度の算出**。既存`required_speed.py`は勾配・制限速度を「現在地点の値1つ」で駅まで一定と仮定するが、本モジュールは**前方プロファイルを積分**する（東西線は惰行減速度が+0.29〜−1.34 km/h/sと符号ごと反転するため必須）。提供するもの: ATC現示の先読み天井／先行列車のCBTC現示（停止限界＝先行の最後尾から50m手前）／プロファイル対応の制動距離／維持帯（勾配と乗り心地T_min=5秒から帯幅と使用ノッチ対を算出）／惰行到達可能性／モード別目標速度（normal=標準運転曲線v_std・delay_recovery=天井直下の維持帯・anti_mid_stop=先行クリア時間から算出・spacing=車間の均し）。`targets()`が0.45ms/回で毎ステップ呼べる。単体実行で各区間の目標速度を一覧表示
- `optimal_curve_dp.py` — **検証用**。運転パターン（力行→定速保持→惰行→制動）を仮定せず、位置×速度の格子上で動的計画法を解いてエネルギー最小の理論下界を求める。`generate_standard_curve_multi.py`の解が最適から離れていないかの確認に使う（実測で力行仕事は理論下界の約13%増・ただしノッチ切替は1/4以下）
- `prompt_multi.py` — 複数駅間版のLLM評価プロンプト。5ノッチ・可変維持帯・ATC先読み・標準運転曲線基準・停車中の発車判断・後続列車を含む。既存プロンプトとは別ファイルで保全

### 複数駅間最適化・山形版（`*_ymulti` 系・新規／既存には非干渉）
**既存路線（山形鉄道フラワー長井線）で複数駅間を1エピソードで走る**ための第3の系統。
対象は**羽前成田 → 白兎 → 蚕桑**（3駅・2駅間）で、**中間駅（白兎）での発車判断**を学習させる。
行動空間は**既存と同じ3ノッチ**（力行/惰行/制動）なので、単一区間版の結果とそのまま比較できる。
設計・根拠・実行手順は`docs_複数駅間_山形_設計メモ.md`にまとめてある。

**既存の`apex2.py`関連スクリプトは一切変更していない**。`train.py`・`track.py`・`actions.py`・
`model.py`・`segment_tree.py`・`histogram_loss.py`・`generate_standard_curve.py`は
**読み取り専用でimportするだけ**で、物理モデル・路線参照・制動曲線の逆積分を既存と完全に一致させている。

- `config_ymulti.py` — 区間・ダイヤ・停車・先行パターン・出力先の設定。**白兎→蚕桑の標準運転時間は130秒**
  （`input/Station.csv`の`rt`=180秒は1377mに対し最短118.8秒で余裕61秒あり緩すぎ、駅停車の遅延が下流に効かない。
  Station.csvは書き換えずこちらで持つ）。標準停車30秒／最低停車30秒／**最大停車300秒**／出発間隔120秒／
  列車長20m／CBTC停止限界は**先行の先頭から70m**（＝最後尾から50m）
- `brake_curve_ymulti.py` — **駅から逆積分した制動曲線**。`train.req_stop_dist`と`required_speed.brake_stop_distance_m`は
  「制動開始地点の勾配・曲線が停止まで一定」と仮定するが、蚕桑の制動開始点直前にR=400mの曲線があり
  **制動距離を5.4m過小評価**する（実際に先行列車生成で5.75mの過走が発生）。逆積分なら数cm精度
- `standard_curve_ymulti.py` — 2区間の標準運転曲線と**`v_std`ルックアップ表**の生成。既存`generate_standard_curve.StandardCurveSolver`を
  再利用する。羽前成田→白兎は定速保持65.25km/h（180秒・誤差+0.022m）、白兎→蚕桑は54.50km/h（130秒・誤差+0.018m）、
  通算340秒で累積標準ダイヤと一致。出力は`standard_curve_ymulti/`
- `required_speed_ymulti.py` — **勾配プロファイルを積分する**目標速度。既存`required_speed.py`の一定勾配仮定では、
  白兎→蚕桑（+11.4‰が1km→−2.3‰）で惰行減速度が−0.581→−0.097km/h/sと6倍変わるため誤差が大きい。
  **惰行到達可能性**（`coast_probe`）も提供する。1ステップ2.27ms（既存3.59ms/回より速い）
- `generate_forward_train_ymulti.py` — 先行列車パターン（**白兎・蚕桑の2駅で停車**）。
  惰行ポイント40〜65km/h × 白兎{30,45,60}秒 × 蚕桑{30,60,120,180}秒 = **312種**を`input/f_train_ymulti/`へ。
  蚕桑の120/180秒が「急病人救護など」の長時間停車＝自列車が白兎に留まるべき局面。停止位置誤差 最大0.142m・駅間停車0件
- `reward_features_ymulti.py` — 状態スキーマ（49列）・特徴量（72次元）・**モード判定をルールで一本化**。
  既存はモードNNのargmaxを併用していたが条件成立中でも`normal↔delay_recovery`が反転する実測問題があるため、
  プロンプトの定義と`decide_mode()`を1対1に対応させた。モードは`normal`/`delay_recovery`/`anti_mid_stop`/
  `spacing`（枠のみ）/**`hold_at_station`**（駅停車中の発車判断）
- `environment_ymulti.py` — **複数駅走行＋駅停車フェーズ**を持つ環境（観測**40次元**）。
  中間駅では`done`にせず停車フェーズへ遷移し、**3ノッチを「発車（力行）／待機（制動）」の2択に読み替える**
  （惰行は禁止。待機と同一結果になりQ学習のmax演算子が過大評価を累積するため）。停車中は`time_step`を1.0秒に固定。
  **既存`environment2.py`から直した実バグ4件**（①失敗判定を到着判定より先に行う＝先行に追突しても到着成功になっていた
  ②異常接近を距離だけで判定 ③信号待ち判定を列車長込みにする ④信号開通後・発車直後の猶予）
- `rule_reward_ymulti.py` — **暫定ルール報酬**。CLAUDE.mdの「報酬はNN出力のみ」制約に対する一時的な例外で、
  報酬NNが用意できるまで環境・DQNの構造を検証するために使う。同時に`prompt_ymulti.py`の**実行可能な下書き**でもあり、
  片方を直したらもう片方も直すこと
- `prompt_ymulti.py` — LLM評価プロンプト（3ノッチ・複数駅間・**駅停車中の発車判断**・惰行到達可能性）。
  既存プロンプトとも`prompt_multi.py`とも別ファイル
- `generate_eval_csv_ymulti.py` — LLM評価用の走行ログCSV生成。**環境をそのまま走らせて生の状態辞書を書き出す**ので、
  LLMが評価する状態とRL実行時に報酬NNが見る状態が構造的に一致する。正例7方策・負例5方策 × 33シナリオ
- `evaluate_csv_with_llm_ymulti.py` — LLM評価ランナー。`--dry-run`／`--workers N`／`--resume`／`--limit N`。
  **停車中の行なのに`mode`が`hold_at_station`でない応答や、`mode=spacing`（後続列車が存在しない）は検証エラーで再試行する**
- `train_reward_network_ymulti.py` / `direct_reward_predictor_ymulti.py` — 報酬NNの学習と推論。
  構成は既存推奨（ゲート＋HL-Gaussian）を踏襲し、**駅停車中の行にサンプル重み**を与える（走行中に比べ圧倒的に少ないため）
- `runcurve_plot_ymulti.py` — 2区間通しの運転曲線と時刻-位置ダイアグラムの描画（標準運転曲線を重ねる）
- `apex_ymulti.py` — Apex DQN 学習エントリポイント。**Double DQNが既定**（複数駅間はエピソードが2倍以上長く、
  駅手前100mの0.1秒刻みが駅の数だけ増えてブートストラップ連鎖が伸びるため）。`--reward rule`で報酬NN無しでも回せる。
  出力は`data_ymulti/`

**データ置き場**（既存と絶対に混ぜない）: `standard_curve_ymulti/` / `input/f_train_ymulti/` /
`評価用csv_Yamagata/` / `評価済ログ_Yamagata/` / `train_reward_csv_direct_Yamagata/` / `data_ymulti/` /
`direct_reward_{model,gate,scaler,manifest}_ymulti.*`

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
- `input/Tozai_line/` — **東京メトロ東西線**の路線データ（複数駅間最適化用に2026-08-19追加）。`Tozai_line_Station.csv`（東陽町16.0118／木場15.0473／門前仲町13.9893／茅場町12.1875 km・`rt`=80/85/140/60秒）・`Tozai_line_grade.csv`（20区間・16.2243〜12.164 kmを被覆）・`Tozai_line_speed_limit.csv`（7区間・45〜75km/h）の3ファイル。**下り列車（B線）のためキロ程は進行方向に沿って減少する**（正常な仕様）。勾配は**進行方向基準**で符号をそのまま使える（東陽町→木場は平均−10.3‰の下り、木場→門前仲町は平均+11.3‰の登り・最大+29.7‰。実データで検証済み）。標準運転間隔はラッシュ時の**140秒**（2分20秒）を採用する。**既存の`track.py`ではそのまま読めない**点に注意（①位置の単調増加を前提としたロジックと逆向き／②勾配フィルタ`-40 < g <= 30`に+29.7‰が0.3‰差で接しており、茅場町まで延長すると±35‰が無言で0.0に潰される／③`curve.csv`に相当するファイルが無い）。複数駅間用の新ローダ`track_multi.py`でこれらを吸収する方針。**車両は東京メトロ15000系10両編成**（`15000系_車両情報.pdf`）を採用する。現行`train.py`の車両モデル（山形鉄道1両編成28t相当）では東西線の標準運転時間が達成できない（木場→門前仲町は全力行97.0秒で標準85秒に12秒不足）が、15000系なら71.3秒／77.3秒で9〜11%の余裕をもって成立する（制動は常用最大3.5ではなく一般的な駅停車ブレーキ2.5 km/h/sを使用）。**`Tozai_line_speed_limit.csv`はCS-ATCの信号現示の切替位置**であり、列車の先頭が越えた時点で現示が変わる（物理的な制限区間そのものではなく、制限に当たらないよう手前で切り替わる設定。よって列車長は考慮しない）。現示の扱いは**予見型**（低下点に到達した時点で既に新現示以下＝手前から緩やかに減速し乗り心地を確保）を既定とし、**追従型**（低下点で制動開始）も切替フラグで実装しておく。**10両編成＝列車長200mのため、CBTC停止限界（先行の最後尾から50m手前に自列車の先頭＝先行の先頭から250m手前）を新たにモデル化する必要がある**（現行実装には列車長の概念が無く、先行の先頭から50mしか引いていないため200m甘い）。列車長を効かせるのはCBTC車間・衝突判定・後続の頭打ち・可視化の4箇所で、ATC現示には適用しない。曲線データは東西線には無いが、対象3駅では影響が小さい（R=400m一律でも力行仕事+6〜7%、比較対象の羽前成田→白兎も距離加重平均0.05kg/tで実質曲線なし）。なお**既存の単一区間手法（`apex2.py`・LLM評価プロンプト・学習済みモデル）は一切変更せず、いつでも実行可能な状態に保つ**方針であり、複数駅間版は`_multi`系の新規スクリプトとして作る（報酬NNの成果物ファイル名も分離して上書き事故を防ぐ）。詳細は`docs_複数駅間最適化_計画.md` §6.5・§6.6・§9.1・付録B2
- `data/` — `apex*.py`実行時の学習ログ・重み（`*.weights.h5`）・走行ダイアグラム等の出力先。停止部分をTASCの制動で上書きした運転曲線PNG（本編と同一書式）と上書き後のCSVログは、学習後に`apply_tasc_to_runcurve.py`を実行すると`<run>/TASC制御/`に生成される（学習ループにはTASCを入れない後処理。詳細は`docs_先行列車対応_設計メモ.md` §30・§31・§33）
- `評価用csv/` — LLM評価前のシミュレータ走行ログ（`evaluate_csv_with_llm.py`の入力）
- `評価用csv_Tozai/` / `評価済ログ_Tozai/` / `train_reward_csv_direct_Tozai/` — **複数駅間版（東西線）専用のデータ系統**。既存の`評価用csv/`・`train_reward_csv_direct/`とは**絶対に混ぜない**（3ノッチvs5ノッチ・プロンプト世代・勾配分布がすべて異なり、プロンプト世代の混在は過去にラベル矛盾の実害を出している）
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
- `runcurve_plot.py` — 運転曲線PNGの共通描画（モード別配色`MODE_COLORS`・駅線・制限速度の階段線）。`apex2.py`のTesterと`apply_tasc_to_runcurve.py`が共有し、両者のPNGを同一書式に保つ
- `apply_tasc_to_runcurve.py` — **学習後の後処理**。DQNの走行ログの停止部分をTASC（停止位置制御）の制動パターンで上書きし、運転曲線PNG（本編と同一書式）と上書き後のCSVログを`<run>/TASC制御/`へ出力する。学習フォルダとcycle番号を渡すとそのcycleの全テストケースを一括処理する。学習ループにTASCは入れない（入れるとTASC作動中は3行動が同一結果になりQ値が膨張して方策が破綻する。詳細は`docs_先行列車対応_設計メモ.md` §30・§33）。引き継ぎ点は「制動パターンに到達した点」、到達しない場合は「最後の制動を開始した点」まで遡って直前のノッチを延長する。延長中は制限速度・CBTC現示を超えない／失速して駅間停車しないようガードする。apex2.py側の呼び出しはコメントアウトで残してあり、学習中に出力させたい場合は戻せる
- `segment_tree.py` — 優先度付き経験再生用の`SumTree`実装
- `required_speed.py` — 必要速度（巡航速度）・ブレーキ停止距離の算出ロジック。`evaluate_csv_with_llm.py`（LLM評価プロンプト生成）と`environment2.py`/`environment3.py`（NN学習・推論）の両方から参照され、算出方法を一致させるための共通モジュール
- `generate_forward_train.py` — 先行列車用の走行パターンCSV（`input/f_train/coast{V}_stop{D}.csv`）を生成するスクリプト。先行列車も自列車と同じ省エネ運転（**惰行ポイント方式**: 出発→惰行ポイントV[km/h]まで力行→惰行→駅に向かって制動→次駅停車→再出発）をしているものとして扱う。V=65が`generate_standard_curve.py`の標準運転曲線と一致し（白兎に181秒で到着）、V=50は標準より遅い運転。この駅間は白兎手前が上り勾配のため惰行のみで駅に届くのはV≒62km/h以上に限られ、V<65では「惰行を続けても制動開始点に届かない」と判定した時点でVまで再加速して駅間停車を避ける（届くと判定した後は再加速しないので、最後の「Vで惰行して駅に向かって減速」フェーズはVによらず必ず現れる）。停止位置誤差は全78パターンで0.00m。先行の出発遅延はCSVでは表現せず、`apex2.py`側で出発間隔（headway）に換算して与える。旧形式（定速走行・`input/f_train_*.csv`、`apex.py`/`apex3.py`が参照）は`--legacy`で再生成できる。詳細は`docs_先行列車対応_設計メモ.md` §32

### LLMによるデータセット作成
- `evaluate_csv_with_llm.py` — `評価用csv/`内の走行ログをLLM APIに送り、ノッチ操作の評価（報酬値・理由）を取得して`評価済ログ/`に出力
- `test_prompt_speed.py` — LLM APIへのプロンプト送信・応答時間計測用スクリプト

### 報酬予測NN（蒸留）の学習・評価
- `train_reward_network.py` / `train_reward_network2.py` / `train_reward_network3.py` — LLM評価済みデータセットから報酬予測NNを学習（それぞれ旧回帰／回帰／分類）
- `direct_reward_predictor.py` / `direct_reward_predictor2.py` / `direct_reward_predictor3.py` — 学習済みNNをロードして推論する予測器クラス（各`environment*.py`から利用される）
- `histogram_loss.py` — **HL-Gaussianの共通モジュール**。ビン中心・打ち切り正規分布の教師・期待値読み出し・ゲートとの合成を一本化し、学習側（`train_reward_network2.py`）と推論側（`direct_reward_predictor2.py`）・解析側で定義が食い違わないようにする
- `reward_predictor.py` — `RewardWeightPredictor`。環境要素ごとの重み付けを予測する旧方式の予測器（`environment.py`のみが使用）
- `analyze_reward_nn_vs_llm.py` — LLMラベル分布と3系統のNN出力分布を比較・可視化し、LLM／NNそれぞれの評価の妥当性を検証する
- `analyze_reward_imbalance.py` — ラベル不均衡による予測バイアスの診断。balanced-MAE・校正直線の傾き・予測std比・真値ビン別バイアス・shot別MAEを算出し、`--tags`で複数モデルを同一分割で並べられる
- `train_reward_heads.py` / `compare_reward_heads.py` — **出力ヘッド／損失を差し替えた比較専用**の学習・評価ツール。本番モデルを上書きしないので安全に実行でき、新しいヘッドを本番へ入れる前の検証に使う（詳細は`docs/報酬NN出力ヘッド比較レポート.html`）
- `check_reward_distribution.py` — `train_reward_csv_direct/`内データの報酬分布を可視化
- `evaluate_result.py` — 個別走行ログCSV（`comp/`）に対する報酬の比較・検証

#### 回帰NNの出力ヘッド（2026-08-17に HL-Gaussian へ移行）
`train_reward_network2.py`の回帰器は既定で**HL-Gaussianヘッド**（Imani & White, ICML 2018）になった。
出力を値域ビンのsoftmaxにし、教師を「ラベル中心の打ち切り正規分布」に変換して交差エントロピーで学習する。

- **既定のビン設定** — 核19ビン（幅0.05）＋ 両端に**予備ビン1個ずつ** ＝ 計21ビン、中心 0.05〜1.05、σ=0.0375
  - 予備ビンは必須。ビン中心をラベル範囲（0.1〜1.0）そのものに置くと端のラベルでガウスの裾が切れ、
    `y=1.0`の教師期待値が0.963になって端を原理的に当てられなくなる（校正直線の傾きが約0.045低下）
- **推論時の合成** — `reward = (1 − ゲート確率) × Σ fᵢcᵢ`。ハード閾値・`clip(0.1,1.0)`・`round(_,1)`をすべて撤廃
  - 旧構成では回帰器の生出力の26.7%が値域外に出て`clip`が破綻を隠していた。softmaxの期待値なら原理的に値域内
  - `clip`下限0.1と0.1丸めのせいで報酬0.1が実質出力されなかった問題（DQN実走行で0.22%）が解消する
  - ゲート確率が0.5をまたぐ瞬間に報酬が0↔0.5以上へ飛ぶ不連続が無くなる
- **ヘッドの判別** — `direct_reward_manifest.json`の`regressor_head`（`head`／`centers`／`composition`）を推論側・解析側が読む。
  このキーが無い旧マニフェストは従来のスカラー回帰（Huber＋ハード合成＋0.1丸め）として扱われるため、**過去のモデル資産はそのまま動く**
  - モデルの出力次元とマニフェストのビン数が食い違う場合は起動時に例外を出す（報酬が黙って壊れるのを防ぐ）
- **旧構成の再現** — `python train_reward_network2.py --head scalar`
- **ゲート分類器は分離のまま維持** — 単一ヘッドへ統合する案（ゼロアトム）は、真値0.0のMAEが0.017→0.041〜0.062へ悪化したため不採用

**注意**: ヘッドを切り替えたら**必ず`train_reward_network2.py`で再学習**すること。モデル・スケーラ・マニフェストは
同時に更新されるので、この3点セットの世代を揃えて運用する。

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

# 複数駅間版（東西線）の路線諸元の確認・標準運転曲線の生成
python track_multi.py tozai                                  # 駅・ATC現示・勾配の一覧
python train_multi.py                                        # 5ノッチ×勾配の加速度表
python generate_standard_curve_multi.py                      # 全駅間の標準運転曲線を standard_curve_multi/ へ
python generate_standard_curve_multi.py --section 0           # 東陽町→木場のみ
python generate_forward_train_multi.py                       # 先行列車パターンを input/f_train_multi/ へ
python generate_eval_csv_multi.py                            # LLM評価用の走行ログを 評価用csv_Tozai/ へ
python evaluate_csv_with_llm_multi.py --dry-run              # APIを呼ばずプロンプトを確認
python evaluate_csv_with_llm_multi.py --workers 6            # LLM評価の本番実行（.env が必要）

# 複数駅間版・山形（羽前成田→白兎→蚕桑・3ノッチ）
python config_ymulti.py                                      # 区間・ダイヤ・停車設定の確認
python brake_curve_ymulti.py                                 # 制動距離（逆積分 vs 一定勾配近似）
python required_speed_ymulti.py                              # 目標速度・惰行到達可能性の一覧
python standard_curve_ymulti.py                              # 標準運転曲線とv_std表 → standard_curve_ymulti/
python generate_forward_train_ymulti.py --jobs 6             # 先行列車312種 → input/f_train_ymulti/
python generate_eval_csv_ymulti.py --rows 4500               # LLM評価用CSV → 評価用csv_Yamagata/
python evaluate_csv_with_llm_ymulti.py --dry-run             # APIを呼ばずプロンプトを検証
python evaluate_csv_with_llm_ymulti.py --workers 6           # LLM評価の本番実行（.env が必要）
python train_reward_network_ymulti.py                        # 報酬NNの学習（train_reward_csv_direct_Yamagata/）
python apex_ymulti.py                                        # Apex DQN 学習（報酬NNがあれば自動で使用）
python apex_ymulti.py --reward rule                          # 報酬NN無しで環境・DQNの構造を検証

# 回帰NNの学習（train_reward_csv_direct/のデータを使用・既定はHL-Gaussianヘッド）
python train_reward_network2.py
python train_reward_network2.py --head scalar        # 旧構成（Huber回帰）の再現
python train_reward_network2.py --bins 19 --guard-bins 1 --sigma-ratio 0.75   # 既定値（明示指定する場合）

# LLMによるデータセット評価（.envにLLM_API_URL/LLM_API_KEYが必要）
python evaluate_csv_with_llm.py

# NN出力とLLMラベルの分布比較
python analyze_reward_nn_vs_llm.py

# ラベル不均衡による予測バイアスの診断（複数モデルを同一分割で比較できる）
python analyze_reward_imbalance.py                   # 本番モデルを診断
python analyze_reward_imbalance.py --noise-check     # ラベルの既約ノイズ（改善余地の上限）も測る

# 出力ヘッドの比較（本番モデルは上書きしない。新ヘッドを本番へ入れる前の検証用）
python train_reward_heads.py --variants baseline hl_gauss
python compare_reward_heads.py --tags _base _hlg

# QNetworkの学習具合（Qテーブルの埋まり具合）を可視化（data/以下の最新重みを自動選択）
python analyze_qnet_coverage.py --overlay-csv comp/12100_0.csv

# TASC制御で停止部分を上書きした運転曲線・CSVの生成（学習後の後処理）
python apply_tasc_to_runcurve.py data/<run> <cycle>          # そのcycleの全テストケース
python apply_tasc_to_runcurve.py data/<run>                  # cycle省略＝最新cycle
python apply_tasc_to_runcurve.py data/<run> <cycle> --cases 0 3 14
python apply_tasc_to_runcurve.py --csv data/<run>/<cycle>_0.csv

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