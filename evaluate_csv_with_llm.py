import os
import glob
import csv
import json
import re
import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

from required_speed import calculate_required_speed, brake_stop_distance_m, calculate_no_stop_target_speed

load_dotenv()

# ==========================================
# 1. LLM API呼び出し関数（リトライ機能付き）
# ==========================================

def _call_openai_api_once(prompt_text: str) -> str:
    """API通信処理を1回だけ実行する（リトライなし）"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")

    if not api_key or not base_url:
        raise RuntimeError("APIキー未設定")

    client = OpenAI(api_key=api_key, base_url=base_url)
    start = time.time()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "あなたは列車の自動運転制御を評価するエキスパートです。必ず指示されたJSONフォーマットのみを出力してください。"},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.0,
        timeout=90.0,
        #extra_body={"reasoning_effort": "low"},
    )
    elapsed = time.time() - start
    print(f"    [応答時間: {elapsed:.2f}秒]")
    return response.choices[0].message.content

def _parse_eval_json(result_text: str) -> Dict[str, Any]:
    """LLM応答からJSONを抽出する。末尾カンマなど軽微なフォーマット崩れは自動修復してから読む。"""
    clean_text = result_text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # 「, }」「, ]」のような末尾カンマを除去して再試行
        repaired = re.sub(r",(\s*[}\]])", r"\1", clean_text)
        return json.loads(repaired)

def call_llm_for_eval(prompt_text: str, max_retries: int = 3) -> Tuple[str, Optional[float], str]:
    """直接評価モード用。通信エラー・レスポンスのフォーマット崩れのどちらでも
    同じリトライ回数の中でAPI呼び出しからやり直す（フォーマット崩れの応答をそのまま解析しても
    意味がないため）。最大リトライ回数を超えても解消しない場合はreward=Noneを返す。
    戻り値は (reason, reward, mode)。modeは運転モードラベル（normal/delay_recovery/anti_mid_stop）。"""
    last_error = "不明なエラー"
    for attempt in range(max_retries):
        print(f"    [APIリクエスト送信中... (試行 {attempt + 1}/{max_retries})]")
        try:
            result_text = _call_openai_api_once(prompt_text)
            print("    [APIレスポンス受信成功！]")
        except Exception as e:
            print(f"    [API通信エラー]: {e}")
            last_error = f"通信エラー: {e}"
            if attempt < max_retries - 1:
                sleep_time = 5 * (attempt + 1)
                print(f"    -> {sleep_time}秒後に再試行します...")
                time.sleep(sleep_time)
            continue

        try:
            eval_data = _parse_eval_json(result_text)
            reason = eval_data.get("reason", "理由の出力なし")
            reward = float(eval_data.get("reward", 0.0))
            reward = max(-1.0, min(1.0, reward))
            mode = eval_data.get("mode", "normal") or "normal"
            return reason, reward, mode
        except Exception as e:
            print(f"    [LLM解析エラー]: {e}\nレスポンス内容: {result_text}")
            last_error = f"解析エラー: {e}"
            if attempt < max_retries - 1:
                print("    -> フォーマット不正のため再試行します...")
                time.sleep(2.0)

    print("    -> 最大リトライ回数に達したため、この行の評価は失敗として扱います（rewardは付与しません）。")
    return f"評価失敗（要再処理）: {last_error}", None, ""

# ==========================================
# 2. プロンプト生成関数
# ==========================================
#評価ルール改訂: 即0.0は安全違反のみに限定し、運転品質（ちんたら・早すぎるブレーキ・停止位置誤差）は
#0.1～0.7の段階減点に変更（ラベルが0.0/1.0に二極化して蒸留NNの較正が崩れるのを防ぐため）
def generate_eval_prompt(features: Dict[str, Any]) -> str:
    """② 直接評価モード用のプロンプト（完全版）"""
    system_instruction = """
あなたは熟練の鉄道運転士として列車の自動運転を評価する「運転監督エキスパート」です。

現在の走行状況とその先の線路状況を分析し、現在の運転操作（current_notch）が適切であったかを0.0（極めて危険・不適切）〜1.0（極めて優秀・適切）の範囲で評価してください。

# 評価対象
重要：rewardは「現在のノッチ操作」が適切かどうかのみを評価すること。運転全体の出来栄えや過去のミス、将来の結果を評価してはならない。

評価対象は以下のみである。
- 現在の走行状況
- 現在の運転操作(current_notch)

# 評価優先順位（絶対）
以下の優先順位で評価すること。
1. 制限速度順守
2. CBTC信号現示順守
3. 停止位置達成可能性
4. フェーズに対するノッチ操作の適切性
5. ノコギリ運転
6. 先行列車と後続列車の運転間隔の適切性
7. 定時性
8. 省エネ性

下位項目が優秀でも、
上位項目に重大な問題がある場合は高評価を与えてはならない。

# 基本運転方針
1駅間の理想的な運転は「力行（加速）→惰行→ブレーキ（減速）」の3段階を簡潔に行うことである。
- 各ノッチは7秒以上保持することが望ましい（短時間での切替の繰り返しはノコギリ運転として減点）。
- 遅延なく駅に正確に停車して運転を完了させることが最も価値が高い。
  低速走行で時間を稼ぐような運転は、たとえ安全であっても決して高く評価してはならない。
  （ただし後述の「駅間停車防止モード」で先行列車に塞がれ、target_speed_no_stop に沿って
   低速で進んでいる場合は、機外停車を回避するための正当な運転でありこの限りではない。）

--------------------------------------------------
# 運転モード（先行列車を考慮した状況判断）
--------------------------------------------------
評価に先立ち、現在の走行状況が下記のどの運転モードかを判定し、判定結果を"mode"として出力する。
判定は以下の優先順位に従い、上位モードの条件が成立していれば下位モードには入らない。

優先順位（高い順）: 安全 ＞ 駅間停車防止モード ＞ 遅延回復モード ＞ 通常運転モード
※安全（制限速度・CBTC現示遵守・衝突回避）はモードに関わらず絶対最優先。モード判定はあくまで
　「加速・巡航フェーズでどの基準速度を目標にするか」「機外停車待機をどう評価するか」を切り替えるものである。

## 通常運転モード（mode = "normal"）
- 条件: 駅間停車防止・遅延回復のいずれにも該当しない標準的な走行。
- 基準速度: required_speed。省エネ性（早めの惰行）を重視する従来通りの評価を行う。

## 遅延回復モード（mode = "delay_recovery"）
- 条件: 計画ダイヤに対する遅延が30秒以上、かつ駅間停車防止に該当しない。
- 基準速度: required_speed。※今回の改訂では評価基準は通常運転モードと同じでよい
　（将来、制限速度近くまでの力行を高評価に差別化予定）。

## 駅間停車防止モード（mode = "anti_mid_stop"）
- 条件: 先行列車が遅延して自列車の前を塞いでおり、このまま required_speed まで加速すると
　先行に追いついて機外停車（駅間停車）に至ると予測される場合。
　判定の強い根拠となる材料:
　  - target_speed_no_stop が required_speed より 5km/h以上低い（＝先行が塞いでいるとPython側が算出）
　  - 先行列車が自列車の次駅を「未発車」、または先行クリア残時間が大きい
　  - 先行列車の遅延が大きく、標準運転間隔に対して先行に追いつく見込みがある
- 基準速度: target_speed_no_stop（機外停車を避ける加速上限）。評価は後述
　「# 駅間停車防止モードの評価基準」に従う。
- 発動と解放: 駅出発時点で発動し、先行が次駅を発車して前方がクリア（先行クリア残時間が小さく
　CBTC現示に余裕がある）になったら解放し、通常/遅延回復モードに戻る。

--------------------------------------------------
# フェーズ別評価基準
--------------------------------------------------
※【モードによる読み替え】以下のフェーズ別基準は通常運転モード・遅延回復モードを前提に記述している。
　mode = "anti_mid_stop"（駅間停車防止モード）の場合は、下記の基準速度 required_speed を
　target_speed_no_stop に読み替え、矛盾する場合は必ず「# 駅間停車防止モードの評価基準」を優先すること。
　特に加速フェーズの「力行継続＝高評価」は、駅間停車防止モードでは target_speed_no_stop を超える力行に対しては
　適用せず、過剰加速として減点する。

## 加速フェーズ
目的：必要速度（required_speed）に到達するための安定した加速

高評価
- 力行を継続している
- ノコギリ運転がない
- CBTC制限内

※このフェーズで直前のノッチが長時間のブレーキ（例: 30秒以上）であるのは、駅停車中の
  停止保持ブレーキであり正当である。これを「不要なブレーキ」「遅延の原因」として
  減点してはならない（評価対象は現在のノッチ操作のみである）。

減点
- 不必要な惰行
- 不必要なブレーキ

大幅減点（reward=0.1～0.3）
- 先行列車も存在しないのに低速で惰行または減速している（遅延を拡大させる運転。速度が低いほど・その状態が長く続いているほど0.1に近づけること）
- 頻繁なノッチ切替（詳細は「ノコギリ運転評価」の基準に従うこと）

--------------------------------------------------
## 巡航フェーズ
目的：定時性と省エネ性の両立

現在速度が必要速度を十分満たしている場合は惰行を高く評価する。
必要速度（required_speed）は、加速にかかる時間・惰行による自然減速・ブレーキにかかる時間と距離を
考慮した走行シミュレーションによりあらかじめ算出済みの値である。current_statusのrequired_speedを
必ずそのまま評価に使用し、LLM自身が平均速度などから必要速度を再計算してはならない。
current_speed ≥ required_speed であれば惰行で定時運行が可能、
current_speed < required_speed であれば力行による加速が必要であることを意味する。

高評価
- 惰行を活用している
- 制限速度に余裕がある
- ノコギリ運転がない
- 先行列車と後続列車の両方が存在する場合、自列車と先行列車、後続列車の車間がある程度同じである。（誤差±400m程度）
- 下り勾配を活用した速度の維持（無駄な加速を行っていない）


減点
- 必要以上の力行（required_speed＋10km/hまでは許容とする）
- 下り勾配で、現在速度がすでに「制限速度−10km/h」以上に達しているにも関わらず力行している（勾配を活用できていない）

大幅減点（reward=0.1～0.3）【巡航フェーズのちんたら運転】
- 先行列車がいない（またはCBTC信号現示に十分な余裕がある）にも関わらず、
  required_speedを大きく下回る低速への減速・ブレーキ、または低速のままの走行。
  惰行・ブレーキ、および小刻みな力行（短時間で保持せず速度回復の意思がないもの）を対象とする。
  特に「ブレーキで必要もなく速度を落とす」操作は遅延を直接拡大させるため必ずこの対象とすること。
  current_speedがrequired_speedから下に乖離しているほど0.1に近づけること。
  ※持続的な力行による速度回復はこの対象外である（低速状態からの回復は正しい操作。
  ただし先行列車がいないのに直前の自らの長時間ブレーキで減速していた場合は、
  次の【自ら減速した後の再力行】を適用すること）。

大幅減点（reward=0.1～0.3）【自ら減速した後の再力行】
- 先行列車・制限速度・信号現示のいずれの理由もないのに、直前の長時間ブレーキ
  （prev_notchがブレーキ（減速）かつprev_notch_duration≥7秒）で大きく減速し、
  その直後から力行している場合。自ら速度を捨てたうえで「必要速度への加速」を装う運転であり、
  減速と再加速でエネルギーを二重に浪費するため、current_speed < required_speedであっても
  高く評価してはならない。required_speedからの下方乖離が大きいほど（=より深く減速していたほど）
  0.1に近づけること。
  【適用範囲の限定・厳守】このルールは「走行フェーズが巡航フェーズ」かつ「先行列車なし
  （または600m超）」の場合のみに適用する。以下には絶対に適用してはならない：
  - 駅出発直後の加速フェーズ（直前のブレーキは駅停車時の停止保持であり正当）
  - 先行列車が600m以内に存在する場合（直前のブレーキはCBTC対応の可能性があり、
    力行による速度回復は「先行列車解放時の特例」により高評価とする）

大幅減点
- ノコギリ運転（詳細は「ノコギリ運転評価」の基準に従うこと）

--------------------------------------------------
## 次駅減速フェーズ
目的：所定停止位置への停止

- 定時性より停止位置精度を優先する。
- 先行列車に接近している場合はCBTCの信号現示に従っているかを重視し先行列車の手前に停止できているかを評価すること。
- 評価には必ずreq_stop_distを使用すること。
- LLM独自の停止距離計算は禁止。
- required_speedはこのフェーズでは評価に使用しないこと（定時到着までの残り時間が0の場合、制限速度と同値が表示される仕様のため）。

### ブレーキ開始判定
delta_stop（= dist_to_next_station - req_stop_dist）はPython側で算出済みの値である。
current_statusのdelta_stopの値を必ずそのまま評価に使用し、LLM自身がdist_to_next_stationとreq_stop_distから再計算してはならない。

### 評価目安
"次駅減速フェーズ"では，まず先行列車が在線しているかを確認し，在線している場合は下記の"先行列車が居る場合"を優先して評価してください。
先行列車が居ない、もしくは十分な距離（600m以上）を保てている場合は"ブレーキをかけている場合"と"ブレーキをかけていない場合"を参考に評価してください。
また、実際の運転では、ブレーキ応答の遅れなどにより、current_speedとsignal_speedが完全に一致することやdelta_stop=0となる場合は少ないです。
そのためcurrent_speedとsignal_speedの速度差が±2km/h程度の場合、およびdelta_stopが±5m以内の場合は「制御上自然な誤差」として評価してください。

先行列車が居る場合
- current_speed ≒ signal_speed （2つの速度の差が±2km/h程度）かつ ブレーキ中
  → 高評価（たとえ，駅の手前に停止する可能性があっても信号現示に従っているとして高く評価してください）
- current_speed > signal_speed かつ ブレーキをかけていない
  → 信号を無視しているとして大幅減点（reward=0.0）
- delta_stop > 0 かつ 加速をしている （かつ CBTCの信号現示を2km/h以上オーバーしていない）
  → 先行列車が動き出し，自列車も駅に向かって加速が出来ているとして，高く評価してください。

ブレーキをかけている場合（停止位置誤差の大きさに応じて必ず段階的に評価すること）
※評価の厳しさは現在速度（current_speed）で変える。高速域では行動決定が1秒間隔で
  1ステップ10～19m進むため、ブレーキ開始タイミングに±5m程度の量子化誤差が必然的に生じる。
  一方、低速域（駅への最終進入）は0.1秒間隔で数cm単位の精密な制御が可能なため、
  必要ブレーキ距離とのマージン（delta_stop）をより小さく要求する。

【高速域: current_speed ≥ 15km/h】（量子化誤差を踏まえた基準）
- 高評価（reward=0.8～1.0）：|delta_stop| ≤ 5m
- やや減点（reward=0.5～0.7）：5m < delta_stop ≤ 15m（誤差が小さいほど0.7に近づけること）
- 減点（reward=0.3～0.5）：15m < delta_stop ≤ 30m（ブレーキ開始が早すぎる。誤差が小さいほど0.5に近づけること）
- 大幅減点（reward=0.1～0.2）：delta_stop > 30m（大幅に手前に停止することがほぼ確実）
- 大幅減点（reward=0.0）：delta_stop < -5m（オーバーランがほぼ確実。安全に関わるため0.0とする）

【低速域: current_speed < 15km/h】（精密制御が可能なため、より厳しく評価する）
- 高評価（reward=0.9～1.0）：|delta_stop| ≤ 1m
- やや減点（reward=0.7～0.8）：1m < |delta_stop| ≤ 2m
- 減点（reward=0.4～0.6）：2m < delta_stop ≤ 4m（停止位置が手前になる。誤差が小さいほど0.6に近づけること）
- 大幅減点（reward=0.2～0.3）：delta_stop > 4m（低速なのに大幅に手前で止まる。ブレーキが強すぎ/早すぎ）
- 大幅減点（reward=0.0）：delta_stop < -1m（オーバーランがほぼ確実。低速なので回避可能なはずであり厳しく減点する）

ブレーキをかけていない場合
- 高評価：delta_stop >= 0m（このタイミングでブレーキをかけると駅手前に停車する可能性があるためブレーキをかけていないのは適切です）
  ※ただし、先行列車も無いのに、低速度（current_speed < 25）で惰行または小刻みな力行をして停止位置へ進む動きは、
  遅延を増大させる「ちんたら運転」であるためreward=0.1～0.3としてください（速度が低いほど・遅延が大きいほど0.1に近づけること）。<-【重要】
  低速のまま停止位置に到達しようとする運転は、惰行・力行のどちらのノッチであっても高く評価してはならない。
  （高評価に値するのは、停止位置に向けて適切なタイミングでブレーキをかけて停止する運転のみである。
  適切なタイミングの許容幅は上記「ブレーキをかけている場合」と同じく速度依存で、高速域±5m・低速域±1mとする）
- やや減点：-2m < delta_stop < 0m
- 大幅減点（reward=0.0）：delta_stop < -2m（オーバーランリスクが高いため）

--------------------------------------------------
## 駅停車完了フェーズ
- 速度0km/hで停止済み状態。停止位置誤差を評価する。
- 誤差の大きさに応じて必ず以下の段階評価とすること（一律0.0にしてはならない）。
- ただし，CBTCの信号現示が0km/hであれば，駅手前であっても高評価とする。

- |dist_to_next_station| ≤ 1m → reward = 1.0
- 1m < |dist_to_next_station| ≤ 3m → reward = 0.8
- 3m < |dist_to_next_station| ≤ 5m → reward = 0.5
- 5m < |dist_to_next_station| ≤ 10m → reward = 0.2
- |dist_to_next_station| > 10m → reward = 0.0

（CBTCの信号現示が0km/hであれば，停止位置誤差に関わらず先行列車衝突回避としてreward = 1.0とする）
※ただし駅間停車防止モード（mode = "anti_mid_stop"）で、駅ではなく駅間で機外停車して先行待ちしている状態は
　この例外の対象外であり、中立 reward=0.5 とする（上記「# 駅間停車防止モードの評価基準」を優先）。


--------------------------------------------------
# 駅間停車防止モードの評価基準（mode = "anti_mid_stop" の場合のみ適用）
--------------------------------------------------
このモードでは先行列車が塞いでいるため、required_speed まで加速すると機外停車に至る。
基準速度は required_speed ではなく target_speed_no_stop（機外停車を避ける加速上限）を用いる。
目的は「早めに惰行して機外停車を未然に回避しつつ、無駄に遅くなりすぎない」運転である。
評価はあくまで現在のノッチ操作に対して行う（機外停車という結果そのものを遡って罰しない）。

【最重要・基準速度の徹底】このモードでは先行列車が塞ぐため、current_speed が required_speed を
下回って走行するのが正しい運転である。current_speed が required_speed を下回ること自体を
「遅延リスク」「ちんたら運転」として減点してはならない。このモードの速度評価の基準は
required_speed ではなく target_speed_no_stop のみであり、required_speed との比較は用いないこと。

## 加速・巡航フェーズ（速度・余裕がある局面）
- 現在速度が target_speed_no_stop ＋5km/h 以内（上限付近まで）の加速・維持 → 高評価
- 現在速度が target_speed_no_stop を ＋5km/h を超えて上回る方向への力行
  → 減点（先行に追いついて機外停車を招く過剰加速。超過幅が大きいほど0.1に近づける）
- 現在速度が target_speed_no_stop を超えている状態からの早めの惰行（上限へ収束させる） → 高評価
- 現在速度が target_speed_no_stop を 10km/h を超えて下回る過度な低速・惰行のしすぎ
  → 減点（無駄な遅延。ただし停止回避のための余裕は許容する）
  ※このモードでは先行が塞いでいるため、target_speed_no_stop 付近までの低速走行は「ちんたら運転」ではなく
    正当である。ちんたら運転として減点するのは、target_speed_no_stop を 10km/h を超えて下回る場合のみとすること。
  ※【重要・力行の除外】力行（加速）で target_speed_no_stop に向けて速度を回復している最中は、たとえ
    現在速度が target_speed_no_stop を 10km/h 超下回っていても、この低速減点の対象外とし高評価とする
    （上限に向けた加速は適切な操作である）。低速減点の対象は「惰行・ブレーキ・小刻みな力行（短時間で
    保持せず速度回復の意思がないもの）」に限る。

## 減速・CBTC接近時（先行に迫っている局面）
- CBTC現示の低下に沿って減速している最中のブレーキ操作
  → 高評価（衝突回避・その瞬間の正しい操作。これは全モード共通）

## 機外停車して先行待ちしている状態（＝駅間での停止・速度0・CBTC現示≒0km/h）
- 【機外停車の判定】速度がほぼ0km/hで、かつ次駅まで十分な距離が残っている
  （dist_to_next_station が停止位置精度の範囲を大きく超える。目安として概ね30m以上）にもかかわらず停止している状態を指す。
  次駅の所定停止位置付近（dist_to_next_station がほぼ0）での停止は「駅停車」であり機外停車ではない（駅停車完了フェーズの基準で評価する）。
- この機外停車待機は、このモードでは中立（reward=0.5前後）とする。安全ではあるが、本来は早めの惰行で回避すべきだった事態のため、
  高評価（1.0）にはしない。動機付けを「早め惰行で流して進む(>0.5) ＞ 機外停車待機(0.5) ＞ 衝突(0.0)」の
  単調な関係にするためである。
  ※先行列車に衝突している場合（接近距離40m未満）は全モードで reward=0.0（変更なし）。

## 先行がクリアした後（解放）
- 先行が次駅を発車し先行クリア残時間が小さくなると、target_speed_no_stop は required_speed 付近まで上昇する。
  この状態での駅に向けた再加速は適切であり高評価とする（「先行列車解放時の特例」と同じ扱い）。

--------------------------------------------------
# CBTCについて
--------------------------------------------------
- CBTC停止限界距離は先行列車の50m手前である。
- 先行列車が存在する場合はまずCBTC制御を優先して評価すること。
- ブレーキをかけており駅の手前に停止する可能性があっても、それが先行列車接近に伴うCBTCによる信号現示の低下による減速であれば高く評価してください。

## 評価ルール
- current_speed > signal_speed かつ ブレーキをかけていない
  → 信号を無視しているとして大幅減点（reward=0.0）
- current_speed ≒ signal_speed （2つの速度の差が±2km/h程度）かつ ブレーキ中
  → 高評価（たとえ，駅の手前に停止する可能性があっても信号現示に従っているとして高く評価してください）
- current_speed < signal_speed
  → CBTC順守の項目としては問題なし（Step3はOK）。
    【重要】ただしこれは「上限を守っているか」の判定にすぎない。信号現示を下回ってさえいれば
    高評価になるわけではなく、低速走行そのものを高評価とする根拠にしてはならない。
    信号現示より大幅に低い速度での走行・減速が適切かどうかは、各フェーズの評価基準
    （required_speedとの比較、ちんたら運転ルール）で別途判断すること。

--------------------------------------------------
# 先行列車解放時の特例
--------------------------------------------------
先行列車待ちで停止した後、signal_speed > 0となった場合は、次駅減速フェーズ中であっても再加速は適切な操作である。
「減速フェーズだから加速は不適切」と判断してはならない。
【重要】この特例は、巡航フェーズの「ちんたら運転」「自ら減速した後の再力行」の減点より優先する。
先行列車がいる（またはいた）影響で低速になった後の、力行による速度回復を減点してはならない。

--------------------------------------------------
# ノコギリ運転評価
--------------------------------------------------
以下をすべて満たした場合のみノコギリ運転として減点する。
たとえその行動が定時到着のための加速や，減速に繋がっていたとしても乗り心地やエネルギー効率性に影響するとして減点してください。

## 条件1
holding_time < 7秒

## 条件2
prev_notch_duration < 7秒

## 条件3
以下のような反転操作
- 力行→惰行
- 惰行→力行
- 力行→減速
- 減速→力行
- 減速→惰行
- 惰行→減速

## 条件4
その反転に合理的理由が存在しない
※合理的理由の例：CBTC信号現示の低下への対応、および駅手前100m以内・低速（25km/h未満）での
  停止位置調整を目的としたブレーキ⇔惰行の切替。これらはノコギリ運転として減点しないこと。

## 評価ルール
条件1〜4をすべて満たしノコギリ運転と判定した場合、prev_notch_durationの値に応じてrewardの上限を以下とする
（他の評価項目が優秀でも、この上限を超えるrewardを与えてはならない）。
- 2秒未満 → 0.0点
- 2～5秒 → 0.1～0.3点
- 5～7秒 → 0.3～0.5点
- 7秒以上 → ノコギリ運転に該当せず、減点なし

--------------------------------------------------
# 即reward0.0ルール
--------------------------------------------------
このルールは「安全に直結する違反」のみを対象とする。
以下に該当する場合はrewardを必ず0.0とする。

- 制限速度超過
- signal_speed超過
- オーバーラン走行中
- オーバーランほぼ確実（ブレーキをかけていないのに delta_stop < -2m、またはブレーキ中でも delta_stop < -5m）
- 先行列車に衝突している場合（先行列車との接近距離が40mを切った場合）

※以下の「運転品質」の問題はこのルールの対象外である。一律0.0とせず、
各フェーズの評価基準・ノコギリ運転評価に従って0.1～0.7の段階的な減点で評価すること。
- ちんたら運転（低速走行で時間を稼ぐ運転。減速フェーズ・巡航フェーズとも。
  必要のないブレーキによる減速も含む）→ reward=0.1～0.3
- 自ら減速した後の再力行（巡航フェーズで理由なく長時間ブレーキ後に力行）→ reward=0.1～0.3
- 早すぎるブレーキ（駅手前停止の可能性）→ delta_stopの誤差に応じて段階減点
- 停車完了時の停止位置誤差 → 誤差に応じて段階減点（駅停車完了フェーズの基準を参照）
- ノコギリ運転 → prev_notch_durationに応じた上限を適用

--------------------------------------------------
# 出力ルール
--------------------------------------------------
以下のStep1〜Step10の順に、必ずすべてのStepを内部で確認すること。
一つのStepの結論が後続のStepの前提になる場合は、その内容を踏まえて次のStepの判断を行うこと。
いずれかのStepを省略・無視して評価してはならない。

Step1 即reward0.0ルール確認（上記「即reward0.0ルール」のいずれかに該当しないかを判断する）
Step2 制限速度確認
Step3 CBTC確認
Step4 運転モード判定（上記「# 運転モード」の優先順位に従い mode を決定する。
       駅間停車防止モードの場合、以降のフェーズ評価で用いる基準速度を required_speed から
       target_speed_no_stop に切り替える）
Step5 停止位置達成可能性確認
Step6 フェーズ評価（Step4で判定したモードに応じた基準速度を使用する。
       mode = "anti_mid_stop" の場合は「# 駅間停車防止モードの評価基準」に従う）
Step7 ノコギリ運転評価
Step8 先行列車と後続列車の運転間隔の適切性
Step9 定時性・省エネ性評価
Step10 総合評価（Step1〜Step9を踏まえたrewardの最終判断。Step1で該当ありと判断した場合はrewardを必ず0.0とする）

ただし出力においては、各Stepの推論過程を長文で書く必要はない。
Step4の判定結果は "mode"（"normal"／"delay_recovery"／"anti_mid_stop" のいずれか）として必ず出力すること。
残りのStepそれぞれについて、"OK"（問題なし）／"NG"（問題あり。10文字程度の短い補足を括弧内に付けてよい）／"対象外"（評価対象外）
のいずれかを"checks"に必ずすべて記録すること（省略不可）。
そのうえで、reward低下・上昇の決め手となった理由のみを"reason"に一文（50～100文字程度）で簡潔にまとめること。
"""

    current_status = f"""
# 現在の走行状況と運転操作
- 走行フェーズ: {features['phase']}
- **現在の運転操作**: {features['current_notch']} （継続時間: {features['holding_time']} 秒） <-- 【重要】この操作の適切性を評価してください。
- **直前の運転操作**: {features['prev_notch']} （継続時間: {features['prev_notch_duration']} 秒） <-- 【重要】保持時間が共に7秒未満の場合、ノコギリ運転のルールを確認してください。
- 速度情報: 制限速度 {features['speed_limit']} km/h (CBTC信号現示 {features['signal_speed']:.1f} km/h) に対し、現在 {features['current_speed']:.1f} km/h で走行中
- 定時運行に必要な巡航速度（required_speed、算出済み。惰行への切替目安）: {features['required_speed']:.1f} km/h
- 次駅への情報: 次駅までの残り距離 {features['dist_to_next_station']:.1f} m（マイナスは過走）に対し，定時到着まで残り {features['time_to_next_station']} 秒
- 駅停車に必要なブレーキ距離: {features['req_stop_dist']:.2f} m
- 停止余裕距離（delta_stop = dist_to_next_station - req_stop_dist、算出済み。ブレーキ開始判定に使用）: {features['delta_stop']:.2f} m
- 運行状況: 計画ダイヤに対し {features['delay']} 秒の遅延
- 現在の勾配: {features['current_gradient']} ‰
- 前方の制限情報: {features['next_limit_info']}
- 前方の勾配情報: {features['next_gradient_info']}
- 先行列車の状況: {features['forward_info']}
- 後続列車の状況: {features['backward_info']}

## 先行列車スナップショット（モード判定・駅間停車防止モード用）
- 先行列車の遅延: {features['forward_train_delay']:.0f} 秒（正＝先行が遅れている）
- 先行列車が自列車の次駅を発車済みか: {features['forward_departed_next'] or '不明'}
- 標準運転間隔（先行が自列車より何秒先に出発したか）: {features['standard_headway']:.0f} 秒
- 先行クリア残時間（先行が自列車の次駅を発車するまでの残り秒数、算出済み）: {features['forward_clear_remaining_time']:.0f} 秒
- 機外停車を避ける加速上限（target_speed_no_stop、算出済み。駅間停車防止モードでの惰行/加速の基準速度）: {features['target_speed_no_stop']:.1f} km/h
"""
    output_format = """
# 出力指示
"mode"には、Step4で判定した運転モード（"normal"／"delay_recovery"／"anti_mid_stop" のいずれか）を必ず記載すること。
"checks"には、Step1〜Step3・Step5〜Step9それぞれの判定（"OK"／"NG(補足)"／"対象外"）を必ずすべて記載すること（キーの省略不可）。
"reason"には、各Stepの説明を並べるのではなく、rewardを決定づけた要因のみを50〜100文字程度の一文で簡潔に記述すること。
rewardは0.0～1.0（0.1刻み）で出力し、"reason"の内容と整合させること。
なお、"immediate_zero_rule"が"NG"の場合、rewardは必ず0.0とすること。

{
  "mode": "anti_mid_stop",
  "checks": {
    "immediate_zero_rule": "OK",
    "speed_limit": "OK",
    "cbtc": "OK",
    "stop_position": "対象外",
    "phase": "NG(上限超えの力行)",
    "sawtooth": "OK",
    "train_interval": "対象外",
    "punctuality_energy": "対象外"
  },
  "reason": "先行が塞ぎ駅間停車防止モード。target_speed_no_stopを超える力行は機外停車を招くため減点し0.2とする。",
  "reward": 0.2
}
"""
    return system_instruction + current_status + output_format

# ==========================================
# 3. CSV処理（ヘッダの確実な書き込み）
# ==========================================

def process_csv_files(input_dir="評価用csv", output_dir="評価済ログ"):
    if not os.path.exists(input_dir):
        print(f"エラー: ディレクトリ '{input_dir}' が見つかりません。")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = sorted(csv_files) # <--- これを追加
    
    if not csv_files:
        print(f"'{input_dir}' の中にCSVファイルがありません。")
        return

    headers = [
        "time", "train_id", "phase", "current_notch", "holding_time",
        "prev_notch", "prev_notch_duration",
        "speed_limit", "signal_speed", "current_speed", "required_speed",
        "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delay",
        "current_gradient", "next_limit_info", "next_gradient_info",
        "forward_info", "backward_info",
        # ▼ 先行スナップショット＋加速上限（駅間停車防止モード用・二重蒸留の入力/ラベル）
        "forward_train_delay", "forward_clear_remaining_time", "forward_departed_next",
        "standard_headway", "target_speed_no_stop", "mode",
        "reward", "reason"
    ]

    output_csv_path = os.path.join(output_dir, "llm_evaluated_dataset.csv")
    failed_csv_path = os.path.join(output_dir, "llm_evaluation_failed_rows.csv")

    # 【修正】ファイルの存在チェックを外し、実行のたびに必ず新規ファイルとして作成してヘッダを書き込む（前回のデータは上書きリセットされます）
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f_init:
        writer = csv.writer(f_init)
        writer.writerow(headers)
    print(f"出力ファイル '{output_csv_path}' を新規作成し、ヘッダを書き込みました。")

    # 通信エラー・フォーマット崩れが最大リトライ回数を超えても解消しなかった行を退避するファイル
    # （reward=0.0として本来のデータセットに混入させると、正常な低評価と区別できなくなるため）
    with open(failed_csv_path, mode='w', newline='', encoding='utf-8-sig') as f_init:
        writer = csv.writer(f_init)
        writer.writerow(["source_file"] + headers)

    for file_path in csv_files:
        print(f"\n=== 処理開始: {os.path.basename(file_path)} ===")
        with open(file_path, mode='r', encoding='utf-8-sig') as f_in:
            reader = csv.DictReader(f_in)
            
            for row_idx, row in enumerate(reader):
                print(f"[{os.path.basename(file_path)}] 行番号: {row_idx + 1} (time: {row.get('time', 'N/A')}) を処理中...")
                
                features = {
                    "phase": row.get("phase", ""),
                    "current_notch": row.get("current_notch", ""),
                    "holding_time": float(row.get("holding_time", 0.0) or 0.0),
                    "prev_notch": row.get("prev_notch", ""),
                    "prev_notch_duration": float(row.get("prev_notch_duration", 0.0) or 0.0),
                    "speed_limit": float(row.get("speed_limit", 0.0) or 0.0),
                    "signal_speed": float(row.get("signal_speed", 0.0) or 0.0),
                    "current_speed": float(row.get("current_speed", 0.0) or 0.0),
                    "dist_to_next_station": float(row.get("dist_to_next_station", 0.0) or 0.0),
                    "time_to_next_station": float(row.get("time_to_next_station", 0.0) or 0.0),
                    "req_stop_dist": float(row.get("req_stop_dist", 0.0) or 0.0),
                    "delay": float(row.get("delay", 0.0) or 0.0),
                    "current_gradient": float(row.get("current_gradient", 0.0) or 0.0),
                    "next_limit_info": row.get("next_limit_info", ""),
                    "next_gradient_info": row.get("next_gradient_info", ""),
                    "forward_info": row.get("forward_info", ""),
                    "backward_info": row.get("backward_info", ""),
                    # ▼ 先行スナップショット（駅間停車防止モード用・新CSV列）。
                    #   旧CSV／先行なしの場合は列が無く、下記デフォルトで通常モード相当に縮退する。
                    "forward_train_delay": float(row.get("forward_train_delay", 0.0) or 0.0),
                    "forward_clear_remaining_time": float(row.get("forward_clear_remaining_time", 0.0) or 0.0),
                    "forward_departed_next": row.get("forward_departed_next", ""),
                    "standard_headway": float(row.get("standard_headway", 0.0) or 0.0),
                }

                # 【重要】旧CSVのreq_stop_dist列は「減速度2.5km/h/s一定＋空走1秒」の簡易モデルで
                # 算出されており、走行抵抗・勾配抵抗を無視しているため実挙動と一致しない。
                # delta_stopに基づくブレーキ開始タイミングの評価がずれないよう、
                # train.pyの実ダイナミクスと同一の物理モデルで再計算して上書きする。
                features["req_stop_dist"] = brake_stop_distance_m(
                    features["current_speed"], features["current_gradient"]
                )

                # 必要速度（巡航速度）の算出
                features["required_speed"] = calculate_required_speed(
                    current_speed=features["current_speed"],
                    dist_to_next_station=features["dist_to_next_station"],
                    time_to_next_station=features["time_to_next_station"],
                    speed_limit=features["speed_limit"],
                    current_gradient=features["current_gradient"],
                )

                # 停止余裕距離（delta_stop）の算出。LLMには計算済みの値として渡し、再計算させない。
                features["delta_stop"] = features["dist_to_next_station"] - features["req_stop_dist"]

                # 機外停車回避の加速上限（駅間停車防止モードの基準速度）。
                # 先行クリア残時間0（先行なし・クリア済み）の場合は required_speed と同値になる。
                features["target_speed_no_stop"] = calculate_no_stop_target_speed(
                    current_speed=features["current_speed"],
                    dist_to_next_station=features["dist_to_next_station"],
                    time_to_next_station=features["time_to_next_station"],
                    forward_clear_remaining_time=features["forward_clear_remaining_time"],
                    speed_limit=features["speed_limit"],
                    current_gradient=features["current_gradient"],
                )

                # LLM評価
                prompt = generate_eval_prompt(features)
                reason, reward, mode = call_llm_for_eval(prompt)

                out_row = [
                    row.get("time", ""), row.get("train_id", ""), features["phase"],
                    features["current_notch"], features["holding_time"],
                    features["prev_notch"], features["prev_notch_duration"],
                    features["speed_limit"], features["signal_speed"], features["current_speed"],
                    round(features["required_speed"], 1),
                    features["dist_to_next_station"], features["time_to_next_station"],
                    features["req_stop_dist"], features["delay"],
                    features["current_gradient"], features["next_limit_info"],
                    features["next_gradient_info"], features["forward_info"],
                    features["backward_info"],
                    round(features["forward_train_delay"], 1),
                    round(features["forward_clear_remaining_time"], 1),
                    features["forward_departed_next"],
                    round(features["standard_headway"], 1),
                    round(features["target_speed_no_stop"], 1), mode,
                    reward, reason
                ]

                if reward is None:
                    # 最大リトライ回数を超えても評価に失敗した行は本データセットに含めず退避する
                    print("    [警告] この行は評価に失敗したため、メインデータセットには含めず退避しました。")
                    with open(failed_csv_path, mode='a', newline='', encoding='utf-8-sig') as f_fail:
                        writer = csv.writer(f_fail)
                        writer.writerow([os.path.basename(file_path)] + out_row)
                else:
                    # 1行評価するごとに追記モードで開いて保存
                    with open(output_csv_path, mode='a', newline='', encoding='utf-8-sig') as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow(out_row)

                time.sleep(1.0)

        print(f"=== 処理完了: {os.path.basename(file_path)} ===")
    print(f"\n全ての評価が完了しました。")

if __name__ == "__main__":
    import sys
    # 使い方:
    #   python evaluate_csv_with_llm.py                     … 既定（入力=評価用csv / 出力=評価済ログ）
    #   python evaluate_csv_with_llm.py <入力dir> [出力dir]  … 3PC並列評価などでdirを指定
    #     例: python evaluate_csv_with_llm.py 評価用csv_先行あり_20260724/PC1 評価済ログ_PC1
    args = sys.argv[1:]
    if len(args) == 0:
        process_csv_files()
    elif len(args) == 1:
        process_csv_files(input_dir=args[0])
    else:
        process_csv_files(input_dir=args[0], output_dir=args[1])