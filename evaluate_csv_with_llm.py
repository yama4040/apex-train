import os
import glob
import csv
import json
import re
import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

from required_speed import calculate_required_speed

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

def call_llm_for_eval(prompt_text: str, max_retries: int = 3) -> Tuple[str, Optional[float]]:
    """直接評価モード用。通信エラー・レスポンスのフォーマット崩れのどちらでも
    同じリトライ回数の中でAPI呼び出しからやり直す（フォーマット崩れの応答をそのまま解析しても
    意味がないため）。最大リトライ回数を超えても解消しない場合はreward=Noneを返す。"""
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
            return reason, reward
        except Exception as e:
            print(f"    [LLM解析エラー]: {e}\nレスポンス内容: {result_text}")
            last_error = f"解析エラー: {e}"
            if attempt < max_retries - 1:
                print("    -> フォーマット不正のため再試行します...")
                time.sleep(2.0)

    print("    -> 最大リトライ回数に達したため、この行の評価は失敗として扱います（rewardは付与しません）。")
    return f"評価失敗（要再処理）: {last_error}", None

# ==========================================
# 2. プロンプト生成関数
# ==========================================
#評価ルール厳密化（特にreward=0.0について）
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

--------------------------------------------------
# フェーズ別評価基準
--------------------------------------------------
## 加速フェーズ
目的：必要速度（required_speed）に到達するための安定した加速

高評価
- 力行を継続している
- ノコギリ運転がない
- CBTC制限内

減点
- 不必要な惰行
- 不必要なブレーキ
- 頻繁なノッチ切替

大幅減点（reward=0.0）
- 先行列車も存在しないのに低速で惰行または減速している

--------------------------------------------------
## 巡航フェーズ
目的：定時性と省エネ性の両立

現在速度が必要速度を十分満たしている場合は惰行を高く評価する。
必要速度（required_speed）は、加速にかかる時間・惰行による自然減速・ブレーキにかかる時間と距離を
考慮した走行シミュレーションによりあらかじめ算出済みの値である。current_statusのrequired_speedを
必ずそのまま評価に使用し、LLM自身が平均速度などから必要速度を再計算してはならない。

高評価
- 惰行を活用している
- 制限速度に余裕がある
- ノコギリ運転がない
- 先行列車と後続列車の両方が存在する場合、自列車と先行列車、後続列車の車間がある程度同じである。（誤差±400m程度）
- 下り勾配を活用した速度の維持（無駄な加速を行っていない）


減点
- 必要以上の力行
- 惰行不足
- 下り勾配かつ速度が"制限速度 - 10km/h"にも関わらず加速を行っている（勾配を活用できていない）

大幅減点（reward=0.0）
- ノコギリ運転

--------------------------------------------------
## 次駅減速フェーズ
目的：所定停止位置への停止

- 定時性より停止位置精度を優先する。
- 先行列車に接近している場合はCBTCの信号現示に従っているかを重視し先行列車の手前に停止できているかを評価すること。
- 評価には必ずreq_stop_distを使用すること。
- LLM独自の停止距離計算は禁止。

### ブレーキ開始判定
delta_stop（= dist_to_next_station - req_stop_dist）はPython側で算出済みの値である。
current_statusのdelta_stopの値を必ずそのまま評価に使用し、LLM自身がdist_to_next_stationとreq_stop_distから再計算してはならない。

### 評価目安
"次駅減速フェーズ"では，まず先行列車が在線しているかを確認し，在線している場合は下記の"先行列車が居る場合"を優先して評価してください。
先行列車が居ない、もしくは十分な距離（600m以上）を保てている場合は"ブレーキをかけている場合"と"ブレーキをかけていない場合"を参考に評価してください。
また、実際の運転では、ブレーキ応答の遅れなどにより、current_speedとsignal_speed、が完全に一致するこやdelta_stop=0となる場合は少ないです。
そのためcurrent_speedとsignal_speedの速度差が±2km/h程度の場合は「制御上自然な誤差」、delta_stopの値が±2m以内の場合はとして評価してください。

先行列車が居る場合
- current_speed ≒ signal_speed （2つの速度の差が±2km/h程度）かつ ブレーキ中
  → 高評価（たとえ，駅の手前に停止する可能性があっても信号現示に従っているとして高く評価してください）
- current_speed > signal_speed かつ ブレーキをかけていない
  → 信号を無視しているとして大幅減点（reward=0.0）
- delta_stop > 0 かつ 加速をしている （かつ CBTCの信号現示を2km/h以上オーバーしていない）
  → 先行列車が動き出し，自列車も駅に向かって加速が出来ているとして，高く評価してください。

ブレーキをかけている場合
- 高評価：|delta_stop| ≤ 2m
- やや減点：2m < |delta_stop| ≤ 5m
- 大幅減点（reward=0.0）：|delta_stop| > 5m

ブレーキをかけていない場合
- 高評価：delta_stop >= 0m（このタイミングでブレーキをかけると駅手前に停車する可能性があるためブレーキをかけていないのは適切です）
  ※ただし、先行列車も無いのに、低速度（current_speed < 25）で惰行をして停止位置へ進む動きは、遅延を増大させる動きであるためreward=0.0点としてください。<-【重要】
- やや減点：-2m くelta_stop <0m 
- 大幅減点（reward=0.0）：delta_stop < -2m（オーバーランリスクが高いため）

--------------------------------------------------
## 駅停車完了フェーズ
- 速度0km/hで停止済み状態。停止位置誤差を評価する。
- ただし，CBTCの信号現示が0km/hであれば，駅手前であっても高評価とする。

|dist_to_next_station| ≤ 1m
→ reward = 1.0

|dist_to_next_station| > 10m
→ reward = 0.0（CBTCの信号現示が0km/hであれば，先行列車衝突回避としてreward = 1.0とする）

1m〜10mの範囲は誤差に応じて段階的に減点する。

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
  → 高評価（信号現示に従っている運転として高く評価してください）

--------------------------------------------------
# 先行列車解放時の特例
--------------------------------------------------
先行列車待ちで停止した後、signal_speed > 0となった場合は、次駅減速フェーズ中であっても再加速は適切な操作である。
「減速フェーズだから加速は不適切」と判断してはならない。

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

## 評価ルール
下記はprev_notch_durationの値に応じた評価基準である。
- 2秒未満：0.0点
- 2～5秒 → 0.1～0.3点
- 5～7秒 → 0.3～0.5点
- 7秒以上 → 減点なし

--------------------------------------------------
# 即reward0.0ルール
--------------------------------------------------
以下に該当する場合はrewardを必ず0.0とする。

- 制限速度超過
- signal_speed超過
- オーバーラン走行中
- オーバーランほぼ確実
- 駅手前停止ほぼ確実
- 駅停車誤差±10m超
- 先行列車に衝突している場合（先行列車との接近距離が40mを切った場合）
- 先行列車が在線して無いにも関わらず低速度（current_speed < 25）で惰行をする行為（詳しくは減速フェーズの評価ルールを参照）

--------------------------------------------------
# 出力ルール
--------------------------------------------------
以下のStep1〜Step9の順に、必ずすべてのStepを内部で確認すること。
一つのStepの結論が後続のStepの前提になる場合は、その内容を踏まえて次のStepの判断を行うこと。
いずれかのStepを省略・無視して評価してはならない。

Step1 即reward0.0ルール確認（上記「即reward0.0ルール」のいずれかに該当しないかを判断する）
Step2 制限速度確認
Step3 CBTC確認
Step4 停止位置達成可能性確認
Step5 フェーズ評価
Step6 ノコギリ運転評価
Step7 先行列車と後続列車の運転間隔の適切性
Step8 定時性・省エネ性評価
Step9 総合評価（Step1〜Step8を踏まえたrewardの最終判断。Step1で該当ありと判断した場合はrewardを必ず0.0とする）

ただし出力においては、各Stepの推論過程を長文で書く必要はない。
Step1〜Step8それぞれについて、"OK"（問題なし）／"NG"（問題あり。10文字程度の短い補足を括弧内に付けてよい）／"対象外"（評価対象外）
のいずれかを"checks"に必ずすべて記録すること（省略不可）。
そのうえで、reward低下・上昇の決め手となった理由のみを"reason"に一文（100～200文字程度）で簡潔にまとめること。
"""

    current_status = f"""
# 現在の走行状況と運転操作
- 走行フェーズ: {features['phase']}
- **現在の運転操作**: {features['current_notch']} （継続時間: {features['holding_time']} 秒） <-- 【重要】この操作の適切性を評価してください。
- **直前の運転操作**: {features['prev_notch']} （継続時間: {features['prev_notch_duration']} 秒） <-- 【重要】保持時間が共に5秒未満の場合、ノコギリ運転のルールを確認してください。
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
"""
    output_format = """
# 出力指示
"checks"には、Step1〜Step8それぞれの判定（"OK"／"NG(補足)"／"対象外"）を必ずすべて記載すること（キーの省略不可）。
"reason"には、Step1〜Step8の説明を並べるのではなく、rewardを決定づけた要因のみを100〜200文字程度の一文で簡潔に記述すること。
rewardは0.0～1.0（0.1刻み）で出力し、"reason"の内容と整合させること。
なお、"immediate_zero_rule"が"NG"の場合、rewardは必ず0.0とすること。

{
  "checks": {
    "immediate_zero_rule": "OK",
    "speed_limit": "OK",
    "cbtc": "OK",
    "stop_position": "対象外",
    "phase": "OK",
    "sawtooth": "NG(2.0秒で反転)",
    "train_interval": "対象外",
    "punctuality_energy": "NG(必要以上の力行)"
  },
  "reason": "ノコギリ運転に該当し、かつ定時性に余裕があるため力行継続は省エネ性に欠けると判断し0.0とする。",
  "reward": 0.0
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
        "forward_info", "backward_info", "reward", "reason"
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
                    "backward_info": row.get("backward_info", "")
                }

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

                # LLM評価
                prompt = generate_eval_prompt(features)
                reason, reward = call_llm_for_eval(prompt)

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
                    features["backward_info"], reward, reason
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
    process_csv_files()