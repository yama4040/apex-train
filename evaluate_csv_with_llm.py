import os
import glob
import csv
import json
import time
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. LLM API呼び出し関数（リトライ機能付き）
# ==========================================

def _call_openai_api_with_retry(prompt_text: str, max_retries: int = 3) -> str:
    """API通信処理（エラー時やフリーズ時の自動リトライ機能付き）"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    
    if not api_key or not base_url:
        return "APIキー未設定によるダミー"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    for attempt in range(max_retries):
        try:
            print(f"    [APIリクエスト送信中... (試行 {attempt + 1}/{max_retries})]")
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "あなたは列車の自動運転制御を評価するエキスパートです。必ず指示されたJSONフォーマットのみを出力してください。"},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                timeout=60.0,
            )
            print("    [APIレスポンス受信成功！]")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [API通信エラー]: {e}")
            if attempt < max_retries - 1:
                sleep_time = 5 * (attempt + 1)
                print(f"    -> {sleep_time}秒後に再試行します...")
                time.sleep(sleep_time)
            else:
                print("    -> 最大リトライ回数に達したため、この行の評価をスキップ（0.0）します。")
                return f"エラー発生: {e}"

def call_llm_for_eval(prompt_text: str) -> Tuple[str, float]:
    """直接評価モード用"""
    result_text = _call_openai_api_with_retry(prompt_text)
    if result_text.startswith("エラー発生") or result_text.startswith("APIキー未設定"):
        return result_text, 0.0

    try:
        clean_text = result_text.strip().replace("```json", "").replace("```", "")
        eval_data = json.loads(clean_text)
        reason = eval_data.get("reason", "理由の出力なし")
        reward = float(eval_data.get("reward", 0.0))
        reward = max(0.0, min(1.0, reward))
        return reason, reward
    except Exception as e:
        print(f"    [LLM解析エラー]: {e}\nレスポンス内容: {result_text}")
        return f"解析エラー: {e}", 0.0

# ==========================================
# 2. プロンプト生成関数（変更なし）
# ==========================================
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
目的：必要速度に到達するための安定した加速

高評価
- 力行を継続している
- ノコギリ運転がない
- CBTC制限内

減点
- 不必要な惰行
- 不必要なブレーキ
- 頻繁なノッチ切替

大幅減点
- 先行列車も存在しないのに低速で惰行または減速している

--------------------------------------------------
## 巡航フェーズ
目的：定時性と省エネ性の両立

現在速度が必要速度を十分満たしている場合は惰行を高く評価する。
必要速度は「現在の速度」と「残り時間」から平均速度を算出し、その平均速度を1.3～1.4倍した速度とする。

高評価
- 惰行を活用している
- 制限速度に余裕がある
- ノコギリ運転がない
- 先行列車と後続列車の両方が存在する場合、自列車と先行列車、後続列車の車間がある程度同じである。（誤差±400m程度）



減点
- 必要以上の力行
- 惰行不足
- ノコギリ運転

--------------------------------------------------
## 次駅減速フェーズ
目的：所定停止位置への停止

- 定時性より停止位置精度を優先する。
- 先行列車に接近している場合はCBTCの信号現示に従っているかを重視し先行列車の手前に停止できているかを評価すること。
- 評価には必ずreq_stop_distを使用すること。
- LLM独自の停止距離計算は禁止。

### ブレーキ開始判定
下記の式を用いて評価を行う。
- delta_stop = dist_to_next_station - req_stop_dist

### 評価目安
"次駅減速フェーズ"では，まず先行列車が在線しているかを確認し，在線している場合は下記の"先行列車が居る場合"を優先して評価してください。
先行列車居ない，もしくは十分な距離（600m以上）を保てているある場合は"ブレーキをかけている場合"と"ブレーキをかけていない場合"を参考に評価してください。

先行列車が居る場合
- current_speed ≒ signal_speed （2つの速度の差が±2km/h程度）かつ ブレーキ中
  → 高評価（たとえ，駅の手前に停止する可能性があっても信号現示に従っているとして高く評価してください）
- current_speed > signal_speed かつ ブレーキをかけていない
  → 信号を無視しているとして大幅減点

ブレーキをかけている場合
- 高評価：|delta_stop| ≤ 3m
- やや減点：3m < |delta_stop| ≤ 5m
- 大幅減点：|delta_stop| > 5m

ブレーキをかけていない場合
- 高評価：delta_stop >= 0m（このタイミングでブレーキをかけると駅手前に停車する可能性があるため）
- やや減点：-5m < delta_stop <0m
- 大幅減点：delta_stop < -5m（オーバーランリスクが高いため）

--------------------------------------------------
## 駅停車完了フェーズ
速度0km/hで停止済み状態。停止位置誤差を評価する。

|dist_to_next_station| ≤ 1m
→ reward = 1.0

|dist_to_next_station| > 10m
→ reward = 0.0

1m〜10mの範囲は誤差に応じて段階的に減点する。

--------------------------------------------------
# CBTCについて
--------------------------------------------------
- CBTC停止限界距離は先行列車の50m手前である。
- 先行列車が存在する場合はまずCBTC制御を優先して評価すること。
- ブレーキをかけており駅の手前に停止する可能性があっても、それが先行列車接近に伴うCBTCによる信号現示の低下による減速であれば高く評価してください。

## 評価ルール
- current_speed > signal_speed かつ ブレーキをかけていない
  → 信号を無視しているとして大幅減点
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

## 条件1
holding_time < 10秒

## 条件2
prev_notch_duration < 10秒

##条件3
以下のような反転操作
- 力行→惰行
- 惰行→力行
- 力行→減速
- 減速→力行

## 評価ルール
- 5秒未満 → 大きな減点
- 5〜10秒 → 小さな減点
- 10秒以上 → 減点なし

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

--------------------------------------------------
# 出力ルール
--------------------------------------------------
Step1 制限速度確認
Step2 CBTC確認
Step3 停止位置達成可能性確認
Step4 フェーズ評価
Step5 ノコギリ運転評価
Step6 先行列車と後続列車の運転間隔の適切性
Step7 定時性・省エネ性評価
Step8 総合評価

以上の順序で評価すること。
"""

    current_status = f"""
# 現在の走行状況と運転操作
- 走行フェーズ: {features['phase']}
- **現在の運転操作**: {features['current_notch']} （継続時間: {features['holding_time']} 秒） <-- 【重要】この操作の適切性を評価してください。
- **直前の運転操作**: {features['prev_notch']} （継続時間: {features['prev_notch_duration']} 秒） <-- 【重要】保持時間が共に5秒未満の場合、ノコギリ運転のルールを確認してください。
- 速度情報: 制限速度 {features['speed_limit']} km/h (CBTC信号現示 {features['signal_speed']:.1f} km/h) に対し、現在 {features['current_speed']:.1f} km/h で走行中
- 次駅への情報: 次駅までの残り距離 {features['dist_to_next_station']:.1f} m（マイナスは過走）に対し，定時到着まで残り {features['time_to_next_station']} 秒
- 駅停車に必要なブレーキ距離: {features['req_stop_dist']:.2f} m
- 運行状況: 計画ダイヤに対し {features['delay']} 秒の遅延
- 現在の勾配: {features['current_gradient']} ‰
- 前方の制限情報: {features['next_limit_info']}
- 前方の勾配情報: {features['next_gradient_info']}
- 先行列車の状況: {features['forward_info']}
- 後続列車の状況: {features['backward_info']}
"""
    output_format = """
# 出力指示
reasonでは、Step1〜Step8の分析を踏まえ、現在の運転操作（current_notch）が適切であったかを簡潔に説明してください。
説明は150〜200文字程度とし、rewardは0.0～1.0（0.1刻み）で出力すること。

{
  "reason": "速度は制限内ですが、直前の加速から2.0秒で惰行に切り替えるノコギリ運転が発生しています。また定時到着に余裕があるにもかかわらず加速を続けていたため省エネ性に欠けます。安全性は保たれていますが快適性と効率性の観点から不適切です。",
  "reward": 0.5
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
        "speed_limit", "signal_speed", "current_speed",  
        "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delay", 
        "current_gradient", "next_limit_info", "next_gradient_info",
        "forward_info", "backward_info", "reward", "reason"
    ]

    output_csv_path = os.path.join(output_dir, "llm_evaluated_dataset.csv")

    # 【修正】ファイルの存在チェックを外し、実行のたびに必ず新規ファイルとして作成してヘッダを書き込む（前回のデータは上書きリセットされます）
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f_init:
        writer = csv.writer(f_init)
        writer.writerow(headers)
    print(f"出力ファイル '{output_csv_path}' を新規作成し、ヘッダを書き込みました。")

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

                # LLM評価
                prompt = generate_eval_prompt(features)
                reason, reward = call_llm_for_eval(prompt)

                out_row = [
                    row.get("time", ""), row.get("train_id", ""), features["phase"],
                    features["current_notch"], features["holding_time"],
                    features["prev_notch"], features["prev_notch_duration"],
                    features["speed_limit"], features["signal_speed"], features["current_speed"],
                    features["dist_to_next_station"], features["time_to_next_station"],
                    features["req_stop_dist"], features["delay"],
                    features["current_gradient"], features["next_limit_info"],
                    features["next_gradient_info"], features["forward_info"],
                    features["backward_info"], reward, reason
                ]
                
                # 1行評価するごとに追記モードで開いて保存
                with open(output_csv_path, mode='a', newline='', encoding='utf-8-sig') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(out_row)
                
                time.sleep(1.0) 

        print(f"=== 処理完了: {os.path.basename(file_path)} ===")
    print(f"\n全ての評価が完了しました。")

if __name__ == "__main__":
    process_csv_files()