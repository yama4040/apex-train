import os
import json
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from evaluate_csv_with_llm import generate_eval_prompt
from required_speed import calculate_required_speed

load_dotenv()

SYSTEM_CONTENT = "あなたは列車の自動運転制御を評価するエキスパートです。必ず指示されたJSONフォーマットのみを出力してください。"


def call_and_measure(label: str, prompt_text: str, reasoning_effort: Optional[str] = None,
                      timeout: float = 300.0) -> Dict[str, Any]:
    """指定プロンプトを送信し、応答時間を計測して表示する。結果はdictで返す（比較表作成用）。

    reasoning_effort: "low"/"medium"/"high"を指定すると、gpt-oss系モデルの
    内部思考（reasoning）の強度をクライアント側から上書きする（サーバーが対応していれば有効）。
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    if not api_key or not base_url:
        print("APIキー未設定のため実行できません。")
        return {"label": label, "reasoning_effort": reasoning_effort, "elapsed": None, "error": "APIキー未設定"}

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"\n=== {label} ===")
    print(f"プロンプト文字数: {len(prompt_text)}")
    if reasoning_effort:
        print(f"reasoning_effort: {reasoning_effort}")
    start = time.time()
    try:
        extra_kwargs = {}
        if reasoning_effort:
            extra_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            timeout=timeout,
            **extra_kwargs,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        print(f"応答時間: {elapsed:.2f}秒")
        print(f"応答内容:\n{content}")
        if response.usage:
            print(f"トークン使用量: {response.usage}")

        reward = None
        reason = None
        try:
            clean = content.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean)
            reward = data.get("reward")
            reason = data.get("reason")
        except Exception:
            pass

        return {
            "label": label,
            "reasoning_effort": reasoning_effort,
            "elapsed": elapsed,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "reward": reward,
            "reason": reason,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[エラー発生]（{elapsed:.2f}秒経過）: {e}")
        return {"label": label, "reasoning_effort": reasoning_effort, "elapsed": elapsed, "error": str(e)}


def build_dummy_real_prompt() -> str:
    """generate_eval_promptと同じ形式のダミーデータで実際のプロンプトを再現する"""
    features = {
        "phase": "巡航",
        "current_notch": "惰行",
        "holding_time": 5.0,
        "prev_notch": "力行1",
        "prev_notch_duration": 8.0,
        "speed_limit": 70.0,
        "signal_speed": 70.0,
        "current_speed": 65.0,
        "dist_to_next_station": 800.0,
        "time_to_next_station": 60.0,
        "req_stop_dist": 300.0,
        "delay": 0.0,
        "current_gradient": 0.0,
        "next_limit_info": "なし",
        "next_gradient_info": "なし",
        "forward_info": "なし",
        "backward_info": "なし",
    }
    features["required_speed"] = calculate_required_speed(
        current_speed=features["current_speed"],
        dist_to_next_station=features["dist_to_next_station"],
        time_to_next_station=features["time_to_next_station"],
        speed_limit=features["speed_limit"],
        current_gradient=features["current_gradient"],
    )
    features["delta_stop"] = features["dist_to_next_station"] - features["req_stop_dist"]
    return generate_eval_prompt(features)


if __name__ == "__main__":
    # まず適当な短いプロンプトでAPIが正常に応答するかを確認する。
    sanity_result = call_and_measure("疎通確認（短いプロンプト）", "こんにちは。挨拶だけ返してください。")
    if sanity_result.get("error"):
        print("\n短いプロンプトへの応答に失敗したため、以降の処理を中断します。")
        raise SystemExit(1)

    # reasoning_effortの度合いによる速度・評価結果の差を比較する。
    # default(サーバー既定=フルパワー)は271秒かけて504になることが既に判明しているため、
    # 必要であればEFFORT_LEVELSの先頭に None を追加して比較対象に含めること（数分待つ覚悟が必要）。
    EFFORT_LEVELS = ["low", "medium", "high"]

    real_prompt = build_dummy_real_prompt()
    results = []
    for effort in EFFORT_LEVELS:
        result = call_and_measure(f"実際の評価プロンプト（reasoning_effort={effort}）",
                                   real_prompt, reasoning_effort=effort)
        results.append(result)
        time.sleep(1.0)

    # 比較表を表示
    print("\n\n=== 比較まとめ ===")
    print(f"{'reasoning_effort':<10} {'応答時間(秒)':<12} {'completion_tokens':<18} {'reward':<8} reason")
    for r in results:
        if r.get("error"):
            print(f"{str(r['reasoning_effort']):<10} エラー: {r['error']}")
            continue
        elapsed_str = f"{r['elapsed']:.2f}" if r["elapsed"] is not None else "-"
        tokens_str = str(r.get("completion_tokens", "-"))
        reward_str = str(r.get("reward", "-"))
        reason_str = r.get("reason", "-")
        print(f"{str(r['reasoning_effort']):<10} {elapsed_str:<12} {tokens_str:<18} {reward_str:<8} {reason_str}")
