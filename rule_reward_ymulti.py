# -*- coding: utf-8 -*-
"""
暫定ルール報酬（複数駅間版・フェーズ2用）。

【位置づけ】CLAUDE.md の制約「報酬はLLMを蒸留したNNの出力のみ」に対する**一時的な例外**である。
報酬NNが用意できるまで環境・DQNの構造を検証するために使い、フェーズ3で必ず
`direct_reward_predictor_ymulti`（LLM蒸留NN）へ置き換える。

同時に、この関数は `prompt_ymulti.py` に書く評価軸の**実行可能な下書き**でもある。
両者が食い違うとルール報酬で学習した方策とLLM評価が矛盾するため、
プロンプトを直したらこちらも直すこと（逆も同じ）。

戻り値は 0.0〜1.0。0.5 が中立で、環境側でゼロ中心化される。
"""
import config_ymulti as CFG
import reward_features_ymulti as rf


def _f(raw, key, default=0.0):
    return rf._f(raw, key, default)


def _s(raw, key):
    return rf._s(raw, key)


# =============================================================================
# 駅停車中の発車判断
# =============================================================================
def evaluate_hold_at_station(raw):
    """駅停車中の「発車するか／待機を続けるか」を評価する。

    ユーザー指定の評価基準:
      ・通常時 … 標準停車時間（30秒）以内の停車か
      ・先行あり … 先行が通常どおりなら通常時と同じ。標準停車からの超過が長くなるほど減点し、
                   停車1分（＝標準+30秒）で最低評価。
                   ただし先行が次の駅で長く停車する要素がある場合は、駅にとどまることを高く評価する。
                   発車は先行が運転を再開した後。
    """
    dwell = _f(raw, "dwell_elapsed")
    dwell_min = _f(raw, "dwell_min", CFG.DWELL_MIN)
    notch = _s(raw, "current_notch")
    departing = (notch == "力行（加速）中")

    # 最低停車時間までは乗降のため発車できない＝行動の余地がないので中立
    if dwell < dwell_min:
        return 0.5

    clear_remaining = _f(raw, "forward_clear_remaining_time")
    reach = _f(raw, "time_to_stop_limit")
    observed = _f(raw, "forward_observed_delay")
    f_dwell = _f(raw, "forward_dwell_elapsed")
    departed = _s(raw, "forward_departed_next") == "発車済み"
    f_exist, _d, _v = rf.extract_forward_info(_s(raw, "forward_info"))

    over = max(0.0, dwell - CFG.STD_DWELL)
    # 標準停車からの超過に対する減点係数（0秒で1.0 → DWELL_PENALTY_FULL_S 秒で0.0）
    decay = max(0.0, 1.0 - over / CFG.DWELL_PENALTY_FULL_S)

    # --- 先行がいない／既に発車済み＝通常時 ---
    if f_exist < 0.5 or departed or clear_remaining <= 0.0:
        return 0.9 if departing else 0.35 * decay

    # --- 塞ぎが確定している: 今発車すると停止限界に着く前に先行がまだ次駅にいる ---
    #     「先行が次の駅で長く停車する要素がある」＝標準停車を超えて停車中と観測できている状態。
    blocked = (clear_remaining > reach) or (observed > 0.0)
    if blocked:
        # 駅にとどまることを高く評価する（駅間に列車を溜めない運転整理の原則にも合致）
        return 0.85 if not departing else 0.1

    # --- まだ判断できない: 先行が次駅に到着して間もなく、延着かどうか不明 ---
    unknown = (f_dwell > 0.0 and observed <= 0.0) or (clear_remaining > reach - 20.0)
    if unknown:
        # 待てば情報が増えるので短時間の待機は許容する。ただし長引くほど下げる。
        return 0.5 if departing else 0.35 + 0.25 * decay

    # --- 塞ぎなし＝定時発車が最良 ---
    return 0.9 if departing else 0.35 * decay


# =============================================================================
# 走行中
# =============================================================================
def evaluate_running(raw, mode):
    speed = _f(raw, "current_speed")
    limit = _f(raw, "speed_limit")
    signal = _f(raw, "signal_speed")
    required = _f(raw, "required_speed")
    target_ns = _f(raw, "target_speed_no_stop", required)
    v_std = _f(raw, "v_std")
    dist = _f(raw, "dist_to_next_station")
    req_stop = _f(raw, "req_stop_dist")
    delta_stop = dist - req_stop
    phase = _s(raw, "phase")
    notch = _s(raw, "current_notch")
    prev = _s(raw, "prev_notch")
    holding = _f(raw, "holding_time")
    prev_dur = _f(raw, "prev_notch_duration")
    coast_ok = _f(raw, "coast_reachable") >= 0.5
    c_acc = _f(raw, "coast_accel")
    p_acc = _f(raw, "power_accel")

    accel = (notch == "力行（加速）中")
    brake = (notch == "ブレーキ（減速）中")
    coast = (notch == "惰行中")

    # --- 絶対ルール: 現示・制限の超過 ---
    if limit > 0.0 and speed > limit + 0.5:
        return 0.0 if accel else 0.3
    if signal > 0.0 and speed > signal + 0.5:
        return 0.0 if accel else 0.3

    # --- 駅停車完了: 停止位置誤差の段階評価 ---
    if phase == "駅停車完了（速度0km/h）":
        err = abs(dist)          # 駅までの残り[m]（負なら過走）
        if err <= 1.0:
            return 1.0
        if err <= 3.0:
            return 0.8
        if err <= 5.0:
            return 0.6
        if err <= 10.0:
            return 0.4
        return 0.1

    # --- ちんたら運転（極低速での惰性走行）は明確に減点する ---
    # 現示が開いているのに数km/hで這って進むのは、時間・エネルギーの両面で最悪であり、
    # 環境側の停滞検出でエピソードが打ち切られる領域でもある。
    # 「信号待ちから復帰したら速やかに加速する」という挙動をここで作る。
    if speed < 5.0 and dist > 30.0 and signal > 5.0:
        return 0.9 if accel else 0.05

    # --- 次駅への減速フェーズ ---
    if phase == "次駅への減速フェーズ（駅手前400m以内）":
        if delta_stop <= 0.0:
            # 制動しないと止まれない
            return 0.9 if brake else 0.0
        if delta_stop <= 15.0:
            return 0.9 if brake else 0.5
        if delta_stop <= 60.0:
            # 制動開始が近い。惰行で詰めるのが最良、早すぎる制動は減点
            if coast:
                return 0.9
            return 0.35 if brake else 0.3
        # まだ余裕がある。力行は無駄、制動はもっと無駄
        if coast:
            return 0.85
        if accel:
            return 0.3 if speed < required else 0.1
        return 0.15

    # --- 駅間停車防止モード ---
    if mode == "anti_mid_stop":
        margin = speed - target_ns
        if speed <= 0.5 and dist > 20.0:
            return 0.5          # 既に機外停車している。ノッチ操作を責めても仕方ないので中立
        if margin > 5.0:
            return 0.05 if accel else (0.75 if (coast or brake) else 0.3)
        if margin > 0.0:
            return 0.2 if accel else 0.8
        if margin > -10.0:
            return 0.85 if coast else (0.6 if accel else 0.5)
        # 下げすぎ（無駄に遅い）
        return 0.5 if accel else 0.3

    # --- 出発直後の加速フェーズ ---
    if phase == "駅出発直後の加速フェーズ（20秒以内）":
        if accel:
            return 0.95 if speed < min(required, v_std) + 5.0 else 0.4
        return 0.3

    # --- 上り勾配で力行しても伸びない局面は、力行継続が正しい ---
    steep_up = (p_acc < 0.45)
    if steep_up and accel and speed < required - 1.0:
        return 0.9

    # --- ノコギリ運転（短時間でのノッチ往復）は減点 ---
    hunting = (holding < 7.0 and prev_dur < 7.0
               and notch != prev and not (steep_up and accel))
    if hunting:
        return 0.15

    # --- 惰行で駅に届かない（+11.4‰で失速する）局面での惰行は減点 ---
    if coast and not coast_ok and dist > 400.0:
        return 0.2

    # --- 巡航・遅延回復 ---
    dev = speed - v_std
    if mode == "delay_recovery":
        # 定時到達に制限速度びたづきが必要。力行の継続を高く評価する
        if accel:
            return 0.95
        if coast:
            return 0.4
        return 0.1
    # 通常運転: 標準運転曲線への追従を基準にする
    if speed > required + 3.0:
        return 0.2 if accel else 0.85          # 速すぎる。惰行・制動で落とすのが正しい
    if dev > 5.0:
        return 0.25 if accel else 0.8
    if dev < -5.0:
        return 0.85 if accel else 0.3          # 標準より遅い。力行で戻す
    # 標準運転曲線の近く。惰行の継続が最も省エネ
    if coast:
        return 0.9
    if accel:
        return 0.6 if speed < required else 0.35
    return 0.2                                   # 理由のない制動


# =============================================================================
# 入口
# =============================================================================
def evaluate(raw, mode=None):
    """生の状態辞書 → 0.0〜1.0 の評価値。"""
    mode = mode or raw.get("mode") or rf.decide_mode(raw)
    if mode == "hold_at_station":
        return evaluate_hold_at_station(raw)
    return evaluate_running(raw, mode)
