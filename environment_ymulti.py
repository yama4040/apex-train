# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田 → 白兎 → 蚕桑）の強化学習環境。

既存 `environment2.py`（単一駅間・apex2.py が使用）は**一切変更しない**。
本環境が新たに担うのは以下。

  1. **1エピソード＝2駅間**（羽前成田→白兎→蚕桑）。中間駅の白兎では `done` にせず停車フェーズへ遷移する。
  2. **駅停車フェーズ**。3ノッチを「発車（力行）／待機（制動）」の2択に読み替える
     （惰行は禁止。待機と同一結果になり Q学習の max 演算子が過大評価を累積するため）。
  3. **通算ダイヤ**（標準停車30秒を挟む）に基づく残り時間・遅延。
  4. **標準運転曲線 v_std との速度差**を観測に持つ。
  5. **列車長20mを織り込んだCBTC現示**（停止限界＝先行の先頭から70m手前）。

物理は `train.Train`（3ノッチ・山形28t）、路線は `track.Track` をそのまま使う（読み取り専用）。

報酬:
  既定は LLM を蒸留した報酬NN（`direct_reward_predictor_ymulti`）の出力のみ。
  NNがまだ無い段階では `reward_mode="rule"` の**暫定ルール報酬**で動かせる
  （`rule_reward_ymulti.py`。フェーズ3でNNに必ず置き換える一時的な例外）。
"""
import os
import re
import codecs
import random
import bisect

import numpy as np
import pandas as pd

import config_ymulti as CFG
import reward_features_ymulti as rf
import required_speed_ymulti as rsm
from brake_curve_ymulti import get_brake_curve, get_lookup
from standard_curve_ymulti import VStdTable
from train import Train
from actions import Actions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 先行列車の次駅停車の「先読み」想定（既存 environment2 と同じ思想。設計メモ §15）。
# 発車直後の猶予[秒]。駅を発車した瞬間は定義上 0km/h であり、駅の停止窓の外にいるため、
# 「駅の手前で停止した＝駅間停車」の判定にそのまま掛かってしまう。
# 既存 environment2 はこの猶予が無く、発車1歩目に惰行・制動を選ぶと即座に失敗終了していた。
# 複数駅間では中間駅の発車判断の直後にも同じ崖が現れるため、短い猶予を設ける。
DEPART_GRACE_S = 3.0

# 信号開通後の再起動猶予[秒]。CBTC現示に従って停止限界に止まったあと先行が進み、
# 現示が開いた**その瞬間**に「駅の手前で停止している＝駅間停車」と判定してしまうと、
# 正しい信号待ちが必ず失敗に終わる（実測: 先行が100mを越えて進んだ1ステップで失敗）。
# 現示が開いてからこの秒数の間は再起動の猶予とし、それを過ぎても動かなければ失敗とする。
SIGNAL_CLEAR_GRACE_S = 5.0

DELAY_DWELL_THRESHOLD_S = 30.0    # 先行の前駅出発遅延[秒]の閾値
ASSUMED_DELAYED_DWELL_S = 45.0    # そのときの想定次駅停車時間[秒]
OBSERVED_DWELL_LOOKAHEAD_S = 15.0 # 観測中（標準超過で停車中）の先読みマージン[秒]


def _read_stations():
    with codecs.open(os.path.join(BASE_DIR, "input/Station.csv"), "r", "utf-8", "ignore") as f:
        st = pd.read_csv(f)
    return [{"index": i, "name": str(st["name"][i]), "position": float(st["position"][i])}
            for i in CFG.STATION_INDICES]


class EnvironmentYMulti:
    """羽前成田→白兎→蚕桑を1エピソードで走る環境。"""

    # 観測ベクトルの次元（apex_ymulti.py がネットワーク定義に使う）
    NUM_STATES = 40
    NUM_ACTIONS = len(Actions)

    def __init__(self, time_step=1.0, load_reward_predictor=True, reward_mode="auto"):
        self.__time_step = time_step
        self.base_time_step = time_step
        self.stations = _read_stations()
        self.n_sections = len(CFG.RUNNING_TIMES)
        self.sched_arrival = CFG.scheduled_arrival_times()
        self.sched_departure = CFG.scheduled_departure_times()
        self.lookup = get_lookup()
        self.v_std_table = VStdTable()
        self._csv_cache = {}

        # --- 報酬源 ---
        self.reward_mode = reward_mode
        self.reward_predictor = None
        if load_reward_predictor and reward_mode in ("auto", "nn"):
            try:
                from direct_reward_predictor_ymulti import DirectRewardPredictorYMulti
                p = DirectRewardPredictorYMulti()
                if p.is_loaded:
                    self.reward_predictor = p
                    self.reward_mode = "nn"
            except Exception as e:
                print(f"[環境] 報酬NNを読み込めませんでした: {e}")
        if self.reward_predictor is None and reward_mode in ("auto", "rule"):
            self.reward_mode = "rule"
            if reward_mode == "auto":
                print("[環境] 報酬NNが無いため暫定ルール報酬で動作します"
                      "（rule_reward_ymulti.py。フェーズ3でNNに置き換えること）")
        self.last_llm_reward = 0.0
        self.current_mode = "normal"
        self.executed_action = None

    # ------------------------------------------------------------------ reset
    def reset(self, delay=0.0, weight_correction=1.0, f_train_csv=None, headway=None,
              forward_train_delay=None, start_section=0, start_position_offset=0.0):
        """エピソードを初期化する。

        Args:
            delay: 自列車の出発遅延[秒]
            f_train_csv: 先行列車CSVのパス（None で先行なし）
            headway: 出発間隔[秒]（先行が自列車より何秒先に起点駅を出たか）
            forward_train_delay: 先行の出発遅延[秒]（既定は 標準出発間隔 − headway）
            start_section: 開始する区間index（0=羽前成田発）
            start_position_offset: 駅間途中から開始する場合のオフセット[km]
        """
        self.section = int(start_section)
        self.departure_station = self.stations[self.section]
        self.arrival_station = self.stations[self.section + 1]
        start_position = self.departure_station["position"] + start_position_offset
        self.train = Train(self.arrival_station["position"], start_position, 0.0, weight_correction)
        self.wc = weight_correction

        # 時刻は「起点駅の標準発車を0とする通算時刻」
        self.t = self.sched_departure[self.section] + delay
        if start_position_offset != 0.0:
            self.t += self.v_std_table.t_std(self.section, start_position)
        self.ego_delay = float(delay)
        self.section_start_t = self.t
        self.episode_start_t = self.t

        self.is_dwelling = False
        self.dwell_elapsed = 0.0
        self.dwell_station_index = None
        self.departed_by_agent = False   # 直近の発車がエージェントの判断だったか（強制発車と区別）

        self._last_speed_limit = max(self.train.current_speed_limit, 1.0)
        self._stall_check_t = self.t
        self._stall_check_pos = self.train.position
        self._stopped_free_since = None   # 信号に抑えられずに停止し始めた時刻[s]

        self.pre_action = Actions.deceleration
        self.holding_time = 30.0
        self.prev_notch = None
        self.prev_notch_duration = 0.0
        self.last_llm_reward = 0.0
        self.current_mode = "normal"
        self.executed_action = None
        self.goal_reached = False
        self.failed = False
        self.fail_reason = ""
        # 分析用の記録
        self.dwell_log = []       # [{"station", "arrive_t", "dwell", "forced"}]

        # --- 先行列車 ---
        self.forward_csv_path = f_train_csv
        self.fowerd_train = None
        self.fowerd_train_controls = None
        self.headway = headway
        if forward_train_delay is not None:
            self.forward_train_delay = float(forward_train_delay)
        elif headway is not None:
            self.forward_train_delay = max(0.0, CFG.STD_DEPARTURE_INTERVAL - float(headway))
        else:
            self.forward_train_delay = 0.0

        if f_train_csv:
            ctr = self.read_csv(f_train_csv)
            self.fowerd_train_controls = [
                {"time": i, "position": float(ctr["position"][i]), "speed": float(ctr["speed"][i])}
                for i in range(len(ctr))]
            if headway is None:
                raise ValueError("先行列車CSVを指定する場合は headway（出発間隔[秒]）が必要です")
            idx = min(int(float(headway) + self.t), len(self.fowerd_train_controls) - 1)
            c = self.fowerd_train_controls[idx]
            self.fowerd_train = Train(self.stations[-1]["position"], c["position"], c["speed"], 1.0)
        self._refresh_forward_times()

        return self.normalized_state

    # -------------------------------------------------------- 先行列車の時刻
    def _refresh_forward_times(self):
        """先行が**現在の次駅**に到着/発車するCSV時刻を求め直す（区間が進むたびに呼ぶ）。"""
        self.forward_arrive_time = None
        self.forward_depart_time = None
        ctr = self.fowerd_train_controls
        if not ctr:
            return
        P = self.arrival_station["position"]
        stopped = False
        for c in ctr:
            if abs(c["position"] - P) < 0.05 and c["speed"] < 0.5:
                if self.forward_arrive_time is None:
                    self.forward_arrive_time = float(c["time"])
                stopped = True
            elif stopped and c["speed"] > 0.5 and c["position"] >= P - 0.05:
                self.forward_depart_time = float(c["time"])
                break

    @property
    def _forward_csv_time(self):
        return self.t + (self.headway if self.headway is not None else 0.0)

    def _sync_forward(self, dt):
        """先行列車をこのステップ完了後の時刻に同期する（CSVの記録軌道を線形補間で再生）。"""
        if self.fowerd_train is None:
            return
        ctr = self.fowerd_train_controls
        tau = self.t + dt + (self.headway if self.headway is not None else 0.0)
        i = int(tau)
        if i >= len(ctr) - 1:
            self.fowerd_train.set_states(ctr[-1]["speed"], ctr[-1]["position"])
        else:
            f = tau - i
            p = ctr[i]["position"] + (ctr[i + 1]["position"] - ctr[i]["position"]) * f
            v = ctr[i]["speed"] + (ctr[i + 1]["speed"] - ctr[i]["speed"]) * f
            self.fowerd_train.set_states(v, p)

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.is_dwelling:
            return self._step_dwell(action)
        return self._step_running(action)

    # ---- 走行中 ----
    def _step_running(self, action):
        dt = self.time_step
        self._sync_forward(dt)
        action_enum = Actions(action)
        self.train.step(action_enum, dt)
        self.executed_action = action_enum
        self.t += dt

        raw = self._build_raw_state(action_enum)
        llm_reward = self._evaluate(raw)
        reward = (llm_reward - 0.5) * (dt / self.base_time_step)

        # 【重要】失敗判定を到着判定より**先**に行う。
        # 先行列車が次駅に停車したままそこへ突っ込むと、位置は停止窓に入るため
        # 到着を先に判定すると「追突したのに到着成功」になってしまう（実測で発生）。
        done = self._check_failures()
        if not done and self._in_stop_window():
            # 到着イベント: 停止精度は「時間あたりのレート」ではなく1回のイベントの評価なので、
            # 終端ステップだけは time_step スケール（駅前0.1秒＝×0.1）を外して満額で与える。
            reward = (llm_reward - 0.5)
            if self.section >= self.n_sections - 1:
                done = True
                self.goal_reached = True
            else:
                self._enter_dwell()

        self._update_notch_history(action_enum, dt)
        return self.normalized_state, reward, done

    # ---- 駅停車中 ----
    def _step_dwell(self, action):
        dt = CFG.DWELL_TIME_STEP
        # 発車可否は「このステップを終えた時点の停車経過」で判定する（forbidden_action と一致させる）。
        # ステップ後で見ないと、最短でも DWELL_MIN + 1 秒の停車になり標準停車30秒ちょうどが作れない。
        departable = (self.dwell_elapsed + dt) >= CFG.DWELL_MIN
        forced = (self.dwell_elapsed + dt) >= CFG.DWELL_MAX
        action_enum = Actions(action)
        # 惰行は停車中禁止だが、万一渡された場合は待機として扱う（安全側）
        if action_enum == Actions.coasting:
            action_enum = Actions.deceleration
        self.executed_action = action_enum

        self._sync_forward(dt)
        self.t += dt
        self.dwell_elapsed += dt

        raw = self._build_raw_state(action_enum)
        llm_reward = self._evaluate(raw)
        reward = (llm_reward - 0.5) * (dt / self.base_time_step)

        depart = forced or (departable and action_enum == Actions.acceleration)
        if depart:
            self.departed_by_agent = not forced
            self.dwell_log[-1]["dwell"] = self.dwell_elapsed
            self.dwell_log[-1]["forced"] = forced
            self._leave_dwell()

        done = False
        if self.t >= CFG.total_scheduled_time() + CFG.TIME_OVER_MARGIN_S:
            done = True
            self.failed = True
            self.fail_reason = "タイムオーバー（停車中）"

        self._update_notch_history(action_enum, dt)
        return self.normalized_state, reward, done

    # ------------------------------------------------------- 停車の出入り
    def _enter_dwell(self):
        """中間駅に到着した。停車フェーズへ遷移し、**次の区間を見る状態**に切り替える。

        停車中の判断対象は「次の駅間で機外停車するか」なので、到着した時点で
        arrival_station を次の駅へ進める（dist_to_next_station 等が次区間を指す）。
        """
        self.dwell_log.append({"station": self.arrival_station["name"],
                               "arrive_t": self.t, "dwell": None, "forced": False,
                               "stop_error_m": (self.position - self.arrival_station["position"]) * 1000.0})
        self.is_dwelling = True
        self.dwell_elapsed = 0.0
        self.dwell_station_index = self.section + 1
        # 位置を駅にぴったり据えて次区間の基準にする（数cmの残差が距離計算に混ざるのを防ぐ）
        self.train.set_states(0.0, self.arrival_station["position"])
        self.section += 1
        self.departure_station = self.stations[self.section]
        self.arrival_station = self.stations[self.section + 1]
        self.train.TARGET_STATION = self.arrival_station["position"]
        self._refresh_forward_times()
        self._last_speed_limit = max(self.train.current_speed_limit, self._last_speed_limit)

    def _leave_dwell(self):
        """発車する。加速フェーズ判定・停滞検出の基準を打ち直す。"""
        self.is_dwelling = False
        self.dwell_elapsed = 0.0
        self.section_start_t = self.t
        self.episode_start_t = self.t
        self._stall_check_t = self.t
        self._stall_check_pos = self.train.position
        self._stopped_free_since = None
        self.pre_action = Actions.deceleration
        self.holding_time = 30.0
        self.prev_notch = None
        self.prev_notch_duration = 0.0

    def _update_notch_history(self, action_enum, dt):
        if self.pre_action == action_enum:
            self.holding_time += dt
        else:
            self.prev_notch = self.pre_action
            self.prev_notch_duration = self.holding_time
            self.holding_time = dt
        self.pre_action = action_enum

    # ------------------------------------------------------------ 終了条件
    def _in_stop_window(self):
        p = self.arrival_station["position"]
        return (self.position >= p - CFG.STOP_WINDOW_BEFORE_KM
                and self.position <= p + CFG.STOP_WINDOW_AFTER_KM
                and self.speed <= CFG.STOP_SPEED_KMH)

    def _check_failures(self):
        """走行中の失敗判定。`True` を返したらエピソード終了。"""
        # ① タイムオーバー（累積標準ダイヤ基準）
        if self.t >= CFG.total_scheduled_time() + CFG.TIME_OVER_MARGIN_S:
            self.failed = True
            self.fail_reason = "タイムオーバー"
            return True
        # ② 先行列車への異常接近（速度によらず距離で判定する）
        #    CBTC停止限界は先行の先頭から70m手前なので、そこに正しく停止するのは設計上の正解。
        #    40m まで詰めた時点で異常接近とする（30m の余裕を残す）。
        #    ※既存 environment2 は「停止したときだけ」判定していたため、
        #      駅に停車中の先行へ走行したまま突っ込むケースを検出できなかった。
        fp = self.fowerd_train_position
        if fp is not None and self.position >= fp - CFG.COLLISION_MARGIN_KM:
            self.failed = True
            self.fail_reason = "先行列車への異常接近"
            return True
        # ③ オーバーラン
        if self.speed > 0.0 and self.position > self.arrival_station["position"] + CFG.STOP_WINDOW_AFTER_KM:
            self.failed = True
            self.fail_reason = "オーバーラン"
            return True
        # ④ 手前での停止（駅間停車）。ただし**CBTC現示に従って止まったのは正しい信号待ち**。
        #    発車直後の DEPART_GRACE_S 秒は 0km/h が正常なので判定しない。
        stopped_short = (self.speed <= CFG.STOP_SPEED_KMH
                         and self.t - self.section_start_t > DEPART_GRACE_S
                         and self.position < self.arrival_station["position"] - CFG.STOP_WINDOW_BEFORE_KM)
        if stopped_short and not self.held_by_signal:
            if self._stopped_free_since is None:
                self._stopped_free_since = self.t          # 現示が開いた／元から信号待ちではない
            if self.t - self._stopped_free_since > SIGNAL_CLEAR_GRACE_S:
                self.failed = True
                self.fail_reason = "駅間停車（駅の手前で停止）"
                return True
        else:
            self._stopped_free_since = None
        # ⑤ 停滞（ちんたら運転）検出
        # CBTC現示に抑えられている間は前進できないのが正しいので、判定窓を伸ばさず
        # チェックポイントを打ち直し続ける。こうしないと信号待ちの時間が「前進していない時間」
        # として蓄積し、現示が開いた直後に停滞と誤判定される。
        if self.held_by_signal:
            self._stall_check_t = self.t
            self._stall_check_pos = self.position
            return False
        near = self.station_remaining_distance <= 0.4
        window = 10.0 if near else 30.0
        in_final = self.station_remaining_distance <= 0.02
        if self.t - self._stall_check_t >= window:
            progress = self.position - self._stall_check_pos
            min_progress = 0.005 if near else 0.025
            hit = (progress < min_progress) if not in_final else False
            self._stall_check_t = self.t
            self._stall_check_pos = self.position
            if hit:
                self.failed = True
                self.fail_reason = "停滞（ちんたら運転）"
                return True
        return False

    # ------------------------------------------------------------ 報酬の評価
    def _evaluate(self, raw):
        """生の状態辞書 → 0.0〜1.0 の評価値。モードもここで確定する。"""
        self.current_mode = rf.decide_mode(raw)
        raw["mode"] = self.current_mode
        value = 0.5
        if self.reward_mode == "nn" and self.reward_predictor is not None:
            try:
                value = self.reward_predictor.predict_reward(raw)
            except Exception as e:
                print(f"[推論エラー] {e}")
                value = 0.5
        else:
            import rule_reward_ymulti as rr
            value = rr.evaluate(raw)
        value = max(0.0, min(1.0, float(value)))
        self.last_llm_reward = value
        return value

    # ------------------------------------------------- 生の状態辞書（正準）
    def _build_raw_state(self, action_enum):
        """`reward_features_ymulti.RAW_COLS` と同じキーを持つ生の状態辞書を作る。

        1ステップにつき1度だけ作り、報酬推論・観測ベクトル・CSV出力で使い回す
        （既存 environment2 は required_speed 系を報酬側と観測側で二重に計算していた）。
        """
        pos = self.position
        v = self.speed
        arr = self.arrival_station["position"]
        dist_km = arr - pos
        limit = self.current_speed_limit
        signal = self.cbtc_signal_speed
        grade = self.lookup.grade(pos)

        rem_time = self.remaining_time
        clear_remaining = self.forward_clear_remaining_time

        if self.is_dwelling:
            # 停車中は自列車が止まっているので、走行系の指標は「発車直後の状況」を表す値にする
            req = rsm.calculate_required_speed(0.0, pos, arr, max(rem_time, 0.0), limit)
            target_ns = rsm.calculate_no_stop_target_speed(0.0, pos, arr, max(rem_time, 0.0),
                                                          clear_remaining, limit)
            req_stop = 0.0
            coast_ok, coast_v = 1.0, 0.0
            reach = rsm.time_to_stop_limit(pos, arr, speed_limit=limit, v_cruise=req)
        else:
            req = rsm.calculate_required_speed(v, pos, arr, rem_time, limit)
            target_ns = rsm.calculate_no_stop_target_speed(v, pos, arr, rem_time,
                                                          clear_remaining, limit)
            req_stop = rsm.station_stop_distance_m(v, arr)
            ok, coast_v = rsm.coast_probe(pos, v, arr)   # 到達可否と到達速度を1回の積分で得る
            coast_ok = 1.0 if ok else 0.0
            reach = 0.0

        v_std = self.v_std_table.v_std(self.section, pos)

        raw = {
            "run_id": "",
            "time": self.t,
            "section": self.section,
            "position": pos,
            "phase": self._phase_str(),
            "current_notch": self._notch_str(action_enum),
            "holding_time": self.holding_time,
            "prev_notch": self._prev_notch_str(),
            "prev_notch_duration": self.prev_notch_duration,
            "current_speed": v,
            "speed_limit": limit,
            "signal_speed": signal,
            "required_speed": req,
            "target_speed_no_stop": target_ns,
            "v_std": v_std,
            "v_std_deviation": v - v_std,
            "dist_to_next_station": dist_km * 1000.0,
            "time_to_next_station": max(0.0, rem_time),
            "req_stop_dist": req_stop,
            "delta_stop": dist_km * 1000.0 - req_stop,
            "current_gradient": grade,
            "coast_accel": rsm.coast_accel(pos, v),
            "power_accel": rsm.power_accel(pos, v),
            "next_limit_info": self._next_limit_info(),
            "next_gradient_info": self._next_gradient_info(),
            "coast_reachable": coast_ok,
            "coast_arrival_speed": coast_v,
            "delay": max(0.0, -rem_time),
            "total_delay": self.total_delay,
            "stations_remaining": self.n_sections - self.section,
            "total_remaining_distance": (self.stations[-1]["position"] - pos) * 1000.0,
            "total_remaining_time": CFG.total_scheduled_time() - self.t,
            "is_dwelling": 1.0 if self.is_dwelling else 0.0,
            "dwell_elapsed": self.dwell_elapsed if self.is_dwelling else 0.0,
            "dwell_min": CFG.DWELL_MIN,
            "dwell_max": CFG.DWELL_MAX,
            "dwell_over_std": max(0.0, self.dwell_elapsed - CFG.STD_DWELL) if self.is_dwelling else 0.0,
            "time_to_stop_limit": reach,
            "forward_info": self._forward_info(),
            "forward_train_delay": self.forward_train_delay,
            "standard_headway": self.headway if self.headway is not None else 0.0,
            "forward_clear_remaining_time": clear_remaining,
            "forward_observed_delay": self.forward_observed_delay,
            "forward_dwell_elapsed": self.forward_dwell_elapsed,
            "forward_departed_next": self.forward_departed_next,
            "backward_info": "後続列車なし",
            "mode": "",
            "reward": "",
            "reason": "",
        }
        self.last_raw_state = raw
        return raw

    # ------------------------------------------------------- 文字列ヘルパー
    def _phase_str(self):
        if self.is_dwelling:
            return "駅停車中（発車判断）"
        if self.station_remaining_distance * 1000.0 <= 10.0 and self.speed <= 0.5:
            return "駅停車完了（速度0km/h）"
        if self.t - self.section_start_t <= 20.0:
            return "駅出発直後の加速フェーズ（20秒以内）"
        if self.station_remaining_distance <= 0.4:
            return "次駅への減速フェーズ（駅手前400m以内）"
        secs = self.train.front_sections
        if len(secs) > 1 and secs[1]["speed_limit"] < secs[0]["speed_limit"] and secs[0]["distance"] <= 0.5:
            return "制限速度区間に接近中（500m以内に制限区間在り）"
        return "巡航フェーズ（駅間走行中）"

    def _notch_str(self, action_enum):
        if action_enum == Actions.acceleration:
            return "力行（加速）中"
        if action_enum == Actions.deceleration:
            return "ブレーキ（減速）中"
        return "惰行中"

    def _prev_notch_str(self):
        a = self.prev_notch
        if a is None:
            return "なし（または停止）"
        if a == Actions.acceleration:
            return "力行（加速）"
        if a == Actions.deceleration:
            return "ブレーキ（減速）"
        return "惰行"

    def _next_limit_info(self):
        secs = self.train.front_sections
        if len(secs) > 1 and secs[0]["distance"] <= 0.5:
            return f"{int(secs[0]['distance']*1000)}m先に制限速度{int(secs[1]['speed_limit'])}km/h"
        return "この先制限速度なし"

    def _next_gradient_info(self):
        grades = self.train.front_grades
        if len(grades) > 1 and grades[0]["distance"] <= 0.5:
            cur = grades[0]["grade"]
            nxt = grades[1]["grade"]
            d = int(grades[0]["distance"] * 1000)
            if nxt != 0:
                return f"{d}m先に{'上り' if nxt > 0 else '下り'}勾配{abs(nxt)}‰あり"
            if cur != 0:
                return f"{d}m先で{'上り' if cur > 0 else '下り'}勾配{abs(cur)}‰が終わり平坦になる"
        return "この先目立った勾配なし"

    def _forward_info(self):
        if self.fowerd_train is None:
            return "先行列車なし"
        d = self.fowerd_train_remaining_distance * 1000.0
        v = self.fowerd_train.speed
        if v < 0.5:
            return f"前方 {d:.1f}m 先に停車中"
        return f"前方 {d:.1f}m 先を {v:.1f}km/h で走行中"

    # ------------------------------------------------------------ プロパティ
    @property
    def speed(self):
        return self.train.speed

    @property
    def position(self):
        return self.train.position

    @property
    def time_step(self):
        if self.is_dwelling:
            return CFG.DWELL_TIME_STEP
        if self.position < self.arrival_station["position"] - 0.1:
            return self.__time_step
        return self.__time_step * 0.1

    @property
    def station_remaining_distance(self):
        return self.arrival_station["position"] - self.position

    @property
    def remaining_time(self):
        """次駅の標準到着時刻までの残り時間[秒]（通算ダイヤ基準）。負なら遅延。"""
        return self.sched_arrival[self.section + 1] - self.t

    @property
    def total_delay(self):
        """現在位置・停車経過に対する標準ダイヤからの遅延[秒]（0以上）。"""
        if self.is_dwelling:
            std = self.sched_arrival[self.section] + min(self.dwell_elapsed, CFG.STD_DWELL)
        else:
            std = self.sched_departure[self.section] + \
                self.v_std_table.t_std(self.section, self.position)
        return max(0.0, self.t - std)

    @property
    def current_speed_limit(self):
        lim = self.train.current_speed_limit
        if lim <= 0.0:
            # 駅を数cm過走すると front_sections が空になり0を返すためフォールバックする
            return self._last_speed_limit
        self._last_speed_limit = lim
        return lim

    @property
    def fowerd_train_position(self):
        return None if self.fowerd_train is None else self.fowerd_train.position

    @property
    def fowerd_train_remaining_distance(self):
        if self.fowerd_train_position is None:
            return self.station_remaining_distance
        return self.fowerd_train_position - self.position

    @property
    def held_by_signal(self):
        """CBTC現示に抑えられて停止している（＝正しい信号待ち）状態か。

        停止限界は先行の先頭から `CBTC_HEAD_MARGIN_KM`（70m）手前なので、
        そこに停止した列車は先行の先頭から70m後方にいる。判定には少し余裕を見て120mとする。
        現示そのものが0付近の場合も信号待ちとみなす。

        ※既存 environment2 は「先行の100m手前」を境にしていたが、列車長20mを織り込むと
          停止限界が70mになり、正しく停止した列車が先行の前進とともに100mの外へ出て
          「駅間停車」と誤判定されていた（実測: 先行が30m進んだ瞬間に失敗扱い）。
        """
        fp = self.fowerd_train_position
        if fp is None:
            return False
        if self.fowerd_train_remaining_distance <= CFG.CBTC_HEAD_MARGIN_KM + 0.05:
            return True
        return self.cbtc_signal_speed <= 1.0

    @property
    def cbtc_signal_speed(self):
        """CBTC指示速度[km/h]（停止限界＝先行の先頭から70m手前。列車長20mを織り込む）"""
        return rsm.cbtc_signal_speed(self.position, self.fowerd_train_position,
                                     self.current_speed_limit)

    @property
    def forward_clear_remaining_time(self):
        """先行が自列車の次駅を発車するまでの残り秒数[s]（因果的・観測情報のみ・0以上）。"""
        if self.fowerd_train is None or self.forward_arrive_time is None:
            return 0.0
        tau = self._forward_csv_time
        if self.forward_depart_time is not None and tau >= self.forward_depart_time:
            return 0.0
        assumed = (ASSUMED_DELAYED_DWELL_S
                   if self.forward_train_delay >= DELAY_DWELL_THRESHOLD_S else CFG.STD_DWELL)
        obs = self.forward_observed_delay
        if obs > 0.0:
            assumed = max(assumed, CFG.STD_DWELL + obs + OBSERVED_DWELL_LOOKAHEAD_S)
        return max(0.0, self.forward_arrive_time + assumed - tau)

    @property
    def forward_observed_delay(self):
        """先行が自列車の次駅で標準停車(30秒)を超えて停車している観測遅延[秒]（0以上）。"""
        if self.fowerd_train is None or self.forward_arrive_time is None:
            return 0.0
        tau = self._forward_csv_time
        if self.forward_depart_time is not None and tau >= self.forward_depart_time:
            return 0.0
        if tau < self.forward_arrive_time:
            return 0.0
        return max(0.0, tau - (self.forward_arrive_time + CFG.STD_DWELL))

    @property
    def forward_dwell_elapsed(self):
        """先行が自列車の次駅に停車してからの経過時間[秒]（0以上）。"""
        if self.fowerd_train is None or self.forward_arrive_time is None:
            return 0.0
        tau = self._forward_csv_time
        if self.forward_depart_time is not None and tau >= self.forward_depart_time:
            return 0.0
        return max(0.0, tau - self.forward_arrive_time)

    @property
    def forward_departed_next(self):
        if self.fowerd_train is None or self.forward_depart_time is None:
            return ""
        return "発車済み" if self._forward_csv_time >= self.forward_depart_time else "未発車"

    # ------------------------------------------------------------ 行動の制約
    @property
    def forbidden_action(self):
        """[惰行, 力行, 制動] の禁止フラグ。**同時に3つ禁止してはならない**
        （masked_qs が全て −inf になり argmax が壊れる）。"""
        coasting = acceleration = deceleration = False
        if self.is_dwelling:
            # 停車中は「発車（力行）／待機（制動）」の2択。惰行は常に禁止する。
            coasting = True
            if self.dwell_elapsed + CFG.DWELL_TIME_STEP < CFG.DWELL_MIN:
                acceleration = True        # 最低停車時間まで発車できない（待機のみ）
            elif self.dwell_elapsed + CFG.DWELL_TIME_STEP >= CFG.DWELL_MAX:
                deceleration = True        # 最大停車時間に達したら強制発車（発車のみ）
            return np.array([coasting, acceleration, deceleration])

        # 走行中（既存 environment2 と同じ扱い）
        if self.speed > self.current_speed_limit:
            acceleration = True            # 制限超過時は力行のみ禁止（惰行は許可）
        fp = self.fowerd_train_position
        if fp is not None and self.position > fp:
            acceleration = True            # 先行を追い越した＝衝突域。安全側で惰行も禁止
            coasting = True
        return np.array([coasting, acceleration, deceleration])

    # ------------------------------------------------------------ 観測ベクトル
    @property
    def normalized_state(self):
        """DQN観測ベクトル（40次元）。各値はおおむね 0〜1 または −1〜1 に正規化する。"""
        raw = getattr(self, "last_raw_state", None)
        if raw is None or abs(raw["time"] - self.t) > 1e-9:
            # reset 直後や発車直後は最新の生状態がまだ無いので作る
            raw = self._build_raw_state(self.pre_action)

        v = self.speed
        arr = self.arrival_station["position"]
        dist = self.station_remaining_distance
        pre_c = 1.0 if self.pre_action == Actions.coasting else 0.0
        pre_a = 1.0 if self.pre_action == Actions.acceleration else 0.0
        pre_d = 1.0 if self.pre_action == Actions.deceleration else 0.0

        req_stop_km = raw["req_stop_dist"] / 1000.0
        margin_stop = dist - req_stop_km
        f_speed = self.fowerd_train.speed if self.fowerd_train is not None else 80.0

        phase = raw["phase"]
        grade = raw["current_gradient"]
        ng = self.train.front_grades
        next_grade_dist = ng[0]["distance"] if len(ng) > 1 and ng[0]["distance"] <= 0.5 else None
        next_grade_val = ng[1]["grade"] if len(ng) > 1 and ng[0]["distance"] <= 0.5 else None
        nl = self.train.front_sections
        next_limit_dist = nl[0]["distance"] if len(nl) > 1 and nl[0]["distance"] <= 0.5 else None
        next_limit_val = nl[1]["speed_limit"] if len(nl) > 1 and nl[0]["distance"] <= 0.5 else None

        mode_oh = [0.0] * CFG.MODE_DIM
        mode_oh[CFG.MODE_INDEX.get(self.current_mode, 0)] = 1.0

        return np.array([
            # --- 自列車の走行状態（1-8） ---
            v / 80.0,
            (max(dist, -0.5) + 0.5) / 3.0,
            (max(min(dist, 0.2), -0.05) + 0.05) * 4.0,
            np.clip(self.remaining_time / 200.0, -1.0, 1.5),
            min(self.holding_time, 30.0) / 30.0,
            pre_c, pre_a, pre_d,
            # --- 速度の基準（9-14） ---
            self.current_speed_limit / 80.0,
            raw["signal_speed"] / 80.0,
            raw["required_speed"] / 80.0,
            raw["target_speed_no_stop"] / 80.0,
            raw["v_std"] / 80.0,
            np.clip(raw["v_std_deviation"] / 20.0, -1.0, 1.0),
            # --- 停止余裕（15-16） ---
            min(req_stop_km, 1.0),
            np.clip(margin_stop, -0.5, 1.5) / 1.5,
            # --- フェーズone-hot（17-22） ---
            1.0 if phase == "駅停車中（発車判断）" else 0.0,
            1.0 if phase == "駅出発直後の加速フェーズ（20秒以内）" else 0.0,
            1.0 if phase == "巡航フェーズ（駅間走行中）" else 0.0,
            1.0 if phase == "制限速度区間に接近中（500m以内に制限区間在り）" else 0.0,
            1.0 if phase == "次駅への減速フェーズ（駅手前400m以内）" else 0.0,
            1.0 if phase == "駅停車完了（速度0km/h）" else 0.0,
            # --- 勾配・惰行到達可能性（23-28） ---
            np.clip(grade / 20.0, -1.0, 1.0),
            (next_grade_dist / 0.5) if next_grade_dist is not None else 1.0,
            np.clip((next_grade_val if next_grade_val is not None else grade) / 20.0, -1.0, 1.0),
            (next_limit_dist / 0.5) if next_limit_dist is not None else 1.0,
            (next_limit_val if next_limit_val is not None else self.current_speed_limit) / 80.0,
            raw["coast_reachable"],
            # --- 先行列車（29-33） ---
            (max(self.fowerd_train_remaining_distance, -0.5) + 0.5) / 3.0,
            f_speed / 80.0,
            min(raw["forward_clear_remaining_time"], 300.0) / 300.0,
            min(raw["forward_dwell_elapsed"], 300.0) / 300.0,
            1.0 if raw["forward_departed_next"] == "発車済み" else 0.0,
            # --- 駅停車・通算ダイヤ（34-38） ---
            1.0 if self.is_dwelling else 0.0,
            min(self.dwell_elapsed, CFG.DWELL_MAX) / CFG.DWELL_MAX,
            # 「先行クリア残時間 − 停止限界到達時間」。正なら今発車すると機外停車する見込み。
            np.clip(raw["forward_clear_remaining_time"] - raw["time_to_stop_limit"],
                    -200.0, 200.0) / 200.0,
            (self.n_sections - self.section) / float(self.n_sections),
            np.clip(raw["total_remaining_time"] / 400.0, -1.0, 1.5),
            # --- 直前ノッチの継続時間（39） ---
            min(self.prev_notch_duration, 30.0) / 30.0,
            # --- 運転モード（40：hold_at_station を1次元で表す） ---
            mode_oh[CFG.MODE_INDEX["hold_at_station"]],
        ], dtype=np.float32)

    # -------------------------------------------------------------- ユーティリティ
    def read_csv(self, path):
        if path in self._csv_cache:
            return self._csv_cache[path]
        with codecs.open(path, "r", "utf-8", "ignore") as f:
            csv = pd.read_csv(f)
        self._csv_cache[path] = csv
        return csv

    @property
    def latest_rewards_info(self):
        return [self.last_llm_reward]
