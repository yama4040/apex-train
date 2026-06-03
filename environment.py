from train import Train
from actions import Actions
import numpy as np
import random
import pandas as pd
import codecs
import math

STEP_INTERVAL=30  # LLMによる重み更新のインターバル（ステップ数）

# 新しく作成した報酬予測器をインポート
try:
    from reward_predictor import RewardWeightPredictor
except ImportError:
    RewardWeightPredictor = None

class Environment:
    def __init__(self, time_step=1.0):
        self.__time_step = time_step
        self.MAX_SECTIONS = 6
        self.MAX_CURVES=4
        self.MAX_GRADES=8
        stations_csv = self.read_csv("./input/Station.csv")
        self.stations = []
        for i in range(len(stations_csv)):
            self.stations.append({"position": stations_csv["position"][i], "running_time": stations_csv["rt"][i]})
            
        # LLM報酬予測モデルの初期化（プロセス毎に1度だけ）
        self.weight_predictor = RewardWeightPredictor() if RewardWeightPredictor else None
        
        # ▼【追加】Tri-Drive報酬の記録用変数
        self.last_w_surv = 0.0
        self.last_R_surv = 0.0
        self.last_w_conf = 0.0
        self.last_R_conf = 0.0
        self.last_w_comp = 0.0
        self.last_R_comp = 0.0
        
        # ▼ 追加: 保持しておく現在の重み（初期値）
        self.current_w_surv = 0.33
        self.current_w_conf = 0.33
        self.current_w_comp = 0.33

    def reset(self, departure_index=None, delay=0.0, weight_correction=1.0, fowerd_train_position_offset=None, start_position_offset=0.0, fowerd_train_controls=None):
        if departure_index is None:
            departure_index = random.randrange(1)
        self.t = 0.0
        self.departure_station = self.stations[departure_index]
        self.arrival_station = self.stations[departure_index + 1]
        start_position=self.departure_station["position"]+start_position_offset
        self.train = Train(self.arrival_station["position"], start_position, 0.0,weight_correction)
        self.fowerd_train=None
        
        if (fowerd_train_controls): 
            ftc=self.read_csv(fowerd_train_controls)
            self.fowerd_train_controls=[]
            for i in range(len(ftc)):
                action=Actions.deceleration
                if (ftc["action"][i]=="Actions.coasting"): action=Actions.coasting
                elif (ftc["action"][i]=="Actions.acceleration"): action=Actions.acceleration
                elif (ftc["action"][i]=="Actions.deceleration"): action=Actions.deceleration
                self.fowerd_train_controls.append({"time":i,"position":ftc["position"][i],"speed":ftc["speed"][i], "action": action})
            self.fowerd_train=Train(self.arrival_station["position"], self.fowerd_train_controls[0]["position"],self.fowerd_train_controls[0]["speed"],1.0)
        self.fowerd_train_position_offset=fowerd_train_position_offset
        
        if (self.fowerd_train_position is not None or start_position_offset !=0.0):
            sr_csv = self.read_csv(f"./input/sr_{departure_index}.csv")
            self.standerd_running = []
            for i in range(len(sr_csv)):
                self.standerd_running.append({"position": sr_csv["position"][i], "time": sr_csv["time"][i]})
        if (start_position_offset!=0.0):
            for i in range(len(self.standerd_running)):
                self.t=self.standerd_running[i]["time"]+delay
                break
                
        self.pre_action = Actions.deceleration  
        self.holding_time = 30  
        self.current_accel = 0.0 # ジャーク計算用（1ステップ前の実際の加速度）
        
        return self.normalized_state

    def step(self, action):
        if (self.fowerd_train is not None and self.fowerd_train.speed<=0.0 and self.fowerd_train.position>self.arrival_station["position"]): self.fowerd_train=None
        if (self.fowerd_train is not None and self.t>self.fowerd_train_controls[-1]["time"]): self.fowerd_train.step(Actions.deceleration,1)
        elif (self.fowerd_train is not None and round(self.t%1,1)==0): self.fowerd_train.step(self.fowerd_train_controls[int(self.t)]["action"],1)
        
        action_enum = Actions(action)
        time_step = self.time_step
        
        # --- 状態の事前保存（ジャーク・消費電力計算用） ---
        pre_accel = self.current_accel
        pre_speed = self.speed
        
        # --- 列車の状態を更新 ---
        self.train.step(action_enum, time_step)
        self.t += time_step
        
        # 実際の物理的な加速度を計算して保存 (Δv / Δt)
        self.current_accel = (self.speed - pre_speed) / time_step
        
        # --- LLMによる重みの予測（30ステップ=約30秒に1回だけ更新） ---
        # self.t がタイムステップの整数倍になったタイミングで、かつ30で割り切れる時
        step_count = int(self.t / self.time_step)
        
        # エピソード開始直後(step=0) または 30ステップごとに更新
        if step_count % STEP_INTERVAL == 0 and self.weight_predictor:
            state_info = {
                'speed_limit': self.current_speed_limit,
                'current_speed': self.speed,
                'dist_to_next_station': self.station_remaining_distance,
                'delay': max(0.0, self.t - self.fixed_running_time),
                'current_gradient': self.train.front_grades[0]["grade"] if len(self.train.front_grades) > 0 else 0.0,
                'phase': self._get_current_phase_str(),
                'notch': self._get_current_notch_str(action_enum),
                'next_limit_info': self._get_next_limit_info(),
                'next_gradient_info': self._get_next_gradient_info()
            }
            try:
                # 推論してクラス変数に保存（次の更新までこれを使い回す）
                self.current_w_surv, self.current_w_conf, self.current_w_comp = self.weight_predictor.predict_weights(state_info)
            except Exception:
                pass # エラー時は前回の重みをそのまま維持

        # 今回のステップの計算には、保持されている最新の重みを使用する
        w_surv = self.current_w_surv
        w_conf = self.current_w_conf
        w_comp = self.current_w_comp
        
        # --- 終了判定 (done) の処理 ---
        done = False
        if self.t >= self.departure_station["running_time"] + 60.0:
            done = True
        if self.speed <= 0.0 and self.position >= self.arrival_station["position"] - 0.01:
            done = True
        if self.fowerd_train_position is not None and self.speed <= 0 and self.position >= self.fowerd_train_position - 0.1:
            done = True

        # ==========================================
        # Tri-Drive 報酬計算ブロック
        # ==========================================
        # TODO: 以下の係数は学習の進捗に合わせてチューニングしてください
        alpha1, alpha2, alpha3 = 1.0, 1.0, 1.0
        beta1, beta2 = 1.0, 1.0
        eta, gamma, rho, sigma = 1.0, 1.0, 1.0, 1.0

        # 【1. Survival (安全/恒常性)】
        v_excess = max(0.0, self.speed - self.current_speed_limit)
        delay_error = max(0.0, self.t - self.fixed_running_time)
        stop_error = abs(self.station_remaining_distance) if done else 0.0
        # 目的地到着時のボーナス（エコドライブによる「サボり」を防止）
        goal_reward = 20.0 if (done and stop_error < 0.01 and self.position >= self.arrival_station["position"] - 0.01) else 0.0 
        R_surv = - (alpha1 * v_excess) - (alpha2 * delay_error) - (alpha3 * stop_error) + goal_reward

        # 【2. Confidence (確信度/乗り心地)】
        jerk = abs(self.current_accel - pre_accel)
        reverse_penalty = 1.0 if (self.pre_action == Actions.acceleration and action_enum == Actions.deceleration) or \
                                 (self.pre_action == Actions.deceleration and action_enum == Actions.acceleration) else 0.0
        R_conf = - (beta1 * jerk) - (beta2 * reverse_penalty)

        # 【3. Competence (能力/エネルギー効率)】
        # 力行(加速)時のみの消費電力を近似（motor_acceleration * speed）
        power_consume = (self.train.motor_acceleration * self.speed * time_step) if action_enum == Actions.acceleration else 0.0
        # ブレーキ（運動エネルギーの無駄捨て）に対するペナルティ
        brake_penalty = 1.0 if action_enum == Actions.deceleration else 0.0
        
        forward_penalty = 0.0
        if self.fowerd_train_position is not None:
            distance_to_forward = max(0.1, self.fowerd_train_remaining_distance)
            forward_penalty = math.exp(-distance_to_forward / sigma)
            
        R_comp = - (eta * power_consume) - (gamma * brake_penalty) - (rho * forward_penalty)

        # --- 最終的な報酬の合算とスケーリング ---
        reward = (w_surv * R_surv) + (w_conf * R_conf) + (w_comp * R_comp)
        reward = reward / 30.0  # DQNのQ値発散防止のためのクリッピング
        
        # ▼【追加】分析用に各値をオブジェクトに保持させておく
        self.last_w_surv = w_surv
        self.last_R_surv = R_surv
        self.last_w_conf = w_conf
        self.last_R_conf = R_conf
        self.last_w_comp = w_comp
        self.last_R_comp = R_comp

        # アクション保持時間の更新
        if self.pre_action == action_enum:
            self.holding_time += time_step
        else:
            self.holding_time = time_step
        self.pre_action = action_enum

        return self.normalized_state, reward, done

    # --- 以下、LLM推論用のヘルパーメソッド群 ---
    def _get_current_phase_str(self):
        if self.t <= 20.0:
            return "駅出発直後の加速フェーズ（駅発車20秒以内）"
        elif self.station_remaining_distance <= 0.4:
            return "次駅への減速フェーズ（駅手前400m以内）"
        elif self.current_speed_limit < 1000 and self.speed > self.current_speed_limit and self.train.section_remaining_distance <= 0.5:
             return "制限速度接近フェーズ（500m以内に制限速度があり、かつ現在速度が制限速度を超えている）"
        else:
            return "巡航フェーズ（駅間走行中）"

    def _get_current_notch_str(self, action_enum):
        if action_enum == Actions.acceleration: return "力行（加速）中"
        elif action_enum == Actions.deceleration: return "ブレーキ（減速）中"
        else: return "惰行中"

    def _get_next_limit_info(self):
        sections = self.train.front_sections
        if len(sections) > 1:
            dist_km = sections[0]["distance"]
            next_limit = sections[1]["speed_limit"]
            if dist_km <= 0.5: # 500m以内
                return f"{int(dist_km*1000)}m先に制限速度{int(next_limit)}km/hあり"
        return "この先制限速度なし"

    def _get_next_gradient_info(self):
        grades = self.train.front_grades
        if len(grades) > 1:
            dist_km = grades[0]["distance"]
            next_grade = grades[1]["grade"]
            if dist_km <= 0.5 and next_grade != 0:
                direction = "上り" if next_grade > 0 else "下り"
                return f"{int(dist_km*1000)}m先に{direction}勾配{abs(next_grade)}‰あり"
        return "この先目立った勾配なし"

    # 以下、既存のプロパティ群 (normalized_state, raw_state, forbidden_action 等) は変更なし
    @property
    def normalized_state(self):
        pre_action_c = 1.0 if self.pre_action == Actions.coasting else 0
        pre_action_a = 1.0 if self.pre_action == Actions.acceleration else 0
        pre_action_d = 1.0 if self.pre_action == Actions.deceleration else 0
        return [
            self.speed / 80.0,
            (max(self.station_remaining_distance,-0.5)+0.5)/2,
            (max(min(self.station_remaining_distance,0.2),-0.05)+0.05)*4,
            self.remaining_time / 360.0,
            min(self.holding_time,30) / 30.0,
            pre_action_c,
            pre_action_a,
            pre_action_d,
            (max(self.fowerd_train_remaining_distance,-0.5)+0.5)/2,
        ]

    @property
    def raw_state(self):
        return [
            self.speed,
            self.station_remaining_distance,
            self.remaining_time,
            self.holding_time,
            self.pre_action,
            self.station_remaining_distance,
            self.fowerd_train_remaining_distance
        ]

    @property
    def forbidden_action(self):
        acceleration = False
        coasting = False
        deceleration = False
        if self.speed > self.current_speed_limit:
            acceleration = True
            coasting = True
        if self.fowerd_train_position is not None and self.position > self.fowerd_train_position:
            acceleration = True
            coasting = True
        return np.array([coasting, acceleration, deceleration])

    @property
    def station_remaining_distance(self):
        return self.arrival_station["position"] - self.position

    @property
    def fowerd_train_position(self):
        if (self.fowerd_train_position_offset is None and self.fowerd_train is None): return None
        if (self.fowerd_train is None): return self.departure_station["position"]+self.fowerd_train_position_offset
        else : return self.fowerd_train.position

    @property
    def fowerd_train_remaining_distance(self):
        if (self.fowerd_train_position is None): return self.station_remaining_distance
        return self.fowerd_train_position-self.position
    
    @property
    def remaining_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"] - self.t
        return self.fixed_running_time - self.t
    
    @property
    def fixed_running_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"]
        for i in range(len(self.standerd_running)):
            if (self.fowerd_train_position<self.standerd_running[i]["position"]):
                return self.standerd_running[i]["time"]
        return self.departure_station["running_time"]

    @property
    def current_speed_limit(self):
        return self.train.current_speed_limit

    @property
    def speed(self):
        return self.train.speed

    @property
    def position(self):
        return self.train.position

    @property
    def time_step(self):
        if self.position<self.arrival_station["position"]-0.1:
            return self.__time_step
        else:
            return self.__time_step*0.1

    def read_csv(self, path):
        with codecs.open(path, "r", "utf-8", "ignore") as f:
            return pd.read_csv(f)
    
    # ▼【追加】外部から各報酬と重みを取得するためのプロパティ
    @property
    def latest_rewards_info(self):
        return [
            self.last_w_surv, self.last_R_surv,
            self.last_w_conf, self.last_R_conf,
            self.last_w_comp, self.last_R_comp
        ]