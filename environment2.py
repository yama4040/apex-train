from train import Train
from actions import Actions
import numpy as np
import random
import pandas as pd
import codecs
import math

# 単一評価値予測器をインポート
try:
    from direct_reward_predictor import DirectRewardPredictor
except ImportError:
    DirectRewardPredictor = None

class Environment:
    def __init__(self, time_step=1.0, load_reward_predictor=True):
        self.__time_step = time_step
        self._csv_cache = {}
        self.MAX_SECTIONS = 6
        self.MAX_CURVES = 4
        self.MAX_GRADES = 8
        stations_csv = self.read_csv("./input/Station.csv")
        self.stations = []
        for i in range(len(stations_csv)):
            self.stations.append({"position": stations_csv["position"][i], "running_time": stations_csv["rt"][i]})
            
        # LLM直接報酬予測モデルの初期化（プロセス毎に1度だけ）
        self.reward_predictor = DirectRewardPredictor() if DirectRewardPredictor else None
        
        # ▼フラグがTrueの時だけモデルを初期化する
        if load_reward_predictor and DirectRewardPredictor:
            self.reward_predictor = DirectRewardPredictor()
        else:
            self.reward_predictor = None
        
        # 分析・CSV記録用の変数
        self.last_llm_reward = 0.0

    def reset(self, departure_index=None, delay=0.0, weight_correction=1.0, fowerd_train_time_offset=None, start_position_offset=0.0, fowerd_train_controls=None):
        if departure_index is None:
            departure_index = random.randrange(1)
        self.t = 0.0
        self.departure_station = self.stations[departure_index]
        self.arrival_station = self.stations[departure_index + 1]
        start_position = self.departure_station["position"] + start_position_offset
        self.train = Train(self.arrival_station["position"], start_position, 0.0, weight_correction)
        self.fowerd_train = None
        self.fowerd_train_time_offset = fowerd_train_time_offset  # 追加
        
        if (fowerd_train_controls): 
            ftc = self.read_csv(fowerd_train_controls)
            self.fowerd_train_controls = []
            for i in range(len(ftc)):
                action = Actions.deceleration
                if (ftc["action"][i] == "Actions.coasting"): action = Actions.coasting
                elif (ftc["action"][i] == "Actions.acceleration"): action = Actions.acceleration
                elif (ftc["action"][i] == "Actions.deceleration"): action = Actions.deceleration
                self.fowerd_train_controls.append({"time": i, "position": ftc["position"][i], "speed": ftc["speed"][i], "action": action})
            
            # ▼【変更】時間オフセット(headway)に基づいて初期位置を割り出す
            if len(self.fowerd_train_controls) > 0 and self.fowerd_train_time_offset is not None:
                # CSVインデックス（経過秒数）の計算。最大値を超えないように制限
                start_idx = min(int(self.fowerd_train_time_offset), len(self.fowerd_train_controls) - 1)
                
                self.fowerd_train = Train(
                    self.arrival_station["position"], 
                    self.fowerd_train_controls[start_idx]["position"], 
                    self.fowerd_train_controls[start_idx]["speed"], 
                    1.0
                )
            else:
                self.fowerd_train = None
                print(f"\n[警告] {fowerd_train_controls} 先行列車なしとして扱います。")
                
        if (self.fowerd_train_position is not None or start_position_offset != 0.0):
            sr_csv = self.read_csv(f"./input/sr_{departure_index}.csv")
            self.standerd_running = []
            for i in range(len(sr_csv)):
                self.standerd_running.append({"position": sr_csv["position"][i], "time": sr_csv["time"][i]})
        if (start_position_offset != 0.0):
            for i in range(len(self.standerd_running)):
                self.t = self.standerd_running[i]["time"] + delay
                break
                
        self.pre_action = Actions.deceleration  
        self.holding_time = 30  
        # ▼▼▼ 新規追加: 直前のノッチ操作と保持時間の初期化 ▼▼▼
        self.prev_notch = None # 初期状態
        self.prev_notch_duration = 0.0
        # ▲▲▲ 新規追加 ▲▲▲
        self.last_llm_reward = 0.0
        
        return self.normalized_state

    def step(self, action):
        # 既存コード：先行列車が終点に着いたら消去
        #if (self.fowerd_train is not None and self.fowerd_train.speed <= 0.0 and self.fowerd_train.position > self.arrival_station["position"]): 
            #self.fowerd_train = None
            
        # ▼【変更】先行列車のアクション適用
        if self.fowerd_train is not None:
            # 現在の自列車時刻 ＋ 出発間隔 ＝ 先行列車の稼働時間
            f_t = int(self.t + self.fowerd_train_time_offset)
            
            if f_t >= len(self.fowerd_train_controls):
                # CSVのデータを超えたらとりあえずブレーキ
                self.fowerd_train.step(Actions.deceleration, 1)
            elif round(self.t % 1, 1) == 0:
                self.fowerd_train.step(self.fowerd_train_controls[f_t]["action"], 1)
        
        action_enum = Actions(action)
        time_step = self.time_step
        
        # 列車の状態を更新
        self.train.step(action_enum, time_step)
        self.t += time_step
        
        # ▼▼▼ 追加: LLMへ渡すための「事前計算」 ▼▼▼
        current_holding_time = self.holding_time + time_step if self.pre_action == action_enum else time_step
        
        # 実際にLLMに渡す段階での「直前のノッチ情報」を整理
        if self.pre_action != action_enum:
            # まさに今ノッチが切り替わった瞬間なら、これまでの行動が「直前のノッチ」になる
            current_prev_notch = self.pre_action
            current_prev_duration = self.holding_time
        else:
            # 切り替わっていなければ、保持している「直前のノッチ」をそのまま使う
            current_prev_notch = self.prev_notch
            current_prev_duration = self.prev_notch_duration
            
        def get_prev_notch_str(act):
            if act is None: return "なし（または停止）"
            if act == Actions.acceleration: return "力行（加速）"
            elif act == Actions.deceleration: return "ブレーキ（減速）"
            elif act == Actions.coasting: return "惰行"
            return "なし（または停止）"
        # ▲▲▲ 追加 ▲▲▲
        
        # --- 1. LLMによる評価値 (スコア) の推論 ---
        llm_reward = 0.0
        if self.reward_predictor:
            # ▼【追加】先行列車の情報をプロンプト形式のテキストに変換
            if self.fowerd_train is not None:
                # 距離は km なので m に変換してフォーマット
                f_dist_m = self.fowerd_train_remaining_distance * 1000.0
                forward_info_str = f"前方 {f_dist_m:.1f}m 先を {self.fowerd_train.speed:.1f}km/h で走行中"
            else:
                forward_info_str = "先行列車なし"

            # ▼▼▼ 追加: 要求ブレーキ距離の計算 ▼▼▼
            v_ms = max(0.0, self.speed / 3.6)
            decel_ms2 = 2.5 / 3.6
            fallback_req_dist = (v_ms ** 2) / (2 * decel_ms2) + (v_ms * self.time_step)
            req_dist_val = fallback_req_dist
            # ▲▲▲ 追加ここまで ▲▲▲

            # ▼▼▼ 【重要修正】'signal_speed' を追加して報酬予測側へ渡す ▼▼▼
            state_info = {
                'speed_limit': self.current_speed_limit,
                'signal_speed': self.cbtc_signal_speed,  # <--- 【追加】CBTC指示速度
                'current_speed': self.speed,
                'dist_to_next_station': self.station_remaining_distance * 1000.0, 
                'time_to_next_station': self.remaining_time,  
                'req_stop_dist': req_dist_val,
                'holding_time': current_holding_time, 
                'prev_notch': get_prev_notch_str(current_prev_notch),
                'prev_notch_duration': current_prev_duration,
                'delay': max(0.0, self.t - self.fixed_running_time),
                'current_gradient': self.train.front_grades[0]["grade"] if len(self.train.front_grades) > 0 else 0.0,
                'phase': self._get_current_phase_str(),
                'current_notch': self._get_current_notch_str(action_enum), 
                'next_limit_info': self._get_next_limit_info(),
                'next_gradient_info': self._get_next_gradient_info(),
                'forward_info': forward_info_str, 
                'backward_info': "後続列車なし"
            }
            try:
                llm_reward = self.reward_predictor.predict_reward(state_info)
            except Exception as e:
                print(f"[推論エラー]: {e}")
                pass
        
        llm_reward = max(0.0, min(1.0, llm_reward))  # 0.0〜1.0に強制クリップ
        self.last_llm_reward = llm_reward  # 分析保存用

        # --- 2. 終了判定 (done) と 絶対ルールの判定 ---
        done = False
        goal_reward = 0.0
        fail_penalty = 0.0
        
        # タイムオーバー（大幅な遅延失敗）
        if self.t >= self.departure_station["running_time"] + 60.0:
            fail_penalty = -10.0
            
        # 到着駅に近づいている場合の報酬
        if self.position >= self.arrival_station["position"] - 0.01 and self.speed <= 0.0:
            goal_reward = (max(min((abs(self.arrival_station["position"] - self.position)) * 1000, 10), 0.1) ** (-0.5)) - 0.31623
            
        # 先行列車への異常接近・衝突判定
        if self.fowerd_train_position is not None and self.speed <= 0 and self.position >= self.fowerd_train_position - 0.1:
            fail_penalty = -15.0

        # 駅のオーバーラン判定（駅を通り過ぎてまだ走っている場合）
        if self.speed > 0.0 and self.position > self.arrival_station["position"] + 0.005:
            fail_penalty = -5.0
            
        # 駅の規定距離より手前で停止してしまった場合は強制終了
        #if self.speed <= 0.0 and self.position < self.arrival_station["position"] - 0.01:
            #if self.fowerd_train_position is None or self.position < self.fowerd_train_position - 0.1:
                #done = True
        
        

        # --- 3. 最終的な報酬の合算 ---
        reward = llm_reward
        
        # 60秒以上遅延した場合，10m以内に停車，先行列車の手前で停止できた（100メートル手前）場合，エピソード終了
        if self.t >= self.departure_station["running_time"] + 60.0:
            done = True
        if self.speed<=0.0 and self.position>=self.arrival_station["position"]-0.01:
            done=True
        if self.fowerd_train_position is not None and self.speed<=0 and self.position>=self.fowerd_train_position-0.1:
            done=True
        
        # ▼▼▼ 変更: アクション保持時間の更新と直前ノッチの保存 ▼▼▼
        if self.pre_action == action_enum:
            self.holding_time += time_step
        else:
            self.prev_notch = self.pre_action
            self.prev_notch_duration = self.holding_time
            self.holding_time = time_step
        self.pre_action = action_enum
        

        return self.normalized_state, reward, done

    # --- LLM推論用の状態テキスト生成ヘルパー群 ---
    def _get_current_phase_str(self):
        if self.t <= 20.0:
            return "駅出発直後の加速フェーズ（20秒以内）"
        elif self.station_remaining_distance <= 0.4:
            return "次駅への減速フェーズ（駅手前400m以内）"
        elif self.current_speed_limit < 1000 and self.speed > self.current_speed_limit and self.train.section_remaining_distance <= 0.5:
            return "制限速度区間に接近中（500m以内に制限区間在り）"
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
            if dist_km <= 0.5:
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
    
    def _calc_brake_distance(self, test_speed):
        """
        修士本論の運動方程式(式2)および train.py の減速ロジックに完全準拠。
        現在速度からフルブレーキ（DECELERATE）で停止するまでの距離(km)を算出する。
        """
        if test_speed <= 0.0:
            return 0.0
            
        sim_speed = test_speed
        sim_position = 0.0  # ブレーキ開始位置からの相対距離[km]
        
        train = self.train
        grade_res = train.grade_resistance
        curve_res = train.curve_resistance
        time_step = train.time_step_base  # 0.01秒
        
        while sim_speed > 0:
            # 式(3): 走行抵抗の計算
            travel_res = 2.39 + 0.0224 * sim_speed + 0.00062 * (sim_speed**2)
            
            # ▼▼▼ train.pyと全く同じ運動方程式の実装 ▼▼▼
            # ブレーキ時の力 (Force) として train.DECELERATE を代入
            force = train.DECELERATE
            
            # 式(2): 加速度 accel [km/h/s] の算出
            accel = ((((force - travel_res) * train.WEIGTH_CORRECTION) - (grade_res + curve_res)) / train.FACTOR_OF_INERTIA)
            
            # 速度の更新: 速度 [km/h] += 加速度 [km/h/s] * 時間 [s]
            sim_speed += accel * time_step
            
            if sim_speed < 0:
                sim_speed = 0.0
                
            # 位置の更新: 距離 [km] += 速度 [km/h] * 時間 [h]
            sim_position += sim_speed * (time_step / 3600.0)
            # ▲▲▲ 修正ここまで ▲▲▲
            
        return sim_position


    # 外部（Tester）からLLMの出力値を抜くための分析用プロパティ
    @property
    def latest_rewards_info(self):
        return [self.last_llm_reward]

    # --- ▼▼▼ 【重要修正】DQN用観測ベクトルを10次元に拡張 ▼▼▼ ---
    @property
    def normalized_state(self):
        pre_action_c = 1.0 if self.pre_action == Actions.coasting else 0
        pre_action_a = 1.0 if self.pre_action == Actions.acceleration else 0
        pre_action_d = 1.0 if self.pre_action == Actions.deceleration else 0
        return [
            self.speed / 80.0,
            (max(self.station_remaining_distance, -0.5) + 0.5) / 2,
            (max(min(self.station_remaining_distance, 0.2), -0.05) + 0.05) * 4,
            self.remaining_time / 360.0,
            min(self.holding_time, 30) / 30.0,
            pre_action_c,
            pre_action_a,
            pre_action_d,
            (max(self.fowerd_train_remaining_distance, -0.5) + 0.5) / 2,
            self.cbtc_signal_speed / 80.0  # <--- 【追加】10次元目の特徴量（正規化されたCBTC指示速度）
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
            self.fowerd_train_remaining_distance,
            self.cbtc_signal_speed  # <--- 【追加】生値
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
        if self.fowerd_train is None: 
            return None
        return self.fowerd_train.position
    
    @property
    def fowerd_train_remaining_distance(self):
        if (self.fowerd_train_position is None): return self.station_remaining_distance
        return self.fowerd_train_position - self.position
    
    @property
    def remaining_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"] - self.t
        return self.fixed_running_time - self.t
    
    @property
    def fixed_running_time(self):
        if (self.fowerd_train_position is None): return self.departure_station["running_time"]
        for i in range(len(self.standerd_running)):
            if (self.fowerd_train_position < self.standerd_running[i]["position"]):
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
        if self.position < self.arrival_station["position"] - 0.1:
            return self.__time_step
        else:
            return self.__time_step * 0.1

    def read_csv(self, path):
        if path in self._csv_cache:
            return self._csv_cache[path]
        with codecs.open(path, "r", "utf-8", "ignore") as f:
            csv = pd.read_csv(f)
        self._csv_cache[path] = csv
        return csv

    @property
    def cbtc_signal_speed(self):
        """
        先行列車の50m手前を停止限界とするCBTC減速パターン（指示速度[km/h]）を計算する。
        先行列車がいない（またはパターンがない）場合は、その区間の基本制限速度を返す。
        """
        base_limit = self.current_speed_limit
        if self.fowerd_train_position is None:
            return base_limit
            
        target_distance = self.fowerd_train_remaining_distance - 0.05
        if target_distance <= 0.0:
            return 0.0
            
        low = 0.0
        high = base_limit
        cbtc_speed = base_limit
        
        for _ in range(15):
            mid = (low + high) / 2.0
            if self._calc_brake_distance(mid) <= target_distance:
                cbtc_speed = mid
                low = mid
            else:
                high = mid
                
        return min(cbtc_speed, base_limit)