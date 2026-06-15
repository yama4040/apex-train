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

    def reset(self, departure_index=None, delay=0.0, weight_correction=1.0, fowerd_train_position_offset=None, start_position_offset=0.0, fowerd_train_controls=None):
        if departure_index is None:
            departure_index = random.randrange(1)
        self.t = 0.0
        self.departure_station = self.stations[departure_index]
        self.arrival_station = self.stations[departure_index + 1]
        start_position = self.departure_station["position"] + start_position_offset
        self.train = Train(self.arrival_station["position"], start_position, 0.0, weight_correction)
        self.fowerd_train = None
        
        if (fowerd_train_controls): 
            ftc = self.read_csv(fowerd_train_controls)
            self.fowerd_train_controls = []
            for i in range(len(ftc)):
                action = Actions.deceleration
                if (ftc["action"][i] == "Actions.coasting"): action = Actions.coasting
                elif (ftc["action"][i] == "Actions.acceleration"): action = Actions.acceleration
                elif (ftc["action"][i] == "Actions.deceleration"): action = Actions.deceleration
                self.fowerd_train_controls.append({"time": i, "position": ftc["position"][i], "speed": ftc["speed"][i], "action": action})
            # ▼【変更】リストが空でない（データが1行以上ある）場合のみ初期化する
            if len(self.fowerd_train_controls) > 0:
                self.fowerd_train = Train(self.arrival_station["position"], self.fowerd_train_controls[0]["position"], self.fowerd_train_controls[0]["speed"], 1.0)
            else:
                self.fowerd_train = None
                print(f"\n[警告] {fowerd_train_controls} にデータがありません！先行列車なしとして扱います。")
        self.fowerd_train_position_offset = fowerd_train_position_offset
        
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
        if (self.fowerd_train is not None and self.fowerd_train.speed <= 0.0 and self.fowerd_train.position > self.arrival_station["position"]): self.fowerd_train = None
        if (self.fowerd_train is not None and self.t > self.fowerd_train_controls[-1]["time"]): self.fowerd_train.step(Actions.deceleration, 1)
        elif (self.fowerd_train is not None and round(self.t % 1, 1) == 0): self.fowerd_train.step(self.fowerd_train_controls[int(self.t)]["action"], 1)
        
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

            # ▼▼▼ 追加: 欠落していた要求ブレーキ距離の計算 ▼▼▼
            v_ms = max(0.0, self.speed / 3.6)
            decel_ms2 = 2.5 / 3.6
            fallback_req_dist = (v_ms ** 2) / (2 * decel_ms2) + (v_ms * self.time_step)
            req_dist_val = fallback_req_dist
            # ▲▲▲ 追加ここまで ▲▲▲

            state_info = {
                'speed_limit': self.current_speed_limit,
                'current_speed': self.speed,
                'dist_to_next_station': self.station_remaining_distance * 1000.0, 
                'time_to_next_station': self.remaining_time,  
                'req_stop_dist': req_dist_val,  # <--- これでエラーが解消します
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
                # エラー発生時はログを残す（デバッグ用）
                print(f"[推論エラー]: {e}")
                pass
        
        self.last_llm_reward = llm_reward  # 分析保存用

        # --- 2. 終了判定 (done) と 絶対ルールの判定 ---
        done = False
        goal_reward = 0.0
        fail_penalty = 0.0
        
        # タイムオーバー（大幅な遅延失敗）
        if self.t >= self.departure_station["running_time"] + 60.0:
            #done = True
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
            
        # ▼▼▼ 新規追加: 駅の規定距離より手前で停止してしまった場合は強制終了 ▼▼▼
        # ※先行列車の後ろで止まっている場合（上記の既存条件）とは切り分ける
        if self.speed <= 0.0 and self.position < self.arrival_station["position"] - 0.01:
            if self.fowerd_train_position is None or self.position < self.fowerd_train_position - 0.1:
                done = True
                # 必要に応じて「途中で止まってしまったペナルティ」を強化学習の報酬に加算することも検討してください
                # fail_penalty = -10.0 
        # ▲▲▲ 新規追加 ▲▲▲


        # --- 3. 最終的な報酬の合算 ---
        # 道中の良さ(LLM評価: -1~1)に、エピソードの結末（成功・失敗）をアドオンする
        #reward = llm_reward + goal_reward + fail_penalty
        reward = llm_reward
        
         #60秒以上遅延した場合，10m以内に停車，先行列車の手前で停止できた（100メートル手前）場合，エピソード終了
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
            # ノッチが切り替わった場合、これまでの操作を「直前」として退避
            self.prev_notch = self.pre_action
            self.prev_notch_duration = self.holding_time
            self.holding_time = time_step
        self.pre_action = action_enum
        # ▲▲▲ 変更 ▲▲▲
        

        return self.normalized_state, reward, done

    # --- LLM推論用の状態テキスト生成ヘルパー群 ---
    def _get_current_phase_str(self):
        # ▼【変更】学習用CSVと一言一句一致させます
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

    # 外部（Tester）からLLMの出力値を抜くための分析用プロパティ
    @property
    def latest_rewards_info(self):
        return [self.last_llm_reward]

    # --- 以下、既存のプロパティ群は一切変更なし ---
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
        if (self.fowerd_train is None): return self.departure_station["position"] + self.fowerd_train_position_offset
        else: return self.fowerd_train.position

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
