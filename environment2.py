from train import Train
from actions import Actions
import numpy as np
import random
import pandas as pd
import codecs
import math

from required_speed import calculate_required_speed, brake_stop_distance_m

# 単一評価値予測器をインポート
try:
    from direct_reward_predictor2 import DirectRewardPredictor
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
                
        # 標準走行曲線（計画ダイヤ）は現在位置基準の遅延計算にも使用するため常にロードする
        self.standerd_running = []
        try:
            sr_csv = self.read_csv(f"./input/sr_{departure_index}.csv")
            for i in range(len(sr_csv)):
                self.standerd_running.append({"position": sr_csv["position"][i], "time": sr_csv["time"][i]})
        except Exception:
            # 標準走行曲線が無い区間では位置基準の遅延計算をスキップする（従来定義で代用）
            pass

        if (start_position_offset != 0.0):
            # 駅間途中から発車する場合は、開始位置に対応する計画通過時刻＋出発遅延を初期時刻とする
            # （旧実装は標準走行曲線の先頭行の時刻を使っており、開始位置と時刻が対応していなかった）
            scheduled = self._scheduled_time_at(start_position)
            self.t = (scheduled if scheduled is not None else 0.0) + delay
        else:
            # 駅から発車する場合も出発遅延を初期時刻に反映する
            # （旧実装ではdelay引数が無視されており、遅延ありのシナリオが機能していなかった）
            self.t = delay

        # フェーズ判定（駅出発直後20秒以内）用にエピソード開始時刻を保持する
        self.episode_start_t = self.t

        # 停滞（ちんたら運転）検出用のチェックポイント
        self._stall_check_t = self.t
        self._stall_check_pos = self.train.position

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

            current_gradient_val = self.train.front_grades[0]["grade"] if len(self.train.front_grades) > 0 else 0.0

            # ▼▼▼ 要求ブレーキ距離の計算 ▼▼▼
            # train.pyの実ダイナミクス（減速ノッチ2.5km/h/s＋走行抵抗＋勾配抵抗）と一致する
            # 物理モデル（required_speed.py）で算出し、LLM評価データセット
            # （evaluate_csv_with_llm.py）・apex2.pyのCSV出力と完全に統一する。
            req_dist_val = brake_stop_distance_m(self.speed, current_gradient_val)
            # ▲▲▲ ここまで ▲▲▲

            # ▼▼▼ 追加: 必要速度（巡航速度）の算出。evaluate_csv_with_llm.pyと同一ロジック（required_speed.py） ▼▼▼
            required_speed_val = calculate_required_speed(
                current_speed=self.speed,
                dist_to_next_station=self.station_remaining_distance * 1000.0,
                time_to_next_station=self.remaining_time,
                speed_limit=self.current_speed_limit,
                current_gradient=current_gradient_val,
            )
            # ▲▲▲ 追加ここまで ▲▲▲

            # ▼▼▼ 【重要修正】'signal_speed' を追加して報酬予測側へ渡す ▼▼▼
            state_info = {
                'speed_limit': self.current_speed_limit,
                'signal_speed': self.cbtc_signal_speed,  # <--- 【追加】CBTC指示速度
                'current_speed': self.speed,
                'required_speed': required_speed_val,  # <--- 【追加】必要速度（巡航速度）
                'dist_to_next_station': self.station_remaining_distance * 1000.0,
                # 残り時間は0でクリップし、超過分はdelayで表現する（apex2.pyのCSV出力と同一の扱い）
                'time_to_next_station': max(0.0, self.remaining_time),
                'req_stop_dist': req_dist_val,
                'holding_time': current_holding_time,
                'prev_notch': get_prev_notch_str(current_prev_notch),
                'prev_notch_duration': current_prev_duration,
                'delay': self.current_delay,
                'current_gradient': current_gradient_val,
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
        #llm_reward = (llm_reward * 2) - 1
        self.last_llm_reward = llm_reward  # 分析保存用

        # --- 2. 終了判定 (done) と 絶対ルールの判定 ---
        done = False
        goal_reward = 0.0
        fail_penalty = 0.0
        
        """
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
        """
        

        # --- 3. 最終的な報酬の合算 ---
        # 【制約】報酬はLLMを蒸留したNNの出力のみとする（時間ペナルティや終端ボーナス等の
        # 環境側の報酬加算は禁止）。ちんたら運転などの問題行動への対処は、
        # LLM評価プロンプトのルール（ちんたら運転=0.0）と下記のエピソード終了条件で行う。
        # 【重要】NN出力は「1秒あたりの評価値」とみなし、ステップ幅でスケールする。
        # 駅手前100mではtime_stepが0.1秒に短縮されるため、スケールしないと実時間あたりの
        # 報酬密度が10倍になり、「駅手前に留まり続けて報酬を稼ぐ」搾取行動が最適になってしまう。
        reward = llm_reward * (time_step / self.__time_step)

        # --- 4. エピソード終了条件 ---
        goal_reached = False
        failed = False  # 分析・ログ用（報酬には使用しない）

        # ① タイムオーバー（大幅な遅延）
        if self.t >= self.departure_station["running_time"] + 60.0:
            done = True
            failed = True

        # ② 目標達成（駅の許容範囲内に停止）
        # -10m(0.01km) 〜 +5m(0.005km) の範囲で速度0になったら無事到着として終了
        if self.position >= self.arrival_station["position"] - 0.01 and self.position <= self.arrival_station["position"] + 0.005 and self.speed <= 0.0:
            done = True
            goal_reached = True

        # ③ 先行列車への異常接近・衝突判定
        if self.fowerd_train_position is not None and self.speed <= 0.0 and self.position >= self.fowerd_train_position - 0.05:
            done = True
            failed = True

        # ④ オーバーラン判定（駅を通り過ぎた）
        # 駅の許容停止位置（+5m）を越えて走っている場合は即終了
        if self.speed > 0.0 and self.position > self.arrival_station["position"] + 0.005:
            done = True
            failed = True

        # ⑤ 【重要】手前での停止（ショートオーバーラン）
        # 駅に届いていない（-10m未満）のに実質的に停止してしまった場合。
        # 判定を「速度0」から「0.5km/h以下」に拡大：速度を0.005km/h等に保つクリープ走行で
        # この判定を回避し、駅手前でホバリングして報酬を稼ぐ搾取行動が観測されたため。
        # 正常なブレーキ停車では速度が0.5km/hを下回るのは停止位置の直前（数cm手前）であり、
        # 駅の10m以上手前でこの速度域に入ること自体が停止失敗を意味する。
        if self.speed <= 0.5 and self.position < self.arrival_station["position"] - 0.01:
            # ただし、先行列車がいて、その手前で止まった場合は正しい「信号待ち」なので除外
            if self.fowerd_train_position is None or self.position < self.fowerd_train_position - 0.1:
                done = True  # 即座にエピソードを打ち切る
                failed = True

        # ⑥ 【追加】停滞（ちんたら運転）検出
        # 極低速で走り続ければ「実質停止」の判定（⑤）を回避してエピソードを引き延ばせて
        # しまうため（実測では平均2.4km/hの匍匐走行が観測された）、進捗ベースで打ち切る。
        # 駅手前400m以内（減速フェーズ）ではtime_step短縮による報酬稼ぎの温床になるため、
        # 判定窓を10秒に短縮し、より早く停滞を検出する。
        near_station = self.station_remaining_distance <= 0.4
        stall_window = 10.0 if near_station else 30.0
        if self.t - self._stall_check_t >= stall_window:
            progress = self.position - self._stall_check_pos
            # 10秒窓: 5m未満（平均1.8km/h未満）／ 30秒窓: 25m未満（平均3km/h未満）
            min_progress = 0.005 if near_station else 0.025
            if not goal_reached:
                if self.fowerd_train_position is None:
                    # 先行列車がいない場合: 規定の前進がなければちんたら運転として打ち切る
                    if progress < min_progress:
                        done = True
                        failed = True
                elif self.position < self.fowerd_train_position - 0.1 and progress < 0.005:
                    # 先行列車がいる場合: 信号待ちの可能性があるため、ほぼ完全な停滞
                    # （窓内で5m未満）のみを対象とし、先行列車の直後（100m以内）での待機は除外する
                    done = True
                    failed = True
            self._stall_check_t = self.t
            self._stall_check_pos = self.position

        """
        # 60秒以上遅延した場合，10m以内に停車，先行列車の手前で停止できた（50メートル手前）場合，エピソード終了
        if self.t >= self.departure_station["running_time"] + 60.0:
            done = True
        if self.speed<=0.0 and self.position>=self.arrival_station["position"]-0.01:
            done=True
        if self.fowerd_train_position is not None and self.speed<=0 and self.position>=self.fowerd_train_position-0.05:
            done=True
        """
        
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
        # 駅停車完了（駅の10m以内かつ実質停止）。apex2.pyのテスター側のLLM評価用CSV生成と同一条件。
        # この分岐がないと、停車した瞬間の終端報酬が「減速フェーズ」として評価されてしまい、
        # プロンプトの「停車完了フェーズ＝停止位置誤差の段階評価」ルールがRL中に一度も発動しない。
        # また、DQN観測ベクトルのphase_5（停車完了）も常に0の死んだ入力になってしまう。
        if self.station_remaining_distance * 1000.0 <= 10.0 and self.speed <= 0.1:
            return "駅停車完了（速度0km/h）"
        # 出発遅延がある場合もエピソード開始からの経過時間で判定する
        if self.t - getattr(self, 'episode_start_t', 0.0) <= 20.0:
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

    # --- DQN観測ベクトル用の数値版ヘルパー（0.5km以内に変化がなければNoneを返す） ---
    def _get_next_limit_numeric(self):
        sections = self.train.front_sections
        if len(sections) > 1:
            dist_km = sections[0]["distance"]
            next_limit = sections[1]["speed_limit"]
            if dist_km <= 0.5:
                return dist_km, next_limit
        return None, None

    def _get_next_gradient_numeric(self):
        grades = self.train.front_grades
        if len(grades) > 1:
            dist_km = grades[0]["distance"]
            next_grade = grades[1]["grade"]
            if dist_km <= 0.5:
                return dist_km, next_grade
        return None, None
    
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
            
            # ▼▼▼ 修論 式(4.1) u=1 と同じ構造の運動方程式（train.pyのstep()と一致） ▼▼▼
            # dv/dt = DECELERATE/kw - (Rr/kw + Rg + Rc)/28.34467 （WEIGTH_CORRECTION = 1/kw に相当）
            # ※旧実装はDECELERATEを引張力[kg/t]として式に代入していたが、
            #   DECELERATEは減速度[km/h/s]そのものであるため誤りだった。
            accel = ((((0 - travel_res) * train.WEIGTH_CORRECTION) - (grade_res + curve_res)) / train.FACTOR_OF_INERTIA)
            accel += train.DECELERATE * train.WEIGTH_CORRECTION
            
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

    # --- ▼▼▼ 【重要修正】DQN用観測ベクトルを25次元に拡張 ▼▼▼ ---
    @property
    def normalized_state(self):
        # 1. 既存の行動フラグ
        pre_action_c = 1.0 if self.pre_action == Actions.coasting else 0.0
        pre_action_a = 1.0 if self.pre_action == Actions.acceleration else 0.0
        pre_action_d = 1.0 if self.pre_action == Actions.deceleration else 0.0

        # 2. 追加特徴量の計算
        req_stop_dist = self._calc_brake_distance(self.speed)
        margin_stop_dist = self.station_remaining_distance - req_stop_dist
        f_speed = self.fowerd_train.speed if self.fowerd_train is not None else 80.0 # 先行列車がいない場合は80km/hとする

        # 3. フェーズのOne-Hotエンコーディング
        phase_str = self._get_current_phase_str()
        phase_1 = 1.0 if phase_str == "駅出発直後の加速フェーズ（20秒以内）" else 0.0
        phase_2 = 1.0 if phase_str == "巡航フェーズ（駅間走行中）" else 0.0
        phase_3 = 1.0 if phase_str == "制限速度区間に接近中（500m以内に制限区間在り）" else 0.0
        phase_4 = 1.0 if phase_str == "次駅への減速フェーズ（駅手前400m以内）" else 0.0
        phase_5 = 1.0 if phase_str == "駅停車完了（速度0km/h）" else 0.0

        # 4. 勾配・この先の制限速度変化・直前操作の継続時間（新規追加分）
        current_gradient = self.train.front_grades[0]["grade"] if len(self.train.front_grades) > 0 else 0.0
        next_grade_dist, next_grade_val = self._get_next_gradient_numeric()
        next_limit_dist, next_limit_val = self._get_next_limit_numeric()
        # 該当なしの場合は「0.5km先までの変化なし」「現在値から変化なし」を意味する値をデフォルトにする
        next_grade_dist_norm = (next_grade_dist / 0.5) if next_grade_dist is not None else 1.0
        next_grade_val_final = next_grade_val if next_grade_val is not None else current_gradient
        next_limit_dist_norm = (next_limit_dist / 0.5) if next_limit_dist is not None else 1.0
        next_limit_val_final = next_limit_val if next_limit_val is not None else self.current_speed_limit

        # 5. 観測ベクトルの構築（合計 25次元）
        # ※各値はニューラルネットワークが学習しやすいよう、おおよそ -1.0 〜 1.0 または 0.0 〜 1.0 にスケーリングしています
        return np.array([
            # --- 既存の入力（一部スケーリング見直し） ---
            self.speed / 80.0,                                               # 1. 現在の速度
            (max(self.station_remaining_distance, -0.5) + 0.5) / 2.0,        # 2. 駅までの距離（広域）
            (max(min(self.station_remaining_distance, 0.2), -0.05) + 0.05) * 4.0, # 3. 駅までの距離（ズーム）
            self.remaining_time / 360.0,                                     # 4. 残り時間
            min(self.holding_time, 30.0) / 30.0,                             # 5. 同じ操作を続けている時間
            pre_action_c,                                                    # 6. 直前の行動（惰行）
            pre_action_a,                                                    # 7. 直前の行動（加速）
            pre_action_d,                                                    # 8. 直前の行動（減速）
            (max(self.fowerd_train_remaining_distance, -0.5) + 0.5) / 2.0,   # 9. 先行列車までの距離

            # --- 新規追加の入力 ---
            self.cbtc_signal_speed / 80.0,                                   # 10. CBTCの信号現示
            self.current_speed_limit / 80.0,                                 # 11. 路線制限速度
            req_stop_dist / 1.0,                                             # 12. 必要なブレーキ距離（最大1km程度を想定）
            np.clip(margin_stop_dist, -0.5, 1.5) / 1.5,                      # 13. 停車余裕マージン（負の値はオーバーラン危険域）
            phase_1,                                                         # 14. フェーズ：加速
            phase_2,                                                         # 15. フェーズ：巡航
            phase_3,                                                         # 16. フェーズ：制限接近
            phase_4,                                                         # 17. フェーズ：減速
            phase_5,                                                         # 18. フェーズ：停車完了
            f_speed / 80.0,                                                  # 19. 先行列車の速度

            # --- 勾配・この先の制限速度変化・直前操作継続時間（今回追加） ---
            np.clip(current_gradient / 40.0, -1.0, 1.0),                     # 20. 現在の勾配
            next_grade_dist_norm,                                            # 21. この先の勾配変化までの距離（0.5km換算、変化なしなら1.0）
            np.clip(next_grade_val_final / 40.0, -1.0, 1.0),                 # 22. この先の勾配値（変化なしなら現在の勾配と同値）
            next_limit_dist_norm,                                            # 23. この先の制限速度変化までの距離（0.5km換算、変化なしなら1.0）
            next_limit_val_final / 80.0,                                     # 24. この先の制限速度（変化なしなら現在の制限速度と同値）
            min(self.prev_notch_duration, 30.0) / 30.0                       # 25. 直前操作（1つ前のノッチ）の継続時間
        ], dtype=np.float32)

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

    def _scheduled_time_at(self, position_km):
        """標準走行曲線（計画ダイヤ）から指定位置の計画通過時刻[s]を線形補間で求める。
        標準走行曲線が無い場合はNoneを返す。"""
        sr = getattr(self, 'standerd_running', None)
        if not sr:
            return None
        if position_km <= sr[0]["position"]:
            return sr[0]["time"]
        for i in range(1, len(sr)):
            if position_km <= sr[i]["position"]:
                p0, t0 = sr[i - 1]["position"], sr[i - 1]["time"]
                p1, t1 = sr[i]["position"], sr[i]["time"]
                if p1 <= p0:
                    return t1
                return t0 + (t1 - t0) * (position_km - p0) / (p1 - p0)
        return sr[-1]["time"]

    @property
    def current_delay(self):
        """標準運転時間を過ぎてから何秒経ったか[s]。
        駅間走行の残り時間（remaining_time）が負になった場合、その絶対値が遅延時間となる
        （例：残り時間が-10秒ならdelay=10秒）。残り時間がある間は0。"""
        return max(0.0, -self.remaining_time)

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