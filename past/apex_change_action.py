"""
・行動を変えたポイントをすべて抽出して-1.0～+1.0の範囲で報酬をLLMが与えることで、その行動を評価
"""


import time
import datetime
import os


import ray
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# Ray の起動待機時間を延長
os.environ['RAY_raylet_start_wait_time_s'] = '120'
# 【今回追加】Rayの過敏なメモリ監視・強制終了を無効化する
#os.environ['RAY_memory_monitor_refresh_ms'] = '0'

from segment_tree import SumTree
from model import QNetwork

from environment import Environment
from actions import Actions
import random
import sys
#sys.dont_write_bytecode = True

from openai import OpenAI
import json
# APIキーの設定（環境変数 GEMINI_API_KEY から読み込むのが安全です）
# 事前にターミナルで export GEMINI_API_KEY="あなたのAPIキー" を実行してください
"""
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
"""
tf.config.set_visible_devices([], "GPU")

#このクラスがRayによって別プロセスとして非同期に実行
@ray.remote
class Actor:
    def __init__(self, pid, epsilon, dir_name, gamma, num_states, time_step):
        tf.config.set_visible_devices([], "GPU")
        self.pid = pid
        self.time_step = time_step
        self.num_states = num_states
        self.env = Environment(self.time_step)
        self.dir_name = dir_name

        self.q_network = QNetwork(self.num_states)
        self.epsilon = epsilon  #探索率
        self.__gamma = gamma    #割引率
        self.buffer = []

        self.define_network()
        self.episode_rewards = 0
        self.episode_count = 0

    # ローカルLLM (OpenAI互換API) を呼び出すメソッド
    def evaluate_with_local_llm(self, log_text):
        """ローカルLLMに運転ログを評価させ、追加報酬のJSONを返す"""
        try:
            # ▼▼▼ Clientが存在しない初回のみ初期化する ▼▼▼
            if not hasattr(self, "llm_client"):
                api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
                base_url = os.environ.get("OPENAI_BASE_URL")
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                self.llm_client = OpenAI(**client_kwargs)
            # ▲▲▲ 修正ここまで ▲▲▲
            
            # ▼▼▼ 役割とルールだけを定義する「システムプロンプト」 ▼▼▼
            system_prompt = """
            あなたは熟練の鉄道運転士であり、DQN強化学習を指導するメンターです。
            提供された【行動変化の抽出ログ】にある複数の「決断ポイント（*マークのStep）」すべてに対して、路線全体を見据えたマクロな視点から評価を行ってください。
            また、評価を行う際は、その決断ポイントに至るまでの「保持時間（惰行・加速・減速の状態をどれだけ長く維持していたか）」や「速度」「残時間」「残距離」「制限速度」「勾配」
            「先行列車との距離」などの要素も考慮に入れてください。
            
            【運転士の好み】
            - 乗り心地を重視し、頻繁な行動変更を避ける傾向がある。
            - 頻繁なノッチ操作をせず、定時運行を実現するために、戦略的なタイミングで行動を変えることを好む。
            - 無駄なブレーキを避けるため、先行列車に近いづいているときは早めに惰行に移ることを好む。
            
            【評価基準（追加報酬の目安：-1.0 ～ 1.0）】
            ・正の報酬 (0.0 ～ 1.0)：直前の行動の「保持時間」が十分に長く（乗り心地が良い）、かつ定時性や省エネの観点からそのタイミングでの行動変更が戦略的に完璧な場合。
            ・負の報酬 (-1.0 ～ -0.1)：保持時間が短すぎる（ガチャガチャ切り替えている）、または速度や遅延の状況から見て、行動を変えるタイミングが早すぎる・遅すぎる場合。
            ・ニュートラル (0.0)：特に致命的な問題がない場合。

            【出力形式】
            必ず以下のJSONスキーマに従い、すべての決断ポイントの評価を配列で出力してください。
            {
                "evaluations": [
                    {
                        "step": <int: 評価対象のStep番号（*マークの行）>,
                        "llm_reward": <float: 追加報酬の値（-1.0 〜 1.0）>,
                        "reason": "<string: その判断に対する評価理由（50文字程度）>"
                    }
                ]
            }
            """

            # OpenAI APIの形式でリクエストを送信
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": log_text}
                ],
                # JSON形式での出力を強制（ローカルモデルが対応している場合）
                response_format={ "type": "json_object" },
                temperature=0.3 # 評価を安定させるため少し低めに設定
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"[Local LLM API Error] {e}")
            # エラー時は学習を止めないように報酬0で返す
            return {"llm_reward": 0.0, "reason": "API Error"}
    
    #ネットワークの定義と初期化
    def define_network(self):
        env = Environment(self.time_step)
        state = env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    #エピソードを実行して経験を収集し、TD誤差を計算して返す
    def rollout(self, current_weights):
        self.episode_count += 1
        self.q_network.set_weights(current_weights)
        r=random.random()   #エピソードの初期化方法をランダムに選択
        #第一引数：出発駅のインデックス，第二引数：出発駅の遅延時間，第三引数：重量補正値，第四引数：前方列車の位置オフセット，第五引数：出発位置のオフセット，第六引数：前方列車の制御データ（CSVファイル）
        if (r<0.5):
            self.state = self.env.reset(11,0.0,1.0)
        elif (r<0.75):
            self.state = self.env.reset(11,0.0,1.0,random.uniform(0.2,1.5))
        else:
            self.state = self.env.reset(11,random.uniform(0.0,20),1.0,None,random.uniform(0.2,1.5))
        self.episode_rewards = 0
        done = False
        
        # エピソードの実行と経験の収集
        while not done:
            state = self.state  #現在の状態
            
            # ▼▼▼ 修正: sample_action から Q値(qs) も受け取る ▼▼▼
            action, qs = self.q_network.sample_action(np.array(state)[np.newaxis,...], self.epsilon, self.env.forbidden_action)
            # ▲▲▲
            
            priority_correction=(0.1-(min(max(self.env.station_remaining_distance,0.0),0.1)+0.001))*500+1
            
           # ▼▼▼ 修正：シミュレータのメソッドを呼ばず、配列から現在位置のデータを独自に検索する ▼▼▼
            current_pos = self.env.position  # 現在の列車のキロ程（位置）
            
            # 1. 制限速度の安全な検索
            safe_limit = 0.0
            sections = self.env.train.track.sections
            for i in range(len(sections) - 1):
                if current_pos < sections[i + 1]["start"]:
                    safe_limit = sections[i]["speed_limit"]
                    break
            else:
                if len(sections) > 0:
                    safe_limit = sections[-1]["speed_limit"]
            
            # 2. 勾配の安全な検索
            safe_gradient = 0.0
            grades = self.env.train.track.grade
            for i in range(len(grades) - 1):
                if grades[i]["start"] <= current_pos and current_pos <= grades[i + 1]["start"]:
                    safe_gradient = grades[i]["grade"]
                    break
            else:
                if len(grades) > 0 and current_pos >= grades[-1]["start"]:
                    safe_gradient = grades[-1]["grade"]

            # エラー防止のための安全な数値変換
            safe_f_train_dist = self.env.fowerd_train_remaining_distance if self.env.fowerd_train_remaining_distance is not None else 999.0

            # raw_info に独自検索した安全な値（safe_limit, safe_gradient）を格納
            raw_info = {
                "speed": float(self.env.speed),
                "distance": float(self.env.station_remaining_distance),
                "rem_time": float(self.env.remaining_time),
                "hold_time": float(self.env.holding_time),
                "limit": float(safe_limit),          # ← 追加復活！
                "gradient": float(safe_gradient),    # ← 追加復活！
                "f_train_dist": float(safe_f_train_dist),
                "qs": np.array(qs).flatten().tolist()
            }
            # ▲▲▲ 修正ここまで ▲▲▲

            next_state, reward, done = self.env.step(action)
            nest_forbidden_action=self.env.forbidden_action 
            self.episode_rewards += reward  
            
            transition = (state, action, reward, next_state, done, nest_forbidden_action, self.gamma, priority_correction, raw_info)
            self.buffer.append(transition)
            self.state = next_state
        
        # -------------------------------------------------------------
        # (既存のコード) エピソードのループが終了し、transitions が完成した直後
        # -------------------------------------------------------------

        # === LLMによる自動評価と報酬補正（追加部分） ===
        # 学習が進んだ100エピソード目以降で発動 (self.episode_countの変数がActorにある前提です。なければ適宜調整してください)
       # review_interval = 50  # NエピソードごとにLLM評価を行う
        
        #if hasattr(self, 'episode_count') and self.episode_count >= 10 and (self.episode_count + self.pid) % review_interval == 0:
        # 全体を通じて「約50エピソードに1回」のペースにするための確率（1/50 = 2%）
       # === LLMによる自動評価と報酬補正（API保護＆CSV圧縮版） ===
        
        # 初回のみ、最後にAPIを呼んだ時間を初期化（Actorごとに10秒の時差をつける！）
        if not hasattr(self, 'last_api_call_time'):
            # Actor 0 は現在時刻-90秒で即時実行可能、Actor 14 は+50秒（現在時刻+140-90）まで待機
            self.last_api_call_time = time.time() - 90 + (self.pid * 10)

        current_time = time.time()
        
        start_episode=200  # 10エピソード以降でLLM評価を開始する
        review_episode_interval = 10
        
        if hasattr(self, 'episode_count') and self.episode_count >= start_episode and (self.episode_count % review_episode_interval == self.pid % review_episode_interval):
            if (current_time - self.last_api_call_time) > 10:
                if len(self.buffer) > 10:
                    self.last_api_call_time = current_time
                    
                    # 1. 決断ポイント（ノッチが変化した瞬間）の抽出
                    decision_points = []
                    prev_action = self.buffer[0][1]
                    for i in range(1, len(self.buffer)):
                        curr_action = self.buffer[i][1]
                        if prev_action != curr_action:
                            decision_points.append(i)
                            prev_action = curr_action
                    
                    # APIトークン保護のため、決断ポイントの上限を20箇所に制限
                    MAX_EVALS = 20
                    if len(decision_points) > MAX_EVALS:
                        # 等間隔にサンプリングして全体を評価させる
                        indices = np.linspace(0, len(decision_points) - 1, MAX_EVALS, dtype=int)
                        decision_points = [decision_points[idx] for idx in indices]

                    if len(decision_points) > 0:
                        log_text = "【路線全体の情報】\n"
                        log_text += f"エピソード総ステップ数: {len(self.buffer)}\n\n"
                        log_text += "【行動変化の抽出ログ】\n"
                        
                        for dp in decision_points:
                            log_text += f"--- 決断ポイント Step {dp} ---\n"
                            log_text += "Step,残距(km),残時間(s),速度(km/h),制限,勾配(‰),先行(km),保持(s),行動\n"
                            
                            start_idx = max(0, dp - 1)
                            end_idx = min(len(self.buffer), dp + 2)
                            
                            for i in range(start_idx, end_idx):
                                t = self.buffer[i]
                                raw = t[8]
                                marker = " *" if i == dp else ""
                                action_str = ["惰行", "加速", "減速"][t[1]]
                                log_text += f"{i},{round(raw['distance'],2)},{round(raw['rem_time'],1)},{round(raw['speed'],1)},{round(raw['limit'],1)},{round(raw['gradient'],1)},{round(raw['f_train_dist'],2)},{round(raw['hold_time'],1)},{action_str}{marker}\n"
                            log_text += "\n"

                        print(f"[Actor {self.pid}] エピソード終了。マクロ評価を実行中... (決断ポイント: {len(decision_points)}箇所)")
                        llm_result = self.evaluate_with_local_llm(log_text)
                        
                        # 2. 評価結果のパースと保存
                        evals_dict = {}
                        if "evaluations" in llm_result:
                            log_file_path = os.path.join(self.dir_name, "llm_eval.jsonl")
                            human_log_path = os.path.join(self.dir_name, "human_readable_log.txt")
                            
                            with open(log_file_path, "a", encoding="utf-8") as f, open(human_log_path, "a", encoding="utf-8") as hf:
                                hf.write(f"========== Actor {self.pid} / Episode {self.episode_count} (Macro Eval) ==========\n")
                                hf.write(log_text + "\n【LLMの評価結果】\n")
                                
                                for item in llm_result["evaluations"]:
                                    step_num = item.get("step")
                                    reward_val = float(item.get("llm_reward", 0.0))
                                    reason = item.get("reason", "")
                                    
                                    if step_num is not None:
                                        evals_dict[step_num] = reward_val
                                        
                                        log_entry = {
                                            "actor_id": self.pid,
                                            "episode": self.episode_count,
                                            "step": step_num,
                                            "reward": reward_val,
                                            "reason": reason
                                        }
                                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                                        hf.write(f"- Step {step_num} | 追加報酬: {reward_val} | 理由: {reason}\n")
                                
                                hf.write("========================================================================\n\n")

                        # 3. 区間報酬（インターバル・リワード）の適用
                        # 行動が変わるまで、その区間全体にLLMの評価報酬を継続して加算する
                        current_llm_reward = 0.0
                        current_action = None
                        
                        for i in range(len(self.buffer)):
                            action = self.buffer[i][1]
                            
                            # 実際の行動が切り替わったら、いったん追加報酬をリセット
                            if action != current_action:
                                current_llm_reward = 0.0
                                current_action = action
                                
                            # このステップが評価対象（決断ポイント）であれば報酬額を更新
                            if i in evals_dict:
                                current_llm_reward = max(-0.1, min(0.1, evals_dict[i]))
                                
                            # 現在の区間報酬が0でなければ加算して上書き
                            if current_llm_reward != 0.0:
                                t = list(self.buffer[i])
                                t[2] += current_llm_reward
                                self.buffer[i] = tuple(t)

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # (既存のコード) この直後に self.q_network を使ってTQ値とTD誤差(td_errors)を計算する処理が続く
        # -------------------------------------------------------------

        states = np.vstack([transition[0] for transition in self.buffer])
        actions = np.array([transition[1] for transition in self.buffer])
        rewards = np.vstack([transition[2] for transition in self.buffer])
        next_states = np.vstack([transition[3] for transition in self.buffer])
        dones = np.vstack([transition[4] for transition in self.buffer])
        next_forbidden_actions=np.vstack([transition[5] for transition in self.buffer])
        gammas=np.vstack([transition[6] for transition in self.buffer])
        
        
        
        next_qvalues = self.q_network(next_states)      #次の状態に対するQ値をネットワークで予測
        next_qvalues=next_qvalues+(next_forbidden_actions*-1.0 * (10**12))  #次の状態での禁止行動に対して非常に大きな負の値を加算して、これらの行動が選択されないようにする(-1兆)
        next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)   #次の状態での最大Q値を持つ行動を選択
        next_actions_onehot = tf.one_hot(next_actions, len(Actions))        #次の状態での最大Q値を持つ行動をワンホットエンコード
        #次の状態での最大Q値を計算（禁止行動は非常に大きな負の値が加算されているため、これらの行動は選択されない）
        next_maxQ = tf.reduce_sum(next_qvalues * next_actions_onehot, axis=1, keepdims=True)
        #ターゲットとなるQ値(TQ)を計算（報酬 + 割引率 * (1 - エピソード終了フラグ) * 次の状態での最大Q値）
        TQ = rewards + gammas * (1 - dones) * next_maxQ

        qvalues = self.q_network(states)    #現在の状態に対するQ値をネットワークで予測
        actions_onehot = tf.one_hot(actions, len(Actions))  #現在の状態で実際に選択した行動をワンホットエンコード
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)  #現在の状態で実際に選択した行動のQ値を計算
        
        #TD誤差を計算（ターゲット値 - 現在のQ値）
        #td_errors = (TQ - Q).numpy().flatten()
        td_errors = (TQ - Q).numpy().flatten().tolist()
        #transitions = self.buffer
        #self.buffer = []
        
        # ▼▼▼ 修正: Rayのシリアライズバグ（SIGSEGV / opcode）を完全に防ぐ ▼▼▼
        # NumPyスカラー型が混入したままRay通信を行うとActorのメモリが破壊されるため、
        # 通信前にすべての要素を純粋なPythonの型（int, float, bool, list）にサニタイズ（洗浄）します。
        clean_transitions = []
        for t in self.buffer:
            s = np.array(t[0], dtype=float).tolist()  # 状態を純粋なPythonリストに変換
            a = int(t[1])                             # 行動を純粋なintに変換
            r = float(t[2])                           # 報酬を純粋なfloatに変換
            ns = np.array(t[3], dtype=float).tolist() # 次の状態を純粋なPythonリストに変換
            done_flag = bool(t[4])                    # 終了フラグを純粋なboolに変換
            nfa = np.array(t[5], dtype=float).tolist()# 禁止行動を純粋なPythonリストに変換
            g = float(t[6])                           # 割引率を純粋なfloatに変換
            pc = float(t[7])                          # 優先度補正値を純粋なfloatに変換
            
            clean_transitions.append((s, a, r, ns, done_flag, nfa, g, pc))
            
        self.buffer = []

        # 計算したTD誤差，サニタイズ済みの1エピソード分の経験リスト，このActorのIDを返す
        return td_errors, clean_transitions, self.pid
        # ▲▲▲ 修正ここまで ▲▲▲
    
    @property
    def gamma(self):
        return self.__gamma**(self.env.time_step/self.time_step)

#このクラスは経験リプレイバッファを管理し、優先度付き経験リプレイのサンプリングと優先度の更新を行う
class Replay:
    def __init__(self, buffer_size, save_dir):

        self.buffer_size = buffer_size  #リプレイバッファの最大サイズ
        self.priorities = SumTree(capacity=self.buffer_size)    #優先度を管理するセグメントツリー
        self.buffer = [None] * self.buffer_size #実際の経験を保存するリスト

        #優先度の計算に使用する指数（α）。αが0の場合は通常の経験リプレイ（ランダム）、1の場合は完全な優先度付き経験リプレイになる。
        self.alpha = 0.6

        self.count = 0
        self.is_full = False
        
    #経験追加メソッド
    def add(self, td_errors, transitions):
        assert len(td_errors) == len(transitions)
        #優先度の計算：TD誤差の絶対値に小さな定数を加えてからα乗することで、優先度を計算。
        #これにより、TD誤差が大きい経験ほど優先的にサンプリングされるようになる。
        priorities = (np.abs(td_errors) + 0.001) ** self.alpha
        for priority, transition in zip(priorities, transitions):
            #self.priorities[self.count] = priority * transition[-1] #優先度に補正値を掛けることで、駅近の経験の優先度をさらに高くする
            # ▼▼▼ 修正: transition[-1] を transition[7] に変更 ▼▼▼
            self.priorities[self.count] = priority * transition[7] 
            # ▲▲▲ 修正ここまで ▲▲▲
            self.buffer[self.count] = transition
            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    #優先度の更新メソッド
    def update_priority(self, sampled_indices, td_errors, priority_corrections):
        assert len(sampled_indices) == len(td_errors)
        for idx, td_error, priority_correction in zip(sampled_indices, td_errors, priority_corrections):
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.priorities[idx] = priority*priority_correction

    #優先度付き経験リプレイのサンプリングメソッド
    def sample_minibatch(self, batch_size, beta):

        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        #: compute prioritized experience replay weights
        weights = []    #サンプリングされた経験の重みを格納するリスト
        current_size = len(self.buffer) if self.is_full else self.count #現在のリプレイバッファのサイズを計算
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum() #その経験が選ばれる確率を計算（自分の優先度/全優先度の合計）
            weight = (prob * current_size) ** (-beta)   #重要度サンプリングの重みを計算（(1/(N*P(i)))^β）。これにより、優先度が高い経験ほど重みが小さくなり、学習のバイアスを補正する。
            weights.append(weight)
        weights = np.array(weights) / max(weights)  #重みの正規化

        experiences = [self.buffer[idx] for idx in sampled_indices]

        return sampled_indices, weights, experiences

#ネットワークの学習を担当
@ray.remote(num_gpus=1)
class Learner:
    def __init__(self,num_states, time_step):
        physical_devices = tf.config.list_physical_devices('GPU')
        #tf.config.set_visible_devices(physical_devices[0], 'GPU')
        if len(physical_devices) > 0:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
        else:
            print("Warning: GPUが見つかりません。CPUでLearnerを実行します。")
        self.init_weight="11_12850.hdf5"
        self.num_states = num_states
        self.time_step = time_step
        self.q_network = QNetwork(self.num_states)
        self.target_q_network = QNetwork(self.num_states)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    #ネットワークの定義と初期化
    def define_network(self):
        env = Environment(self.time_step)
        state = env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))
        #if (self.init_weight is not None):self.q_network.load_weights(self.init_weight)
        self.target_q_network(np.atleast_2d(state))
        self.target_q_network.set_weights(self.q_network.get_weights())
        current_weights = self.q_network.get_weights()
        return current_weights

    #ミニバッチを使用してネットワークを更新し、更新された重みとTD誤差、優先度補正値を返す
    def update_network(self, minibatchs):
        indices_all = []
        td_errors_all = []
        priority_correction_all=[]

        for (indices, weights, transitions) in minibatchs:
            states, actions, rewards, next_states, dones, next_forbidden_actions, gammas, priority_correction = zip(*transitions)
            states = np.vstack(states)
            actions = np.array(actions)
            rewards = np.vstack(rewards)
            next_states = np.vstack(next_states)
            dones = np.vstack(dones)
            next_forbidden_actions=np.vstack(next_forbidden_actions)
            gammas = np.vstack(gammas)
            priority_correction=np.vstack(priority_correction)
            
            #次の状態に対するQ値をネットワークで予測し、禁止行動に対して非常に大きな負の値を加算して、これらの行動が選択されないようにする
            next_qvalues = self.target_q_network(next_states)   #次の状態に対するQ値をネットワークで予測
            next_qvalues=next_qvalues+(next_forbidden_actions*-1.0 * (10**12))  #次の状態での禁止行動に対して非常に大きな負の値を加算して、これらの行動が選択されないようにする(-1兆)
            next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)   #次の状態での最大Q値を持つ行動を選択
            next_actions_onehot = tf.one_hot(next_actions, len(Actions))
            next_maxQ = tf.reduce_sum(next_qvalues * next_actions_onehot, axis=1, keepdims=True)
            TQ = rewards + gammas * (1 - dones) * next_maxQ

            with tf.GradientTape() as tape:
                qvalues = self.q_network(states)
                actions_onehot = tf.one_hot(actions, len(Actions))
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)
                td_errors = tf.square(TQ - Q)   #TD誤差の二乗を計算（損失関数として使用）
                loss = tf.reduce_mean(weights * td_errors)  #重要度サンプリングの重みを掛けたTD誤差の平均を損失関数とすることで、優先度付き経験リプレイのバイアスを補正する

            grads = tape.gradient(loss, self.q_network.trainable_variables) #損失関数に対するネットワークの重みの勾配を計算
            grads, _ = tf.clip_by_global_norm(grads, 10.0)  #勾配のクリッピングを行うことで、勾配爆発を防止する
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))  #ネットワークの重みを更新
            
            #更新された重みとTD誤差、優先度補正値を返すために、各ミニバッチの結果をリストに追加
            indices_all += indices
            td_errors_all += td_errors.numpy().flatten().tolist()
            priority_correction_all+=priority_correction.flatten().tolist()

        current_weights = self.q_network.get_weights()
        self.target_q_network.set_weights(current_weights)
        return current_weights, indices_all, td_errors_all, priority_correction_all


@ray.remote
class Tester:
    def __init__(self, num_states, time_step):
        tf.config.set_visible_devices([], "GPU")
        self.num_states = num_states
        self.time_step = time_step
        self.q_network = QNetwork(self.num_states)
        self.define_network()

    def define_network(self):
        env = Environment(self.time_step)
        state = env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    def test_play(self, current_weights, dir_name, file_name):
        plt.switch_backend('agg')
        os.makedirs(dir_name, exist_ok=True)
        self.q_network.set_weights(current_weights)
        self.q_network.save_weights(dir_name+file_name+".weights.h5")
        episode_rewards = 0
        
        #14種類のテストケースを生成（遅延時間、前方列車の位置オフセット、出発位置のオフセットの組み合わせ）
        test_cases=[{"delay": 0.0, "fowerd_train_position_offset": None, "start_position_offset": 0.0}]
        """
        for ftp in np.linspace(0.2, 1.5, 4):
            test_cases.append({"delay": 0.0, "fowerd_train_position_offset": ftp, "start_position_offset": 0.0})
        for delay in np.linspace(0.0, 10.0, 3):
            for sp in np.linspace(0.2,1.5,3):
                test_cases.append({"delay": delay, "fowerd_train_position_offset": None, "start_position_offset": sp})
        """
        # ftp, delay, sp をすべて float() でキャスト
        for ftp in np.linspace(0.2, 1.5, 4):
            test_cases.append({"delay": 0.0, "fowerd_train_position_offset": float(ftp), "start_position_offset": 0.0})
        for delay in np.linspace(0.0, 10.0, 3):
            for sp in np.linspace(0.2,1.5,3):
                test_cases.append({"delay": float(delay), "fowerd_train_position_offset": None, "start_position_offset": float(sp)})
        
        full_reward=0
        ci=0
        
        #各テストケースに対してエピソードを実行し、報酬を記録してCSVファイルとグラフを保存
        for tc in test_cases:
            env = Environment(self.time_step)
            state = env.reset(11,tc["delay"],1.0,tc["fowerd_train_position_offset"],tc["start_position_offset"])
            speeds = []
            positions = []
            done = False
            plt.figure(dpi=200, figsize=(10, 10))
            plt.xlabel("Position[km]")
            plt.ylabel("Speed[km/h]")
            plt.plot([env.departure_station["position"], env.arrival_station["position"]], [0, 0], "k-", lw=3)
            plt.plot([env.departure_station["position"], env.departure_station["position"]], [0, 100], "k-", lw=3)
            plt.plot([env.arrival_station["position"], env.arrival_station["position"]], [0, 100], "k-", lw=3)
            if (tc["fowerd_train_position_offset"] is not None):
                plt.plot([env.fowerd_train_position, env.fowerd_train_position], [0, 100], "k-", lw=3)
            sec_start = env.position
            front_sections=env.train.front_sections
            for fsi in range(len(front_sections)):
                plt.plot([sec_start, sec_start + front_sections[fsi]["distance"]], [front_sections[fsi]["speed_limit"], front_sections[fsi]["speed_limit"]], "k-", lw=1)
                if (fsi>0):
                    plt.plot([sec_start, sec_start], [front_sections[fsi]["speed_limit"], front_sections[fsi-1]["speed_limit"]], "k-", lw=1)
                sec_start += front_sections[fsi]["distance"]

            f = open(f"{dir_name}{file_name}_{ci}.csv", "w", newline="")
            writer = csv.writer(f)
            
            #エピソードの実行とデータの記録
            while not done:
                speeds.append(env.speed)
                positions.append(env.position)
                t_state = env.raw_state
                n_state=env.normalized_state
                #if (tc["fowerd_train_position_offset"] is not None):
                #    print(f"{env.fowerd_train_position}, {env.position}, {env.forbidden_action}")
                action,qs = self.q_network.sample_action(np.array(state)[np.newaxis,...], 0.0,env.forbidden_action)
                next_state, reward, done = env.step(int(action))
                t_state=[*t_state,*n_state,*qs,reward]
                writer.writerow(t_state)
                episode_rewards += reward
                if ci==0: full_reward=reward
                state = next_state
            writer.writerow([*env.raw_state,*env.normalized_state])
            f.close()
            plt.plot(positions, speeds, "r")
            plt.savefig(f"{dir_name}{file_name}_{ci}.png")
            plt.close("all")
            ci+=1
        #return episode_rewards, file_name, full_reward
        # 報酬を純粋なfloatにキャストして返す
        return float(episode_rewards), file_name, float(full_reward)


def main(num_actors, gamma, num_states, time_step=1.0):
    """
    s = time.time()
    now = datetime.datetime.now()
    dir_name = "data/" + now.strftime("%Y%m%d%H%M%S")
    dir_name += "/"
    os.mkdir(dir_name)
    """
    
    s = time.time()
    now = datetime.datetime.now()
    
    # スクリプトがあるディレクトリの絶対パスを取得（/home/haru1/apex-train/）
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 絶対パスで保存先ディレクトリを作成
    dir_name = os.path.join(base_dir, "data", now.strftime("%Y%m%d%H%M%S")) + "/"
    
    # os.makedirsを使い、dataフォルダが存在しなくても親ごと安全に作成する
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(dir_name + "replay/", exist_ok=True)
    
    #ray.init()
    # ▼▼▼ 修正後：APIキー等の環境変数を子プロセス（Actor）に引き継がせる ▼▼▼
    llm_env = {}
    if os.environ.get("OPENAI_API_KEY"):
        llm_env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    if os.environ.get("OPENAI_BASE_URL"):
        llm_env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL")
        
    if llm_env:
        ray.init(runtime_env={"env_vars": llm_env})
    else:
        ray.init()
    
    
    history = []
    
    #各Actorの探索率を0.001から0.4まで線形に変化させた配列を作成。
    #これにより、異なるActorが異なる程度の探索を行うようになり，学習の初期段階では多くの探索が行われ、後半ではより安定した行動選択が促される。
    epsilons = np.linspace(0.001, 0.4, num_actors,dtype=np.float32)     
    beta=0.4
    #actors = [Actor.remote(pid=i, epsilon=epsilons[i],dir_name=dir_name,  gamma=gamma, num_states=num_states, time_step=time_step) for i in range(num_actors)]
    # float() で包んで純粋なPython型に洗浄して渡す
    actors = [Actor.remote(pid=i, epsilon=float(epsilons[i]), dir_name=dir_name,  gamma=gamma, num_states=num_states, time_step=time_step) for i in range(num_actors)]

    replay = Replay(buffer_size=2**20, save_dir=dir_name+"replay/")

    learner = Learner.remote(num_states=num_states, time_step=time_step)
    current_weights = ray.get(learner.define_network.remote())
    current_weights = ray.put(current_weights)

    tester = Tester.remote(num_states, time_step)

    wip_actors = [actor.rollout.remote(current_weights) for actor in actors]

    #学習を始める前に，各Actorがエピソードを実行して経験を収集し、TD誤差を計算してリプレイバッファに追加する。
    #この初期の経験が学習の開始に必要な多様なデータを提供する。
    for _ in range(30):
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        td_errors, transitions, pid = ray.get(finished[0])
        replay.add(td_errors, transitions)
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])

    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    wip_learner = [learner.update_network.remote(minibatchs)]
    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    wip_tester = tester.test_play.remote(current_weights, dir_name, "0")


    update_cycles = 1
    actor_cycles = 0
    t=time.time()
    
    # 新しく追加されたデータの件数をカウントする変数
    new_transitions_added = 0  
    # Learnerを1回動かすのに必要な最低限の新規データ数（例: 200〜500程度に設定）
    # ※この値を大きくするほど、Learnerの更新頻度が下がり過学習を防げます
    MIN_NEW_TRANSITIONS_PER_UPDATE = 300

    #無限ループで学習継続
    #元に戻すときはapex_gem.pyからコピーしてください。
    # ▼▼▼ 無限ループを完全に書き換え ▼▼▼
    while True:
        actor_cycles += 1
        
        # 1. 完了したActorを「全て」回収する
        while True:  
            finished, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)
            if len(finished) > 0:
                td_errors, transitions, pid = ray.get(finished[0])
                replay.add(td_errors, transitions)
                wip_actors.extend([actors[pid].rollout.remote(current_weights)])
                new_transitions_added += len(transitions)
            else:
                break

        # 2. Learnerが動いていて、完了していれば回収する
        if len(wip_learner) > 0:
            finished_learner, _ = ray.wait(wip_learner, timeout=0)
            if len(finished_learner) > 0:
                current_weights_raw, indices, td_errors, priority_correction = ray.get(finished_learner[0])
                current_weights = ray.put(current_weights_raw)
                replay.update_priority(indices, td_errors, priority_correction)
                wip_learner = [] # Learnerを待機状態（空）にする

        # 3. Learnerが待機状態で、かつ新しいデータが十分に溜まっていれば次をスタート
        if len(wip_learner) == 0 and new_transitions_added >= MIN_NEW_TRANSITIONS_PER_UPDATE:
            minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
            wip_learner = [learner.update_network.remote(minibatchs)]
            
            new_transitions_added = 0
            beta = min(beta + 0.6 / 20000.0, 1.0)
            update_cycles += 1
            print(f"learner {time.time()-t:.2f}s, update_cycles: {update_cycles}, beta: {beta:.4f}")
            t = time.time()

            if update_cycles % 50 == 0:
                test_score, file_name, full_rewoard = ray.get(wip_tester)
                print(file_name, test_score, beta)
                history.append((update_cycles , test_score))
                with open(dir_name + "history.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , test_score))
                with open(dir_name + "history_f.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , full_rewoard))
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                sys.stdout.flush()

        # 空回りによるCPU100%消費を防ぐ微小スリープ
        time.sleep(0.01)


if __name__ == "__main__":
    #main(num_actors=50, gamma=0.9975, num_states=9, time_step=1.0)
    main(num_actors=15, gamma=0.9975, num_states=9, time_step=1.0)
