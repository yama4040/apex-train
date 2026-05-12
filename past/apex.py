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

from openai import OpenAI

from segment_tree import SumTree
from model import QNetwork

from environment import Environment
from actions import Actions
import random
import sys
import json

tf.config.set_visible_devices([], "GPU")


#このクラスがRayによって別プロセスとして非同期に実行
@ray.remote
class Actor:
    # ▼▼▼ 【修正】引数に num_actors と max_important_steps を追加 ▼▼▼
    def __init__(self, pid, epsilon, gamma, num_states, time_step, log_dir, num_actors, max_important_steps):
        tf.config.set_visible_devices([], "GPU")
        self.pid = pid
        self.time_step = time_step
        self.num_states = num_states
        self.env = Environment(self.time_step)
        self.log_dir = log_dir
        self.num_actors = num_actors                 # Actor総数
        self.max_important_steps = max_important_steps # ガンマ最大修正数
    # ▲▲▲ ここまで ▲▲▲

        self.q_network = QNetwork(self.num_states)
        self.epsilon = epsilon  #探索率
        self.__gamma = gamma    #割引率
        self.buffer = []
        
       

        self.define_network()
        self.episode_rewards = 0
        self.episode_count = 0 # ログファイル名用

    #ネットワークの定義と初期化
    def define_network(self):
        env = Environment(self.time_step)
        state = env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))
    
    # ▼▼▼ LLM評価メソッド（エラー可視化＆CSV入力対応＆モード切り替え対応） ▼▼▼
    def evaluate_gammas_with_local_llm(self, log_csv, base_gamma, total_steps):
        """ローカルLLMにCSV形式のログを評価させ、結果を返す"""
        try:
            if not hasattr(self, "llm_client"):
                api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
                base_url = os.environ.get("OPENAI_BASE_URL")
                
                import httpx
                # タイムアウトの設定
                timeout_config = httpx.Timeout(300.0, connect=10.0)
                client_kwargs = {"api_key": api_key, "timeout": timeout_config}
                
                if base_url:
                    client_kwargs["base_url"] = base_url
                self.llm_client = OpenAI(**client_kwargs)
                
            """====================
            ☆ステップ抽出両設定☆
            ===================="""    
            #max_important_steps = 50  # 重要ステップ抽出モードの最大ステップ数（0にすると全ステップ評価モード）

            # --- モードに応じた指示の切り替え ---
            if self.max_important_steps > 0:
                # 【モードA】重要ステップ抽出モード
                mode_instruction = f"""
                全ステップを出力する必要はありません。
                あなたが「このステップの行動は重要だ」と判断したステップ必ず{self.max_important_steps}個を抽出し、
                0.0000 ～ 1.0000 の範囲でガンマ値を付与してください。
                あなたが決定した{self.max_important_steps}個以外のステップには、すべてベースガンマ（{base_gamma}）を適用します。
                それを考慮したうえで各ステップのガンマ値を決定してください。
                出力形式:
                {{
                    "important_steps": [
                        {{"step": <int>, "gamma": <float>,"}}
                    ]
                }}
                """
            else:
                # 【モードB】全ステップ評価モード (max_important_steps = 0)
                mode_instruction = f"""
                提供された全ステップ（計 {total_steps} 個）に対して、それぞれ 0.0000 ～ 1.0000 の範囲でガンマ値を決定してください。
                配列の長さは必ず {total_steps} にしてください。
                出力形式:
                {{
                    "gammas": [<float>, <float>, ...]
                }}
                """

            system_prompt = f"""
            あなたは鉄道運転士でありDQNのメンターです。
            提供された【走行ログ(CSV形式: 残り距離(km), 先行列車までの残り距離(km), 速度(km/h), 行動, 報酬)】を分析し、
            省エネや定時到着に貢献した行動を特定し、TQ値の計算に使用する割引率（ガンマ）を決定してください。
            省エネや定時到着、停止位置精度に貢献した行動を特定し、TQ値の計算に使用する直接的な重み（ガンマ）を決定してください。
            TQ値は「TQ_t = 報酬_t + Σ (ガンマ_k * 報酬_k)」というように、未来の報酬にあなたの決定したガンマを掛けたものの総和として計算されます。
            最終的な成功に大きく寄与した重要な行動のステップには1.0000に近い重みを、無関係な行動には0.0000に近い重みを付与してください。
            {mode_instruction}
            余計な挨拶や説明文は一切含めず、JSONデータのみを出力してください。
            """

            print(f"[Actor {self.pid}] ---- LLMにリクエストを送信します ----")
            print(f"[Actor {self.pid}] CSVデータ量: {len(log_csv)} 文字, Step数: {total_steps}")
            sys.stdout.flush() 

            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": log_csv}
                ],
                response_format={ "type": "json_object" }, 
                temperature=0.1,
                # ▼▼▼ 【追加】出力の最大トークン数を制限（物理ブレーキ） ▼▼▼
                #max_tokens=len(log_csv)+1000
                # ▲▲▲ ここまで ▲▲▲
            )
            
            elapsed = time.time() - start_time
            print(f"[Actor {self.pid}] ---- LLMから応答がありました (所要時間: {elapsed:.2f}秒) ----")
            sys.stdout.flush()

            content = response.choices[0].message.content
            
            # =========================================================
            # プロンプトと応答内容をテキストファイルに出力 (ご要望の処理)
            # =========================================================
            try:
                # ▼▼▼ 【修正】main関数から受け取った self.log_dir を使用する ▼▼▼
                log_file_path = os.path.join(self.log_dir, f"llm_prompt_result_actor_{self.pid}.txt")
                with open(log_file_path, "a", encoding="utf-8") as f:
                    # ▼▼▼ 【追加】エピソード1回目（初回）の時だけ、ファイルの冒頭に設定を書き込む ▼▼▼
                    if self.episode_count == 1:
                        f.write("***************** EXPERIMENT SETTINGS *****************\n")
                        f.write(f"Total Actors     : {self.num_actors}\n")
                        f.write(f"Max Gamma Edits  : {self.max_important_steps}\n")
                        f.write(f"Base Gamma Value : {base_gamma}\n")
                        f.write("*******************************************************\n\n")
                    # ▲▲▲ ここまで ▲▲▲
                    f.write(f"==================== Episode {self.episode_count} ====================\n")
                # ▲▲▲ ここまで ▲▲▲
                    f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    # ▼▼▼ 【追加】実行時間（所要時間）を記載 ▼▼▼
                    f.write(f"Elapsed Time: {elapsed:.2f} seconds\n\n")
                    # ▲▲▲ ここまで ▲▲▲
                    f.write("---------- System Prompt ----------\n")
                    f.write(f"{system_prompt.strip()}\n\n")
                    f.write("---------- User Prompt (CSV Log) ----------\n")
                    f.write(f"{log_csv.strip()}\n\n")
                    f.write("---------- LLM Response ----------\n")
                    f.write(f"{content.strip()}\n")
                    f.write("========================================================\n\n")
            except Exception as e:
                print(f"[Actor {self.pid}] プロンプト/応答ログの保存エラー: {e}")
                sys.stdout.flush()
            # =========================================================

            result_json = json.loads(content)
            
            # ▼▼▼ 【修正箇所】モードに応じた結果の受け取り方 ▼▼▼
            # 初期値は全て base_gamma (0.9975など) で埋めておく
            final_gammas = [base_gamma] * total_steps

            if self.max_important_steps > 0:
                # 【モードA】の解釈: "important_steps" キーから取得し、指定ステップだけガンマを上書きする
                important_steps = result_json.get("important_steps", [])
                for item in important_steps:
                    s_idx = item.get("step")
                    g_val = item.get("gamma")
                    # ステップ番号が正常な範囲内かチェック
                    if isinstance(s_idx, int) and 0 <= s_idx < total_steps and isinstance(g_val, (int, float)):
                        final_gammas[s_idx] = float(g_val)
                        #print(f"[Actor {self.pid}] 💡 Step {s_idx} のガンマを {g_val} に強化！")
            else:
                # 【モードB】の解釈: 従来の "gammas" 配列として受け取る
                llm_array = result_json.get("gammas", [])
                if len(llm_array) == total_steps:
                    final_gammas = [float(g) for g in llm_array]
                    print(f"[Actor {self.pid}] 成功！正しい長さのガンマ配列を取得しました。")
                else:
                    print(f"[Actor {self.pid}] ⚠️ 警告: 配列長不一致 (期待値:{total_steps}, 実際:{len(llm_array)})")
            
            sys.stdout.flush()
            return final_gammas
            # ▲▲▲ 修正はここまで ▲▲▲

        except Exception as e:
            # タイムアウトやパースエラーの理由を出力する
            print(f"[Actor {self.pid}] ❌ LLM API エラー発生: {e}")
            sys.stdout.flush()
            return [base_gamma] * total_steps
    # ▲▲▲ ここまで ▲▲▲

    #エピソードを実行して経験を収集し、TD誤差を計算して返す
    def rollout(self, current_weights):
        self.q_network.set_weights(current_weights)
        r=random.random()   
        if (r<0.5):
            self.state = self.env.reset(11,0.0,1.0)
        elif (r<0.75):
            self.state = self.env.reset(11,0.0,1.0,random.uniform(0.2,1.5))
        else:
            self.state = self.env.reset(11,random.uniform(0.0,20),1.0,None,random.uniform(0.2,1.5))
        self.episode_rewards = 0
        done = False
        
        # ▼ CSVヘッダーの修正: 先行列車距離と報酬を追加
        episode_logs = ["残り距離(km),先行列車までの残り距離(km),速度(km/h),行動,報酬"]
        
        step_count = 0
        action_map = {0: "惰行", 1: "加速", 2: "減速"}
        
        # エピソードの実行と経験の収集
        while not done:
            state = self.state
            action, _ = self.q_network.sample_action(np.array(state)[np.newaxis,...], self.epsilon, self.env.forbidden_action)
            
            current_speed = self.env.speed
            remain_dist = self.env.station_remaining_distance 
            
            # ▼ 先行列車までの距離を取得
            if self.env.fowerd_train_position is not None:
                forward_dist = f"{(self.env.fowerd_train_position - self.env.position):.3f}"
            else:
                forward_dist = "なし"
            
            priority_correction=(0.1-(min(max(self.env.station_remaining_distance,0.0),0.1)+0.001))*500+1
            next_state, reward, done = self.env.step(action)
            nest_forbidden_action=self.env.forbidden_action
            self.episode_rewards += reward
            
            # ▼ ログに報酬を含める
            action_ja = action_map.get(int(action), str(int(action)))
            csv_line = f"{remain_dist:.3f},{forward_dist},{current_speed:.1f},{action_ja},{reward:.3f}"
            episode_logs.append(csv_line)
            step_count += 1
            
            # 一旦ベースガンマでタプルとして保存（※報酬はそのまま保存）
            transition = (state, action, reward, next_state, done, nest_forbidden_action, self.gamma, priority_correction)
            self.buffer.append(transition)
            self.state = next_state
        
        # =========================================================
        # LLMによる動的ガンマの取得とバッファの上書き (TQの直接計算)
        # =========================================================
        self.episode_count += 1
        log_csv_for_llm = "\n".join(episode_logs)
        total_steps = len(self.buffer)
        
        llm_gammas = self.evaluate_gammas_with_local_llm(log_csv_for_llm, self.__gamma, total_steps)
        
        # ▼ TQ値の逆順計算アルゴリズム
        future_sum = 0.0
        TQ_array = np.zeros(total_steps)
        for i in reversed(range(total_steps)):
            r_i = self.buffer[i][2]  # 現在のステップの報酬
            g_i = llm_gammas[i]      # LLMが決定したそのステップへの重み
            
            # TQ_t = r_t + Σ(γ_k * r_k)
            TQ_array[i] = r_i + future_sum
            
            # 次のステップ(手前のステップ)のために現在の重み付き報酬を加算
            future_sum += g_i * r_i

        # バッファ内のタプルを上書き（報酬をTQ値に置き換える）
        for i in range(total_steps):
            temp_transition = list(self.buffer[i])
            temp_transition[2] = float(TQ_array[i]) # 報酬だった部分をTQ値に上書き
            temp_transition[6] = float(llm_gammas[i])
            self.buffer[i] = tuple(temp_transition)

        states = np.vstack([transition[0] for transition in self.buffer])
        actions = np.array([transition[1] for transition in self.buffer])
        TQs = np.vstack([transition[2] for transition in self.buffer]) # 保存したTQを取り出す
        
        # 現在の状態に対するQ値をネットワークで予測
        qvalues = self.q_network(states)    
        actions_onehot = tf.one_hot(actions, len(Actions))  
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)  
        
        # TD誤差の計算（ベルマン方程式のTarget推論を省略し、直接算出済みのTQ値との差分をとる）
        td_errors = (TQs - Q).numpy().flatten()
        
        transitions = self.buffer
        self.buffer = []

        return td_errors, transitions, self.pid
    
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
            self.priorities[self.count] = priority * transition[-1] #優先度に補正値を掛けることで、駅近の経験の優先度をさらに高くする
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
            #self.priorities[int(idx)] = float(priority*priority_correction)


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
            # ▼ 報酬(rewards)として格納されていた場所に、Actorで算出済みのTQ値が入っている
            states, actions, TQs, next_states, dones, next_forbidden_actions, gammas, priority_correction = zip(*transitions)
            
            states = np.vstack(states)
            actions = np.array(actions)
            TQs = np.vstack(TQs).astype(np.float32) # LLMによる重みで確定した正解ラベル
            
            # weightsをテンソル計算で使うため、必要に応じて型変換（安全策）
            weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
            priority_correction = np.vstack(priority_correction)
            
            # 【削除】従来のベルマン更新（target_q_networkを使った次状態のmaxQ推論と、報酬+割引の計算）は
            # Actor側ですでにTQ値として算出済みのため、ここでは一切行いません。

            with tf.GradientTape() as tape:
                # 現在のネットワークで状態sに対するQ値を予測
                qvalues = self.q_network(states)
                actions_onehot = tf.one_hot(actions, len(Actions))
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)
                
                # ▼ 誤差計算の変更: target_q_networkの推論値ではなく、Actorが計算した確定TQ値をそのまま使う
                td_errors = tf.square(TQs - Q)   # TD誤差の二乗を計算（損失関数として使用）
                loss = tf.reduce_mean(weights * td_errors)  # 重要度サンプリングの重みを掛けたTD誤差の平均

            grads = tape.gradient(loss, self.q_network.trainable_variables) #損失関数に対するネットワークの重みの勾配を計算
            grads, _ = tf.clip_by_global_norm(grads, 10.0)  #勾配のクリッピングを行うことで、勾配爆発を防止する
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))  #ネットワークの重みを更新
            
            #更新された重みとTD誤差、優先度補正値を返すために、各ミニバッチの結果をリストに追加
            indices_all += indices
            td_errors_all += td_errors.numpy().flatten().tolist()
            priority_correction_all+=priority_correction.flatten().tolist()

        current_weights = self.q_network.get_weights()
        
        # 今回の手法では推論にTarget Networkは使いませんが、コード全体の互換性のため重み同期は残しておきます
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
        self.q_network.set_weights(current_weights)
        #self.q_network.save_weights(dir_name+file_name+".hdf5")
        self.q_network.save_weights(dir_name+file_name+".weights.h5")
        episode_rewards = 0
        
        #14種類のテストケースを生成（遅延時間、前方列車の位置オフセット、出発位置のオフセットの組み合わせ）
        test_cases=[{"delay": 0.0, "fowerd_train_position_offset": None, "start_position_offset": 0.0}]
        for ftp in np.linspace(0.2, 1.5, 4):
            test_cases.append({"delay": 0.0, "fowerd_train_position_offset": ftp, "start_position_offset": 0.0})
        for delay in np.linspace(0.0, 10.0, 3):
            for sp in np.linspace(0.2,1.5,3):
                test_cases.append({"delay": delay, "fowerd_train_position_offset": None, "start_position_offset": sp})
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
                next_state, reward, done = env.step(action)
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
        return episode_rewards, file_name, full_reward


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
    
    ray.init()
    history = []
    
    #各Actorの探索率を0.001から0.4まで線形に変化させた配列を作成。
    #これにより、異なるActorが異なる程度の探索を行うようになり，学習の初期段階では多くの探索が行われ、後半ではより安定した行動選択が促される。
    epsilons = np.linspace(0.001, 0.4, num_actors,dtype=np.float32)     
    beta=0.4
   # ▼▼▼ 【修正】ガンマの最大修正数をここで一括管理し、Actorに渡す ▼▼▼
    """""""""""""""""""""""""""
    ☆最大修正ガンマ数指定☆
    """""""""""""""""""""""""""
    MAX_IMPORTANT_STEPS = 50
    
    actors = [Actor.remote(pid=i, epsilon=epsilons[i], gamma=gamma, num_states=num_states, 
                           time_step=time_step, log_dir=dir_name, 
                           num_actors=num_actors, max_important_steps=MAX_IMPORTANT_STEPS) for i in range(num_actors)]
    # ▲▲▲ ここまで ▲▲▲
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
    wip_learner = learner.update_network.remote(minibatchs)
    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    wip_tester = tester.test_play.remote(current_weights, dir_name, "0")


    update_cycles = 1
    actor_cycles = 0
    t = time.time()
    
    # ▼▼▼ 【追加】新しく追加された経験（ステップ）の数をカウントする変数 ▼▼▼
    new_transitions_added = 0
    # ▲▲▲

    #無限ループで学習継続
    while True:
        actor_cycles += 1
        
        # Actorのロールアウト完了を待ち、完了したActorから経験を取得
        while True:  
            finished, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)
            if (len(finished) > 0):
                td_errors, transitions, pid = ray.get(finished[0])
                replay.add(td_errors, transitions)
                
                # ▼▼▼ 【追加】追加された経験のステップ数をカウント ▼▼▼
                new_transitions_added += len(transitions)
                # ▲▲▲
                
                wip_actors.extend([actors[pid].rollout.remote(current_weights)])
            else: 
                break

        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        
        # ▼▼▼ 【重要修正】Learnerの更新条件に「新規データが十分に集まっているか」を追加 ▼▼▼
        # 例：新しいステップが 500 個以上溜まるまでは、Learnerは次の更新を行わない
        update_interval = 500
        if finished_learner and new_transitions_added >= update_interval:
            current_weights, indices, td_errors, priority_correction = ray.get(finished_learner[0])
            
            # 共有メモリ参照を切り離し
            indices = np.array(indices).copy()
            td_errors = np.array(td_errors).copy()
            
            # ▼▼▼ 【追加】学習を開始するので、カウントを消費する ▼▼▼
            new_transitions_added -= update_interval
            # ▲▲▲
            
            wip_learner = learner.update_network.remote(minibatchs)
            current_weights = ray.put(current_weights)
            
            replay.update_priority(indices, td_errors, priority_correction)
            minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
            beta = min(beta + 0.6 / 20000.0, 1.0)
            update_cycles += 1
            actor_cycles = 0
            
            print(f"learner {time.time()-t:.2f}s, update_cycles: {update_cycles}, beta: {beta:.4f}, wait_buffer: {new_transitions_added}")
            t = time.time()

            if update_cycles % 50 == 0:
                test_score, file_name, full_rewoard = ray.get(wip_tester)
                print(f"Test Result: {file_name}, Score: {test_score}, Beta: {beta:.4f}")
                history.append((update_cycles , test_score))
                with open(dir_name + "history.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , test_score))
                with open(dir_name + "history_f.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , full_rewoard))
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                sys.stdout.flush()


if __name__ == "__main__":
    #main(num_actors=50, gamma=0.9975, num_states=9, time_step=1.0)
    main(num_actors=2, gamma=0.5000, num_states=9, time_step=1.0)
