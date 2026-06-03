#報酬直接予測モデルを使用して、Apex DQNアルゴリズムで列車の制御を学習するためのコード。
import time
import datetime
import os

# ▼【究極のメモリ対策1】Linuxのマルチスレッドによるメモリ抱え込み(アリーナ増殖)を1つに制限して封じる
os.environ['MALLOC_ARENA_MAX'] = '1'
# ▼【究極のメモリ対策2】TensorFlowの不要なCPU最適化(メモリ食い)をオフにする
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import ray
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import psutil  # ← 【追加】メモリ使用量を取得するためのライブラリ
import gc
import ctypes  # ← 【追加】C言語のライブラリを呼び出すための標準モジュール

# Ray の起動待機時間を延長
os.environ['RAY_raylet_start_wait_time_s'] = '120'
# 【今回追加】Rayの過敏なメモリ監視・強制終了を無効化する
#os.environ['RAY_memory_monitor_refresh_ms'] = '0'

from segment_tree import SumTree
from model import QNetwork

from environment2 import Environment
from actions import Actions
import random
import sys

tf.config.set_visible_devices([], "GPU")

#このクラスがRayによって別プロセスとして非同期に実行
@ray.remote
class Actor:
    def __init__(self, pid, epsilon, gamma, num_states, time_step):
        tf.config.set_visible_devices([], "GPU")
        self.pid = pid
        self.time_step = time_step
        self.num_states = num_states
        self.env = Environment(self.time_step)

        self.q_network = QNetwork(self.num_states)
        self.epsilon = epsilon  #探索率
        self.__gamma = gamma    #割引率
        self.buffer = []

        self.define_network()
        
        # ▼【追加】バッチサイズ(None)を許容する静的グラフを作成し、キャッシュ増殖を完全に防ぐ
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.num_states], dtype=tf.float32)])
        def predict_q_batch(x):
            return self.q_network(x, training=False)
        self.predict_q_batch = predict_q_batch
        
        self.episode_rewards = 0
        

    #ネットワークの定義と初期化
    def define_network(self):
        #env = Environment(self.time_step)
        state = self.env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    #エピソードを実行して経験を収集し、TD誤差を計算して返す
    #エピソードを実行して経験を収集し、TD誤差を計算して返す
    def rollout(self, current_weights):
        for var, weight in zip(self.q_network.variables, current_weights):
            var.assign(weight)

        # 様々な状況の先行列車CSVのリスト（事前に作成しておく）
        f_train_csv_list = [
            "input/f_train_low50.csv",
            "input/f_train_delay0_stop30.csv",
            "input/f_train_delay5_stop45.csv"
        ]

        r = random.random()
        # 遅延は常に0〜20秒のランダム値を与える
        random_delay = random.uniform(0.0, 20.0) 
        
        if (r < 0.4):
            # 40%: 単独走行（先行列車なし）
            self.state = self.env.reset(11, random_delay, 1.0)
        elif (r < 0.7):
            # 30%: 固定の距離オフセットのみ（従来の簡易学習）
            self.state = self.env.reset(11, random_delay, 1.0, fowerd_train_position_offset=random.uniform(0.2, 1.5))
        else:
            # 30%: 移動閉塞シミュレーション（外部CSVの先行列車を追従）
            random_csv = random.choice(f_train_csv_list)
            self.state = self.env.reset(11, random_delay, 1.0, fowerd_train_controls=random_csv)
            
        self.episode_rewards = 0
        done = False
        
        while not done:
            state = self.state  
            
            # ▼【大改修1】sample_action()を使わず、Numpyで行動選択して Tensorの流出を完全に防ぐ
            state_tensor = tf.convert_to_tensor(np.array(state)[np.newaxis,...], dtype=tf.float32)
            qs = self.predict_q_batch(state_tensor).numpy()[0]
            forbidden = self.env.forbidden_action
            
            if random.random() < self.epsilon:
                valid_actions = [i for i, f in enumerate(forbidden) if not f]
                action = random.choice(valid_actions)
            else:
                masked_qs = qs.copy()
                masked_qs[forbidden] = -np.inf
                action = int(np.argmax(masked_qs))  # 必ず純粋なPythonのint型にする！
            
            priority_correction=(0.1-(min(max(self.env.station_remaining_distance,0.0),0.1)+0.001))*500+1
            next_state, reward, done = self.env.step(action)
            nest_forbidden_action=self.env.forbidden_action 
            self.episode_rewards += reward  
            
            # ここに入る action が int 型になったため、Ray の通信リークが消滅します
            transition = (state, action, reward, next_state, done, nest_forbidden_action, self.gamma, priority_correction)
            self.buffer.append(transition)
            self.state = next_state

        states = np.vstack([transition[0] for transition in self.buffer])
        actions = np.array([transition[1] for transition in self.buffer])
        rewards = np.vstack([transition[2] for transition in self.buffer])
        next_states = np.vstack([transition[3] for transition in self.buffer])
        dones = np.vstack([transition[4] for transition in self.buffer])
        next_forbidden_actions=np.vstack([transition[5] for transition in self.buffer])
        gammas=np.vstack([transition[6] for transition in self.buffer])
        
        chunk_size = 512
        next_qvalues_list = []
        qvalues_list = []
        for i in range(0, len(states), chunk_size):
            next_chunk = tf.convert_to_tensor(next_states[i:i+chunk_size], dtype=tf.float32)
            curr_chunk = tf.convert_to_tensor(states[i:i+chunk_size], dtype=tf.float32)
            next_qvalues_list.append(self.predict_q_batch(next_chunk))
            qvalues_list.append(self.predict_q_batch(curr_chunk))
            
        # ▼【大改修2】推論結果をNumpy配列に変換し、TD誤差計算を「全てNumpy」で行う（TF Eagerのメモリ断片化を防止）
        next_qvalues = tf.concat(next_qvalues_list, axis=0).numpy()
        qvalues = tf.concat(qvalues_list, axis=0).numpy()
        
        next_qvalues = next_qvalues + (next_forbidden_actions * -1.0 * (10**12))  
        next_actions = np.argmax(next_qvalues, axis=1)   
        next_actions_onehot = np.eye(len(Actions))[next_actions]        
        next_maxQ = np.sum(next_qvalues * next_actions_onehot, axis=1, keepdims=True)
        TQ = rewards + gammas * (1 - dones) * next_maxQ

        actions_onehot = np.eye(len(Actions))[actions]  
        Q = np.sum(qvalues * actions_onehot, axis=1, keepdims=True)  
        
        td_errors = (TQ - Q).flatten()
        transitions = self.buffer
        self.buffer = []
        
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        return td_errors, transitions, self.pid
    
    @property
    def gamma(self):
        return self.__gamma**(self.env.time_step/self.time_step)
    
    def get_memory(self):
        return {
            "pid": self.pid,
            "memory_gb":
                psutil.Process(os.getpid()).memory_info().rss
                / 1024**3,
            "gc_objects": len(gc.get_objects()),
             "buffer_len": len(self.buffer),
             "tensor_count":
            len([
                x for x in gc.get_objects()
                if isinstance(x, tf.Tensor)
            ])
        }

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
            states, actions, rewards, next_states, dones,next_forbidden_actions,gammas,priority_correction = zip(*transitions)
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
        self.env = Environment(self.time_step)
        self.define_network()

    def define_network(self):
        #env = Environment(self.time_step)
        state = self.env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    def test_play(self, current_weights, dir_name, file_name):
        plt.switch_backend('agg')
        self.q_network.set_weights(current_weights)
        #self.q_network.save_weights(dir_name+file_name+".hdf5")
        self.q_network.save_weights(dir_name+file_name+".weights.h5")
        episode_rewards = 0
        
        test_cases = []
        
        # === シミュレーション1: 通常走行 ===
        test_cases.append({"delay": 0.0, "f_train_csv": None, "desc": "Sim1_Normal"})
        
        # === シミュレーション2: 遅延あり（先行列車なし） ===
        for delay in [5.0, 10.0, 15.0]:
            test_cases.append({"delay": delay, "f_train_csv": None, "desc": f"Sim2_Delay_{delay}s"})
            
        # === シミュレーション3: 先行列車あり ===
        # ① 低速モード (50km/h走行)
        test_cases.append({"delay": 0.0, "f_train_csv": "input/f_train_low50.csv", "desc": "Sim3_Low50"})
        
        # ② 先行列車遅延モード + 駅停車時間
        # (※事前に対応する組み合わせのCSVを作成し、inputフォルダに配置しておく必要があります)
        for f_delay in [0, 5, 10]:
            for stop_time in [30, 45, 60]:
                csv_path = f"input/f_train_delay{f_delay}_stop{stop_time}.csv"
                test_cases.append({"delay": 0.0, "f_train_csv": csv_path, "desc": f"Sim3_fDelay{f_delay}_Stop{stop_time}"})
        full_reward=0
        tc0_cumulative_reward = 0  # ←【追加】テストケース0の累積報酬を保持する変数
        ci=0
        
        #各テストケースに対してエピソードを実行し、報酬を記録してCSVファイルとグラフを保存
        env = self.env
        for tc in test_cases:
            # env.resetに fowerd_train_controls としてCSVのパスを渡す
            state = env.reset(11, tc["delay"], 1.0, fowerd_train_controls=tc["f_train_csv"])
            speeds = []
            positions = []
            times = []  # ▼【追加】自列車の時間を記録するリスト
            f_speeds = []
            f_positions = []
            f_times = []  # ▼【追加】先行列車の時間を記録するリスト
            
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
            
            # ▼【追加】データの列が何を表しているか分かりやすいようにヘッダーを書き込む
            header = [
               # raw_state (7次元)
                "raw_speed", "raw_stat_dist", "raw_rem_time", "raw_hold_time", "raw_pre_action", "raw_stat_dist2", "raw_fw_dist",
                # normalized_state (9次元)
                "norm_speed", "norm_stat_dist", "norm_stat_dist_clip", "norm_rem_time", "norm_hold_time", 
                "norm_pre_act_c", "norm_pre_act_a", "norm_pre_act_d", "norm_fw_dist",
                # Q-values (3次元) & 合計Reward, LLM評価値
                "Q_coast", "Q_accel", "Q_decel", "total_reward", "llm_reward"
            ]
            writer.writerow(header)
            
            #エピソードの実行とデータの記録
            while not done:
                speeds.append(env.speed)
                positions.append(env.position)
                times.append(env.t)  # ▼【追加】現在時刻を記録
                
                # 先行列車が存在する場合、その情報も記録する
                if env.fowerd_train is not None:
                    f_positions.append(env.fowerd_train.position)
                    f_speeds.append(env.fowerd_train.speed)
                    f_times.append(env.t)  # ▼【追加】先行列車用の時間も記録
                    
                t_state = env.raw_state
                n_state=env.normalized_state
                
                # ▼ Tester側もNumpy化して Tensorの流出を防ぐ
                state_tensor = tf.convert_to_tensor(np.array(state)[np.newaxis,...], dtype=tf.float32)
                qs = self.q_network(state_tensor, training=False).numpy()[0]
                forbidden = env.forbidden_action
                
                masked_qs = qs.copy()
                masked_qs[forbidden] = -np.inf
                action = int(np.argmax(masked_qs))
                
                next_state, reward, done = env.step(action)
                
                # ▼【追加】environmentからTri-Driveの重みと値を取得
                tri_drive_info = env.latest_rewards_info
                
                # ▼【変更】出力リストの末尾に tri_drive_info を結合して書き込む
                t_state=[*t_state, *n_state, *qs, reward, *tri_drive_info]
                writer.writerow(t_state)
                
                episode_rewards += reward
                if ci==0: 
                    full_reward=reward
                    tc0_cumulative_reward += reward # ←【追加】テストケース0なら毎ステップの報酬を足し合わせる
                state = next_state
            
            # 終了時の状態書き込み（必要であればここにも追加できますが、通常は最後のステップだけでOKです）
            # writer.writerow([*env.raw_state,*env.normalized_state])
            f.close()
            # ==========================================
            # ① 従来のランカーブ（横: 位置, 縦: 速度）の保存
            # ==========================================
            plt.plot(positions, speeds, "r-", label="Own Train")
            if len(f_positions) > 0:
                plt.plot(f_positions, f_speeds, "b--", label="Forward Train")
            plt.legend(loc="upper right")
            plt.savefig(f"{dir_name}{file_name}_{ci}.png")
            
            fig = plt.gcf()
            fig.clf()
            plt.close(fig)

            # ==========================================
            # ② 新規追加：ダイヤグラム（横: 時間, 縦: 位置）の保存
            # ==========================================
            plt.figure(dpi=200, figsize=(10, 6))
            plt.xlabel("Time [s]")
            plt.ylabel("Position [km]")
            
            # 駅の位置に水平線を引く
            plt.axhline(y=env.departure_station["position"], color='k', linestyle='-', lw=2)
            plt.axhline(y=env.arrival_station["position"], color='k', linestyle='-', lw=2)
            
            # 自列車の軌跡（赤）
            plt.plot(times, positions, "r-", label="Own Train")
            
            # 先行列車の軌跡（青点線）
            if len(f_positions) > 0:
                plt.plot(f_times, f_positions, "b--", label="Forward Train")
            
            # 縦軸（距離）の表示範囲をご要望通りに設定（出発駅の少し手前から、到着駅+100m程度まで）
            plt.ylim(env.departure_station["position"] - 0.05, env.arrival_station["position"] + 0.1)
            
            plt.legend(loc="lower right")
            plt.grid(True) # 時間と距離が分かりやすいようにグリッド線を追加
            
            # ファイル名の末尾に _diagram をつけて保存
            plt.savefig(f"{dir_name}{file_name}_{ci}_diagram.png")
            
            fig2 = plt.gcf()
            fig2.clf()
            plt.close(fig2)
            
            ci+=1
        return episode_rewards, file_name, full_reward, tc0_cumulative_reward


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
    actors = [Actor.remote(pid=i, epsilon=epsilons[i], gamma=gamma, num_states=num_states, time_step=time_step) for i in range(num_actors)]

    #replay = Replay(buffer_size=2**20, save_dir=dir_name+"replay/")
    replay = Replay(buffer_size=2**17, save_dir=dir_name+"replay/")

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
        # ▼ このような行を追加して、ターミナルに進捗を表示させる
        #print(f"現在データ収集（ウォームアップ）中... バッファ数: {replay.tree.n_entries}")

    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    wip_learner = learner.update_network.remote(minibatchs)
    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    wip_tester = tester.test_play.remote(current_weights, dir_name, "0")


    update_cycles = 1
    actor_cycles = 0
    t=time.time()

    #無限ループで学習継続
    while True:
        actor_cycles += 1
        
        #Actorのロールアウトが完了するのを待ち、完了したActorからTD誤差と経験を取得してリプレイバッファに追加し、次のロールアウトを開始する。
        while True:  
            finished, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)
            if (len(finished)>0):
                td_errors, transitions, pid= ray.get(finished[0])
                replay.add(td_errors, transitions)
                wip_actors.extend([actors[pid].rollout.remote(current_weights)])
            else: break

        
        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        
        #Learnerのネットワーク更新が完了するのを待ち、完了したら更新された重みとTD誤差、優先度補正値を取得してリプレイバッファの優先度を更新し、
        # 新しいミニバッチをサンプリングして次のネットワーク更新を開始する。
        if finished_learner:
            current_weights, indices, td_errors, priority_correction = ray.get(finished_learner[0])
            wip_learner = learner.update_network.remote(minibatchs)
            current_weights = ray.put(current_weights)
            #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
            replay.update_priority(indices, td_errors, priority_correction)
            minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
            beta=min(beta+0.6/20000.0,1.0)
            update_cycles += 1
            actor_cycles = 0
            # ▼【追加】現在のバッファサイズを計算（一杯なら最大サイズ、そうでなければ現在のカウント数）
            current_buffer_size = replay.buffer_size if replay.is_full else replay.count
            
            # ▼【変更】ログに buffer_size を追加（ついでに見やすいように小数点以下も丸めています）
            print(f"learner {time.time()-t:.2f}s, update_cycles: {update_cycles}, beta: {beta:.5f}, buffer_size: {current_buffer_size} / {replay.buffer_size}")
            t=time.time()

            if update_cycles % 50 == 0:
                # ▼【追加】50サイクルごとに強制的にメモリのゴミ箱を空にする
                gc.collect()
                # システム全体のメモリ使用量をチェックして見やすく表示
                mem = psutil.virtual_memory()
                used_gb = (mem.total - mem.available) / (1024**3)
                total_gb = mem.total / (1024**3)
                print(f"==== [System Memory] 使用率: {mem.percent}% ({used_gb:.2f} GB / {total_gb:.2f} GB) ====")
                process = psutil.Process(os.getpid())

                print(
                    "[Driver]",
                    process.memory_info().rss / 1024**3
                )
                for actor in actors:
                    print(ray.get(actor.get_memory.remote()))

                # ▼【変更】戻り値を4つ受け取るように修正（tc0_reward を追加）
                test_score, file_name, full_rewoard, tc0_reward = ray.get(wip_tester)
                print(file_name, test_score, beta)
                history.append((update_cycles , test_score))
                with open(dir_name + "history.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , test_score))
                with open(dir_name + "history_f.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , full_rewoard))
                # ▼【追加】history_0.csv にテストケース0の累積報酬を保存する
                with open(dir_name + "history_0.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , tc0_reward))
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                sys.stdout.flush()


if __name__ == "__main__":
    #main(num_actors=50, gamma=0.9975, num_states=9, time_step=1.0)
    main(num_actors=5, gamma=0.9975, num_states=9, time_step=1.0)
