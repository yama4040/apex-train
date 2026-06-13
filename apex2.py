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
import psutil
import gc
from collections import deque
import ctypes

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

# ▼【追加】TensorFlowのメモリ動的確保（必要な分だけ確保し、がめつく全取りしない）
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def free_ray_refs(*refs):
    flat_refs = []
    for ref in refs:
        if ref is None:
            continue
        if isinstance(ref, (list, tuple, set)):
            flat_refs.extend(x for x in ref if x is not None)
        else:
            flat_refs.append(ref)
    if not flat_refs:
        return
    try:
        ray._private.internal_api.free(flat_refs, local_only=False)
    except Exception:
        try:
            ray.internal.free(flat_refs)
        except Exception:
            pass

@ray.remote
class Actor:
    def __init__(self, pid, epsilon, gamma, num_states, time_step):
        tf.config.set_visible_devices([], "GPU")
        self.pid = pid
        self.time_step = time_step
        self.num_states = num_states
        self.env = Environment(self.time_step)

        self.q_network = QNetwork(self.num_states)
        self.epsilon = epsilon
        self.__gamma = gamma
        self.buffer = []

        self.define_network()
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.num_states], dtype=tf.float32)])
        def predict_q_batch(x):
            return self.q_network(x, training=False)
        self.predict_q_batch = predict_q_batch
        
        self.episode_rewards = 0

    def define_network(self):
        state = self.env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    def rollout(self, current_weights):
        for var, weight in zip(self.q_network.variables, current_weights):
            var.assign(weight)

        f_train_csv_list = [
            "input/f_train_low50.csv",
            "input/f_train_delay0_stop30.csv",
            "input/f_train_delay5_stop45.csv"
        ]

        r = random.random()
        random_delay = random.uniform(0.0, 20.0) 
        
        if (r < 0.4):
            self.state = self.env.reset(11, random_delay, 1.0)
        elif (r < 0.7):
            self.state = self.env.reset(11, random_delay, 1.0, fowerd_train_position_offset=random.uniform(0.2, 1.5))
        else:
            random_csv = random.choice(f_train_csv_list)
            self.state = self.env.reset(11, random_delay, 1.0, fowerd_train_controls=random_csv)
            
        self.episode_rewards = 0
        done = False
        
        while not done:
            state = self.state  
            
            state_tensor = tf.convert_to_tensor(np.array(state)[np.newaxis,...], dtype=tf.float32)
            qs = self.predict_q_batch(state_tensor).numpy()[0]
            del state_tensor
            forbidden = self.env.forbidden_action
            
            if random.random() < self.epsilon:
                valid_actions = [i for i, f in enumerate(forbidden) if not f]
                action = random.choice(valid_actions)
            else:
                masked_qs = qs.copy()
                masked_qs[forbidden] = -np.inf
                action = int(np.argmax(masked_qs)) 
            
            priority_correction=(0.1-(min(max(self.env.station_remaining_distance,0.0),0.1)+0.001))*500+1
            next_state, reward, done = self.env.step(action)
            nest_forbidden_action=self.env.forbidden_action 
            self.episode_rewards += reward  
            
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
            next_qvalues_list.append(self.predict_q_batch(next_chunk).numpy())
            qvalues_list.append(self.predict_q_batch(curr_chunk).numpy())
            del next_chunk, curr_chunk
            
        next_qvalues = np.concatenate(next_qvalues_list, axis=0)
        qvalues = np.concatenate(qvalues_list, axis=0)
        
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
        del states, actions, rewards, next_states, dones, next_forbidden_actions, gammas
        del next_qvalues_list, qvalues_list, next_qvalues, qvalues
        del next_actions, next_actions_onehot, next_maxQ, TQ, actions_onehot, Q
        
        try:
            gc.collect()
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
            "memory_gb": psutil.Process(os.getpid()).memory_info().rss / 1024**3,
            "gc_objects": len(gc.get_objects()),
            "buffer_len": len(self.buffer),
            "tensor_count": len([x for x in gc.get_objects() if isinstance(x, tf.Tensor)])
        }

class Replay:
    def __init__(self, buffer_size, save_dir):
        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)
        self.buffer = [None] * self.buffer_size
        self.alpha = 0.6
        self.count = 0
        self.is_full = False
        
    def add(self, td_errors, transitions):
        assert len(td_errors) == len(transitions)
        priorities = (np.abs(td_errors) + 0.001) ** self.alpha
        for priority, transition in zip(priorities, transitions):
            self.priorities[self.count] = priority * transition[-1]
            self.buffer[self.count] = transition
            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def update_priority(self, sampled_indices, td_errors, priority_corrections):
        assert len(sampled_indices) == len(td_errors)
        for idx, td_error, priority_correction in zip(sampled_indices, td_errors, priority_corrections):
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.priorities[idx] = priority*priority_correction

    def sample_minibatch(self, batch_size, beta):
        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]
        weights = []
        current_size = len(self.buffer) if self.is_full else self.count
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * current_size) ** (-beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)
        experiences = [self.buffer[idx] for idx in sampled_indices]
        return sampled_indices, weights, experiences

@ray.remote(num_gpus=1)
class Learner:
    def __init__(self,num_states, time_step):
        physical_devices = tf.config.list_physical_devices('GPU')
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

    def define_network(self):
        env = Environment(self.time_step, load_reward_predictor=False)
        state = env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))
        self.target_q_network(np.atleast_2d(state))
        self.target_q_network.set_weights(self.q_network.get_weights())
        current_weights = self.q_network.get_weights()
        return current_weights

    def set_weights(self, weights):
        self.q_network.set_weights(weights)
        self.target_q_network.set_weights(weights)
        return True

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
            
            next_qvalues = self.target_q_network(next_states)
            next_qvalues=next_qvalues+(next_forbidden_actions*-1.0 * (10**12))
            next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)
            next_actions_onehot = tf.one_hot(next_actions, len(Actions))
            next_maxQ = tf.reduce_sum(next_qvalues * next_actions_onehot, axis=1, keepdims=True)
            TQ = rewards + gammas * (1 - dones) * next_maxQ

            weights_t = tf.convert_to_tensor(np.asarray(weights, dtype=np.float32).reshape(-1, 1))

            with tf.GradientTape() as tape:
                qvalues = self.q_network(states)
                actions_onehot = tf.one_hot(actions, len(Actions))
                Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)
                
                td_delta = TQ - Q
                td_errors = tf.square(td_delta) 
                loss = tf.reduce_mean(weights_t * td_errors)  

            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0) 
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
            
            indices_all += indices
            td_errors_all += td_delta.numpy().flatten().tolist()
            priority_correction_all += priority_correction.flatten().tolist()
            del states, actions, rewards, next_states, dones, next_forbidden_actions, gammas
            del priority_correction, next_qvalues, next_actions, next_actions_onehot, next_maxQ, TQ
            del qvalues, actions_onehot, Q, td_errors, loss, grads
            del indices, weights, transitions

        current_weights = self.q_network.get_weights()
        self.target_q_network.set_weights(current_weights)
        
        try:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
            
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
        state = self.env.reset(11,0.0)
        self.q_network(np.atleast_2d(state))

    def test_play(self, current_weights, dir_name, file_name):
        plt.switch_backend('agg')
        self.q_network.set_weights(current_weights)
        self.q_network.save_weights(dir_name+file_name+".weights.h5")
        episode_rewards = 0
        
        test_cases = []
        
        test_cases.append({"delay": 0.0, "f_train_csv": None, "desc": "Sim1_Normal"})
        for delay in [5.0, 10.0, 15.0]:
            test_cases.append({"delay": delay, "f_train_csv": None, "desc": f"Sim2_Delay_{delay}s"})
        test_cases.append({"delay": 0.0, "f_train_csv": "input/f_train_low50.csv", "desc": "Sim3_Low50"})
        
        for f_delay in [0, 5, 10]:
            for stop_time in [30, 45, 60]:
                csv_path = f"input/f_train_delay{f_delay}_stop{stop_time}.csv"
                test_cases.append({"delay": 0.0, "f_train_csv": csv_path, "desc": f"Sim3_fDelay{f_delay}_Stop{stop_time}"})
        full_reward=0
        tc0_cumulative_reward = 0 
        ci=0
        
        env = self.env
        for tc in test_cases:
            state = env.reset(11, tc["delay"], 1.0, fowerd_train_controls=tc["f_train_csv"])
            speeds = []
            positions = []
            times = []
            f_speeds = []
            f_positions = []
            f_times = []
            
            done = False
            plt.figure(dpi=200, figsize=(10, 10))
            plt.xlabel("Position[km]")
            plt.ylabel("Speed[km/h]")
            plt.plot([env.departure_station["position"], env.arrival_station["position"]], [0, 0], "k-", lw=3)
            plt.plot([env.departure_station["position"], env.departure_station["position"]], [0, 100], "k-", lw=3)
            plt.plot([env.arrival_station["position"], env.arrival_station["position"]], [0, 100], "k-", lw=3)
            if env.fowerd_train_position is not None:
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
            
            header = [
                "raw_speed", "raw_stat_dist", "raw_rem_time", "raw_hold_time", "raw_pre_action", "raw_stat_dist2", "raw_fw_dist",
                "norm_speed", "norm_stat_dist", "norm_stat_dist_clip", "norm_rem_time", "norm_hold_time", 
                "norm_pre_act_c", "norm_pre_act_a", "norm_pre_act_d", "norm_fw_dist",
                "Q_coast", "Q_accel", "Q_decel", "total_reward", "llm_reward"
            ]
            writer.writerow(header)
            
            while not done:
                speeds.append(env.speed)
                positions.append(env.position)
                times.append(env.t)
                
                if env.fowerd_train is not None:
                    f_positions.append(env.fowerd_train.position)
                    f_speeds.append(env.fowerd_train.speed)
                    f_times.append(env.t)
                    
                t_state = env.raw_state
                n_state=env.normalized_state
                
                state_tensor = tf.convert_to_tensor(np.array(state)[np.newaxis,...], dtype=tf.float32)
                qs = self.q_network(state_tensor, training=False).numpy()[0]
                forbidden = env.forbidden_action
                
                masked_qs = qs.copy()
                masked_qs[forbidden] = -np.inf
                action = int(np.argmax(masked_qs))
                
                next_state, reward, done = env.step(action)
                
                tri_drive_info = env.latest_rewards_info
                t_state=[*t_state, *n_state, *qs, reward, *tri_drive_info]
                writer.writerow(t_state)
                
                episode_rewards += reward
                if ci==0: 
                    full_reward=reward
                    tc0_cumulative_reward += reward
                state = next_state
            
            f.close()
            plt.plot(positions, speeds, "r-", label="Own Train")
            if len(f_positions) > 0:
                plt.plot(f_positions, f_speeds, "b--", label="Forward Train")
            plt.legend(loc="upper right")
            plt.savefig(f"{dir_name}{file_name}_{ci}.png")
            

            plt.figure(dpi=200, figsize=(10, 6))
            plt.xlabel("Time [s]")
            plt.ylabel("Position [km]")
            
            plt.axhline(y=env.departure_station["position"], color='k', linestyle='-', lw=2)
            plt.axhline(y=env.arrival_station["position"], color='k', linestyle='-', lw=2)
            
            plt.plot(times, positions, "r-", label="Own Train")
            
            if len(f_positions) > 0:
                plt.plot(f_times, f_positions, "b--", label="Forward Train")
            
            plt.ylim(env.departure_station["position"] - 0.05, env.arrival_station["position"] + 0.1)
            
            plt.legend(loc="lower right")
            plt.grid(True)
            
            plt.savefig(f"{dir_name}{file_name}_{ci}_diagram.png")
            
            plt.close('all')
            
            ci+=1
            
        try:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
            
        return episode_rewards, file_name, full_reward, tc0_cumulative_reward


def main(num_actors, gamma, num_states, time_step=1.0):
    
    s = time.time()
    now = datetime.datetime.now()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_name = os.path.join(base_dir, "data", now.strftime("%Y%m%d%H%M%S")) + "/"
    
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(dir_name + "replay/", exist_ok=True)
    
    ray.init()
    history = deque(maxlen=1000)
    
    epsilons = np.linspace(0.001, 0.4, num_actors,dtype=np.float32)     
    beta=0.4
    actors = [Actor.remote(pid=i, epsilon=epsilons[i], gamma=gamma, num_states=num_states, time_step=time_step) for i in range(num_actors)]

    replay = Replay(buffer_size=2**19, save_dir=dir_name+"replay/")

    learner = Learner.remote(num_states=num_states, time_step=time_step)
    define_ref = learner.define_network.remote()
    current_weights = ray.get(define_ref)
    free_ray_refs(define_ref)
    current_weights = ray.put(current_weights)
    old_weight_refs = deque()

    tester = Tester.remote(num_states, time_step)

    wip_actors = [actor.rollout.remote(current_weights) for actor in actors]

    for _ in range(30):
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        result_ref = finished[0]
        td_errors, transitions, pid = ray.get(result_ref)
        replay.add(td_errors, transitions)
        free_ray_refs(result_ref)
        del td_errors, transitions
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])

    minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
    minibatchs_ref = ray.put(minibatchs)
    wip_learner = learner.update_network.remote(minibatchs_ref)
    
    wip_tester = tester.test_play.remote(current_weights, dir_name, "0")

    update_cycles = 1
    actor_cycles = 0
    t=time.time()

    while True:
        actor_cycles += 1
        
        while True:  
            finished, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)
            if (len(finished)>0):
                result_ref = finished[0]
                td_errors, transitions, pid= ray.get(result_ref)
                replay.add(td_errors, transitions)
                free_ray_refs(result_ref)
                del td_errors, transitions
                wip_actors.extend([actors[pid].rollout.remote(current_weights)])
            else: break

        
        finished_learner, _ = ray.wait([wip_learner], timeout=0)
        
        if finished_learner:
            learner_ref = finished_learner[0]
            new_weights, indices, td_errors, priority_correction = ray.get(learner_ref)
            
            free_ray_refs(learner_ref)
            free_ray_refs(minibatchs_ref)
            
            old_weight_refs.append(current_weights)
            current_weights = ray.put(new_weights)
            del new_weights
            while len(old_weight_refs) > max(100, num_actors * 4):
                free_ray_refs(old_weight_refs.popleft())
            
            replay.update_priority(indices, td_errors, priority_correction)
            del indices, td_errors, priority_correction
            
            minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
            minibatchs_ref = ray.put(minibatchs)
            wip_learner = learner.update_network.remote(minibatchs_ref)
            
            beta=min(beta+0.6/20000.0,1.0)
            update_cycles += 1
            actor_cycles = 0
            
            current_buffer_size = replay.buffer_size if replay.is_full else replay.count
            print(f"learner {time.time()-t:.2f}s, update_cycles: {update_cycles}, beta: {beta:.5f}, buffer_size: {current_buffer_size} / {replay.buffer_size}")
            t=time.time()

            if update_cycles % 50 == 0:
                gc.collect()
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
                    memory_ref = actor.get_memory.remote()
                    print(ray.get(memory_ref))
                    free_ray_refs(memory_ref)

                tester_ref = wip_tester
                test_score, file_name, full_rewoard, tc0_reward = ray.get(tester_ref)
                free_ray_refs(tester_ref)
                print(file_name, test_score, beta)
                history.append((update_cycles , test_score))
                with open(dir_name + "history.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , test_score))
                with open(dir_name + "history_f.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , full_rewoard))
                with open(dir_name + "history_0.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow((file_name , tc0_reward))
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                sys.stdout.flush()

            # ▼【大改修】Actor, Learner, Tester を一斉に再生成（1000サイクルごと）
            if update_cycles % 1000 == 0:
                print(f"=== [Memory Reset] Actor, Learner, Testerを再生成します (update_cycles: {update_cycles}) ===")
                
                # 1. 進行中の全タスクをキャンセル（Actorタスクに force=True は使えないため外す）
                try:
                    for wip in wip_actors:
                        ray.cancel(wip)
                    ray.cancel(wip_tester)
                    ray.cancel(wip_learner)
                except Exception:
                    pass
                del wip_actors, wip_tester, wip_learner
                free_ray_refs(minibatchs_ref)
                
                # 2. 古い全プロセスを強制終了して破棄
                for actor in actors:
                    ray.kill(actor)
                ray.kill(learner)
                ray.kill(tester)
                del actors, learner, tester
                
                # 3. メインプロセスのゴミ箱を空にする
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                
                # 4. 全プロセスを新しいメモリ空間で再作成
                actors = [Actor.remote(pid=i, epsilon=epsilons[i], gamma=gamma, num_states=num_states, time_step=time_step) for i in range(num_actors)]
                learner = Learner.remote(num_states=num_states, time_step=time_step)
                tester = Tester.remote(num_states, time_step)
                
                # 5. Learnerの初期化と「最新の重み」の完全引き継ぎ
                ray.get(learner.define_network.remote())
                ray.get(learner.set_weights.remote(current_weights))
                
                # 6. 各タスクを再開
                wip_actors = [actor.rollout.remote(current_weights) for actor in actors]
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                
                minibatchs = [replay.sample_minibatch(batch_size=512, beta=beta) for _ in range(64)]
                minibatchs_ref = ray.put(minibatchs)
                wip_learner = learner.update_network.remote(minibatchs_ref)
                
                print("=== 全プロセスの再生成完了 ===")

if __name__ == "__main__":
    main(num_actors=5, gamma=0.9975, num_states=9, time_step=1.0)