# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田 → 白兎 → 蚕桑）の Apex DQN 学習エントリポイント。

既存 `apex2.py` / `apex.py` / `apex3.py` は**一切変更しない**。出力先も `data_ymulti/` に分離する。

既存 apex2.py との違い
  1. 環境が `environment_ymulti.EnvironmentYMulti`（**1エピソード＝2駅間＋中間駅の停車判断**）
  2. 観測が30次元 → **40次元**（停車中フラグ・停車経過・標準運転曲線との差・通算ダイヤなど）
  3. **Double DQN を既定にする。** 複数駅間ではエピソードが2倍以上長く、駅手前100mの0.1秒刻みが
     駅の数だけ増えるためブートストラップ連鎖が伸び、vanilla max の過大評価バイアスが
     蓄積しやすい（CLAUDE.md「残る副作用」・計画書 §8.3）。`--no-double` で従来動作に戻せる。
  4. 先行列車は `input/f_train_ymulti/`（白兎・蚕桑の2駅で停車するパターン）を使う。

行動空間は既存と同じ3ノッチ（`actions.Actions`）。停車中は環境側で
「力行＝発車 / 制動＝待機」に読み替え、惰行を `forbidden_action` で禁止する。

使い方:
    python apex_ymulti.py                      # 既定（Actor 6並列・Double DQN）
    python apex_ymulti.py --actors 4
    python apex_ymulti.py --reward rule        # 報酬NNが未学習のうちは暫定ルール報酬で回す
"""
import os

# メモリ対策（既存 apex2.py と同じ）
os.environ["MALLOC_ARENA_MAX"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_raylet_start_wait_time_s"] = "120"

import sys
import gc
import csv
import json
import time
import ctypes
import random
import argparse
import datetime
from collections import deque

import numpy as np
import ray

# psutil は「メモリ使用量の表示」だけに使う任意依存。
# 現在の .venv には入っておらず requirements.txt にも記載が無い（既存 apex2.py はここで
# ImportError になる）。学習の本体に影響しないので、無い場合は表示を省いて動かす。
try:
    import psutil
except ImportError:
    psutil = None
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")

from segment_tree import SumTree
from model import QNetwork
from actions import Actions

import config_ymulti as CFG
import reward_features_ymulti as rf
from environment_ymulti import EnvironmentYMulti
import runcurve_plot_ymulti as rcp

tf.config.set_visible_devices([], "GPU")
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

NUM_STATES = EnvironmentYMulti.NUM_STATES
NUM_ACTIONS = len(Actions)


def free_ray_refs(*refs):
    flat = []
    for ref in refs:
        if ref is None:
            continue
        if isinstance(ref, (list, tuple, set)):
            flat.extend(x for x in ref if x is not None)
        else:
            flat.append(ref)
    if not flat:
        return
    try:
        ray._private.internal_api.free(flat, local_only=False)
    except Exception:
        try:
            ray.internal.free(flat)
        except Exception:
            pass


# =============================================================================
# 学習シナリオ（Actor がランダムに選ぶ）
# =============================================================================
def sample_scenario(rng):
    """学習用のシナリオを1つ引く。

    先行なし4割 / 先行あり6割。先行ありは惰行ポイント・両駅の停車時間・出発間隔をランダムに選ぶ。
    自列車の出発遅延は 0〜60秒（既存 apex2 は0〜20秒だが、複数駅間では
    **自列車が遅れているほど先行の延着を観測しやすくなる**ため、発車判断の学習に効く）。
    """
    ego_delay = rng.uniform(0.0, 60.0)
    if rng.random() < 0.4:
        return {"delay": ego_delay}
    coast = rng.choice(CFG.F_COAST_SPEEDS_TRAIN)
    b = rng.choice(CFG.F_DWELL_B)
    c = rng.choice(CFG.F_DWELL_C)
    headway = rng.uniform(40.0, 120.0)
    return {"delay": ego_delay,
            "f_train_csv": CFG.f_train_csv(coast, b, c),
            "headway": headway}


def build_test_cases():
    """検証（Tester）用のテストケース。"""
    tc = [{"desc": "Sim1_Normal", "kw": {}}]
    for d in (15.0, 30.0, 60.0):
        tc.append({"desc": f"Sim2_EgoDelay{int(d)}s", "kw": {"delay": d}})
    for coast in CFG.F_COAST_SPEEDS_TEST:
        for hw in (120.0, 90.0, 60.0):
            for b, c in ((30, 30), (30, 120), (30, 180)):
                tc.append({
                    "desc": f"Sim3_v{coast}_hw{int(hw)}_b{b}_c{c}",
                    "kw": {"f_train_csv": CFG.f_train_csv(coast, b, c), "headway": hw},
                })
    return tc


TEST_CASES = build_test_cases()


# =============================================================================
# Actor
# =============================================================================
@ray.remote
class Actor:
    def __init__(self, pid, epsilon, gamma, time_step, reward_mode, double_dqn, seed):
        tf.config.set_visible_devices([], "GPU")
        self.pid = pid
        self.time_step = time_step
        self.epsilon = epsilon
        self.__gamma = gamma
        self.double_dqn = double_dqn
        self.rng = random.Random(seed)
        self.env = EnvironmentYMulti(time_step, reward_mode=reward_mode)
        self.q_network = QNetwork(NUM_STATES)
        self.buffer = []
        state = self.env.reset()
        self.q_network(np.atleast_2d(state))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, NUM_STATES], dtype=tf.float32)])
        def predict_q_batch(x):
            return self.q_network(x, training=False)
        self.predict_q_batch = predict_q_batch
        self.episode_rewards = 0.0

    @property
    def gamma(self):
        """実時間あたりの割引を揃える（駅手前0.1秒・停車中1.0秒でステップ幅が変わるため）。"""
        return self.__gamma ** (self.env.time_step / self.time_step)

    def rollout(self, current_weights):
        for var, w in zip(self.q_network.variables, current_weights):
            var.assign(w)

        state = self.env.reset(**sample_scenario(self.rng))
        self.episode_rewards = 0.0
        done = False
        steps = 0
        while not done and steps < 12000:
            st = tf.convert_to_tensor(np.array(state)[np.newaxis, ...], dtype=tf.float32)
            qs = self.predict_q_batch(st).numpy()[0]
            del st
            forbidden = self.env.forbidden_action
            if self.rng.random() < self.epsilon:
                valid = [i for i, f in enumerate(forbidden) if not f]
                action = self.rng.choice(valid)
            else:
                masked = qs.copy()
                masked[forbidden] = -np.inf
                action = int(np.argmax(masked))

            # 駅の直前ほど優先度を上げる（既存 apex2 と同じ考え方）
            priority_correction = (0.1 - (min(max(self.env.station_remaining_distance, 0.0), 0.1) + 0.001)) * 500 + 1
            gamma = self.gamma
            next_state, reward, done = self.env.step(action)
            next_forbidden = self.env.forbidden_action
            self.episode_rewards += reward
            executed = self.env.executed_action
            if executed is not None:
                action = int(executed)
            self.buffer.append((state, action, reward, next_state, done,
                                next_forbidden, gamma, priority_correction))
            state = next_state
            steps += 1

        td_errors = self._td_errors()
        transitions = self.buffer
        self.buffer = []
        info = {"pid": self.pid, "steps": steps, "reward": self.episode_rewards,
                "goal": self.env.goal_reached, "failed": self.env.failed,
                "reason": self.env.fail_reason}
        try:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return td_errors, transitions, self.pid, info

    def _td_errors(self):
        states = np.vstack([t[0] for t in self.buffer])
        actions = np.array([t[1] for t in self.buffer])
        rewards = np.vstack([t[2] for t in self.buffer])
        next_states = np.vstack([t[3] for t in self.buffer])
        dones = np.vstack([t[4] for t in self.buffer])
        next_forbidden = np.vstack([t[5] for t in self.buffer])
        gammas = np.vstack([t[6] for t in self.buffer])

        chunk = 512
        nq, cq = [], []
        for i in range(0, len(states), chunk):
            nq.append(self.predict_q_batch(
                tf.convert_to_tensor(next_states[i:i + chunk], dtype=tf.float32)).numpy())
            cq.append(self.predict_q_batch(
                tf.convert_to_tensor(states[i:i + chunk], dtype=tf.float32)).numpy())
        next_q = np.concatenate(nq, axis=0)
        q = np.concatenate(cq, axis=0)

        next_q_masked = next_q + (next_forbidden * -1.0 * (10 ** 12))
        next_actions = np.argmax(next_q_masked, axis=1)
        onehot = np.eye(NUM_ACTIONS)[next_actions]
        next_maxQ = np.sum(next_q_masked * onehot, axis=1, keepdims=True)
        TQ = rewards + gammas * (1 - dones) * next_maxQ
        Q = np.sum(q * np.eye(NUM_ACTIONS)[actions], axis=1, keepdims=True)
        return (TQ - Q).flatten()

    def get_memory(self):
        rss = (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3) if psutil else None
        return {"pid": self.pid, "memory_gb": rss, "buffer_len": len(self.buffer)}


# =============================================================================
# Replay（既存 apex2.py と同一の優先度付き経験再生）
# =============================================================================
class Replay:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=buffer_size)
        self.buffer = [None] * buffer_size
        self.alpha = 0.6
        self.count = 0
        self.is_full = False

    def add(self, td_errors, transitions):
        priorities = (np.abs(td_errors) + 0.001) ** self.alpha
        for p, tr in zip(priorities, transitions):
            self.priorities[self.count] = p * tr[-1]
            self.buffer[self.count] = tr
            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def update_priority(self, indices, td_errors, corrections):
        for idx, td, c in zip(indices, td_errors, corrections):
            self.priorities[idx] = (abs(td) + 0.001) ** self.alpha * c

    def sample_minibatch(self, batch_size, beta):
        idxs = [self.priorities.sample() for _ in range(batch_size)]
        size = len(self.buffer) if self.is_full else self.count
        weights = []
        for i in idxs:
            prob = self.priorities[i] / self.priorities.sum()
            weights.append((prob * size) ** (-beta))
        weights = np.array(weights) / max(weights)
        return idxs, weights, [self.buffer[i] for i in idxs]


# =============================================================================
# Learner
# =============================================================================
@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, time_step, double_dqn):
        devs = tf.config.list_physical_devices("GPU")
        if devs:
            tf.config.set_visible_devices(devs[0], "GPU")
        else:
            print("[Learner] GPUが見つかりません。CPUで実行します。")
        self.time_step = time_step
        self.double_dqn = double_dqn
        self.q_network = QNetwork(NUM_STATES)
        self.target_q_network = QNetwork(NUM_STATES)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def define_network(self):
        env = EnvironmentYMulti(self.time_step, load_reward_predictor=False, reward_mode="rule")
        state = env.reset()
        self.q_network(np.atleast_2d(state))
        self.target_q_network(np.atleast_2d(state))
        self.target_q_network.set_weights(self.q_network.get_weights())
        return self.q_network.get_weights()

    def set_weights(self, weights):
        self.q_network.set_weights(weights)
        self.target_q_network.set_weights(weights)
        return True

    def update_network(self, minibatchs):
        indices_all, td_all, corr_all = [], [], []
        for (indices, weights, transitions) in minibatchs:
            states, actions, rewards, next_states, dones, next_forbidden, gammas, corr = zip(*transitions)
            states = np.vstack(states)
            actions = np.array(actions)
            rewards = np.vstack(rewards)
            next_states = np.vstack(next_states)
            dones = np.vstack(dones)
            next_forbidden = np.vstack(next_forbidden)
            gammas = np.vstack(gammas)
            corr = np.vstack(corr)

            mask = next_forbidden * -1.0 * (10 ** 12)
            next_q_target = self.target_q_network(next_states) + mask
            if self.double_dqn:
                # Double DQN: 行動選択はオンラインネット、評価はターゲットネット。
                # 複数駅間はエピソードが長く、vanilla max だと過大評価が蓄積しやすい。
                next_q_online = self.q_network(next_states) + mask
                next_actions = tf.cast(tf.argmax(next_q_online, axis=1), tf.int32)
            else:
                next_actions = tf.cast(tf.argmax(next_q_target, axis=1), tf.int32)
            onehot = tf.one_hot(next_actions, NUM_ACTIONS)
            next_maxQ = tf.reduce_sum(next_q_target * onehot, axis=1, keepdims=True)
            TQ = rewards + gammas * (1 - dones) * next_maxQ

            w = tf.convert_to_tensor(np.asarray(weights, dtype=np.float32).reshape(-1, 1))
            with tf.GradientTape() as tape:
                qvalues = self.q_network(states)
                Q = tf.reduce_sum(qvalues * tf.one_hot(actions, NUM_ACTIONS), axis=1, keepdims=True)
                td_delta = TQ - Q
                loss = tf.reduce_mean(w * tf.square(td_delta))
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

            indices_all += list(indices)
            td_all += td_delta.numpy().flatten().tolist()
            corr_all += corr.flatten().tolist()

        weights_out = self.q_network.get_weights()
        self.target_q_network.set_weights(weights_out)
        try:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return weights_out, indices_all, td_all, corr_all


# =============================================================================
# Tester
# =============================================================================
@ray.remote
class Tester:
    def __init__(self, time_step, reward_mode):
        tf.config.set_visible_devices([], "GPU")
        self.time_step = time_step
        self.env = EnvironmentYMulti(time_step, reward_mode=reward_mode)
        self.q_network = QNetwork(NUM_STATES)
        state = self.env.reset()
        self.q_network(np.atleast_2d(state))

    def test_play(self, current_weights, dir_name, file_name):
        self.q_network.set_weights(current_weights)
        self.q_network.save_weights(os.path.join(dir_name, file_name + ".weights.h5"))
        env = self.env
        summary = []
        total_reward = 0.0

        for ci, tc in enumerate(TEST_CASES):
            state = env.reset(**tc["kw"])
            positions, speeds, modes, times = [], [], [], []
            f_positions, f_times = [], []
            rows = []
            ep_reward = 0.0
            done = False
            steps = 0
            while not done and steps < 12000:
                qs = np.array(self.q_network(np.atleast_2d(state)))[0]
                forbidden = env.forbidden_action
                masked = qs.copy()
                masked[forbidden] = -np.inf
                action = int(np.argmax(masked))
                raw_before_t = env.t
                state, reward, done = env.step(action)
                ep_reward += reward
                raw = env.last_raw_state
                positions.append(env.position)
                speeds.append(env.speed)
                modes.append(env.current_mode)
                times.append(env.t)
                if env.fowerd_train is not None:
                    f_positions.append(env.fowerd_train.position)
                    f_times.append(env.t)
                rows.append([raw.get(c, "") for c in rf.RAW_COLS]
                            + [f"{q:.5f}" for q in qs]
                            + [f"{reward:.5f}", f"{env.last_llm_reward:.5f}"]
                            + [f"{v:.6f}" for v in state])
                steps += 1

            total_reward += ep_reward
            dwell = env.dwell_log[0] if env.dwell_log else {}
            info = {
                "case": ci, "desc": tc["desc"], "reward": ep_reward, "steps": steps,
                "goal": bool(env.goal_reached), "failed": bool(env.failed),
                "reason": env.fail_reason, "arrival_t": env.t,
                "schedule_t": CFG.total_scheduled_time(),
                "delay_s": env.t - CFG.total_scheduled_time(),
                "stop_error_m": (env.position - env.stations[-1]["position"]) * 1000.0,
                "dwell_s": dwell.get("dwell"),
                "dwell_forced": dwell.get("forced"),
                "mid_stop_error_m": dwell.get("stop_error_m"),
            }
            summary.append(info)

            # --- CSV ---
            header = list(rf.RAW_COLS) + [f"q_{a.name}" for a in Actions] \
                + ["step_reward", "llm_reward"] + [f"obs_{i}" for i in range(NUM_STATES)]
            with open(os.path.join(dir_name, f"{file_name}_{ci}.csv"), "w", newline="",
                      encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            with open(os.path.join(dir_name, f"{file_name}_{ci}_meta.json"), "w",
                      encoding="utf-8") as f:
                json.dump({**info, "kw": {k: v for k, v in tc["kw"].items()}},
                          f, ensure_ascii=False, indent=2)

            # --- PNG ---
            station_positions = [s["position"] for s in env.stations]
            limit_sections = env.train.track.get_front_sections(
                station_positions[0], station_positions[-1])
            secs = []
            pos = station_positions[0]
            for s in limit_sections:
                secs.append({"start": pos, "distance": s["distance"], "speed_limit": s["speed_limit"]})
                pos += s["distance"]
            title = (f"{tc['desc']} / 報酬 {ep_reward:+.1f} / 到着 {env.t:.0f}s"
                     f"（標準 {CFG.total_scheduled_time():.0f}s）/ 停車 {info['dwell_s']}s"
                     f" / 停止誤差 {info['stop_error_m']:+.2f}m")
            rcp.plot_run_curve(os.path.join(dir_name, f"{file_name}_{ci}.png"),
                               positions, speeds, modes, station_positions, secs, title)
            rcp.plot_diagram(os.path.join(dir_name, f"{file_name}_{ci}_diagram.png"),
                             times, positions, station_positions, title,
                             f_times or None, f_positions or None)

        with open(os.path.join(dir_name, f"{file_name}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        n_goal = sum(1 for s in summary if s["goal"])
        return total_reward, file_name, n_goal, summary[0]["reward"]


# =============================================================================
# main
# =============================================================================
def main(num_actors=6, gamma=0.9975, time_step=1.0, reward_mode="auto",
         double_dqn=True, test_interval=50):
    now = datetime.datetime.now()
    base = os.path.abspath(os.path.dirname(__file__))
    dir_name = os.path.join(base, CFG.DATA_DIR, now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"num_actors": num_actors, "gamma": gamma, "time_step": time_step,
                   "reward_mode": reward_mode, "double_dqn": double_dqn,
                   "num_states": NUM_STATES, "stations": CFG.STATION_INDICES,
                   "running_times": CFG.RUNNING_TIMES, "dwell_min": CFG.DWELL_MIN,
                   "dwell_max": CFG.DWELL_MAX, "test_cases": [t["desc"] for t in TEST_CASES]},
                  f, ensure_ascii=False, indent=2)
    print(f"[apex_ymulti] 出力先: {dir_name}")
    print(f"  観測 {NUM_STATES} 次元 / 行動 {NUM_ACTIONS} / Double DQN={double_dqn} / "
          f"報酬={reward_mode} / テストケース {len(TEST_CASES)} 件")

    ray.init()
    epsilons = np.linspace(0.001, 0.4, num_actors, dtype=np.float32)
    beta = 0.4
    actors = [Actor.remote(pid=i, epsilon=float(epsilons[i]), gamma=gamma, time_step=time_step,
                           reward_mode=reward_mode, double_dqn=double_dqn, seed=1000 + i)
              for i in range(num_actors)]
    replay = Replay(buffer_size=2 ** 19)
    learner = Learner.remote(time_step=time_step, double_dqn=double_dqn)
    ref = learner.define_network.remote()
    current_weights = ray.get(ref)
    free_ray_refs(ref)
    current_weights = ray.put(current_weights)
    old_weight_refs = deque()
    tester = Tester.remote(time_step, reward_mode)

    wip_actors = [a.rollout.remote(current_weights) for a in actors]
    ep_stats = deque(maxlen=200)
    for _ in range(30):
        finished, wip_actors = ray.wait(wip_actors, num_returns=1)
        td, trans, pid, info = ray.get(finished[0])
        replay.add(td, trans)
        ep_stats.append(info)
        free_ray_refs(finished[0])
        wip_actors.append(actors[pid].rollout.remote(current_weights))

    minibatchs = [replay.sample_minibatch(512, beta) for _ in range(64)]
    minibatchs_ref = ray.put(minibatchs)
    wip_learner = learner.update_network.remote(minibatchs_ref)
    wip_tester = tester.test_play.remote(current_weights, dir_name, "0")

    update_cycles = 1
    t = time.time()
    try:
        while True:
            while True:
                finished, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)
                if not finished:
                    break
                td, trans, pid, info = ray.get(finished[0])
                replay.add(td, trans)
                ep_stats.append(info)
                free_ray_refs(finished[0])
                wip_actors.append(actors[pid].rollout.remote(current_weights))

            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if not finished_learner:
                continue

            new_weights, indices, td_errors, corr = ray.get(finished_learner[0])
            free_ray_refs(finished_learner[0], minibatchs_ref)
            old_weight_refs.append(current_weights)
            current_weights = ray.put(new_weights)
            while len(old_weight_refs) > max(100, num_actors * 4):
                free_ray_refs(old_weight_refs.popleft())
            replay.update_priority(indices, td_errors, corr)
            minibatchs = [replay.sample_minibatch(512, beta) for _ in range(64)]
            minibatchs_ref = ray.put(minibatchs)
            wip_learner = learner.update_network.remote(minibatchs_ref)
            beta = min(beta + 0.6 / 20000.0, 1.0)
            update_cycles += 1

            size = replay.buffer_size if replay.is_full else replay.count
            if ep_stats:
                steps = np.mean([e["steps"] for e in ep_stats])
                goal = np.mean([1.0 if e["goal"] else 0.0 for e in ep_stats])
                rew = np.mean([e["reward"] for e in ep_stats])
            else:
                steps = goal = rew = 0.0
            print(f"learner {time.time()-t:.2f}s cycle={update_cycles} beta={beta:.5f} "
                  f"buffer={size}/{replay.buffer_size} | 直近エピソード: "
                  f"平均{steps:.0f}歩 到達率{goal*100:.0f}% 平均報酬{rew:+.1f}")
            t = time.time()

            if update_cycles % test_interval == 0:
                gc.collect()
                if psutil is not None:
                    mem = psutil.virtual_memory()
                    print(f"==== [Memory] {mem.percent}% "
                          f"({(mem.total-mem.available)/1024**3:.2f}/{mem.total/1024**3:.2f} GB) ====")
                score, fname, n_goal, tc0 = ray.get(wip_tester)
                free_ray_refs(wip_tester)
                print(f"[Tester] cycle={fname} 合計報酬={score:+.1f} "
                      f"到達 {n_goal}/{len(TEST_CASES)} ケース / 通常運転={tc0:+.1f}")
                with open(os.path.join(dir_name, "history.csv"), "a", newline="") as f:
                    csv.writer(f).writerow((fname, score, n_goal, tc0, steps, goal, rew))
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                sys.stdout.flush()

            # 1000サイクルごとに全プロセスを作り直す（メモリ肥大対策・既存 apex2 と同じ）
            if update_cycles % 1000 == 0:
                print(f"=== [Memory Reset] cycle={update_cycles} ===")
                try:
                    for w in wip_actors:
                        ray.cancel(w)
                    ray.cancel(wip_tester)
                    ray.cancel(wip_learner)
                except Exception:
                    pass
                for a in actors:
                    ray.kill(a)
                ray.kill(learner)
                ray.kill(tester)
                del actors, learner, tester, wip_actors, wip_tester, wip_learner
                free_ray_refs(minibatchs_ref)
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                actors = [Actor.remote(pid=i, epsilon=float(epsilons[i]), gamma=gamma,
                                       time_step=time_step, reward_mode=reward_mode,
                                       double_dqn=double_dqn, seed=1000 + i + update_cycles)
                          for i in range(num_actors)]
                learner = Learner.remote(time_step=time_step, double_dqn=double_dqn)
                tester = Tester.remote(time_step, reward_mode)
                ray.get(learner.define_network.remote())
                ray.get(learner.set_weights.remote(current_weights))
                wip_actors = [a.rollout.remote(current_weights) for a in actors]
                wip_tester = tester.test_play.remote(current_weights, dir_name, str(update_cycles))
                minibatchs = [replay.sample_minibatch(512, beta) for _ in range(64)]
                minibatchs_ref = ray.put(minibatchs)
                wip_learner = learner.update_network.remote(minibatchs_ref)
                print("=== 再生成完了 ===")
    except KeyboardInterrupt:
        print("\n中断しました。")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="複数駅間版（羽前成田→白兎→蚕桑）の Apex DQN 学習")
    ap.add_argument("--actors", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9975, help="1秒あたりの割引率")
    ap.add_argument("--time-step", type=float, default=1.0)
    ap.add_argument("--reward", choices=["auto", "nn", "rule"], default="auto",
                    help="auto=報酬NNがあればNN・無ければ暫定ルール / nn=NN固定 / rule=暫定ルール固定")
    ap.add_argument("--no-double", action="store_true", help="Double DQN を無効にする（従来動作）")
    ap.add_argument("--test-interval", type=int, default=50)
    a = ap.parse_args()
    main(num_actors=a.actors, gamma=a.gamma, time_step=a.time_step, reward_mode=a.reward,
         double_dqn=not a.no_double, test_interval=a.test_interval)
