"""報酬予測器（二重蒸留の推論側・2026-07-25改修）

RL実行時のパイプライン:
  状態辞書 → 状態特徴量(reward_features) → モードNN(argmax) → mode one-hot
           → [正規化状態 | mode one-hot] → 評価NN(ゲート+回帰) → 報酬

特徴量エンジニアリングは reward_features（共有モジュール）に一本化し、学習側と一致させる。
モデルが無い場合は0.0（後方互換）。モードNNが無い場合は mode=normal 固定で動作する。
"""
import os
import numpy as np
import tensorflow as tf
import joblib

import reward_features as rf


class DirectRewardPredictor:
    def __init__(self,
                 model_path='direct_reward_model2.h5',
                 scaler_path='direct_reward_scaler2.pkl',
                 gate_path='direct_reward_gate2.h5',
                 mode_model_path='mode_model.h5',
                 mode_scaler_path='mode_scaler.pkl'):
        self.is_loaded = False
        self.model = self.gate_model = self.scaler = None
        self.mode_model = self.mode_scaler = None
        self.last_mode = 'normal'
        self.state_dim = len(rf.STATE_FEATURE_COLS)

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。報酬は0.0を返します。")
            return

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)
        self.is_loaded = True

        n_state = getattr(self.scaler, 'n_features_in_', self.state_dim)
        if n_state != self.state_dim:
            print(f"[Warning] スケーラーの特徴量数({n_state})が現行の状態特徴量数({self.state_dim})と一致しません。"
                  f"reward_features.STATE_FEATURE_COLS とモデル世代を確認してください。")
        input_dim = n_state + rf.MODE_DIM

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_dim], dtype=tf.float32)])
        def predict_fn(x):
            return self.model(x, training=False)
        self.predict_fn = predict_fn

        # ゲート分類器（無ければ回帰器のみ・後方互換）
        if os.path.exists(gate_path):
            self.gate_model = tf.keras.models.load_model(gate_path, compile=False)

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_dim], dtype=tf.float32)])
            def predict_gate_fn(x):
                return self.gate_model(x, training=False)
            self.predict_gate_fn = predict_gate_fn
        else:
            print(f"[Warning] {gate_path} が見つかりません。ゲートなし（回帰器のみ）で動作します。")

        # モードNN（無ければ / 次元不一致なら mode=normal 固定・後方互換）
        if os.path.exists(mode_model_path) and os.path.exists(mode_scaler_path):
            mode_scaler = joblib.load(mode_scaler_path)
            n_mode = getattr(mode_scaler, 'n_features_in_', self.state_dim)
            if n_mode != self.state_dim:
                # モードNNの世代が古い（例: forward_observed_delay 追加前の52次元）と
                # mode_scaler.transform で分類例外（features不一致）になるため、ロードせず
                # mode=normal 固定に退避する。評価NN側の次元チェックと同じ思想。
                print(f"[Warning] モードNNの特徴量数({n_mode})が現行の状態特徴量数({self.state_dim})と不一致のため、"
                      f"モードNNを無効化し mode=normal 固定で動作します。train_mode_network.py を再学習してください。")
            else:
                self.mode_model = tf.keras.models.load_model(mode_model_path, compile=False)
                self.mode_scaler = mode_scaler

                @tf.function(input_signature=[tf.TensorSpec(shape=[1, n_mode], dtype=tf.float32)])
                def predict_mode_fn(x):
                    return self.mode_model(x, training=False)
                self.predict_mode_fn = predict_mode_fn
        else:
            print(f"[Warning] {mode_model_path}/{mode_scaler_path} が見つかりません。mode=normal 固定で動作します。")

    def _infer_mode(self, x_state_raw):
        """状態特徴量(1,state_dim) → (mode one-hot(1,MODE_DIM), mode文字列)。"""
        if self.mode_model is None:
            oh = rf.mode_to_onehot('normal').reshape(1, -1)
            return oh, 'normal'
        x_scaled = self.mode_scaler.transform(x_state_raw)
        probs = self.predict_mode_fn(tf.convert_to_tensor(x_scaled, dtype=tf.float32)).numpy()[0]
        idx = int(np.argmax(probs))
        mode_str = rf.MODE_CLASSES_ACTIVE[idx] if idx < len(rf.MODE_CLASSES_ACTIVE) else 'normal'
        return rf.mode_to_onehot(mode_str).reshape(1, -1), mode_str

    def predict_reward(self, state_info):
        if not self.is_loaded:
            return 0.0
        try:
            x_state = rf.state_vector(state_info).reshape(1, -1)  # (1, state_dim)
            mode_oh, mode_str = self._infer_mode(x_state)
            self.last_mode = mode_str

            x_state_scaled = self.scaler.transform(x_state)
            X = np.hstack([x_state_scaled, mode_oh]).astype(np.float32)
            x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

            if self.gate_model is not None:
                zero_prob = float(self.predict_gate_fn(x_tensor).numpy()[0][0])
                if zero_prob >= 0.5:
                    return 0.0
                reg_value = float(self.predict_fn(x_tensor).numpy()[0][0])
                return round(min(max(reg_value, 0.1), 1.0), 1)

            reg_value = float(self.predict_fn(x_tensor).numpy()[0][0])
            return round(reg_value, 1)
        except Exception as e:
            print(f"[推論例外発生] {e}")
            return 0.0
