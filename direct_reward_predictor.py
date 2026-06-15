import numpy as np
import tensorflow as tf
import joblib
import os
import re

class DirectRewardPredictor:
    def __init__(self, model_path='direct_reward_model.h5', scaler_path='direct_reward_scaler.pkl'):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
            
            # ▼変更1: 次元数を 34 から 43 に変更（特徴量追加に合わせる）
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 43], dtype=tf.float32)])
            def predict_fn(x):
                return self.model(x, training=False)
            self.predict_fn = predict_fn
            
        else:
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。デフォルトの0.0を使用します。")
            self.is_loaded = False
        
        self.phase_categories = [
            "駅出発直後の加速フェーズ（20秒以内）", 
            "巡航フェーズ（駅間走行中）", 
            "制限速度区間に接近中（500m以内に制限区間在り）", 
            "次駅への減速フェーズ（駅手前400m以内）",
            "駅停車完了（速度0km/h）"
        ]
        self.notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
        self.prev_notch_categories = ["惰行", "力行（加速）", "ブレーキ（減速）"]

    def predict_reward(self, state_info):
        if not self.is_loaded:
            return 0.0
            
        # 推論時のノッチ文字列置換
        notch_str = state_info.get('current_notch', '') # engineからのキー名に合わせて 'current_notch' も取得できるようにフォールバック
        if not notch_str:
            notch_str = state_info.get('notch', '')
        if notch_str == '停止・その他':
            notch_str = 'ブレーキ（減速）中'
            
        prev_notch_str = state_info.get('prev_notch', 'ブレーキ（減速）')
        if prev_notch_str in ['なし（または停止）', 'なし', '停止・その他']:
            prev_notch_str = 'ブレーキ（減速）'

        # 1. 保持時間の分離
        hold_time = float(state_info.get('holding_time', 0.0))
        prev_duration = float(state_info.get('prev_notch_duration', 0.0))
        
        hold_coast = hold_time if "惰行中" in notch_str else 0.0
        hold_accel = hold_time if "力行" in notch_str else 0.0
        hold_decel = hold_time if "ブレーキ" in notch_str else 0.0
        
        # 2. 状態変数の取得
        speed = float(state_info.get('current_speed', 0.0))
        limit = float(state_info.get('speed_limit', 0.0))
        dist = float(state_info.get('dist_to_next_station', 0.0))
        req_dist = float(state_info.get('req_stop_dist', 0.0))
        time_to_next = float(state_info.get('time_to_next_station', 0.0))
        delay = float(state_info.get('delay', 0.0))
        current_gradient = float(state_info.get('current_gradient', 0.0))
        
        next_limit_flag, next_limit_dist, next_limit_speed = self._extract_limit_info(state_info.get('next_limit_info', ''))
        next_gradient_flag, next_gradient_dist, next_gradient_val = self._extract_gradient_info(state_info.get('next_gradient_info', ''))
        f_exist, f_distance, f_speed = self._extract_forward_info(state_info.get('forward_info', ''))
        b_exist, b_distance, b_speed = self._extract_backward_info(state_info.get('backward_info', ''))

        # =====================================================================
        # ▼▼▼ 変更2: 新しい派生特徴量のリアルタイム計算 ▼▼▼
        # =====================================================================
        margin_speed = limit - speed
        margin_stop_dist = dist - req_dist
        required_speed_mps = dist / (time_to_next + 1e-3)
        required_cruise_speed = (required_speed_mps * 3.6) + 20.0
        
        is_hunting = (hold_time < 5.0) and (prev_duration < 5.0) and (notch_str != prev_notch_str)
        hunting_score = max(0.0, 5.0 - hold_time) / 5.0 if is_hunting else 0.0
        
        f_relative_speed = speed - f_speed
        f_ttc = f_distance / (f_relative_speed / 3.6 + 1e-3) if f_relative_speed > 0 else 5000.0
        
        b_relative_speed = b_speed - speed
        b_ttc = b_distance / (b_relative_speed / 3.6 + 1e-3) if b_relative_speed > 0 else 5000.0
        
        # =====================================================================
        # ▼▼▼ 変更3: 学習時と同じクリッピング処理の適用 ▼▼▼
        # =====================================================================
        dist = min(dist, 2000.0)
        req_dist = min(req_dist, 2000.0)
        f_distance = min(f_distance, 2000.0)
        f_ttc = min(f_ttc, 5000.0)
        b_distance = min(b_distance, 2000.0)
        b_ttc = min(b_ttc, 5000.0)

        # 3. Categorical の One-Hot ベクトル化
        phase_str = state_info.get('phase', '')
        phase_onehot = [1.0 if cat == phase_str else 0.0 for cat in self.phase_categories]
        notch_onehot = [1.0 if cat == notch_str else 0.0 for cat in self.notch_categories]
        prev_notch_onehot = [1.0 if prev_notch_str == cat else 0.0 for cat in self.prev_notch_categories]

        # 4. 特徴量ベクトルの結合 (43次元：学習スクリプトと完全に同じ順序にする)
        features = [
            hold_coast, hold_accel, hold_decel,
            prev_duration,
            limit, speed, dist, time_to_next, req_dist, delay, current_gradient,
            # --- 追加した派生特徴量 ---
            margin_speed, margin_stop_dist, required_speed_mps, required_cruise_speed, hunting_score,
            f_relative_speed, f_ttc, b_relative_speed, b_ttc,
            # --------------------------
            next_limit_flag, next_limit_dist, next_limit_speed,
            next_gradient_flag, next_gradient_dist, next_gradient_val,
            f_exist, f_distance, f_speed,
            b_exist, b_distance, b_speed
        ] + phase_onehot + notch_onehot + prev_notch_onehot
        
        # 5. スケーリングと推論
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        predicted_tensor = self.predict_fn(tf.convert_to_tensor(X_scaled, dtype=tf.float32))
        
        # ▼変更4: linear出力のため、0.0〜1.0にクリップして丸める
        raw_reward = float(predicted_tensor.numpy()[0][0])
        clipped_reward = max(0.0, min(1.0, raw_reward))
        rounded_reward = round(clipped_reward * 10) / 10.0
        
        del predicted_tensor, X, X_scaled
        
        return rounded_reward

    def _extract_limit_info(self, text):
        text = str(text)
        if "この先制限速度なし" in text:
            return 0.0, 0.0, 0.0
        match = re.search(r'(\d+)m先に制限速度(\d+)km/h', text)
        if match:
            return 1.0, float(match.group(1)), float(match.group(2))
        return 0.0, 0.0, 0.0

    def _extract_gradient_info(self, text):
        text = str(text)
        if "この先目立った勾配なし" in text:
            return 0.0, 0.0, 0.0
        match = re.search(r'(\d+)m先に(上り|下り)勾配(\d+\.?\d*)‰あり', text)
        if match:
            dist = float(match.group(1))
            direction = match.group(2)
            val = float(match.group(3))
            if direction == '下り':
                val = -val
            return 1.0, dist, val
        return 0.0, 0.0, 0.0

    def _extract_forward_info(self, text):
        text = str(text)
        if "先行列車なし" in text or text == "nan":
            return 0.0, 5000.0, 0.0 
        match = re.search(r'前方\s*([\d\.]+)\s*m\s*先を\s*([\d\.]+)\s*km/h', text)
        if match:
            return 1.0, float(match.group(1)), float(match.group(2))
        return 0.0, 5000.0, 0.0

    def _extract_backward_info(self, text):
        text = str(text)
        if "後続列車なし" in text or text == "nan":
            return 0.0, 5000.0, 0.0 
        match = re.search(r'後方\s*([\d\.]+)\s*m\s*後ろを\s*([\d\.]+)\s*km/h', text)
        if match:
            return 1.0, float(match.group(1)), float(match.group(2))
        return 0.0, 5000.0, 0.0