import numpy as np
import tensorflow as tf
import joblib
import os
import re

class DirectRewardPredictor:
    def __init__(self, model_path='direct_reward_model.h5', scaler_path='direct_reward_scaler.pkl'):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
            
            # ▼変更1: 次元数を 33 から 34 に変更
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 34], dtype=tf.float32)])
            def predict_fn(x):
                return self.model(x, training=False)
            self.predict_fn = predict_fn
            
        else:
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。デフォルトの0.0を使用します。")
            self.is_loaded = False
        
        # ▼変更2: フェーズに「駅停車完了（速度0km/h）」を追加
        self.phase_categories = [
            "駅出発直後の加速フェーズ（20秒以内）", 
            "巡航フェーズ（駅間走行中）", 
            "制限速度区間に接近中（500m以内に制限区間在り）", 
            "次駅への減速フェーズ（駅手前400m以内）",
            "駅停車完了（速度0km/h）"
        ]
        self.notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]
        # ▼変更3: 直前のノッチカテゴリを3つに削減
        self.prev_notch_categories = ["惰行", "力行（加速）", "ブレーキ（減速）"]

    def predict_reward(self, state_info):
        if not self.is_loaded:
            return 0.0
            
        # ▼変更3: 推論時のノッチ文字列置換（学習時の前処理と同じにする）
        notch_str = state_info.get('notch', '')
        if notch_str == '停止・その他':
            notch_str = 'ブレーキ（減速）中'
            
        prev_notch_str = state_info.get('prev_notch', 'ブレーキ（減速）')
        if prev_notch_str in ['なし（または停止）', 'なし', '停止・その他']:
            prev_notch_str = 'ブレーキ（減速）'

        # 1. 保持時間の分離 (hold_coast, hold_accel, hold_decel)
        hold_time = float(state_info.get('holding_time', 0.0))
        
        # 置換後の notch_str を用いて判定
        hold_coast = hold_time if "惰行中" in notch_str else 0.0
        hold_accel = hold_time if "力行" in notch_str else 0.0
        hold_decel = hold_time if "ブレーキ" in notch_str else 0.0
        
        # 直前の保持時間を取得
        prev_duration = state_info.get('prev_notch_duration', 0.0)
        
        # 2. Phase の One-Hot ベクトル化
        phase_str = state_info.get('phase', '')
        phase_onehot = [1.0 if cat == phase_str else 0.0 for cat in self.phase_categories]
        
        # 3. Notch の One-Hot ベクトル化
        notch_onehot = [1.0 if cat == notch_str else 0.0 for cat in self.notch_categories]
        
        # 直前ノッチのOne-hotベクトルを作成 (置換後の prev_notch_str を使用)
        prev_notch_onehot = [1.0 if prev_notch_str == cat else 0.0 for cat in self.prev_notch_categories]
        
        # 4. Limit と Gradient の抽出
        next_limit_flag, next_limit_dist, next_limit_speed = self._extract_limit_info(state_info.get('next_limit_info', ''))
        next_gradient_flag, next_gradient_dist, next_gradient_val = self._extract_gradient_info(state_info.get('next_gradient_info', ''))
        
        # 5. 先行列車・後続列車情報の抽出
        f_exist, f_distance, f_speed = self._extract_forward_info(state_info.get('forward_info', ''))
        b_exist, b_distance, b_speed = self._extract_backward_info(state_info.get('backward_info', ''))

        # 6. 特徴量ベクトルの結合 (34次元：学習スクリプトと完全に同じ順序)
        features = [
            hold_coast, hold_accel, hold_decel,
            prev_duration,
            state_info.get('speed_limit', 0.0),
            state_info.get('current_speed', 0.0),
            state_info.get('dist_to_next_station', 0.0),
            state_info.get('time_to_next_station', 0.0),
            state_info.get('req_stop_dist', 0.0),  # ▼変更4: ここに req_stop_dist を追加
            state_info.get('delay', 0.0),
            state_info.get('current_gradient', 0.0),
            next_limit_flag, next_limit_dist, next_limit_speed,
            next_gradient_flag, next_gradient_dist, next_gradient_val,
            f_exist, f_distance, f_speed,
            b_exist, b_distance, b_speed
        ] + phase_onehot + notch_onehot + prev_notch_onehot
        
        # 7. スケーリングと推論
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        predicted_tensor = self.predict_fn(tf.convert_to_tensor(X_scaled, dtype=tf.float32))
        
        raw_reward = float(predicted_tensor.numpy()[0][0])
        rounded_reward = round(raw_reward * 10) / 10.0
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