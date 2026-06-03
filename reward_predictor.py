import numpy as np
import tensorflow as tf
import joblib
import os
import re

class RewardWeightPredictor:
    def __init__(self, model_path='reward_weight_model.h5', scaler_path='reward_weight_scaler.pkl'):
        # モデルが存在する場合のみロード（初回実行時やエラー時のクラッシュを防ぐ）
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
        else:
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。デフォルトの重み(0.33)を使用します。")
            self.is_loaded = False
        
        self.phase_categories = [
            "駅出発直後の加速フェーズ（駅発車20秒以内）", 
            "巡航フェーズ（駅間走行中）", 
            "制限速度接近フェーズ（500m以内に制限速度があり、かつ現在速度が制限速度を超えている）", 
            "次駅への減速フェーズ（駅手前400m以内）"
        ]
        self.notch_categories = ["惰行中", "力行（加速）中", "ブレーキ（減速）中"]

    def predict_weights(self, state_info):
        # モデルが読み込めなかった場合は均等な重みを返す（フォールバック）
        if not self.is_loaded:
            return 0.33, 0.33, 0.33
        
        # 1. One-Hot エンコーディング
        phase_vec = [1.0 if state_info['phase'] == cat else 0.0 for cat in self.phase_categories]
        notch_vec = [1.0 if state_info['notch'] == cat else 0.0 for cat in self.notch_categories]
        
        # 2. テキスト情報のパース（フラグ対応）
        next_limit_flag, next_limit_dist, next_limit_speed = self._extract_limit_info(state_info['next_limit_info'])
        next_gradient_flag, next_gradient_dist, next_gradient_val = self._extract_gradient_info(state_info['next_gradient_info'])
        
        # 3. 特徴量ベクトルの結合 (学習時と順番・次元数を完全一致させる)
        feature_vector = [
            float(state_info['speed_limit']),
            float(state_info['current_speed']),
            float(state_info['dist_to_next_station']),
            float(state_info['delay']),
            float(state_info['current_gradient']),
            next_limit_flag, next_limit_dist, next_limit_speed,
            next_gradient_flag, next_gradient_dist, next_gradient_val
        ] + phase_vec + notch_vec
        
        # 4. スケーリングと推論
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # predict() は使わず、__call__ を使って高速かつ省メモリに推論する
        predicted_tensor = self.model(X_scaled, training=False)
        predicted_weights = predicted_tensor.numpy()[0]
        
        return float(predicted_weights[0]), float(predicted_weights[1]), float(predicted_weights[2])

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
            val = float(match.group(3))
            if match.group(2) == '下り':
                val = -val
            return 1.0, dist, val
        return 0.0, 0.0, 0.0