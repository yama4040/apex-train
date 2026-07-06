import numpy as np
import tensorflow as tf
import joblib
import os
import re

class DirectRewardPredictor:
    def __init__(self, model_path='direct_reward_model2.h5', scaler_path='direct_reward_scaler2.pkl'):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
            
            # ▼▼▼ 【修正】入力次元数を 34 から 43 に完全に同期 ▼▼▼
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 43], dtype=tf.float32)])
            def predict_fn(x):
                return self.model(x, training=False)
            self.predict_fn = predict_fn
            
        else:
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。デフォルトの0.0を使用します。")
            self.is_loaded = False
        
        # 特徴量カラムの並び順（train_reward_network2.py と完全一致）
        self.feature_cols = [
            'hold_coast', 'hold_accel', 'hold_decel',
            'prev_notch_duration',
            'speed_limit', 'signal_speed', 'current_speed', 'dist_to_next_station',
            'time_to_next_station', 'req_stop_dist', 'delay', 'current_gradient',
            'margin_speed', 'margin_signal_speed', 'margin_stop_dist',
            'required_speed', 'speed_margin_to_required', 'hunting_score',
            'f_relative_speed', 'b_relative_speed',
            'next_limit_flag', 'next_limit_dist', 'next_limit_speed',
            'next_gradient_flag', 'next_gradient_dist', 'next_gradient_val',
            'f_exist', 'f_distance', 'f_speed',
            'b_exist', 'b_distance', 'b_speed',
            'phase_駅出発直後の加速フェーズ（20秒以内）', 
            'phase_巡航フェーズ（駅間走行中）', 
            'phase_制限速度区間に接近中（500m以内に制限区間在り）', 
            'phase_次駅への減速フェーズ（駅手前400m以内）',
            'phase_駅停車完了（速度0km/h）',
            'current_notch_惰行中', 
            'current_notch_力行（加速）中', 
            'current_notch_ブレーキ（減速）中',
            'prev_notch_惰行', 
            'prev_notch_力行（加速）', 
            'prev_notch_ブレーキ（減速）'
        ]

    def _preprocess_state(self, s):
        """
        environment2.py から渡される辞書データ(s)を 
        学習時と完全に一致する並び順の 43次元 の数値行列 (1, 43) に前処理変換する。
        """
        # 1. カレントノッチのフラグ化
        is_coast = 1.0 if s['current_notch'] == "惰行中" else 0.0
        is_accel = 1.0 if s['current_notch'] == "力行（加速）中" else 0.0
        is_decel = 1.0 if s['current_notch'] == "ブレーキ（減速）中" else 0.0
        
        # 2. 派生特徴量の動的計算
        margin_speed = s['speed_limit'] - s['current_speed']
        margin_signal_speed = s['signal_speed'] - s['current_speed']
        margin_stop_dist = s['dist_to_next_station'] - s['req_stop_dist']

        # 必要速度（environment2.pyがrequired_speed.pyで算出済みの値をそのまま使用）。
        # 現在速度との差が正なら必要速度未達（力行継続が必要）、負なら惰行への移行余地があることを示す。
        required_speed = s['required_speed']
        speed_margin_to_required = s['current_speed'] - required_speed

        # ノコギリ運転スコア化
        is_hunting = (s['holding_time'] < 5.0) and (s['prev_notch_duration'] < 5.0) and (s['current_notch'] != s['prev_notch'])
        hunting_score = max(0.0, 5.0 - s['holding_time']) / 5.0 if is_hunting else 0.0
        
        # テキスト情報のパース
        nl_flag, nl_dist, nl_speed = self._extract_limit_info(s['next_limit_info'])
        ng_flag, ng_dist, ng_val = self._extract_gradient_info(s['next_gradient_info'])
        f_exist, f_dist, f_speed = self._extract_forward_info(s['forward_info'])
        b_exist, b_dist, b_speed = self._extract_backward_info(s['backward_info'])
        
        # 特徴量のクリッピング（NNのスケーリング崩れ防止）
        dist_to_next = min(s['dist_to_next_station'], 2000.0)
        req_stop = min(s['req_stop_dist'], 2000.0)
        f_dist = min(f_dist, 2000.0)
        b_dist = min(b_dist, 2000.0)
        
        # 相対速度 (TTCは削除されました)
        f_rel_speed = s['current_speed'] - f_speed
        b_rel_speed = b_speed - s['current_speed']
        
        # カラム名に合わせた辞書の作成
        features = {
            'hold_coast': s['holding_time'] * is_coast,
            'hold_accel': s['holding_time'] * is_accel,
            'hold_decel': s['holding_time'] * is_decel,
            'prev_notch_duration': s['prev_notch_duration'],
            'speed_limit': s['speed_limit'],
            'signal_speed': s['signal_speed'],
            'current_speed': s['current_speed'],
            'dist_to_next_station': dist_to_next,
            'time_to_next_station': s['time_to_next_station'],
            'req_stop_dist': req_stop,
            'delay': s['delay'],
            'current_gradient': s['current_gradient'],
            'margin_speed': margin_speed,
            'margin_signal_speed': margin_signal_speed,
            'margin_stop_dist': margin_stop_dist,
            'required_speed': required_speed,
            'speed_margin_to_required': speed_margin_to_required,
            'hunting_score': hunting_score,
            'f_relative_speed': f_rel_speed,
            'b_relative_speed': b_rel_speed,
            'next_limit_flag': nl_flag, 'next_limit_dist': nl_dist, 'next_limit_speed': nl_speed,
            'next_gradient_flag': ng_flag, 'next_gradient_dist': ng_dist, 'next_gradient_val': ng_val,
            'f_exist': f_exist, 'f_distance': f_dist, 'f_speed': f_speed,
            'b_exist': b_exist, 'b_distance': b_dist, 'b_speed': b_speed,
            
            # カテゴリ（テキスト）変数のワンホット化
            'phase_駅出発直後の加速フェーズ（20秒以内）': 1.0 if s['phase'] == "駅出発直後の加速フェーズ（20秒以内）" else 0.0,
            'phase_巡航フェーズ（駅間走行中）': 1.0 if s['phase'] == "巡航フェーズ（駅間走行中）" else 0.0,
            'phase_制限速度区間に接近中（500m以内に制限区間在り）': 1.0 if s['phase'] == "制限速度区間に接近中（500m以内に制限区間在り）" else 0.0,
            'phase_次駅への減速フェーズ（駅手前400m以内）': 1.0 if s['phase'] == "次駅への減速フェーズ（駅手前400m以内）" else 0.0,
            'phase_駅停車完了（速度0km/h）': 1.0 if s['phase'] == "駅停車完了（速度0km/h）" else 0.0,
            
            'current_notch_惰行中': is_coast,
            'current_notch_力行（加速）中': is_accel,
            'current_notch_ブレーキ（減速）中': is_decel,
            
            'prev_notch_惰行': 1.0 if s['prev_notch'] == "惰行" else 0.0,
            'prev_notch_力行（加速）': 1.0 if s['prev_notch'] == "力行（加速）" else 0.0,
            'prev_notch_ブレーキ（減速）': 1.0 if s['prev_notch'] == "ブレーキ（減速）" else 0.0
        }
        
        # 厳密な特徴量の順番でソートして1次元化
        X = np.array([features[col] for col in self.feature_cols], dtype=np.float32)
        return X.reshape(1, -1)

    def predict_reward(self, state_info):
        if not self.is_loaded:
            return 0.0
            
        try:
            # 前処理を実行し (1, 43) 行列に変換
            X_raw = self._preprocess_state(state_info)
            # 読み込まれている StandardScaler で正規化
            X_scaled = self.scaler.transform(X_raw)
            # 高速化された Tensor 推論関数へ入力
            preds = self.predict_fn(tf.convert_to_tensor(X_scaled, dtype=tf.float32))
            
            # ▼▼▼ 修正: 従来のLLMと同じく小数点第1位に丸める ▼▼▼
            rounded_reward = round(float(preds.numpy()[0][0]), 1)
            return rounded_reward
        except Exception as e:
            print(f"[推論例外発生] {e}")
            return 0.0

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