"""報酬予測器（二重蒸留の推論側・2026-07-25改修 / 2026-08-17 HL-Gaussian対応）

RL実行時のパイプライン:
  状態辞書 → 状態特徴量(reward_features) → モードNN(argmax) → mode one-hot
           → [正規化状態 | mode one-hot] → 評価NN(ゲート+回帰) → 報酬

特徴量エンジニアリングは reward_features（共有モジュール）に一本化し、学習側と一致させる。
モデルが無い場合は0.0（後方互換）。モードNNが無い場合は mode=normal 固定で動作する。

回帰器のヘッドは direct_reward_manifest.json の regressor_head で判別する（2026-08-17）。

  hl_gauss（推奨・現行の学習既定）:
      回帰器の出力は値域ビンのsoftmax。予測はビン中心の期待値で、合成は
          reward = (1 - ゲート確率) * Σ f_i c_i
      ハード閾値・clip・0.1丸めをいずれも使わない。
      ・出力がsoftmaxの期待値なので原理的に値域内（旧スカラー回帰は生出力の26.7%が値域外で、
        clipが破綻を隠していた）
      ・clip下限0.1と0.1丸めが無くなるので、報酬0.1が実質出力されない問題が解消する
      ・ゲート確率が0.5をまたいだ瞬間に報酬が0↔0.5以上へ飛ぶ不連続が無くなる

  scalar（旧構成・マニフェストにヘッド情報が無い場合もこちら）:
      従来どおり「ゲート確率>=0.5なら0.0、それ以外は clip(0.1,1.0) して0.1丸め」。
      過去のモデル資産をそのまま読めるよう後方互換で残している。

戻り値は常に 0.0〜1.0 のスカラーで、environment2.py 側の扱いは変わらない。
"""
import os
import json
import numpy as np
import tensorflow as tf
import joblib

import reward_features as rf
import histogram_loss as hl

# 遅延回復モードの判定閾値[km/h]（設計メモ§19）。
# required_speed は speed_limit で頭打ちになるため、制限との差がこの値以下なら「びたづき」＝
# 定時到達に制限速度近くの力行が必要、とみなす。学習データ上、差が0.5〜3km/hの行は全体の1%しかなく、
# プロンプトの記載条件（required ≥ speed_limit − 3km/h）とは実質同一の判定になる。
DELAY_RECOVERY_PIN_TOL = 0.5
# 遅延回復を適用してよいCBTC現示の余裕[km/h]。現示が制限速度よりこれ以上低い場合は
# 先行列車に速度を抑えられている状況とみなし、遅延回復モードにはしない。
CBTC_MARGIN_FOR_RECOVERY = 5.0


class DirectRewardPredictor:
    def __init__(self,
                 model_path='direct_reward_model2.h5',
                 scaler_path='direct_reward_scaler2.pkl',
                 gate_path='direct_reward_gate2.h5',
                 mode_model_path='mode_model.h5',
                 mode_scaler_path='mode_scaler.pkl',
                 manifest_path='direct_reward_manifest.json'):
        self.is_loaded = False
        self.model = self.gate_model = self.scaler = None
        self.mode_model = self.mode_scaler = None
        self.last_mode = 'normal'
        # 回帰器ヘッド（マニフェストが無ければ旧来のスカラー回帰として扱う＝後方互換）
        self.head = 'scalar'
        self.composition = 'hard'
        self.bin_centers = None
        self._load_head_manifest(manifest_path)
        self.state_dim = len(rf.STATE_FEATURE_COLS)
        # 遅延回復のルール判定で参照する特徴量の位置（_infer_mode で使用）
        self._idx_required_speed = rf.STATE_FEATURE_COLS.index('required_speed')
        self._idx_speed_limit = rf.STATE_FEATURE_COLS.index('speed_limit')
        self._idx_signal_speed = rf.STATE_FEATURE_COLS.index('signal_speed')

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"[Warning] {model_path} または {scaler_path} が見つかりません。報酬は0.0を返します。")
            return

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)
        self._check_head_matches_model()
        self.is_loaded = True
        if self.head == 'hl_gauss':
            print(f"[報酬NN] HL-Gaussianヘッド: ビン{len(self.bin_centers)}個 "
                  f"[{self.bin_centers[0]:.3f}, {self.bin_centers[-1]:.3f}] / 合成={self.composition}")

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

    def _load_head_manifest(self, manifest_path):
        """マニフェストから回帰器のヘッド種別とビン中心を読む。

        学習側（train_reward_network2.py）が書いた regressor_head をそのまま使うことで、
        ビン中心の定義が学習・推論で食い違って報酬が静かにずれる事故を防ぐ。
        """
        if not os.path.exists(manifest_path):
            print(f"[Info] {manifest_path} が無いため、回帰器を旧来のスカラー回帰として扱います。")
            return
        try:
            with open(manifest_path, encoding='utf-8') as f:
                info = json.load(f).get('regressor_head')
        except Exception as e:
            print(f"[Warning] {manifest_path} を読めませんでした（{e}）。スカラー回帰として扱います。")
            return
        if not info:
            # 2026-08-17より前のマニフェストにはヘッド情報が無い＝スカラー回帰
            return
        self.head = info.get('head', 'scalar')
        self.composition = info.get('composition', 'hard')
        if self.head == 'hl_gauss':
            centers = info.get('centers')
            if not centers:
                # 中心が保存されていない場合はビン設定から復元する
                centers, _ = hl.make_centers(info.get('bins', hl.DEFAULT_BINS),
                                             info.get('guard_bins', hl.DEFAULT_GUARD))
                centers = np.asarray(centers)
            self.bin_centers = np.asarray(centers, dtype=np.float32)

    def _check_head_matches_model(self):
        """モデルの出力次元とマニフェストのヘッド情報が食い違っていないか検証する。

        マニフェストだけ更新してモデルを差し替え忘れる（またはその逆）と、報酬が
        黙って壊れたまま学習が進んでしまうため、起動時に必ず突き合わせる。
        """
        units = int(self.model.output_shape[-1])
        if self.head == 'hl_gauss':
            if self.bin_centers is None or units != len(self.bin_centers):
                raise ValueError(
                    f"回帰器の出力次元({units})とマニフェストのビン数"
                    f"({0 if self.bin_centers is None else len(self.bin_centers)})が一致しません。"
                    f"direct_reward_model2.h5 と direct_reward_manifest.json の世代を揃えてください"
                    f"（train_reward_network2.py で再学習すると両方が更新されます）。")
        elif units != 1:
            raise ValueError(
                f"マニフェストはスカラー回帰ヘッドですが、回帰器の出力次元は{units}です。"
                f"direct_reward_manifest.json が古い可能性があります。")

    def _infer_mode(self, x_state_raw):
        """状態特徴量(1,state_dim) → (mode one-hot(1,MODE_DIM), mode文字列)。

        遅延回復モードはルールで決定する（設計メモ§19）。
          required_speed が speed_limit に「びたづき」（＝定時到達には制限速度近くで
          走り続ける必要がある）→ delay_recovery、そうでなければ normal。
          required_speed が制限を下回っている場合はその速度で惰行しても遅延しないため normal でよい。
        anti_mid_stop（先行に塞がれている）はモード分類NNの判定を優先する
        （優先順位: 安全 ＞ anti_mid_stop ＞ delay_recovery ＞ normal）。

        ※以前は3クラスすべてをモード分類NNのargmaxで決めていたが、NNは required_speed 以外の
        　無関係な特徴量（holding_time・ノッチ種別など）にも反応し、条件が成立し続けている区間でも
        　delay_recovery ↔ normal を反転させていた（run 20260804122227 ci3で確認）。
        　ルール化により判定が決定的になり、プロンプトのモード定義と実行時の挙動が一致する。
        """
        # 先にNNで anti_mid_stop かどうかを判定する（先行列車の塞ぎは状態量の組合せで決まるため）
        nn_mode = None
        if self.mode_model is not None:
            x_scaled = self.mode_scaler.transform(x_state_raw)
            probs = self.predict_mode_fn(tf.convert_to_tensor(x_scaled, dtype=tf.float32)).numpy()[0]
            idx = int(np.argmax(probs))
            nn_mode = rf.MODE_CLASSES_ACTIVE[idx] if idx < len(rf.MODE_CLASSES_ACTIVE) else 'normal'
        if nn_mode == 'anti_mid_stop':
            return rf.mode_to_onehot(nn_mode).reshape(1, -1), nn_mode

        # 遅延回復の判定（ルール）
        #  条件1: required_speed が speed_limit にびたづき（定時到達に制限速度近くの力行が必要）
        #  条件2: CBTC現示に余裕がある（先行列車に速度を抑えられていない）
        # 条件2が必要な理由: 先行に接近して現示が下がっている状況では、速度を決めるのは先行であって
        # 遅延回復ではない。実データでも「びたづきかつLLMがnormalと判定した行」は97%が先行あり・
        # 63%がCBTC制限下であり、逆にdelay_recoveryと判定された行は85%がCBTC余裕だった。
        # （なお減速フェーズか否かは判別軸にならない。delay_recovery行の40%は減速フェーズである）
        req = float(x_state_raw[0, self._idx_required_speed])
        lim = float(x_state_raw[0, self._idx_speed_limit])
        sig = float(x_state_raw[0, self._idx_signal_speed])
        pinned = (lim > 0.0 and lim - req <= DELAY_RECOVERY_PIN_TOL)
        cbtc_free = (sig >= lim - CBTC_MARGIN_FOR_RECOVERY)
        mode_str = 'delay_recovery' if (pinned and cbtc_free) else 'normal'
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

            out = self.predict_fn(x_tensor).numpy()[0]
            zero_prob = (float(self.predict_gate_fn(x_tensor).numpy()[0][0])
                         if self.gate_model is not None else 0.0)

            if self.head == 'hl_gauss':
                # ビン中心の期待値。softmaxの重み付き平均なので必ず [centers[0], centers[-1]] 内。
                reg_value = float(np.dot(out, self.bin_centers))
                if self.composition == 'soft' and self.gate_model is not None:
                    # ハードルモデルの期待値。閾値をまたぐ不連続が無く、clip・丸めも不要。
                    value = (1.0 - zero_prob) * reg_value
                elif self.gate_model is not None:
                    value = 0.0 if zero_prob >= 0.5 else reg_value
                else:
                    value = reg_value
                return float(min(max(value, 0.0), 1.0))

            # --- 旧構成（スカラー回帰）: 従来の合成をそのまま維持する ---
            reg_value = float(out[0])
            if self.gate_model is not None:
                if zero_prob >= 0.5:
                    return 0.0
                return round(min(max(reg_value, 0.1), 1.0), 1)
            return round(reg_value, 1)
        except Exception as e:
            print(f"[推論例外発生] {e}")
            return 0.0
