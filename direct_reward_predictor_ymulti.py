# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の報酬予測器（LLM蒸留の推論側）。

既存 `direct_reward_predictor2.py` は**一切変更しない**（apex2.py 系がそのまま使い続ける）。
成果物ファイル名も分離してあるので、既存の `direct_reward_model2.h5` 等を上書きすることはない。

推論パイプライン:
    生の状態辞書 → 状態特徴量(reward_features_ymulti) → **モードをルールで決定** → mode one-hot
                 → [正規化状態 | mode one-hot] → 評価NN（ゲート＋HL-Gaussian回帰）→ 報酬

既存版との違い
  * **モード分類NNを持たない。** モードは `reward_features_ymulti.decide_mode()` で決定的に決める。
    既存版はモードNNのargmaxを併用していたが、条件が成立し続ける区間でも
    normal ↔ delay_recovery が反転する問題が実測されている
    （`direct_reward_predictor2._infer_mode` のコメント参照）。プロンプトに書くモード定義と
    実行時の判定を1対1で対応させるため、本系統ではルールに一本化した。
  * モードに `hold_at_station`（駅停車中の発車判断）を含む。

回帰器のヘッドは `direct_reward_manifest_ymulti.json` の `regressor_head` で判別する
（HL-Gaussian / スカラー回帰）。定義は既存と同じ `histogram_loss.py` を共有する。
"""
import os
import json

import numpy as np
import tensorflow as tf
import joblib

import config_ymulti as CFG
import reward_features_ymulti as rf
import histogram_loss as hl


class DirectRewardPredictorYMulti:
    def __init__(self,
                 model_path=CFG.REWARD_MODEL_PATH,
                 scaler_path=CFG.REWARD_SCALER_PATH,
                 gate_path=CFG.REWARD_GATE_PATH,
                 manifest_path=CFG.REWARD_MANIFEST_PATH,
                 quiet=False):
        self.is_loaded = False
        self.model = self.gate_model = self.scaler = None
        self.last_mode = "normal"
        self.head = "scalar"
        self.composition = "hard"
        self.bin_centers = None
        self.state_dim = rf.STATE_DIM
        self._load_manifest(manifest_path, quiet)

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            if not quiet:
                print(f"[報酬NN(ymulti)] {model_path} または {scaler_path} が見つかりません。"
                      f"train_reward_network_ymulti.py で学習してください。")
            return

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)
        self._check_head_matches_model()

        n_state = getattr(self.scaler, "n_features_in_", self.state_dim)
        if n_state != self.state_dim:
            raise ValueError(
                f"スケーラの特徴量数({n_state})が現行の状態特徴量数({self.state_dim})と一致しません。"
                f"reward_features_ymulti.STATE_FEATURE_COLS を変えたら"
                f"train_reward_network_ymulti.py で再学習してください。")
        input_dim = n_state + rf.MODE_DIM

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_dim], dtype=tf.float32)])
        def predict_fn(x):
            return self.model(x, training=False)
        self.predict_fn = predict_fn

        if os.path.exists(gate_path):
            self.gate_model = tf.keras.models.load_model(gate_path, compile=False)

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_dim], dtype=tf.float32)])
            def predict_gate_fn(x):
                return self.gate_model(x, training=False)
            self.predict_gate_fn = predict_gate_fn
        elif not quiet:
            print(f"[報酬NN(ymulti)] {gate_path} が見つかりません。ゲートなし（回帰器のみ）で動作します。")

        self.is_loaded = True
        if not quiet:
            print(f"[報酬NN(ymulti)] ロード完了: ヘッド={self.head} / 合成={self.composition} / "
                  f"状態{self.state_dim}次元 + モード{rf.MODE_DIM}次元")

    # ------------------------------------------------------------ マニフェスト
    def _load_manifest(self, path, quiet):
        if not os.path.exists(path):
            if not quiet:
                print(f"[報酬NN(ymulti)] {path} が無いため、回帰器をスカラー回帰として扱います。")
            return
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        cols = info.get("state_feature_cols")
        if cols and cols != rf.STATE_FEATURE_COLS:
            raise ValueError(
                f"マニフェストの状態特徴量の並びが現行の reward_features_ymulti と一致しません。"
                f"（マニフェスト{len(cols)}列 / 現行{rf.STATE_DIM}列）"
                f"train_reward_network_ymulti.py で再学習してください。")
        head = info.get("regressor_head")
        if not head:
            return
        self.head = head.get("head", "scalar")
        self.composition = head.get("composition", "hard")
        if self.head == "hl_gauss":
            centers = head.get("centers")
            if not centers:
                centers, _ = hl.make_centers(head.get("bins", hl.DEFAULT_BINS),
                                             head.get("guard_bins", hl.DEFAULT_GUARD))
            self.bin_centers = np.asarray(centers, dtype=np.float32)

    def _check_head_matches_model(self):
        """モデルの出力次元とマニフェストのヘッド情報が食い違っていないか検証する。
        片方だけ更新すると報酬が黙って壊れたまま学習が進むため、起動時に必ず突き合わせる。"""
        units = int(self.model.output_shape[-1])
        if self.head == "hl_gauss":
            n = 0 if self.bin_centers is None else len(self.bin_centers)
            if units != n:
                raise ValueError(
                    f"回帰器の出力次元({units})とマニフェストのビン数({n})が一致しません。"
                    f"{CFG.REWARD_MODEL_PATH} と {CFG.REWARD_MANIFEST_PATH} の世代を揃えてください。")
        elif units != 1:
            raise ValueError(
                f"マニフェストはスカラー回帰ヘッドですが、回帰器の出力次元は{units}です。")

    # ------------------------------------------------------------------ 推論
    def predict_reward(self, state_info):
        """生の状態辞書 → 0.0〜1.0 の報酬。"""
        if not self.is_loaded:
            return 0.0
        try:
            mode_str = state_info.get("mode") or rf.decide_mode(state_info)
            self.last_mode = mode_str
            x_state = rf.state_vector(state_info).reshape(1, -1)
            mode_oh = rf.mode_to_onehot(mode_str).reshape(1, -1)
            X = np.hstack([self.scaler.transform(x_state), mode_oh]).astype(np.float32)
            xt = tf.convert_to_tensor(X, dtype=tf.float32)

            out = self.predict_fn(xt).numpy()[0]
            zero_prob = (float(self.predict_gate_fn(xt).numpy()[0][0])
                         if self.gate_model is not None else 0.0)

            if self.head == "hl_gauss":
                reg = float(np.dot(out, self.bin_centers))
                if self.composition == "soft" and self.gate_model is not None:
                    value = (1.0 - zero_prob) * reg
                elif self.gate_model is not None:
                    value = 0.0 if zero_prob >= 0.5 else reg
                else:
                    value = reg
                return float(min(max(value, 0.0), 1.0))

            reg = float(out[0])
            if self.gate_model is not None and zero_prob >= 0.5:
                return 0.0
            return float(min(max(reg, 0.0), 1.0))
        except Exception as e:
            print(f"[推論例外(ymulti)] {e}")
            return 0.0


if __name__ == "__main__":
    p = DirectRewardPredictorYMulti()
    print(f"is_loaded={p.is_loaded}")
