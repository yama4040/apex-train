"""通常運転モードのみの報酬予測器（比較実験用・2026-08-11）。

8月中間報告の指摘「遅延回復について、通常運転のみと遅延回復モードを導入した場合の
運転曲線を比較したほうが有用性を比較できるのでは」に対応するアブレーション実装。

現行系（apex2 / environment2 / direct_reward_predictor2）との違いは**モードだけ**:
  ・モード判定を行わず、常に mode = "normal" として評価NNに入力する。
  ・したがって遅延回復モード・駅間停車防止モードの基準は一切適用されない。
  ・状態特徴量・ネットワーク構造・報酬スケーリングは現行系と同一（公平な比較のため）。

読み込むモデルは通常運転モードのみで学習した別ファイル（*_normal.*）。
現行系のモデル（direct_reward_model2.h5 等）は読まないため、両者は完全に独立している。

※ apex2 系のコードは一切変更していない。本ファイルは DirectRewardPredictor を継承し、
　 モード推論のみ差し替える。
"""
import numpy as np

import reward_features as rf
from direct_reward_predictor2 import DirectRewardPredictor


class NormalOnlyRewardPredictor(DirectRewardPredictor):
    """モードを常に normal に固定した報酬予測器。"""

    def __init__(self,
                 model_path='direct_reward_model_normal.h5',
                 scaler_path='direct_reward_scaler_normal.pkl',
                 gate_path='direct_reward_gate_normal.h5'):
        # モードNNは使わないので、存在しないパスを渡してロードさせない。
        # マニフェストも専用パスにする（現行系の direct_reward_manifest.json を読むと、
        # 現行系がHL-Gaussianヘッドのとき通常運転モデル=スカラー回帰と食い違ってしまう）。
        super().__init__(model_path=model_path,
                         scaler_path=scaler_path,
                         gate_path=gate_path,
                         mode_model_path='__none__',
                         mode_scaler_path='__none__',
                         manifest_path='direct_reward_manifest_normal.json')
        self.last_mode = 'normal'

    def _infer_mode(self, x_state_raw):
        """常に normal を返す（モード判定を行わない）。"""
        return rf.mode_to_onehot('normal').reshape(1, -1), 'normal'
