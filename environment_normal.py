"""通常運転モードのみの環境（比較実験用・2026-08-11）。

environment2.Environment を継承し、**報酬予測器だけ**を通常運転モード固定版に差し替える。
それ以外（状態特徴量30次元・time_stepスケーリング・forbidden_action・remaining_time など）は
現行系とまったく同一で、比較が「モードの有無」だけの差になるようにしてある。

QNetwork入力のモードone-hot（26〜30番目のうち27〜30番目）は、environment2.step() が
`self.current_mode = reward_predictor.last_mode` で更新する仕組みのため、
予測器を差し替えるだけで自動的に常に normal（[1,0,0,0]）になる。

※ apex2 系のコード（environment2.py 等）は一切変更していない。
"""
from environment2 import Environment as _Environment2

try:
    from direct_reward_predictor_normal import NormalOnlyRewardPredictor
except ImportError:
    NormalOnlyRewardPredictor = None


class Environment(_Environment2):
    """通常運転モードのみを用いる環境。"""

    def __init__(self, time_step=1.0, load_reward_predictor=True):
        # 親の __init__ で現行系の予測器が生成されるため、生成させずに差し替える
        super().__init__(time_step=time_step, load_reward_predictor=False)
        if load_reward_predictor and NormalOnlyRewardPredictor is not None:
            self.reward_predictor = NormalOnlyRewardPredictor()
        else:
            self.reward_predictor = None
