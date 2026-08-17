# -*- coding: utf-8 -*-
"""HL-Gaussian（Histogram Loss with Gaussian target）の共通モジュール（2026-08-17）

Imani & White, "Improving Regression Performance with Distributional Losses" (ICML 2018) の
HL-Gaussian を、報酬予測NNの回帰器に適用するための最小限の道具立て。
学習側（train_reward_network2.py / train_reward_heads.py）と推論側（direct_reward_predictor2.py）
で「ビン中心の定義」が食い違うと報酬が静かにずれるため、ここに一本化する。

考え方:
  スカラー回帰の代わりに、値域を K 個のビンに切った softmax を出力する。
  教師は「ラベルを中心とする打ち切り正規分布」をビンごとに積分した確率 p_i で、
  損失は交差エントロピー -Σ p_i log f_i(x)。予測はビン中心の期待値 Σ f_i(x) c_i。

  ・ℓ2/Huber の勾配ノルムは |f(x)-y| に比例して大きく変動するのに対し、
    HL-Gaussian は Σ|p_i - f_i(x)| ≤ 1 に収まる（論文の命題1）。
    条件付き平均へ縮む力が弱く、ダイナミックレンジが潰れにくい。
  ・ガウス目標が隣接ビンへ質量を分配するため、希薄なラベル同士で統計的強度を共有できる。
    本研究のデータでは 0.6 / 0.7 が特に希薄で、実測で MAE が15〜17%改善した（2026-08-17）。
  ・出力が softmax の期待値なので値域外が原理的に出ない。現行の線形出力は
    テスト行の26.7%が [0.1, 1.0] の外に出ており clip が破綻を隠していた。

【予備ビン(guard)が必須である理由】
  ビン中心をラベルの値域そのもの（0.1〜1.0）に置くと、端のラベルでガウスの裾が
  サポート外に切られ、教師分布の期待値自体がラベルからずれる。実測値:
      guard=0: y=1.0 の教師期待値 0.963 / y=0.1 が 0.137（最大ズレ 0.0369）
      guard=1: y=1.0 の教師期待値 0.998 / y=0.1 が 0.102（最大ズレ 0.0024）
  guard=0 では端のラベルを原理的に当てられず、校正直線の傾きが約0.045下がる。
  既定は guard=1（両端に1ビンずつ余分に置く）。
"""
import numpy as np
from scipy.special import erf

# 非0ラベルの値域。LLMの評価値は 0.0〜1.0 の0.1刻みで、0.0はゲート分類器が担当する。
NONZERO_LO, NONZERO_HI = 0.1, 1.0

# 推奨既定値（docs/報酬NN出力ヘッド比較レポート.html の推奨構成）
DEFAULT_BINS = 19          # 核ビン数。19で幅0.05となりラベルの0.1刻みが全て中心に一致する
DEFAULT_GUARD = 1          # 両端の予備ビン数
DEFAULT_SIGMA_RATIO = 0.75  # σ ÷ ビン幅。Imaniは「σ=ビン半径」、実務上0.75前後が定番


def make_centers(bins=DEFAULT_BINS, guard=DEFAULT_GUARD, lo=NONZERO_LO, hi=NONZERO_HI):
    """ビン中心の配列とビン幅を返す。

    核ビンは [lo, hi] を bins 等分した中心に置き、その外側に guard 個ずつ予備ビンを足す。
    既定（bins=19, guard=1）なら中心は 0.05, 0.10, ..., 1.05 の21個で、幅は0.05。
    """
    if bins < 2:
        raise ValueError('bins は2以上にしてください。')
    if guard < 0:
        raise ValueError('guard は0以上にしてください。')
    width = (hi - lo) / (bins - 1)
    centers = np.linspace(lo - guard * width, hi + guard * width, bins + 2 * guard)
    return centers.astype(np.float64), float(width)


def gaussian_bin_targets(y, centers, width, sigma):
    """ラベル y(N,) → 打ち切り正規分布のビン確率 (N, len(centers))。

    p_i = (F(l_i + w) - F(l_i)) / Z,  F は N(y, sigma^2) のCDF
    サポートは [centers[0]-w/2, centers[-1]+w/2]、Z は打ち切り分の正規化。
    各行の和は 1 になる。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    centers = np.asarray(centers, dtype=np.float64)
    edges = np.concatenate([centers - width / 2.0, [centers[-1] + width / 2.0]])
    cdf = 0.5 * (1.0 + erf((edges[None, :] - y) / (np.sqrt(2.0) * sigma)))
    p = np.diff(cdf, axis=1)
    total = p.sum(axis=1, keepdims=True)
    # サポート外・σが極端に小さい等で総和が0に潰れた行は最近傍ビンへ全質量を置く（安全弁）
    bad = (total[:, 0] <= 1e-12)
    if bad.any():
        p[bad] = 0.0
        p[bad, np.abs(centers[None, :] - y[bad]).argmin(axis=1)] = 1.0
        total[bad] = 1.0
    return (p / total).astype(np.float32)


def expectation(probs, centers):
    """softmax出力 → ビン中心の期待値。学習側・推論側で共通の読み出し。"""
    return np.asarray(probs, dtype=np.float64) @ np.asarray(centers, dtype=np.float64)


def make_expected_mae(centers):
    """交差エントロピーで学習するヘッドを、スカラー回帰と同じ土俵（予測値のMAE）で監視する指標。

    y_true は目標分布なので、その期待値（≒元のラベル）と予測期待値の差を測る。
    EarlyStopping / ReduceLROnPlateau の基準を損失の種類によらず揃えられる。
    """
    import tensorflow as tf
    c = tf.constant(np.asarray(centers, dtype=np.float32))

    def expected_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(tf.reduce_sum(y_pred * c, axis=-1) -
                                     tf.reduce_sum(y_true * c, axis=-1)))
    return expected_mae


def head_manifest(bins=DEFAULT_BINS, guard=DEFAULT_GUARD, sigma_ratio=DEFAULT_SIGMA_RATIO):
    """マニフェストに書き込むヘッド情報。推論側はこれを読んでビン中心を復元する。"""
    centers, width = make_centers(bins, guard)
    return {
        'head': 'hl_gauss',
        'bins': bins,
        'guard_bins': guard,
        'sigma_ratio': sigma_ratio,
        'width': width,
        'sigma': sigma_ratio * width,
        'centers': centers.tolist(),
        'units': len(centers),
        # 推論時の合成方法。soft = (1 - ゲート確率) * 期待値（閾値・clip・roundなし）
        'composition': 'soft',
    }


def scalar_head_manifest():
    """従来のスカラー回帰ヘッド（Huber）のマニフェスト情報。"""
    return {'head': 'scalar', 'units': 1, 'composition': 'hard'}


# --------------------------------------------------- 解析スクリプト向けの共通読み出し
def load_head_info(manifest_path='direct_reward_manifest.json'):
    """マニフェストから回帰器ヘッドの情報を読む。無ければスカラー回帰とみなす（後方互換）。

    direct_reward_predictor2.DirectRewardPredictor と同じ判別規則。解析スクリプトが
    モデルの生出力を「スカラー値」に直すときは read_regressor / compose_reward を使う。
    """
    import json
    import os
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding='utf-8') as f:
                info = json.load(f).get('regressor_head')
            if info:
                return info
        except Exception:
            pass
    return scalar_head_manifest()


def read_regressor(raw_out, head_info):
    """回帰器の生出力 (N, units) → スカラー予測値 (N,)。ヘッド差を吸収する。"""
    raw_out = np.asarray(raw_out)
    if head_info.get('head') == 'hl_gauss':
        centers = head_info.get('centers')
        if not centers:
            centers, _ = make_centers(head_info.get('bins', DEFAULT_BINS),
                                      head_info.get('guard_bins', DEFAULT_GUARD))
        return expectation(raw_out, np.asarray(centers, dtype=np.float64))
    return raw_out.flatten()


def compose_reward(reg, gate_prob, head_info):
    """回帰値とゲート確率 → 実行時と同一の合成結果。

    soft（HL-Gaussian の既定）: (1 - ゲート確率) * 回帰値
    hard（旧スカラー回帰）    : ゲート確率>=0.5 なら 0.0、それ以外は clip(0.1, 1.0)
    """
    reg = np.asarray(reg, dtype=np.float64)
    if gate_prob is None:
        return np.clip(reg, 0.0, 1.0) if head_info.get('composition') == 'soft' \
            else np.clip(reg, 0.1, 1.0)
    gate_prob = np.asarray(gate_prob, dtype=np.float64)
    if head_info.get('composition') == 'soft':
        return np.clip((1.0 - gate_prob) * reg, 0.0, 1.0)
    return np.where(gate_prob >= 0.5, 0.0, np.clip(reg, 0.1, 1.0))
