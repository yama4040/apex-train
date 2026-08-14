# -*- coding: utf-8 -*-
"""報酬予測NNの出力ヘッド／損失関数を差し替えた比較用モデルを学習するスクリプト（2026-08-13）

train_reward_network2.py は**一切変更しない**。本スクリプトは幹（128-64-32・BN/Dropout/L2）・
データ・分割・スケーラ・サンプル重みを現行と揃えたうえで、出力ヘッドと損失だけを差し替えた
モデルを学習し、analyze_reward_imbalance.py がそのまま読める命名規約で保存する。

  direct_reward_model2<tag>.h5 / direct_reward_gate2<tag>.h5 / direct_reward_scaler2<tag>.pkl
  ＋ direct_reward_head2<tag>.json（ヘッド種別・ビン中心などのメタ情報。本スクリプト独自）

学習する3系統（--variants で選択）:

  baseline  (tag=_base)  現行と同一。Huber回帰器 ＋ 別学習のゲート分類器。
                         ※ 出荷済み direct_reward_model2.h5 は別のデータ断面で学習された
                           可能性があるため、同一条件で必ず引き直して比較の基準にする。

  hl_gauss  (tag=_hlg)   【提案①】回帰器のみ HL-Gaussian 化。
                         出力を K ビンの softmax にし、教師をラベル中心の打ち切り正規分布に
                         変換して交差エントロピーを取る（Imani & White, ICML 2018）。
                         予測はビン中心の期待値。ゲートは baseline のものをそのまま流用する
                         （＝「回帰の損失だけを変えた」統制比較にするため）。

  zero_atom (tag=_atom)  【提案②】ゼロアトム付き HL。ゲートを廃し単一ヘッドに統合。
                         出力は K+1 ビンの softmax で、ビン0を構造ゼロ専用のアトム、
                         ビン1..K を (0,1] のガウス目標に割り当てる。
                         予測は E[y] = Σ_{i>=1} f_i c_i （ハードルモデルの期待値）。
                         閾値・clip・round がすべて不要になる。

【重要な前提】
  analyze_reward_imbalance.py の冒頭に記録されているとおり、2026-08-13 の調査では
  「幻の中間値」の主因は損失関数ではなく**ラベル側の矛盾**（ノコギリ運転の減点規則を
  LLMがCSVバッチによって適用したりしなかったりしている）と結論づけられており、
  Balanced MSE は全指標で悪化した。①②も同じ理由で効かない可能性がある。
  本スクリプトはその可否を安価に確かめるためのもので、改善を保証するものではない。

使い方:
  python train_reward_heads.py                       # 3系統すべてを学習
  python train_reward_heads.py --variants hl_gauss   # 一部だけ学習
  python train_reward_heads.py --bins 19             # 非0側のビン数を変える（既定10=ラベル刻みと一致）
  python train_reward_heads.py --sigma-ratio 1.0     # σ/ビン幅（既定0.75）
  python train_reward_heads.py --epochs 50           # 動作確認用に短く

  学習後の比較は compare_reward_heads.py で行う。
"""
import os
import sys
import json
import argparse
import datetime

import numpy as np
import joblib
from scipy.special import erf

import matplotlib
matplotlib.use('Agg')  # train_reward_network2.py と同じ理由（qtaggだとSIGABRTで落ちる環境がある）

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import reward_features as rf
import train_reward_network2 as t2   # 読み取り専用で定数・関数を借りる（変更はしない）

RANDOM_STATE = 42
TEST_SIZE = 0.2
ZERO_THRESHOLD = t2.ZERO_THRESHOLD               # 0.05
STOP_PHASE_WEIGHT = t2.STOP_PHASE_WEIGHT         # 3.0
STOP_PHASE_COL = t2.STOP_PHASE_COL
CEILING_LABEL_WEIGHT = t2.CEILING_LABEL_WEIGHT   # 2.5
CEILING_LABEL_THRESHOLD = t2.CEILING_LABEL_THRESHOLD

# 非0ラベルの値域。LLMの出力は 0.1 刻みなので既定ではビン中心がラベル値と一対一に対応する。
NONZERO_LO, NONZERO_HI = 0.1, 1.0

VARIANT_TAGS = {'baseline': '_base', 'hl_gauss': '_hlg', 'zero_atom': '_atom'}


# ------------------------------------------------------------------ ビンと目標分布
def make_centers(k):
    """非0側のビン中心とビン幅を返す。k=10 なら 0.1,0.2,...,1.0（＝ラベル刻みと一致）。"""
    if k < 2:
        raise ValueError('--bins は2以上にしてください。')
    centers = np.linspace(NONZERO_LO, NONZERO_HI, k)
    width = (NONZERO_HI - NONZERO_LO) / (k - 1)
    return centers.astype(np.float64), float(width)


def gaussian_bin_targets(y, centers, width, sigma):
    """ラベル y を、中心 centers・幅 width のビン上の打ち切り正規分布へ変換する。

    Imani & White (2018) の HL-Gaussian の p_i:
        p_i = (F(l_i + w) - F(l_i)) / Z,  F は N(y, sigma^2) のCDF
    サポートは [centers[0]-w/2, centers[-1]+w/2]。Z は打ち切り分の正規化。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    edges = np.concatenate([centers - width / 2.0, [centers[-1] + width / 2.0]])  # (k+1,)
    z = (edges[None, :] - y) / (np.sqrt(2.0) * sigma)
    cdf = 0.5 * (1.0 + erf(z))
    p = np.diff(cdf, axis=1)
    total = p.sum(axis=1, keepdims=True)
    # サポート外・σが極端に小さい等で総和が0に潰れた行は、最近傍ビンへ全質量を置く（安全弁）
    bad = (total[:, 0] <= 1e-12)
    if bad.any():
        p[bad] = 0.0
        p[bad, np.abs(centers[None, :] - y[bad]).argmin(axis=1)] = 1.0
        total[bad] = 1.0
    return (p / total).astype(np.float32)


def expectation(probs, centers):
    """softmax出力 → ビン中心の期待値。"""
    return (np.asarray(probs, dtype=np.float64) @ np.asarray(centers, dtype=np.float64)).astype(np.float32)


# ------------------------------------------------------------------ モデル
def build_trunk(input_dim, out_units, out_activation):
    """train_reward_network2.build_model と同一の幹。出力層だけ差し替える。"""
    l2_reg = l2(1e-4)
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(32, kernel_regularizer=l2_reg), BatchNormalization(), Activation('relu'), Dropout(0.1),
        Dense(out_units, activation=out_activation),
    ])


def make_expected_mae(centers):
    """交差エントロピーで学習するヘッドを、baseline と同じ土俵（予測値のMAE）で監視する指標。

    y_true は目標分布なので、その期待値（≒元のラベル）と予測期待値の差を測る。
    これにより EarlyStopping の基準を全系統で揃えられる。
    """
    c = tf.constant(np.asarray(centers, dtype=np.float32))

    def expected_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(tf.reduce_sum(y_pred * c, axis=-1) -
                                     tf.reduce_sum(y_true * c, axis=-1)))
    return expected_mae


def compile_regressor(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss=Huber(delta=1.0), metrics=['mae'])
    return model


def compile_histogram(model, centers):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=[make_expected_mae(centers)])
    return model


def compile_gate(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def callbacks(monitor):
    return [EarlyStopping(monitor=monitor, patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-6, verbose=0)]


# ------------------------------------------------------------------ サンプル重み
def build_sample_weights(y, is_stop, use_bin_weight, use_ceiling):
    """現行 train_reward_network2 と同一の重み付け（頻度逆数sqrt × 停車フェーズ × 上限ラベル）。"""
    if use_bin_weight:
        w, _ = t2.compute_bin_sample_weights(y.reshape(-1, 1))
    else:
        w = np.ones(len(y), dtype=np.float32)
    w = w * np.where(is_stop, STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    if use_ceiling:
        ceiling = (y >= CEILING_LABEL_THRESHOLD) & is_stop
        w = w * np.where(ceiling, CEILING_LABEL_WEIGHT, 1.0).astype(np.float32)
    return (w / w.mean()).astype(np.float32)


# ------------------------------------------------------------------ 各系統の学習
def train_gate(data, args, tag):
    """ゲート分類器（reward=0.0か否か）。baseline と hl_gauss で共有する。"""
    print(f"\n--- ゲート分類器を学習（tag={tag}）---")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    d = data
    y0_tr = (d['y_tr'] <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    y0_te = (d['y_te'] <= ZERO_THRESHOLD).astype(np.float32).reshape(-1, 1)
    sw = np.where(d['stop_tr'], STOP_PHASE_WEIGHT, 1.0).astype(np.float32)
    sw = sw / sw.mean()
    gate = compile_gate(build_trunk(d['X_tr'].shape[1], 1, 'sigmoid'))
    gate.fit(d['X_tr'], y0_tr, sample_weight=sw,
             validation_data=(d['X_te'], y0_te), epochs=args.epochs, batch_size=64,
             callbacks=callbacks('val_loss'), verbose=args.verbose)
    return gate


def train_baseline(data, args, gate):
    """現行と同一：非0行のみで Huber 回帰。"""
    print("\n--- baseline: Huber回帰器を学習 ---")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    d = data
    nz_tr, nz_te = d['y_tr'] > ZERO_THRESHOLD, d['y_te'] > ZERO_THRESHOLD
    sw = build_sample_weights(d['y_tr'][nz_tr], d['stop_tr'][nz_tr], True, True)
    model = compile_regressor(build_trunk(d['X_tr'].shape[1], 1, 'linear'))
    monitor = 'val_mae' if args.monitor == 'mae' else 'val_loss'
    model.fit(d['X_tr'][nz_tr], d['y_tr'][nz_tr].reshape(-1, 1), sample_weight=sw,
              validation_data=(d['X_te'][nz_te], d['y_te'][nz_te].reshape(-1, 1)),
              epochs=args.epochs, batch_size=64, callbacks=callbacks(monitor), verbose=args.verbose)
    meta = {'head': 'scalar', 'centers': None, 'sigma': None,
            'note': '現行 train_reward_network2.py と同一構成（Huber + ゲート合成）'}
    return model, gate, meta


def train_hl_gauss(data, args, gate):
    """提案①：非0行のみで HL-Gaussian（softmax K ビン ＋ ガウス目標のCE）。"""
    centers, width = make_centers(args.bins)
    sigma = args.sigma_ratio * width
    print(f"\n--- hl_gauss: HL-Gaussian回帰器を学習（K={args.bins} 幅={width:.4f} σ={sigma:.4f}）---")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    d = data
    nz_tr, nz_te = d['y_tr'] > ZERO_THRESHOLD, d['y_te'] > ZERO_THRESHOLD
    P_tr = gaussian_bin_targets(d['y_tr'][nz_tr], centers, width, sigma)
    P_te = gaussian_bin_targets(d['y_te'][nz_te], centers, width, sigma)
    sw = build_sample_weights(d['y_tr'][nz_tr], d['stop_tr'][nz_tr],
                              args.sample_weight != 'none', args.sample_weight != 'none')
    model = compile_histogram(build_trunk(d['X_tr'].shape[1], args.bins, 'softmax'), centers)
    monitor = 'val_expected_mae' if args.monitor == 'mae' else 'val_loss'
    model.fit(d['X_tr'][nz_tr], P_tr, sample_weight=sw,
              validation_data=(d['X_te'][nz_te], P_te),
              epochs=args.epochs, batch_size=64, callbacks=callbacks(monitor), verbose=args.verbose)
    meta = {'head': 'hl_gauss', 'centers': centers.tolist(), 'sigma': sigma, 'width': width,
            'sample_weight': args.sample_weight,
            'note': 'ゲートは baseline と同一のものを共有（回帰の損失だけを変えた統制比較）'}
    return model, gate, meta


def train_zero_atom(data, args):
    """提案②：全行で ゼロアトム＋ガウス目標（softmax K+1 ビンのCE）。ゲートなし。"""
    centers, width = make_centers(args.bins)
    sigma = args.sigma_ratio * width
    all_centers = np.concatenate([[0.0], centers])  # ビン0 = 構造ゼロのアトム
    print(f"\n--- zero_atom: ゼロアトム付きHLを学習（K+1={args.bins + 1} σ={sigma:.4f}）---")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    d = data

    def targets(y):
        P = np.zeros((len(y), args.bins + 1), dtype=np.float32)
        zero = y <= ZERO_THRESHOLD
        P[zero, 0] = 1.0
        if (~zero).any():
            P[~zero, 1:] = gaussian_bin_targets(y[~zero], centers, width, sigma)
        return P

    sw = build_sample_weights(d['y_tr'], d['stop_tr'],
                              args.sample_weight != 'none', args.sample_weight != 'none')
    model = compile_histogram(build_trunk(d['X_tr'].shape[1], args.bins + 1, 'softmax'), all_centers)
    monitor = 'val_expected_mae' if args.monitor == 'mae' else 'val_loss'
    model.fit(d['X_tr'], targets(d['y_tr']), sample_weight=sw,
              validation_data=(d['X_te'], targets(d['y_te'])),
              epochs=args.epochs, batch_size=64, callbacks=callbacks(monitor), verbose=args.verbose)
    meta = {'head': 'zero_atom', 'centers': all_centers.tolist(), 'sigma': sigma, 'width': width,
            'sample_weight': args.sample_weight,
            'note': 'ビン0が構造ゼロのアトム。予測は E[y]=Σ_{i>=1} f_i c_i（閾値・clip・round不要）'}
    return model, None, meta


# ------------------------------------------------------------------ 保存
def save_variant(tag, model, gate, scaler, meta, args, n_train, n_test):
    model.save(f'direct_reward_model2{tag}.h5')
    if gate is not None:
        gate.save(f'direct_reward_gate2{tag}.h5')
    else:
        # 前回の残骸が残っていると analyze_reward_imbalance が誤ってゲート合成してしまう
        stale = f'direct_reward_gate2{tag}.h5'
        if os.path.exists(stale):
            os.remove(stale)
            print(f"[注意] ゲート不要の系統のため {stale} を削除しました。")
    joblib.dump(scaler, f'direct_reward_scaler2{tag}.pkl')
    meta = dict(meta)
    meta.update({'tag': tag, 'bins': args.bins, 'sigma_ratio': args.sigma_ratio,
                 'zero_threshold': ZERO_THRESHOLD, 'state_dim': scaler.n_features_in_,
                 'mode_dim': rf.MODE_DIM, 'n_train': int(n_train), 'n_test': int(n_test),
                 'random_state': RANDOM_STATE, 'monitor': args.monitor,
                 'trained_at': datetime.datetime.now().isoformat(timespec='seconds')})
    with open(f'direct_reward_head2{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"保存: direct_reward_model2{tag}.h5 / "
          f"{'gate2' + tag + '.h5 / ' if gate is not None else ''}"
          f"scaler2{tag}.pkl / head2{tag}.json")


# ------------------------------------------------------------------ データ
def prepare_data(csv_dir):
    """train_reward_network2.main と同一のデータ・分割・スケーラを再現する。"""
    X_state, mode_onehot, y, state_cols = t2.load_and_preprocess_data(csv_dir)
    y = y.flatten().astype(np.float32)
    _, bin_idx = t2.compute_bin_sample_weights(y.reshape(-1, 1))
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=bin_idx)
    scaler = StandardScaler().fit(X_state[idx_tr])
    X_tr = np.hstack([scaler.transform(X_state[idx_tr]), mode_onehot[idx_tr]]).astype(np.float32)
    X_te = np.hstack([scaler.transform(X_state[idx_te]), mode_onehot[idx_te]]).astype(np.float32)
    stop_col = state_cols.index(STOP_PHASE_COL)
    print(f"学習 {len(idx_tr)}行 / テスト {len(idx_te)}行（seed{RANDOM_STATE}・0.1刻み層化）")
    print(f"入力次元 {X_tr.shape[1]} = 状態{X_state.shape[1]} + mode{rf.MODE_DIM}")
    return {'X_tr': X_tr, 'X_te': X_te, 'y_tr': y[idx_tr], 'y_te': y[idx_te],
            'stop_tr': X_state[idx_tr][:, stop_col] >= 0.5,
            'stop_te': X_state[idx_te][:, stop_col] >= 0.5,
            'scaler': scaler, 'n_train': len(idx_tr), 'n_test': len(idx_te)}


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', nargs='*', default=['baseline', 'hl_gauss', 'zero_atom'],
                    choices=list(VARIANT_TAGS), help='学習する系統')
    ap.add_argument('--csv-dir', default='train_reward_csv_direct')
    ap.add_argument('--bins', type=int, default=10,
                    help='非0側のビン数。既定10でビン中心が0.1,0.2,...,1.0（ラベル刻みと一致）')
    ap.add_argument('--sigma-ratio', type=float, default=0.75,
                    help='σ ÷ ビン幅。Imani は「σ=ビン半径」、実務上は0.75前後が定番')
    ap.add_argument('--sample-weight', choices=['same', 'none'], default='same',
                    help='HL系のサンプル重み。same=現行と同一 / '
                         'none=頻度逆数重みと上限ラベル重みを外す（停車フェーズ重みは残す）。'
                         'HL-Gaussは頻度逆数重みを不要にする、という主張の検証用')
    ap.add_argument('--monitor', choices=['mae', 'native'], default='mae',
                    help='EarlyStoppingの基準。mae=予測値のMAEで全系統を揃える（既定） / '
                         'native=各損失のval_loss（baselineは出荷レシピと一致）')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--verbose', type=int, default=1)
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    data = prepare_data(args.csv_dir)
    n_zero = int((data['y_tr'] <= ZERO_THRESHOLD).sum())
    print(f"学習データのラベル0.0: {n_zero}行 / 非0.0: {len(data['y_tr']) - n_zero}行")

    # ゲートは baseline と hl_gauss で共有する（回帰の損失だけを変えた比較にするため）
    gate = None
    if {'baseline', 'hl_gauss'} & set(args.variants):
        gate = train_gate(data, args, '_base')

    for name in args.variants:
        tag = VARIANT_TAGS[name]
        print("\n" + "=" * 78)
        print(f"系統 '{name}'  →  tag='{tag}'")
        print("=" * 78)
        if name == 'baseline':
            model, g, meta = train_baseline(data, args, gate)
        elif name == 'hl_gauss':
            model, g, meta = train_hl_gauss(data, args, gate)
        else:
            model, g, meta = train_zero_atom(data, args)
        save_variant(tag, model, g, data['scaler'], meta, args, data['n_train'], data['n_test'])

    print("\n完了。比較は次で行えます:")
    print("  python compare_reward_heads.py --tags " +
          ' '.join(VARIANT_TAGS[v] for v in args.variants))
    return 0


if __name__ == '__main__':
    sys.exit(main())
