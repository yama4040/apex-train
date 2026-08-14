# -*- coding: utf-8 -*-
"""報酬予測NNの出力ヘッド（現行 / 提案① HL-Gaussian / 提案② ゼロアトム）を比較評価する（2026-08-13）

train_reward_network2.py・direct_reward_predictor2.py・analyze_reward_imbalance.py は
**いずれも変更しない**。本スクリプトは analyze_reward_imbalance.py の指標一式をそのまま
再利用（import）し、ヘッド種別の違いを吸収する予測器と、「データ数の多いラベルに
引っ張られていないか」を直接測る追加指標だけを足したもの。

評価の主眼＝多数ラベルへの引っ張られ:

  [analyze_reward_imbalance.py から再利用する指標]
    - balanced-MAE（ビン等重み）と many/medium/few-shot 別MAE
    - 校正直線の傾き slope（1.0が無バイアス／小さいほど中央へ縮む）
    - 予測std ÷ 真値std（1未満なら分散が潰れている）
    - 真値ビン別のバイアスと、出力分布 ÷ 真値分布の比
    - 報酬符号の逆転率・0.5潰れ率・Spearman ρ・順位一致率・モード別bMAE・ゲート性能

  [本スクリプトで追加する指標]
    - 頻度重心への収縮係数 β : bias(v) を (頻度重心 − v) に回帰した傾き。
                               β=0 引っ張られなし / β=1 完全に頻度重心へ潰れている。
    - 頻出ビンへの吸引率      : 真値が少数ラベル(few/medium-shot)の行のうち、
                               予測が多数ラベル(many-shot)のビンへ落ちた割合。
    - 過剰生産比 vs 学習頻度  : (予測件数÷真値件数) と学習件数の順位相関。
                               正に大きいほど「多いラベルほど余計に吐いている」。
    - 出力分布のエントロピー比: H(予測) ÷ H(真値)。1未満なら少数の値へ集中している。
    - 分布距離                : 総変動距離 TV と Wasserstein-1。
    - 0.1丸めの影響           : round(_,1) によるMAE増分（HL系は丸めを外せるため）。

【重要な前提】
  analyze_reward_imbalance.py の冒頭に記録のとおり、2026-08-13 の調査では「幻の中間値」の
  主因は損失関数ではなく**ラベル側の矛盾**（ノコギリ運転の減点規則の適用揺れ）と結論づけられ、
  Balanced MSE は全指標で悪化した。①②も同様に効かない可能性があり、本スクリプトはその
  可否を安価に判定するためのもの。改善が出なかった場合は損失を追わずデータ側を直すこと。

使い方:
  python train_reward_heads.py                                   # 先に3系統を学習
  python compare_reward_heads.py                                 # 既定 --tags _base _hlg _atom
  python compare_reward_heads.py --tags '' _base _hlg _atom      # 出荷モデル('')も含める
  python compare_reward_heads.py --no-plot                       # 図を出さない
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reward_features as rf
import analyze_reward_imbalance as ari   # 読み取り専用で指標一式を借りる（変更はしない）

ZERO_THRESHOLD = ari.ZERO_THRESHOLD
MANY_SHOT, FEW_SHOT = 1000, 100          # ari.evaluate の shot 定義と揃える

# dataviz の検証済みカテゴリ配色（スロット1..3はall-pairsで色差要件を満たす組み合わせ）。
# 4系統以上を並べる場合はスロット4以降を使うが、その範囲の色差保証は無い。
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
GRID_ALPHA = 0.25


# ------------------------------------------------------------------ 予測
def load_head_meta(tag):
    """direct_reward_head2<tag>.json を読む。無ければ現行のスカラー回帰とみなす。"""
    path = f'direct_reward_head2{tag}.json'
    if not os.path.exists(path):
        return {'head': 'scalar', 'centers': None,
                'note': f'{path} が無いため現行のスカラー回帰ヘッドとして扱います。'}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def predict_any(tag, X_state, mode_onehot, idx_te, atom_decision='expect'):
    """ヘッド種別を吸収して、analyze_reward_imbalance.evaluate が受け取る形の予測を作る。

    返す dict のキーは ari.predict_combined と同一:
      raw   : 実行時に environment2 へ渡る値（丸め前）
      round : 0.1丸め後（現行 predict_reward が返す粒度）
      reg   : 「回帰部分の生出力」に相当する量（ゼロ判定を除いた条件付き期待値）
      gate  : ゼロ判定の確率（zero_atom ではアトムの確率）

    atom_decision は zero_atom ヘッドの読み出し方（提案②の設計判断そのもの）:
      'expect' … raw = E[y] = Σ_{i>=1} f_i c_i。完全に連続で閾値が無い。
                 ただし厳密な 0.0 を出すには f_0 がほぼ1である必要があり、
                 ラベル0.0が最多の本データでは 0.0 を出しにくい側に倒れる。
      'gate'   … raw = 0.0 (f_0>=0.5) / E[y|y>0] (それ以外)。現行の実行時意味論と同じ
                 ハード判定。0.0は出しやすいが、閾値をまたぐ不連続が残る。
    """
    import joblib
    import tensorflow as tf

    model_path, scaler_path = f'direct_reward_model2{tag}.h5', f'direct_reward_scaler2{tag}.pkl'
    for p in (model_path, scaler_path):
        if not os.path.exists(p):
            print(f"[skip] tag='{tag}': {p} が見つかりません。")
            return None, None
    meta = load_head_meta(tag)
    head = meta.get('head', 'scalar')

    scaler = joblib.load(scaler_path)
    if getattr(scaler, 'n_features_in_', None) != X_state.shape[1]:
        print(f"[skip] tag='{tag}': スケーラ次元 {scaler.n_features_in_} != 現行特徴量 "
              f"{X_state.shape[1]}（モデル世代が古い。再学習が必要）")
        return None, None
    X = np.hstack([scaler.transform(X_state[idx_te]), mode_onehot[idx_te]]).astype(np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)
    out = model.predict(X, verbose=0)

    if head == 'zero_atom':
        centers = np.asarray(meta['centers'], dtype=np.float64)   # [0.0, 0.1, ..., 1.0]
        probs = np.asarray(out, dtype=np.float64)
        p_zero = probs[:, 0]
        cond = (probs[:, 1:] @ centers[1:]) / np.maximum(1.0 - p_zero, 1e-9)  # E[y | y>0]
        if atom_decision == 'gate':
            raw = np.where(p_zero >= 0.5, 0.0, cond)              # 現行と同じハード判定
        else:
            raw = probs @ centers                                 # E[y]（ハードルモデルの期待値）
        gate_p = p_zero
        reg = cond
    else:
        gate_path = f'direct_reward_gate2{tag}.h5'
        if head == 'hl_gauss':
            centers = np.asarray(meta['centers'], dtype=np.float64)  # [0.1, ..., 1.0]
            reg = np.asarray(out, dtype=np.float64) @ centers        # 期待値。既に[0.1,1.0]に有界
        else:
            reg = np.asarray(out, dtype=np.float64).flatten()
        if os.path.exists(gate_path):
            gate = tf.keras.models.load_model(gate_path, compile=False)
            gate_p = gate.predict(X, verbose=0).flatten()
            # 現行 direct_reward_predictor2.predict_reward と同一の合成
            raw = np.where(gate_p >= 0.5, 0.0, np.clip(reg, 0.1, 1.0))
        else:
            print(f"[注意] tag='{tag}': ゲートが無いため回帰器のみで評価します。")
            gate_p = np.zeros(len(reg))
            raw = np.clip(reg, 0.1, 1.0)

    pred = {'raw': raw.astype(np.float64), 'round': np.round(raw, 1).astype(np.float64),
            'reg': reg.astype(np.float64), 'gate': np.asarray(gate_p, dtype=np.float64)}
    return pred, meta


# ------------------------------------------- 追加指標（多数ラベルへの引っ張られ）
def frequency_pull_metrics(pred, y_te, train_cnt, min_bin_n=10):
    """「データ数の多いラベルに引っ張られていないか」を数値化する。"""
    p_raw, p_round = pred['raw'], pred['round']
    b_true = ari.to_bin(y_te)
    b_pred = ari.to_bin(p_round)
    test_cnt = np.bincount(b_true, minlength=11)
    pred_cnt = np.bincount(b_pred, minlength=11)

    # --- 頻度重心への収縮係数 β ---------------------------------------------
    # bias(v) = α + β・(m − v)。m は学習ラベルの頻度重み付き平均（＝素直な回帰が縮む先）。
    centroid = float((np.arange(11) / 10 * train_cnt).sum() / max(train_cnt.sum(), 1))
    sel = [i for i in range(11) if test_cnt[i] >= min_bin_n]
    v = np.array([i / 10 for i in sel])
    bias = np.array([p_raw[b_true == i].mean() - i / 10 for i in sel])
    w = np.array([test_cnt[i] for i in sel], dtype=np.float64)
    A = np.vstack([centroid - v, np.ones(len(v))]).T
    beta, alpha = np.linalg.lstsq(A * np.sqrt(w)[:, None], bias * np.sqrt(w), rcond=None)[0]

    # --- 頻出ビンへの吸引率 ---------------------------------------------------
    many = np.where(train_cnt > MANY_SHOT)[0]
    rare_rows = ~np.isin(b_true, many)
    if len(many) == 0:
        # 多数ラベルが1つも無い（＝データが小さい）場合は定義できない
        attracted = on_target = float('nan')
    elif rare_rows.sum():
        attracted = float(np.isin(b_pred[rare_rows], many).mean())
        on_target = float((b_pred[rare_rows] == b_true[rare_rows]).mean())
    else:
        attracted = on_target = float('nan')

    # --- 過剰生産比と学習頻度の順位相関 --------------------------------------
    ok = test_cnt > 0
    over = pred_cnt[ok] / test_cnt[ok]
    rho_freq = spearmanr(over, train_cnt[ok])[0] if ok.sum() >= 3 else float('nan')

    # --- 分布の集中度と距離 ---------------------------------------------------
    def entropy(c):
        q = c / max(c.sum(), 1)
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())
    ent_ratio = entropy(pred_cnt) / max(entropy(test_cnt), 1e-9)
    q_pred = pred_cnt / max(pred_cnt.sum(), 1)
    q_true = test_cnt / max(test_cnt.sum(), 1)
    tv = float(0.5 * np.abs(q_pred - q_true).sum())
    w1 = float(np.abs(np.cumsum(q_pred) - np.cumsum(q_true)).sum() * 0.1)

    # --- 0.1丸めの影響 --------------------------------------------------------
    round_cost = float(np.abs(p_round - y_te).mean() - np.abs(p_raw - y_te).mean())

    # --- 真値ビンごとの予測のばらつき（潰れの検出）----------------------------
    bin_std = np.array([p_raw[b_true == i].std() if test_cnt[i] >= min_bin_n else np.nan
                        for i in range(11)])

    return {'beta': float(beta), 'alpha': float(alpha), 'centroid': centroid,
            'attracted': attracted, 'on_target': on_target, 'rho_freq': float(rho_freq),
            'ent_ratio': ent_ratio, 'tv': tv, 'w1': w1, 'round_cost': round_cost,
            'bin_std': bin_std, 'test_cnt': test_cnt, 'pred_cnt': pred_cnt,
            'bias_bins': sel, 'bias_vals': bias}


def report_frequency_pull(results):
    print("\n" + "=" * 78)
    print("⑤ 多数ラベルへの引っ張られ（本比較の主眼）")
    print("=" * 78)
    m0 = results[0]['pull']
    many = [f"{i/10:.1f}" for i in range(11) if results[0]['train_cnt'][i] > MANY_SHOT]
    print(f"学習ラベルの頻度重心 = {m0['centroid']:.3f} / many-shotビン(学習>{MANY_SHOT}件) = {many}")

    head = f"{'指標':<34}" + ''.join(f"{('tag=' + (r['tag'] or 'なし')):>18}" for r in results)
    print("\n" + head)
    print("-" * len(head))

    def row(label, key, fmt='{:.3f}', better='low', note=''):
        vals = [r['pull'][key] for r in results]
        line = f"{label:<34}" + ''.join(f"{fmt.format(v):>18}" for v in vals)
        finite = [v for v in vals if np.isfinite(v)]
        if len(results) >= 2 and len(finite) == len(vals):
            best = min(vals) if better == 'low' else max(vals)
            line += f"   ← 良: {fmt.format(best)}"
        print(line)
        if note:
            print(f"{'':<34}{note}")

    row('収縮係数 β (0が理想)', 'beta', '{:+.3f}',
        note='  bias(v) を (頻度重心−v) に回帰した傾き。1に近いほど頻度重心へ潰れている。')
    row('頻出ビンへの吸引率 (低いほど良)', 'attracted', '{:.2%}',
        note=f'  真値が少数ラベルの行のうち、予測が many-shot ビンへ落ちた割合。')
    row('  同・真値ビン的中率 (高いほど良)', 'on_target', '{:.2%}', better='high')
    row('過剰生産比 vs 学習頻度 ρ (0が理想)', 'rho_freq', '{:+.3f}',
        note='  正に大きいほど「学習件数の多いラベルほど余計に吐いている」。')
    row('出力分布エントロピー比 (1が理想)', 'ent_ratio', '{:.3f}', better='high',
        note='  1未満＝少数の値へ集中。1超＝真値より散らばっている。')
    row('総変動距離 TV (低いほど良)', 'tv', '{:.3f}')
    row('Wasserstein-1 (低いほど良)', 'w1', '{:.4f}')
    row('0.1丸めによるMAE増分', 'round_cost', '{:+.4f}',
        note='  HL系は連続出力なので丸めを外せる。外した場合の伸びしろ。')

    print("\n[真値ビン別の予測の標準偏差（0に近いビン＝その真値で予測が1点に潰れている）]")
    hdr = f"{'真値':>5} {'テスト件数':>9}" + ''.join(f"{('tag=' + (r['tag'] or 'なし')):>18}" for r in results)
    print(hdr)
    for i in range(11):
        if results[0]['pull']['test_cnt'][i] == 0:
            continue
        line = f"{i/10:5.1f} {results[0]['pull']['test_cnt'][i]:9d}"
        for r in results:
            s = r['pull']['bin_std'][i]
            line += f"{('—' if not np.isfinite(s) else f'{s:.3f}'):>18}"
        print(line)


# ------------------------------------------------------------------ 図
def plot_frequency_pull(results, train_cnt, out_path, jp):
    t = (lambda ja, en: ja if jp else en)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(11) / 10

    # (a) 学習件数 vs ビン別バイアス。少数ラベルほどバイアスが大きければ「引っ張られ」。
    ax = axes[0, 0]
    ax.axhline(0, color='0.35', lw=1)
    for i, r in enumerate(results):
        pull = r['pull']
        cnt = np.maximum(train_cnt[pull['bias_bins']], 1)
        ax.plot(cnt, pull['bias_vals'], 'o', ms=8, color=PALETTE[i % len(PALETTE)],
                markeredgecolor='white', markeredgewidth=2,
                label=f"tag='{r['tag'] or 'なし'}'  β={pull['beta']:+.3f}")
    # 真値の注記は、その学習件数における全系統の最上点の上へ置く（点と重ならないように）
    for bi in range(11):
        tops = [r['pull']['bias_vals'][r['pull']['bias_bins'].index(bi)]
                for r in results if bi in r['pull']['bias_bins']]
        if tops:
            ax.annotate(f"{bi/10:.1f}", (max(train_cnt[bi], 1), max(tops)),
                        textcoords='offset points', xytext=(0, 9), fontsize=8,
                        color='0.35', ha='center')
    ax.set_xscale('log')
    ax.set_title(t('学習件数 vs 真値ビン別バイアス（点の注記＝真値）',
                   'Train count vs per-bin bias'))
    ax.set_xlabel(t('そのラベルの学習件数（対数）', 'train count (log)'))
    ax.set_ylabel(t('予測平均 − 真値', 'bias'))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=GRID_ALPHA)

    # (b) 過剰生産比。1.0を超えるビン＝真値より余計に吐いている値。
    ax = axes[0, 1]
    bw = 0.8 / max(len(results), 1)
    never = []          # 「その値を一度も出していない」＝対数軸では棒が描けないので別途印を打つ
    for i, r in enumerate(results):
        pull = r['pull']
        ok = pull['test_cnt'] > 0
        ratio = np.full(11, np.nan)
        ratio[ok] = pull['pred_cnt'][ok] / pull['test_cnt'][ok]
        xs = np.arange(11) + i * bw - 0.4 + bw / 2
        zero = np.where(ratio == 0)[0]
        ratio[zero] = np.nan
        never.append((xs[zero], PALETTE[i % len(PALETTE)]))
        ax.bar(xs, ratio, width=bw * 0.9,
               color=PALETTE[i % len(PALETTE)], label=f"tag='{r['tag'] or 'なし'}'")
    ax.axhline(1.0, color='0.35', lw=1)
    ax.set_yscale('log')
    lo = ax.get_ylim()[0]
    for xs, c in never:
        if len(xs):
            ax.plot(xs, np.full(len(xs), lo), marker='x', ls='none', ms=7, mew=2, color=c)
    if any(len(xs) for xs, _ in never):
        ax.annotate(t('×＝その値を一度も出していない', 'x = value never produced'),
                    xy=(0.02, 0.96), xycoords='axes fraction', va='top',
                    fontsize=8, color='0.35')
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels([f"{i/10:.1f}" for i in range(11)])
    ax.set_title(t('出力の過剰生産比（予測件数 ÷ 真値件数・1.0が理想）',
                   'Over-production ratio (pred / true count)'))
    ax.set_xlabel(t('報酬値', 'reward'))
    ax.set_ylabel(t('比', 'ratio'))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=GRID_ALPHA, axis='y')

    # (c) 少数ラベル行の予測の落ち先。多数ラベルの位置に山が立てば吸引されている。
    ax = axes[1, 0]
    many = np.where(train_cnt > MANY_SHOT)[0]
    for i, r in enumerate(results):
        b_true, b_pred = ari.to_bin(r['y_te']), ari.to_bin(r['pred']['round'])
        rare = ~np.isin(b_true, many)
        h = np.bincount(b_pred[rare], minlength=11) / max(rare.sum(), 1)
        ax.bar(np.arange(11) + i * bw - 0.4 + bw / 2, h, width=bw * 0.9,
               color=PALETTE[i % len(PALETTE)],
               label=f"tag='{r['tag'] or 'なし'}'  " +
                     t(f"吸引率{r['pull']['attracted']:.1%}", f"pull {r['pull']['attracted']:.1%}"))
    for mb in many:
        ax.axvspan(mb - 0.45, mb + 0.45, color='0.85', zorder=0)
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels([f"{i/10:.1f}" for i in range(11)])
    ax.set_title(t('少数ラベル行の予測の落ち先（灰帯＝多数ラベルのビン）',
                   'Where rare-label rows are predicted (gray = many-shot bins)'))
    ax.set_xlabel(t('予測値', 'prediction'))
    ax.set_ylabel(t('割合', 'share'))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=GRID_ALPHA, axis='y')

    # (d) 真値ビン別の予測の標準偏差。0付近＝その真値で予測が1点に潰れている。
    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.plot(x, r['pull']['bin_std'], 'o-', ms=7, lw=2, color=PALETTE[i % len(PALETTE)],
                markeredgecolor='white', markeredgewidth=1.5,
                label=f"tag='{r['tag'] or 'なし'}'")
    ax.set_title(t('真値ビン別の予測の標準偏差（0に近い＝1点に潰れている）',
                   'Std of predictions within each true bin'))
    ax.set_xlabel(t('真値', 'true'))
    ax.set_ylabel(t('予測の標準偏差', 'std of prediction'))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n図を保存しました: {out_path}")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', nargs='*', default=['_base', '_hlg', '_atom'],
                    help="比較するモデルの接尾辞。'' は出荷モデル direct_reward_model2.h5")
    ap.add_argument('--csv-dir', default='train_reward_csv_direct')
    ap.add_argument('--atom-decision', choices=['expect', 'gate', 'both'], default='both',
                    help='提案②の読み出し方。expect=E[y]（連続・閾値なし） / '
                         'gate=f_0>=0.5なら0.0（現行と同じ意味論） / both=両方を並べる（既定）')
    ap.add_argument('--out', default='reward_head_comparison.png',
                    help='analyze_reward_imbalance と同じ4面の図の出力先')
    ap.add_argument('--out-pull', default='reward_head_frequency_pull.png',
                    help='多数ラベルへの引っ張られを見る4面の図の出力先')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    jp = ari.setup_japanese_font()
    df, X_state, mode_onehot, y, state_cols = ari.load_dataset(args.csv_dir)
    modes = df['mode'].values

    label_cnt = ari.report_label_distribution(y, modes)
    idx_tr, idx_te = ari.split_indices(y)
    print(f"\n学習 {len(idx_tr)}行 / テスト {len(idx_te)}行（全系統で同一分割・seed42層化）")
    train_cnt = np.bincount(ari.to_bin(y)[idx_tr], minlength=11)
    y_te, modes_te = y[idx_te], modes[idx_te]

    results = []
    for tag in args.tags:
        head = load_head_meta(tag).get('head', 'scalar')
        if head == 'zero_atom':
            decisions = ['expect', 'gate'] if args.atom_decision == 'both' else [args.atom_decision]
        else:
            decisions = ['expect']
        for dec in decisions:
            pred, meta = predict_any(tag, X_state, mode_onehot, idx_te, dec)
            if pred is None:
                break
            # 同じモデルを2通りの読み出しで並べるため、表示上のタグを分ける
            disp = f'{tag}:{dec}' if head == 'zero_atom' and len(decisions) > 1 else tag
            print(f"\n[ヘッド] tag='{tag or 'なし'}' → {meta.get('head')}"
                  + (f" 読み出し={dec}" if head == 'zero_atom' else '')
                  + (f" / {meta['note']}" if meta.get('note') else ''))
            r = ari.evaluate(disp, pred, y_te, modes_te, train_cnt)
            r['pull'] = frequency_pull_metrics(pred, y_te, train_cnt)
            r['pred'], r['y_te'], r['train_cnt'], r['head'] = pred, y_te, train_cnt, head
            results.append(r)

    if not results:
        print("\n評価できたモデルがありません。先に train_reward_heads.py を実行してください。")
        return 1
    if len(results) >= 2:
        ari.report_summary(results)
    report_frequency_pull(results)

    if not args.no_plot:
        ari.plot_report(results, label_cnt, args.out, jp)
        plot_frequency_pull(results, train_cnt, args.out_pull, jp)

    print("\n" + "=" * 78)
    print("読み方: balanced-MAE・校正傾き slope・収縮係数 β・頻出ビン吸引率 の4つを主に見る。")
    print("        ①②が baseline に対してこれらを改善していなければ、原因は損失ではなく")
    print("        ラベル側の矛盾（analyze_reward_imbalance.py 冒頭の記録）である可能性が高い。")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
