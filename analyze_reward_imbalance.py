# -*- coding: utf-8 -*-
"""報酬予測NNの「ラベル不均衡による予測バイアス」を診断・比較するスクリプト（2026-08-13）

train_reward_network2.py とは独立の読み取り専用ツール（reward_features のみに依存）。
学習側の構成を変えたとき、新旧のモデルを同一のテスト分割で並べて比較する用途。

背景:
  LLMが付けた報酬ラベルは 0.0(32%) と 1.0(26%) に山があり中央(0.6〜0.7)が1%未満というU字型で、
  不均衡比は約38倍ある。この分布のまま素直に回帰すると予測が学習平均へ縮み（中央回帰）、
  「LLMがほとんど付けない中間値」を高頻度で出力するようになる（実測: 予測0.7が真値の約8倍）。

  ただし2026-08-13の調査で、この中間値出力の主因は損失関数ではなく**ラベル側の矛盾**と判明した。
  全体では近傍ラベルの標準偏差は半径0.2以下で0.000（ラベルは特徴量から完全に決まる）なのに、
  幻の中間値が出る領域だけは半径0.03（ほぼ同一入力）でも0.20が残り、高低を分ける特徴量も無い
  （最大効果量0.027σ）。原因は「ノコギリ運転なら上限0点」という規則をLLMが適用したりしなかったり
  していることで、CSVバッチ別に適用率が77.4%(19ファイル)と16.6%(26ファイル)へ二極化している。
  → 損失関数の工夫でこの誤差は減らない（Balanced MSE を試した結果は全指標で悪化した）。
  データ側の矛盾を解消してから手法を評価すること。

測る指標:
  [不均衡回帰の一般指標 / Yang et al., ICML 2021 のプロトコル準拠]
    - MAE（件数重み）と balanced-MAE（ビン等重み）、many/medium/few-shot 別MAE
    - 真値ビン別のバイアス、校正直線の傾き（1.0で無バイアス、小さいほど中央回帰）
    - 予測の標準偏差 / 真値の標準偏差（1未満なら分散が潰れている）
  [本研究固有・RL側への影響]
    - 報酬符号の逆転率: environment2 は reward=(llm_reward-0.5)*scale なので 0.5 が褒め/罰の境界
    - 0.5潰れ率: 予測が0.1丸めでちょうど0.5になる＝報酬0＝学習信号が消える行の割合
    - 順序保存: Spearman ρ、真値差0.2以上のペアの順位一致率（RLでは絶対値より順位が効く）
    - モード別の balanced-MAE（delay_recovery が normal の写像に引かれていないか）

使い方:
  python analyze_reward_imbalance.py                     # 本番モデル(direct_reward_model2.h5)を診断
  python analyze_reward_imbalance.py --tags '' _new      # 2つのモデルを同一分割で並べて比較
  python analyze_reward_imbalance.py --noise-check       # ラベルの既約ノイズ（改善余地）も測る

ファイル名の規約:
  --tags に渡した接尾辞 <tag> ごとに、以下の3ファイルを読む（<tag>='' が本番モデル）。
  比較したいモデルはこの命名で保存しておくこと。
    direct_reward_model2<tag>.h5 / direct_reward_gate2<tag>.h5 / direct_reward_scaler2<tag>.pkl
  ゲートが無い場合は回帰器のみで評価する（合成せず clip(0.1,1.0) のみ）。
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 学習スクリプトと同じ理由（Qtのxcb初期化失敗でSIGABRTするのを避ける）
import matplotlib.pyplot as plt
from matplotlib import font_manager

import joblib
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr

import reward_features as rf

# train_reward_network2.py と同じ定数（分割・ゲート閾値を一致させる）
ZERO_THRESHOLD = 0.05
RANDOM_STATE = 42
TEST_SIZE = 0.2
BINS = np.arange(11)  # 0.0〜1.0 の0.1刻み（ラベルは実際にこの11値しか取らない）

_JP_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
]


def setup_japanese_font():
    """日本語フォントを matplotlib に登録する。見つからなければ False（英語ラベルにする）。"""
    for path in _JP_FONT_FILES:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
        return True
    return False


# ------------------------------------------------------------------ データ
def load_dataset(csv_dir):
    """train_reward_network2.load_and_preprocess_data と同一の読み込み。"""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    df_list, skipped, empty = [], [], []
    for file in csv_files:
        temp = pd.read_csv(file, encoding='utf-8-sig')
        if 'required_speed' not in temp.columns:
            skipped.append(os.path.basename(file))
            continue
        if len(temp) == 0:
            empty.append(os.path.basename(file))
            continue
        df_list.append(temp)
    df = pd.concat(df_list, ignore_index=True)
    if 'mode' not in df.columns:
        df['mode'] = 'normal'
    df['mode'] = df['mode'].fillna('normal').replace('', 'normal')
    print(f"CSV {len(csv_files)}件（旧形式除外 {len(skipped)}件 / 空 {len(empty)}件）→ 合計 {len(df)} 行")

    X_state, state_cols = rf.build_state_matrix(df)
    mode_onehot = np.stack([rf.mode_to_onehot(m) for m in df['mode']]).astype(np.float32)
    y = df['reward'].astype(np.float32).values
    return df, X_state, mode_onehot, y, state_cols


def to_bin(y):
    return np.clip(np.round(np.asarray(y).flatten() * 10).astype(np.int32), 0, 10)


def split_indices(y):
    """学習時と同一の層化分割（seed42・0.1刻みビンで層化）を再現する。"""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=to_bin(y))


# ------------------------------------------------------ ①データセットの不均衡
def report_label_distribution(y, modes):
    print("\n" + "=" * 78)
    print("① ラベル分布（学習データ全体・0.1刻み）")
    print("=" * 78)
    b = to_bin(y)
    cnt = np.bincount(b, minlength=11)
    tot = cnt.sum()
    print(f"{'label':>6} {'count':>8} {'ratio':>8}  bar")
    for i in range(11):
        print(f"{i/10:6.1f} {cnt[i]:8d} {cnt[i]/tot:7.2%}  " + '#' * int(60 * cnt[i] / max(cnt.max(), 1)))
    nz = cnt[cnt > 0]
    print(f"不均衡比（最多/最少）: {cnt.max() / max(nz.min(), 1):.0f} 倍 / "
          f"平均 {y.mean():.3f} 中央値 {np.median(y):.3f} 標準偏差 {y.std():.3f}")
    print("→ 単峰＋ロングテールではなく『両端に山・中央が谷』のU字型。少数派は中央(0.6〜0.7)側にある。")

    print("\n--- モード別（モード×ラベルの同時分布の偏り）---")
    for m in rf.MODE_CLASSES:
        sel = (modes == m)
        if sel.sum() == 0:
            continue
        c = np.bincount(b[sel], minlength=11)
        print(f"[{m:14s}] n={sel.sum():6d} : " + ' '.join(f"{i/10:.1f}:{c[i]}" for i in range(11)))
    return cnt


def report_label_noise(X_state, modes, y, scaler_path):
    """同一・近傍の入力に対するラベルのばらつき＝既約誤差（ベイズ誤差）の目安。
    これが balanced-MAE に近ければ手法改善の余地は小さい、という判断材料になる。"""
    from sklearn.neighbors import NearestNeighbors
    print("\n" + "=" * 78)
    print("② ラベルの既約ノイズ（改善余地があるかの判定）")
    print("=" * 78)
    if not os.path.exists(scaler_path):
        print(f"[skip] {scaler_path} が無いため実行できません。")
        return
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X_state).astype(np.float32)
    mode_id = np.array([rf.onehot_index(m) for m in modes])

    key = pd.DataFrame(np.round(Xs, 4)).astype(str).agg('|'.join, axis=1) + '|m' + pd.Series(mode_id).astype(str)
    d = pd.DataFrame({'k': key, 'y': y})
    stats = d.groupby('k')['y'].agg(['size', 'std', 'min', 'max'])
    dup = stats[stats['size'] >= 2]
    print(f"入力が完全一致する行グループ: {len(dup)}組 / 対象 {int(dup['size'].sum())}行 "
          f"({dup['size'].sum()/len(y):.1%})")
    if len(dup):
        print(f"  グループ内ラベル標準偏差の平均 = {dup['std'].mean():.4f}（全体のラベルstd={y.std():.4f}）")
        print(f"  ラベルが完全一致するグループ   = {(dup['std'] == 0).mean():.1%}")
        sub = d[d['k'].isin(dup.index)]
        m = sub.groupby('k')['y'].transform('mean')
        print(f"  → 完全一致グループ内での既約MAE = {np.abs(sub['y'] - m).mean():.4f}")

    rng = np.random.default_rng(0)
    sample = rng.choice(len(Xs), size=min(8000, len(Xs)), replace=False)
    nn = NearestNeighbors(n_neighbors=11).fit(Xs)
    dist, ind = nn.kneighbors(Xs[sample])
    res = []
    for i in range(len(sample)):
        sel = (mode_id[ind[i]] == mode_id[sample[i]]) & (dist[i] <= 0.5)
        if sel.sum() >= 4:
            lab = y[ind[i]][sel]
            res.append(np.abs(lab - lab.mean()).mean())
    if res:
        print(f"特徴量近傍(k<=10・同一モード・距離<=0.5)での既約MAE = {np.mean(res):.4f}（{len(res)}点で評価）")
    print("→ この値より balanced-MAE が明確に大きければ、残差は学習側のバイアスであり改善余地がある。")


# ------------------------------------------------------------ ②モデルの診断
def predict_combined(tag, X_state, mode_onehot, idx_te):
    """推論時（direct_reward_predictor2.predict_reward）と同一の合成ロジックで予測を作る。"""
    model_path = f'direct_reward_model2{tag}.h5'
    gate_path = f'direct_reward_gate2{tag}.h5'
    scaler_path = f'direct_reward_scaler2{tag}.pkl'
    for p in (model_path, scaler_path):
        if not os.path.exists(p):
            print(f"[skip] tag='{tag}': {p} が見つかりません。")
            return None
    import tensorflow as tf

    scaler = joblib.load(scaler_path)
    if getattr(scaler, 'n_features_in_', None) != X_state.shape[1]:
        print(f"[skip] tag='{tag}': スケーラ次元 {scaler.n_features_in_} != 現行特徴量 {X_state.shape[1]}"
              f"（モデル世代が古い。再学習が必要）")
        return None
    X = np.hstack([scaler.transform(X_state[idx_te]), mode_onehot[idx_te]]).astype(np.float32)

    # 回帰器のヘッド（スカラー回帰 / HL-Gaussian）を判別する。
    # 判別規則・読み出し・合成は histogram_loss に一本化してあり、
    # direct_reward_predictor2 の実行時挙動と一致する。
    import histogram_loss as hl
    import json as _json
    reg = tf.keras.models.load_model(model_path, compile=False)
    units = int(reg.output_shape[-1])

    # 探索順: 比較用ヘッドjson（train_reward_heads.py）→ 本番マニフェスト（tag=''のみ）→ 出力次元から推定
    head_info = None
    head_json = f'direct_reward_head2{tag}.json'
    if os.path.exists(head_json):
        head_info = _json.load(open(head_json, encoding='utf-8'))
    elif tag == '':
        head_info = hl.load_head_info()
    if head_info is None or units != head_info.get('units', 1 if head_info.get('head') == 'scalar'
                                                   else len(head_info.get('centers') or [])):
        if units == 1:
            head_info = hl.scalar_head_manifest()   # 出力1次元なら旧来のスカラー回帰と断定できる
        else:
            print(f"[skip] tag='{tag}': 回帰器の出力次元が {units} ですが、ビン中心を書いた"
                  f" '{head_json}' がありません。ヘッドを特定できないため評価できません。")
            return None
    if head_info.get('head') != 'scalar':
        print(f"[情報] tag='{tag}': 回帰器ヘッド={head_info['head']} / "
              f"合成={head_info.get('composition', 'hard')} / ビン{units}個")
    reg_p = hl.read_regressor(reg.predict(X, verbose=0), head_info)
    if os.path.exists(gate_path):
        gate = tf.keras.models.load_model(gate_path, compile=False)
        gate_p = gate.predict(X, verbose=0).flatten()
        combined = hl.compose_reward(reg_p, gate_p, head_info)
    else:
        print(f"[注意] tag='{tag}': ゲートが無いため回帰器のみで評価します。")
        gate_p = np.zeros(len(reg_p))
        combined = hl.compose_reward(reg_p, None, head_info)
    # environment2 が実際に受け取る値は0.1丸め後（predict_reward の round(...,1)）
    return {'raw': combined, 'round': np.round(combined, 1), 'reg': reg_p, 'gate': gate_p}


def evaluate(tag, pred, y_te, modes_te, train_cnt):
    """1モデル分の指標をまとめて算出・表示し、サマリ用のdictを返す。"""
    p, pr = pred['raw'], pred['round']
    b = to_bin(y_te)
    print("\n" + "=" * 78)
    print(f"③ モデル診断  tag='{tag}'  (direct_reward_model2{tag}.h5)")
    print("=" * 78)

    # --- 真値ビン別のバイアス（中央回帰の直接確認）---
    print(f"{'true':>5} {'n':>6} {'pred平均':>9} {'pred中央':>9} {'bias':>8} {'MAE':>7}")
    bin_mean, bin_mae, bin_true, bin_n = [], [], [], []
    for i in range(11):
        m = b == i
        if m.sum() == 0:
            continue
        print(f"{i/10:5.1f} {m.sum():6d} {p[m].mean():9.3f} {np.median(p[m]):9.3f} "
              f"{p[m].mean()-i/10:+8.3f} {np.abs(pr[m]-y_te[m]).mean():7.3f}")
        bin_mean.append(p[m].mean()); bin_mae.append(np.abs(pr[m] - y_te[m]).mean())
        bin_true.append(i / 10); bin_n.append(int(m.sum()))
    bin_mean, bin_mae, bin_true = np.array(bin_mean), np.array(bin_mae), np.array(bin_true)

    # 校正直線の傾き（非0ラベル・1.0なら無バイアス、小さいほど中央へ縮む）
    sel = bin_true > ZERO_THRESHOLD
    A = np.vstack([bin_true[sel], np.ones(sel.sum())]).T
    slope, intercept = np.linalg.lstsq(A, bin_mean[sel], rcond=None)[0]

    mae = np.abs(pr - y_te).mean()
    bmae = bin_mae.mean()
    shots = {'many': train_cnt > 1000, 'medium': (train_cnt >= 100) & (train_cnt <= 1000),
             'few': (train_cnt > 0) & (train_cnt < 100)}
    shot_mae = {}
    for name, s in shots.items():
        m = np.isin(b, np.where(s)[0])
        shot_mae[name] = np.abs(pr[m] - y_te[m]).mean() if m.sum() else float('nan')

    print(f"\n[不均衡回帰の指標]")
    print(f"  MAE(件数重み)          = {mae:.4f}")
    print(f"  balanced-MAE(ビン等重み)= {bmae:.4f}   ← 少数ラベルを平等に見た誤差（主指標）")
    for name, s in shots.items():
        bins = [round(x/10, 1) for x in np.where(s)[0]]
        print(f"  {name:6s}-shot MAE = {shot_mae[name]:.4f}  bins={bins}")
    print(f"  校正直線の傾き slope   = {slope:.3f} (intercept={intercept:+.3f})  ← 1.0が無バイアス")
    nz = y_te > ZERO_THRESHOLD
    std_ratio = p[nz].std() / y_te[nz].std()
    print(f"  予測std/真値std(非0)   = {std_ratio:.3f}  ← 1未満なら分散が潰れている")

    # 回帰器の生出力レンジ。値域外が多いと推論時のclipが破綻を隠し、
    # clip前で計算される学習中のval_maeが実運用の値と乖離する（エポック選択が当てにならない）。
    reg = pred['reg']
    oor = ((reg < 0.1) | (reg > 1.0)).mean()
    print(f"  回帰器の生出力(clip前)  = [{reg.min():+.3f}, {reg.max():+.3f}] 値域外={oor:.1%}"
          + ("   ← ★clipが破綻を隠している。出力ヘッドの有界化を検討" if oor > 0.05 else ""))

    # --- 出力分布（中間値の吐き出しが起きていないか）---
    print(f"\n[出力値の分布 vs 真値の分布（0.1丸め後）]")
    tc = np.bincount(b, minlength=11)
    pc = np.bincount(to_bin(pr), minlength=11)
    print(f"{'値':>5} {'真値':>7} {'予測':>7} {'比':>7}")
    for i in range(11):
        ratio = pc[i] / tc[i] if tc[i] else float('nan')
        print(f"{i/10:5.1f} {tc[i]:7d} {pc[i]:7d} {ratio:7.2f}")

    # --- RL側への影響 ---
    tr_sign = np.sign(np.round(y_te - 0.5, 6))
    pr_sign = np.sign(np.round(pr - 0.5, 6))
    flip = (tr_sign * pr_sign < 0)
    dead = (pr == 0.5) & (y_te != 0.5)   # 報酬が0になり学習信号が消える行
    print(f"\n[RL側への影響: reward=(llm-0.5)*scale]")
    print(f"  報酬符号の逆転率        = {flip.mean():.2%} "
          f"(罰→褒め {int(((tr_sign<0)&(pr_sign>0)).sum())} / 褒め→罰 {int(((tr_sign>0)&(pr_sign<0)).sum())})")
    print(f"  0.5潰れ率(報酬0で信号消失)= {dead.mean():.2%}")
    print(f"  報酬std 予測/真値        = {(pr-0.5).std()/(y_te-0.5).std():.3f}（褒め罰のコントラスト）")

    rho = spearmanr(pr, y_te)[0]
    r = pearsonr(pr, y_te)[0]
    rng = np.random.default_rng(0)
    i1, i2 = rng.integers(0, len(y_te), 200000), rng.integers(0, len(y_te), 200000)
    m = np.abs(y_te[i1] - y_te[i2]) >= 0.2 - 1e-6
    a, c = i1[m], i2[m]
    conc = (np.sign(y_te[a] - y_te[c]) == np.sign(pr[a] - pr[c])).mean()
    print(f"  Spearman ρ={rho:.4f} / Pearson r={r:.4f} / 真値差0.2以上のペア順位一致={conc:.2%}")

    # --- モード別 ---
    print(f"\n[モード別]")
    mode_bmae = {}
    for name in rf.MODE_CLASSES:
        sm = modes_te == name
        if sm.sum() == 0:
            continue
        per = [np.abs(pr[sm & (b == i)] - y_te[sm & (b == i)]).mean() for i in range(11) if (sm & (b == i)).sum() >= 10]
        mode_bmae[name] = np.mean(per)
        bias_str = ' '.join(f"{i/10:.1f}:{np.mean(pr[sm & (b == i)] - i/10):+.2f}"
                            for i in range(11) if (sm & (b == i)).sum() >= 10)
        print(f"  [{name:14s}] n={sm.sum():5d} MAE={np.abs(pr[sm]-y_te[sm]).mean():.4f} "
              f"bMAE={mode_bmae[name]:.4f}")
        print(f"      ビン別bias: {bias_str}")

    # --- ゲート ---
    tz, pz = y_te <= ZERO_THRESHOLD, pred['gate'] >= 0.5
    tp, fn, fp = int((tz & pz).sum()), int((tz & ~pz).sum()), int((~tz & pz).sum())
    print(f"\n[ゲート分類器] 再現率(0→0)={tp/max(tp+fn,1):.3f} 適合率={tp/max(tp+fp,1):.3f} "
          f"誤って0にした非0行={fp}件")

    return {'tag': tag, 'mae': mae, 'bmae': bmae, 'slope': slope, 'std_ratio': std_ratio,
            'flip': flip.mean(), 'dead': dead.mean(), 'rho': rho, 'conc': conc,
            'shot': shot_mae, 'mode_bmae': mode_bmae,
            'bin_true': bin_true, 'bin_mean': bin_mean, 'bin_mae': bin_mae,
            'pred_hist': pc, 'true_hist': tc}


def report_summary(results):
    print("\n" + "=" * 78)
    print("④ 手法の比較サマリ（テスト分割は全手法で同一・seed42層化）")
    print("=" * 78)
    head = f"{'指標':<28}" + ''.join(f"{('tag=' + (r['tag'] or 'なし')):>18}" for r in results)
    print(head)
    print("-" * len(head))

    def row(label, key, fmt='{:.4f}', better='low', getter=None):
        vals = [(getter(r) if getter else r[key]) for r in results]
        line = f"{label:<28}" + ''.join(f"{fmt.format(v):>18}" for v in vals)
        if len(vals) >= 2 and all(np.isfinite(vals)):
            best = min(vals) if better == 'low' else max(vals)
            line += f"   ← 良: {fmt.format(best)}"
        print(line)

    row('MAE (件数重み)', 'mae')
    row('balanced-MAE ★主指標', 'bmae')
    row('  many-shot MAE', None, getter=lambda r: r['shot']['many'])
    row('  medium-shot MAE', None, getter=lambda r: r['shot']['medium'])
    row('校正の傾き slope (1.0が理想)', 'slope', '{:.3f}', better='high')
    row('予測std/真値std (1.0が理想)', 'std_ratio', '{:.3f}', better='high')
    row('報酬符号の逆転率', 'flip', '{:.2%}')
    row('0.5潰れ率(信号消失)', 'dead', '{:.2%}')
    row('Spearman ρ', 'rho', '{:.4f}', better='high')
    row('順位一致率(真値差0.2以上)', 'conc', '{:.2%}', better='high')
    for m in rf.MODE_CLASSES:
        if all(m in r['mode_bmae'] for r in results):
            row(f'bMAE [{m}]', None, getter=lambda r, mm=m: r['mode_bmae'][mm])


def plot_report(results, label_cnt, out_path, jp):
    t = (lambda ja, en: ja if jp else en)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.arange(len(results)))
    x = np.arange(11) / 10

    ax = axes[0, 0]
    n_series = 1 + len(results)                 # 真値 + 各モデル
    bw = 0.085 / n_series                       # ビン間隔0.1に収まる幅で横並びにする
    base = x - 0.085 / 2 + bw / 2
    ax.bar(base, label_cnt, width=bw, color='0.4', label=t('LLMラベル(真値)', 'LLM label'))
    for i, r in enumerate(results):
        # 予測はテスト分割上の件数なので、真値ヒストグラムと同じ総数にスケールして重ねる
        scaled = r['pred_hist'] * label_cnt.sum() / max(r['true_hist'].sum(), 1)
        ax.bar(base + (i + 1) * bw, scaled, width=bw, color=colors[i], alpha=0.9,
               label=f"pred tag='{r['tag'] or 'なし'}'")
    ax.set_yscale('log')
    ax.set_title(t('ラベル分布と予測分布（対数軸・予測は全体件数にスケール）',
                   'Label vs prediction distribution (log)'))
    ax.set_xlabel(t('報酬値', 'reward')); ax.set_ylabel(t('件数', 'count')); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label=t('理想（無バイアス）', 'ideal'))
    for i, r in enumerate(results):
        ax.plot(r['bin_true'], r['bin_mean'], 'o-', color=colors[i],
                label=f"tag='{r['tag'] or 'なし'}' slope={r['slope']:.3f}")
    ax.set_title(t('校正曲線：真値ビン別の予測平均（傾き<1＝中央回帰）',
                   'Calibration: mean prediction per true bin'))
    ax.set_xlabel(t('真値', 'true')); ax.set_ylabel(t('予測平均', 'mean prediction'))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    w = 0.8 / max(len(results), 1)
    for i, r in enumerate(results):
        ax.bar(r['bin_true'] * 10 + i * w - 0.4, r['bin_mae'], width=w, color=colors[i],
               label=f"tag='{r['tag'] or 'なし'}' bMAE={r['bmae']:.4f}")
    ax.set_xticks(np.arange(11)); ax.set_xticklabels([f"{i/10:.1f}" for i in range(11)])
    ax.set_title(t('真値ビン別MAE（平均＝balanced-MAE）', 'Per-bin MAE'))
    ax.set_xlabel(t('真値', 'true')); ax.set_ylabel('MAE'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.bar(r['bin_true'] * 10 + i * w - 0.4, r['bin_mean'] - r['bin_true'], width=w, color=colors[i],
               label=f"tag='{r['tag'] or 'なし'}'")
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(np.arange(11)); ax.set_xticklabels([f"{i/10:.1f}" for i in range(11)])
    ax.set_title(t('真値ビン別バイアス（＋は過大評価・−は過小評価）', 'Per-bin bias'))
    ax.set_xlabel(t('真値', 'true')); ax.set_ylabel(t('予測平均 − 真値', 'bias'))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n図を保存しました: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', nargs='*', default=[''],
                    help="比較するモデルの接尾辞。例: --tags '' _new （空文字は本番モデル）")
    ap.add_argument('--csv-dir', default='train_reward_csv_direct')
    ap.add_argument('--noise-check', action='store_true',
                    help="ラベルの既約ノイズ（ベイズ誤差の目安）も測る（数分かかる）")
    ap.add_argument('--out', default='reward_imbalance_report.png')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    jp = setup_japanese_font()
    df, X_state, mode_onehot, y, state_cols = load_dataset(args.csv_dir)
    modes = df['mode'].values

    label_cnt = report_label_distribution(y, modes)
    idx_tr, idx_te = split_indices(y)
    print(f"\n学習 {len(idx_tr)}行 / テスト {len(idx_te)}行（学習時と同一の分割を再現）")
    train_cnt = np.bincount(to_bin(y)[idx_tr], minlength=11)

    if args.noise_check:
        report_label_noise(X_state, modes, y, f'direct_reward_scaler2{args.tags[0]}.pkl')

    y_te, modes_te = y[idx_te], modes[idx_te]
    results = []
    for tag in args.tags:
        pred = predict_combined(tag, X_state, mode_onehot, idx_te)
        if pred is not None:
            results.append(evaluate(tag, pred, y_te, modes_te, train_cnt))

    if not results:
        print("\n評価できたモデルがありません。--tags と保存済みファイル名を確認してください。")
        return 1
    if len(results) >= 2:
        report_summary(results)
    if not args.no_plot:
        plot_report(results, label_cnt, args.out, jp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
