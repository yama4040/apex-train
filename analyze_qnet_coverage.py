"""
QNetworkの学習具合（Qテーブルの埋まり具合に相当）を可視化するスクリプト。

【目的】
apex2.pyで学習したQNetwork（model.py / 25次元入力・3行動出力）に対し、
「速度 × 駅までの距離」の2軸を格子状にスイープした人工状態ベクトルを一括推論し、
状態空間のどの領域でQ値がどう学習されているかをヒートマップで可視化する。
表形式Q学習の「テーブルの埋まり具合」に相当する診断を関数近似モデルに対して行う。

【出力（--outdir 以下、既定 qnet_analysis/）】
- qnet_sweep_wide.png : 駅まで0〜2.2km（巡航〜減速の全域）の3面ヒートマップ
- qnet_sweep_zoom.png : 駅まで0〜150m（time_stepが0.1秒に短縮される駅直前領域）の3面ヒートマップ
  各図の3面は左から
    1) max Q        … 最良行動のQ値の大きさ（過大評価・発散のチェック）
    2) best action  … 貪欲方策マップ（惰行/力行/ブレーキのどれを選ぶか）
    3) Q gap        … 最良行動と次点のQ値差（ほぼ0の領域＝行動を区別できていない「未学習に近い」領域）
- コンソールに領域別（巡航/減速/駅直前）の要約統計を表示

【グリッド以外の状態量（シナリオ固定値）】
2軸以外の23次元は「定時運行・先行列車なし・平坦・制限70km/h」のシナリオで
environment2.pyのnormalized_stateと同一のスケーリング式により埋める。
フェーズone-hotは距離・速度から environment2._get_current_phase_str と同じ規則で導出する
（ただし時刻依存の「出発直後の加速フェーズ」と路線データ依存の「制限接近フェーズ」は
 グリッド上では再現できないため巡航として扱う。この2フェーズの領域は本図では評価対象外）。

【使い方】
    python analyze_qnet_coverage.py                          # data/ 以下の最新の .weights.h5 を自動選択
    python analyze_qnet_coverage.py --weights data/20260715004414/xxx.weights.h5
    python analyze_qnet_coverage.py --overlay-csv comp/12100_0.csv   # 実走行の訪問状態を重ね描き
    python analyze_qnet_coverage.py --pre-action accel       # 直前ノッチのシナリオを変更
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # 非対話実行（表示ブロック防止）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from model import QNetwork
from required_speed import brake_stop_distance_m

# ===== シナリオ定数（environment2.pyの正規化式と対応させること） =====
NUM_STATES = 25
SPEED_LIMIT = 70.0        # 路線制限速度[km/h]（comp/のログと同じ0.875=70/80に対応）
CBTC_SIGNAL = 70.0        # CBTC信号現示[km/h]
SCHED_AVG_SPEED = 43.6    # 計画ダイヤの平均速度[km/h]（≒2.178km/180s。「定時運行中」の残り時間の算出に使用）
HOLDING_TIME = 5.0        # 現ノッチの継続時間[s]のシナリオ値
PREV_NOTCH_DURATION = 10.0  # 直前ノッチの継続時間[s]のシナリオ値

# 行動の表示定義（actions.py: coasting=0, acceleration=1, deceleration=2）
ACTION_NAMES = ["coast", "accel", "decel"]
# 固定順の categorical 配色（Paul Tol bright: 色覚多様性に配慮した3色）
ACTION_COLORS = ["#4477AA", "#EE6677", "#228833"]


def find_latest_weights():
    """data/ 以下で最も新しい .weights.h5 を返す。"""
    candidates = glob.glob(os.path.join("data", "*", "*.weights.h5"))
    if not candidates:
        raise FileNotFoundError("data/*/ 以下に .weights.h5 が見つかりません。--weights で指定してください。")
    return max(candidates, key=os.path.getmtime)


def build_state_grid(dist_km, speed_kmh, pre_action):
    """
    速度×距離の格子から25次元の正規化状態ベクトル群を構築する。
    正規化式は environment2.py の normalized_state と1対1で対応させている。

    dist_km: shape (D,) 駅までの残距離[km]
    speed_kmh: shape (V,) 現在速度[km/h]
    pre_action: 0/1/2（惰行/力行/ブレーキ）のシナリオ値
    戻り値: states shape (V*D, 25), 形状復元用の (V, D)
    """
    D, V = len(dist_km), len(speed_kmh)

    # 速度ごとのブレーキ停止距離[km]を事前計算（環境の_calc_brake_distanceと同一の物理モデル）
    brake_dist_km = np.array([brake_stop_distance_m(v, 0.0) / 1000.0 for v in speed_kmh])

    vv, dd = np.meshgrid(speed_kmh, dist_km, indexing="ij")   # shape (V, D)
    bb = np.repeat(brake_dist_km[:, None], D, axis=1)          # shape (V, D)

    # 定時運行シナリオ: 残り時間[s] = 残距離 ÷ 計画平均速度
    remaining_time = dd / SCHED_AVG_SPEED * 3600.0

    # フェーズ判定（environment2._get_current_phase_str の距離・速度依存部分と同一規則）
    phase_stop = ((dd * 1000.0 <= 10.0) & (vv <= 0.1)).astype(np.float32)
    phase_decel = ((dd <= 0.4) & (phase_stop == 0)).astype(np.float32)
    phase_cruise = ((phase_stop == 0) & (phase_decel == 0)).astype(np.float32)

    pre_c = 1.0 if pre_action == 0 else 0.0
    pre_a = 1.0 if pre_action == 1 else 0.0
    pre_d = 1.0 if pre_action == 2 else 0.0

    ones = np.ones_like(vv, dtype=np.float32)
    states = np.stack([
        vv / 80.0,                                                        # 1. 現在の速度
        (np.maximum(dd, -0.5) + 0.5) / 2.0,                               # 2. 駅までの距離（広域）
        (np.clip(dd, -0.05, 0.2) + 0.05) * 4.0,                           # 3. 駅までの距離（ズーム）
        remaining_time / 360.0,                                           # 4. 残り時間
        ones * min(HOLDING_TIME, 30.0) / 30.0,                            # 5. 同じ操作の継続時間
        ones * pre_c,                                                     # 6. 直前の行動（惰行）
        ones * pre_a,                                                     # 7. 直前の行動（加速）
        ones * pre_d,                                                     # 8. 直前の行動（減速）
        (np.maximum(dd, -0.5) + 0.5) / 2.0,                               # 9. 先行列車までの距離（先行なし＝駅距離と同値）
        ones * CBTC_SIGNAL / 80.0,                                        # 10. CBTC信号現示
        ones * SPEED_LIMIT / 80.0,                                        # 11. 路線制限速度
        bb / 1.0,                                                         # 12. 必要ブレーキ距離
        np.clip(dd - bb, -0.5, 1.5) / 1.5,                                # 13. 停車余裕マージン
        np.zeros_like(vv),                                                # 14. フェーズ：加速（時刻依存のためグリッドでは対象外）
        phase_cruise,                                                     # 15. フェーズ：巡航
        np.zeros_like(vv),                                                # 16. フェーズ：制限接近（路線依存のため対象外）
        phase_decel,                                                      # 17. フェーズ：減速
        phase_stop,                                                       # 18. フェーズ：停車完了
        ones * 80.0 / 80.0,                                               # 19. 先行列車の速度（先行なし＝80km/h扱い）
        np.zeros_like(vv),                                                # 20. 現在の勾配（平坦）
        ones,                                                             # 21. 勾配変化までの距離（変化なし=1.0）
        np.zeros_like(vv),                                                # 22. この先の勾配値
        ones,                                                             # 23. 制限変化までの距離（変化なし=1.0）
        ones * SPEED_LIMIT / 80.0,                                        # 24. この先の制限速度
        ones * min(PREV_NOTCH_DURATION, 30.0) / 30.0,                     # 25. 直前ノッチの継続時間
    ], axis=-1).astype(np.float32)

    return states.reshape(-1, NUM_STATES), (V, D)


def robust_limits(values, lo=1.0, hi=99.0):
    """外れ値（Q値の発散）で配色が潰れないよう、パーセンタイルで表示レンジを決める。"""
    return np.percentile(values, lo), np.percentile(values, hi)


def plot_sweep(q, dist_km, speed_kmh, title, save_path, overlay=None):
    """
    3面ヒートマップ（max Q / best action / Q gap）を描画して保存する。
    q: shape (V, D, 3)
    overlay: (dist[km], speed[km/h]) の実走行訪問点。Noneなら重ね描きなし
    """
    q_sorted = np.sort(q, axis=-1)
    max_q = q_sorted[:, :, -1]
    gap = q_sorted[:, :, -1] - q_sorted[:, :, -2]
    best = np.argmax(q, axis=-1)

    extent = [dist_km[0], dist_km[-1], speed_kmh[0], speed_kmh[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=150, sharey=True)

    # --- 1) max Q（過大評価・発散チェック。逐次配色＋ロバストレンジ） ---
    vmin, vmax = robust_limits(max_q)
    im0 = axes[0].imshow(max_q, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"max Q  (display range: p1={vmin:.2f}, p99={vmax:.2f})")
    fig.colorbar(im0, ax=axes[0], label="max Q")

    # --- 2) 貪欲方策マップ（categorical・固定順3色） ---
    cmap = ListedColormap(ACTION_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axes[1].imshow(best, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)
    axes[1].set_title("best action (greedy policy)")
    axes[1].legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(ACTION_COLORS, ACTION_NAMES)],
                   loc="upper right", framealpha=0.9)

    # --- 3) 行動間ギャップ（≈0の領域＝行動を区別できていない領域） ---
    gvmax = np.percentile(gap, 99.0)
    im2 = axes[2].imshow(gap, origin="lower", aspect="auto", extent=extent,
                         cmap="magma", vmin=0.0, vmax=max(gvmax, 1e-6))
    axes[2].set_title(f"Q gap = Q(1st) - Q(2nd)  (p99={gvmax:.3f})")
    fig.colorbar(im2, ax=axes[2], label="Q gap")

    for ax in axes:
        ax.set_xlabel("distance to station [km]")
        # 制限速度と、time_stepが0.1秒になる駅手前100mの境界を補助線で示す
        ax.axhline(SPEED_LIMIT, color="white", lw=0.8, ls="--", alpha=0.7)
        if dist_km[0] <= 0.1 <= dist_km[-1]:
            ax.axvline(0.1, color="white", lw=0.8, ls=":", alpha=0.7)
        if overlay is not None:
            od, ov = overlay
            mask = (od >= dist_km[0]) & (od <= dist_km[-1]) & (ov >= speed_kmh[0]) & (ov <= speed_kmh[-1])
            ax.scatter(od[mask], ov[mask], s=4, c="white", edgecolors="black",
                       linewidths=0.3, alpha=0.8, label="visited (run log)")
    axes[0].set_ylabel("speed [km/h]")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")
    return max_q, gap, best


def print_region_summary(label, dist_km, speed_kmh, max_q, gap, best):
    """領域別（巡航/減速/駅直前）の要約統計をコンソールに表示する。"""
    dd = np.repeat(dist_km[None, :], len(speed_kmh), axis=0)
    regions = {
        "巡航域 (>0.4km)": dd > 0.4,
        "減速域 (0.1〜0.4km)": (dd <= 0.4) & (dd > 0.1),
        "駅直前 (<=0.1km, 0.1秒step)": dd <= 0.1,
    }
    print(f"\n===== {label} 領域別サマリ =====")
    print(f"{'領域':<28}{'mean|Q|':>10}{'max|Q|':>10}{'mean gap':>10}{'coast%':>8}{'accel%':>8}{'decel%':>8}")
    for name, mask in regions.items():
        if not mask.any():
            continue
        shares = [np.mean(best[mask] == a) * 100 for a in range(3)]
        print(f"{name:<28}{np.abs(max_q[mask]).mean():>10.3f}{np.abs(max_q[mask]).max():>10.3f}"
              f"{gap[mask].mean():>10.4f}{shares[0]:>7.1f}%{shares[1]:>7.1f}%{shares[2]:>7.1f}%")


def load_overlay(csv_path):
    """apex2.pyのTester出力CSV（comp/やdata/配下）から実走行の訪問状態（距離・速度）を読み込む。"""
    df = pd.read_csv(csv_path)
    return df["raw_stat_dist"].to_numpy(), df["raw_speed"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description="QNetworkの状態空間グリッドスイープ診断")
    parser.add_argument("--weights", default=None, help="学習済み重み(.weights.h5)。省略時はdata/以下の最新を使用")
    parser.add_argument("--outdir", default="qnet_analysis", help="出力ディレクトリ")
    parser.add_argument("--overlay-csv", default=None, help="実走行ログCSV（raw_stat_dist/raw_speed列を持つTester出力）を重ね描き")
    parser.add_argument("--pre-action", default="coast", choices=["coast", "accel", "decel"],
                        help="直前ノッチのシナリオ値（既定: coast）")
    args = parser.parse_args()

    weights_path = args.weights or find_latest_weights()
    pre_action = ["coast", "accel", "decel"].index(args.pre_action)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[重み] {weights_path}")
    print(f"[シナリオ] 直前ノッチ={args.pre_action}, 制限={SPEED_LIMIT}km/h, 定時運行・先行列車なし・平坦")

    # モデル構築（一度callして形状を確定させてから重みをロード）
    q_network = QNetwork(NUM_STATES)
    q_network(np.zeros((1, NUM_STATES), dtype=np.float32))
    q_network.load_weights(weights_path)

    overlay = load_overlay(args.overlay_csv) if args.overlay_csv else None

    # 広域: 駅まで0〜2.2km × 速度0〜80km/h ／ ズーム: 駅直前150m × 0〜40km/h
    sweeps = [
        ("wide", np.linspace(0.0, 2.2, 221), np.linspace(0.0, 80.0, 161)),
        ("zoom", np.linspace(0.0, 0.15, 151), np.linspace(0.0, 40.0, 161)),
    ]
    for name, dist_km, speed_kmh in sweeps:
        states, (V, D) = build_state_grid(dist_km, speed_kmh, pre_action)
        q = q_network(states).numpy().reshape(V, D, 3)
        title = f"QNetwork sweep [{name}]  weights: {os.path.basename(weights_path)}  (pre-action: {args.pre_action})"
        save_path = os.path.join(args.outdir, f"qnet_sweep_{name}.png")
        max_q, gap, best = plot_sweep(q, dist_km, speed_kmh, title, save_path, overlay)
        print_region_summary(name, dist_km, speed_kmh, max_q, gap, best)


if __name__ == "__main__":
    main()
