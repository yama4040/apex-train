# 運転曲線PNGの共通描画モジュール（2026-08-18）
#
# apex2.py の Tester が出力する運転曲線と、TASC制動で上書きした運転曲線
# （apply_tasc_to_runcurve.py）を**同一書式**にするために、配色と背景描画をここに集約する。
# 片方だけ書式が変わると2枚を並べた比較ができなくなるため、定義は必ずこのモジュールに置く。

import matplotlib.pyplot as plt

# 運転モード→運転曲線の色（PNGでモード切替を視認するため）。
#   通常=赤 / 遅延回復=緑 / 機外停車防止(駅間停車防止)=オレンジ / 運転間隔調整=紫
MODE_COLORS = {"normal": "red", "delay_recovery": "green", "anti_mid_stop": "orange", "spacing": "purple"}
MODE_LABELS = {"normal": "Normal", "delay_recovery": "DelayRecovery",
               "anti_mid_stop": "AntiMidStop", "spacing": "Spacing"}


def curve_background_params(env):
    """運転曲線の背景を描くのに必要な情報を env（reset直後）から取り出す。

    エピソードが進むと env の状態（位置・先行列車）が変わって再現できないため、
    リセット直後にこの辞書を控えておき、あとから draw_curve_background に渡す。
    """
    return {
        "dep_pos": env.departure_station["position"],
        "arr_pos": env.arrival_station["position"],
        "fw_pos": env.fowerd_train_position,
        "sec_start": env.position,
        "front_sections": [dict(fs) for fs in env.train.front_sections],
    }


def draw_curve_background(bg):
    """運転曲線（位置-速度）の背景を描く。図の新規作成もここで行う。

    駅の黒線・先行列車の初期位置・区間ごとの制限速度の階段線を描く。
    """
    plt.figure(dpi=200, figsize=(10, 10))
    plt.xlabel("Position[km]")
    plt.ylabel("Speed[km/h]")
    plt.plot([bg["dep_pos"], bg["arr_pos"]], [0, 0], "k-", lw=3)
    plt.plot([bg["dep_pos"], bg["dep_pos"]], [0, 100], "k-", lw=3)
    plt.plot([bg["arr_pos"], bg["arr_pos"]], [0, 100], "k-", lw=3)
    if bg["fw_pos"] is not None:
        plt.plot([bg["fw_pos"], bg["fw_pos"]], [0, 100], "k-", lw=3)
    sec_start = bg["sec_start"]
    front_sections = bg["front_sections"]
    for fsi in range(len(front_sections)):
        plt.plot([sec_start, sec_start + front_sections[fsi]["distance"]],
                 [front_sections[fsi]["speed_limit"], front_sections[fsi]["speed_limit"]], "k-", lw=1)
        if fsi > 0:
            plt.plot([sec_start, sec_start],
                     [front_sections[fsi]["speed_limit"], front_sections[fsi - 1]["speed_limit"]], "k-", lw=1)
        sec_start += front_sections[fsi]["distance"]


def plot_curve_by_mode(ax_plot, x, y, modes):
    """運転曲線をモード別に色分けして描画する。x,y,modes は同一長（各点＝各ステップ）。
    連続する同一モードの点をまとめて1本の線として描き（描画効率化）、
    モードごとに凡例ラベルを1度だけ付ける。ax_plot は plt.plot（または Axes.plot）を想定。"""
    n = min(len(x), len(y), len(modes))
    if n == 0:
        return

    def norm(k):
        return k if k in MODE_COLORS else "normal"

    seen = set()
    i = 0
    while i < n - 1:
        m = norm(modes[i])
        j = i
        while j < n - 1 and norm(modes[j]) == m:
            j += 1
        lbl = None
        if m not in seen:
            seen.add(m)
            lbl = f"Own Train ({MODE_LABELS.get(m, m)})"
        # 点 i..j を1本で描画。次の区間は点jを共有して連結する。
        ax_plot(x[i:j + 1], y[i:j + 1], color=MODE_COLORS[m], lw=1.3, label=lbl)
        i = j
