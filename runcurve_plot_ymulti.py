# -*- coding: utf-8 -*-
"""
複数駅間版（羽前成田→白兎→蚕桑）の運転曲線・ダイアグラム描画。

既存 `runcurve_plot.py`（apex2.py と apply_tasc_to_runcurve.py が共有）は**一切変更しない**。
複数駅間版は「2区間を通しで描く」「駅停車の区間をダイアグラム上に示す」「標準運転曲線を重ねる」
という要件が増えるため別モジュールにする。

`apex_ymulti.py` の Tester と、学習後の後処理スクリプトが共有して同一書式を保つ。
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config_ymulti as CFG

# 運転モード → 運転曲線の色
MODE_COLORS = {"normal": "red", "delay_recovery": "green", "anti_mid_stop": "orange",
               "spacing": "purple", "hold_at_station": "black"}
MODE_LABELS = {"normal": "Normal", "delay_recovery": "DelayRecovery",
               "anti_mid_stop": "AntiMidStop", "spacing": "Spacing",
               "hold_at_station": "HoldAtStation"}

_JP_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
]
_JP_READY = None


def setup_japanese_font():
    """日本語フォントを設定する。見つからなければ False（英語表記へフォールバック）。"""
    global _JP_READY
    if _JP_READY is not None:
        return _JP_READY
    for p in _JP_FONTS:
        if not os.path.exists(p):
            continue
        try:
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            _JP_READY = True
            return True
        except Exception:
            continue
    _JP_READY = False
    return False


def _v_std_curves():
    """標準運転曲線（位置→速度）を区間ごとに読む。無ければ None。"""
    import csv
    out = []
    for k in range(len(CFG.RUNNING_TIMES)):
        path = os.path.join(CFG.STANDARD_CURVE_DIR, f"v_std_{CFG.STATION_INDICES[k]}.csv")
        if not os.path.exists(path):
            return None
        xs, vs = [], []
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                xs.append(float(r["position"]))
                vs.append(float(r["v_std"]))
        out.append((xs, vs))
    return out


def plot_run_curve(path, positions, speeds, modes, station_positions, limit_sections,
                   title="", forward_positions=None, show_std=True):
    """運転曲線（位置-速度）を描く。連続する同一モードをまとめて色分けする。"""
    jp = setup_japanese_font()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    # 背景: 駅の黒線・制限速度の階段線
    for k, p in enumerate(station_positions):
        ax.axvline(p, color="k", lw=2.5)
        name = CFG.STATION_NAMES_JA[CFG.STATION_INDICES[k]] if jp else str(CFG.STATION_INDICES[k])
        ax.text(p, 76, name, ha="center", va="bottom", fontsize=11)
    for sec in limit_sections:
        ax.plot([sec["start"], sec["start"] + sec["distance"]],
                [sec["speed_limit"], sec["speed_limit"]], "k-", lw=1)

    # 標準運転曲線（比較基準）
    if show_std:
        std = _v_std_curves()
        if std:
            for i, (xs, vs) in enumerate(std):
                ax.plot(xs, vs, color="gray", lw=1.2, ls="--",
                        label=("標準運転曲線" if jp else "Standard") if i == 0 else None)

    # 本線: モード別に色分け
    n = min(len(positions), len(speeds), len(modes))
    seen = set()
    i = 0
    while i < n - 1:
        m = modes[i] if modes[i] in MODE_COLORS else "normal"
        j = i
        while j < n - 1 and (modes[j] if modes[j] in MODE_COLORS else "normal") == m:
            j += 1
        lbl = None
        if m not in seen:
            seen.add(m)
            lbl = f"Own Train ({MODE_LABELS.get(m, m)})"
        ax.plot(positions[i:j + 1], speeds[i:j + 1], color=MODE_COLORS[m], lw=1.4, label=lbl)
        i = j

    if forward_positions:
        for p in forward_positions:
            ax.axvline(p, color="dimgray", lw=1.0, ls=":")

    ax.set_xlabel("位置 [km]" if jp else "Position [km]")
    ax.set_ylabel("速度 [km/h]" if jp else "Speed [km/h]")
    ax.set_ylim(0, 82)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_diagram(path, times, positions, station_positions, title="",
                 f_times=None, f_positions=None):
    """ダイアグラム（時刻-位置）を描く。駅停車中は水平線になるので停車が一目で分かる。"""
    jp = setup_japanese_font()
    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    for k, p in enumerate(station_positions):
        ax.axhline(p, color="k", lw=1.0, ls="--", alpha=0.6)
        name = CFG.STATION_NAMES_JA[CFG.STATION_INDICES[k]] if jp else str(CFG.STATION_INDICES[k])
        ax.text(0, p, name, ha="left", va="bottom", fontsize=10)
    if f_times and f_positions:
        ax.plot(f_times, f_positions, color="dimgray", lw=1.4, ls="--",
                label="先行列車" if jp else "Forward")
    ax.plot(times, positions, color="red", lw=1.8, label="自列車" if jp else "Own")
    # 標準ダイヤ
    arr = CFG.scheduled_arrival_times()
    ax.plot(arr, station_positions, color="steelblue", lw=1.0, ls=":", marker="o", ms=3,
            label="標準ダイヤ" if jp else "Schedule")
    ax.set_xlabel("時刻 [s]" if jp else "Time [s]")
    ax.set_ylabel("位置 [km]" if jp else "Position [km]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
