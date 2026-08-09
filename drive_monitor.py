"""運転曲線モニター（テストケースログの再生アプリケーション）

apex2.py の Tester が出力したテストケースのログCSVを読み込み、時間に応じて
運転曲線・ダイヤグラム・列車の動き・自列車実況をリアルタイムに再生する
デスクトップアプリケーション（PyQt5 + matplotlib）。

    python drive_monitor.py                       # GUIでCSVを選択
    python drive_monitor.py data/<run>/0_13.csv   # 起動時に読み込む
    python drive_monitor.py A.csv B.csv           # 2本を重ねて比較

■ 対応するログ形式
  1. 新形式（2026-08 以降の apex2.py が出力）
       data/<run>/<file>_<ci>.csv       … 末尾に time/position/speed_limit/fw_position/
                                           fw_speed/mode/action/gradient/fw_dwell_elapsed の9列
       data/<run>/<file>_<ci>_meta.json … テストケース説明・先行遅延・先行の駅停車時間・
                                           駅名/位置・標準運転時間・制限速度プロファイル
  2. 旧形式（上記の列を持たない過去のログ）
       raw列（raw_speed / raw_stat_dist / raw_cbtc_signal / raw_fw_dist / モードone-hot）から
       復元する。時刻・勾配・先行列車速度は同じrun内の
       data/<run>/LLM評価用/<file>_<ci>_llm.csv があればそこから取得し、
       無ければ time_step 規則（駅手前100m以内で0.1秒、それ以外1.0秒）で再構成する。
       絶対位置は input/Station.csv の到着駅位置から復元する。

■ 実況の「残り時間」について
  ログの raw_rem_time（environment2.remaining_time）は先行列車がいる場合
  「先行の位置から引いた標準運転時間 − 経過時刻」となるため、先行が進まない間は値が
  止まったり増えたりして時計として読めない。本モニターでは
    残り時間 = 標準運転時間（出発駅のrt）− 経過時刻
  を表示する（先行なしの場合は raw_rem_time と完全一致）。
"""

from __future__ import annotations

import argparse
import codecs
import ctypes
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 旧形式ログの位置復元に使う既定値。apex2.py の Tester は env.reset(11, ...) 固定のため、
# 出発駅=index 11（羽前成田）、到着駅=index 12（白兎）が既定のテストケースとなる。
DEFAULT_DEPARTURE_INDEX = 11

# 黒背景テーマ。暗い背景上で視認できるよう、モード色は明度・彩度を上げた値を使う
BG_FIG = "#0b0d12"        # ウィンドウ／Figureの地の色
BG_AXES = "#14181f"       # グラフ内側
FG_TEXT = "#e8ebf2"       # 文字・目盛
FG_DIM = "#9aa3b2"        # 補助的な文字
GRID_COLOR = "#2b313d"    # グリッド
RAIL_COLOR = "#cfd6e4"    # 線路・駅の縦線

MODE_COLORS = {"normal": "#ff6b5e", "delay_recovery": "#41d98a",
               "anti_mid_stop": "#ffb02e", "spacing": "#b39bff"}
MODE_LABELS = {"normal": "通常運転", "delay_recovery": "遅延回復",
               "anti_mid_stop": "駅間停車防止", "spacing": "間隔調整"}
MODE_KEYS = ["normal", "delay_recovery", "anti_mid_stop", "spacing"]
ACTION_LABELS = {0: "惰行", 1: "力行", 2: "ブレーキ"}

FORWARD_COLOR = "#4da3ff"
CBTC_COLOR = "#ff4d6d"
RUN_LINESTYLES = ["-", "--"]  # 1本目=実線、2本目（比較）=破線

# 列車の模式図を実スケールで描くための諸元（CLAUDE.md「その他備考」準拠）。
# 以前は見た目重視で線路長の4.5%（≒100m）の箱を描いていたため、車間が数百mあっても
# 先行列車と重なって衝突しているように見えていた。実車長で描くことでこれを解消する。
TRAIN_LENGTH_KM = 0.020   # 1両編成＝列車長20m
CBTC_LIMIT_KM = 0.050     # CBTCの停止限界距離50m


# =============================================================================
# 起動環境のセットアップ（Qtプラットフォーム／日本語フォント）
# =============================================================================
def _setup_qt_platform() -> None:
    """xcbプラグインの依存ライブラリが欠けている環境（WSLg等）ではwaylandへ切り替える。

    PyQt5同梱の libqxcb.so は libxcb-icccm 等のシステムライブラリを必要とするが、
    WSLのUbuntuには既定で入っていない。その場合 QApplication 生成時に
    プロセスごとabortしてしまい例外として捕捉できないため、事前に判定しておく。
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    try:
        import PyQt5
        plugin = os.path.join(os.path.dirname(PyQt5.__file__),
                              "Qt5", "plugins", "platforms", "libqxcb.so")
        if not os.path.exists(plugin):
            return
        ctypes.CDLL(plugin)  # 依存が欠けていれば OSError
    except OSError:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
    except Exception:
        pass


_setup_qt_platform()

import matplotlib  # noqa: E402

matplotlib.use("Qt5Agg")

from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

# 日本語フォントの候補。Linux標準／WSLからWindowsフォントを借りる場合／macOSを順に探す。
_JP_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothR.ttc",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]
# 実況パネル用の等幅日本語フォント候補（数値の桁を揃えるため）
_JP_MONO_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/mnt/c/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]

JP_FONT: Optional[str] = None       # 全体で使う日本語フォント名
JP_MONO_FONT: Optional[str] = None  # 実況パネルで使う等幅日本語フォント名


def _register_font(path: str) -> Optional[str]:
    """フォントファイルをmatplotlibとQtに登録し、フォント名を返す。"""
    if not os.path.exists(path):
        return None
    try:
        font_manager.fontManager.addfont(path)
        name = font_manager.FontProperties(fname=path).get_name()
    except Exception:
        return None
    try:
        QtGui.QFontDatabase.addApplicationFont(path)
    except Exception:
        pass
    return name


def _apply_dark_theme() -> None:
    """matplotlibの既定を黒背景テーマに揃える。"""
    matplotlib.rcParams.update({
        "figure.facecolor": BG_FIG,
        "figure.edgecolor": BG_FIG,
        "savefig.facecolor": BG_FIG,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": FG_TEXT,
        "text.color": FG_TEXT,
        "xtick.color": FG_DIM,
        "ytick.color": FG_DIM,
        "grid.color": GRID_COLOR,
        "legend.facecolor": BG_AXES,
        "legend.edgecolor": GRID_COLOR,
        "axes.unicode_minus": False,
    })


def _setup_japanese_font() -> Optional[str]:
    """日本語フォントを登録して既定フォントに設定し、フォント名を返す（無ければNone）。"""
    global JP_FONT, JP_MONO_FONT
    _apply_dark_theme()
    for path in _JP_FONT_FILES:
        name = _register_font(path)
        if name:
            JP_FONT = name
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    for path in _JP_MONO_FONT_FILES:
        name = _register_font(path)
        if name:
            JP_MONO_FONT = name
            break
    return JP_FONT


# =============================================================================
# ログの読み込み
# =============================================================================
@dataclass
class RunLog:
    """1テストケース分の走行ログ（各配列は同一長＝各ステップ）。"""
    path: str
    label: str
    time: np.ndarray            # 時刻[s]
    position: np.ndarray        # 自列車位置[km]
    speed: np.ndarray           # 自列車速度[km/h]
    speed_limit: np.ndarray     # 路線制限速度[km/h]
    signal: np.ndarray          # CBTC信号現示[km/h]
    station_dist: np.ndarray    # 次駅までの距離[m]
    remaining_time: np.ndarray  # ダイヤ上の残り時間[s]＝標準運転時間−経過時刻（単調減少）
    mode: list                  # 運転モード（'normal'等）
    action: np.ndarray          # 選択した行動（0:惰行 1:力行 2:ブレーキ）
    fw_position: np.ndarray     # 先行列車位置[km]（先行なしはnan）
    fw_speed: np.ndarray        # 先行列車速度[km/h]（先行なしはnan）
    gradient: np.ndarray        # 現在位置の勾配[‰]（正=上り勾配）
    fw_dwell: np.ndarray        # 先行列車が次駅に停車してからの経過時間[s]（不明はnan）
    meta: dict = field(default_factory=dict)
    source: str = "new"         # 'new'（新形式）／'legacy'（旧形式から復元）
    notes: list = field(default_factory=list)

    @property
    def has_forward(self) -> bool:
        return bool(np.any(np.isfinite(self.fw_position)))

    @property
    def fw_gap(self) -> np.ndarray:
        """先行列車までの距離[m]（先行なしはnan）。"""
        return (self.fw_position - self.position) * 1000.0


def _read_csv_any(path: str) -> pd.DataFrame:
    with codecs.open(path, "r", "utf-8", "ignore") as f:
        return pd.read_csv(f)


def _load_station_table() -> Optional[pd.DataFrame]:
    path = os.path.join(BASE_DIR, "input", "Station.csv")
    if not os.path.exists(path):
        return None
    try:
        return _read_csv_any(path)
    except Exception:
        return None


def _load_speed_limit_table() -> Optional[pd.DataFrame]:
    path = os.path.join(BASE_DIR, "input", "speed_limit.csv")
    if not os.path.exists(path):
        return None
    try:
        return _read_csv_any(path)
    except Exception:
        return None


def _lookup_speed_limit(positions: np.ndarray) -> np.ndarray:
    """input/speed_limit.csv から位置ごとの制限速度を引く（旧形式ログのフォールバック用）。"""
    table = _load_speed_limit_table()
    if table is None:
        return np.full(len(positions), np.nan)
    starts = table["start"].to_numpy(dtype=float)
    limits = table["speed_limit"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(starts, positions, side="right") - 1, 0, len(limits) - 1)
    return limits[idx]


_FORWARD_INFO_RE = re.compile(r"前方([-\d.]+)m先を([-\d.]+)km/h")


def _load_meta(csv_path: str) -> dict:
    meta_path = os.path.splitext(csv_path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_llm_sibling(csv_path: str, n_rows: int) -> Optional[pd.DataFrame]:
    """同じrun内の LLM評価用/<stem>_llm.csv を探して読む（行数が一致する場合のみ採用）。"""
    directory, base = os.path.split(csv_path)
    stem = os.path.splitext(base)[0]
    cand = os.path.join(directory, "LLM評価用", f"{stem}_llm.csv")
    if not os.path.exists(cand):
        return None
    try:
        df = _read_csv_any(cand)
    except Exception:
        return None
    return df if len(df) == n_rows else None


def _modes_from_onehot(df: pd.DataFrame) -> list:
    cols = ["mode_normal", "mode_delay_recovery", "mode_anti_mid_stop", "mode_spacing"]
    if not all(c in df.columns for c in cols):
        return ["normal"] * len(df)
    arr = df[cols].to_numpy(dtype=float)
    return [MODE_KEYS[int(i)] for i in np.argmax(arr, axis=1)]


def _default_station_positions(meta: dict) -> tuple:
    """（出発駅, 到着駅）の {name, position} を meta もしくは input/Station.csv から決める。"""
    dep = meta.get("departure_station")
    arr = meta.get("arrival_station")
    if dep and arr:
        return dep, arr
    table = _load_station_table()
    if table is None:
        return ({"name": "駅A", "position": 0.0}, {"name": "駅B", "position": 1.0})
    i = DEFAULT_DEPARTURE_INDEX
    return ({"name": str(table["name"][i]), "position": float(table["position"][i])},
            {"name": str(table["name"][i + 1]), "position": float(table["position"][i + 1])})


def load_run(csv_path: str, label: Optional[str] = None) -> RunLog:
    """テストケースのログCSVを読み込んで RunLog を返す。新形式・旧形式の両方に対応する。"""
    csv_path = os.path.abspath(csv_path)
    df = _read_csv_any(csv_path)
    if len(df) == 0:
        raise ValueError(f"行が1件もありません: {csv_path}")
    if "raw_speed" not in df.columns:
        raise ValueError(
            f"テストケースのログCSVではないようです（raw_speed列がありません）: {csv_path}")

    meta = _load_meta(csv_path)
    notes = []
    label = label or os.path.splitext(os.path.basename(csv_path))[0]

    speed = df["raw_speed"].to_numpy(dtype=float)
    station_dist_km = df["raw_stat_dist"].to_numpy(dtype=float)
    signal = df["raw_cbtc_signal"].to_numpy(dtype=float)
    modes = _modes_from_onehot(df)

    if "time" in df.columns and "position" in df.columns:
        # ---- 新形式 ---------------------------------------------------------
        source = "new"
        t = df["time"].to_numpy(dtype=float)
        position = df["position"].to_numpy(dtype=float)
        speed_limit = df["speed_limit"].to_numpy(dtype=float)
        fw_position = pd.to_numeric(df["fw_position"], errors="coerce").to_numpy(dtype=float)
        fw_speed = pd.to_numeric(df["fw_speed"], errors="coerce").to_numpy(dtype=float)
        modes = [str(m) if str(m) in MODE_COLORS else "normal" for m in df["mode"]]
        action = pd.to_numeric(df["action"], errors="coerce").fillna(0).to_numpy(dtype=int)
        gradient = (pd.to_numeric(df["gradient"], errors="coerce").to_numpy(dtype=float)
                    if "gradient" in df.columns else df["norm_gradient"].to_numpy(dtype=float) * 40.0)
        fw_dwell = (pd.to_numeric(df["fw_dwell_elapsed"], errors="coerce").to_numpy(dtype=float)
                    if "fw_dwell_elapsed" in df.columns else np.full(len(df), np.nan))
        if not meta:
            notes.append("meta.json が見つからないため駅位置は input/Station.csv から補完しました")
    else:
        # ---- 旧形式（raw列＋LLM評価用CSVから復元） ---------------------------
        source = "legacy"
        llm = _load_llm_sibling(csv_path, len(df))
        _, arrival = _default_station_positions(meta)
        position = float(arrival["position"]) - station_dist_km

        if llm is not None:
            t = llm["time"].to_numpy(dtype=float)
            speed_limit = llm["speed_limit"].to_numpy(dtype=float)
            fw_gap_m = np.full(len(df), np.nan)
            fw_speed = np.full(len(df), np.nan)
            for i, info in enumerate(llm["forward_info"].astype(str)):
                m = _FORWARD_INFO_RE.search(info)
                if m:
                    fw_gap_m[i] = float(m.group(1))
                    fw_speed[i] = float(m.group(2))
            fw_position = position + fw_gap_m / 1000.0
            gradient = llm["current_gradient"].to_numpy(dtype=float)
            # LLM評価用CSVが持つのは「標準停車30秒を超えた分」なので、超過中のみ総経過時間へ換算する。
            # 停車開始〜30秒の区間は記録が無いため不明（nan）として扱う。
            obs = llm["forward_observed_delay"].to_numpy(dtype=float)
            fw_dwell = np.where(obs > 0.0, obs + 30.0, np.nan)
            notes.append("旧形式ログ: 時刻・制限速度・勾配・先行列車情報を LLM評価用CSV から復元しました"
                         "（先行停車経過は標準30秒超過中のみ表示）")
        else:
            # time_step 規則で時刻を再構成する（environment2.Environment.time_step と同一）。
            # 出発時の遅延ぶんの時刻オフセットは分からないため 0 始まりとする。
            base_dt = float(meta.get("base_time_step", 1.0))
            dt = np.where(station_dist_km > 0.1, base_dt, base_dt * 0.1)
            t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
            if "norm_speed_limit" in df.columns:
                speed_limit = df["norm_speed_limit"].to_numpy(dtype=float) * 80.0
            else:
                speed_limit = _lookup_speed_limit(position)
            fw_dist_km = df["raw_fw_dist"].to_numpy(dtype=float)
            # 先行列車がいない場合 raw_fw_dist は raw_stat_dist と同値になる（environment2の仕様）
            has_fw = not np.allclose(fw_dist_km, station_dist_km)
            if has_fw and "norm_fw_speed" in df.columns:
                fw_position = position + fw_dist_km
                fw_speed = df["norm_fw_speed"].to_numpy(dtype=float) * 80.0
            else:
                fw_position = np.full(len(df), np.nan)
                fw_speed = np.full(len(df), np.nan)
            gradient = (df["norm_gradient"].to_numpy(dtype=float) * 40.0
                        if "norm_gradient" in df.columns else np.zeros(len(df)))
            fw_dwell = np.full(len(df), np.nan)
            notes.append("旧形式ログ: LLM評価用CSVが無いため時刻を time_step 規則から再構成しました"
                         "（出発遅延ぶんの時刻オフセット・先行停車経過は復元不可）")

        if "raw_pre_act" in df.columns:
            # 旧形式には選択行動の列が無い。次ステップの直前ノッチ＝当該ステップの行動で代用する。
            pre_act = df["raw_pre_act"].to_numpy(dtype=float)
            action = np.concatenate([pre_act[1:], pre_act[-1:]]).astype(int)
        else:
            action = np.zeros(len(df), dtype=int)

    if not meta:
        dep, arr = _default_station_positions(meta)
        meta = {"departure_station": dep, "arrival_station": arr, "desc": label}

    # 残り時間はログの raw_rem_time を使わない。environment2 の remaining_time は先行列車がいる場合
    # 「先行の位置から引いた標準運転時間 − 経過時刻」となるため、先行が進まない間は値が止まったり
    # 増えたりして時計として読めない。ここではダイヤ基準の
    #   残り時間 = 標準運転時間(出発駅のrt) − 経過時刻
    # とする（先行なしの場合は raw_rem_time と完全に一致し、出発遅延ぶんも初期値に反映される）。
    std_rt = meta.get("standard_running_time")
    if std_rt is None:
        table = _load_station_table()
        std_rt = float(table["rt"][DEFAULT_DEPARTURE_INDEX]) if table is not None else 180.0
    remaining_time = float(std_rt) - t

    return RunLog(path=csv_path, label=label, time=t, position=position, speed=speed,
                  speed_limit=speed_limit, signal=signal, station_dist=station_dist_km * 1000.0,
                  remaining_time=remaining_time, mode=modes, action=action,
                  fw_position=fw_position, fw_speed=fw_speed,
                  gradient=gradient, fw_dwell=fw_dwell, meta=meta,
                  source=source, notes=notes)


def format_case_title(runs: list) -> str:
    """テストケース説明のタイトル文字列を組み立てる。"""
    if not runs:
        return "運転曲線モニター"
    parts = []
    for run in runs:
        meta = run.meta
        if meta.get("has_forward_train") or run.has_forward:
            bits = ["先行列車あり"]
            if meta.get("forward_delay") is not None:
                bits.append(f"先行遅延{meta['forward_delay']:.0f}秒")
            if meta.get("forward_dwell") is not None:
                bits.append(f"駅停車時間{meta['forward_dwell']:.0f}秒")
            if meta.get("headway") is not None:
                bits.append(f"出発間隔{meta['headway']:.0f}秒")
            body = f"{bits[0]}（{'，'.join(bits[1:])}）" if len(bits) > 1 else bits[0]
        else:
            body = "先行列車なし"
            if meta.get("ego_delay"):
                body += f"（自列車遅延{meta['ego_delay']:.0f}秒）"
        parts.append(f"{run.label}：{body}")
    return "テストケース：" + "　／　".join(parts)


# =============================================================================
# 描画（運転曲線・ダイヤグラム・列車の動き）
# =============================================================================
class MonitorFigure:
    """3つのグラフパネルを管理する。

    上段のFigure（運転曲線・ダイヤグラム）と下段のFigure（列車の動き）に分ける。
    自列車実況はmatplotlibではなくQtウィジェットに描く（CJKテキストのラスタライズが
    1フレーム150ms近くかかり再生がカクつくため）。実況の文字列は set_info コールバックで渡す。
    """

    def __init__(self, fig_top: Figure, fig_bottom: Figure, set_info):
        self.fig_top = fig_top
        self.fig_bottom = fig_bottom
        self.set_info = set_info

        self._legend = None
        # 右端はダイヤグラムの駅名（副軸ラベル）ぶんを空けておく
        gs_t = fig_top.add_gridspec(1, 2, left=0.065, right=0.905, top=0.84, bottom=0.22,
                                    wspace=0.20)
        self.ax_curve = fig_top.add_subplot(gs_t[0, 0])
        self.ax_diagram = fig_top.add_subplot(gs_t[0, 1])
        gs_b = fig_bottom.add_gridspec(1, 1, left=0.065, right=0.985, top=0.82, bottom=0.24)
        self.ax_anim = fig_bottom.add_subplot(gs_b[0, 0])

        self.runs: list = []
        self._suptitle = fig_top.suptitle("CSVを読み込んでください", fontsize=12)
        # blit（差分描画）用の背景キャッシュ。毎フレーム全描画すると数fpsしか出ないため、
        # 静的な軸・グリッド・制限速度線を一度だけ描いて保存し、動く要素だけを重ね描きする。
        self._bg = {}
        for fig in (fig_top, fig_bottom):
            fig.canvas.mpl_connect("resize_event", lambda _e: self.invalidate_background())

    @property
    def figures(self) -> tuple:
        return (self.fig_top, self.fig_bottom)

    # ---------------------------------------------------------------- 静的描画
    def invalidate_background(self) -> None:
        """背景キャッシュを破棄する（リサイズ・ログ入れ替え時）。"""
        self._bg = {}

    def set_runs(self, runs: list) -> None:
        self.runs = runs
        for ax in (self.ax_curve, self.ax_diagram, self.ax_anim):
            ax.clear()
        self.invalidate_background()
        self._suptitle.set_text(format_case_title(runs))
        if not runs:
            for fig in self.figures:
                fig.canvas.draw_idle()
            return

        base = runs[0]
        dep = base.meta.get("departure_station", {"name": "駅A", "position": float(base.position[0])})
        arr = base.meta.get("arrival_station",
                            {"name": "駅B", "position": float(base.position[-1])})
        self.dep, self.arr = dep, arr

        self._setup_curve_axes(runs, dep, arr)
        self._setup_diagram_axes(runs, dep, arr)
        self._setup_anim_axes(runs, dep, arr)
        for fig in self.figures:
            fig.canvas.draw_idle()

    def _all_positions(self, runs: list) -> np.ndarray:
        vals = [r.position for r in runs]
        vals += [r.fw_position[np.isfinite(r.fw_position)] for r in runs]
        vals = [v for v in vals if len(v) > 0]
        return np.concatenate(vals) if vals else np.array([0.0, 1.0])

    def _setup_curve_axes(self, runs, dep, arr) -> None:
        ax = self.ax_curve
        ax.set_title("運転曲線", fontsize=11)
        ax.set_xlabel("位置 [km]")
        ax.set_ylabel("速度 [km/h]")
        ax.grid(True, alpha=0.25)

        # 制限速度の階段線（meta優先、無ければログの制限速度列から）
        sections = runs[0].meta.get("speed_limit_sections")
        if sections:
            xs, ys = [], []
            for sec in sections:
                xs += [sec["start"], sec["start"] + sec["distance"]]
                ys += [sec["speed_limit"], sec["speed_limit"]]
            ax.step(xs, ys, where="post", color=RAIL_COLOR, lw=1.0, label="制限速度")
        else:
            order = np.argsort(runs[0].position)
            ax.step(runs[0].position[order], runs[0].speed_limit[order],
                    where="post", color=RAIL_COLOR, lw=1.0, label="制限速度")

        pos_all = self._all_positions(runs)
        xmin = min(float(dep["position"]), float(pos_all.min())) - 0.05
        xmax = max(float(arr["position"]), float(pos_all.max())) + 0.05
        speed_max = max(80.0, float(max(np.nanmax(r.speed) for r in runs)) + 10.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-9, speed_max)
        ax.axhline(0.0, color=RAIL_COLOR, lw=1.0)
        # 駅名は凡例・曲線とぶつからない0km/h線の下（空白帯）に置く。
        # 端の駅は軸外にはみ出さないよう内側に寄せる。
        for st, ha in ((dep, "left"), (arr, "right")):
            ax.axvline(float(st["position"]), color=RAIL_COLOR, lw=2.0)
            ax.text(float(st["position"]), -1.5, st.get("name", ""),
                    ha=ha, va="top", fontsize=8, color=FG_TEXT,
                    bbox=dict(fc=BG_FIG, ec=RAIL_COLOR, lw=0.5, pad=1.5))

        self.curve_lcs, self.curve_fw, self.curve_head = [], [], []
        for i, run in enumerate(runs):
            lc = LineCollection([], linewidths=2.0, linestyles=RUN_LINESTYLES[i % 2], zorder=3)
            ax.add_collection(lc)
            self.curve_lcs.append(lc)
            fw, = ax.plot([], [], color=FORWARD_COLOR, ls=":", lw=1.8, zorder=2,
                          label="先行列車" if i == 0 and run.has_forward else None)
            self.curve_fw.append(fw)
            head, = ax.plot([], [], "o", ms=7, color=FG_TEXT, mfc=BG_AXES, mew=1.6, zorder=4)
            self.curve_head.append(head)
        self._add_mode_legend(ax, runs)

    def _add_mode_legend(self, ax, runs) -> None:
        """凡例はFigure下端に横1列で置く。運転曲線の中に置くと右上の曲線と重なるため。"""
        used = []
        for run in runs:
            for m in dict.fromkeys(run.mode):
                if m not in used:
                    used.append(m)
        handles = [Line2D([], [], color=MODE_COLORS.get(m, FG_DIM), lw=2.4,
                          label=f"自列車（{MODE_LABELS.get(m, m)}）")
                   for m in used]
        if any(r.has_forward for r in runs):
            handles.append(Line2D([], [], color=FORWARD_COLOR, ls=":", lw=2.0, label="先行列車"))
        handles.append(Line2D([], [], color=RAIL_COLOR, lw=1.2, label="制限速度"))
        if len(runs) > 1:
            handles.append(Line2D([], [], color=FG_DIM, ls="--", lw=2.0,
                                  label=f"比較: {runs[1].label}（破線）"))
        if self._legend is not None:
            self._legend.remove()
        self._legend = self.fig_top.legend(
            handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
            frameon=False, bbox_to_anchor=(0.5, 0.0))
        for txt in self._legend.get_texts():
            txt.set_color(FG_TEXT)

    def _setup_diagram_axes(self, runs, dep, arr) -> None:
        ax = self.ax_diagram
        ax.set_title("ダイヤグラム", fontsize=11)
        ax.set_xlabel("時刻 [s]")
        ax.set_ylabel("位置 [km]")
        ax.grid(True, alpha=0.25)

        t_min = min(float(r.time[0]) for r in runs)
        t_max = max(float(r.time[-1]) for r in runs)
        pos_all = self._all_positions(runs)
        ax.set_xlim(t_min - 2, t_max + 5)
        ax.set_ylim(min(float(dep["position"]), float(pos_all.min())) - 0.05,
                    max(float(arr["position"]), float(pos_all.max())) + 0.05)
        for st, style in ((dep, "-"), (arr, "--")):
            ax.axhline(float(st["position"]), color=RAIL_COLOR, ls=style, lw=1.5)
        # 駅名は右側の副軸の目盛として置く（プロット内に置くと走行線と重なるため）
        sec = ax.secondary_yaxis("right")
        sec.set_yticks([float(dep["position"]), float(arr["position"])])
        sec.set_yticklabels([dep.get("name", ""), arr.get("name", "")], fontsize=8)
        sec.tick_params(colors=FG_TEXT)
        sec.spines["right"].set_color(GRID_COLOR)

        self.diag_lcs, self.diag_fw, self.diag_head, self.diag_cursor = [], [], [], None
        for i, run in enumerate(runs):
            lc = LineCollection([], linewidths=2.0, linestyles=RUN_LINESTYLES[i % 2], zorder=3)
            ax.add_collection(lc)
            self.diag_lcs.append(lc)
            fw, = ax.plot([], [], color=FORWARD_COLOR, ls=":", lw=1.8, zorder=2)
            self.diag_fw.append(fw)
            head, = ax.plot([], [], "o", ms=7, color=FG_TEXT, mfc=BG_AXES, mew=1.6, zorder=4)
            self.diag_head.append(head)
        self.diag_cursor = ax.axvline(t_min, color=FG_DIM, lw=1.0, ls="-", alpha=0.7, zorder=1)

    def _setup_anim_axes(self, runs, dep, arr) -> None:
        """列車の動きの模式図。自列車と先行列車は同一線路なので同じ線の上に置き、
        比較用の2本目のログがある場合のみ、その下に平行な線路をもう1本描く。

        列車は実スケール（列車長20m）で位置の手前側＝後方に伸ばして描く。以前は見た目重視で
        線路長の4.5%（≒100m）の箱を位置中心に描いていたため、車間が数百mあっても
        先行列車と重なって衝突しているように見えていた。あわせてCBTCの停止限界50mを
        自列車前方に表示し、余裕がどれだけあるかを目視できるようにする。
        """
        ax = self.ax_anim
        ax.set_title("列車の動き（列車長20m・CBTC停止限界50mを実スケールで表示）", fontsize=10)
        pos_all = self._all_positions(runs)
        xmin = min(float(dep["position"]), float(pos_all.min())) - 0.08
        xmax = max(float(arr["position"]), float(pos_all.max())) + 0.08
        ax.set_xlim(xmin, xmax)
        ax.set_yticks([])
        ax.set_xlabel("位置 [km]")
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.set_facecolor(BG_FIG)  # 枠のない模式図なのでウィンドウ地の色に合わせる
        self._anim_xlim = (xmin, xmax)

        self._train_w = TRAIN_LENGTH_KM
        self._train_h = 0.34
        # 線路のy座標。1本目=0.0、比較用=-1.05（車間表示が上の線路に被らない間隔）
        self._track_ys = [0.0, -1.05]
        n_tracks = 2 if len(runs) > 1 else 1

        for ti in range(n_tracks):
            y = self._track_ys[ti]
            ax.plot([xmin, xmax], [y, y], color=RAIL_COLOR, lw=2.5, solid_capstyle="butt", zorder=2)
            if n_tracks > 1:
                ax.text(xmin, y + 0.05, f"{'①②'[ti]} {runs[ti].label}",
                        fontsize=8, color=FG_DIM, va="bottom", ha="left")
        # 駅は最下段の線路の下にまとめて表示する
        base_y = self._track_ys[n_tracks - 1]
        for st in (dep, arr):
            x = float(st["position"])
            ax.plot([x, x], [self._track_ys[0], base_y - 0.20], color=RAIL_COLOR, lw=1.5, zorder=1)
            ax.text(x, base_y - 0.28, st.get("name", ""), ha="center", va="top", fontsize=9,
                    color=FG_TEXT,
                    bbox=dict(fc=BG_AXES, ec=RAIL_COLOR, lw=1.0, boxstyle="round,pad=0.25"))
        # 線路と駅名がちょうど収まる高さに合わせる（余白の間延びを防ぐ）
        ax.set_ylim(base_y - 0.72, self._track_ys[0] + 1.25)

        # 自列車（位置＝先頭。後方へ列車長ぶん伸ばす）
        self.anim_bodies, self.anim_labels = [], []
        for i, run in enumerate(runs):
            y = self._track_ys[i] + 0.04
            body = FancyBboxPatch((xmin, y), self._train_w, self._train_h,
                                  boxstyle="square,pad=0", fc=MODE_COLORS["normal"],
                                  ec=FG_TEXT, lw=0.8, zorder=6, mutation_aspect=1.0)
            ax.add_patch(body)
            lbl = ax.text(xmin, y + self._train_h + 0.06, "", ha="right", va="bottom",
                          fontsize=8.5, color=FG_TEXT)
            self.anim_bodies.append((body, y))
            self.anim_labels.append(lbl)

        # 先行列車は自列車と同一線路なので、各ログの線路上に置く
        self.anim_fw_bodies, self.anim_fw_labels, self.anim_gaps = [], [], []
        # CBTC停止限界（自列車先頭から前方50m）の帯
        self.anim_cbtc = []
        for i, run in enumerate(runs):
            y = self._track_ys[i] + 0.04
            body = FancyBboxPatch((xmin, y), self._train_w, self._train_h,
                                  boxstyle="square,pad=0", fc=FORWARD_COLOR,
                                  ec=FG_TEXT, lw=0.8, zorder=6, mutation_aspect=1.0)
            ax.add_patch(body)
            self.anim_fw_bodies.append((body, y))
            # 先行列車のラベルは前方（右）へ寄せる（自列車ラベルとの衝突回避）
            self.anim_fw_labels.append(
                ax.text(xmin, y + self._train_h + 0.34, "", ha="left", va="bottom",
                        fontsize=8.5, color=FORWARD_COLOR))
            # 車間距離の表示（自列車と先行列車の中間、列車マーカーの上）
            self.anim_gaps.append(
                ax.annotate("", xy=(xmin, y + self._train_h + 0.64), ha="center",
                            fontsize=9, color=FG_DIM))
            cbtc = FancyBboxPatch((xmin, y + 0.04), CBTC_LIMIT_KM, self._train_h - 0.08,
                                  boxstyle="square,pad=0", fc=CBTC_COLOR, ec="none",
                                  alpha=0.45, zorder=5, mutation_aspect=1.0)
            ax.add_patch(cbtc)
            self.anim_cbtc.append((cbtc, y + 0.04))

    # ------------------------------------------------------------ フレーム更新
    def _dynamic_artists(self, fig: Figure) -> list:
        """毎フレーム描き直す（＝背景に焼き込まない）アーティスト一覧。"""
        if fig is self.fig_top:
            return (self.curve_lcs + self.curve_fw + self.curve_head
                    + self.diag_lcs + self.diag_fw + self.diag_head + [self.diag_cursor])
        return ([b for b, _ in self.anim_bodies] + self.anim_labels
                + [b for b, _ in self.anim_cbtc]
                + [b for b, _ in self.anim_fw_bodies] + self.anim_fw_labels + self.anim_gaps)

    def _blit(self, fig: Figure) -> None:
        """動く要素を隠した背景を一度だけ描いてキャッシュし、以降は差分だけ重ね描きする。"""
        artists = self._dynamic_artists(fig)
        if self._bg.get(fig) is None:
            visible = [a.get_visible() for a in artists]
            for a in artists:
                a.set_visible(False)
            fig.canvas.draw()
            self._bg[fig] = fig.canvas.copy_from_bbox(fig.bbox)
            for a, v in zip(artists, visible):
                a.set_visible(v)
        fig.canvas.restore_region(self._bg[fig])
        for a in artists:
            a.axes.draw_artist(a)
        fig.canvas.blit(fig.bbox)

    def update(self, sim_t: float) -> None:
        if not self.runs:
            return
        infos = []
        for i, run in enumerate(self.runs):
            k = int(np.searchsorted(run.time, sim_t, side="right"))
            k = int(np.clip(k, 0, len(run.time)))
            self._update_curve(i, run, k)
            self._update_diagram(i, run, k)
            self._update_anim(i, run, k)
            infos.append((run, k))
        self.diag_cursor.set_xdata([sim_t, sim_t])
        self._update_forward_anim(sim_t)
        self.set_info(infos)

        for fig in self.figures:
            try:
                self._blit(fig)
            except Exception:
                # blitが使えない環境では通常描画にフォールバック
                self._bg.pop(fig, None)
                fig.canvas.draw_idle()

    @staticmethod
    def _segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    def _mode_colors(self, run: RunLog, k: int) -> list:
        return [MODE_COLORS.get(m, "0.5") for m in run.mode[:max(k - 1, 0)]]

    def _update_curve(self, i: int, run: RunLog, k: int) -> None:
        if k >= 2:
            self.curve_lcs[i].set_segments(self._segments(run.position[:k], run.speed[:k]))
            self.curve_lcs[i].set_color(self._mode_colors(run, k))
        else:
            self.curve_lcs[i].set_segments([])
        if run.has_forward and k >= 1:
            self.curve_fw[i].set_data(run.fw_position[:k], run.fw_speed[:k])
        else:
            self.curve_fw[i].set_data([], [])
        if k >= 1:
            self.curve_head[i].set_data([run.position[k - 1]], [run.speed[k - 1]])
        else:
            self.curve_head[i].set_data([], [])

    def _update_diagram(self, i: int, run: RunLog, k: int) -> None:
        if k >= 2:
            self.diag_lcs[i].set_segments(self._segments(run.time[:k], run.position[:k]))
            self.diag_lcs[i].set_color(self._mode_colors(run, k))
        else:
            self.diag_lcs[i].set_segments([])
        if run.has_forward and k >= 1:
            self.diag_fw[i].set_data(run.time[:k], run.fw_position[:k])
        else:
            self.diag_fw[i].set_data([], [])
        if k >= 1:
            self.diag_head[i].set_data([run.time[k - 1]], [run.position[k - 1]])
        else:
            self.diag_head[i].set_data([], [])

    def _label_anchor(self, x: float, backward: bool) -> tuple:
        """列車ラベルの位置と揃え方を返す。

        backward=True（自列車）は後方＝左寄せ、False（先行列車）は前方＝右寄せを基本とし、
        軸端に寄って見切れる場合は反対側へ折り返す。
        """
        xmin, xmax = self._anim_xlim
        margin = (xmax - xmin) * 0.16
        if backward:
            if x - self._train_w - margin < xmin:
                return x, "left"
            return x - self._train_w, "right"
        if x + margin > xmax:
            return x - self._train_w, "right"
        return x, "left"

    def _update_anim(self, i: int, run: RunLog, k: int) -> None:
        """自列車を描く。位置は先頭位置なので、車体は後方（−列車長）へ伸ばす。"""
        body, y = self.anim_bodies[i]
        j = max(k - 1, 0)
        x = float(run.position[j])
        body.set_x(x - self._train_w)
        body.set_facecolor(MODE_COLORS.get(run.mode[j], FG_DIM))
        # 先行列車と接近したときにラベルが重ならないよう、自列車のラベルは後方（左）へ寄せる。
        # ただし軸の左端に近い場合は見切れるので前方寄せに切り替える。
        lx, ha = self._label_anchor(x, backward=True)
        self.anim_labels[i].set_position((lx, y + self._train_h + 0.06))
        self.anim_labels[i].set_ha(ha)
        prefix = "自列車" if i == 0 else "比較"
        self.anim_labels[i].set_text(f"{prefix} {run.speed[j]:.0f}km/h")
        # CBTC停止限界の帯（自列車先頭から前方50m）
        cbtc, cy = self.anim_cbtc[i]
        cbtc.set_x(x)
        cbtc.set_visible(run.has_forward)

    def _update_forward_anim(self, sim_t: float) -> None:
        for i, run in enumerate(self.runs):
            body, y = self.anim_fw_bodies[i]
            k = int(np.clip(np.searchsorted(run.time, sim_t, side="right"), 1, len(run.time)))
            j = k - 1
            x = float(run.fw_position[j]) if run.has_forward else np.nan
            if not np.isfinite(x):
                body.set_visible(False)
                self.anim_fw_labels[i].set_text("")
                self.anim_gaps[i].set_text("")
                continue
            body.set_visible(True)
            body.set_x(x - self._train_w)
            lx, ha = self._label_anchor(x, backward=False)
            self.anim_fw_labels[i].set_position((lx, y + self._train_h + 0.34))
            self.anim_fw_labels[i].set_ha(ha)
            self.anim_fw_labels[i].set_text(f"先行 {run.fw_speed[j]:.0f}km/h")
            # 車間は先頭位置どうしの距離（environment2 の先行距離と同義）。
            # 括弧内は先行列車の最後尾と自列車の先頭の距離＝CBTC停止限界50mと比較すべき値。
            own = float(run.position[j])
            gap = (x - own) * 1000.0
            nose_to_tail = gap - TRAIN_LENGTH_KM * 1000.0
            # 中点に置くと軸端で見切れるので、表示範囲の内側にクランプする
            xmin, xmax = self._anim_xlim
            margin = (xmax - xmin) * 0.13
            gx = float(np.clip((x + own) / 2.0, xmin + margin, xmax - margin))
            self.anim_gaps[i].set_position((gx, y + self._train_h + 0.64))
            self.anim_gaps[i].set_text(f"車間 {gap:.0f} m（後尾まで {nose_to_tail:.0f} m）")


# =============================================================================
# 自列車実況（Qtウィジェットに描くHTMLテーブル）
# =============================================================================
def build_info_html(infos: list) -> str:
    """[(RunLog, 進行インデックス), ...] から実況テーブルのHTMLを組み立てる。

    比較ログがある場合は「項目｜ログ1｜ログ2」の3列になり、値を横並びで見比べられる。
    """
    if not infos:
        return ""
    dim = f'<span style="color:{FG_DIM};">'
    rows = []
    for run, k in infos:
        j = int(np.clip(k - 1, 0, len(run.time) - 1))
        gap = run.fw_gap[j]
        mode = run.mode[j]
        grade = float(run.gradient[j])
        dwell = float(run.fw_dwell[j])
        if not run.has_forward:
            dwell_str = f'{dim}― （先行列車なし）</span>'
        elif np.isnan(dwell):
            dwell_str = f'{dim}―</span>'
        elif dwell <= 0.0:
            dwell_str = f'{dim}停車前／発車済み</span>'
        else:
            dwell_str = f"{dwell:.0f} s"
        rows.append({
            "モード": (f'<span style="color:{MODE_COLORS.get(mode, FG_DIM)};">■</span> '
                     f'{MODE_LABELS.get(mode, mode)}'),
            "ノッチ": ACTION_LABELS.get(int(run.action[j]), "-"),
            "現在速度": f"{run.speed[j]:.1f} km/h",
            "信号現示": f"{run.signal[j]:.1f} km/h",
            # 上り勾配を正、下り勾配を負として表示（environment2 と同符号）
            "勾配": f"{grade:+.1f} ‰",
            "先行距離": (f"{gap:.0f} m" if np.isfinite(gap)
                     else f'{dim}― （先行列車なし）</span>'),
            "先行停車経過": dwell_str,
            "駅残距離": f"{run.station_dist[j]:.0f} m",
            "残り時間": f"{run.remaining_time[j]:.1f} s",
        })

    multi = len(rows) > 1
    html = [f'<table cellspacing="0" cellpadding="4" width="100%" '
            f'style="color:{FG_TEXT};">']
    if multi:
        heads = "".join(f'<th align="right">{run.label}</th>' for run, _ in infos)
        html.append(f'<tr><th align="left" style="color:{FG_DIM};">項目</th>{heads}</tr>')
    for key in rows[0]:
        cells = "".join(f'<td align="right"><b>{row[key]}</b></td>' for row in rows)
        html.append(f'<tr><td style="color:{FG_DIM};">{key}</td>{cells}</tr>')
    html.append("</table>")
    return "".join(html)


# =============================================================================
# メインウィンドウ
# =============================================================================
DARK_STYLESHEET = f"""
QWidget {{ background: {BG_FIG}; color: {FG_TEXT}; }}
QGroupBox {{
    border: 1px solid {GRID_COLOR}; border-radius: 6px; margin-top: 10px;
    padding-top: 8px; background: {BG_AXES};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {FG_DIM};
}}
QPushButton {{
    background: #1d2330; border: 1px solid {GRID_COLOR}; border-radius: 4px;
    padding: 5px 12px; color: {FG_TEXT};
}}
QPushButton:hover {{ background: #293142; }}
QPushButton:pressed {{ background: #333d52; }}
QPushButton:disabled {{ color: #55606f; border-color: #1c2029; }}
QDoubleSpinBox {{
    background: {BG_AXES}; border: 1px solid {GRID_COLOR}; border-radius: 4px;
    padding: 3px 6px; color: {FG_TEXT};
}}
QSlider::groove:horizontal {{ height: 5px; background: {GRID_COLOR}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {FORWARD_COLOR}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {FG_TEXT}; width: 12px; margin: -5px 0; border-radius: 6px;
}}
QStatusBar {{ color: {FG_DIM}; }}
QLabel {{ background: transparent; }}
"""


class MonitorWindow(QtWidgets.QMainWindow):

    FPS = 20  # 画面更新レート[Hz]

    def __init__(self, initial_paths: Optional[list] = None):
        super().__init__()
        self.setWindowTitle("運転曲線モニター")
        self.resize(1440, 900)
        self.setStyleSheet(DARK_STYLESHEET)
        self.runs: list = []
        self.sim_t = 0.0
        self.t_start = 0.0
        self.t_end = 1.0
        self._slider_busy = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addLayout(self._build_file_bar())

        # 上段：運転曲線・ダイヤグラム
        self.fig_top = Figure(figsize=(14, 5.4), dpi=100)
        self.canvas_top = FigureCanvas(self.fig_top)
        layout.addWidget(self.canvas_top, stretch=5)

        # 下段：左＝列車の動き、右＝自列車実況（Qtウィジェット）
        lower = QtWidgets.QHBoxLayout()
        self.fig_bottom = Figure(figsize=(8, 3.2), dpi=100)
        self.canvas_bottom = FigureCanvas(self.fig_bottom)
        lower.addWidget(self.canvas_bottom, stretch=3)
        lower.addWidget(self._build_info_panel(), stretch=2)
        layout.addLayout(lower, stretch=3)

        self.monitor = MonitorFigure(self.fig_top, self.fig_bottom, self._set_info)
        layout.addLayout(self._build_control_bar())

        self.status = self.statusBar()
        self.status.showMessage("CSVを読み込んでください")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / self.FPS))
        self.timer.timeout.connect(self._on_tick)
        # 実経過時間で進める。描画が間に合わずフレーム落ちしても倍速指定どおりの速さになる。
        self._clock = QtCore.QElapsedTimer()

        if initial_paths:
            self._load_paths(initial_paths)

    # ------------------------------------------------------------------- UI構築
    def _build_file_bar(self) -> QtWidgets.QLayout:
        bar = QtWidgets.QHBoxLayout()
        self.path_labels = []
        for i, name in enumerate(("ログ1", "ログ2（比較・任意）")):
            btn = QtWidgets.QPushButton(f"{name} を開く…")
            btn.clicked.connect(lambda _, slot=i: self._choose_file(slot))
            lbl = QtWidgets.QLabel("（未選択）")
            lbl.setStyleSheet(f"color:{FG_DIM};")
            lbl.setMinimumWidth(220)
            bar.addWidget(btn)
            bar.addWidget(lbl, stretch=1)
            self.path_labels.append(lbl)
        clear_btn = QtWidgets.QPushButton("比較をクリア")
        clear_btn.clicked.connect(self._clear_second)
        bar.addWidget(clear_btn)
        return bar

    def _build_info_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("自列車実況")
        inner = QtWidgets.QVBoxLayout(box)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setAlignment(QtCore.Qt.AlignTop)
        self.info_label.setStyleSheet(f"font-size: 13px; color:{FG_TEXT};")
        inner.addWidget(self.info_label)
        inner.addStretch(1)
        return box

    def _set_info(self, infos: list) -> None:
        self.info_label.setText(build_info_html(infos))

    def _build_control_bar(self) -> QtWidgets.QLayout:
        bar = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ 開始")
        self.btn_stop = QtWidgets.QPushButton("■ 停止")
        self.btn_reset = QtWidgets.QPushButton("⟲ リセット")
        for b in (self.btn_start, self.btn_stop, self.btn_reset):
            b.setMinimumWidth(110)
            b.setEnabled(False)
            bar.addWidget(b)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset)

        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("倍速:"))
        self.spin_speed = QtWidgets.QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 50.0)
        self.spin_speed.setSingleStep(0.5)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setDecimals(1)
        self.spin_speed.setSuffix(" ×")
        self.spin_speed.setMinimumWidth(90)
        bar.addWidget(self.spin_speed)

        bar.addSpacing(16)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        bar.addWidget(self.slider, stretch=1)

        self.time_label = QtWidgets.QLabel("t =    0.0 s")
        self.time_label.setMinimumWidth(120)
        font = QtGui.QFont("monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.time_label.setFont(font)
        bar.addWidget(self.time_label)
        return bar

    # ---------------------------------------------------------------- ファイル
    def _choose_file(self, slot: int) -> None:
        start_dir = os.path.join(BASE_DIR, "data")
        if self.runs:
            start_dir = os.path.dirname(self.runs[0].path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "テストケースのログCSVを選択", start_dir, "CSVファイル (*.csv);;すべて (*)")
        if not path:
            return
        paths = [r.path for r in self.runs]
        while len(paths) <= slot:
            paths.append(None)
        paths[slot] = path
        self._load_paths([p for p in paths if p])

    def _clear_second(self) -> None:
        if len(self.runs) > 1:
            self._load_paths([self.runs[0].path])

    def _load_paths(self, paths: list) -> None:
        runs, notes = [], []
        for path in paths[:2]:
            try:
                run = load_run(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "読み込みエラー",
                                               f"{os.path.basename(path)} を読み込めませんでした。\n\n{e}")
                return
            runs.append(run)
            notes.extend(run.notes)
        if not runs:
            return

        self.stop()
        self.runs = runs
        self.monitor.set_runs(runs)
        self.t_start = min(float(r.time[0]) for r in runs)
        self.t_end = max(float(r.time[-1]) for r in runs)
        for i, lbl in enumerate(self.path_labels):
            lbl.setText(os.path.basename(runs[i].path) if i < len(runs) else "（未選択）")
        for b in (self.btn_start, self.btn_stop, self.btn_reset):
            b.setEnabled(True)
        self.slider.setEnabled(True)
        self.reset()

        msg = f"{len(runs)}件を読み込みました（{self.t_start:.1f}〜{self.t_end:.1f} s）"
        if notes:
            msg += " ｜ " + " ／ ".join(dict.fromkeys(notes))
        self.status.showMessage(msg)

    # ------------------------------------------------------------------ 再生制御
    def start(self) -> None:
        if not self.runs:
            return
        if self.sim_t >= self.t_end:
            self.sim_t = self.t_start
        self._clock.start()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    def reset(self) -> None:
        self.stop()
        self.sim_t = self.t_start
        self._refresh()

    def _on_tick(self) -> None:
        # 実経過時間[ms]×倍速ぶんだけ進める（1フレームあたりの進みは最大1秒相当に制限）
        elapsed = min(self._clock.restart() / 1000.0, 1.0)
        self.sim_t += elapsed * self.spin_speed.value()
        if self.sim_t >= self.t_end:
            self.sim_t = self.t_end
            self.stop()
        self._refresh()

    def _on_slider(self, value: int) -> None:
        if self._slider_busy or not self.runs:
            return
        self.stop()
        self.sim_t = self.t_start + (self.t_end - self.t_start) * value / 1000.0
        self._refresh(update_slider=False)

    def _refresh(self, update_slider: bool = True) -> None:
        if not self.runs:
            return
        self.monitor.update(self.sim_t)
        self.time_label.setText(f"t = {self.sim_t:6.1f} s")
        if update_slider:
            span = max(self.t_end - self.t_start, 1e-9)
            self._slider_busy = True
            self.slider.setValue(int((self.sim_t - self.t_start) / span * 1000))
            self._slider_busy = False


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="テストケースのログCSVを再生する運転曲線モニター")
    parser.add_argument("csv", nargs="*", help="ログCSV（1〜2件。2件目は比較用）")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    font_name = _setup_japanese_font()
    if font_name:
        app.setFont(QtGui.QFont(font_name, 10))
    else:
        print("[警告] 日本語フォントが見つかりませんでした。文字が豆腐表示になる場合があります。")

    window = MonitorWindow(initial_paths=args.csv)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
