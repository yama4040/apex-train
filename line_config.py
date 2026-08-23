# -*- coding: utf-8 -*-
"""
複数駅間最適化版の路線・車両設定（設計: docs_複数駅間最適化_計画.md §6・§4.7）

【重要】既存の track.py / train.py / input/*.csv は**一切変更しない**（計画書 §9.1）。
このモジュールは複数駅間版（`*_multi.py`）だけが参照する。

路線データの座標系について（§6.2）
    東京メトロ東西線は**下り列車（B線）**のため、キロ程が進行方向に沿って減少する。
    既存 track.py は位置の単調増加を前提としているため、読み込み時に
        x_internal = x0 − x_csv   （x0 = 起点駅のキロ程）
    で「進行方向に単調増加する内部座標[km]」へ変換する。**勾配の符号はそのまま保つ**
    （資料の勾配は進行方向基準であることを実データで確認済み。§6.3）。

速度制限データの意味（§6.2.1）
    Tozai_line_speed_limit.csv の位置は**CS-ATCの信号現示が切り替わる位置**であり、
    列車の**先頭**が越えた時点で現示が変わる。物理的な制限区間ではないため、
    列車長は考慮しない。
"""

# =============================================================================
# 車両
# =============================================================================
VEHICLES = {
    # 東京メトロ 東西線 15000系（10両編成・5M5T）
    #   出典: 15000系_車両情報.pdf（東京地下鉄 鉄道本部 車両部設計課）
    #   走行抵抗係数 A/B/C は指定値
    "metro15000": {
        "name": "東京メトロ15000系(10両)",
        # --- 編成 ---
        "car_length_m": 20.0,
        "n_cars": 10,
        "train_length_km": 0.200,        # 20m × 10両（CBTC車間・可視化に使う。§6.6）
        # --- 走行抵抗 R(v) = A + B·v + C·v²  [kg/t] ---
        "res_a": 2.089,
        "res_b": 0.0394,
        "res_c": 0.000675,
        # --- 引張力（定トルク → 定出力 → 特性域の3領域） ---
        #   F0 は起動加速度 3.3 km/h/s（定員時）を満たす値として逆算する。
        #   V1/V2 はPDFに記載が無いため一般的な通勤電車の値（§13-2 の感度確認対象）。
        "accel_start": 3.3,              # 起動加速度[km/h/s]（定員時）
        "tf_v1": 40.0,                   # 定トルク域の終端[km/h]
        "tf_v2": 75.0,                   # 定出力域の終端[km/h]
        # --- 制動 ---
        #   PDFの 3.5(常用最大) / 5.0(非常) は使わない。駅停車で使う一般的な制動は 2.5。
        #   ATCパターンも同じ 2.5 で構築すること（§6.5）。
        "decelerate": -2.5,              # 制動ノッチ B1 の減速度[km/h/s]
        "decel_service_max": -3.5,       # 常用最大（参考値。現状は未使用）
        "decel_emergency": -5.0,         # 非常（参考値。現状は未使用）
        "design_max_speed": 110.0,
        # --- 中間ノッチ（§4.7）---
        #   P2: 引張力 = min(TF(v), R(v) + GRADE_COMP) → 勾配 g での加速度 = (GRADE_COMP − g)/FI
        #   B2: 制動ノッチ = (GRADE_COMP − R(v))/FI    → 勾配 g での加速度 = −(GRADE_COMP + g)/FI
        "grade_comp": 35.0,              # 打ち消す勾配[‰]（+35‰でP2が定速、−35‰でB2が定速）
    },
    # 既存（山形鉄道1両編成28t相当）— 比較・回帰確認用。train.py と同一の値。
    "yamagata": {
        "name": "既存モデル(1両28t)",
        "car_length_m": 20.0, "n_cars": 1, "train_length_km": 0.020,
        "res_a": 2.39, "res_b": 0.0224, "res_c": 0.00062,
        "accel_start": None,             # 引張力は折れ線で直接定義（下記 legacy_tf）
        "tf_v1": None, "tf_v2": None,
        "decelerate": -2.5,
        "decel_service_max": None, "decel_emergency": None,
        "design_max_speed": 110.0,
        "grade_comp": 35.0,
        "legacy_tf": True,               # train.py の3本の直線をそのまま使う
    },
}

# =============================================================================
# 路線
# =============================================================================
LINES = {
    "tozai": {
        "name": "東京メトロ東西線(下り)",
        "dir": "input/Tozai_line",
        "station_csv": "Tozai_line_Station.csv",
        "grade_csv": "Tozai_line_grade.csv",
        "limit_csv": "Tozai_line_speed_limit.csv",
        "curve_csv": None,               # 曲線データなし（曲線抵抗0で扱う。§6.2.2）
        "direction": "descending",       # キロ程が進行方向に減少する（下りB線）
        "grade_limit": 40.0,             # |勾配| がこれを超えたら例外（既存は30で+35‰を握り潰す。§6.2(2)）
        "std_departure_interval": 140.0, # 標準運転間隔[秒]（ラッシュ時2分20秒）
        "std_dwell": 30.0,               # 標準停車時間[秒]
        "vehicle": "metro15000",
        "atc_mode": "anticipatory",      # (a)予見型（既定）／"pattern" で(b)追従型（§7.4.1）
        # 3駅スコープ（東陽町・木場・門前仲町）。茅場町まで含めたい場合は3→4に伸ばす
        "default_stations": [0, 1, 2],
    },
    "yamagata": {
        "name": "山形鉄道フラワー長井線",
        "dir": "input",
        "station_csv": "Station.csv",
        "grade_csv": "grade.csv",
        "limit_csv": "speed_limit.csv",
        "curve_csv": "curve.csv",
        "direction": "ascending",
        "grade_limit": 30.0,             # 既存 track.py と同じ挙動を保つ
        "std_departure_interval": 120.0,
        "std_dwell": 30.0,
        "vehicle": "yamagata",
        "atc_mode": "anticipatory",
        "default_stations": [11, 12],    # 羽前成田→白兎
    },
}

# =============================================================================
# 共通定数
# =============================================================================
FACTOR_OF_INERTIA = 28.34467   # kg/t → km/h/s の換算（既存 train.py と同一。§6.5）
SUB_DT = 0.01                  # 物理積分の刻み[s]（既存 train.py の time_step_base と同一）
CBTC_STOP_LIMIT_KM = 0.050     # CBTC停止限界（先行の最後尾から自列車の先頭まで）[km]
GRAVITY = 9.80665

# ノッチ（§4.7）。値は Actions の整数値と一致させる（actions_multi.py）
NOTCH_ORDER = ["P1", "P2", "C", "B2", "B1"]
NOTCH_LABEL_JA = {"P1": "力行", "P2": "勾配力行", "C": "惰行", "B2": "勾配ブレーキ", "B1": "制動"}
NOTCH_LABEL_EN = {"P1": "Power", "P2": "GradePower", "C": "Coast", "B2": "GradeBrake", "B1": "Brake"}
NOTCH_COLOR = {"P1": "#d62728", "P2": "#ff7f0e", "C": "#2ca02c", "B2": "#17becf", "B1": "#1f77b4"}


def get_line(name):
    if name not in LINES:
        raise KeyError(f"未知の路線: {name}（利用可能: {list(LINES)}）")
    return LINES[name]


def get_vehicle(name):
    if name not in VEHICLES:
        raise KeyError(f"未知の車両: {name}（利用可能: {list(VEHICLES)}）")
    return VEHICLES[name]
