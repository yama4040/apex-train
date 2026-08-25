# -*- coding: utf-8 -*-
"""
山形鉄道フラワー長井線・複数駅間版（羽前成田 → 白兎 → 蚕桑）の設定モジュール。

【絶対条件】既存の単一区間手法（apex2.py / environment2.py / evaluate_csv_with_llm.py /
train.py / track.py / actions.py / required_speed.py / 学習済みモデル）は**一切変更しない**。
本ファイル以降の `*_ymulti.py` はすべて新規であり、既存モジュールは
**読み取り専用でimportするだけ**である（物理モデルを完全に一致させるため）。

  ・物理（運動方程式・引張力・走行抵抗・制動2.5km/h/s） … `train.Train` をそのまま使う
  ・路線データ（勾配・曲線・制限速度）              … `track.Track` をそのまま使う
  ・行動空間                                        … `actions.Actions`（3ノッチ）をそのまま使う

東西線版（`*_multi.py`）とは別系統である。あちらは5ノッチ・東京メトロ15000系・下りB線であり、
本系統は3ノッチ・山形28t・上り座標。データ置き場も別にして絶対に混ぜない。
"""

# =============================================================================
# 対象区間
# =============================================================================
# input/Station.csv の行index。11=羽前成田 / 12=白兎 / 13=蚕桑
STATION_INDICES = [11, 12, 13]
STATION_NAMES_JA = {11: "羽前成田", 12: "白兎", 13: "蚕桑"}

# 区間ごとの標準運転時間[秒]。STATION_INDICES の隣接ペアに対応する。
#   [0] 羽前成田 → 白兎 : 180秒（input/Station.csv の rt と同じ。既存の標準運転曲線 V=65 と一致）
#   [1] 白兎   → 蚕桑   : 130秒（**本スクリプト群での設定値**。Station.csv の rt=180 は
#                          距離1377mに対して緩すぎ（最短118.8秒＝余裕61秒）、駅停車の遅延が
#                          下流に効かないため。input/Station.csv は書き換えない）
RUNNING_TIMES = [180.0, 130.0]

# 参考: Station.csv 上の rt（比較・検証用。制御には使わない）
RUNNING_TIMES_CSV = [180.0, 180.0]

# =============================================================================
# 駅停車
# =============================================================================
STD_DWELL = 30.0        # 標準停車時間[秒]。これで発車できるのが定時。
DWELL_MIN = 30.0        # 最低停車時間[秒]。乗降のためこれ未満では発車できない（行動を禁止する）
# 最大停車時間[秒]。到達したら強制発車し、エピソードの無限延長を構造的に防ぐ。
# 必要量は「先行が蚕桑を発車する時刻 − 白兎発車から停止限界までの所要（約118秒）」で決まる。
# 最悪ケース（先行が惰行50km/h・白兎60秒停車・蚕桑180秒停車・出発間隔60秒）で257秒必要なため
# 300秒とした（240秒では機外停車を避けられない組合せが残る）。
DWELL_MAX = 300.0
# 「標準停車からの超過がこの秒数に達したら最低評価」（ユーザー指定の評価基準）。
# 停車時間60秒＝標準+30秒 が減点の底。ただし先行が塞いでいる場合はこの減点を適用しない。
DWELL_PENALTY_FULL_S = 30.0

# 停車中のノッチ読み替え（3ノッチ）
#   力行   = 発車
#   制動   = 待機（停止保持）
#   惰行   = 禁止（待機と同一結果になり、Q学習のmax演算子が過大評価を累積するため。設計メモ §26）
DWELL_TIME_STEP = 1.0   # 停車中の制御周期[秒]。駅手前100mの0.1秒刻みを引きずらないよう明示指定する

# =============================================================================
# ダイヤ・列車
# =============================================================================
STD_DEPARTURE_INTERVAL = 120.0   # 羽前成田の標準列車出発間隔[秒]（既存 apex2.py と同値）
TRAIN_LENGTH_KM = 0.020          # 列車長[km]（1両編成20m。CLAUDE.md「その他備考」）
CBTC_STOP_LIMIT_KM = 0.050       # CBTC停止限界[km]（先行の最後尾 ← 自列車の先頭）
# 自列車の先頭が停止すべき点は「先行の先頭 −（列車長 + 停止限界）」＝先行の70m手前。
# ※既存 environment2.py は列車長の概念が無く50mしか引いていない（20m甘い）。
#   本系統では計画書 §6.6 に従い列車長を効かせる。
CBTC_HEAD_MARGIN_KM = TRAIN_LENGTH_KM + CBTC_STOP_LIMIT_KM   # = 0.070 km

# 衝突（異常接近）判定。停止限界にちょうど止まるのは正解挙動なので、その手前で判定する。
COLLISION_MARGIN_KM = 0.040

# =============================================================================
# 停止窓・終了条件
# =============================================================================
STOP_WINDOW_BEFORE_KM = 0.010    # 駅の手前側許容[km]（−10m）
STOP_WINDOW_AFTER_KM = 0.005     # 駅の先側許容[km]（+5m）
STOP_SPEED_KMH = 0.5             # 実質停止とみなす速度[km/h]
# タイムオーバー余裕[秒]。複数駅間では「待機が正解」のシナリオがあるため大きく取る
# （計画書 §4.5。DWELL_MAX 240秒 + 走行の遅れ分を吸収できる幅）。
TIME_OVER_MARGIN_S = 360.0

# =============================================================================
# 先行列車パターン（generate_forward_train_ymulti.py が生成）
# =============================================================================
F_TRAIN_DIR = "input/f_train_ymulti"
# 惰行ポイント[km/h]。学習は40〜65からランダム、検証は下記2点。
F_COAST_SPEEDS_TRAIN = list(range(40, 66))
F_COAST_SPEEDS_TEST = [65, 50]
# 白兎（＝自列車の中間駅B）での先行の停車時間[秒]
F_DWELL_B = [30, 45, 60]
# 蚕桑（＝自列車の終着駅C）での先行の停車時間[秒]。
# 120/180 は「急病人救護など」で長時間停車するシナリオ（＝自列車は白兎に留まるべき局面）。
F_DWELL_C = [30, 60, 120, 180]
# 先行列車を走らせ切るための遠方の目標駅位置[km]（荒砥30.605）
F_TARGET_POSITION = 30.605
F_TOTAL_SECONDS = 1500


def f_train_csv(coast_speed, dwell_b, dwell_c):
    """先行列車CSVのパス（generate_forward_train_ymulti.py と apex_ymulti.py で共有する命名規約）"""
    return f"{F_TRAIN_DIR}/coast{int(coast_speed)}_b{int(dwell_b)}_c{int(dwell_c)}.csv"


ALL_F_TRAIN_CSVS = [f_train_csv(v, b, c)
                    for v in F_COAST_SPEEDS_TRAIN
                    for b in F_DWELL_B
                    for c in F_DWELL_C]

# =============================================================================
# 運転モード
# =============================================================================
# 既存 reward_features.MODE_CLASSES（normal/delay_recovery/anti_mid_stop/spacing）に
# 第5のモード hold_at_station（駅停車中の発車判断）を加えたもの。
# 後続列車は本フェーズでは扱わないため spacing は非アクティブ（one-hotの枠だけ確保）。
MODE_CLASSES = ["normal", "delay_recovery", "anti_mid_stop", "spacing", "hold_at_station"]
MODE_CLASSES_ACTIVE = ["normal", "delay_recovery", "anti_mid_stop", "hold_at_station"]
MODE_DIM = len(MODE_CLASSES)
MODE_INDEX = {m: i for i, m in enumerate(MODE_CLASSES)}

# =============================================================================
# 出力先（既存系統と絶対に混ぜない）
# =============================================================================
STANDARD_CURVE_DIR = "standard_curve_ymulti"
EVAL_CSV_DIR = "評価用csv_Yamagata"
EVALUATED_DIR = "評価済ログ_Yamagata"
TRAIN_CSV_DIR = "train_reward_csv_direct_Yamagata"
DATA_DIR = "data_ymulti"

# 報酬NNの成果物（既存の direct_reward_model2.h5 等を絶対に上書きしないよう名前を分ける）
REWARD_MODEL_PATH = "direct_reward_model_ymulti.h5"
REWARD_GATE_PATH = "direct_reward_gate_ymulti.h5"
REWARD_SCALER_PATH = "direct_reward_scaler_ymulti.pkl"
REWARD_MANIFEST_PATH = "direct_reward_manifest_ymulti.json"


# =============================================================================
# ダイヤの通算（複数駅間）
# =============================================================================
def scheduled_arrival_times():
    """各駅の標準到着時刻[秒]（起点駅の発車を0とする通算ダイヤ）。

    戻り値の長さは len(STATION_INDICES)。先頭は 0.0（起点駅の発車時刻）。
      羽前成田発 0 → 白兎着 180 → （標準停車30秒）→ 白兎発 210 → 蚕桑着 340
    """
    times = [0.0]
    t = 0.0
    for k, rt in enumerate(RUNNING_TIMES):
        t += rt
        times.append(t)
        t += STD_DWELL          # 次の区間の発車は標準停車のぶん後ろへ
    return times


def scheduled_departure_times():
    """各駅の標準発車時刻[秒]（終着駅の要素は None）。"""
    arr = scheduled_arrival_times()
    out = []
    for k in range(len(arr)):
        if k == len(arr) - 1:
            out.append(None)
        elif k == 0:
            out.append(0.0)
        else:
            out.append(arr[k] + STD_DWELL)
    return out


def total_scheduled_time():
    """起点駅発車から終着駅到着までの累積標準ダイヤ[秒]"""
    return scheduled_arrival_times()[-1]


if __name__ == "__main__":
    import codecs as _c
    import pandas as _pd
    with _c.open("./input/Station.csv", "r", "utf-8", "ignore") as f:
        st = _pd.read_csv(f)
    print("=== 山形鉄道フラワー長井線・複数駅間版（3ノッチ） ===")
    arr = scheduled_arrival_times()
    dep = scheduled_departure_times()
    for k, idx in enumerate(STATION_INDICES):
        pos = float(st["position"][idx])
        d = f"{dep[k]:7.1f}" if dep[k] is not None else "      -"
        print(f"  [{idx}] {STATION_NAMES_JA[idx]:<6} {st['name'][idx]:<12} "
              f"位置 {pos:8.4f} km / 標準到着 {arr[k]:7.1f}s / 標準発車 {d}s")
    for k, rt in enumerate(RUNNING_TIMES):
        a = float(st["position"][STATION_INDICES[k]])
        b = float(st["position"][STATION_INDICES[k + 1]])
        print(f"  区間{k}: {STATION_NAMES_JA[STATION_INDICES[k]]} → "
              f"{STATION_NAMES_JA[STATION_INDICES[k+1]]}  {(b-a)*1000:7.1f} m / "
              f"標準運転時間 {rt:.0f}s（Station.csv値 {RUNNING_TIMES_CSV[k]:.0f}s）")
    print(f"  累積標準ダイヤ（起点発→終着着）: {total_scheduled_time():.0f} 秒")
    print(f"  停車: 標準{STD_DWELL:.0f}s / 最低{DWELL_MIN:.0f}s / 最大{DWELL_MAX:.0f}s")
    print(f"  CBTC: 先行の先頭から {CBTC_HEAD_MARGIN_KM*1000:.0f} m 手前が停止限界"
          f"（列車長{TRAIN_LENGTH_KM*1000:.0f}m + 停止限界{CBTC_STOP_LIMIT_KM*1000:.0f}m）")
    print(f"  先行パターン数: {len(ALL_F_TRAIN_CSVS)} 種")
