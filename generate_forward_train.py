"""先行列車の走行パターンCSV（`input/f_train/`）を生成するスクリプト。

【2026-08-14 変更】「惰行ポイント方式」に全面変更した。
従来は目標速度を定速保持する定速走行パターンだったが、先行列車も自列車と同じ
「力行 → 惰行 → 制動」の省エネ運転をしているものとして扱う。

  運転パターン: 出発 → 惰行速度 V まで力行 → 惰行 → 駅に向かって制動 → 停車（次駅停車時間）
                → 再出発（以降は V 付近を保持して走行）

  V（惰行ポイント）の使い分け:
    - V=65km/h … `generate_standard_curve.py` が求めた省エネ最適の標準運転曲線と一致する
                  （標準運転時間180秒でちょうど白兎に停止する）。「先行が標準運転曲線で
                  走ったパターン」の検証ケース。
    - V=50km/h … 標準より遅い運転をしている先行列車の検証ケース。
    - V=40〜65km/h … 学習時に`apex2.py`のActorがランダムに選ぶ範囲。

  【再加速について】この駅間（羽前成田21.112km→白兎23.29km、2.178km）は白兎手前が
  上り勾配（6.1→9.2‰）のため、惰行だけで駅の制動開始点まで到達できるのは V≒62km/h
  以上に限られる。V がそれより低い場合は惰行中に駅間停車してしまうため、
  「このまま惰行を続けても制動開始点に届かない」と判定した時点で V まで再加速する
  （`_coast_reaches_brake_point`）。結果として、
    - V=65 … 純粋な 力行→惰行→制動（再加速なし＝標準運転曲線）
    - V<65 … 力行と惰行を繰り返して V 付近を保ちつつ進み、届く位置まで来たら最後の惰行に入る
  という挙動になる。届くと判定した後は再加速しない（`coast_latched`）ので、
  「V で惰行して駅に向かって減速する」最終フェーズは V によらず必ず現れる。

先行列車の出発遅延は、このCSVでは表現しない（全パターンとも t=0 に出発する）。
遅延は`apex2.py`側で「出発間隔（headway）＝ 標準出発間隔120秒 − 先行遅延」に換算して
`environment2.reset(fowerd_train_time_offset=...)` に渡す（headway換算モデル）。

旧形式（定速走行・CSV内で出発遅延を表現）のCSVは `--legacy` で再生成できる。
`apex.py` / `apex3.py` が参照しているため `input/f_train_*.csv` は残してある。
"""

import argparse
import csv
import os

# train.py や actions.py が同じディレクトリにある前提でインポート
from train import Train
from actions import Actions

# ==== 路線・列車の設定値 ====
# 自列車と同じ駅（羽前成田: 21.112km）から出発させ、headwayぶん先行させて使う
START_POS = 21.112
# 先行列車の終着駅（CSVの記録範囲を走り切らせるための遠方の目標）
TARGET_STATION = 30.605
# 先行列車が停車する駅の位置（白兎: 23.29km）＝自列車の次駅
STOP_STATION_POS = 23.29
# 最大シミュレーション秒数（エピソードが途切れないように長めに設定）
TOTAL_STEPS = 1200

# ==== 制御の設定値 ====
# ノッチ判断の周期[s]。train.py の積分刻み（0.01秒）と揃えることで、
# 制動開始点を0.01秒精度で置ける（1秒粗いと停止位置が十数mずれる）。
CONTROL_DT = 0.01
# 惰行から再加速に移るヒステリシス幅[km/h]（V-REACCEL_BAND まで落ちたら再加速を検討）
REACCEL_BAND = 3.0
# 惰行のみで到達可能かを判定する試行シミュレーションの刻み[s]と、
# 「駅間停車しそう」とみなす速度[km/h]。惰行を続けて制動開始点に達する前にこの速度を
# 下回るなら再加速する。完全停車(0km/h)を閾値にすると駅の手前を3〜8km/hで這って進む
# 非現実的なパターンが混ざるため、15km/hを下回る時点で「停車しそう」と判定する。
# （V=65の標準運転曲線は21.6km/hで制動に入るため、この閾値では再加速が起きず影響しない）
PROBE_DT = 1.0
PROBE_STALL_SPEED = 15.0
# 制動開始点の監視を始める駅までの距離[km]（これより手前では停止距離を計算しない＝高速化）
BRAKE_WATCH_DIST = 0.5

# ==== 生成するパターン ====
OUT_DIR = "input/f_train"
# 検証（Tester）用の惰行ポイント[km/h]: 65=標準運転曲線相当, 50=標準より遅い運転
TEST_COAST_SPEEDS = [65, 50]
# 学習（Actor）用の惰行ポイント[km/h]: 40〜65を1km/h刻みでランダム選択する
TRAIN_COAST_SPEEDS = list(range(40, 66))
# 次駅（白兎）での停車時間[s]
DWELL_TIMES = [30, 45, 60]


def coast_pattern_csv_path(coast_speed, dwell_time_sec):
    """惰行ポイント方式の先行列車CSVのパスを返す（apex2.pyと共有する命名規約）。"""
    return os.path.join(OUT_DIR, f"coast{int(coast_speed)}_stop{int(dwell_time_sec)}.csv")


def _action_str(action):
    if action == Actions.acceleration:
        return "Actions.acceleration"
    if action == Actions.deceleration:
        return "Actions.deceleration"
    return "Actions.coasting"


def _coast_reaches_brake_point(position, speed, stop_pos):
    """現在の位置・速度から惰行のみで駅の制動開始点に到達できるかを判定する。

    到達できる（=このまま惰行して駅に停止できる）なら True、
    途中で PROBE_STALL_SPEED を下回ってしまう（＝駅間停車しそう）なら False。
    停止距離の計算は重いので、駅まで300m以内に入ってからのみ行う
    （65km/hでも停止距離は約120mなので、それ以遠で制動開始点に達することはない）。
    """
    sim = Train(TARGET_STATION, position=position, speed=speed)
    for _ in range(int(600 / PROBE_DT)):
        dist = stop_pos - sim.position
        if dist <= 0.0:
            return True
        if dist <= 0.3 and dist <= sim.req_stop_dist:
            return True
        if sim.speed < PROBE_STALL_SPEED:
            return False
        sim.step(Actions.coasting, PROBE_DT)
    return False


def generate_coast_pattern_csv(filename, coast_speed, dwell_time_sec, stop_pos=STOP_STATION_POS):
    """惰行ポイント方式（力行→惰行→制動→停車→再出発）の先行列車CSVを生成する。"""
    train = Train(TARGET_STATION, position=START_POS, speed=0.0)

    ticks_per_sec = int(round(1.0 / CONTROL_DT))
    phase = "power"          # power / coast / brake / dwell
    stopped_done = False     # 次駅での停車を終えたか（以降は制動開始点を監視しない）
    coast_latched = False    # 惰行のみで制動開始点に届くと判定済み（＝もう再加速しない）
    dwell_elapsed = 0.0
    probe_fail_pos = None    # 直近に「届かない」と判定した位置[km]（近すぎる再判定を省く）
    # 制動開始点の判定を次に行うtick（停止距離の計算は重いので余裕に応じて間引く）
    next_brake_check = 0

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "position", "speed", "action"])

        for k in range(TOTAL_STEPS * ticks_per_sec):
            if phase == "dwell":
                # ① 次駅での停車（ブレーキを込めたまま停車時間を消化する）
                action = Actions.deceleration
                dwell_elapsed += CONTROL_DT
                if dwell_elapsed >= dwell_time_sec:
                    phase = "power"      # 再出発。以降は停車駅がないので V 付近を保持して走る
                    stopped_done = True
            elif phase == "brake":
                # ② 駅に向かって制動中
                action = Actions.deceleration
                if train.speed <= 0.0:
                    phase = "dwell"
                    dwell_elapsed = 0.0
                    # 停止位置の残差（数cm）を吸収して駅位置に正確に据える
                    if abs(train.position - stop_pos) < 0.005:
                        train.set_states(0.0, stop_pos)
            else:
                dist = stop_pos - train.position
                # ③ 制動開始点に達したか（次駅がまだ前方にある場合のみ監視）
                if (not stopped_done) and dist <= BRAKE_WATCH_DIST and k >= next_brake_check:
                    margin_km = dist - train.req_stop_dist
                    if margin_km <= 0.0:
                        phase = "brake"
                    else:
                        # 余裕を使い切るまでの時間の半分だけ次の判定を先送りする
                        v_kmh = max(train.speed, 1.0)
                        wait_ticks = int(margin_km / v_kmh * 3600.0 / CONTROL_DT * 0.5)
                        next_brake_check = k + max(1, wait_ticks)

                if phase == "brake":
                    action = Actions.deceleration
                elif phase == "power":
                    # ④ 惰行ポイントまで力行
                    action = Actions.acceleration
                    if train.speed >= coast_speed:
                        phase = "coast"
                        action = Actions.coasting
                else:
                    # ⑤ 惰行。駅間停車しそうなら惰行ポイントまで再加速する
                    action = Actions.coasting
                    if train.speed <= coast_speed - REACCEL_BAND:
                        if stopped_done:
                            # 停車後は次の停車駅がないので、単純に V 付近を保持して走る
                            phase = "power"
                        elif not coast_latched:
                            # 「50m以上進んでから」再判定する（同じ結論の再計算を省く）
                            if probe_fail_pos is None or train.position - probe_fail_pos >= 0.05:
                                if _coast_reaches_brake_point(train.position, train.speed, stop_pos):
                                    coast_latched = True     # 最後の惰行に入る
                                else:
                                    probe_fail_pos = train.position
                                    phase = "power"
                            else:
                                phase = "power"

            if k % ticks_per_sec == 0:
                writer.writerow([k // ticks_per_sec, round(train.position, 6),
                                 round(train.speed, 2), _action_str(action)])

            train.step(action, CONTROL_DT)

    print(f"Generated: {filename}")


# ============================================================================
# 旧形式（定速走行パターン）の生成。apex.py / apex3.py が参照する
# input/f_train_low50.csv・input/f_train_delay{N}_stop{M}.csv を再生成する用。
# ============================================================================
def generate_forward_train_csv(filename, target_speed, delay_sec, stop_pos=None, stop_time_sec=0):
    # 先行列車のインスタンスを生成
    train = Train(TARGET_STATION, position=START_POS, speed=0.0)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'position', 'speed', 'action'])

        has_stopped = False
        stop_timer = 0

        for t in range(TOTAL_STEPS):
            action = Actions.coasting

            # 1. 出発前の遅延（待機）フェーズ
            if t < delay_sec:
                action = Actions.deceleration
            else:
                # 2. 駅停車フェーズ (stop_pos が指定されている場合)
                if stop_pos is not None and not has_stopped:
                    dist_to_stop = stop_pos - train.position

                    # ▼▼▼ 修正: ブレーキ開始距離を現在の速度から動的に逆算する ▼▼▼
                    v_ms = train.speed / 3.6
                    decel_ms2 = 2.4 / 3.6  # 減速度 2.4 km/h/s
                    # 必要なブレーキ距離(km) ＋ 余裕マージン(10m)
                    req_brake_dist = ((v_ms ** 2) / (2 * decel_ms2)) / 1000.0 + 0.01

                    # 必要な距離に入ったらブレーキ開始
                    if dist_to_stop <= req_brake_dist:
                        action = Actions.deceleration

                        # 完全に停車したらタイマーを回す
                        if train.speed <= 0.0:
                            stop_timer += 1
                            action = Actions.deceleration
                            if stop_timer >= stop_time_sec:
                                has_stopped = True # 停車時間完了、再出発へ
                    else:
                        if train.speed < target_speed:
                            action = Actions.acceleration
                        elif train.speed > target_speed + 2.0:
                            action = Actions.deceleration
                        else:
                            action = Actions.coasting

                # 3. 通常走行フェーズ (停車完了後、または停車駅なし)
                else:
                    if train.speed < target_speed:
                        action = Actions.acceleration
                    elif train.speed > target_speed + 2.0:
                        action = Actions.deceleration
                    else:
                        action = Actions.coasting

            # 現在の時刻、位置、速度、行動を記録
            writer.writerow([t, round(train.position, 6), round(train.speed, 2), _action_str(action)])

            # train.py の物理モデルを1ステップ(1.0秒)進める
            train.step(action, 1.0)

    print(f"Generated: {filename}")


def main():
    parser = argparse.ArgumentParser(description="先行列車の走行パターンCSVを生成する")
    parser.add_argument("--legacy", action="store_true",
                        help="旧形式（定速走行）の input/f_train_*.csv を生成する")
    args = parser.parse_args()

    if args.legacy:
        print("=== 先行列車用CSV（旧形式・定速走行）の生成を開始します ===")
        # ① Sim3_Low50 (遅延なし、50km/hで定速走行、駅停車なし)
        generate_forward_train_csv("input/f_train_low50.csv", target_speed=50.0, delay_sec=0)
        # ② Sim3_Delay_Stop (先行列車遅延 [0, 5, 10] × 停車時間 [30, 45, 60])
        for f_delay in [0, 5, 10]:
            for stop_time in [30, 45, 60]:
                filename = f"input/f_train_delay{f_delay}_stop{stop_time}.csv"
                generate_forward_train_csv(filename, target_speed=50.0, delay_sec=f_delay,
                                           stop_pos=STOP_STATION_POS, stop_time_sec=stop_time)
        return

    print("=== 先行列車用CSV（惰行ポイント方式）の生成を開始します ===")
    # 惰行ポイント 40〜65km/h × 次駅停車時間 [30, 45, 60] の全組み合わせ。
    # うち 65（標準運転曲線相当）と 50 がTesterの検証ケース、全体がActorの学習用。
    speeds = sorted(set(TRAIN_COAST_SPEEDS) | set(TEST_COAST_SPEEDS))
    for coast_speed in speeds:
        for dwell in DWELL_TIMES:
            generate_coast_pattern_csv(coast_pattern_csv_path(coast_speed, dwell),
                                       coast_speed=float(coast_speed), dwell_time_sec=float(dwell))
    print(f"=== 完了: {len(speeds) * len(DWELL_TIMES)} 件を {OUT_DIR}/ に出力しました ===")


if __name__ == "__main__":
    main()
