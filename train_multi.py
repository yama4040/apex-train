# -*- coding: utf-8 -*-
"""
複数駅間最適化版の列車運動モデル（設計: docs_複数駅間最適化_計画.md §4.7・§6.5）

既存 train.py（山形28t・3ノッチ・物理定数がハードコード）は**一切変更しない**。
本モジュールは以下を新規に備える。

  * 車両パラメータの注入（`line_config.VEHICLES`）
  * **走行抵抗・制動減速度・引張力を1箇所に集約**（既存は4箇所に重複していた。§6.5末尾）
  * **5ノッチ**（P1/P2/C/B2/B1）
  * 路線の注入（`track_multi.TrackMulti`）

物理式は既存 train.py と同一構造:
    accel [km/h/s] = (F − R(v) − Rg − Rc) / FACTOR_OF_INERTIA  + （制動ノッチ）
"""
from actions_multi import ActionsMulti
import line_config as LC
from track_multi import TrackMulti

_TRACK_CACHE = {}


def get_track(line_name="tozai"):
    """路線ごとに1つだけ生成してキャッシュする（既存 train.get_shared_track の複数路線版）"""
    if line_name not in _TRACK_CACHE:
        _TRACK_CACHE[line_name] = TrackMulti(line_name)
    return _TRACK_CACHE[line_name]


class Vehicle:
    """車両特性。走行抵抗・引張力・制動をここに集約する。"""

    def __init__(self, name="metro15000"):
        v = LC.get_vehicle(name)
        self.key = name
        self.name = v["name"]
        self.A, self.B, self.C = v["res_a"], v["res_b"], v["res_c"]
        self.DECELERATE = v["decelerate"]
        self.GRADE_COMP = v["grade_comp"]
        self.train_length_km = v["train_length_km"]
        self.design_max_speed = v["design_max_speed"]
        self._legacy = bool(v.get("legacy_tf"))
        if not self._legacy:
            self.V1, self.V2 = v["tf_v1"], v["tf_v2"]
            # 起動加速度を満たす定トルク域の引張力
            self.F0 = v["accel_start"] * LC.FACTOR_OF_INERTIA + self.travel_resistance(0.0)
        self._check()

    # ---------------------------------------------------------------- 基本特性
    def travel_resistance(self, v):
        """走行抵抗[kg/t]"""
        return self.A + self.B * v + self.C * v * v

    def tractive_force(self, v):
        """引張力[kg/t]（P1 力行）"""
        if self._legacy:
            if v < 42.0:
                return -1.489 * v + 92.408
            if v < 68.0:
                return -0.4 * v + 46.68
            return -0.0963 * v + 26.0284
        if v <= self.V1:
            return self.F0
        if v <= self.V2:
            return self.F0 * self.V1 / v
        return self.F0 * self.V1 * self.V2 / (v * v)

    def tractive_force_p2(self, v):
        """引張力[kg/t]（P2 勾配力行）。**P1を超えないようクリップする**（§4.7.1）"""
        return min(self.tractive_force(v), self.travel_resistance(v) + self.GRADE_COMP)

    def brake_decel_b2(self, v):
        """B2 勾配ブレーキの減速ノッチ[km/h/s]（正の値）"""
        return (self.GRADE_COMP - self.travel_resistance(v)) / LC.FACTOR_OF_INERTIA

    def _check(self):
        """運用速度域で P2 < P1 であることを確認する（§4.7.1）。
        等しくなると2つの行動が同一結果になり、Q学習のmax演算子が過大評価を累積する（設計メモ §26）。"""
        if self._legacy:
            return
        bad = [v for v in range(0, int(self.design_max_speed) + 1, 5)
               if self.tractive_force(v) <= self.travel_resistance(v) + self.GRADE_COMP + 1e-9]
        if bad and bad[0] <= 80:
            raise RuntimeError(
                f"[{self.name}] 運用速度域で P2 の引張力が P1 に達します（{bad[0]} km/h 以上）。"
                f"2つの行動が同一になり学習が壊れます。grade_comp か引張力特性を見直してください。")
        self.p2_saturate_speed = bad[0] if bad else None

    # ---------------------------------------------------------------- 加速度
    def accel(self, action, v, grade, curve=0.0, wc=1.0):
        """ノッチ・速度・勾配抵抗・曲線抵抗から加速度[km/h/s]を返す（既存 train.py と同一構造）"""
        a = ActionsMulti(action)
        if a == ActionsMulti.power:
            f = self.tractive_force(v)
        elif a == ActionsMulti.grade_power:
            f = self.tractive_force_p2(v)
        else:
            f = 0.0
        acc = ((f - self.travel_resistance(v)) * wc - (grade + curve)) / LC.FACTOR_OF_INERTIA
        if a == ActionsMulti.braking:
            acc += self.DECELERATE * wc
        elif a == ActionsMulti.grade_brake:
            acc -= self.brake_decel_b2(v) * wc
        return acc


class TrainMulti:
    """列車。位置は**内部座標[km]**（進行方向に増加）で保持する。"""

    def __init__(self, target_station, position=0.0, speed=0.0,
                 line_name="tozai", vehicle_name=None, weight_correction=1.0):
        self.track = get_track(line_name)
        self.vehicle = Vehicle(vehicle_name or self.track.cfg["vehicle"])
        self.time_step_base = LC.SUB_DT
        self.TARGET_STATION = target_station
        self.WEIGTH_CORRECTION = weight_correction
        self.__speed = speed
        self.__position = position
        self.__pre_acceleration = 0.0

    def set_states(self, speed, position):
        self.__speed, self.__position = speed, position

    def step(self, action, time_step):
        n = int(round(time_step / self.time_step_base))
        for _ in range(n):
            if self.__position < 0:
                return
            acc = self.vehicle.accel(action, self.__speed,
                                     self.track.grade(self.__position),
                                     self.track.curve(self.__position),
                                     self.WEIGTH_CORRECTION)
            if self.__speed + acc * self.time_step_base >= 0:
                self.__position += (self.__speed / 3600.0) * self.time_step_base \
                                   + (acc / 3600.0) * (self.time_step_base ** 2)
                self.__speed += acc * self.time_step_base
            else:
                self.__position += (self.__speed ** 2) / (2.0 * abs(acc) * 3600.0) if acc < 0 else 0.0
                self.__speed = 0.0
        self.__pre_acceleration = acc

    @property
    def speed(self):
        return self.__speed

    @property
    def position(self):
        return self.__position

    @property
    def current_speed_limit(self):
        return self.track.atc_limit(self.__position)

    @property
    def grade_resistance(self):
        return self.track.grade(self.__position)

    @property
    def curve_resistance(self):
        return self.track.curve(self.__position)


if __name__ == "__main__":
    import line_config as LC2
    veh = Vehicle("metro15000")
    print(f"=== {veh.name} ===")
    print(f"F0(定トルク域引張力) = {veh.F0:.3f} kg/t  （起動加速度 3.3 km/h/s を満たす値）")
    print(f"P2 が P1 に達する速度: {veh.p2_saturate_speed} km/h（運用域75km/hの外であること）")
    print()
    print("5ノッチの加速度 [km/h/s]")
    hdr = "  勾配   |" + "".join(f"{LC2.NOTCH_LABEL_JA[c]:>10}" for c in LC2.NOTCH_ORDER)
    print(hdr)
    from actions_multi import FROM_CODE
    for g in (-35, -16.3, -14.2, -5, 0, 4, 27, 29.7, 35):
        row = f" {g:+6.1f}‰ |"
        for c in LC2.NOTCH_ORDER:
            a = veh.accel(FROM_CODE[c], 67.0, g)
            row += f"{a:+9.3f}" + ("*" if abs(a) < 0.005 else " ")
        print(row)
    print("  (* = 定速)")
