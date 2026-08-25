# -*- coding: utf-8 -*-
"""
勾配・曲線の位置変化を織り込んだ制動曲線（駅にちょうど停止する速度パターン）。

複数駅間版の各所（先行列車の生成・目標速度の算出・環境のCBTC現示・LLM評価用CSVの生成）が
**同一の制動モデル**を使うための共通モジュール。既存モジュールは読み取り専用でimportする。

  * 物理定数・走行抵抗・制動減速度は `generate_standard_curve.py`（＝`train.py`と検証済みで一致）から取る
  * 路線参照は `generate_standard_curve.TrackLookup`（＝`track.py` と区間境界の扱いまで一致）を使う

なぜ必要か:
    `train.Train.req_stop_dist` と `required_speed.brake_stop_distance_m` は
    **制動開始地点の勾配・曲線が停止まで一定**と仮定している。白兎→蚕桑は
    +11.4‰ の上りが 1km 続いたあと −2.3‰ に反転し、制動開始点の直前に R=400m の曲線があるため、
    この仮定では制動距離を最大 6m 誤り、先行列車パターンが駅を過走した（実測 5.75m）。
    駅から逆向きに積分すればプロファイルがそのまま入るので、数cm精度になる。
"""
import math
from bisect import bisect_left

import generate_standard_curve as _gsc
from track import Track

SUB_DT = _gsc.SUB_DT
FACTOR_OF_INERTIA = _gsc.FACTOR_OF_INERTIA
DECELERATE = _gsc.DECELERATE

_LOOKUP = None
_BRAKE_CACHE = {}


def get_lookup():
    """路線参照（勾配・曲線・制限速度）。プロセス内で1つだけ生成してキャッシュする。"""
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = _gsc.TrackLookup(Track())
    return _LOOKUP


class BrakeCurve:
    """停止位置 `stop_position` にちょうど停止する制動曲線 v_brake(x)。

    駅から逆向きに 0.01 秒刻みで積分し、位置一定間隔（既定1cm）のテーブルに載せ替える。
    """

    def __init__(self, stop_position, v_max=90.0, grid_km=1e-5, back_km=1.2, wc=1.0):
        lk = get_lookup()
        self.stop_position = float(stop_position)
        self.wc = wc
        xs, vs = [self.stop_position], [0.0]
        x, v = self.stop_position, 0.0
        while v < v_max and x > self.stop_position - back_km:
            rr = _gsc.travel_resistance(v)
            rg = lk.grade(x)
            rc = lk.curve(x)
            a = ((0.0 - rr) * wc - (rg + rc)) / FACTOR_OF_INERTIA + DECELERATE * wc
            if a >= 0.0:
                a = -0.0001
            v_prev = v - a * SUB_DT
            x_prev = x - (v_prev / 3600.0) * SUB_DT - (a / 3600.0) * (SUB_DT ** 2)
            x, v = x_prev, v_prev
            xs.append(x)
            vs.append(v)
        # 逆積分の結果は「位置が降順・速度が昇順」で並んでいる。
        # 速度からの逆引き（stop_distance）にはこの並びをそのまま使う。
        self._xs_by_v = list(xs)      # 位置（降順）＝速度昇順に対応
        self._vs_by_v = list(vs)      # 速度（昇順・先頭0）
        xs.reverse()
        vs.reverse()
        self._x0 = xs[0]
        self._dx = grid_km
        n = int((self.stop_position - self._x0) / grid_km) + 2
        # 位置一定間隔へ線形補間で載せ替える（参照コストを O(1) にする）
        table = []
        j = 0
        for i in range(n):
            gx = self._x0 + grid_km * i
            while j < len(xs) - 2 and xs[j + 1] < gx:
                j += 1
            span = xs[j + 1] - xs[j]
            r = 0.0 if span <= 0 else (gx - xs[j]) / span
            table.append(vs[j] + r * (vs[j + 1] - vs[j]))
        self._table = table

    def speed_at(self, position):
        """位置 position における「駅にちょうど停止するための速度」[km/h]。
        この値を上回っていれば直ちに制動が必要。曲線の範囲外（駅から遠い）は inf。"""
        i = int((position - self._x0) / self._dx)
        if i < 0:
            return math.inf
        if i >= len(self._table):
            return 0.0
        return self._table[i]

    def stop_distance(self, speed):
        """速度 speed[km/h] で制動を開始したときに停止までに要する距離[km]。
        制動曲線の逆引き（プロファイルを織り込んだ停止距離）。"""
        if speed <= 0.0:
            return 0.0
        vs, xs = self._vs_by_v, self._xs_by_v
        if speed >= vs[-1]:
            return max(0.0, self.stop_position - xs[-1])
        i = bisect_left(vs, speed)
        if i <= 0:
            return 0.0
        span = vs[i] - vs[i - 1]
        r = 0.0 if span <= 0 else (speed - vs[i - 1]) / span
        x = xs[i - 1] + r * (xs[i] - xs[i - 1])
        return max(0.0, self.stop_position - x)

    def brake_start_position(self, speed):
        """速度 speed[km/h] のときの制動開始位置[km]"""
        return self.stop_position - self.stop_distance(speed)


def get_brake_curve(stop_position, **kw):
    """停止位置ごとに1度だけ構築してキャッシュする。"""
    key = round(float(stop_position), 6)
    if key not in _BRAKE_CACHE:
        _BRAKE_CACHE[key] = BrakeCurve(stop_position, **kw)
    return _BRAKE_CACHE[key]


if __name__ == "__main__":
    import config_ymulti as CFG
    import codecs
    import pandas as pd
    with codecs.open("./input/Station.csv", "r", "utf-8", "ignore") as f:
        st = pd.read_csv(f)
    print("=== 制動曲線（勾配・曲線を織り込んだ停止距離）===")
    for idx in CFG.STATION_INDICES[1:]:
        pos = float(st["position"][idx])
        bc = get_brake_curve(pos)
        print(f"\n{CFG.STATION_NAMES_JA[idx]}（{pos} km）に停止する制動距離")
        for v in (20, 30, 40, 50, 55, 60, 65, 70):
            d = bc.stop_distance(v) * 1000.0
            # 既存の一定勾配モデル（required_speed）との差
            import required_speed as rs
            lk = get_lookup()
            x = bc.brake_start_position(v)
            d_flat = rs.brake_stop_distance_m(v, lk.grade(x) + lk.curve(x))
            print(f"  {v:3d} km/h → {d:7.2f} m （一定勾配近似 {d_flat:7.2f} m / 差 {d-d_flat:+6.2f} m）")
