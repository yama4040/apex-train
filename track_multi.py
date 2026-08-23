# -*- coding: utf-8 -*-
"""
複数駅間最適化版の路線データローダ（設計: docs_複数駅間最適化_計画.md §6.2）

既存 track.py との違い:
  1. **進行方向に単調増加する内部座標へ変換**する（下りB線でキロ程が減少する路線に対応）。
     勾配の符号はそのまま保つ（資料が進行方向基準のため。§6.3）。
  2. **勾配フィルタを緩和**（既存は -40 < g <= 30 で範囲外を無言で 0.0 にする。
     東西線の +29.7‰ は上限0.3‰手前、延伸区間の ±35‰ は握り潰される。§6.2(2)）。
     範囲外は**例外を出す**ことで、黙って平坦扱いになるのを防ぐ。
  3. **曲線データの欠損を許容**（東西線には curve.csv が無い。曲線抵抗0で扱う。§6.2.2）。
  4. 速度制限は**CS-ATCの信号現示**であり、**先頭位置**で判定する（列車長は考慮しない。§6.2.1）。

既存 track.py / train.py / environment2.py は一切変更しない。
"""
import os
import codecs
from bisect import bisect_right

import pandas as pd

import line_config as LC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_csv(path):
    with codecs.open(path, "r", "utf-8", "ignore") as f:
        return pd.read_csv(f)


class TrackMulti:
    """路線データ（駅・勾配・ATC現示・曲線）を内部座標で保持する。

    内部座標 x [km] は**起点駅を0とし進行方向に増加**する。
    元のキロ程との対応は `to_internal()` / `to_kilometrage()` で変換できる。
    """

    def __init__(self, line_name="tozai"):
        self.line_name = line_name
        self.cfg = LC.get_line(line_name)
        self.vehicle = LC.get_vehicle(self.cfg["vehicle"])
        d = os.path.join(BASE_DIR, self.cfg["dir"])
        self.descending = (self.cfg["direction"] == "descending")

        # ---- 駅 ----
        st = _read_csv(os.path.join(d, self.cfg["station_csv"]))
        raw = [{"name": str(st["name"][i]),
                "km": float(st["position"][i]),
                "stay_time": float(st["stay_time"][i]),
                "running_time": float(st["rt"][i])} for i in range(len(st))]
        # 起点＝進行方向の最初の駅（CSVの並び順が進行方向であることを前提とする）
        self.origin_km = raw[0]["km"]
        self.stations = []
        for s in raw:
            self.stations.append({"name": s["name"],
                                  "position": self.to_internal(s["km"]),
                                  "km": s["km"],
                                  "stay_time": s["stay_time"],
                                  "running_time": s["running_time"]})
        # 内部座標が単調増加していることを確認する（CSVの並びが進行方向でない場合を検出）
        for a, b in zip(self.stations, self.stations[1:]):
            if b["position"] <= a["position"]:
                raise ValueError(
                    f"[{line_name}] 駅の内部座標が単調増加していません: "
                    f"{a['name']}({a['position']:.4f}) → {b['name']}({b['position']:.4f})。"
                    f"Station.csv の並びが進行方向になっているか、direction 設定を確認してください。")

        # ---- 勾配 ----
        gl = float(self.cfg["grade_limit"])
        g = _read_csv(os.path.join(d, self.cfg["grade_csv"]))
        segs = []
        for i in range(len(g)):
            s = self.to_internal(float(g["start"][i]))
            e = self.to_internal(float(g["end"][i]))
            val = float(g["grade"][i])
            if abs(val) > gl:
                raise ValueError(
                    f"[{line_name}] 勾配 {val:+.1f}‰ が上限 ±{gl:.0f}‰ を超えています "
                    f"(キロ程 {g['start'][i]}〜{g['end'][i]})。"
                    f"既存 track.py はこれを無言で 0.0 に置き換えるため、"
                    f"line_config の grade_limit を見直してください。")
            lo, hi = (min(s, e), max(s, e))
            segs.append((lo, hi, val))
        segs.sort()
        self.grade_starts = [a for a, _, _ in segs]
        self.grade_ends = [b for _, b, _ in segs]
        self.grade_vals = [v for _, _, v in segs]

        # ---- ATC現示（速度制限） ----
        lm = _read_csv(os.path.join(d, self.cfg["limit_csv"]))
        secs = sorted((self.to_internal(float(lm["start"][i])), float(lm["speed_limit"][i]))
                      for i in range(len(lm)))
        self.limit_starts = [a for a, _ in secs]
        self.limit_vals = [v for _, v in secs]

        # ---- 曲線（無い路線は曲線抵抗0） ----
        self.curve_starts, self.curve_vals = [], []
        cc = self.cfg.get("curve_csv")
        if cc:
            path = os.path.join(d, cc)
            if os.path.exists(path):
                c = _read_csv(path)
                items = []
                for i in range(len(c)):
                    s = self.to_internal(float(c["start"][i]))
                    e = self.to_internal(float(c["end"][i]))
                    r = float(c["curve"][i])
                    lo, hi = (min(s, e), max(s, e))
                    items.append((lo, hi, 800.0 / r if r else 0.0))
                items.sort()
                # 区間の切れ目には抵抗0を挿入する（既存 track.py と同じ扱い）
                for j, (lo, hi, v) in enumerate(items):
                    if j > 0 and round(items[j - 1][1], 4) != round(lo, 4):
                        self.curve_starts.append(items[j - 1][1]); self.curve_vals.append(0.0)
                    self.curve_starts.append(lo); self.curve_vals.append(v)
                    self.curve_ends = None

    # ------------------------------------------------------------------ 座標
    def to_internal(self, km):
        """元のキロ程 → 内部座標[km]（進行方向に増加・起点駅が0）"""
        return (self.origin_km - km) if self.descending else (km - self.origin_km)

    def to_kilometrage(self, x):
        """内部座標[km] → 元のキロ程"""
        return (self.origin_km - x) if self.descending else (self.origin_km + x)

    # ------------------------------------------------------------------ 参照
    def grade(self, x):
        """位置 x [km] の勾配抵抗[kg/t]（＝勾配[‰]。進行方向基準・符号そのまま）"""
        i = bisect_right(self.grade_starts, x) - 1
        if i < 0 or x > self.grade_ends[i]:
            return 0.0
        return self.grade_vals[i]

    def curve(self, x):
        """位置 x [km] の曲線抵抗[kg/t]（データが無い路線は常に0）"""
        if not self.curve_starts:
            return 0.0
        i = bisect_right(self.curve_starts, x) - 1
        return self.curve_vals[i] if i >= 0 else 0.0

    def atc_limit(self, x):
        """位置 x [km] のATC現示[km/h]（**先頭位置で判定**。列車長は考慮しない）"""
        i = bisect_right(self.limit_starts, x) - 1
        if i < 0:
            i = 0
        return self.limit_vals[i]

    def min_limit(self, start, end):
        """区間 [start, end] の最小ATC現示[km/h]"""
        vals = [self.atc_limit(start)]
        for s, v in zip(self.limit_starts, self.limit_vals):
            if start < s <= end:
                vals.append(v)
        return min(vals)

    def limit_sections(self, start, end):
        """[start, end] を現示ごとに区切ったリスト（作図・meta用）"""
        out, pos = [], start
        while pos < end - 1e-12:
            lim = self.atc_limit(pos)
            nxt = end
            for s in self.limit_starts:
                if pos < s < end:
                    nxt = min(nxt, s)
                    break
            out.append({"start": pos, "distance": nxt - pos, "speed_limit": lim})
            pos = nxt
        return out

    def next_limit_drop(self, x, ahead_km=1.0):
        """x から前方 ahead_km 以内で**現示が下がる**最初の地点。
        戻り値 (距離[km], 低下後の現示[km/h])。無ければ (None, None)。"""
        cur = self.atc_limit(x)
        for s, v in zip(self.limit_starts, self.limit_vals):
            if s <= x:
                continue
            if s - x > ahead_km:
                break
            if v < cur - 1e-9:
                return s - x, v
            cur = v
        return None, None

    def grade_sections(self, start, end):
        """[start, end] を勾配ごとに区切ったリスト（作図用）"""
        out = []
        for lo, hi, v in zip(self.grade_starts, self.grade_ends, self.grade_vals):
            a, b = max(lo, start), min(hi, end)
            if b > a:
                out.append({"start": a, "end": b, "grade": v})
        return out


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "tozai"
    t = TrackMulti(name)
    print(f"=== {t.cfg['name']} / {t.vehicle['name']} ===")
    print(f"進行方向: {'キロ程が減少' if t.descending else 'キロ程が増加'} / 起点 {t.origin_km} km")
    print("\n駅（内部座標）")
    for i, s in enumerate(t.stations):
        print(f"  [{i}] {s['name']:<16} km={s['km']:8.4f} → x={s['position']:7.4f}  rt={s['running_time']:.0f}s")
    print("\nATC現示")
    for a, v in zip(t.limit_starts, t.limit_vals):
        print(f"  x={a:7.4f} km （km程 {t.to_kilometrage(a):8.4f}） → {v:5.1f} km/h")
    print("\n各駅間の諸元")
    for i in range(len(t.stations) - 1):
        a, b = t.stations[i], t.stations[i + 1]
        secs = t.grade_sections(a["position"], b["position"])
        tot = sum(s["grade"] * (s["end"] - s["start"]) for s in secs)
        L = b["position"] - a["position"]
        print(f"  {a['name']} → {b['name']}: {L*1000:7.1f} m / rt {a['running_time']:.0f}s "
              f"/ 平均勾配 {tot/L:+6.2f}‰ / 最小現示 {t.min_limit(a['position'], b['position']):.0f} km/h")
        d, v = t.next_limit_drop(a["position"], ahead_km=L)
        if d is not None:
            print(f"      現示低下: 発車から {d*1000:.1f} m 地点で {v:.0f} km/h へ")
