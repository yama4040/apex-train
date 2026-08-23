# -*- coding: utf-8 -*-
"""
複数駅間最適化版の行動定義（設計: docs_複数駅間最適化_計画.md §4.7）

既存 actions.py（3行動）は**一切変更しない**。こちらは5ノッチ版。

  P1 力行        : 最大加速
  P2 勾配力行    : 引張力 = min(TF(v), R(v) + 35)  → 勾配 g での加速度 = (35 − g)/FI
  C  惰行        : 動力なし
  B2 勾配ブレーキ: 制動ノッチ = (35 − R(v))/FI     → 勾配 g での加速度 = −(35 + g)/FI
  B1 制動        : 常用制動 2.5 km/h/s

P2 は +35‰ で、B2 は −35‰ でちょうど定速になる。
どちらも走行抵抗を完全に打ち消すため、加速度が勾配だけの関数になり、平坦では対称に ±1.235 km/h/s。
"""
from enum import IntEnum


class ActionsMulti(IntEnum):
    power = 0            # P1 力行
    grade_power = 1      # P2 勾配力行
    coasting = 2         # C  惰行
    grade_brake = 3      # B2 勾配ブレーキ
    braking = 4          # B1 制動


# 表示・プロンプト用の対応表（line_config の NOTCH_* と同じ並び）
CODE = {ActionsMulti.power: "P1", ActionsMulti.grade_power: "P2",
        ActionsMulti.coasting: "C", ActionsMulti.grade_brake: "B2",
        ActionsMulti.braking: "B1"}
FROM_CODE = {v: k for k, v in CODE.items()}
N_ACTIONS = len(ActionsMulti)
