import pandas as pd
import numpy as np
import sys
import os

# ==========================================
# 設定項目
# ==========================================
# 出発駅の位置(km)。環境に合わせて変更してください。
# 出発駅の位置(km) : 羽前成田
START_POSITION = 21.112

# 制限速度データ(speed_limit.csv)のパス
SPEED_LIMIT_CSV = "./input/speed_limit.csv"
# ==========================================

def calc_tractive_force(speed):
    """ 速度(km/h)から引張力[kg/t]を計算する関数 """
    if speed < 42:
        return -1.489 * speed + 92.408
    elif 42 <= speed < 68:
        return -0.4 * speed + 46.68
    else:
        return -0.0963 * speed + 26.0284

def evaluate_csv(csv_path, start_position, speed_limit_path):
    FACTOR_OF_INERTIA = 28.34467

    try:
        # まず通常のヘッダありを想定して読み込む
        df = pd.read_csv(csv_path)
        
        # 'raw_speed' という列名が存在しない場合、ヘッダなしCSVと判断する
        if 'raw_speed' not in df.columns:
            # ヘッダなしとして再読み込み
            df = pd.read_csv(csv_path, header=None)
            
            # 列数が足りない場合はエラー
            if len(df.columns) < 5:
                print("エラー: CSVの列数が不足しています。")
                return
                
            # environment.py の raw_state 定義に従って必要な列に名前を割り当てる
            df = df.rename(columns={
                0: 'raw_speed',
                1: 'raw_stat_dist',
                2: 'raw_rem_time',
                4: 'raw_pre_action'
            })
            
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 制限速度データの読み込み
    sl_df = None
    if os.path.exists(speed_limit_path):
        sl_df = pd.read_csv(speed_limit_path)
        sl_df = sl_df.sort_values(by='start').reset_index(drop=True)
    else:
        print(f"警告: 制限速度ファイル '{speed_limit_path}' が見つかりません。制限速度超過は計算されません。")

    total_energy = 0.0
    notch_changes = 0
    violation_count = 0
    is_violating = False 

    # 初期状態の取得
    prev_action = df.loc[0, 'raw_pre_action']
    initial_dist = df.loc[0, 'raw_stat_dist']

    for i in range(1, len(df)):
        current_speed = df.loc[i, 'raw_speed']
        current_action = df.loc[i, 'raw_pre_action']
        current_rem_dist = df.loc[i, 'raw_stat_dist']
        
        # 現在位置の推定
        current_position = start_position + (initial_dist - current_rem_dist)

        # タイムステップの計算
        dt = round(df.loc[i-1, 'raw_rem_time'] - df.loc[i, 'raw_rem_time'], 3)
        if dt <= 0:
            dt = 0.1 if current_rem_dist < 0.1 else 1.0

        # 1. 消費エネルギー量の計算 (加速行動: 1 の場合のみ)
        if current_action == 1:
            force = calc_tractive_force(current_speed)
            motor_acceleration = force / FACTOR_OF_INERTIA
            step_energy = motor_acceleration * current_speed * dt
            total_energy += step_energy

        # 2. ノッチ切り替え回数
        if current_action != prev_action:
            notch_changes += 1
        
        # 3. 制限速度の超過判定
        if sl_df is not None:
            limits = sl_df[sl_df['start'] <= current_position]
            current_limit = limits.iloc[-1]['speed_limit'] if len(limits) > 0 else 70

            if current_speed > current_limit:
                if not is_violating:
                    violation_count += 1
                    is_violating = True
            else:
                is_violating = False

        prev_action = current_action

    # 4. 遅延時間 
    final_rem_time = df['raw_rem_time'].iloc[-1]
    delay_time = -final_rem_time

    # 5. 停止位置の誤差
    final_rem_dist = df['raw_stat_dist'].iloc[-1]
    stop_error_m = abs(final_rem_dist) * 1000

    # === 結果の出力 ===
    print(f"=== 評価結果: {csv_path} ===")
    print(f"総消費エネルギー量 : {total_energy:.2f}")
    print(f"遅延時間           : {delay_time:.2f} 秒 (正なら遅延、負なら早着)")
    print(f"ノッチ切り替え回数 : {notch_changes} 回")
    print(f"停止位置の誤差     : {stop_error_m:.2f} m")
    if sl_df is not None:
        print(f"制限速度超過回数   : {violation_count} 回")
    print("===============================\n")
    
if __name__ == "__main__":
    # コマンドライン引数でファイル名を指定可能 (例: python eval.py data.csv)
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "comp/7200_0.csv"
    evaluate_csv(target_csv, START_POSITION, SPEED_LIMIT_CSV)