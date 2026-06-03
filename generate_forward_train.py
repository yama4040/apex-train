import os
import csv
import math

# train.py や actions.py が同じディレクトリにある前提でインポート
from train import Train
from actions import Actions

# ==== 設定値 ====
# 自列車（羽前成田: 21.112km）の約360m前方に先行列車を配置
START_POS = 22.112
TARGET_STATION = 30.605 
# 先行列車が停車する駅の位置（白兎: 23.29km）
STOP_STATION_POS = 23.29
# 最大シミュレーション秒数 (エピソードが途切れないように長めに設定)
TOTAL_STEPS = 1200

def generate_forward_train_csv(filename, target_speed, delay_sec, stop_pos=None, stop_time_sec=0):
    # 先行列車のインスタンスを生成（実際の物理モデルで演算する）
    train = Train(TARGET_STATION, position=START_POS, speed=0.0)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # environment2.py が読み込むためのヘッダ
        writer.writerow(['time', 'position', 'speed', 'action'])
        
        has_stopped = False
        stop_timer = 0
        
        for t in range(TOTAL_STEPS):
            action = Actions.coasting
            
            # 1. 出発前の遅延（待機）フェーズ
            if t < delay_sec:
                action = Actions.deceleration
            
            # 2. 駅への停車および待機フェーズ
            elif stop_pos is not None and not has_stopped:
                dist_to_stop = stop_pos - train.position
                
                # 簡易ブレーキカーブ計算 (現在の速度から減速度2.0km/h/sで止まれる距離)
                v_ms = train.speed / 3.6
                a_ms2 = 2.0 / 3.6
                brake_dist_km = ((v_ms ** 2) / (2 * a_ms2)) / 1000.0
                
                if dist_to_stop <= 0.005 and train.speed <= 1.0:
                    # 停止位置に到達し、ほぼ停止している場合
                    action = Actions.deceleration
                    stop_timer += 1
                    if stop_timer >= stop_time_sec:
                        has_stopped = True # 所定時間停車したので再発車フラグON
                elif dist_to_stop <= brake_dist_km + 0.05: 
                    # ブレーキカーブに入った (+50mの余裕)
                    action = Actions.deceleration
                else:
                    # まだ距離がある場合は目標速度まで力行/惰行
                    if train.speed < target_speed - 2.0:
                        action = Actions.acceleration
                    elif train.speed > target_speed + 2.0:
                        action = Actions.deceleration
                    else:
                        action = Actions.coasting
                        
            # 3. 通常走行（巡航）フェーズ
            else:
                if train.speed < target_speed - 2.0:
                    action = Actions.acceleration
                elif train.speed > target_speed + 2.0:
                    action = Actions.deceleration
                else:
                    action = Actions.coasting
            
            # --- CSVへ書き込むためのアクション文字列の変換 ---
            if action == Actions.acceleration:
                action_str = "Actions.acceleration"
            elif action == Actions.deceleration:
                action_str = "Actions.deceleration"
            else:
                action_str = "Actions.coasting"
                
            # 現在の時刻、位置、速度、行動を記録
            writer.writerow([t, round(train.position, 6), round(train.speed, 2), action_str])
            
            # train.py の物理モデルを1ステップ(1.0秒)進める
            train.step(action, 1.0)
            
    print(f"Generated: {filename}")


if __name__ == "__main__":
    print("=== 先行列車用CSVの生成を開始します ===")
    
    # ① Sim3_Low50 (遅延なし、50km/hで定速走行、駅停車なし)
    generate_forward_train_csv("input/f_train_low50.csv", target_speed=50.0, delay_sec=0)
    
    # ② Sim3_Delay_Stop (先行列車遅延 [0, 5, 10] × 停車時間 [30, 45, 60])
    for f_delay in [0, 5, 10]:
        for stop_time in [30, 45, 60]:
            filename = f"input/f_train_delay{f_delay}_stop{stop_time}.csv"
            generate_forward_train_csv(
                filename, 
                target_speed=65.0,     # 通常区間は65km/hで走行
                delay_sec=f_delay, 
                stop_pos=STOP_STATION_POS, 
                stop_time_sec=stop_time
            )
            
    print("=== すべての生成が完了しました！ ===")