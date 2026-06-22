import os
import csv
import math

# train.py や actions.py が同じディレクトリにある前提でインポート
from train import Train
from actions import Actions

# ==== 設定値 ====
# 自列車（羽前成田: 21.112km）の約360m前方に先行列車を配置
START_POS = 21.112
TARGET_STATION = 30.605 
# 先行列車が停車する駅の位置（白兎: 23.29km）
STOP_STATION_POS = 23.29
# 最大シミュレーション秒数 (エピソードが途切れないように長めに設定)
TOTAL_STEPS = 1200

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
            generate_forward_train_csv(filename, target_speed=50.0, delay_sec=f_delay, stop_pos=STOP_STATION_POS, stop_time_sec=stop_time)