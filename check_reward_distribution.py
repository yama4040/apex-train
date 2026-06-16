import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(csv_dir):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ディレクトリ '{csv_dir}' にCSVファイルが見つかりません。")
    
    print(f"{len(csv_files)}個のCSVファイルを読み込みます...")
    
    # 最新の34次元(物理的停止距離を含む)のヘッダ構成
    columns = ["time", "train_id", "phase", "current_notch", "holding_time", 
               "prev_notch", "prev_notch_duration", 
               "speed_limit", "current_speed", 
               "dist_to_next_station", "time_to_next_station", "req_stop_dist", "delay", "current_gradient", 
               "next_limit_info", "next_gradient_info", "forward_info", "backward_info", "reward", "reason"]
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file, names=columns, skiprows=1)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"合計データ数: {len(df)}行")
    
    return df

def analyze_reward_distribution(df):
    # 報酬列を浮動小数点数として取得し、欠損値があれば除外
    rewards = pd.to_numeric(df['reward'], errors='coerce').dropna()
    
    # 0.1刻みのビンを作成
    # 丸め誤差を考慮して -1.05 から 1.05 の範囲でビンを区切り、中心を -1.0, -0.9...1.0 に合わせる
    bins = np.arange(-1.05, 1.15, 0.1)
    
    # ヒストグラムの計算 (各区間に入っているデータ数をカウント)
    counts, bin_edges = np.histogram(rewards, bins=bins)
    
    print("\n=== 報酬値の分布 (0.1刻み) ===")
    print(" 評価値 | データ件数")
    print("--------|----------")
    for i in range(len(counts)):
        # ビンの中心値を計算（表示用）
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2.0
        
        # カウントが0の行も表示したい場合はそのまま出力
        if counts[i] > 0 or True:
            # -0.0 になるのを防ぐための微小な補正
            display_val = 0.0 if abs(bin_center) < 0.01 else bin_center
            print(f"  {display_val:4.1f}  | {counts[i]:7d} 件")
    print("==============================\n")
        
    # --- グラフの作成と保存 ---
    plt.figure(figsize=(10, 6))
    
    # ヒストグラムの描画
    plt.hist(rewards, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    
    # グラフの見た目の調整
    plt.title('Distribution of LLM Rewards', fontsize=14)
    plt.xlabel('Reward Value', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    
    # X軸の目盛りを -1.0 から 1.0 まで 0.1 刻みに設定
    xticks = np.arange(-1.0, 1.1, 0.1)
    plt.xticks(xticks)
    
    # Y軸にグリッド線を追加
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 画像として保存
    save_path = 'reward_distribution.png'
    plt.savefig(save_path, dpi=300)
    print(f"分布グラフを '{save_path}' として保存しました。")
    
    # 環境によってはウィンドウが開きます（不要な場合はコメントアウト）
    # plt.show()

def main():
    # 学習用CSVが格納されているディレクトリを指定
    csv_dir = 'train_reward_csv_direct'
    
    if not os.path.exists(csv_dir):
        print(f"[エラー] ディレクトリ '{csv_dir}' が見つかりません。")
        print("データセットが配置されているディレクトリ名を正しく設定してください。")
        return
        
    try:
        df = load_data(csv_dir)
        analyze_reward_distribution(df)
    except Exception as e:
        print(f"[エラー] 処理中に問題が発生しました: {e}")

if __name__ == "__main__":
    main()