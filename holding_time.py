import pandas as pd

def add_holding_time(input_file, output_file):
    # CSVファイルの読み込み
    df = pd.read_csv(input_file)
    
    # 列車IDと時間でソートし、インデックスをリセット
    df = df.sort_values(by=['train_id', 'time']).reset_index(drop=True)
    
    # 状態が変わったタイミングを判定（列車IDが変わった、またはノッチが変わった行）
    state_changed = (df['train_id'] != df['train_id'].shift()) | (df['current_notch'] != df['current_notch'].shift())
    
    # 変化したタイミングでグループIDを振り直す（連続している同じ状態は同じIDになる）
    df['state_group'] = state_changed.cumsum()
    
    # 同じグループ内での連番を取得 = 保持時間（0, 1, 2...秒）
    df['holding_time'] = df.groupby('state_group').cumcount().astype(float)
    
    # 計算用に使った列を削除
    df = df.drop(columns=['state_group'])
    
    # カラム（列）の並び順を指定
    columns_order = [
        'time', 'train_id', 'phase', 'current_notch', 'holding_time', 
        'speed_limit', 'current_speed', 'dist_to_next_station', 
        'time_to_next_station', 'delay', 'current_gradient', 
        'next_limit_info', 'next_gradient_info', 'reward', 'reason'
    ]
    df = df[columns_order]
    
    # 新しいCSVとして保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"変換が完了しました。出力ファイル: {output_file}")

# 実行
add_holding_time("llm_eval_data_20260529_232213.csv", "output.csv")