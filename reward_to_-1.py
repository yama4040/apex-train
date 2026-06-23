import pandas as pd
import glob
import os

def process_and_combine_csv():
    # フォルダのパスと出力ファイル名を設定
    input_dir = 'train_reward_csv_direct'
    output_file = 'combined_reward_modified.csv'

    # 指定フォルダ内のすべてのCSVファイルのパスを取得
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    if not csv_files:
        print(f"エラー: ディレクトリ '{input_dir}' にCSVファイルが見つかりません。")
        return

    print(f"{len(csv_files)} 個のCSVファイルを結合・処理します...")

    df_list = []
    
    # 各CSVファイルを読み込んでリストに追加
    for file in csv_files:
        try:
            # 日本語が含まれるため、一般的なエンコーディングで読み込みを試みる
            # utf-8でエラーが出る場合は 'shift_jis' や 'cp932' に変更してください
            df = pd.read_csv(file, encoding='utf-8')
            df_list.append(df)
        except Exception as e:
            print(f"ファイル '{file}' の読み込み中にエラーが発生しました: {e}")

    if df_list:
        # 全てのデータフレームを縦に結合（インデックスは振り直す）
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # 置換前の 0.0 の件数を確認（ログ用）
        zero_count = (combined_df['reward'] == 0.0).sum()
        
        # reward列が 0.0 の行を -1.0 に置換
        combined_df.loc[combined_df['reward'] == 0.0, 'reward'] = -1.0
        
        # 1つのCSVファイルとして出力
        # utf-8-sig を指定することで、Excelで開いた際の日本語の文字化けを防ぎます
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"処理が完了しました！")
        print(f" -> 置換した対象 (reward = 0.0): {zero_count} 件")
        print(f" -> 出力ファイル: '{output_file}' (全 {len(combined_df)} 行)")

if __name__ == "__main__":
    process_and_combine_csv()