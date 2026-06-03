import pandas as pd
import glob
import os

def update_csv_files(target_folder="."):
    """
    指定されたフォルダ内のCSVファイルを読み込み、
    forward_info と backward_info の列を一括追加して上書き保存します。
    """
    # 対象フォルダ内のすべてのCSVファイルを取得
    csv_files = glob.glob(os.path.join(target_folder, "*.csv"))
    
    if not csv_files:
        print("指定されたフォルダにCSVファイルが見つかりません。")
        return

    updated_count = 0
    skipped_count = 0

    print("CSVファイルの更新を開始します...")

    for file_path in csv_files:
        try:
            # CSVの読み込み（文字化け対策としてutf-8想定。エラー時はcp932などを試すよう自動フォールバック）
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp932')

            # 既に新しい列（forward_info）が存在するかチェック
            if 'forward_info' not in df.columns:
                # 'next_gradient_info' 列の次の位置（インデックス）を取得
                if 'next_gradient_info' in df.columns:
                    insert_idx = df.columns.get_loc('next_gradient_info') + 1
                    
                    # 指定位置に新しい列を挿入し、すべての行に固定値をセット
                    df.insert(insert_idx, 'forward_info', '先行列車なし')
                    df.insert(insert_idx + 1, 'backward_info', '後続列車なし')
                    
                    # 上書き保存（Excel等で文字化けしないように utf-8-sig を使用）
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"[更新完了] {os.path.basename(file_path)}")
                    updated_count += 1
                else:
                    print(f"[スキップ] {os.path.basename(file_path)}: 'next_gradient_info' 列が見つかりません。")
                    skipped_count += 1
            else:
                print(f"[スキップ] {os.path.basename(file_path)}: 既に新しい列が存在します。")
                skipped_count += 1
                
        except Exception as e:
            print(f"[エラー] {os.path.basename(file_path)} の処理中にエラーが発生しました: {e}")

    print("-" * 30)
    print(f"処理完了: 更新 {updated_count} 件 / スキップ {skipped_count} 件")

if __name__ == "__main__":
    # 変更したいCSVファイルが入っているフォルダのパスを指定してください。
    # 現在のフォルダ（スクリプトと同じ場所）にあるCSVを処理する場合は "." を指定します。
    # 例: target_dir = "./data/history"
    target_dir = "./train_reward_csv_direct" 
    
    update_csv_files(target_dir)