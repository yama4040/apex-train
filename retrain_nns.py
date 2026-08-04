# 評価NN(train_reward_network2)とモード分類NN(train_mode_network)をセットで再学習するスクリプト。
#
# 学習データ(train_reward_csv_direct)の状態特徴量やmodeラベル分布を変えたら、必ず両方を再学習する
# 必要がある。片方だけの再学習は以下の失敗を招く（設計メモ§16-17・過去2回発生）:
#   1. 特徴量の次元を変えたのにモードNNだけ旧次元 → 実行時に mode_scaler.transform で分類例外
#   2. 新しいmodeクラス(例: delay_recovery)のデータを追加したのにモードNNだけ再学習漏れ
#      → 次元は同じで例外は出ないが、モードNNが新クラスを一切出力できず、DQNがそのモードを永遠に
#        獲得できない（無言の失敗）
#
# 本スクリプトは両者を順に再学習し、最後に整合チェックを行う:
#   - mode / reward の各 manifest の特徴量数が reward_features.STATE_FEATURE_COLS と一致するか
#   - 学習データに存在する全modeクラス（ACTIVE）を、再学習後のモードNNが実際に予測できるか（無言失敗の検出）
#
# ※実行後は必ず DQN(apex2.py) を再起動すること（apex2は起動時に mode_model / direct_reward_model を
#   ロードしてキャッシュするため、稼働中のrunには反映されない）。
#
# 使い方:
#   python retrain_nns.py                       # 評価NN→モードNNの順に再学習＋整合チェック
#   python retrain_nns.py --skip-reward          # モードNNのみ再学習（評価NNは変更なしのとき）
#   python retrain_nns.py --verify-only          # 再学習せず整合チェックのみ
#   python retrain_nns.py --csv-dir <dir>        # 学習データフォルダを指定

import argparse
import collections
import csv
import glob
import json
import os


def retrain(csv_dir, reward_epochs, mode_epochs, skip_reward, skip_mode):
    if not skip_reward:
        print("=" * 64)
        print("[1/2] 評価NN(direct_reward)再学習  train_reward_network2.main()")
        print("=" * 64)
        import train_reward_network2 as trn
        trn.main(csv_dir=csv_dir, epochs=reward_epochs)

    if not skip_mode:
        print("=" * 64)
        print("[2/2] モード分類NN(mode)再学習  train_mode_network.main()")
        print("=" * 64)
        import train_mode_network as tmn
        tmn.main(csv_dir=csv_dir, epochs=mode_epochs)


def _manifest_feat_len(path):
    if not os.path.exists(path):
        return None
    m = json.load(open(path, encoding="utf-8"))
    cols = m.get("state_feature_cols") or m.get("feature_cols") or []
    return len(cols)


def verify(csv_dir):
    """再学習後の整合チェック。片方漏れ・無言のモード獲得失敗を検出する。"""
    import numpy as np  # noqa: F401  (reward_features/predictor が依存)
    import reward_features as rf

    print("=" * 64)
    print("整合チェック")
    print("=" * 64)
    ok = True
    n_feat = len(rf.STATE_FEATURE_COLS)

    # 1) 特徴量数の一致（次元漏れの検出）
    for name, path in [("モードNN", "mode_manifest.json"), ("評価NN", "direct_reward_manifest.json")]:
        ln = _manifest_feat_len(path)
        if ln is None:
            print(f"  {name}: manifest '{path}' が見つかりません ★"); ok = False
        elif ln != n_feat:
            print(f"  {name}: 特徴量数 {ln} ≠ reward_features {n_feat} ★不一致（再学習漏れ）"); ok = False
        else:
            print(f"  {name}: 特徴量数 {ln} = reward_features {n_feat}  OK")

    # 2) 学習データに存在する全modeクラスをモードNNが予測できるか（無言失敗の検出）
    data_modes = collections.Counter()
    rows_by_mode = collections.defaultdict(list)
    for f in glob.glob(os.path.join(csv_dir, "*.csv")):
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                mo = (r.get("mode") or "").strip()
                if mo:
                    data_modes[mo] += 1
                    if len(rows_by_mode[mo]) < 150:
                        rows_by_mode[mo].append(r)
    print(f"  学習データのmode分布: {dict(data_modes)}")

    from direct_reward_predictor2 import DirectRewardPredictor
    rp = DirectRewardPredictor()
    if rp.mode_model is None:
        print("  ★モードNNがロードできません（mode_model=None）"); ok = False
    else:
        for mo, rows in rows_by_mode.items():
            if mo not in rf.MODE_CLASSES_ACTIVE:
                continue  # spacing等ACTIVE外は学習対象外
            preds = collections.Counter()
            for r in rows:
                _, ms = rp._infer_mode(rf.state_vector(r).reshape(1, -1))
                preds[ms] += 1
            hit = preds.get(mo, 0)
            if hit == 0:
                print(f"  mode='{mo}': データ{data_modes[mo]}行 → NNが{mo}を1件も予測せず ★無言失敗"); ok = False
            else:
                print(f"  mode='{mo}': データ{data_modes[mo]}行 → NN予測 {mo} {hit}/{len(rows)}件  OK")

    print("=" * 64)
    if ok:
        print("✅ 整合OK。DQN(apex2.py)を再起動してください（起動時にモデルをキャッシュするため、"
              "稼働中のrunには反映されません）。")
    else:
        print("❌ 問題を検出しました。上記★の項目を確認してください（多くは片方のNNの再学習漏れ）。")
    return ok


def main():
    ap = argparse.ArgumentParser(description="評価NNとモード分類NNをセットで再学習する")
    ap.add_argument("--csv-dir", default="train_reward_csv_direct", help="学習データフォルダ")
    ap.add_argument("--reward-epochs", type=int, default=500)
    ap.add_argument("--mode-epochs", type=int, default=300)
    ap.add_argument("--skip-reward", action="store_true", help="評価NNの再学習をスキップ")
    ap.add_argument("--skip-mode", action="store_true", help="モードNNの再学習をスキップ")
    ap.add_argument("--verify-only", action="store_true", help="再学習せず整合チェックのみ")
    args = ap.parse_args()

    if not args.verify_only:
        retrain(args.csv_dir, args.reward_epochs, args.mode_epochs, args.skip_reward, args.skip_mode)
    verify(args.csv_dir)


if __name__ == "__main__":
    main()
