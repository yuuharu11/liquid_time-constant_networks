#!/bin/bash

# 実験設定: RNNモデル (例: Vanilla RNN) を使用
EXPERIMENT_CONFIG="experiment=rnn/uci_har"

# 結果を保存するルートディレクトリ
RESULTS_DIR="outputs/rnn/uci_har_loso"

# 被験者数 (UCI-HAR は 1～30)
NUM_SUBJECTS=30

# --- LOSO ループ ---
for i in $(seq 15 $NUM_SUBJECTS)
do
  echo "====================================================="
  echo "== Running LOSO for Subject ID: $i"
  echo "====================================================="

  # Hydra のコマンドライン引数で loso_subject を上書き
  # 出力ディレクトリも被験者ごとに変更
  python3 train.py \
    $EXPERIMENT_CONFIG \
    dataset.loso_subject=$i \
    hydra.run.dir="$RESULTS_DIR/subject_$i"\
    wandb.project="UCI-HAR-LOSO" \
    wandb.name="RNN_Subject_$i"\
    
  # エラーが発生したらスクリプトを停止
  if [ $? -ne 0 ]; then
    echo "Error running experiment for subject $i. Aborting."
    exit 1
  fi
done

echo "====================================================="
echo "== LOSO cross-validation finished."
echo "== Results saved in: $RESULTS_DIR"
echo "====================================================="
