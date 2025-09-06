#!/bin/bash

# 実験設定: 上で作成したCNNの設定ファイルを指定
EXPERIMENT_CONFIG="experiment=cnn/uci-har-loso"

# 結果を保存するルートディレクトリ
RESULTS_DIR="outputs/cnn/uci-har-loso"

# 被験者数 (UCI-HAR は 1～30)
NUM_SUBJECTS=30

# --- LOSO ループ ---
for i in $(seq 1 $NUM_SUBJECTS)
do
  echo "====================================================="
  echo "== Running LOSO for Subject ID: $i"
  echo "====================================================="

  # Hydra のコマンドライン引数で loso_subject などを上書き
  python3 train.py \
    $EXPERIMENT_CONFIG \
    dataset.loso_subject=$i \
    hydra.run.dir="$RESULTS_DIR/subject_$i" \
    logger.wandb.project="UCI-HAR-LOSO" \
    logger.wandb.group="cnn-loso-50epochs" \
    logger.wandb.name="CNN_Subject_$i"
    
  # エラーが発生したらスクリリプトを停止
  if [ $? -ne 0 ]; then
    echo "Error running experiment for subject $i. Aborting."
    exit 1
  fi
done

echo "====================================================="
echo "== LOSO cross-validation finished."
echo "== Results saved in: $RESULTS_DIR"
echo "====================================================="