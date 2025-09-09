#!/bin/bash
set -e

# --- 実験設定 ---
MODEL="rnn"
EXPERIMENT_BASE="uci_har"
WANDB_PROJECT="UCI-HAR-CIL"
SEED=42
NUM_TASKS=5 # CILデータローダーで定義したタスク数

RESULTS_DIR="/work/outputs/${MODEL}/uci_har/cil"
CSV_LOG_PATH="/work/csv/uci-har/cil/${MODEL}.csv"

python3 train.py \
  experiment=${MODEL}/${EXPERIMENT_BASE} \
  dataset=uci_har_cil \
  train.seed=$SEED \
  train.pretrained_model_path=/work/outputs/rnn/uci_har/cil/Task_1/checkpoints/last-v1.ckpt \
  dataset.seed=$SEED \
  dataset.task_id=0 \
  train.test_only=true \
  callbacks.experiment_logger.output_file=$CSV_LOG_PATH