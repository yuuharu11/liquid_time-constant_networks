#!/bin/bash
set -e

# --- 実験設定: RNN ---
MODEL_NAME="RNN"
EXPERIMENT_CONFIG="rnn/uci_har"
RESULTS_DIR="/work/outputs/rnn/uci_har/base-cl"
CSV_LOG_PATH="/work/csv/uci-har/base-cl/rnn.csv"
SEED=42

# --- DILタスクのシーケンス定義 ---
TASKS=(
  "6,0.0"   # Task 0: クリーンデータ
  "6,0.1"   # Task 1: 軽微ノイズ
  "6,0.2"   # Task 2: 小ノイズ
  "6,0.3"   # Task 3: 中ノイズ
  "6,0.4"   # Task 4: やや大きめノイズ
  "6,0.5"   # Task 5: 大ノイズ
  "6,0.6"   # Task 6: かなり大きなノイズ
  "6,0.7"   # Task 7: 深刻ノイズ
  "6,0.8"   # Task 8: 深刻ノイズ
  "6,0.9"   # Task 9: 全壊に近いノイズ
  "6,1.0"   # Task 10: 全壊に近いノイズ
)

NUM_TASKS=${#TASKS[@]}

# --- 学習済みモデルパス ---
PRETRAINED_MODEL="/work/outputs/rnn/uci_har_standard/seed_42/checkpoints/last-v4.ckpt"

echo "=================================================================="
echo "--- Starting evaluation on all DIL tasks using pretrained model ---"
echo "=================================================================="

# --- ループでTask 0から最後まで評価 ---
for i in $(seq 0 $(($NUM_TASKS - 1))); do
  IFS=',' read -r task_id noise_level <<< "${TASKS[$i]}"
  TASK_NAME="Task_${i}_ID_${task_id}_Noise_${noise_level}"

  echo ""
  echo "-----------------------------------------------------"
  echo "--- Evaluating on ${TASK_NAME} ---"
  echo "-----------------------------------------------------"

  python3 train.py \
    experiment=$EXPERIMENT_CONFIG \
    train.seed=$SEED \
    dataset.seed=$SEED \
    dataset.task_id=$task_id \
    dataset.noise_level=$noise_level \
    train.pretrained_model_path=$PRETRAINED_MODEL \
    train.test_only=true \
    callbacks.experiment_logger.output_file=$CSV_LOG_PATH
done

echo ""
echo "=================================================================="
echo "--- ✅ All DIL task evaluations finished ---"
echo "Results logged in: $CSV_LOG_PATH"
echo "=================================================================="
