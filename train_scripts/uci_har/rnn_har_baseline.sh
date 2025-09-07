#!/bin/bash
set -e

# --- 実験設定: RNN ---
MODEL_NAME="RNN"
EXPERIMENT_CONFIG="experiment=rnn/uci_har"
RESULTS_DIR="/work/outputs/rnn/uci_har/base-cl"
CSV_LOG_PATH="/work/csv/uci-har/base-cl/rnn.csv"
SEED=42

# --- DILタスクのシーケンス定義 ---
TASKS=(
  "0,0.0"      # Task 0: クリーンなデータ
  "0,0.3"      # Task 1: total_acc 故障 (軽微)
  "1,0.3"      # Task 2: body_acc 故障 (軽微)
  "2,0.3"      # Task 3: body_gyro 故障 (軽微)
  "3,0.6"      # Task 4: 複合故障 (body_acc + gyro)
  "4,0.6"      # Task 5: 複合故障 (body_gyro + total_acc)
  "5,0.6"      # Task 6: 複合故障 (body_acc + total_acc)
  "6,1.0"      # Task 7: 全センサー故障 (深刻)
)
NUM_TASKS=${#TASKS[@]}


# ==================================================================
# --- ループでTask 1から最後まで、それぞれモデルを評価 ---
# ==================================================================
for i in $(seq 1 $(($NUM_TASKS - 1))); do

  # --- 過去の全タスクでの評価 ---

  IFS=',' read -r task_id noise_level <<< "${TASKS[$i]}"

  python3 train.py 
    experiment=$EXPERIMENT_CONFIG \
    train.seed=$SEED \
    dataset.seed=$SEED \
    dataset.task_id=$task_id \
    dataset.noise_level=$noise_level \
    train.pretrained_ckpt_path="/work/outputs/rnn/uci_har_standard/seed_42/checkpoints/last-v4.ckpt" \
    train.test_only=true \
    callbacks.experiment_logger.output_file=$CSV_LOG_PATH \

done

echo ""
echo "=================================================================="
echo "--- ✅ All Continual Learning experiments finished! ---"
echo "=================================================================="