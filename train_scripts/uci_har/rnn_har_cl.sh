#!/bin/bash
set -e

# --- 実験設定: RNN ---
MODEL_NAME="RNN"
EXPERIMENT_CONFIG="experiment=rnn/uci_har"
RESULTS_DIR="/work/outputs/rnn/uci_har/continual_learning"
WANDB_PROJECT="UCI-HAR-Continual-Learning"
CSV_LOG_PATH="/work/csv/uci-har/rnn.csv"
GROUP_NAME="RNN"
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
# --- Task 0 (ベースライン) の訓練 ---
# ==================================================================
TASK_0_NAME="Task_0"
TASK_0_OUTPUT_DIR="${RESULTS_DIR}/${TASK_0_NAME}/train"
echo "=================================================================="
echo "--- 1. Training baseline model on Task 0 (Clean Data) ---"
echo "=================================================================="

python3 train.py \
  experiment=$EXPERIMENT_CONFIG \
  train.seed=$SEED \
  dataset.seed=$SEED \
  dataset.task_id=0 \
  dataset.noise_level=0.0 \
  train.ckpt_path=null \
  trainer.max_epochs=30 \
  hydra.run.dir=$TASK_0_OUTPUT_DIR \
  wandb.project=$WANDB_PROJECT \
  wandb.group=$GROUP_NAME \
  wandb.name="${TASK_0_NAME}_Train" \
  callbacks.experiment_logger.output_file=$CSV_LOG_PATH \

LAST_CHECKPOINT_PATH="${TASK_0_OUTPUT_DIR}/checkpoints/last.ckpt"
echo "✅ Baseline training complete. Checkpoint: ${LAST_CHECKPOINT_PATH}"


# ==================================================================
# --- ループでTask 1から最後まで、それぞれモデルを訓練・評価 ---
# ==================================================================
for i in $(seq 1 $(($NUM_TASKS - 1))); do
  
  # --- a. Task i での訓練 ---
  IFS=',' read -r task_id noise_level <<< "${TASKS[$i]}"
  TASK_NAME="Task_${i}_ID_${task_id}_Noise_${noise_level}"
  OUTPUT_DIR="${RESULTS_DIR}/${TASK_NAME}/train"

  echo ""
  echo "=================================================================="
  echo "--- Training model on ${TASK_NAME} ---"
  echo "=================================================================="
  
  python3 train.py \
    experiment=$EXPERIMENT_CONFIG \
    train.seed=$SEED \
    dataset.seed=$SEED \
    dataset.task_id=$task_id \
    dataset.noise_level=$noise_level \
    train.ckpt_path=$LAST_CHECKPOINT_PATH \
    trainer.max_epochs=10 \
    hydra.run.dir=$OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.group=$GROUP_NAME \
    wandb.name="${TASK_NAME}_Train" \
    callbacks.experiment_logger.output_file=$CSV_LOG_PATH \

  CHECKPOINT_PATH_JUST_TRAINED="${OUTPUT_DIR}/checkpoints/last.ckpt"
  LAST_CHECKPOINT_PATH=$CHECKPOINT_PATH_JUST_TRAINED 
  echo "✅ Training for ${TASK_NAME} complete. Checkpoint: ${CHECKPOINT_PATH_JUST_TRAINED}"

  # --- b. 過去の全タスクでの評価 ---
  echo "--- Evaluating model on ALL previous tasks (0 to ${i}) ---"
  for j in $(seq 0 $i); do
    IFS=',' read -r past_task_id past_noise_level <<< "${TASKS[$j]}"
    PAST_TASK_NAME="Task_${j}_ID_${past_task_id}_Noise_${past_noise_level}"
    EVAL_DIR="${RESULTS_DIR}/${TASK_NAME}/eval_on_${PAST_TASK_NAME}"
    
    echo "---   Testing on ${PAST_TASK_NAME} ---"

    python3 train.py \
      experiment=$EXPERIMENT_CONFIG \
      train.seed=$SEED \
      dataset.seed=$SEED \
      train.pretrained_ckpt_path=$CHECKPOINT_PATH_JUST_TRAINED \
      train.test_only=true \ \
      callbacks.experiment_logger.output_file=$CSV_LOG_PATH \

  done
done

echo ""
echo "=================================================================="
echo "--- ✅ All Continual Learning experiments finished! ---"
echo "=================================================================="