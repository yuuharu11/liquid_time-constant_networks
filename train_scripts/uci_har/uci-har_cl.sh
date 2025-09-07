#!/bin/bash
set -e

# --- モデル種別のループ ---
MODELS=(rnn cnn lstm ncps)
EXPERIMENT_BASE="uci_har/continual_learning"
WANDB_PROJECT="UCI-HAR-Continual-Learning"
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

for MODEL in "${MODELS[@]}"; do
  echo "=================================================================="
  echo "=== Running CL experiments for model: $MODEL ==="
  echo "=================================================================="

  RESULTS_DIR="/work/outputs/${MODEL}/uci_har/continual_learning"
  CSV_LOG_PATH="/work/csv/uci-har/${MODEL}.csv"
  GROUP_NAME="${MODEL^^}"

  # -------------------------------
  # Task 0 ベースライン訓練
  # -------------------------------
  TASK_0_OUTPUT_DIR="${RESULTS_DIR}/Task_0/train"
  echo "--- Training baseline model on Task 0 ---"
  python3 train.py \
    experiment=experiment=${MODEL}/${EXPERIMENT_BASE} \
    train.seed=$SEED \
    dataset.seed=$SEED \
    dataset.task_id=0 \
    dataset.noise_level=0.0 \
    train.ckpt_path=null \
    trainer.max_epochs=30 \
    hydra.run.dir=$TASK_0_OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.group=$GROUP_NAME \
    wandb.name="Task_0_Train" \
    callbacks.experiment_logger.output_file=$CSV_LOG_PATH

  LAST_CHECKPOINT_PATH="${TASK_0_OUTPUT_DIR}/checkpoints/last.ckpt"
  echo "✅ Task 0 training complete. Checkpoint: $LAST_CHECKPOINT_PATH"

  # -------------------------------
  # Task 1 以降のCL訓練
  # -------------------------------
  for i in $(seq 1 $(($NUM_TASKS - 1))); do
    IFS=',' read -r task_id noise_level <<< "${TASKS[$i]}"
    TASK_NAME="Task_${i}_ID_${task_id}_Noise_${noise_level}"
    OUTPUT_DIR="${RESULTS_DIR}/${TASK_NAME}/train"

    echo "--- Training on ${TASK_NAME} ---"
    python3 train.py \
      experiment=experiment=${MODEL}/${EXPERIMENT_BASE} \
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
      callbacks.experiment_logger.output_file=$CSV_LOG_PATH

    LAST_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"

    # 過去タスクの評価
    echo "--- Evaluating on all previous tasks (0 to ${i}) ---"
    for j in $(seq 0 $i); do
      IFS=',' read -r past_task_id past_noise_level <<< "${TASKS[$j]}"
      PAST_TASK_NAME="Task_${j}_ID_${past_task_id}_Noise_${past_noise_level}"
      EVAL_DIR="${RESULTS_DIR}/${TASK_NAME}/eval_on_${PAST_TASK_NAME}"

      echo "--- Testing on ${PAST_TASK_NAME} ---"
      python3 train.py \
        experiment=experiment=${MODEL}/${EXPERIMENT_BASE} \
        train.seed=$SEED \
        dataset.seed=$SEED \
        train.pretrained_ckpt_path=$LAST_CHECKPOINT_PATH \
        train.test_only=true \
        hydra.run.dir=$EVAL_DIR \
        callbacks.experiment_logger.output_file=$CSV_LOG_PATH
    done
  done
done

echo "=================================================================="
echo "--- ✅ All CL experiments finished ---"
echo "=================================================================="
