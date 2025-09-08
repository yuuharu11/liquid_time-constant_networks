#!/bin/bash
set -e

# --- モデル種別のループ ---
MODELS=("RNN" "CNN" "LSTM" "LTC_NCPS")
EXPERIMENT_BASE="uci_har"
WANDB_PROJECT="UCI-HAR-DIL"
SEED=42

# --- DILタスクのシーケンス定義 ---
TASKS=(
  "6,0.0"      # Task 0: クリーンなデータ
  "6,0.1"      # Task 1: total_acc 故障 (軽微)
  "6,0.2"      # Task 2: body_acc 故障 (軽微)
  "6,0.3"      # Task 3: body_gyro 故障 (軽微)
  "6,0.4"      # Task 4: 複合故障 (body_acc + gyro)
  "6,0.5"      # Task 5: 複合故障 (body_gyro + total_acc)
  "6,0.6"      # Task 6: 複合故障 (body_acc + total_acc)
  "6,0.7"      # Task 7: 全センサー故障 (深刻)
  "6,0.8"      # Task 7: 全センサー故障 (深刻)
  "6,0.9"      # Task 7: 全センサー故障 (深刻)
  "6,1.0"      # Task 7: 全センサー故障 (深刻)
)
NUM_TASKS=${#TASKS[@]}

for MODEL in "${MODELS[@]}"; do
  echo "=================================================================="
  echo "=== Running CL experiments for model: $MODEL ==="
  echo "=================================================================="

  MODEL=${MODEL,,}  # 小文字化
  RESULTS_DIR="/work/outputs/${MODEL}/uci_har_standard/dil"
  CSV_LOG_PATH="/work/csv/uci-har/dil/${MODEL}.csv"
  GROUP_NAME="${MODEL^^}"

  # -------------------------------
  # Task 0 ベースライン訓練
  # -------------------------------
  TASK_0_OUTPUT_DIR="${RESULTS_DIR}/Task_0/train"
  echo "--- Training baseline model on Task 0 ---"
  python3 train.py \
    experiment=${MODEL}/${EXPERIMENT_BASE} \
    train.seed=$SEED \
    dataset.seed=$SEED \
    dataset.task_id=0 \
    dataset.noise_level=0.0 \
    train.ckpt=null \
    trainer.max_epochs=30 \
    hydra.run.dir=$TASK_0_OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.group=$GROUP_NAME \
    wandb.name="Task_0_Train" \

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
      experiment=${MODEL}/${EXPERIMENT_BASE} \
      train.seed=$SEED \
      dataset.seed=$SEED \
      dataset.task_id=$task_id \
      dataset.noise_level=$noise_level \
      train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
      trainer.max_epochs=10 \
      hydra.run.dir=$OUTPUT_DIR \
      wandb.project=$WANDB_PROJECT \
      wandb.group=$GROUP_NAME \
      wandb.name="${TASK_NAME}_Train" \

    LAST_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"

    # 過去タスクの評価
    echo "--- Evaluating on all previous tasks (0 to ${i}) ---"
    for j in $(seq 0 $i); do
      IFS=',' read -r past_task_id past_noise_level <<< "${TASKS[$j]}"
      PAST_TASK_NAME="Task_${j}_ID_${past_task_id}_Noise_${past_noise_level}"
      EVAL_DIR="${RESULTS_DIR}/${TASK_NAME}/eval_on_${PAST_TASK_NAME}"

      echo "--- Testing on ${PAST_TASK_NAME} ---"
      python3 train.py \
        experiment=${MODEL}/${EXPERIMENT_BASE} \
        train.seed=$SEED \
        dataset.seed=$SEED \
        dataset.task_id=$past_task_id \
        dataset.noise_level=$past_noise_level \
        train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
        train.test_only=true \
        hydra.run.dir=$EVAL_DIR \
        callbacks.experiment_logger.output_file=$CSV_LOG_PATH
    done
  done
done

echo "=================================================================="
echo "--- ✅ All CL experiments finished ---"
echo "=================================================================="
