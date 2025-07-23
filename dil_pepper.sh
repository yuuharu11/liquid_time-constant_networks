#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
WANDB_PROJECT="ncps_dil_pepper"
OUTPUT_CSV_FILE="/work/test/DIL/pepper.csv"

# --- 結果を保存するためのBash配列を初期化 ---
declare -a results_on_own_task # 自身のタスクでの精度
declare -a results_on_task0    # Task0での精度

task_0_OUTPUT_DIR="/work/outputs/ncps/DIL/pepper/task_0"

python3 train.py \
    experiment=$BASE_EXPERIMENT \
    dataset.noise_ratio=0.0 \
    dataset.seed=0 \
    dataset.permute=false \
    trainer.max_epochs=10 \
    hydra.run.dir=$task_0_OUTPUT_DIR \
    train.test=true \
    wandb.project=$WANDB_PROJECT \
    wandb.name="train_0" \
    callbacks.experiment_logger.output_file="$OUTPUT_CSV_FILE"

task_0_CHECKPOINT_PATH="${task_0_OUTPUT_DIR}/checkpoints/last.ckpt"

# ==================================================================
# --- ループでTask 1から9まで、それぞれモデルを訓練・評価 ---
# ==================================================================
for i in {1..10}; do
  
  # --- 1. Task i での訓練 ---
  RUN_NAME="train_task_${i}"
  prev_task_num=$((i - 1))
  noise_ratio=$(echo "scale=2; $i / 10" | bc)  # 0.1, 0.2, ..., 0.9, 1.0
  last_checkpoint="/work/outputs/ncps/DIL/pepper/task_${prev_task_num}/checkpoints/last.ckpt"
  OUTPUT_DIR="/work/outputs/ncps/DIL/pepper/task_${i}"

  echo ""
  echo "=================================================================="
  echo "--- 1. Training model on Task ${i} (seed=${i}) ---"
  echo "=================================================================="

  
  python3 train.py \
    experiment=$BASE_EXPERIMENT \
    dataset.permute=false \
    dataset.seed=0 \
    dataset.noise_ratio=$noise_ratio \
    trainer.max_epochs=10 \
    train.pretrained_model_path=$last_checkpoint \
    hydra.run.dir=$OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.name="ncps_train_${RUN_NAME}" \
    train.test=true \
    callbacks.experiment_logger.output_file=$OUTPUT_CSV_FILE \

  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
  echo "✅ Training for Task ${i} complete. Checkpoint: ${CHECKPOINT_PATH}"

  # --- 3. Test on all previous tasks ---
  echo "--- Testing model on ALL previous tasks (0 to ${i}) ---"

  # Use seq to generate the sequence for the inner loop
  for j in $(seq 0 $i); do
    noise_ratio_test=$(echo "scale=2; $j / 10" | bc) 
    echo "---   Testing on Task ${j} ---"

    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        train.pretrained_model_path=${CHECKPOINT_PATH} \
        train.test_only=true \
        dataset.noise_ratio=$noise_ratio_test \
        dataset.permute=false \
        dataset.seed=0 \
        callbacks.experiment_logger.output_file="$OUTPUT_CSV_FILE" \

  done
done
echo "All experiments finished!"