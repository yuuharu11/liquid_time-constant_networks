#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
WANDB_PROJECT="ncps_continual_learning"

# --- 結果を保存するためのBash配列を初期化 ---
declare -a results_on_own_task # 自身のタスクでの精度
declare -a results_on_task0    # Task0での精度

task_0_OUTPUT_DIR="/work/outputs/ncps/DIL/random/train_task_0"

python3 train.py \
    experiment=$BASE_EXPERIMENT \
    dataset.seed=0 \
    dataset.permute=false \
    trainer.max_epochs=10 \
    hydra.run.dir=$task_0_OUTPUT_DIR \
    train.test=true \
    wandb.project=$WANDB_PROJECT \
    wandb.name="ncps_train_0" \
    callbacks.experiment_logger.output_file="/work/test/DIL.csv"

task_0_CHECKPOINT_PATH="${task_0_OUTPUT_DIR}/checkpoints/last.ckpt"

# ==================================================================
# --- ループでTask 1から9まで、それぞれモデルを訓練・評価 ---
# ==================================================================
for i in {1..9}; do
  
  # --- 1. Task i での訓練 ---
  RUN_NAME="train_task_${i}"
  prev_task_num=$((i - 1))
  last_checkpoint="/work/outputs/ncps/DIL/random/train_task_${prev_task_num}/checkpoints/last.ckpt"
  OUTPUT_DIR="/work/outputs/ncps/DIL/random/train_task_${i}"

  echo ""
  echo "=================================================================="
  echo "--- 1. Training model on Task ${i} (seed=${i}) ---"
  echo "=================================================================="

  
  python3 train.py \
    experiment=$BASE_EXPERIMENT \
    dataset.permute=true \
    dataset.seed=$i \
    trainer.max_epochs=10 \
    train.pretrained_model_path=$last_checkpoint \
    hydra.run.dir=$OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.name="ncps_train_${RUN_NAME}" \
    train.test=true \
    callbacks.experiment_logger.output_file="/work/test/DIL.csv" \

  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
  echo "✅ Training for Task ${i} complete. Checkpoint: ${CHECKPOINT_PATH}"

  # --- 3. Test on all previous tasks ---
  echo "--- Testing model on ALL previous tasks (0 to ${i}) ---"

  # Use seq to generate the sequence for the inner loop
  for j in $(seq 0 $i); do
      echo "---   Testing on Task ${j} ---"

      # Set the permute flag based on the task number (j)
      PERMUTE_FLAG_TEST=true
      if [ $j -eq 0 ]; then
          PERMUTE_FLAG_TEST=false
      fi

      python3 train.py \
          experiment=$BASE_EXPERIMENT \
          train.pretrained_model_path=${CHECKPOINT_PATH} \
          train.test_only=true \
          dataset.permute=$PERMUTE_FLAG_TEST \
          dataset.seed=$j \
          callbacks.experiment_logger.output_file="/work/test/DIL.csv" \

  done
done


# --- 4. 全ての実験が完了した後、結果をまとめて出力 ---
echo ""
echo "=================================================================="
echo "--- 📜 Final Transfer Learning Test Results ---"
echo "=================================================================="
echo "Trained on | Acc on Own Task | Acc on Task 0"
echo "------------------------------------------------"
echo "=================================================================="
echo "All experiments finished!"