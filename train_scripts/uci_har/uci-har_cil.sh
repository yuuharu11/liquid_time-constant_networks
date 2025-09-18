#!/bin/bash
set -e

# --- 実験設定 ---
MODELS=("rnn" "lstm" "cnn" "ltc_ncps")
EXPERIMENT_BASE="uci_har"
WANDB_PROJECT="UCI-HAR-CIL_EACH"
NUM_TASKS=3
NUM_RUNS=5   # ← 5回実験
SEEDS=(1 2 3 4 5)  # ← 5つの異なるシード値
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "##################################################################"
    echo "=== Run $SEED / $NUM_RUNS ==="
    echo "##################################################################"
    echo "=================================================================="
    echo "=== Running CIL experiments for model: $MODEL (seed=$SEED) ==="
    echo "=================================================================="

    RESULTS_DIR="/work/outputs/${MODEL}/uci_har/cil_each/seed_${SEED}"
    CSV_LOG_PATH="/work/csv/uci-har/cil_each/${MODEL}/seed${SEED}.csv"
    GROUP_NAME="${MODEL^^}-CIL"
    LAST_CHECKPOINT_PATH="null"

    # --- CIL タスクループ ---
    for i in $(seq 0 $(($NUM_TASKS - 1))); do
      TASK_NAME="Task_${i}"
      OUTPUT_DIR="${RESULTS_DIR}/${TASK_NAME}"
      EPOCHS=10

      echo "--- [Phase 1] Training on ${TASK_NAME} ---"
      python3 train.py \
        experiment=${MODEL}/${EXPERIMENT_BASE} \
        dataset=uci_har_cil \
        train.seed=$SEED \
        dataset.seed=$SEED \
        dataset.task_id=$i \
        train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
        train.test=true \
        trainer.max_epochs=$EPOCHS \
        hydra.run.dir=$OUTPUT_DIR \
        wandb.project=$WANDB_PROJECT \
        wandb.group=$GROUP_NAME \
        wandb.name="${MODEL^^}-${TASK_NAME}-seed${SEED}" \
        callbacks.experiment_logger.output_file=$CSV_LOG_PATH

      CHECKPOINT_JUST_TRAINED="${OUTPUT_DIR}/checkpoints/last.ckpt"
      LAST_CHECKPOINT_PATH=$CHECKPOINT_JUST_TRAINED

      # --- [Phase 2] 過去タスクでの評価 ---
      echo "--- Evaluating on all previous tasks (0 to ${i}) ---"
      for j in $(seq 0 $(($i - 1))); do
        PAST_TASK_NAME="Task_${j}"
        EVAL_DIR="${RESULTS_DIR}/${TASK_NAME}/eval_on_${PAST_TASK_NAME}"

        echo "---   Testing on ${PAST_TASK_NAME} ---"
        python3 train.py \
          experiment=${MODEL}/${EXPERIMENT_BASE} \
          dataset=uci_har_cil \
          train.seed=$SEED \
          dataset.seed=$SEED \
          dataset.task_id=$j \
          train.pretrained_model_path=$CHECKPOINT_JUST_TRAINED \
          train.test_only=true \
          callbacks.experiment_logger.output_file=$CSV_LOG_PATH
      done

      # --- [Phase 3] 総合性能での評価 ---
      python3 train.py \
        experiment=${MODEL}/${EXPERIMENT_BASE} \
        dataset=uci_har_cil \
        train.seed=$SEED \
        dataset.seed=$SEED \
        dataset.task_id=$i \
        dataset.overall=true \
        train.pretrained_model_path=$CHECKPOINT_JUST_TRAINED \
        train.test_only=true \
        callbacks.experiment_logger.output_file=$CSV_LOG_PATH
    done
  done
done

echo "=================================================================="
echo "--- ✅ All CIL experiments finished (all seeds) ---"
echo "=================================================================="
