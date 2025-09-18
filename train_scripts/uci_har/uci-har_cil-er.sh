#!/bin/bash
set -e

# --- 実験設定 (編集箇所) ---
EXPERIMENT_CONFIG="ltc_ncps/uci_har"
RESULTS_DIR="/work/outputs/ltc_ncps/uci_har/cil_each/er_sweep"
WANDB_PROJECT="UCI-HAR-CIL_EACH-ER"
SEEDS=(1 2 3 4 5)      # SEEDを複数指定
NUM_TASKS=3
EPOCHS=10

# --- ハイパーパラメータースイープ設定 ---
MEMORY_SIZES=(0 50 100 500 1000 5000)
REPLAY_BATCH_SIZES=(10 50 100 500)

echo "🚀 Starting CIL with Experience Replay sweep for UCI-HAR..."
echo "=================================================================="

# --- ハイパーパラメータのループ ---
for seed in "${SEEDS[@]}"; do
  for mem_size in "${MEMORY_SIZES[@]}"; do
    for replay_bs in "${REPLAY_BATCH_SIZES[@]}"; do

      SWEEP_NAME="seed${seed}_mem${mem_size}_bs${replay_bs}"
      echo ""
      echo "--- Running Sweep: ${SWEEP_NAME} ---"

      LAST_CHECKPOINT_PATH=""
      BUFFER_PATH="/work/buffer/cil_each/er_buffer_${SWEEP_NAME}.pt"
      CSV_LOG_PATH="/work/csv/uci-har/cil-each/er/${SWEEP_NAME}.csv"

      rm -f $BUFFER_PATH
      mkdir -p "$(dirname "$CSV_LOG_PATH")"

      for i in $(seq 0 $(($NUM_TASKS - 1))); do
        
        TASK_NAME="Task_${i}"
        OUTPUT_DIR="${RESULTS_DIR}/${SWEEP_NAME}/${TASK_NAME}"
        GROUP_NAME="${SWEEP_NAME}"

        echo "--- Training ${TASK_NAME} ---"

        CMD="python3 train.py \
            experiment=${EXPERIMENT_CONFIG} \
            dataset=uci_har_cil \
            train.seed=$seed \
            dataset.seed=$seed \
            dataset.task_id=$i \
            trainer.max_epochs=$EPOCHS \
            hydra.run.dir=$OUTPUT_DIR \
            wandb.project=$WANDB_PROJECT \
            wandb.group=$GROUP_NAME \
            wandb.name=\"${SWEEP_NAME}-${TASK_NAME}\" \
            callbacks.experiment_logger.output_file=$CSV_LOG_PATH \
            train.replay._name_=exact_replay \
            train.replay.memory_size=$mem_size \
            train.replay.batch_size=$replay_bs \
            train.replay.buffer_path=$BUFFER_PATH"

        if [ -n "$LAST_CHECKPOINT_PATH" ]; then
          CMD="$CMD train.pretrained_model_path=$LAST_CHECKPOINT_PATH"
        fi
        
        eval $CMD
        LAST_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"

        echo "--- Evaluating ${TASK_NAME} model on all learned tasks (0 to ${i}) ---"
        for j in $(seq 0 $i); do
          PAST_TASK_NAME="Task_${j}"
          EVAL_DIR="${OUTPUT_DIR}/eval_on_${PAST_TASK_NAME}"
          echo "---   Testing on ${PAST_TASK_NAME} ---"
          python3 train.py \
              experiment=${EXPERIMENT_CONFIG} \
              dataset=uci_har_cil \
              train.seed=$seed \
              dataset.seed=$seed \
              dataset.task_id=$j \
              hydra.run.dir=$EVAL_DIR \
              train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
              train.test_only=true \
              callbacks.experiment_logger.output_file=$CSV_LOG_PATH
        done

        python3 train.py \
            experiment=${EXPERIMENT_CONFIG} \
            dataset=uci_har_cil \
            train.seed=$seed \
            dataset.seed=$seed \
            dataset.task_id=$i \
            dataset.overall=true \
            train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
            train.test_only=true \
            callbacks.experiment_logger.output_file=$CSV_LOG_PATH

      done
      echo "Sweep ${SWEEP_NAME} finished."
      echo "=================================================================="
    done
  done
done

echo "✅ All CIL experiments finished."
echo "=================================================================="
