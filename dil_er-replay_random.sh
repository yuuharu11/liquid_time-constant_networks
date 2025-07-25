#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
WANDB_PROJECT="ncps_er_hyperparam_sweep" # プロジェクト名を一新

# --- ループさせたいハイパーパラメータのリストを定義 ---
MEMORY_SIZES=(500 3000 5000)
BATCH_SIZES=(8 32 128)

# ==================================================================
# --- memory_sizeとbatch_sizeの全ての組み合わせでループ ---
# ==================================================================
for mem_size in "${MEMORY_SIZES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do

    echo ""
    echo "##################################################################"
    echo "### Starting Run: memory_size=${mem_size}, replay_batch_size=${bs} ###"
    echo "##################################################################"

    # --- 各実験ごとに、ユニークな出力先とCSVファイルを定義 ---
    RUN_ID="mem${mem_size}_bs${bs}"
    BASE_OUTPUT_DIR="/work/outputs/ncps/DIL/er_sweep/${RUN_ID}"
    CSV_OUTPUT_FILE="/work/test/DIL/er_sweep_${RUN_ID}.csv"
    
    # --- Step 1: Task 0 での初期訓練 (リプレイバッファの作成) ---
    TASK0_DIR="${BASE_OUTPUT_DIR}/train_task_0"
    echo "--- Training on Task 0 (seed=0) to create the replay buffer ---"

    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        train.replay._name_=exact_replay \
        train.replay.memory_size=$mem_size \
        train.replay.batch_size=$bs \
        train.replay.n_replay=1 \
        train.replay.buffer_path="${BASE_OUTPUT_DIR}/replay_buffer.pt" \
        dataset.seed=0 \
        dataset.permute=false \
        trainer.max_epochs=10 \
        hydra.run.dir=$TASK0_DIR \
        wandb.project=$WANDB_PROJECT \
        wandb.name="${RUN_ID}_task_0" \
        callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE

    # --- Step 2: ループでTask 1から9まで、Task 0を復習しながら連続学習 ---
    for i in {1..9}; do
      
      prev_task_num=$((i - 1))
      last_checkpoint="${BASE_OUTPUT_DIR}/train_task_${prev_task_num}/checkpoints/last.ckpt"
      RUN_NAME="train_task_${i}"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

      echo ""
      echo "=================================================================="
      echo "--- Training model on Task ${i} (seed=${i}), replaying Task 0 ---"
      echo "=================================================================="
      
      python3 train.py \
        experiment=$BASE_EXPERIMENT \
        train.replay._name_=exact_replay \
        train.replay.memory_size=$mem_size \
        train.replay.batch_size=$bs \
        train.replay.n_replay=1 \
        train.replay.buffer_path="${BASE_OUTPUT_DIR}/replay_buffer.pt" \
        dataset.permute=true \
        dataset.seed=$i \
        trainer.max_epochs=10 \
        train.pretrained_model_path=$last_checkpoint \
        hydra.run.dir=$OUTPUT_DIR \
        wandb.project=$WANDB_PROJECT \
        wandb.name="${RUN_ID}_task_${i}" \
        callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE

      CURRENT_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
      echo "✅ Training for Task ${i} complete. Checkpoint: ${CURRENT_CHECKPOINT_PATH}"

      # --- 3. 訓練済みモデルを、過去の全タスクで評価する ---
      echo "--- Testing model on ALL previous tasks (0 to ${i}) ---"

      for j in $(seq 0 $i); do
        echo "---   Testing on Task ${j} ---"

        PERMUTE_FLAG_TEST=true
        if [ $j -eq 0 ]; then
          PERMUTE_FLAG_TEST=false
        fi

        python3 train.py \
            experiment=$BASE_EXPERIMENT \
            train.pretrained_model_path=${CURRENT_CHECKPOINT_PATH} \
            train.test_only=true \
            dataset.permute=$PERMUTE_FLAG_TEST \
            dataset.seed=$j \
            callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE
      done
    done

  done # batch_size loop
done # memory_size loop

echo ""
echo "=================================================================="
echo "--- 📜 All hyperparameter sweep experiments finished! ---"
echo "=================================================================="