#!/bin/bash
set -e

# --- 実験設定 (編集箇所) ---
EXPERIMENT_CONFIG="ltc_ncps/uci_har" # Hydraの実験設定ファイル
RESULTS_DIR="/work/outputs/ltc_ncps/cil/er_sweep" # 結果を保存するベースディレクトリ
WANDB_PROJECT="UCI-HAR-CIL-ER-Sweep" # Weights & Biases のプロジェクト名
SEED=42
NUM_TASKS=5 # CILタスクの総数
EPOCHS=30

# --- ハイパーパラメータースイープ設定 ---
MEMORY_SIZES=(100 250)
REPLAY_BATCH_SIZES=(32 64 96)

echo "🚀 Starting CIL with Experience Replay sweep for UCI-HAR..."
echo "=================================================================="

# --- ハイパーパラメータのループ ---
for mem_size in "${MEMORY_SIZES[@]}"; do
  for replay_bs in "${REPLAY_BATCH_SIZES[@]}"; do

    SWEEP_NAME="mem${mem_size}_bs${replay_bs}"
    echo ""
    echo "--- Running Sweep: ${SWEEP_NAME} ---"

    # --- スイープごとの初期化 ---
    LAST_CHECKPOINT_PATH="" # 最初のタスクでは pretrained_model_path を使わない
    BUFFER_PATH="/work/buffer/er_buffer_${SWEEP_NAME}.pt"
    CSV_LOG_PATH="/work/csv/uci-har/cil-er/${SWEEP_NAME}.csv"

    # 以前のバッファが残っていれば削除
    rm -f $BUFFER_PATH
    # ログファイルのディレクトリを作成
    mkdir -p "$(dirname "$CSV_LOG_PATH")"


    # --- CIL タスクループ (0からNUM_TASKS-1まで) ---
    for i in $(seq 0 $(($NUM_TASKS - 1))); do
      
      TASK_NAME="Task_${i}"
      OUTPUT_DIR="${RESULTS_DIR}/${SWEEP_NAME}/${TASK_NAME}"
      GROUP_NAME="${SWEEP_NAME}" # WandBでのグループ名

      echo "--- Training ${TASK_NAME} ---"

      # --- 学習コマンドの組み立て ---
      CMD="python3 train.py \
          experiment=${EXPERIMENT_CONFIG} \
          dataset=uci_har_cil \
          train.seed=$SEED \
          dataset.seed=$SEED \
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

      # Task 0 以外では、前のチェックポイントを読み込む
      if [ -n "$LAST_CHECKPOINT_PATH" ]; then
        CMD="$CMD train.pretrained_model_path=$LAST_CHECKPOINT_PATH"
      fi
      
      # --- 学習の実行 ---
      eval $CMD

      # --- チェックポイントのパスを更新 ---
      LAST_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
      
      # --- 過去タスクを含む全タスクでの評価 ---
      echo "--- Evaluating ${TASK_NAME} model on all learned tasks (0 to ${i}) ---"
      for j in $(seq 0 $i); do
        
        PAST_TASK_NAME="Task_${j}"
        EVAL_DIR="${OUTPUT_DIR}/eval_on_${PAST_TASK_NAME}"
        
        echo "---   Testing on ${PAST_TASK_NAME} ---"
        python3 train.py \
            experiment=${EXPERIMENT_CONFIG} \
            dataset=uci_har_cil \
            train.seed=$SEED \
            dataset.seed=$SEED \
            dataset.task_id=$j \
            hydra.run.dir=$EVAL_DIR \
            train.pretrained_model_path=$LAST_CHECKPOINT_PATH \
            train.test_only=true \
            callbacks.experiment_logger.output_file=$CSV_LOG_PATH
      done
      echo "--------------------------------------------------"

      echo "---  Training and evaluation Joint Training ---"
      python3 train.py \
          experiment=${EXPERIMENT_CONFIG} \
          dataset=uci_har_cil \
          train.seed=$SEED \
          trainer.max_epochs=$EPOCHS \
          dataset.seed=$SEED \
          dataset.task_id=$i \
          train.test=true \
          callbacks.experiment_logger.output_file=$CSV_LOG_PATH

    done
    echo "Sweep ${SWEEP_NAME} finished."
    echo "=================================================================="
  done
done

echo "✅ All CIL experiments finished."
echo "=================================================================="