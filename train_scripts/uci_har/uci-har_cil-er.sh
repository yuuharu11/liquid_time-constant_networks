#!/bin/bash
set -e

# --- 共通の実験設定 ---
EXPERIMENT_CONFIG="ltc_ncps/uci_har_cil" # CIL用の実験設定ファイル
LOADER_BATCH_SIZE=200
SEED=42

# --- CILのシナリオ設定 (2クラスずつ増加) ---
# Task 0: クラス 0, 1
# Task 1: クラス 0, 1, 2, 3
# Task 2: クラス 0, 1, 2, 3, 4, 5
TASK_CLASSES=(
    "[3,4]"
    "[3,4,5]"
    "[0,3,4,5]"
    "[0,1,3,4,5]"
    "[0,1,2,3,4,5]"
)

# --- ERのハイパーパラメータースイープ設定 ---
MEMORY_SIZES=(500 1500 3000)
REPLAY_BATCH_SIZES=(32 64 96)

echo "Starting CIL with Experience Replay sweep for UCI-HAR..."

# --- ハイパーパラメータのループ ---
for mem_size in "${MEMORY_SIZES[@]}"; do
  for replay_bs in "${REPLAY_BATCH_SIZES[@]}"; do

    echo ""
    echo "=================================================================="
    echo "--- Running Sweep: memory_size=${mem_size}, replay_batch_size=${replay_bs} ---"
    echo "=================================================================="

    # リプレイバッファのパスをスイープごとに一意に設定
    BUFFER_PATH="/work/buffer/er_buffer_mem${mem_size}_bs${replay_bs}.pt"
    # 以前のバッファが残っていれば、新しいスイープの開始時に削除
    rm -f $BUFFER_PATH

    # --- 各タスクのループ ---
    for task_id in ${!TASK_CLASSES[@]}; do
      
      CLASSES_FOR_TASK=${TASK_CLASSES[$task_id]}
      RUN_NAME="mem${mem_size}_bs${replay_bs}/task${task_id}"

      echo "--- Training Task ${task_id} on classes ${CLASSES_FOR_TASK} ---"

      python3 train.py \
          experiment=$EXPERIMENT_CONFIG \
          train.seed=$SEED \
          hydra.run.dir="/work/outputs/ltc-ncps/cil/er/${RUN_NAME}" \
          train.pretrained_model_path="${last_path}" \
          train.replay._name_=exact_replay \
          train.replay.memory_size=$mem_size \
          train.replay.batch_size=$replay_bs \
          train.replay.buffer_path=$BUFFER_PATH \

        last_path="/work/outputs/ltc-ncps/cil/er/${RUN_NAME}/checkpoints/last.ckpt" # 次のタスクの初期化に使用
    done

    # --- 全タスク学習後の最終評価 ---
    # ここに、全タスクのテストデータを使った最終評価スクリプトの呼び出しなどを追加できます
    echo "--- Final evaluation for sweep (mem ${mem_size}, bs ${replay_bs}) can be performed now ---"

  done
done