#!/bin/bash
set -e

# --- 実験設定: モデルとタスク ---
MODEL_NAMES=("RNN" "CNN" "LSTM" "LTC_NCPS")
EXPERIMENT_BASE="uci_har"
RESULTS_BASE="/work/outputs"
CSV_BASE="/work/csv/uci-har/base"
SEED=42

TASKS=(
  "6,0.0"   # Task 0: クリーンデータ
  "6,0.1"   # Task 1: 軽微ノイズ
  "6,0.2"   # Task 2: 小ノイズ
  "6,0.3"   # Task 3: 中ノイズ
  "6,0.4"   # Task 4: やや大きめノイズ
  "6,0.5"   # Task 5: 大ノイズ
  "6,0.6"   # Task 6: かなり大きなノイズ
  "6,0.7"   # Task 7: 深刻ノイズ
  "6,0.8"   # Task 8: 深刻ノイズ
  "6,0.9"   # Task 9: 全壊に近いノイズ
  "6,1.0"   # Task 10: 全壊に近いノイズ
)

NUM_TASKS=${#TASKS[@]}

# --- モデルループ ---
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    EXPERIMENT_CONFIG="${MODEL_NAME,,}/${EXPERIMENT_BASE}"  # 小文字に変換してパスに使用
    RESULTS_DIR="${RESULTS_BASE}/${MODEL_NAME,,}/${EXPERIMENT_BASE}/base-cl"
    CSV_LOG_PATH="${CSV_BASE}/${MODEL_NAME,,}.csv"
    PRETRAINED_MODEL="${RESULTS_BASE}/${MODEL_NAME,,}/${EXPERIMENT_BASE}_standard/seed_${SEED}/checkpoints/last-base.ckpt"

    echo ""
    echo "=================================================================="
    echo "--- Starting evaluation for model: ${MODEL_NAME} ---"
    echo "=================================================================="

    # --- タスクループ ---
    for i in $(seq 0 $(($NUM_TASKS - 1))); do
        IFS=',' read -r task_id noise_level <<< "${TASKS[$i]}"
        TASK_NAME="Task_${i}_ID_${task_id}_Noise_${noise_level}"

        echo ""
        echo "-----------------------------------------------------"
        echo "--- Evaluating on ${TASK_NAME} ---"
        echo "-----------------------------------------------------"

        python3 train.py \
            experiment=$EXPERIMENT_CONFIG \
            train.seed=$SEED \
            dataset.seed=$SEED \
            dataset.task_id=$task_id \
            dataset.noise_level=$noise_level \
            train.pretrained_model_path=$PRETRAINED_MODEL \
            train.test_only=true \
            callbacks.experiment_logger.output_file=$CSV_LOG_PATH
    done

done

echo ""
echo "=================================================================="
echo "--- ✅ All models and DIL task evaluations finished ---"
echo "=================================================================="
