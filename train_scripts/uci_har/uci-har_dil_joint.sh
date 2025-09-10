#!/bin/bash
set -e

# --- 実験設定 ---
# ループで実行したいモデルのリスト
MODELS=("rnn" "cnn" "lstm" "ltc_ncps") 
EXPERIMENT_BASE="uci_har"
RESULTS_BASE="/work/outputs"
CSV_BASE="/work/csv/uci-har/dil-ex/im" # 結果を保存するCSVのパス
SEED=42

# ループで試したいノイズレベルのリスト
TASK_ID=(0 1 2 3 4 5 6)
NOISE_LEVEL=-1.0

# --- モデルループ (一番外側) ---
for MODEL in "${MODELS[@]}"; do
    EXPERIMENT_CONFIG="${MODEL}/${EXPERIMENT_BASE}"
    CSV_LOG_PATH="${CSV_BASE}/${MODEL}.csv"
    

    echo ""
    echo "=================================================================="
    echo "--- Starting evaluation for model: ${MODEL} ---"
    echo "=================================================================="

    # --- Noise Level ループ (内側) ---
    for task_id in "${TASK_ID[@]}"; do

        TASK_NAME="TaskID_${task_id}_Noise_${NOISE_LEVEL}"
        echo "---   Testing with Noise Level: ${NOISE_LEVEL} ---"

        python3 train.py \
            experiment=$EXPERIMENT_CONFIG \
            trainer.max_epochs=10 \
            train.seed=$SEED \
            dataset.seed=$SEED \
            dataset=uci_har_dil \
            dataset.task_id=$task_id \
            dataset.noise_level=$NOISE_LEVEL \
            dataset.joint_training=true \
            train.test=true \
            callbacks.experiment_logger.output_file=$CSV_LOG_PATH
    done
done

echo ""
echo "=================================================================="
echo "--- ✅ All models and noise level evaluations finished! ---"
echo "=================================================================="