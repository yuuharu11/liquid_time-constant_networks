#!/bin/bash
set -e

# --- 実験設定 ---
# ループで実行したいモデルのリスト
MODELS=("rnn" "cnn" "lstm" "ncps") 
EXPERIMENT_BASE="uci_har"
RESULTS_BASE="/work/outputs"
CSV_BASE="/work/csv/uci-har/dil/im" # 結果を保存するCSVのパス
SEED=42

# ループで試したいノイズレベルのリスト
NOISE_LEVELS=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
# 評価対象のタスクID（6 = 全てのセンサー）
TASK_ID=6 

# --- モデルループ (一番外側) ---
for MODEL in "${MODELS[@]}"; do
    EXPERIMENT_CONFIG="${MODEL}/${EXPERIMENT_BASE}"
    CSV_LOG_PATH="${CSV_BASE}/${MODEL}.csv"
    

    echo ""
    echo "=================================================================="
    echo "--- Starting evaluation for model: ${MODEL} ---"
    echo "=================================================================="

    # --- Noise Level ループ (内側) ---
    for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do

        TASK_NAME="TaskID_${TASK_ID}_Noise_${NOISE_LEVEL}"
        echo "---   Testing with Noise Level: ${NOISE_LEVEL} ---"

        python3 train.py \
            experiment=$EXPERIMENT_CONFIG \
            trainer.max_epochs=10 \
            train.seed=$SEED \
            dataset.seed=$SEED \
            dataset=uci_har_dil \
            dataset.task_id=$TASK_ID \
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