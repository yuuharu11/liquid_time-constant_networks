#!/bin/bash
set -e

# --- 実験設定: モデルとタスク ---
MODEL_NAMES=("NCP")
EXPERIMENT_BASE="uci_har"
RESULTS_BASE="/work/outputs"
CSV_BASE="/work/csv/temp" # 結果の保存先
SEED=1

# --- テストするノイズレベルのリスト ---
NOISE_LEVELS=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.2" "1.4" "1.6" "1.8" "2.0")

# --- モデルループ (一番外側) ---
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_NAME_LOWER=${MODEL_NAME,,}
    EXPERIMENT_CONFIG="${MODEL_NAME_LOWER}/${EXPERIMENT_BASE}"
    
    # 事前に訓練した、クリーンなデータでのベースラインモデルを指定
    PRETRAINED_MODEL="${RESULTS_BASE}/${MODEL_NAME_LOWER}/${EXPERIMENT_BASE}_standard/seed_${SEED}/checkpoints/last-30epoch.ckpt"

    echo ""
    echo "=================================================================="
    echo "--- Starting evaluation for model: ${MODEL_NAME} ---"
    echo "=================================================================="

    # --- Task ID ループ (0から6まで) ---
    for task_id in {0..6}; do

        echo "-----------------------------------------------------"
        echo "--- Evaluating for Task ID: ${task_id} (Sensor Group ${task_id})"
        echo "-----------------------------------------------------"


      CSV_LOG_PATH="${CSV_BASE}/sensor_${task_id}/${MODEL_NAME_LOWER}.csv"
        # --- Noise Level ループ (内側) ---
        for noise_level in "${NOISE_LEVELS[@]}"; do

            TASK_NAME="TaskID_${task_id}_Noise_${noise_level}"
            echo "---   Testing with Noise Level: ${noise_level} ---"

            python3 train.py \
                experiment=$EXPERIMENT_CONFIG \
                train.seed=$SEED \
                dataset.seed=$SEED \
                dataset=uci_har_dil \
                dataset.task_id=$task_id \
                dataset.noise_level=$noise_level \
                train.pretrained_model_path=$PRETRAINED_MODEL \
                train.test_only=true \
                callbacks.experiment_logger.output_file=$CSV_LOG_PATH
        done
    done
done

echo ""
echo "=================================================================="
echo "--- ✅ All models and DIL task evaluations finished ---"
echo "=================================================================="