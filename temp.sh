#!/bin/bash
set -e

# --- テストしたいハイパーパラメータのリストを定義 ---
MEMORY_SIZES=(500 3000 5000)
BATCH_SIZES=(8 32 128)

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
# チェックポイントが保存されている親ディレクトリ
BASE_CHECKPOINT_DIR="/work/outputs/ncps/DIL/er_sweep_replay"
# テスト結果のCSVを保存するディレクトリ
CSV_OUTPUT_DIR="/work/"

# ==================================================================
# --- memory_sizeとbatch_sizeの全ての組み合わせでループ ---
# ==================================================================
for mem_size in "${MEMORY_SIZES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    
    RUN_ID="mem${mem_size}_bs${bs}"
    echo ""
    echo "------------------------------------------------------------------"
    echo "--- Testing combination: ${RUN_ID} ---"
    echo "------------------------------------------------------------------"

    # --- この組み合わせに対応するパスを動的に生成 ---
    CHECKPOINT_PATH="${BASE_CHECKPOINT_DIR}/${RUN_ID}/train_task_0/checkpoints/last.ckpt"
    CSV_OUTPUT_FILE="${CSV_OUTPUT_DIR}/test_results_${RUN_ID}.csv"

    # --- チェックポイントファイルが存在するか確認 ---
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "[WARNING] Checkpoint not found, skipping: ${CHECKPOINT_PATH}"
        continue # 次のループへ
    fi

    # --- テストを実行 ---
    # train.test_only=true を使い、指定したチェックポイントでテストのみ行う
    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        train.pretrained_model_path=$CHECKPOINT_PATH \
        train.test_only=true \
        dataset.seed=0 \
        dataset.permute=false \
        callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE

    echo "✅ Test for ${RUN_ID} complete. Results saved to ${CSV_OUTPUT_FILE}"

  done # batch_size ループ終了
done # memory_size ループ終了

echo ""
echo "=================================================================="
echo "--- 📜 All hyperparameter tests finished! ---"
echo "=================================================================="