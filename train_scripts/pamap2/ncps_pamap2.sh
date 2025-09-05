#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/pamap2"

# --- ループさせたいハイパーパラメータのリストを定義 ---
D_MODEL_LIST=(32 40 64 128)
# 試行回数分のseedを定義 (3回実行)
SEED_LIST=(21 123 567) 

echo "Starting multi-seed hyperparameter sweep for LTC-NCPs on PAMAP2..."

# --- seedのループを追加 ---
for seed in "${SEED_LIST[@]}"; do
  for d_model in "${D_MODEL_LIST[@]}"; do
    RUN_NAME="d_model${d_model}_seed${seed}"
    OUTPUT_DIR="outputs/ltc_ncps/pamap2/${RUN_NAME}"

    echo ""
    echo "=================================================================="
    echo "--- Running experiment: ${RUN_NAME} ---"
    echo "=================================================================="

    # 実行コマンド
    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        trainer.max_epochs=50 \
        hydra.run.dir=$OUTPUT_DIR \
        model.d_model=$d_model \
        train.seed=$seed \
        dataset.seed=$seed \
        wandb.project="PAMAP2" \
        wandb.tags="[ncps,PAMAP2,repeat]" \
        wandb.name="ncps-d_model${d_model}-seed${seed}"
  done
done