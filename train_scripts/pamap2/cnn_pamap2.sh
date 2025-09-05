#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="cnn/pamap2"

# --- ループさせたいハイパーパラメータのリストを定義 ---
D_MODEL_LIST=(64 128 256) # チャンネル数
N_LAYERS_LIST=(4 6 8)      # 層数
echo "Starting hyperparameter sweep for CNN on PAMAP2..."

for d_model in "${D_MODEL_LIST[@]}"; do
  for n_layers in "${N_LAYERS_LIST[@]}"; do

    RUN_NAME="d_model${d_model}_layers${n_layers}"
    OUTPUT_DIR="outputs/cnn/pamap2/${RUN_NAME}"

    echo ""
    echo "=================================================================="
    echo "--- Running experiment: ${RUN_NAME} ---"
    echo "=================================================================="

    # 実行コマンド
    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        trainer.max_epochs=200 \
        hydra.run.dir=$OUTPUT_DIR \
        model.d_model=$d_model \
        model.n_layers=$n_layers \
        wandb.name="cnn-${RUN_NAME}"
  done
done