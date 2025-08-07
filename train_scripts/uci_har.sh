#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="rnn/uci_har"
WANDB_PROJECT="rnn-uci_har"

# --- ループさせたいハイパーパラメータのリストを定義 ---
UNITS_LIST=(8 16 32 64 128)
N_LAYERS_LIST=(1 2 3 4)

echo "Starting hyperparameter sweep for RNN on UCI HAR..."

# --- ネストしたforループで、全ての組み合わせを実行 ---
for units in "${UNITS_LIST[@]}"; do
  for n_layers in "${N_LAYERS_LIST[@]}"; do
    # --- 各実験で、ディレクトリ名とWandBの実行名を動的に生成 ---
    RUN_NAME="new_units${units}_layers${n_layers}"
    OUTPUT_DIR="outputs/rnn/uci_har/${RUN_NAME}"

    echo ""
    echo "=================================================================="
    echo "--- Running experiment: ${RUN_NAME} ---"
    echo "=================================================================="

    # 実行コマンド
    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        model.n_layers=$n_layers \
        model.d_model=$units \
        trainer.max_epochs=10 \
        hydra.run.dir=$OUTPUT_DIR \
        wandb.project=$WANDB_PROJECT \
        wandb.name="rnn_${RUN_NAME}_UCI_HAR"
  done
done
