#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
WANDB_PROJECT="ltc_mnist"

# --- ループさせたいハイパーパラメータのリストを定義 ---
UNITS_LIST=(16 32 64 128 256)
N_LAYERS_LIST=(1 2 3)
ODE_UNFOLDS_LIST=(3 6 9)


echo "Starting hyperparameter sweep for LTC on MNIST..."

# --- ネストしたforループで、全ての組み合わせを実行 ---
for units in "${UNITS_LIST[@]}"; do
  for n_layers in "${N_LAYERS_LIST[@]}"; do
    for ode_unfolds in "${ODE_UNFOLDS_LIST[@]}"; do
      # --- 各実験で、ディレクトリ名とWandBの実行名を動的に生成 ---
      RUN_NAME="units${units}_layers${n_layers}_odeunfolds${ode_unfolds}"
      OUTPUT_DIR="/work/outputs/ncps/mnist/${RUN_NAME}/checkpoints/last.ckpt"

      echo ""
      echo "=================================================================="
      echo "--- Running experiment: ${RUN_NAME} ---"
      echo "=================================================================="

      # 実行コマンド
      python3 train.py \
          experiment=$BASE_EXPERIMENT \
          train.test_only=true \
          train.ckpt=$OUTPUT_DIR \
          callbacks.experiment_logger.output_file="/work/test/mnist.csv"
    done
  done
done
