#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/uci_har"
WANDB_PROJECT="uci_har"

# --- ループさせたいハイパーパラメータのリストを定義 ---
UNITS_LIST=(128)
N_LAYERS_LIST=(1 2)
ODE_UNFOLDS_LIST=(1)
input=9
output=6


echo "Starting hyperparameter sweep for LTC on MNIST..."

# --- ネストしたforループで、全ての組み合わせを実行 ---
for units in "${UNITS_LIST[@]}"; do
  for n_layers in "${N_LAYERS_LIST[@]}"; do
    for ode_unfolds in "${ODE_UNFOLDS_LIST[@]}"; do
      # --- 各実験で、ディレクトリ名とWandBの実行名を動的に生成 ---
      RUN_NAME="new_units${units}_layers${n_layers}_unfolds${ode_unfolds}"
      OUTPUT_DIR="outputs/ncps/uci_har/${RUN_NAME}"

      echo ""
      echo "=================================================================="
      echo "--- Running experiment: ${RUN_NAME} ---"
      echo "=================================================================="

      # 実行コマンド
      python3 train.py \
          experiment=$BASE_EXPERIMENT \
          model.n_layers=$n_layers \
          model.layer.units.1.units=$units \
          model.layer.ode_unfolds=$ode_unfolds \
          trainer.max_epochs=100 \
          hydra.run.dir=$OUTPUT_DIR \
          wandb.project=$WANDB_PROJECT \
          wandb.name="ncps_${RUN_NAME}_UCI_HAR"
    done
  done
done
