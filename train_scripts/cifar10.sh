#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/cifar10"
WANDB_PROJECT="ncps_cifar10"

# --- 変更して試したい `units` のリストを定義 ---
UNITS=256

# --- 最適だったハイパーパラメータを固定 ---
D_MODEL=128
OUTPUT_UNITS=10
ODE_UNFOLDS=(2 3)
lr=5e-3
N_LAYERS=1

# --- `units` ごとにループを実行 ---
for ode_unfolds in "${ODE_UNFOLDS[@]}"; do

  # --- 各実験で、ユニークな名前と出力先を生成 ---
  RUN_NAME="ode_unfolds${ode_unfolds}_conv2d"
  OUTPUT_DIR="outputs/ncps/cifar10/${RUN_NAME}"

  echo ""
  echo "=================================================================="
  echo "--- Running experiment: ${RUN_NAME} ---"
  echo "=================================================================="

  # 実行コマンド
  python3 train.py \
      experiment=$BASE_EXPERIMENT \
      model.n_layers=$N_LAYERS \
      model.d_model=$D_MODEL \
      model.layer.units.1.units=$UNITS \
      model.layer.units.2.output_units=$OUTPUT_UNITS \
      model.layer.ode_unfolds=$ode_unfolds \
      optimizer.lr=$lr \
      loader.batch_size=128 \
      trainer.max_epochs=150 \
      train.test=true \
      hydra.run.dir=$OUTPUT_DIR \
      wandb.project=$WANDB_PROJECT \
      wandb.name=$RUN_NAME
      
done

echo ""
echo "=================================================================="
echo "--- 📜 All final sweep experiments finished! ---"
echo "=================================================================="