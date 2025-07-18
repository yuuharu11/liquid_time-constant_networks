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
      OUTPUT_DIR="outputs/ncps/mnist/${RUN_NAME}"

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
          trainer.max_epochs=10 \
          hydra.run.dir=$OUTPUT_DIR \
          wandb.project=$WANDB_PROJECT \
          wandb.name="ncps_${RUN_NAME}_MNIST"
    done
  done
done

echo ""
echo "finished! ---"
echo "start mnist continual learning experiments with NCPs"
# Bashの正しいforループ構文に修正
for i in {1..10}; do
  echo "running $i th continual learning experiment"
  OUTPUT_DIR="outputs/ncps/continual_learning_${i}"
  RUN_NAME="continual_learning_${i}"

  # 実行コマンド
  python3 train.py \
    experiment=$BASE_EXPERIMENT \
    dataset.permute=true \
    dataset.seed=$i \
    model.n_layers=1 \
    model.layer.units.1.units=64 \
    model.layer.ode_unfolds=6 \
    trainer.max_epochs=10 \
    hydra.run.dir=$OUTPUT_DIR \
    wandb.project=$WANDB_PROJECT \
    wandb.name="ncps_continual_learning_${RUN_NAME}_MNIST"
done