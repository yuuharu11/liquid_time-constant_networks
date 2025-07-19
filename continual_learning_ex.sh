#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/mnist"
WANDB_PROJECT="ltc_mnist"
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