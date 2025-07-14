#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc/ltc-mnist-small"
WANDB_PROJECT="ltc_mnist"
LEARNING_RATE=0.01

# --- ループさせたいハイパーパラメータのリストを定義 ---
N_LAYERS_LIST=(2 3)
D_MODEL_LIST=(32 64)

echo "Starting hyperparameter sweep for LTC on MNIST..."
echo "n_layers values: ${N_LAYERS_LIST[@]}"
echo "d_model values: ${D_MODEL_LIST[@]}"
echo "Learning rate: ${LEARNING_RATE}"

# --- ネストしたforループで、全ての組み合わせを実行 ---
for n_layers in "${N_LAYERS_LIST[@]}"; do
  for d_model in "${D_MODEL_LIST[@]}"; do
    
    # --- 各実験で、ディレクトリ名とWandBの実行名を動的に生成 ---
    RUN_NAME="layers${n_layers}_dmodel${d_model}"
    OUTPUT_DIR="outputs/ltc/${RUN_NAME}"

    echo ""
    echo "=================================================================="
    echo "--- Running experiment: ${RUN_NAME} ---"
    echo "=================================================================="

    # 実行コマンド
    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        model.n_layers=$n_layers \
        model.d_model=$d_model \
        trainer.max_epochs=5 \
        model.layer.cell.d_model=$d_model \
        hydra.run.dir=$OUTPUT_DIR \
        wandb.project=$WANDB_PROJECT \
        wandb.name=$RUN_NAME
  
  done
done

echo ""
echo "finished! ---"