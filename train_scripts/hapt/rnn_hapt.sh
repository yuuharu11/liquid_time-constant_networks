#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="rnn/hapt"
WANDB_PROJECT="HAPT" # 全てのHAPT実験をまとめるプロジェクト

# --- ループさせたいハイパーパラメータのリストを定義 ---
D_MODEL_LIST=(80 128) # ユニット数
N_LAYERS_LIST=(1 2 3)      # 層数
SEED=42                    # 固定seed

echo "Starting hyperparameter sweep for RNN on HAPT dataset..."

# --- ネストしたforループで、全ての組み合わせを実行 ---
for d_model in "${D_MODEL_LIST[@]}"; do
  for n_layers in "${N_LAYERS_LIST[@]}"; do
    
    RUN_NAME="d_model${d_model}_layers${n_layers}"
    OUTPUT_DIR="outputs/rnn/hapt/${RUN_NAME}"

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
        model.n_layers=$n_layers \
        train.seed=$SEED \
        dataset.seed=$SEED \
        wandb.project=$WANDB_PROJECT \
        wandb.name="rnn-${RUN_NAME}" \
        wandb.group="rnn" # WandB上で'rnn'グループにまとめる
  done
done