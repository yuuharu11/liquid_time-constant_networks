#!/bin/bash
set -e

# --- 実験の基本設定 ---
BASE_EXPERIMENT="ltc_ncps/pamap2"
WANDB_PROJECT="ltc_ncps-pamap2"

# --- ループさせたいハイパーパラメータのリストを定義 ---
# d_modelがモデルのサイズを決める唯一のパラメータになる
D_MODEL_LIST=(32 40 64 128)

echo "Starting hyperparameter sweep for LTC-NCPs on PAMAP2..."

# --- d_modelのループのみで実行 ---
for d_model in "${D_MODEL_LIST[@]}"; do
  # --- 各実験で、ディレクトリ名とWandBの実行名を動的に生成 ---
  # RUN_NAMEをd_modelのみを反映するように修正
  RUN_NAME="d_model${d_model}"
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
      wandb.project=$WANDB_PROJECT \
      wandb.name="ltc_ncps_${RUN_NAME}_PAMAP2" \
      model.d_model=$d_model # model.d_modelを上書きするだけでOK
done