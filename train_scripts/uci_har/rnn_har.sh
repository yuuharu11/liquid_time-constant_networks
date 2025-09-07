#!/bin/bash

# 実験設定: uci_har.yaml を使用
EXPERIMENT_CONFIG="experiment=rnn/uci_har"

# 結果を保存するルートディレクトリ
RESULTS_DIR="outputs/rnn/uci_har_standard"

# 試行するシード値のリスト
SEEDS=(42 123 555 777 1000)

echo "====================================================="
echo "== Running 5 experiments with different seeds for RNN"
echo "====================================================="

# --- シード値を変更しながらループ ---
for seed in "${SEEDS[@]}"
do
  echo "-----------------------------------------------------"
  echo "== Running Experiment with Seed: $seed"
  echo "-----------------------------------------------------"

  # Hydra のコマンドライン引数で seed を上書き
  # 出力ディレクトリとWandBの名前もシードごとに変更
  python3 train.py \
    $EXPERIMENT_CONFIG \
    train.seed=$seed \
    dataset.seed=$seed \
    hydra.run.dir="$RESULTS_DIR/seed_$seed" \
    trainer.max_epochs=50 \
    wandb.project="UCI-HAR-Standard" \
    wandb.group="RNN_100k" \
    wandb.name="rnn_100k_seed_$seed"

  # エラーが発生したらスクリプトを停止
  if [ $? -ne 0 ]; then
    echo "Error running experiment for seed $seed. Aborting."
    exit 1
  fi
done

echo "====================================================="
echo "== All 5 experiments finished."
echo "== Results saved in: $RESULTS_DIR"
echo "====================================================="