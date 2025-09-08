#!/bin/bash

# 実験設定: LSTM用の設定ファイルを指定
EXPERIMENT_CONFIG="experiment=lstm/uci_har"

# 結果を保存するルートディレクトリ
RESULTS_DIR="outputs/lstm/uci_har_standard"

# 試行するシード値のリスト
SEEDS=(42 123 555 777 1000)

echo "====================================================="
echo "== Running 5 experiments with different seeds for LSTM"
echo "====================================================="

for seed in "${SEEDS[@]}"
do
  echo "-----------------------------------------------------"
  echo "== Running Experiment with Seed: $seed"
  echo "-----------------------------------------------------"

  python3 train.py \
    $EXPERIMENT_CONFIG \
    train.seed=$seed \
    dataset.seed=$seed \
    dataset.task_id=6 \
    dataset.noise_level=0.5 \
    hydra.run.dir="$RESULTS_DIR/seed_$seed" \
    wandb.project="UCI-HAR-Standard" \
    wandb.group="LSTM_100k_noise" \
    wandb.name="lstm_100k_seed_$seed" \
    trainer.max_epochs=50

  if [ $? -ne 0 ]; then
    echo "Error running experiment for seed $seed. Aborting."
    exit 1
  fi
done

echo "====================================================="
echo "== All 5 experiments finished."
echo "== Results saved in: $RESULTS_DIR"
echo "====================================================="