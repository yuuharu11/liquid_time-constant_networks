#!/bin/bash
set -e

# ==================================================================
# --- 1. 実験で試行するラムダ値のリスト ---
# ==================================================================
# ここに試したいラムダの値をスペース区切りで指定します。
# (例: 0はEWCなしのベースライン、1, 10, 100... はEWCの効果を確認)
LAMBDA_VALUES=(10 100 500 5000 10000 50000)


# ==================================================================
# --- 2. 各ラムダ値に対して、連続学習の全プロセスを実行するループ ---
# ==================================================================
for EWC_LAMBDA in "${LAMBDA_VALUES[@]}"; do

    echo "##################################################################"
    echo "###   STARTING NEW EXPERIMENT RUN WITH LAMBDA = ${EWC_LAMBDA}   ###"
    echo "##################################################################"

    # --- 実験の基本設定 ---
    BASE_EXPERIMENT="ltc_ncps/mnist"
    WANDB_PROJECT="ncps_ewc_continual_learning"

    # --- 各ラムダ値に応じたファイルパスを定義 ---
    BASE_OUTPUT_DIR="/work/outputs/ncps/DIL/ewc_lambda${EWC_LAMBDA}"
    CSV_OUTPUT_FILE="/work/test/DIL/ewc_lambda${EWC_LAMBDA}.csv"
    EWC_PARAMS_FILE="${BASE_OUTPUT_DIR}/ewc_params.pt"


    # ==================================================================
    # --- Step 1: Task 0 での初期訓練 (EWCの基礎となる重要度を計算) ---
    # ==================================================================
    TASK0_DIR="${BASE_OUTPUT_DIR}/train_task_0"
    echo "--- Training on Task 0 (seed=0) and computing initial Fisher matrix for lambda=${EWC_LAMBDA} ---"

    python3 train.py \
        experiment=$BASE_EXPERIMENT \
        train.regularization._name_=ewc \
        train.regularization.lambda=$EWC_LAMBDA \
        train.regularization.param_path=$EWC_PARAMS_FILE \
        dataset.seed=0 \
        dataset.permute=false \
        trainer.max_epochs=10 \
        hydra.run.dir=$TASK0_DIR \
        train.test=true \
        wandb.project=$WANDB_PROJECT \
        wandb.name="ewc${EWC_LAMBDA}_task_0" \
        callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE


    # ==================================================================
    # --- Step 2: ループでTask 1から9まで、EWCを使いながら連続学習 ---
    # ==================================================================
    for i in {1..9}; do
    
        # --- 前のタスクのチェックポイントパスを計算 ---
        prev_task_num=$((i - 1))
        last_checkpoint="${BASE_OUTPUT_DIR}/train_task_${prev_task_num}/checkpoints/last.ckpt"

        # --- 現在のタスクの情報を設定 ---
        RUN_NAME="train_task_${i}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

        echo ""
        echo "=================================================================="
        echo "--- Training model on Task ${i} (seed=${i}) with EWC (lambda=${EWC_LAMBDA}) ---"
        echo "=================================================================="
        
        python3 train.py \
            experiment=$BASE_EXPERIMENT \
            train.regularization._name_=ewc \
            train.regularization.lambda=$EWC_LAMBDA \
            train.regularization.param_path=$EWC_PARAMS_FILE \
            dataset.permute=true \
            dataset.seed=$i \
            trainer.max_epochs=10 \
            train.pretrained_model_path=$last_checkpoint \
            hydra.run.dir=$OUTPUT_DIR \
            wandb.project=$WANDB_PROJECT \
            wandb.name="ewc${EWC_LAMBDA}_task_${i}" \
            train.test=true \
            callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE

        CURRENT_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
        echo "✅ Training for Task ${i} complete. Checkpoint: ${CURRENT_CHECKPOINT_PATH}"

        # --- 3. 訓練済みモデルを、過去の全タスクで評価する ---
        echo "--- Testing model on ALL previous tasks (0 to ${i}) ---"

        for j in $(seq 0 $i); do
            echo "---      Testing on Task ${j} ---"

            PERMUTE_FLAG_TEST=true
            if [ $j -eq 0 ]; then
                PERMUTE_FLAG_TEST=false
            fi

            python3 train.py \
                experiment=$BASE_EXPERIMENT \
                train.pretrained_model_path=${CURRENT_CHECKPOINT_PATH} \
                train.test_only=true \
                dataset.permute=$PERMUTE_FLAG_TEST \
                dataset.seed=$j \
                callbacks.experiment_logger.output_file=$CSV_OUTPUT_FILE
        done
    done

    echo ""
    echo "=================================================================="
    echo "--- ✅ Experiment for LAMBDA=${EWC_LAMBDA} finished! ---"
    echo "--- Results have been appended to: ${CSV_OUTPUT_FILE} ---"
    echo "=================================================================="
    echo ""

done # --- ラムダ値のループ終了 ---

echo "🎉🎉🎉 All experiments for all lambda values are complete! 🎉🎉🎉"