#!/usr/bin/env python3
"""
機械学習パイプラインのメイン実行スクリプト

使用方法:
    python train.py                    # デフォルト設定で実行
    python train.py dataset=cifar      # 異なるデータセットを使用
    python train.py model=resnet       # 異なるモデルを使用
"""

import os
import warnings
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from rich import print

# レジストリシステムと utilities
import src.utils as utils
from src.utils import registry

# 警告を抑制（オプション）
warnings.filterwarnings("ignore", category=UserWarning)

# TensorFloat32を有効化（大きなモデルの訓練を高速化）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_logger(cfg: DictConfig) -> WandbLogger:
    """Weights & Biases ロガーをセットアップ"""
    logger = WandbLogger(
        project="sequence_models",
        name=f"experiment-{cfg.model.get('_name_', 'unknown')}",
        tags=[cfg.dataset.get('_name_', 'unknown')],
        offline=True,  # 開発中はオフライン
    )
    return logger


def setup_callbacks(cfg: DictConfig) -> list:
    """PyTorch Lightning コールバックをセットアップ（レジストリ経由）"""
    callbacks = []
    
    # モデルチェックポイント
    checkpoint_config = DictConfig({
        '_target_': 'src.callbacks.model_checkpoint.ModelCheckpoint',
        'monitor': 'val_loss',
        'mode': 'min',
        'save_top_k': 3,
        'filename': '{epoch:02d}-{val_loss:.3f}',
        'save_last': True,
        'verbose': True,
    })
    callbacks.append(utils.instantiate(registry.callbacks, checkpoint_config))
    
    # Early stopping
    early_stop_config = DictConfig({
        '_target_': 'src.callbacks.early_stopping.EarlyStopping',
        'monitor': 'val_loss',
        'patience': 10,
        'mode': 'min',
        'verbose': True,
    })
    callbacks.append(utils.instantiate(registry.callbacks, early_stop_config))
    
    # Learning rate monitor
    lr_monitor_config = DictConfig({
        '_target_': 'src.callbacks.learning_rate_monitor.LearningRateMonitor',
        'logging_interval': 'epoch',
    })
    callbacks.append(utils.instantiate(registry.callbacks, lr_monitor_config))
    
    return callbacks


def create_dataloader(cfg: DictConfig):
    """データローダーを作成"""
    print(f"[cyan]📊 データセット準備: {cfg.dataset._name_}[/cyan]")
    
    # データセットをレジストリから作成
    dataset = utils.instantiate(registry.dataset, cfg.dataset)
    
    # データローダー設定
    loader_config = cfg.loader
    
    # データローダーを作成
    train_loader = dataset.train_dataloader(
        batch_size=loader_config.batch_size,
        shuffle=True,
        num_workers=loader_config.get('num_workers', 4),
    )
    
    val_loader = dataset.val_dataloader(
        batch_size=loader_config.batch_size,
        shuffle=False,
        num_workers=loader_config.get('num_workers', 4),
    )
    
    test_loader = dataset.test_dataloader(
        batch_size=loader_config.batch_size,
        shuffle=False,
        num_workers=loader_config.get('num_workers', 4),
    )
    
    print(f"   ✅ データローダー作成完了")
    print(f"   📈 訓練データ: {len(train_loader)} バッチ")
    print(f"   📊 検証データ: {len(val_loader)} バッチ")
    print(f"   🧪 テストデータ: {len(test_loader)} バッチ")
    
    return train_loader, val_loader, test_loader, dataset


def create_task_and_model(cfg: DictConfig, dataset):
    """タスクとモデルを作成"""
    print(f"[green]🏗️  モデルとタスク作成[/green]")
    
    # モデルの設定を準備
    model_config = cfg.model.copy()
    model_config.d_input = dataset.d_input
    
    # モデルを作成
    model = utils.instantiate(registry.model, model_config)
    print(f"   ✅ モデル作成: {model.__class__.__name__}")
    
    # タスクの設定を準備
    task_config = cfg.task.copy()
    task_config.model = model_config  # モデル設定をタスクに渡す
    task_config.d_output = dataset.d_output
    task_config.optimizer = cfg.optimizer
    task_config.scheduler = cfg.get('scheduler', None)
    
    # タスクを作成
    task = utils.instantiate(registry.task, task_config)
    
    # モデルをタスクに設定
    task.set_model(model)
    
    print(f"   ✅ タスク作成: {task.__class__.__name__}")
    print(f"   📊 入力次元: {dataset.d_input}")
    print(f"   🎯 出力次元: {dataset.d_output}")
    
    # モデル情報を表示
    if hasattr(model, 'get_info'):
        info = model.get_info()
        print(f"   📊 パラメータ数: {info.get('parameters', 'N/A'):,}")
    
    return task


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """メイン関数"""
    print("[bold blue]🚀 機械学習パイプライン開始[/bold blue]")
    print(f"⚙️  設定: {OmegaConf.to_yaml(cfg)}")
    
    # シードの設定
    if hasattr(cfg, 'seed'):
        pl.seed_everything(cfg.seed, workers=True)
        print(f"🌱 シード設定: {cfg.seed}")
    
    # データローダーの作成
    train_loader, val_loader, test_loader, dataset = create_dataloader(cfg)
    
    # タスクとモデルの作成
    task = create_task_and_model(cfg, dataset)
    
    # コールバックとロガーの設定
    logger = setup_logger(cfg)
    callbacks = setup_callbacks(cfg)
    
    # トレーナーの設定
    trainer_config = cfg.trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        precision=trainer_config.get('precision', 32),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=cfg.get('deterministic', False),
    )
    
    print(f"[green]🏃‍♂️ 訓練開始[/green]")
    
    # 訓練の実行
    start_time = time.time()
    trainer.fit(task, train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"[green]✅ 訓練完了 (時間: {training_time:.2f}s)[/green]")
    
    # テストの実行
    if test_loader is not None:
        print(f"[blue]🧪 テスト実行[/blue]")
        test_result = trainer.test(task, test_loader)
        print(f"[blue]✅ テスト完了[/blue]")
        print(f"📊 テスト結果: {test_result}")
    
    print("[bold green]🎉 パイプライン完了![/bold green]")


if __name__ == "__main__":
    main()
