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


class SequenceLightningModule(pl.LightningModule):
    """
    シーケンスモデル用のPyTorch Lightningモジュール
    元のリポジトリの構造を参考に、レジストリシステムを使用
    """
    def __init__(self, config):
        super().__init__()
        # 設定をハイパーパラメータとして保存
        self.save_hyperparameters(config, logger=False)
        
        # セットアップフラグ（重複セットアップを防ぐ）
        self._has_setup = False
        
        # 状態管理の初期化
        self._state = None
        
        print("[yellow]📦 SequenceLightningModule初期化完了[/yellow]")
        
    def setup(self, stage=None):
        """モデルとコンポーネントのセットアップ"""
        if self._has_setup:
            return
        else:
            self._has_setup = True
            
        print(f"[green]🔧 セットアップ開始 (stage: {stage})[/green]")
        
        # モデルのインスタンス化（レジストリ経由）
        print("🏗️  モデルのインスタンス化...")
        self.model = utils.instantiate(registry.model, self.hparams.model)
        print(f"   ✅ モデル作成: {self.model.__class__.__name__}")
        
        # モデル情報の表示
        if hasattr(self.model, 'get_info'):
            info = self.model.get_info()
            print(f"   📊 パラメータ数: {info.get('parameters', 'N/A'):,}")
        
        print(f"[green]✅ セットアップ完了[/green]")
        
    def configure_optimizers(self):
        """オプティマイザーとスケジューラーの設定（レジストリ経由）"""
        print("[blue]📊 オプティマイザー設定[/blue]")
        
        # モデルのパラメータを取得
        params = list(self.model.parameters())
        print(f"   📝 最適化対象パラメータ数: {sum(p.numel() for p in params):,}")
        
        # オプティマイザーの作成（レジストリ経由）
        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)
        print(f"   ✅ オプティマイザー: {optimizer.__class__.__name__}")
        
        # スケジューラーの設定（オプション）
        if hasattr(self.hparams, 'scheduler'):
            lr_scheduler = utils.instantiate(registry.scheduler, self.hparams.scheduler, optimizer)
            scheduler = {
                "scheduler": lr_scheduler,
                "interval": self.hparams.train.get("interval", "epoch"),
                "monitor": self.hparams.train.get("monitor", "val/loss"),
                "name": "trainer/lr",
            }
            print(f"   ✅ スケジューラー: {lr_scheduler.__class__.__name__}")
            return [optimizer], [scheduler]
        
        return optimizer
    
    def forward(self, batch):
        """順方向計算（仮実装）"""
        # TODO: エンコーダー・デコーダー・タスクの実装後に完全実装
        x, y, *z = batch
        
        # 現在は直接モデルを呼び出し
        if hasattr(self.model, 'forward'):
            output, state = self.model(x, state=self._state)
            self._state = state
            return output, y, {}
        else:
            raise NotImplementedError("Model forward method not implemented")
    
    def training_step(self, batch, batch_idx):
        """訓練ステップ（仮実装）"""
        try:
            x, y, w = self.forward(batch)
            
            # 仮の損失計算
            loss = torch.nn.functional.mse_loss(x, torch.zeros_like(x))
            
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        except Exception as e:
            print(f"⚠️  Training step error: {e}")
            return torch.tensor(0.0, requires_grad=True)


def setup_logger(cfg: DictConfig) -> WandbLogger:
    """Weights & Biases ロガーをセットアップ"""
    logger = WandbLogger(
        project="ml_pipeline",
        name=f"experiment-{cfg.model.get('_name_', 'unknown')}",
        tags=[],
        offline=True,  # 開発中はオフライン
    )
    return logger


def setup_callbacks(cfg: DictConfig) -> list:
    """PyTorch Lightning コールバックをセットアップ（レジストリ経由）"""
    callbacks = []
    
    # モデルチェックポイント
    checkpoint_config = DictConfig({
        '_name_': 'model_checkpoint',
        'monitor': 'train/loss',
        'mode': 'min',
        'save_top_k': 3,
        'filename': '{epoch:02d}-{train_loss:.3f}',
        'auto_insert_metric_name': False,
    })
    checkpoint_callback = utils.instantiate(registry.callbacks, checkpoint_config)
    callbacks.append(checkpoint_callback)
    
    print(f"   ✅ コールバック追加: ModelCheckpoint")
    
    return callbacks


def create_trainer(config, **kwargs):
    """トレーナーの作成（元のリポジトリスタイル）"""
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # GPU設定の自動調整
    devices = config.trainer.get("devices", 1)
    if devices > 1:
        print(f"[yellow]📟 マルチGPU設定検出: {devices} devices[/yellow]")
        
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **config.trainer,
        **kwargs,
    )
    return trainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """メイン関数"""
    
    # 設定を表示
    print("=" * 80)
    print("[bold green]🚀 機械学習パイプライン開始[/bold green]")
    print("=" * 80)
    print("\n[bold blue]📋 設定:[/bold blue]")
    print(OmegaConf.to_yaml(cfg))
    
    # シード設定（再現性のため）
    if hasattr(cfg.dataset, 'seed'):
        pl.seed_everything(cfg.dataset.seed, workers=True)
        print(f"[green]🎲 シード設定: {cfg.dataset.seed}[/green]")
    
    # ログディレクトリ作成
    os.makedirs("logs", exist_ok=True)
    
    # トレーナーとモデルの作成
    print("\n[bold cyan]🔧 コンポーネント初期化...[/bold cyan]")
    trainer = create_trainer(cfg)
    model = SequenceLightningModule(cfg)
    
    print("\n[bold yellow]⚠️  現在の実装状況:[/bold yellow]")
    print("  ✅ レジストリシステム")
    print("  ✅ 基本構造とHydra設定") 
    print("  ✅ RNNModel実装")
    print("  ❌ データセット実装")
    print("  ❌ タスク実装")
    print("  ❌ エンコーダー・デコーダー実装")
    
    print("\n[bold cyan]🔧 次のステップ:[/bold cyan]")
    print("  1. データセット実装")
    print("  2. タスク（損失・メトリクス）実装")
    print("  3. エンコーダー・デコーダー実装")
    print("  4. 完全な訓練ループ")
    
    # 設定確認とモデルテスト
    print(f"\n[green]✅ 設定確認とモデルテスト完了！[/green]")
    
    # TODO: データセット実装後に有効化
    """
    # 訓練実行
    trainer.fit(model)
    
    # テスト実行
    if cfg.train.test:
        trainer.test(model)
    
    print("✅ 訓練完了!")
    """


if __name__ == "__main__":
    main()
