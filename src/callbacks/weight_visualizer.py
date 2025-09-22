import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class WeightVisualizerCallback(Callback):
    """
    エポックの終わりに重み行列とマスクを可視化してWandBに記録するコールバック
    """
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Nエポックごとに実行
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # PackNetアーキテクチャでない場合は何もしない
        if pl_module.arch_name != "packnet" or pl_module.pruner is None:
            return

        print(f"\n[Visualizer] Logging weights and masks for epoch {trainer.current_epoch + 1}...")

        # モデルの各レイヤーをループ
        for module_idx, module in enumerate(pl_module.model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # 1. 重み行列を取得
                weights = module.weight.data.clone().cpu().numpy()
                
                # 2. 対応するマスクを取得
                mask = pl_module.pruner.current_masks.get(module_idx)
                if mask is None:
                    continue
                mask = mask.clone().cpu().numpy()

                # ログ用の名前を作成 (例: layer.0.weights)
                layer_name = f"module.{module_idx}.{type(module).__name__}"

                # Figureを作成
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{layer_name} | Epoch {trainer.current_epoch + 1}', fontsize=16)

                # 重みの可視化 (Heatmap)
                sns.heatmap(weights.reshape(weights.shape[0], -1), ax=axes[0], cmap='viridis', cbar=False)
                axes[0].set_title('Weights')
                axes[0].set_xlabel('Neurons (Flattened)')
                axes[0].set_ylabel('Neurons')


                # マスクの可視化 (タスクIDで色分け)
                sns.heatmap(mask.reshape(mask.shape[0], -1), ax=axes[1], cmap='tab10', cbar=True)
                axes[1].set_title('PackNet Mask (Task IDs)')
                axes[1].set_xlabel('Neurons (Flattened)')
                

                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # WandBに画像として記録
                trainer.logger.experiment.log({
                    f"weights/{layer_name}": wandb.Image(fig),
                    "trainer/global_step": trainer.global_step # これがないとstepがずれる
                })
                
                # Figureを閉じてメモリを解放
                plt.close(fig)