import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class WeightVisualizerCallback(Callback):
    """
    エポックの終わりに全ての重み行列とマスクを自動で可視化しWandBに記録するコールバック
    """
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        print(f"\n[Visualizer] Logging weights and masks on train end.")

        # named_parametersで全パラメータを取得
        for name, param in pl_module.model.named_parameters():
            # パラメータが2次元以上なら可視化
            if param.ndim < 2:
                continue

            weights = param.data.clone().cpu().numpy()
            # マスクはパラメータ名で取得（PackNetのcurrent_masksのキーがparam名の場合）
            mask = pl_module.pruner.current_masks.get(name)
            if mask is None:
                continue
            mask = mask.clone().cpu().numpy()

            # Figure作成
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'{name} | Epoch {trainer.current_epoch + 1}', fontsize=16)

            sns.heatmap(weights.reshape(weights.shape[0], -1), ax=axes[0], cmap='viridis', cbar=False)
            axes[0].set_title('Weights')
            axes[0].set_xlabel('Flattened')
            axes[0].set_ylabel('Rows')

            sns.heatmap(mask.reshape(mask.shape[0], -1), ax=axes[1], cmap='tab10', cbar=True)
            axes[1].set_title('PackNet Mask (Task IDs)')
            axes[1].set_xlabel('Flattened')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            trainer.logger.experiment.log({
                f"weights/{name}": wandb.Image(fig),
                "trainer/global_step": trainer.global_step
            })
            plt.close(fig)