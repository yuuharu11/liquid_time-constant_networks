import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich import print

class MemoryMonitor(Callback):
    """
    A PyTorch Lightning callback to monitor memory usage during training.
    """

    def __init__(self, enable=True):
        self.enable = enable

    def log_memory_usage(self, hook_name: str):
        if torch.cuda.is_available():
            # Allocating memory at the start of the hook
            allocated_memory = torch.cuda.memory_allocated() / 1024**2
            # Peak memory usage during the hook
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"[MemoryMonitor] {hook_name:<25}: "
                  f"Allocated = {allocated_memory:.2f} MB, "
                  f"Peak = {peak_memory:.2f} MB")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.enable: return
        self.log_memory_usage("on_train_epoch_start")

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int) -> None:
        if not self.enable: return
        self.log_memory_usage("on_train_batch_start")

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.enable: return
        # This hook is called after the backward pass
        self.log_memory_usage("on_after_backward")

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int) -> None:
        if not self.enable: return
        # This hook is called after the training batch ends (after optimizer.step())
        self.log_memory_usage("on_train_batch_end")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.enable: return
        self.log_memory_usage("on_train_epoch_end")
        # Reset peak memory at the end of the epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()