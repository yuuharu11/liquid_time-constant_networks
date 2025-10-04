import torch
from pytorch_lightning import Callback, Trainer
from ptflops import get_model_complexity_info
import logging
import os

log = logging.getLogger(__name__)

class ProfilingCallback(Callback):
    """
    学習開始時にモデルのFLOPsとパラメータ数を計算し、
    最初の数ステップの詳細なプロファイル情報をTensorBoard用に保存するCallback。
    """

    def __init__(self, warmup_steps=1, active_steps=1, dirpath="profiler_logs", enable=False):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.dirpath = dirpath
        self.profiler = None
        self.enable = enable

    def on_train_start(self, trainer, pl_module):
        if not self.enable:
            return

        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=self.warmup_steps, active=self.active_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.dirpath),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.profiler.__enter__()
        log.info("Profiler initialized.")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.enable and self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self.enable and self.profiler:
            self.profiler.__exit__(None, None, None)
            log.info(f"Profiler trace saved to '{self.dirpath}'. Use TensorBoard to view.")
            self.enable = False
