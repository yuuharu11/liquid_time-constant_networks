import torch
from pytorch_lightning import Callback
import logging
import os
import socket
import datetime

log = logging.getLogger(__name__)

class ProfilingCallback(Callback):
    def __init__(self, warmup_steps=1, active_steps=5, dirpath="logs/profiler", enable=False):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.dirpath = dirpath
        self.profiler = None
        self.enable = enable

    def on_train_start(self, trainer, pl_module):
        if not self.enable:
            return

        # ランごとのユニークなサブディレクトリ
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.dirpath, run_name)
        os.makedirs(run_dir, exist_ok=True)

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1,
                                             warmup=self.warmup_steps,
                                             active=self.active_steps,
                                             repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.profiler.__enter__()
        log.info(f"Profiler initialized. Saving to {run_dir}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.enable and self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self.enable and self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            log.info(f"Profiler trace saved. Run 'tensorboard --logdir={self.dirpath}' to view.")
            self.enable = False
