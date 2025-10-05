import torch
from pytorch_lightning import Callback
from pytorch_lightning.profilers import PyTorchProfiler
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
        self.enable = enable
        self.profiler = None

    def on_train_start(self, trainer, pl_module):
        if not self.enable:
            return

        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(
            self.dirpath,
            run_name,
            "plugins", "profile", socket.gethostname()
        )
        os.makedirs(run_dir, exist_ok=True)

        # ✅ PyTorchProfiler は trainer に登録して使う
        self.profiler = PyTorchProfiler(
            dirpath=run_dir,
            filename="profile",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)
        )

        trainer.profiler = self.profiler
        log.info(f"Profiler initialized. Saving traces to {run_dir}")

    def on_train_end(self, trainer, pl_module):
        if self.enable and self.profiler is not None:
            log.info(f"Profiler trace saved. Run 'tensorboard --logdir={self.dirpath}' to view.")
