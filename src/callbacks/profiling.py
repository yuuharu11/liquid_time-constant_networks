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
        """
        学習が始まる直前に一度だけ呼び出されるフック。
        ここでFLOPsとパラメータ数を計算します。
        """
        if not self.enable:
            return  
        
        log.info("Calculating model complexity (FLOPs and Parameters)...")
        try:
            # データローダからダミーの入力バッチを取得
            # `trainer.datamodule` は PyTorch Lightning 1.7 以降で推奨
            if hasattr(trainer, 'datamodule'):
                dummy_input, _ = next(iter(trainer.datamodule.train_dataloader()))
            else: # 古いバージョンとの互換性のため
                dummy_input, _ = next(iter(trainer.train_dataloader))
                
            # 入力テンソルの形状を取得
            # (batch, seq_len, features) -> (seq_len, features) のように、バッチサイズ1で計算
            input_shape = tuple(dummy_input.shape[1:])

            # ptflopsで計算
            macs, params = get_model_complexity_info(
                pl_module.model,
                input_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )

            # GFLOPsに変換 (MACs * 2 ≈ FLOPs)
            gflops = (macs * 2) / 1e9
            mparams = params / 1e6

            # 結果をログに出力し、ロガー（TensorBoardなど）に保存
            log.info(f"Model Complexity: {gflops:.2f} GFLOPs, {mparams:.2f} M Params")
            pl_module.log("model_GFLOPs", gflops, rank_zero_only=True)
            pl_module.log("model_MParams", mparams, rank_zero_only=True)

        except Exception as e:
            log.warning(f"Failed to calculate model complexity: {e}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        各トレーニングステップの開始時に呼び出されるフック。
        torch.profilerを制御します。
        """
        if not self.enable:
            return
        
        if self.profiler is None:
            # プロファイラを初期化
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=self.warmup_steps, active=self.active_steps),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.dirpath, "trace")),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            self.profiler.__enter__()

        if batch_idx < self.warmup_steps + self.active_steps + 1:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        """
        トレーニング終了時にプロファイラを停止します。
        """        
        if self.enable and self.profiler:
            self.profiler.__exit__(None, None, None)
            log.info(f"Profiler trace saved to '{self.dirpath}/trace'. Use TensorBoard to view.")
            self.enable = False  # 一度だけ実行するために無効化