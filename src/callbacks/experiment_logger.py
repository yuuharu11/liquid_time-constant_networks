import csv
import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from rich import print

# --- Monitor Callback ---

class TrainingMonitor(Callback):
    """Measure training epoch duration and peak memory usage."""
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        self.batch_peak_memory = []
        if pl_module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.device.type == 'cuda':
            self.batch_peak_memory.append(torch.cuda.max_memory_allocated(pl_module.device))
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.perf_counter() - self.epoch_start_time
        pl_module.log("training/epoch_duration_sec", epoch_duration, on_step=False, on_epoch=True)
        
        if self.batch_peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.batch_peak_memory) / (1024**2)
            pl_module.log("training/avg_peak_mb", avg_peak_memory_mb, on_step=False, on_epoch=True)
            print(f"\n[cyan]TrainingMonitor: Avg Peak Memory this Epoch: {avg_peak_memory_mb:.2f} MB, Duration: {epoch_duration:.2f} sec[/cyan]")

class InferenceMonitor(Callback):
    """Measure inference latency and peak memory usage during testing."""
    def on_test_epoch_start(self, trainer, pl_module):
        self.latencies = []
        self.peak_memory = []
        print("\n[yellow]InferenceMonitor: Starting measurement...[/yellow]")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.start_time = time.perf_counter()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize()
            self.peak_memory.append(torch.cuda.max_memory_allocated(pl_module.device))
        
        latency = time.perf_counter() - self.start_time
        self.latencies.append(latency)

    def on_test_epoch_end(self, trainer, pl_module):
        stable_latencies = self.latencies[1:] if len(self.latencies) > 1 else self.latencies
        avg_stable_latency_ms = np.mean(stable_latencies) * 1000
        p95_latency_ms = np.percentile(stable_latencies, 95) * 1000
        
        pl_module.log("inference/avg_latency_ms", avg_stable_latency_ms)
        pl_module.log("inference/p95_latency_ms", p95_latency_ms)

        if self.peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.peak_memory) / (1024**2)
            pl_module.log("inference/avg_peak_mb", avg_peak_memory_mb)
        
        print(f"[green]InferenceMonitor: Avg Latency: {avg_stable_latency_ms:.2f} ms/batch, Avg Memory: {avg_peak_memory_mb:.2f} MB[/green]")

class CSVSummaryCallback(Callback):
    """Append training and testing results to a CSV file."""
    def __init__(self, output_file="results/summary.csv"):
        super().__init__()
        self.output_file = output_file
        self.training_results = {}
        self.headers = [
            "ステータス", "モデル", "データセット", "LSTMCell", "optimizer", "scheduler",
            "batch", "model.n_layers", "model.d_model", "units", "output_units",
            "エポック数", "ode_solver_unfolds", "input_mapping",
            "検証精度 (Val Acc)", "テスト精度 (Test Acc)", "平均レイテンシ (ms/バッチ)",
            "p95 レイテンシ (ms/バッチ)", "学習時間/epoch", "訓練時 Memoey Allocated [MB]",
            "推論時 Memoey Allocated [MB]", "チェックポイントパス", "wandbリンク", "備考"
        ]

    def on_train_end(self, trainer: Trainer, pl_module):
        """Call when training ends to capture final results."""
        hparams = pl_module.hparams
        metrics = trainer.callback_metrics

        self.training_results["モデル"] = hparams.model._name_
        self.training_results["データセット"] = hparams.dataset._name_
        self.training_results["LSTMCell"] = hparams.model.layer.get("mixed_memory", "N/A")
        self.training_results["optimizer"] = "adamw"
        self.training_results["scheduler"] = "cosine_warmup"
        self.training_results["batch"] = hparams.loader.batch_size
        self.training_results["model.n_layers"] = hparams.model.get("n_layers", "N/A")
        self.training_results["model.d_model"] = hparams.model.get("d_model", "N/A")
        
        units_list = hparams.model.layer.get("units", [])
        self.training_results["units"] = next((item.get("units") for item in units_list if "units" in item), "N/A")
        self.training_results["output_units"] = next((item.get("output_units") for item in units_list if "output_units" in item), "N/A")

        self.training_results["エポック数"] = trainer.current_epoch + 1
        self.training_results["ode_solver_unfolds"] = hparams.model.layer.get("ode_unfolds", "N/A")
        self.training_results["input_mapping"] = hparams.model.layer.get("input_mapping", "N/A")

        # 最終エポックの値を記録
        self.training_results["検証精度 (Val Acc)"] = metrics.get(f"val/{hparams.task.get('metric', 'accuracy')}", -1).item() 
        self.training_results["学習時間/epoch"] = metrics.get("training/epoch_duration_sec", -1).item()
        self.training_results["訓練時 Memoey Allocated [MB]"] = metrics.get("training/avg_peak_mb", -1).item()
        
        print("\n[bold cyan]CSVSummaryCallback: 訓練結果をキャプチャしました。テスト完了後に記録します。[/bold cyan]")

    def on_test_end(self, trainer: Trainer, pl_module):
        """全てのテストが完了した時に呼び出される"""
        results = self.training_results.copy() # 訓練結果をコピー
        metrics = trainer.callback_metrics

        # テスト結果を追加
        results["テスト精度 (Test Acc)"] = metrics.get(f"final/test/{pl_module.hparams.task.get('metric', 'accuracy')}", -1).item()
        results["平均レイテンシ (ms/バッチ)"] = metrics.get("inference/avg_latency_ms", -1).item()
        results["p95 レイテンシ (ms/バッチ)"] = metrics.get("inference/p95_latency_ms", -1).item()
        results["推論時 Memoey Allocated [MB]"] = metrics.get("inference/avg_peak_mb", -1).item()

        # その他情報を取得
        results["ステータス"] = "完了"
        if isinstance(trainer.logger, WandbLogger):
            results["wandbリンク"] = trainer.logger.experiment.url
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                results["チェックポイントパス"] = cb.best_model_path or "N/A"
        
        # CSVファイルに書き出す
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        file_exists = os.path.isfile(self.output_file)
        with open(self.output_file, "a", newline="", encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if not file_exists:
                writer.writeheader()
            
            row_data = {h: results.get(h, "") for h in self.headers}
            writer.writerow(row_data)
            
        print(f"\n[bold magenta]📊 全ての実験結果を {self.output_file} に記録しました。[/bold magenta]")