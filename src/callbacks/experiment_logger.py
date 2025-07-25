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

# --- Monitor Callbacks (変更の必要なし) ---

class TrainingMonitor(Callback):
    """訓練中のGPUメモリ使用量とエポックごとの学習時間を計測する"""
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
        pl_module.log("training/epoch_duration_sec", epoch_duration, on_step=False, on_epoch=True, logger=True)
        
        if self.batch_peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.batch_peak_memory) / (1024**2)
            pl_module.log("training/avg_peak_mb", avg_peak_memory_mb, on_step=False, on_epoch=True, logger=True)
            print(f"\n[cyan]TrainingMonitor: Avg Peak Memory this Epoch: {avg_peak_memory_mb:.2f} MB, Duration: {epoch_duration:.2f} sec[/cyan]")

class InferenceMonitor(Callback):
    """テスト（推論）中のレイテンシとGPUメモリ使用量を計測する"""
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
        if not self.latencies: return
        stable_latencies = self.latencies[1:] if len(self.latencies) > 1 else self.latencies
        avg_stable_latency_ms = np.mean(stable_latencies) * 1000
        p95_latency_ms = np.percentile(stable_latencies, 95) * 1000
        
        pl_module.log("inference/avg_latency_ms", avg_stable_latency_ms, logger=True)
        pl_module.log("inference/p95_latency_ms", p95_latency_ms, logger=True)

        avg_peak_memory_mb = 0.0
        if self.peak_memory and pl_module.device.type == 'cuda':
            avg_peak_memory_mb = np.mean(self.peak_memory) / (1024**2)
            pl_module.log("inference/avg_peak_mb", avg_peak_memory_mb, logger=True)
        
        print(f"[green]InferenceMonitor: Avg Latency: {avg_stable_latency_ms:.2f} ms/batch, Avg Memory: {avg_peak_memory_mb:.2f} MB[/green]")

# --- 主役となるCSVロガー (ここを修正) ---

class CSVSummaryCallback(Callback):
    """訓練とテストの結果をまとめてCSVファイルに追記する"""
    def __init__(self, output_file="results/summary.csv"):
        super().__init__()
        self.output_file = output_file
        self.results_cache = {}
        self.has_written_this_run = False
        self.headers = [
            "ステータス", "モデル", "データセット", "dataset_seed", "LSTMCell", "optimizer", "scheduler",
            "batch", "model.n_layers", "model.d_model", "units", "output_units",
            "エポック数", "ode_solver_unfolds", "input_mapping",
            "検証精度 (Val Acc)", "テスト精度 (Test Acc)", "平均レイテンシ (ms/バッチ)",
            "p95 レイテンシ (ms/バッチ)", "学習時間/epoch", "訓練時 Memoey Allocated [MB]",
            "推論時 Memoey Allocated [MB]", "チェックポイントパス", "wandbリンク", "備考"
        ]

    def _get_metric(self, metrics_dict, key, default=-1.0):
        """メトリクスを安全に取得し、テンソルなら.item()を呼ぶ補助関数"""
        value = metrics_dict.get(key, default)
        return value.item() if isinstance(value, torch.Tensor) else value

    def _capture_hparams(self, pl_module):
        """ハイパーパラメータをキャッシュに保存する補助関数"""
        hparams = pl_module.hparams
        self.results_cache["モデル"] = hparams.model._name_
        self.results_cache["データセット"] = hparams.dataset._name_
        self.results_cache["dataset_seed"] = hparams.dataset.get("seed", "N/A")
        self.results_cache["LSTMCell"] = hparams.model.layer.get("mixed_memory", "N/A")
        self.results_cache["optimizer"] = "adamw"
        self.results_cache["scheduler"] = "cosine_warmup"
        self.results_cache["batch"] = hparams.loader.batch_size
        self.results_cache["model.n_layers"] = hparams.model.get("n_layers", "N/A")
        self.results_cache["model.d_model"] = hparams.model.get("d_model", "N/A")
        
        units_list = hparams.model.layer.get("units", [])
        self.results_cache["units"] = next((item.get("units") for item in units_list if "units" in item), "N/A")
        self.results_cache["output_units"] = next((item.get("output_units") for item in units_list if "output_units" in item), "N/A")
        self.results_cache["ode_solver_unfolds"] = hparams.model.layer.get("ode_unfolds", "N/A")
        self.results_cache["input_mapping"] = hparams.model.layer.get("input_mapping", "N/A")

    def on_train_end(self, trainer: Trainer, pl_module):
        """訓練完了時に呼び出され、訓練結果を一時保存する"""
        self._capture_hparams(pl_module)
        metrics = trainer.logged_metrics

        self.results_cache["エポック数"] = trainer.current_epoch + 1
        val_metric_key = f"val/{pl_module.hparams.task.get('metric', 'accuracy')}"
        self.results_cache["検証精度 (Val Acc)"] = self._get_metric(metrics, val_metric_key)
        self.results_cache["学習時間/epoch"] = self._get_metric(metrics, "training/epoch_duration_sec")
        self.results_cache["訓練時 Memoey Allocated [MB]"] = self._get_metric(metrics, "training/avg_peak_mb")
        
        print("\n[bold cyan]CSVSummaryCallback: 訓練結果をキャプチャしました。テスト完了後に記録します。[/bold cyan]")

    def on_test_start(self, trainer: Trainer, pl_module):
        """テスト開始時に呼び出され、書き込みフラグをリセットする"""
        self.has_written_this_run = False

    def on_test_end(self, trainer: Trainer, pl_module):
        """テスト完了時に呼び出され、全ての情報をまとめて一度だけ書き込む"""
        if self.has_written_this_run:
            return  # 2回目以降の呼び出しは無視 (重複書き込み防止)

        # test_onlyモードの場合、訓練情報がないので、ここでhparamsをキャプチャする
        if not self.results_cache:
            self._capture_hparams(pl_module)
            self.results_cache["エポック数"] = trainer.max_epochs # 設定ファイルから取得

        metrics = trainer.callback_metrics

        # テスト結果を追加
        test_metric_key = f"final/test/{pl_module.hparams.task.get('metric', 'accuracy')}"
        self.results_cache["テスト精度 (Test Acc)"] = self._get_metric(metrics, test_metric_key)
        self.results_cache["平均レイテンシ (ms/バッチ)"] = self._get_metric(metrics, "inference/avg_latency_ms")
        self.results_cache["p95 レイテンシ (ms/バッチ)"] = self._get_metric(metrics, "inference/p95_latency_ms")
        self.results_cache["推論時 Memoey Allocated [MB]"] = self._get_metric(metrics, "inference/avg_peak_mb")

        # その他情報を取得
        self.results_cache["ステータス"] = "完了"
        if isinstance(trainer.logger, WandbLogger) and trainer.logger.experiment:
            self.results_cache["wandbリンク"] = trainer.logger.experiment.url
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                # test_only時はbest_model_pathがないので、ロードしたckptパスを使う
                ckpt_path = trainer.ckpt_path or "N/A"
                self.results_cache["チェックポイントパス"] = cb.best_model_path or ckpt_path
        
        # CSVファイルに書き出す
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        file_exists = os.path.isfile(self.output_file)
        with open(self.output_file, "a", newline="", encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if not file_exists:
                writer.writeheader()
            
            row_data = {h: self.results_cache.get(h, "") for h in self.headers}
            writer.writerow(row_data)
        
        self.has_written_this_run = True # 書き込み完了フラグを立てる
        self.results_cache = {} # 次の実験のためにキャッシュをクリア
        print(f"\n[bold magenta]📊 全ての実験結果を {self.output_file} に記録しました。[/bold magenta]")