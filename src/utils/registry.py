optimizer = {
    "adamw": "torch.optim.AdamW",
}

scheduler = {
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
}

model = {
    "sequence": "src.models.sequence.SequenceModel",
}

wirings = {
    "wiring": "src.models.wirings.Wiring",
    "fully_connected": "src.models.wirings.FullyConnected",
    "ncp": "src.models.wirings.NCP",
    "auto_ncp": "src.models.wirings.AutoNCP",
    "random": "src.models.wirings.Random",
}

layer = {
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "rnn_original": "src.models.sequence.rnns.rnn_original.RNN",
    "lstm": "src.models.baseline.lstm.TorchLSTM",
    "ltc_for_ncps": "src.models.ncps.ltc.LTC",
    "cfc": "src.models.ncps.cfc.CfC",
    "wired_cfc": "src.models.ncps.wired_cfc.WiredCfC",
}

cell = {
    "rnn": "src.models.sequence.rnns.cells.rnn.RNNCell",
    "ltc": "src.models.ncps.cells.ltc.LTCCell",
    "ltc_for_ncps": "src.models.ncps.cells.ltc_cell.LTCCell",
    "cfc": "src.models.ncps.cells.cfc_cell.CfCCell",
    "wired_cfc": "src.models.ncps.cells.wired_cfc_cell.WiredCfCCell",
}

task = {
    "multiclass_classification": "src.tasks.classification.MulticlassClassification",
    "classification": "src.tasks.classification.MulticlassClassification",
}

callbacks = {
    "score": "src.callbacks.score.Score",
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
    # 自作コールバック
    "memory_monitor": "src.callbacks.memory_monitor.MemoryMonitor",
    "latency_monitor": "src.callbacks.latency_monitor.LatencyMonitor",
    "experiment_logger": "src.callbacks.experiment_logger.CSVSummaryCallback",
}
