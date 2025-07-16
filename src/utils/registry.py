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

layer = {
    "rnn": "src.models.sequence.rnns.rnn.RNN",
    "rnn_original": "src.models.sequence.rnns.rnn_original.RNN",
    "lstm": "src.models.baseline.lstm.TorchLSTM",
}

cell = {
    "ltc": "src.models.sequence.rnns.cells.ltc.LTCCell",
    "rnn": "src.models.sequence.rnns.cells.rnn.RNNCell",
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
}
