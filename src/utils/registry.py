optimizer = {
    "adamw": "torch.optim.AdamW"
}

scheduler = {
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
}

model = {
    # NN系
    "RNNModel": "src.models.rnn.RNNModel",
}

layer = {
    "rnn": "src.models.rnn.RNNModel",
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
}