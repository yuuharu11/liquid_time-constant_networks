optimizer = {
    "adamw": "torch.optim.AdamW",
    "adam": "torch.optim.Adam",
    "sgd": "torch.optim.SGD",
    "rmsprop": "torch.optim.RMSprop",
}

scheduler = {
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "step": "torch.optim.lr_scheduler.StepLR",
    "exponential": "torch.optim.lr_scheduler.ExponentialLR",
    "reduce_on_plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
}

model = {
    "sequence": "src.models.sequence.SequenceModel",
}

layer = {
    "rnn": "src.models.sequence.rnns.rnn.RNN",
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
}
