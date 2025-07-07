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
    # Base models
    "RNNModel": "src.models.rnn.RNNModel",
    "LinearModel": "src.models.rnn.LinearModel", 
    "MLPModel": "src.models.rnn.MLPModel",
    "IdentityModel": "src.models.base.IdentityModel",
    # Sequence models
    "SequenceModule": "src.models.base.SequenceModule",
}

layer = {
    "rnn": "src.models.rnn.RNNModel",
    "linear": "src.models.rnn.LinearModel",
    "mlp": "src.models.rnn.MLPModel",
    "activation": "src.models.nn.layers.Activation",
    "normalization": "src.models.nn.layers.Normalization",
    "dropout": "src.models.nn.layers.DropoutModule",
}

task = {
    "multiclass_classification": "src.tasks.classification.MulticlassClassification",
    "classification": "src.tasks.classification.MulticlassClassification",
}

callbacks = {
    # Custom callbacks
    "model_checkpoint": "src.callbacks.model_checkpoint.ModelCheckpoint",
    "early_stopping": "src.callbacks.early_stopping.EarlyStopping", 
    "learning_rate_monitor": "src.callbacks.learning_rate_monitor.LearningRateMonitor",
    # PyTorch Lightning built-in callbacks
    "pl_model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "pl_early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "pl_learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
}

dataset = {
    "mnist": "src.dataloaders.basic.MNIST",
    "cifar10": "src.dataloaders.basic.CIFAR10", 
    "speech_commands": "src.dataloaders.basic.SpeechCommands",
}