"""
Base task class for machine learning tasks
元のリポジトリのタスク構造を参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseTask(pl.LightningModule, ABC):
    """
    Base class for all ML tasks
    PyTorch Lightning + Hydra構成をサポート
    """
    
    def __init__(
        self,
        model: dict,
        optimizer: dict = None,
        scheduler: dict = None,
        loss_fn: Union[str, dict] = "cross_entropy",
        metrics: list = None,
        **kwargs
    ):
        super().__init__()
        
        # Model will be instantiated by calling code
        self._model_config = model
        self.model = None
        
        # Optimizer and scheduler configs
        self.optimizer_config = optimizer or {"_target_": "torch.optim.AdamW", "lr": 1e-3}
        self.scheduler_config = scheduler
        
        # Loss function
        self.loss_fn = self._create_loss_fn(loss_fn)
        
        # Metrics
        self.metrics = metrics or []
        self.metric_fns = {}
        self._setup_metrics()
        
        # For tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def set_model(self, model):
        """Set the model after instantiation"""
        self.model = model
    
    def _create_loss_fn(self, loss_config):
        """Create loss function from config"""
        if isinstance(loss_config, str):
            if loss_config == "cross_entropy":
                return nn.CrossEntropyLoss()
            elif loss_config == "mse":
                return nn.MSELoss()
            elif loss_config == "bce":
                return nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unknown loss function: {loss_config}")
        elif isinstance(loss_config, dict):
            # TODO: Implement dynamic loss creation from config
            loss_type = loss_config.get("_target_", "cross_entropy")
            if "cross_entropy" in loss_type.lower():
                return nn.CrossEntropyLoss()
            else:
                return nn.CrossEntropyLoss()
        else:
            return loss_config
    
    def _setup_metrics(self):
        """Setup metric functions"""
        for metric in self.metrics:
            if metric == "accuracy":
                self.metric_fns[metric] = self._accuracy
            elif metric == "top5_accuracy":
                self.metric_fns[metric] = self._top5_accuracy
            # Add more metrics as needed
    
    @abstractmethod
    def forward(self, batch):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def compute_loss(self, batch, outputs):
        """Compute loss - must be implemented by subclasses"""
        pass
    
    def compute_metrics(self, batch, outputs, prefix=""):
        """Compute metrics for the batch"""
        metrics = {}
        
        for metric_name, metric_fn in self.metric_fns.items():
            try:
                value = metric_fn(batch, outputs)
                metrics[f"{prefix}{metric_name}"] = value
            except Exception as e:
                # Skip metric if computation fails
                pass
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self.forward(batch)
        loss = self.compute_loss(batch, outputs)
        
        # Compute metrics
        metrics = self.compute_metrics(batch, outputs, prefix="train_")
        
        # Log loss and metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self.forward(batch)
        loss = self.compute_loss(batch, outputs)
        
        # Compute metrics
        metrics = self.compute_metrics(batch, outputs, prefix="val_")
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        outputs = self.forward(batch)
        loss = self.compute_loss(batch, outputs)
        
        # Compute metrics
        metrics = self.compute_metrics(batch, outputs, prefix="test_")
        
        # Log loss and metrics
        self.log("test_loss", loss, on_epoch=True)
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Import here to avoid circular imports
        from ..utils.config import instantiate
        
        # Instantiate optimizer
        optimizer = instantiate(self.optimizer_config, params=self.parameters())
        
        if self.scheduler_config is None:
            return optimizer
        
        # Instantiate scheduler
        scheduler = instantiate(self.scheduler_config, optimizer=optimizer)
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",  # or "step"
            "frequency": 1,
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
    
    def _accuracy(self, batch, outputs):
        """Compute accuracy metric"""
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Get targets
        if hasattr(batch, 'targets'):
            targets = batch.targets
        elif isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            targets = batch[1]  # Assume (inputs, targets) format
        else:
            return 0.0
        
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == targets).float()
        return correct.mean()
    
    def _top5_accuracy(self, batch, outputs):
        """Compute top-5 accuracy metric"""
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Get targets
        if hasattr(batch, 'targets'):
            targets = batch.targets
        elif isinstance(batch, dict) and 'targets' in batch:
            targets = batch['targets']
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            targets = batch[1]
        else:
            return 0.0
        
        # Get top-5 predictions
        _, top5_preds = torch.topk(logits, k=5, dim=-1)
        targets_expanded = targets.unsqueeze(1).expand_as(top5_preds)
        correct = (top5_preds == targets_expanded).any(dim=1).float()
        return correct.mean()
    
    def get_model_info(self):
        """Get model information"""
        if self.model is not None and hasattr(self.model, 'get_info'):
            return self.model.get_info()
        else:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {"parameters": total_params}
