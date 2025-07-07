"""
Multi-class classification task
MNIST, CIFAR-10などの分類タスク用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

from .base import BaseTask


class MulticlassClassification(BaseTask):
    """
    Multi-class classification task
    MNISTなどのシーケンス分類に使用
    """
    
    def __init__(
        self,
        model: dict,
        optimizer: dict = None,
        scheduler: dict = None,
        d_output: int = 10,  # Number of classes
        loss_fn: Union[str, dict] = "cross_entropy",
        metrics: list = None,
        pooling: str = "mean",  # For sequence models: "mean", "max", "last"
        **kwargs
    ):
        # Default metrics for classification
        if metrics is None:
            metrics = ["accuracy"]
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            **kwargs
        )
        
        self.d_output = d_output
        self.pooling = pooling
        
        # Classification head will be set after model is instantiated
        self.head = None
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def set_model(self, model):
        """Set the model and create classification head"""
        super().set_model(model)
        
        # Determine model output dimension
        if hasattr(model, 'd_model'):
            d_model = model.d_model
        elif hasattr(model, 'output_dim'):
            d_model = model.output_dim
        else:
            # Try to infer from forward pass
            dummy_input = torch.randn(1, 10, getattr(model, 'd_input', 1))
            with torch.no_grad():
                dummy_output, _ = model(dummy_input)
                if dummy_output.dim() == 3:
                    d_model = dummy_output.shape[-1]
                else:
                    d_model = dummy_output.shape[-1]
        
        # Create classification head
        self.head = nn.Linear(d_model, self.d_output)
    
    def forward(self, batch):
        """
        Forward pass for classification
        
        Args:
            batch: Input batch (inputs, targets) or dict with 'inputs', 'targets'
            
        Returns:
            Dict with 'logits' and other info
        """
        # Extract inputs
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch['inputs']
            targets = batch.get('targets', None)
        else:
            inputs = batch
            targets = None
        
        # Forward through model
        model_output, final_state = self.model(inputs)
        
        # Handle sequence outputs
        if model_output.dim() == 3:  # (batch, seq_len, d_model)
            if self.pooling == "mean":
                # Average pooling over sequence
                pooled_output = model_output.mean(dim=1)
            elif self.pooling == "max":
                # Max pooling over sequence
                pooled_output, _ = model_output.max(dim=1)
            elif self.pooling == "last":
                # Take last timestep
                pooled_output = model_output[:, -1, :]
            else:
                # Default to mean pooling
                pooled_output = model_output.mean(dim=1)
        else:
            pooled_output = model_output
        
        # Classification head
        logits = self.head(pooled_output)
        
        return {
            'logits': logits,
            'model_output': model_output,
            'final_state': final_state,
            'pooled_output': pooled_output,
        }
    
    def compute_loss(self, batch, outputs):
        """Compute classification loss"""
        # Extract targets
        if isinstance(batch, (list, tuple)):
            _, targets = batch
        elif isinstance(batch, dict):
            targets = batch['targets']
        else:
            raise ValueError("Cannot extract targets from batch")
        
        logits = outputs['logits']
        loss = self.loss_fn(logits, targets)
        
        return loss
    
    def predict(self, inputs):
        """Make predictions on inputs"""
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            
            batch = (inputs, None)  # No targets for prediction
            outputs = self.forward(batch)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            return {
                'predictions': preds.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'logits': logits.cpu().numpy(),
            }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for PyTorch Lightning"""
        outputs = self.forward(batch)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': logits,
        }
    
    def configure_metrics(self):
        """Configure additional metrics specific to classification"""
        try:
            import torchmetrics
            
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", 
                num_classes=self.d_output
            )
            self.val_acc = torchmetrics.Accuracy(
                task="multiclass", 
                num_classes=self.d_output
            )
            self.test_acc = torchmetrics.Accuracy(
                task="multiclass", 
                num_classes=self.d_output
            )
            
            if self.d_output >= 5:
                self.train_acc5 = torchmetrics.Accuracy(
                    task="multiclass", 
                    num_classes=self.d_output,
                    top_k=5
                )
                self.val_acc5 = torchmetrics.Accuracy(
                    task="multiclass", 
                    num_classes=self.d_output,
                    top_k=5
                )
                self.test_acc5 = torchmetrics.Accuracy(
                    task="multiclass", 
                    num_classes=self.d_output,
                    top_k=5
                )
        except ImportError:
            # Fall back to custom metrics
            pass
    
    def get_task_info(self):
        """Get task-specific information"""
        info = self.get_model_info()
        info.update({
            "task_type": "multiclass_classification",
            "num_classes": self.d_output,
            "pooling": self.pooling,
            "loss_function": str(self.loss_fn),
            "metrics": self.metrics,
        })
        return info
