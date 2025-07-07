"""
Neural Network utilities for models
元のリポジトリのutils機能を参考に実装
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PassthroughSequential(nn.Sequential):
    """
    通常のSequentialとは異なり、引数をそのまま通すSequential
    元のリポジトリで使用されている構造
    """
    
    def forward(self, *args, **kwargs):
        for module in self:
            if len(args) == 1 and not kwargs:
                # 単一引数の場合
                args = (module(args[0]),)
            else:
                # 複数引数またはキーワード引数がある場合
                output = module(*args, **kwargs)
                if isinstance(output, tuple):
                    args = output
                    kwargs = {}
                else:
                    args = (output,)
                    kwargs = {}
        
        return args[0] if len(args) == 1 else args


def get_activation(activation: str) -> nn.Module:
    """活性化関数を文字列から取得"""
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),  # SiLU is the same as Swish
        'identity': nn.Identity(),
    }
    
    if activation.lower() not in activations:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return activations[activation.lower()]


def init_weights(module: nn.Module, method: str = 'xavier_uniform'):
    """重みの初期化"""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight)
        elif method == 'orthogonal':
            nn.init.orthogonal_(module.weight)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """モデルの詳細情報を取得"""
    total_params = count_parameters(model)
    
    info = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'total_parameters_readable': f"{total_params:,}",
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 4 bytes per parameter (float32)
    }
    
    # レイヤー別パラメータ数
    layer_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 最下層のモジュールのみ
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_params[name] = params
    
    info['layer_parameters'] = layer_params
    
    return info
