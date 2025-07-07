"""
RNNベースのシーケンスモデル実装

設定からの主要パラメータ:
- d_model: 隠れ層の次元数 (256)
- hidden_activation: 活性化関数 (tanh)
- orthogonal: 直交初期化を使用するか (False)
- output_mode: 出力モード (state)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any


class RNNCell(nn.Module):
    """
    カスタムRNNセル実装
    PyTorchの標準RNNよりも柔軟な設定が可能
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        orthogonal: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # 重み行列の定義
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(hidden_size))
        
        # 活性化関数の設定
        if activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "relu":
            self.activation_fn = torch.relu
        elif activation == "sigmoid":
            self.activation_fn = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 重みの初期化
        self._initialize_weights(orthogonal)
    
    def _initialize_weights(self, orthogonal: bool):
        """重みの初期化"""
        if orthogonal:
            # 直交初期化（勾配爆発/消失を防ぐ）
            nn.init.orthogonal_(self.weight_ih)
            nn.init.orthogonal_(self.weight_hh)
        else:
            # Xavier初期化
            nn.init.xavier_uniform_(self.weight_ih)
            nn.init.xavier_uniform_(self.weight_hh)
        
        # バイアスをゼロで初期化
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
    
    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        RNNセルの順方向計算
        
        Args:
            input_tensor: (batch_size, input_size)
            hidden: (batch_size, hidden_size)
            
        Returns:
            new_hidden: (batch_size, hidden_size)
        """
        # RNNの標準的な計算: h_new = activation(W_ih @ x + W_hh @ h + b)
        ih = torch.mm(input_tensor, self.weight_ih.t()) + self.bias_ih
        hh = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        new_hidden = self.activation_fn(ih + hh)
        
        return new_hidden


class RNNModel(nn.Module):
    """
    RNNベースのシーケンスモデル
    
    設定可能なパラメータ:
    - d_model: 隠れ層の次元数
    - hidden_activation: 活性化関数
    - orthogonal: 直交初期化の使用
    - output_mode: 出力モード ("state", "sequence", "last")
    """
    
    def __init__(
        self,
        d_input: int = 1,  # デフォルト値、実際は動的に設定
        d_model: int = 256,
        cell: Dict[str, Any] = None,
        output_mode: str = "state",
        **kwargs
    ):
        super().__init__()
        
        # セル設定のパース
        if cell is None:
            cell = {}
        
        self.d_input = d_input
        self.d_model = d_model
        self.output_mode = output_mode
        
        # RNNセルの初期化
        self.rnn_cell = RNNCell(
            input_size=d_input,
            hidden_size=d_model,
            activation=cell.get("hidden_activation", "tanh"),
            orthogonal=cell.get("orthogonal", False),
        )
        
        # 出力層（必要に応じて）
        if output_mode in ["last", "sequence"]:
            self.output_projection = nn.Linear(d_model, d_model)
        
        # メトリクス記録用
        self.metrics = {}
    
    def default_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        デフォルトの初期状態を作成
        
        Args:
            batch_size: バッチサイズ
            device: テンソルのデバイス
            
        Returns:
            初期隠れ状態: (batch_size, d_model)
        """
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros(batch_size, self.d_model, device=device)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順方向計算
        
        Args:
            x: 入力シーケンス (batch_size, seq_len, d_input)
            state: 初期隠れ状態 (batch_size, d_model)
            
        Returns:
            output: 出力 (形状はoutput_modeによる)
            final_state: 最終隠れ状態 (batch_size, d_model)
        """
        batch_size, seq_len, d_input = x.shape
        
        # 入力次元の動的更新
        if hasattr(self.rnn_cell, 'input_size') and self.rnn_cell.input_size != d_input:
            self._update_input_size(d_input)
        
        # 初期状態の設定
        if state is None:
            state = self.default_state(batch_size, x.device)
        
        # シーケンス処理
        outputs = []
        current_state = state
        
        for t in range(seq_len):
            # 時刻tの入力: (batch_size, d_input)
            x_t = x[:, t, :]
            
            # RNNセルによる計算
            current_state = self.rnn_cell(x_t, current_state)
            outputs.append(current_state)
        
        # 出力の整形
        if self.output_mode == "state":
            # 状態をそのまま返す
            output = current_state
        elif self.output_mode == "last":
            # 最後の出力のみ
            output = self.output_projection(current_state)
        elif self.output_mode == "sequence":
            # 全シーケンス出力
            sequence_outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
            output = self.output_projection(sequence_outputs)
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        
        # メトリクス更新
        self._update_metrics(current_state)
        
        return output, current_state
    
    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        単一ステップの推論（リアルタイム処理用）
        
        Args:
            x_t: 時刻tの入力 (batch_size, d_input)
            state: 現在の隠れ状態 (batch_size, d_model)
            
        Returns:
            新しい隠れ状態 (batch_size, d_model)
        """
        return self.rnn_cell(x_t, state)
    
    def _update_input_size(self, new_input_size: int):
        """入力サイズの動的更新（必要に応じて）"""
        if new_input_size != self.d_input:
            print(f"Updating input size from {self.d_input} to {new_input_size}")
            self.d_input = new_input_size
            
            # 新しいRNNセルを作成
            old_cell = self.rnn_cell
            self.rnn_cell = RNNCell(
                input_size=new_input_size,
                hidden_size=self.d_model,
                activation=old_cell.activation,
                orthogonal=False,  # 再初期化時は通常の初期化を使用
            )
    
    def _update_metrics(self, hidden_state: torch.Tensor):
        """隠れ状態のメトリクスを更新"""
        with torch.no_grad():
            # 隠れ状態のノルム
            hidden_norm = torch.norm(hidden_state, dim=-1).mean()
            
            # 隠れ状態の分散
            hidden_var = torch.var(hidden_state, dim=-1).mean()
            
            self.metrics.update({
                "hidden_norm": hidden_norm.item(),
                "hidden_variance": hidden_var.item(),
            })
    
    def get_info(self) -> Dict[str, Any]:
        """モデルの情報を返す"""
        return {
            "model_type": "RNN",
            "d_input": self.d_input,
            "d_model": self.d_model,
            "activation": self.rnn_cell.activation,
            "orthogonal": "orthogonal" in str(self.rnn_cell.weight_ih),
            "output_mode": self.output_mode,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
