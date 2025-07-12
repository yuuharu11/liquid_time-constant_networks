#!/usr/bin/env python3
"""
パラメータ数比較スクリプト
"""
import sys
sys.path.append('/work')

import torch
from src.models.sequence.rnns.cells.basic import RNNCell
from src.models.sequence.rnns.cells.ltc import LTCCell

def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models():
    """RNNとLTCのパラメータ数を比較"""
    d_input = 784  # MNIST
    
    print("=== パラメータ数比較 ===")
    print(f"入力次元: {d_input}")
    print()
    
    # 異なるd_modelでテスト
    d_models = [64, 128, 256]
    
    for d_model in d_models:
        print(f"--- d_model = {d_model} ---")
        
        try:
            # RNN
            rnn = RNNCell(d_input=d_input, d_model=d_model)
            rnn_params = count_parameters(rnn)
            print(f"RNN:  {rnn_params:8,} パラメータ")
            
            # LTC
            ltc = LTCCell(d_input=d_input, d_model=d_model)
            ltc_params = count_parameters(ltc)
            ratio = ltc_params / rnn_params if rnn_params > 0 else 0
            print(f"LTC:  {ltc_params:8,} パラメータ (RNNの{ratio:.1f}倍)")
            
        except Exception as e:
            print(f"Error with d_model={d_model}: {e}")
        
        print()

if __name__ == "__main__":
    compare_models()
