#!/usr/bin/env python3
"""
RNNModel の動作テスト
"""

import torch
import sys
import os

# パスを追加
sys.path.append('/work')

from src.models import RNNModel

def test_rnn_model():
    print("🧪 RNNModel テスト開始")
    print("=" * 50)
    
    # モデルの設定
    config = {
        'd_input': 784,  # MNIST画像を平坦化したサイズ
        'd_model': 256,
        'cell': {
            'hidden_activation': 'tanh',
            'orthogonal': False,
        },
        'output_mode': 'state'
    }
    
    # モデルの作成
    model = RNNModel(**config)
    print(f"✅ モデル作成成功")
    print(f"   - 入力次元: {model.d_input}")
    print(f"   - 隠れ次元: {model.d_model}")
    print(f"   - 出力モード: {model.output_mode}")
    
    # モデル情報の表示
    info = model.get_info()
    print(f"\n📊 モデル情報:")
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    # テストデータの作成
    batch_size = 32
    seq_len = 28  # MNIST画像の高さ（各行を1時刻として処理）
    d_input = 28  # MNIST画像の幅
    
    x = torch.randn(batch_size, seq_len, d_input)
    print(f"\n🔢 テストデータ:")
    print(f"   - 形状: {x.shape}")
    print(f"   - バッチサイズ: {batch_size}")
    print(f"   - シーケンス長: {seq_len}")
    print(f"   - 入力次元: {d_input}")
    
    # 順方向計算
    print(f"\n🚀 順方向計算テスト...")
    
    # デフォルト状態の作成
    initial_state = model.default_state(batch_size)
    print(f"   - 初期状態形状: {initial_state.shape}")
    
    # モデルの実行
    output, final_state = model(x, state=initial_state)
    print(f"   - 出力形状: {output.shape}")
    print(f"   - 最終状態形状: {final_state.shape}")
    
    # ステップ実行のテスト
    print(f"\n⚡ ステップ実行テスト...")
    x_t = torch.randn(batch_size, d_input)
    state_t = model.default_state(batch_size)
    new_state = model.step(x_t, state_t)
    print(f"   - 単一ステップ入力形状: {x_t.shape}")
    print(f"   - 単一ステップ出力形状: {new_state.shape}")
    
    # メトリクスの確認
    print(f"\n📈 メトリクス:")
    for key, value in model.metrics.items():
        print(f"   - {key}: {value:.4f}")
    
    print(f"\n✅ 全テスト完了！")

if __name__ == "__main__":
    test_rnn_model()
