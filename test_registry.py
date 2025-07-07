#!/usr/bin/env python3
"""
レジストリシステムのテスト
"""

import sys
sys.path.append('/work')

from omegaconf import DictConfig
from src.utils import registry, instantiate

def test_registry_system():
    print("🧪 レジストリシステムテスト開始")
    print("=" * 50)
    
    # 1. モデルレジストリのテスト
    print("📦 モデルレジストリテスト")
    model_config = DictConfig({
        '_name_': 'RNNModel',
        'd_input': 784,
        'd_model': 128,
        'cell': {
            'hidden_activation': 'tanh',
            'orthogonal': False,
        },
        'output_mode': 'state'
    })
    
    try:
        model = instantiate(registry.model, model_config)
        print(f"   ✅ モデル作成成功: {type(model).__name__}")
        print(f"   - パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ モデル作成失敗: {e}")
    
    # 2. オプティマイザーレジストリのテスト
    print("\n⚙️ オプティマイザーレジストリテスト")
    optimizer_config = DictConfig({
        '_name_': 'adamw',
        'lr': 1e-3,
        'weight_decay': 1e-4
    })
    
    try:
        # ダミーパラメータでテスト
        import torch
        dummy_params = [torch.randn(10, requires_grad=True)]
        optimizer = instantiate(registry.optimizer, optimizer_config, dummy_params)
        print(f"   ✅ オプティマイザー作成成功: {type(optimizer).__name__}")
        print(f"   - 学習率: {optimizer.param_groups[0]['lr']}")
        print(f"   - 重み減衰: {optimizer.param_groups[0]['weight_decay']}")
    except Exception as e:
        print(f"   ❌ オプティマイザー作成失敗: {e}")
    
    # 3. _target_での直接指定テスト
    print("\n🎯 _target_直接指定テスト")
    target_config = DictConfig({
        '_target_': 'src.models.rnn.RNNModel',
        'd_input': 28,
        'd_model': 64,
        'output_mode': 'state'
    })
    
    try:
        model_direct = instantiate(registry.model, target_config)
        print(f"   ✅ 直接指定成功: {type(model_direct).__name__}")
    except Exception as e:
        print(f"   ❌ 直接指定失敗: {e}")
    
    # 4. レジストリ内容の確認
    print(f"\n📋 レジストリ内容:")
    print(f"   - Models: {list(registry.model.keys())}")
    print(f"   - Optimizers: {list(registry.optimizer.keys())}")
    print(f"   - Schedulers: {list(registry.scheduler.keys())}")
    
    print(f"\n✅ レジストリシステムテスト完了！")

if __name__ == "__main__":
    test_registry_system()
