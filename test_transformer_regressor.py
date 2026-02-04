"""Test script for TransformerTrackRegressor instantiation and parameter count."""

import torch
from hepattn.models.transformer_regressor import TransformerTrackRegressor
from hepattn.models.mamba_regressor import MambaTrackRegressor


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_transformer_single_cls():
    """Test Transformer with single CLS token."""
    print("\n=== Transformer Single-CLS ===")
    model = TransformerTrackRegressor(
        input_dim=27,
        dim=128,
        num_layers=3,
        num_heads=4,
        attn_type="torch",
        dense_hidden_scale=2,
        norm="RMSNorm",
        dropout=0.0,
        head_dropout=0.1,
        head_hidden_dim=256,
        use_delta_prediction=True,
        use_delta_phi=False,  # Hybrid mode
        use_multi_cls=False,
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    B, seq_len = 4, 50
    inputs = {
        'hit_features': torch.randn(B, seq_len, 27),
        'hit_mask': torch.ones(B, seq_len, dtype=torch.bool),
    }
    
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Regression output shape: {outputs['regression'].shape}")
    print(f"Charge logit shape: {outputs['charge_logit'].shape}")
    print(f"CLS embedding shape: {outputs['cls_embedding'].shape}")
    print("✓ Forward pass successful")
    
    return model


def test_transformer_multi_cls():
    """Test Transformer with multi-CLS tokens."""
    print("\n=== Transformer Multi-CLS ===")
    model = TransformerTrackRegressor(
        input_dim=27,
        dim=128,
        num_layers=3,
        num_heads=4,
        attn_type="torch",
        dense_hidden_scale=2,
        norm="RMSNorm",
        dropout=0.0,
        head_dropout=0.1,
        head_hidden_dim=256,
        use_delta_prediction=True,
        use_delta_phi=False,  # Hybrid mode
        use_multi_cls=True,
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    B, seq_len = 4, 50
    inputs = {
        'hit_features': torch.randn(B, seq_len, 27),
        'hit_mask': torch.ones(B, seq_len, dtype=torch.bool),
    }
    
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Regression output shape: {outputs['regression'].shape}")
    print(f"Charge logit shape: {outputs['charge_logit'].shape}")
    print(f"CLS embedding shape: {outputs['cls_embedding'].shape}")
    print("✓ Forward pass successful")
    
    return model


def test_mamba_single_cls():
    """Test Mamba with single CLS token for comparison."""
    print("\n=== Mamba Single-CLS (baseline) ===")
    model = MambaTrackRegressor(
        input_dim=27,
        dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba2=True,
        headdim=32,
        norm="RMSNorm",
        dropout=0.0,
        head_dropout=0.1,
        head_hidden_dim=256,
        use_delta_prediction=True,
        use_delta_phi=False,  # Hybrid mode
        use_multi_cls=False,
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    return model


def test_mamba_multi_cls():
    """Test Mamba with multi-CLS token for comparison."""
    print("\n=== Mamba Multi-CLS (baseline) ===")
    model = MambaTrackRegressor(
        input_dim=27,
        dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba2=True,
        headdim=32,
        norm="RMSNorm",
        dropout=0.0,
        head_dropout=0.1,
        head_hidden_dim=256,
        use_delta_prediction=True,
        use_delta_phi=False,  # Hybrid mode
        use_multi_cls=True,
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    return model


def main():
    print("=" * 60)
    print("Testing TransformerTrackRegressor and comparing to Mamba")
    print("=" * 60)
    
    # Test Transformer models
    transformer_single = test_transformer_single_cls()
    transformer_multi = test_transformer_multi_cls()
    
    # Test Mamba models for comparison
    mamba_single = test_mamba_single_cls()
    mamba_multi = test_mamba_multi_cls()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 60)
    
    models = {
        "Transformer Single-CLS": transformer_single,
        "Transformer Multi-CLS": transformer_multi,
        "Mamba Single-CLS": mamba_single,
        "Mamba Multi-CLS": mamba_multi,
    }
    
    for name, model in models.items():
        total, _ = count_parameters(model)
        print(f"{name:30s}: {total:>10,} params")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
