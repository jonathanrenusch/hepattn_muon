from hepattn.models.transformer_regressor import TransformerTrackRegressor

# 2-layer Transformer
model = TransformerTrackRegressor(dim=128, num_layers=2, num_heads=4, input_dim=27, head_hidden_dim=256, use_multi_cls=False)
print(f"Transformer 2-layer single-CLS: {sum(p.numel() for p in model.parameters()):,}")

model = TransformerTrackRegressor(dim=128, num_layers=2, num_heads=4, input_dim=27, head_hidden_dim=256, use_multi_cls=True)
print(f"Transformer 2-layer multi-CLS: {sum(p.numel() for p in model.parameters()):,}")
