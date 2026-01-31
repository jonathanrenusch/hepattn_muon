#!/usr/bin/env python
"""Test the updated forward signature."""
import torch
from hepattn.models.mamba_regressor import MambaTrackRegressor

# Test the new forward signature
model = MambaTrackRegressor(
    input_dim=18, dim=128, num_layers=1, d_state=16, d_conv=4,
    expand=2, use_mamba2=True, headdim=32, norm='RMSNorm',
).cuda()

# Create test inputs as dict (matching dataloader output)
inputs = {
    'hit_features': torch.randn(4, 32, 18).cuda(),
    'hit_mask': torch.ones(4, 32, dtype=torch.bool).cuda(),
    'sequence_lengths': torch.tensor([32, 32, 32, 32]).cuda(),
}

# Test forward
outputs = model(inputs)
print('Forward pass works!')
print(f'  regression shape: {outputs["regression"].shape}')
print(f'  charge_logit shape: {outputs["charge_logit"].shape}')

# Test predict
preds = model.predict(outputs)
print('Predict works!')
print(f'  eta: {preds["eta"].shape}, phi: {preds["phi"].shape}, pt: {preds["pt"].shape}')
print()
print('All tests passed!')
