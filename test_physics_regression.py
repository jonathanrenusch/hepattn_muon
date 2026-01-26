#!/usr/bin/env python
"""Test physics-informed regression changes."""
import torch
from hepattn.models.mamba_regressor import MambaTrackRegressor
from hepattn.models.task_per_track import MambaRegressionTask, angular_difference
import math

print("=== Testing Physics-Informed Regression ===\n")

# Test model output shape
print("1. Testing model output shape...")
model = MambaTrackRegressor(
    input_dim=18, dim=128, num_layers=1, d_state=16, d_conv=4,
    expand=2, use_mamba2=True, headdim=32, norm='LayerNorm',
    dropout=0.0, head_dropout=0.2, head_hidden_dim=64,
).cuda()

x = torch.randn(8, 32, 18).cuda()
mask = torch.ones(8, 32, dtype=torch.bool).cuda()
outputs = model(x, mask)

print(f"   regression shape: {outputs['regression'].shape} (expected: [8, 4])")
assert outputs['regression'].shape == (8, 4), "Wrong regression output shape!"
print("   ✓ Model output shape correct\n")

# Test prediction recovery
print("2. Testing prediction recovery...")
preds = model.predict(outputs)
print(f"   eta shape: {preds['eta'].shape}")
print(f"   phi shape: {preds['phi'].shape}")
print(f"   pt shape: {preds['pt'].shape}")
print(f"   charge shape: {preds['charge'].shape}")
print(f"   phi range: [{preds['phi'].min():.2f}, {preds['phi'].max():.2f}] (should be ~[-π, π])")
print(f"   pt range: [{preds['pt'].min():.2f}, {preds['pt'].max():.2f}] (should be positive)")
print("   ✓ Prediction recovery works\n")

# Test angular difference
print("3. Testing angular difference (periodicity handling)...")
phi1 = torch.tensor([math.pi - 0.1])
phi2 = torch.tensor([-math.pi + 0.1])
diff = angular_difference(phi1, phi2)
print(f"   phi1 = π - 0.1, phi2 = -π + 0.1")
print(f"   angular diff = {diff.item():.4f} (expected ~0.2)")
assert abs(diff.item() - 0.2) < 0.01, "Angular difference not handling periodicity!"
print("   ✓ Angular difference handles periodicity correctly\n")

# Test loss computation
print("4. Testing loss computation...")
task = MambaRegressionTask(
    regression_weight=1.0,
    classification_weight=1.0,
)

targets = {
    'eta': torch.randn(8).cuda() * 1.5,
    'phi': (torch.rand(8).cuda() * 2 - 1) * math.pi,  # [-π, π]
    'pt': torch.rand(8).cuda() * 150 + 10,  # [10, 160] GeV
    'charge': torch.randint(0, 2, (8,)).cuda().float(),
}

losses = task.loss(outputs, targets)
print(f"   loss: {losses['loss'].item():.4f}")
print(f"   loss_eta: {losses['loss_eta'].item():.4f}")
print(f"   loss_phi: {losses['loss_phi'].item():.4f}")
print(f"   loss_pt: {losses['loss_pt'].item():.4f}")
print(f"   loss_charge: {losses['loss_charge'].item():.4f}")
print("   ✓ Loss computation works\n")

# Test metrics computation
print("5. Testing metrics computation...")
metrics = task.metrics(outputs, targets)
print(f"   mae_eta: {metrics['mae_eta'].item():.4f}")
print(f"   mae_phi: {metrics['mae_phi'].item():.4f} rad")
print(f"   mae_pt: {metrics['mae_pt'].item():.2f} GeV")
print(f"   rel_res_pt: {metrics['rel_res_pt'].item():.4f}")
print(f"   charge_accuracy: {metrics['charge_accuracy'].item():.4f}")
print("   ✓ Metrics computation works\n")

print("=== All tests passed! ===")
