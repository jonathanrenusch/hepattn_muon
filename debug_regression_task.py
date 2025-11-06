#!/usr/bin/env python
"""Debug script to identify issues with WeightedPoolingObjectHitRegressionTask."""

import torch
import torch.nn as nn
from hepattn.models.task import WeightedPoolingObjectHitRegressionTask

# Create a mock task
task = WeightedPoolingObjectHitRegressionTask(
    name="test_regr",
    input_hit="hit",
    input_object="query",
    output_object="track",
    target_object="particle",
    hit_fields=["x", "y", "z"],
    fields=["pt", "eta", "phi"],
    loss_weight=1.0,
    dim=32,
    assignment_threshold=0.1,
    inverse_fields=["pt"],
    regression_hidden_dim=144,
    regression_num_layers=3,
    dropout=0.1,
)

# Create mock inputs
batch_size = 2
num_tracks = 2
num_hits = 10
dim = 32

# Mock data
x = {
    "query_embed": torch.randn(batch_size, num_tracks, dim),
    "hit_embed": torch.randn(batch_size, num_hits, dim),
    "hit_x": torch.randn(batch_size, num_hits),
    "hit_y": torch.randn(batch_size, num_hits),
    "hit_z": torch.randn(batch_size, num_hits),
    "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    "track_hit_logit": torch.randn(batch_size, num_tracks, num_hits),
}

targets = {
    "particle_valid": torch.tensor([[True, False], [True, True]], dtype=torch.bool),
    "particle_pt": torch.tensor([[10.0, 5.0], [20.0, 15.0]]),
    "particle_eta": torch.tensor([[0.5, 1.0], [0.3, 0.8]]),
    "particle_phi": torch.tensor([[1.0, 2.0], [1.5, 2.5]]),
}

print("=" * 80)
print("TESTING WeightedPoolingObjectHitRegressionTask")
print("=" * 80)

# Test forward pass
print("\n1. Testing forward pass...")
try:
    outputs = task.forward(x)
    print(f"   ✓ Forward pass successful")
    print(f"   Output shape: {outputs['track_regr'].shape}")
    print(f"   Output stats: min={outputs['track_regr'].min():.4f}, max={outputs['track_regr'].max():.4f}")
    print(f"   Has NaN: {torch.isnan(outputs['track_regr']).any()}")
    print(f"   Has Inf: {torch.isinf(outputs['track_regr']).any()}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test loss computation
print("\n2. Testing loss computation...")
try:
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    print(f"   ✓ Loss computation successful")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss is NaN: {torch.isnan(loss).any()}")
    print(f"   Loss is Inf: {torch.isinf(loss).any()}")
    print(f"   Loss is zero: {loss.item() == 0}")
except Exception as e:
    print(f"   ✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test gradient computation
print("\n3. Testing gradient computation...")
try:
    # Zero gradients
    task.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_stats = []
    
    for name, param in task.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_stats.append({
                'name': name,
                'shape': param.shape,
                'grad_norm': grad_norm,
                'grad_min': param.grad.min().item(),
                'grad_max': param.grad.max().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'is_zero': (grad_norm == 0),
            })
    
    if not has_grad:
        print(f"   ✗ No gradients computed!")
    else:
        print(f"   ✓ Gradients computed")
        
        # Check for issues
        zero_grads = [s for s in grad_stats if s['is_zero']]
        nan_grads = [s for s in grad_stats if s['has_nan']]
        
        if zero_grads:
            print(f"   ⚠ Warning: {len(zero_grads)} parameters have zero gradients:")
            for s in zero_grads[:5]:  # Show first 5
                print(f"      - {s['name']}: {s['shape']}")
        
        if nan_grads:
            print(f"   ✗ Error: {len(nan_grads)} parameters have NaN gradients:")
            for s in nan_grads:
                print(f"      - {s['name']}: {s['shape']}")
        
        if not zero_grads and not nan_grads:
            print(f"   ✓ All gradients are valid and non-zero")
            print(f"   Gradient norm stats:")
            norms = [s['grad_norm'] for s in grad_stats]
            print(f"      - Min norm: {min(norms):.6e}")
            print(f"      - Max norm: {max(norms):.6e}")
            print(f"      - Mean norm: {sum(norms)/len(norms):.6e}")

except Exception as e:
    print(f"   ✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with edge case: all tracks invalid
print("\n4. Testing edge case: all tracks invalid...")
targets_invalid = {
    "particle_valid": torch.tensor([[False, False], [False, False]], dtype=torch.bool),
    "particle_pt": torch.tensor([[10.0, 5.0], [20.0, 15.0]]),
    "particle_eta": torch.tensor([[0.5, 1.0], [0.3, 0.8]]),
    "particle_phi": torch.tensor([[1.0, 2.0], [1.5, 2.5]]),
}
try:
    outputs = task.forward(x)
    loss_dict = task.loss(outputs, targets_invalid)
    loss = loss_dict['smooth_l1']
    print(f"   Loss with all invalid: {loss.item():.6f}")
    print(f"   Loss is NaN: {torch.isnan(loss).any()}")
except Exception as e:
    print(f"   ✗ Failed with all invalid targets: {e}")

# Test prediction
print("\n5. Testing prediction...")
try:
    preds = task.predict(outputs)
    print(f"   ✓ Prediction successful")
    for field in task.fields:
        pred_key = f"track_{field}"
        if pred_key in preds:
            pred_val = preds[pred_key]
            print(f"   {field}: shape={pred_val.shape}, min={pred_val.min():.4f}, max={pred_val.max():.4f}")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
