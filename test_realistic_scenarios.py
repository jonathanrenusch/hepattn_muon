#!/usr/bin/env python
"""Test with more realistic scenarios that might occur during training."""

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

print("=" * 80)
print("REALISTIC SCENARIO TESTS")
print("=" * 80)

# Test 1: No hits assigned (all assignment probabilities below threshold)
print("\n1. Testing with no hits assigned (all probs < threshold)...")
batch_size = 2
num_tracks = 2
num_hits = 10
dim = 32

x = {
    "query_embed": torch.randn(batch_size, num_tracks, dim),
    "hit_embed": torch.randn(batch_size, num_hits, dim),
    "hit_x": torch.randn(batch_size, num_hits),
    "hit_y": torch.randn(batch_size, num_hits),
    "hit_z": torch.randn(batch_size, num_hits),
    "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    # Very negative logits -> all probs << threshold
    "track_hit_logit": torch.ones(batch_size, num_tracks, num_hits) * -10,
}

targets = {
    "particle_valid": torch.tensor([[True, True], [True, False]], dtype=torch.bool),
    "particle_pt": torch.tensor([[10.0, 20.0], [5.0, 15.0]]),
    "particle_eta": torch.tensor([[0.5, 0.3], [1.0, 0.8]]),
    "particle_phi": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
}

try:
    outputs = task.forward(x)
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    print(f"   ✓ Loss: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss).any()}")
    print(f"   Has Inf: {torch.isinf(loss).any()}")
    
    # Check gradients
    loss.backward()
    has_nan_grad = any(torch.isnan(p.grad).any() for p in task.parameters() if p.grad is not None)
    print(f"   Has NaN gradients: {has_nan_grad}")
    task.zero_grad()
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Very low pT targets (near zero)
print("\n2. Testing with very low pT targets...")
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
    "particle_valid": torch.tensor([[True, True], [True, False]], dtype=torch.bool),
    "particle_pt": torch.tensor([[0.001, 0.1], [1e-5, 15.0]]),  # Very low pT
    "particle_eta": torch.tensor([[0.5, 0.3], [1.0, 0.8]]),
    "particle_phi": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
}

try:
    outputs = task.forward(x)
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    print(f"   ✓ Loss: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss).any()}")
    print(f"   Has Inf: {torch.isinf(loss).any()}")
    
    loss.backward()
    has_nan_grad = any(torch.isnan(p.grad).any() for p in task.parameters() if p.grad is not None)
    print(f"   Has NaN gradients: {has_nan_grad}")
    task.zero_grad()
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Mixed valid/invalid tracks across batch
print("\n3. Testing mixed valid/invalid across batch...")
targets = {
    "particle_valid": torch.tensor([[False, False], [True, True]], dtype=torch.bool),
    "particle_pt": torch.tensor([[10.0, 20.0], [5.0, 15.0]]),
    "particle_eta": torch.tensor([[0.5, 0.3], [1.0, 0.8]]),
    "particle_phi": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
}

try:
    outputs = task.forward(x)
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    print(f"   ✓ Loss: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss).any()}")
    
    loss.backward()
    has_nan_grad = any(torch.isnan(p.grad).any() for p in task.parameters() if p.grad is not None)
    print(f"   Has NaN gradients: {has_nan_grad}")
    task.zero_grad()
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Training loop simulation
print("\n4. Simulating mini training loop (10 steps)...")
optimizer = torch.optim.Adam(task.parameters(), lr=0.001)
losses = []

for step in range(10):
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
        "particle_valid": torch.rand(batch_size, num_tracks) > 0.5,
        "particle_pt": torch.rand(batch_size, num_tracks) * 50 + 1,  # 1-51 GeV
        "particle_eta": torch.randn(batch_size, num_tracks),
        "particle_phi": torch.rand(batch_size, num_tracks) * 6.28 - 3.14,
    }
    
    optimizer.zero_grad()
    outputs = task.forward(x)
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"   ✗ Step {step}: Loss is NaN/Inf!")
        break
    
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

if len(losses) == 10:
    print(f"   ✓ Completed 10 steps")
    print(f"   Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"   All losses finite: {all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)}")
else:
    print(f"   ✗ Only completed {len(losses)} steps")

print("\n" + "=" * 80)
print("TESTS COMPLETE")
print("=" * 80)
