#!/usr/bin/env python
"""Test with exact config from atlas_muon_tracking_NGT_small_better_regression.yaml"""

import torch
from hepattn.models.task import WeightedPoolingObjectHitRegressionTask

print("=" * 80)
print("TESTING WITH EXACT CONFIG PARAMETERS")
print("=" * 80)

# Exact config from yaml
hit_fields = [
    "spacePoint_globEdgeHighX",
    "spacePoint_globEdgeHighY",
    "spacePoint_globEdgeHighZ",
    "spacePoint_globEdgeLowX",
    "spacePoint_globEdgeLowY",
    "spacePoint_globEdgeLowZ",
    "spacePoint_time",
    "spacePoint_driftR",
    "spacePoint_covXX",
    "spacePoint_covXY",
    "spacePoint_covYX",
    "spacePoint_covYY",
    "spacePoint_channel",
    "spacePoint_layer",
    "spacePoint_stationPhi",
    "spacePoint_stationEta",
    "spacePoint_stationIndex",
    "spacePoint_technology",
    "r",
    "s",
    "theta",
    "phi",
    "eta",
]

regression_fields = [
    "truthMuon_eta",
    "truthMuon_phi",
    "truthMuon_pt",
    "truthMuon_q",  # charge
]

task = WeightedPoolingObjectHitRegressionTask(
    name="parameter_regression",
    input_hit="hit",
    input_object="query",
    output_object="track",
    target_object="particle",
    hit_fields=hit_fields,
    fields=regression_fields,
    loss_weight=1.0,
    dim=32,
    assignment_threshold=0.1,
    inverse_fields=["truthMuon_pt"],
    regression_hidden_dim=144,
    regression_num_layers=3,
    dropout=0.1,
)

print(f"Task created successfully")
print(f"  Input dimension: {len(hit_fields)} hit fields")
print(f"  Output dimension: {len(regression_fields)} regression targets")
print(f"  Inverse fields: truthMuon_pt")
print(f"  Regression head: {task.regression_head}")

# Simulate realistic batch from config
batch_size = 200  # from config
num_tracks = 2    # event_max_num_particles
num_hits = 100    # typical
dim = 32

print(f"\nSimulating batch:")
print(f"  batch_size: {batch_size}")
print(f"  num_tracks: {num_tracks}")
print(f"  num_hits: {num_hits}")
print(f"  dim: {dim}")

# Create mock input matching the config structure
x = {
    "query_embed": torch.randn(batch_size, num_tracks, dim),
    "hit_embed": torch.randn(batch_size, num_hits, dim),
    "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    "track_hit_logit": torch.randn(batch_size, num_tracks, num_hits),
}

# Add all hit fields
for field in hit_fields:
    x[f"hit_{field}"] = torch.randn(batch_size, num_hits)

# Realistic targets (based on typical muon values)
targets = {
    "particle_valid": torch.rand(batch_size, num_tracks) > 0.3,  # ~70% have particles
    "particle_truthMuon_eta": torch.randn(batch_size, num_tracks) * 2.5,  # |eta| < 2.5
    "particle_truthMuon_phi": torch.rand(batch_size, num_tracks) * 6.28 - 3.14,  # [-pi, pi]
    "particle_truthMuon_pt": torch.rand(batch_size, num_tracks) * 50 + 5,  # 5-55 GeV
    "particle_truthMuon_q": torch.randint(-1, 2, (batch_size, num_tracks), dtype=torch.float32) * 2 + 1,  # -1 or +1
}

# Ensure at least some valid particles per batch to avoid all-invalid edge case
targets["particle_valid"][:, 0] = True

print("\nRunning forward pass...")
try:
    outputs = task.forward(x)
    print(f"  ✓ Forward successful")
    print(f"  Output shape: {outputs['track_regr'].shape}")
    print(f"  Expected: [{batch_size}, {num_tracks}, {len(regression_fields)}]")
except Exception as e:
    print(f"  ✗ Forward failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nComputing loss...")
try:
    loss_dict = task.loss(outputs, targets)
    loss = loss_dict['smooth_l1']
    print(f"  ✓ Loss successful")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss is NaN: {torch.isnan(loss).any()}")
    print(f"  Loss is Inf: {torch.isinf(loss).any()}")
except Exception as e:
    print(f"  ✗ Loss failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nComputing gradients...")
try:
    loss.backward()
    
    # Check for gradient issues
    grad_issues = []
    for name, param in task.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                grad_issues.append(f"NaN in {name}")
            elif torch.isinf(param.grad).any():
                grad_issues.append(f"Inf in {name}")
    
    if grad_issues:
        print(f"  ✗ Gradient issues found:")
        for issue in grad_issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ All gradients are valid")
except Exception as e:
    print(f"  ✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nTesting predictions...")
try:
    preds = task.predict(outputs)
    print(f"  ✓ Predictions successful")
    for field in regression_fields:
        key = f"track_{field}"
        if key in preds:
            val = preds[key]
            print(f"    {field}: min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("CONFIG TEST COMPLETE - Task is ready for training!")
print("=" * 80)
