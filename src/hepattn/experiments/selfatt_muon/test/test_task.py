"""
Test script for SelfAttentionCorrelationTask.

This script tests:
1. Task initialization
2. Forward pass (computing self-similarity matrix)
3. Prediction (thresholding)
4. Loss computation (BCE)
"""

import torch
import torch.nn as nn

# Add parent to path for imports
import sys
sys.path.insert(0, '/shared/tracking/hepattn_muon/src')

from hepattn.models.task import SelfAttentionCorrelationTask


def test_task_initialization():
    """Test that the task initializes correctly."""
    print("=" * 60)
    print("TEST: Task Initialization")
    print("=" * 60)
    
    task = SelfAttentionCorrelationTask(
        name="hit_correlation",
        input_object="hit",
        target_field="particle_hit_corr",
        dim=64,
        threshold=0.5,
        loss_fn="bce",
        has_intermediate_loss=False,
    )
    
    print(f"Task name: {task.name}")
    print(f"Input object: {task.input_object}")
    print(f"Target field: {task.target_field}")
    print(f"Dimension: {task.dim}")
    print(f"Threshold: {task.threshold}")
    print(f"Loss function: {task.loss_fn}")
    print(f"Projection layer: {task.proj}")
    print("✓ Task initialization successful\n")
    
    return task


def test_forward_pass(task):
    """Test the forward pass of the task."""
    print("=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    num_hits = 20
    dim = 64
    
    # Create mock embeddings
    x = {
        "hit_embed": torch.randn(batch_size, num_hits, dim),
    }
    
    print(f"Input shape: hit_embed = {x['hit_embed'].shape}")
    
    # Run forward pass
    outputs = task(x)
    
    print(f"\nOutputs keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
        print(f"    min = {value.min().item():.4f}, max = {value.max().item():.4f}, mean = {value.mean().item():.4f}")
    
    # Check expected output shape
    expected_key = "hit_particle_hit_corr_logit"
    assert expected_key in outputs, f"Expected key '{expected_key}' not in outputs"
    assert outputs[expected_key].shape == (batch_size, num_hits, num_hits), \
        f"Expected shape {(batch_size, num_hits, num_hits)}, got {outputs[expected_key].shape}"
    
    print("✓ Forward pass successful\n")
    
    return outputs


def test_predict(task, outputs):
    """Test the prediction method."""
    print("=" * 60)
    print("TEST: Predict")
    print("=" * 60)
    
    preds = task.predict(outputs)
    
    print(f"Prediction keys: {list(preds.keys())}")
    for key, value in preds.items():
        print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
        if value.dtype == torch.bool:
            print(f"    num_true = {value.sum().item()}, num_false = {(~value).sum().item()}")
        else:
            print(f"    min = {value.min().item():.4f}, max = {value.max().item():.4f}")
    
    print("✓ Predict successful\n")
    
    return preds


def test_loss(task, outputs):
    """Test the loss computation."""
    print("=" * 60)
    print("TEST: Loss Computation")
    print("=" * 60)
    
    batch_size = outputs["hit_particle_hit_corr_logit"].shape[0]
    num_hits = outputs["hit_particle_hit_corr_logit"].shape[1]
    
    # Create mock targets
    # Simulate 2 particles per event
    targets = {
        "hit_particle_hit_corr": torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool),
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    }
    
    # Create some structure in targets (particle 0: hits 0-9, particle 1: hits 10-14)
    for b in range(batch_size):
        # Particle 0's innermost hit at index 0, correlates with hits 0-9
        targets["hit_particle_hit_corr"][b, 0, 0:10] = True
        # Particle 1's innermost hit at index 10, correlates with hits 10-14
        targets["hit_particle_hit_corr"][b, 10, 10:15] = True
        # Mark some hits as padding
        targets["hit_valid"][b, 15:] = False
    
    print(f"Target shapes:")
    print(f"  hit_particle_hit_corr: {targets['hit_particle_hit_corr'].shape}")
    print(f"  hit_valid: {targets['hit_valid'].shape}")
    print(f"  num_true in target: {targets['hit_particle_hit_corr'].sum().item()}")
    print(f"  num_valid hits: {targets['hit_valid'].sum().item()}")
    
    # Compute loss
    try:
        losses = task.loss(outputs, targets)
        print(f"\nLoss outputs:")
        for key, value in losses.items():
            print(f"  {key}: {value.item():.6f}")
        print("✓ Loss computation successful\n")
    except Exception as e:
        print(f"✗ Loss computation FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        raise
    
    return losses


def test_gradient_flow(task):
    """Test that gradients flow correctly through the task."""
    print("=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)
    
    batch_size = 2
    num_hits = 10
    dim = 64
    
    # Create mock embeddings with gradients
    x = {
        "hit_embed": torch.randn(batch_size, num_hits, dim, requires_grad=True),
    }
    
    # Create mock targets
    targets = {
        "hit_particle_hit_corr": torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool),
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    }
    targets["hit_particle_hit_corr"][:, 0, 0:5] = True
    
    # Forward pass
    outputs = task(x)
    
    # Compute loss
    losses = task.loss(outputs, targets)
    total_loss = sum(losses.values())
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    print(f"Input gradient shape: {x['hit_embed'].grad.shape}")
    print(f"Input gradient norm: {x['hit_embed'].grad.norm().item():.6f}")
    print(f"Projection weight gradient norm: {task.proj.weight.grad.norm().item():.6f}")
    
    assert x['hit_embed'].grad is not None, "No gradient on input"
    assert task.proj.weight.grad is not None, "No gradient on projection weights"
    
    print("✓ Gradient flow successful\n")


def main():
    print("\n" + "=" * 60)
    print("SELF-ATTENTION CORRELATION TASK TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    task = test_task_initialization()
    outputs = test_forward_pass(task)
    preds = test_predict(task, outputs)
    losses = test_loss(task, outputs)
    test_gradient_flow(task)
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
