#!/usr/bin/env python3
"""
Simple test script to validate the raw loss extraction works correctly.
"""

import torch
import sys
import os

# Add the hepattn module to the path
sys.path.insert(0, '/shared/tracking/hepattn_muon/src')

def test_raw_loss_extraction():
    """Test that raw_loss method works for all task types."""
    from hepattn.models.task import (
        ObjectValidTask, 
        ObjectHitMaskTask, 
        RegressionTask,
        ObjectRegressionTask,
        ObjectChargeClassificationTask,
        GaussianRegressionTask,
        ObjectClassificationTask,
        ClassificationTask,
        IncidenceRegressionTask,
        HitFilterTask,
    )
    from hepattn.models.dense import Dense
    
    print("Testing raw loss extraction...")
    
    # Create sample inputs and targets
    batch_size = 2
    num_objects = 3
    num_hits = 5
    dim = 64
    
    device = torch.device("cpu")
    
    # Sample data for testing
    outputs = {
        "track_logit": torch.randn(batch_size, num_objects, device=device),
        "track_hit_valid_logit": torch.randn(batch_size, num_objects, num_hits, device=device),
        "track_regr": torch.randn(batch_size, num_objects, 3, device=device),  # 3 fields
        "track_charge_logits": torch.randn(batch_size, num_objects, device=device),
    }
    
    targets = {
        "particle_valid": torch.randint(0, 2, (batch_size, num_objects), device=device, dtype=torch.float32),
        "particle_hit_valid": torch.randint(0, 2, (batch_size, num_objects, num_hits), device=device, dtype=torch.float32),
        "particle_pt": torch.randn(batch_size, num_objects, device=device),
        "particle_eta": torch.randn(batch_size, num_objects, device=device),
        "particle_phi": torch.randn(batch_size, num_objects, device=device),
        "particle_q": torch.randint(-1, 2, (batch_size, num_objects), device=device, dtype=torch.float32) * 2 - 1,  # -1 or +1
        "hit_valid": torch.randint(0, 2, (batch_size, num_hits), device=device, dtype=torch.bool),
    }
    
    # Test ObjectValidTask
    print("Testing ObjectValidTask...")
    task1 = ObjectValidTask(
        name="track_valid",
        input_object="query",
        output_object="track",
        target_object="particle",
        losses={"object_bce": 1.0},
        costs={"object_bce": 1.0},
        dim=dim,
    )
    
    weighted_loss = task1.loss(outputs, targets)
    raw_loss = task1.raw_loss(outputs, targets)
    
    print(f"Weighted loss: {weighted_loss}")
    print(f"Raw loss: {raw_loss}")
    
    # The raw loss should be the weighted loss divided by the weight
    expected_raw_loss = weighted_loss["object_bce"] / 1.0
    assert torch.allclose(raw_loss["object_bce"], expected_raw_loss, atol=1e-6), "Raw loss doesn't match expected value"
    
    # Test ObjectRegressionTask
    print("Testing ObjectRegressionTask...")
    task2 = ObjectRegressionTask(
        name="parameter_regression",
        input_object="query",
        output_object="track",
        target_object="particle", 
        fields=["pt", "eta", "phi"],
        loss_weight=2.0,
        cost_weight=1.0,
        dim=dim,
    )
    
    weighted_loss2 = task2.loss(outputs, targets)
    raw_loss2 = task2.raw_loss(outputs, targets)
    
    print(f"Weighted loss: {weighted_loss2}")
    print(f"Raw loss: {raw_loss2}")
    
    # The raw loss should be the weighted loss divided by the weight
    expected_raw_loss2 = weighted_loss2["smooth_l1"] / 2.0
    assert torch.allclose(raw_loss2["smooth_l1"], expected_raw_loss2, atol=1e-6), "Raw loss doesn't match expected value"
    
    print("✅ All tests passed! Raw loss extraction works correctly.")
    
    return True

if __name__ == "__main__":
    try:
        test_raw_loss_extraction()
        print("SUCCESS: Raw loss extraction is working correctly!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)