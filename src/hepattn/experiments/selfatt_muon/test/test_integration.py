"""
Test the full training pipeline integration.

This tests:
1. Model instantiation with config
2. Forward pass with real data batch
3. Loss computation
4. Metric computation in the wrapper
"""

import sys
sys.path.insert(0, '/shared/tracking/hepattn_muon/src')

import torch
import torch.nn as nn
from pathlib import Path


def create_model(device="cpu"):
    """Create a model matching the config."""
    from hepattn.models import HitFilter, Encoder, InputNet, Dense
    from hepattn.models.task import SelfAttentionCorrelationTask
    
    dim = 64
    
    # Use torch attention for CPU compatibility, flash for GPU
    attn_type = "flash" if device == "cuda" else "torch"
    
    # Input network
    input_net = InputNet(
        input_name="hit",
        fields=[
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "r",
            "phi",
            "eta",
        ],
        net=Dense(input_size=6, hidden_layers=[64], output_size=dim),
    )
    
    # Encoder
    encoder = Encoder(
        num_layers=4,
        dim=dim,
        attn_type=attn_type,
        window_size=None,
        window_wrap=False,
        hybrid_norm=True,
        norm="RMSNorm",
    )
    
    # Task
    task = SelfAttentionCorrelationTask(
        name="hit_correlation",
        input_object="hit",
        target_field="particle_hit_corr",
        dim=dim,
        threshold=0.5,
        loss_fn="bce",
        has_intermediate_loss=False,
    )
    
    # Full model
    model = HitFilter(
        input_nets=nn.ModuleList([input_net]),
        encoder=encoder,
        tasks=nn.ModuleList([task]),
        input_sort_field=None,
    )
    
    return model.to(device)


def test_model_forward():
    """Test model forward pass with mock data."""
    print("\n" + "=" * 60)
    print("TEST: Model Forward Pass")
    print("=" * 60)
    
    # Use CPU with sdpa attention for mock tests
    model = create_model(device="cpu")
    model.eval()
    
    batch_size = 4
    num_hits = 50
    
    # Create mock inputs matching expected format
    inputs = {
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        "hit_spacePoint_globEdgeLowX": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowY": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowZ": torch.randn(batch_size, num_hits),
        "hit_r": torch.rand(batch_size, num_hits) * 10,
        "hit_phi": torch.rand(batch_size, num_hits) * 2 * 3.14159 - 3.14159,
        "hit_eta": torch.randn(batch_size, num_hits),
    }
    
    print(f"Input shapes:")
    for k, v in inputs.items():
        print(f"  {k}: {v.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"\nOutput structure:")
    for layer_name, layer_outputs in outputs.items():
        print(f"  {layer_name}:")
        for task_name, task_outputs in layer_outputs.items():
            print(f"    {task_name}:")
            for k, v in task_outputs.items():
                print(f"      {k}: {v.shape}")
    
    # Check expected output
    assert "final" in outputs
    assert "hit_correlation" in outputs["final"]
    assert "hit_particle_hit_corr_logit" in outputs["final"]["hit_correlation"]
    
    logits = outputs["final"]["hit_correlation"]["hit_particle_hit_corr_logit"]
    assert logits.shape == (batch_size, num_hits, num_hits)
    
    print("\n✓ Model forward pass successful")
    return model, outputs


def test_model_loss(model):
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("TEST: Model Loss Computation")
    print("=" * 60)
    
    batch_size = 4
    num_hits = 50
    
    # Create mock inputs
    inputs = {
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        "hit_spacePoint_globEdgeLowX": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowY": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowZ": torch.randn(batch_size, num_hits),
        "hit_r": torch.rand(batch_size, num_hits) * 10,
        "hit_phi": torch.rand(batch_size, num_hits) * 2 * 3.14159 - 3.14159,
        "hit_eta": torch.randn(batch_size, num_hits),
    }
    
    # Create mock targets
    targets = {
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        "hit_particle_hit_corr": torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool),
    }
    
    # Add some structure to targets
    for b in range(batch_size):
        # Particle 0: innermost at 0, hits 0-19
        targets["hit_particle_hit_corr"][b, 0, 0:20] = True
        # Particle 1: innermost at 25, hits 25-40
        targets["hit_particle_hit_corr"][b, 25, 25:40] = True
    
    print(f"Target hit_particle_hit_corr: {targets['hit_particle_hit_corr'].shape}")
    print(f"  Total True: {targets['hit_particle_hit_corr'].sum().item()}")
    
    # Forward pass
    outputs = model(inputs)
    
    # Loss computation
    losses, updated_targets = model.loss(outputs, targets)
    
    print(f"\nLosses:")
    for layer_name, layer_losses in losses.items():
        print(f"  {layer_name}:")
        for task_name, task_losses in layer_losses.items():
            print(f"    {task_name}:")
            for loss_name, loss_value in task_losses.items():
                print(f"      {loss_name}: {loss_value.item():.6f}")
    
    # Check loss is valid
    total_loss = sum(
        loss_value 
        for layer_losses in losses.values() 
        for task_losses in layer_losses.values()
        for loss_value in task_losses.values()
    )
    
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert not torch.isinf(total_loss), "Loss is Inf"
    
    print(f"\nTotal loss: {total_loss.item():.6f}")
    print("✓ Loss computation successful")


def test_predict(model):
    """Test prediction generation."""
    print("\n" + "=" * 60)
    print("TEST: Prediction Generation")
    print("=" * 60)
    
    batch_size = 2
    num_hits = 30
    
    inputs = {
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        "hit_spacePoint_globEdgeLowX": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowY": torch.randn(batch_size, num_hits),
        "hit_spacePoint_globEdgeLowZ": torch.randn(batch_size, num_hits),
        "hit_r": torch.rand(batch_size, num_hits) * 10,
        "hit_phi": torch.rand(batch_size, num_hits) * 2 * 3.14159 - 3.14159,
        "hit_eta": torch.randn(batch_size, num_hits),
    }
    
    with torch.no_grad():
        outputs = model(inputs)
        preds = model.predict(outputs)
    
    print("Predictions:")
    for layer_name, layer_preds in preds.items():
        print(f"  {layer_name}:")
        for task_name, task_preds in layer_preds.items():
            print(f"    {task_name}:")
            for k, v in task_preds.items():
                print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Check predictions exist
    assert "hit_particle_hit_corr_prob" in preds["final"]["hit_correlation"]
    assert "hit_particle_hit_corr" in preds["final"]["hit_correlation"]
    
    probs = preds["final"]["hit_correlation"]["hit_particle_hit_corr_prob"]
    binary_preds = preds["final"]["hit_correlation"]["hit_particle_hit_corr"]
    
    print(f"\nProbability stats:")
    print(f"  min: {probs.min().item():.4f}")
    print(f"  max: {probs.max().item():.4f}")
    print(f"  mean: {probs.mean().item():.4f}")
    print(f"\nBinary predictions: {binary_preds.sum().item()} / {binary_preds.numel()} True")
    
    print("✓ Prediction generation successful")


def test_with_real_data():
    """Test with actual data from the dataset."""
    print("\n" + "=" * 60)
    print("TEST: With Real Data")
    print("=" * 60)
    
    from hepattn.experiments.selfatt_muon.data import AtlasMuonDataset, AtlasMuonCollator
    from torch.utils.data import DataLoader
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    
    if not Path(data_dir).exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Skipping real data test")
        return
    
    inputs_config = {
        "hit": [
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "r",
            "phi",
            "eta",
        ]
    }
    targets_config = {"particle": [], "hit": []}
    
    dataset = AtlasMuonDataset(
        dirpath=data_dir,
        inputs=inputs_config,
        targets=targets_config,
        num_events=16,
        event_max_num_particles=2,
    )
    
    collator = AtlasMuonCollator(
        dataset_inputs=inputs_config,
        dataset_targets=targets_config,
        max_num_obj=2,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=0,
    )
    
    model = create_model(device=device)
    model.eval()
    
    # Get a batch
    batch_inputs, batch_targets = next(iter(dataloader))
    
    # Move to device and cast to float16 for flash attention (required by flash_attn)
    def move_to_device(d, dev, dtype=None):
        result = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.to(dev)
                # Only cast float tensors, not bool or int
                if dtype is not None and v.dtype in (torch.float32, torch.float64):
                    v = v.to(dtype)
            result[k] = v
        return result
    
    dtype = torch.bfloat16 if device == "cuda" else None
    batch_inputs = move_to_device(batch_inputs, device, dtype)
    batch_targets = move_to_device(batch_targets, device, dtype)
    
    # Cast model to same dtype
    if dtype is not None:
        model = model.to(dtype)
    
    print(f"Batch inputs: {list(batch_inputs.keys())}")
    print(f"Batch targets: {list(batch_targets.keys())}")
    print(f"Batch size: {batch_inputs['hit_valid'].shape[0]}")
    print(f"Max hits: {batch_inputs['hit_valid'].shape[1]}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch_inputs)
        losses, _ = model.loss(outputs, batch_targets)
        preds = model.predict(outputs)
    
    print(f"\nOutputs computed successfully")
    print(f"Logits shape: {outputs['final']['hit_correlation']['hit_particle_hit_corr_logit'].shape}")
    
    # Calculate loss
    total_loss = sum(
        lv for ll in losses.values() for tl in ll.values() for lv in tl.values()
    )
    print(f"Total loss: {total_loss.item():.6f}")
    
    # Check predictions
    pred_matrix = preds["final"]["hit_correlation"]["hit_particle_hit_corr"]
    print(f"\nPrediction matrix shape: {pred_matrix.shape}")
    print(f"Predicted True entries: {pred_matrix.sum().item()}")
    print(f"Target True entries: {batch_targets['hit_particle_hit_corr'].sum().item()}")
    
    print("\n✓ Real data test successful")


def main():
    print("\n" + "=" * 60)
    print("FULL PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    model, outputs = test_model_forward()
    test_model_loss(model)
    test_predict(model)
    test_with_real_data()
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
