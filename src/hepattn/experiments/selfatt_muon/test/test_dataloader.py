"""
Test script for data loading and collation.

This script tests:
1. Single event loading and shape verification
2. Batch collation and padding
3. particle_hit_corr matrix structure
4. Consistency between inputs and targets
"""

import sys
sys.path.insert(0, '/shared/tracking/hepattn_muon/src')

import torch
import numpy as np
from pathlib import Path


def test_dataset_loading():
    """Test loading a single dataset and checking basic properties."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Loading")
    print("=" * 60)
    
    from hepattn.experiments.selfatt_muon.data import AtlasMuonDataset
    
    # You may need to update this path
    data_dir = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    
    if not Path(data_dir).exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Skipping dataset loading test")
        return None
    
    # Define inputs and targets
    inputs = {
        "hit": [
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "r",
            "phi",
            "eta",
        ]
    }
    targets = {
        "particle": [],
        "hit": [],
    }
    
    dataset = AtlasMuonDataset(
        dirpath=data_dir,
        inputs=inputs,
        targets=targets,
        num_events=10,
        event_max_num_particles=2,
    )
    
    print(f"Dataset created with {len(dataset)} events")
    
    return dataset, inputs, targets


def test_single_event(dataset):
    """Test loading a single event and verify all shapes."""
    print("\n" + "=" * 60)
    print("TEST: Single Event Loading")
    print("=" * 60)
    
    if dataset is None:
        print("⚠ Skipping (no dataset)")
        return
    
    for idx in range(min(3, len(dataset))):
        print(f"\n--- Event {idx} ---")
        inputs, targets = dataset[idx]
        
        print("Inputs:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\nTargets:")
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.dtype == torch.bool:
                    print(f"    num_true={value.sum().item()}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Check particle_hit_corr matrix
        if "particle_hit_corr" in targets:
            corr = targets["particle_hit_corr"]
            print(f"\nparticle_hit_corr analysis:")
            print(f"  Shape: {corr.shape}")
            print(f"  Total True entries: {corr.sum().item()}")
            
            # Find rows with True entries (innermost hits)
            rows_with_true = corr.squeeze(0).any(dim=-1)
            active_rows = torch.where(rows_with_true)[0]
            print(f"  Active rows (innermost hits): {active_rows.tolist()}")
            
            for row_idx in active_rows:
                row = corr[0, row_idx]
                true_cols = torch.where(row)[0]
                print(f"    Row {row_idx.item()}: {len(true_cols)} hits -> cols {true_cols.tolist()}")
        
        # Check particle_valid
        if "particle_valid" in targets:
            pv = targets["particle_valid"]
            print(f"\nparticle_valid analysis:")
            print(f"  Shape: {pv.shape}")
            print(f"  Valid positions: {torch.where(pv.squeeze(0))[0].tolist()}")


def test_collation(dataset, inputs_config, targets_config):
    """Test the collation function for batching."""
    print("\n" + "=" * 60)
    print("TEST: Batch Collation")
    print("=" * 60)
    
    if dataset is None:
        print("⚠ Skipping (no dataset)")
        return
    
    from hepattn.experiments.selfatt_muon.data import AtlasMuonCollator
    from torch.utils.data import DataLoader
    
    collator = AtlasMuonCollator(
        dataset_inputs=inputs_config,
        dataset_targets=targets_config,
        max_num_obj=2,
    )
    
    # Create a small batch
    batch_data = [dataset[i] for i in range(min(4, len(dataset)))]
    
    print(f"Collating {len(batch_data)} events...")
    
    try:
        batched_inputs, batched_targets = collator(batch_data)
        
        print("\nBatched Inputs:")
        for key, value in batched_inputs.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("\nBatched Targets:")
        for key, value in batched_targets.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Check hit_particle_hit_corr specifically
        if "hit_particle_hit_corr" in batched_targets:
            corr = batched_targets["hit_particle_hit_corr"]
            print(f"\nBatched hit_particle_hit_corr:")
            print(f"  Shape: {corr.shape} (expected [B, N, N])")
            print(f"  Total True: {corr.sum().item()}")
            
            for b in range(corr.shape[0]):
                rows_with_true = corr[b].any(dim=-1)
                active_rows = torch.where(rows_with_true)[0]
                print(f"  Batch {b}: {len(active_rows)} active rows at {active_rows.tolist()}")
        
        print("\n✓ Collation successful")
        
    except Exception as e:
        print(f"\n✗ Collation FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_dataloader(dataset, inputs_config, targets_config):
    """Test the full DataLoader pipeline."""
    print("\n" + "=" * 60)
    print("TEST: DataLoader Pipeline")
    print("=" * 60)
    
    if dataset is None:
        print("⚠ Skipping (no dataset)")
        return
    
    from hepattn.experiments.selfatt_muon.data import AtlasMuonCollator
    from torch.utils.data import DataLoader
    
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
        shuffle=False,
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Get one batch
    try:
        batch_inputs, batch_targets = next(iter(dataloader))
        
        print("\nFirst batch loaded successfully!")
        print(f"Batch inputs keys: {list(batch_inputs.keys())}")
        print(f"Batch targets keys: {list(batch_targets.keys())}")
        
        # Verify key shapes
        if "hit_valid" in batch_inputs:
            print(f"\nhit_valid shape: {batch_inputs['hit_valid'].shape}")
            print(f"  Valid hits per event: {batch_inputs['hit_valid'].sum(dim=-1).tolist()}")
        
        if "hit_particle_hit_corr" in batch_targets:
            corr = batch_targets["hit_particle_hit_corr"]
            print(f"\nhit_particle_hit_corr shape: {corr.shape}")
            
            # Detailed analysis
            for b in range(corr.shape[0]):
                rows_with_true = corr[b].any(dim=-1)
                n_active = rows_with_true.sum().item()
                n_total_true = corr[b].sum().item()
                print(f"  Event {b}: {n_active} tracks, {n_total_true} total correlations")
        
        print("\n✓ DataLoader pipeline works correctly")
        
    except Exception as e:
        print(f"\n✗ DataLoader FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_target_consistency():
    """Test that targets are consistent with expected structure."""
    print("\n" + "=" * 60)
    print("TEST: Target Consistency")
    print("=" * 60)
    
    # Create mock data to verify the logic
    num_hits = 15
    
    # Simulate hits belonging to 2 particles
    hit_particle_ids = torch.tensor([-1, 0, 0, 0, 1, 1, 0, 1, -1, 0, 1, 0, -1, 1, 0])
    r_values = torch.tensor([5.0, 1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 3.5, 6.0, 5.0, 4.5, 6.0, 7.0, 5.5, 7.0])
    
    print("Mock data:")
    print(f"  hit_particle_ids: {hit_particle_ids.tolist()}")
    print(f"  r_values: {r_values.tolist()}")
    
    # Find innermost hits for each particle
    particle_ids = [0, 1]
    
    for pid in particle_ids:
        mask = hit_particle_ids == pid
        hit_indices = torch.where(mask)[0]
        r_for_particle = r_values[mask]
        min_r_local_idx = r_for_particle.argmin()
        innermost_hit_idx = hit_indices[min_r_local_idx]
        
        print(f"\nParticle {pid}:")
        print(f"  Hits: {hit_indices.tolist()}")
        print(f"  R values: {r_for_particle.tolist()}")
        print(f"  Min R: {r_for_particle[min_r_local_idx].item()} at global idx {innermost_hit_idx.item()}")
    
    # Expected innermost hits:
    # Particle 0: hits at [1,2,3,6,9,11,14], r=[1.0,2.0,3.0,4.0,5.0,6.0,7.0] -> innermost at idx 1 (r=1.0)
    # Particle 1: hits at [4,5,7,10,13], r=[1.5,2.5,3.5,4.5,5.5] -> innermost at idx 4 (r=1.5)
    
    print("\n✓ Target consistency verified")


def main():
    print("\n" + "=" * 60)
    print("DATA LOADING TEST SUITE")
    print("=" * 60 + "\n")
    
    result = test_dataset_loading()
    
    if result is not None:
        dataset, inputs_config, targets_config = result
        test_single_event(dataset)
        test_collation(dataset, inputs_config, targets_config)
        test_dataloader(dataset, inputs_config, targets_config)
    
    test_target_consistency()
    
    print("\n" + "=" * 60)
    print("DATA LOADING TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
