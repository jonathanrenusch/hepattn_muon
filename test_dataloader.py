#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import yaml
import numpy as np
from pathlib import Path

def test_dataloader():
    print("Testing DataLoader initialization...")
    
    try:
        from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
        print("✓ DataModule imported successfully")
    except Exception as e:
        print(f"✗ DataModule import failed: {e}")
        return
    
    # Load config
    config_path = "src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config loaded successfully")
        
        inputs = config['data']['inputs']
        targets = config['data']['targets']
        print(f"✓ Inputs: {list(inputs.keys())}")
        print(f"✓ Targets: {list(targets.keys())}")
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return
    
    # Test data path
    data_path = "/scratch/ml_test_data_156000_hdf5"
    if not os.path.exists(data_path):
        print(f"✗ Data path does not exist: {data_path}")
        return
    print(f"✓ Data path exists: {data_path}")
    
    # Try to create DataModule
    try:
        datamodule = AtlasMuonDataModule(
            train_dir=data_path,
            val_dir=data_path,
            test_dir=data_path,
            num_workers=1,
            num_train=5,
            num_val=5,
            num_test=5,
            batch_size=1,
            inputs=inputs,
            targets=targets,
        )
        print("✓ DataModule created successfully")
        
        # Setup
        datamodule.setup("test")
        print("✓ DataModule setup completed")
        
        # Get dataloader
        test_dataloader = datamodule.test_dataloader()
        print(f"✓ DataLoader created with {len(test_dataloader)} batches")
        
        # Try to get first batch
        print("Attempting to get first batch...")
        for i, (inputs_batch, targets_batch) in enumerate(test_dataloader):
            print(f"✓ Successfully loaded batch {i}")
            print(f"  Sample ID: {targets_batch['sample_id'][0].item()}")
            print(f"  Input keys: {list(inputs_batch.keys())}")
            print(f"  Target keys: {list(targets_batch.keys())}")
            
            # Check for plotting_spacePoint_truthLink
            if 'plotting_spacePoint_truthLink' in inputs_batch:
                print(f"  ✓ plotting_spacePoint_truthLink found with shape: {inputs_batch['plotting_spacePoint_truthLink'].shape}")
            else:
                print(f"  ✗ plotting_spacePoint_truthLink not found")
                print(f"  Available input keys: {list(inputs_batch.keys())}")
            
            if i >= 1:  # Test only first 2 batches
                break
                
        print("✓ DataLoader test completed successfully!")
        
    except Exception as e:
        print(f"✗ DataModule creation/usage failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()
