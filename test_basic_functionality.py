#!/usr/bin/env python3

import sys
import numpy as np
import h5py
import yaml
from pathlib import Path

def test_basic_functionality():
    print("Testing basic functionality...")
    
    # Test file paths
    eval_path = "/shared/tracking/hepattn_muon/src/logs/ATLAS-Muon-6H100-600K_20250831-T195418/ckpts/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5"
    data_path = "/scratch/ml_test_data_156000_hdf5"
    
    print(f"Checking eval file: {eval_path}")
    print(f"Exists: {Path(eval_path).exists()}")
    
    print(f"Checking data dir: {data_path}")
    print(f"Exists: {Path(data_path).exists()}")
    
    # Test loading metadata
    try:
        with open(Path(data_path) / 'metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        print("✓ Metadata loaded successfully")
        print(f"  Hit features: {len(metadata['hit_features'])}")
        print(f"  Track features: {len(metadata['track_features'])}")
    except Exception as e:
        print(f"✗ Metadata loading failed: {e}")
        return
    
    # Test loading index arrays
    try:
        file_indices = np.load(Path(data_path) / 'event_file_indices.npy')
        row_indices = np.load(Path(data_path) / 'event_row_indices.npy')
        print(f"✓ Index arrays loaded: {len(file_indices)} events")
    except Exception as e:
        print(f"✗ Index array loading failed: {e}")
        return
    
    # Test loading first event from eval file
    try:
        with h5py.File(eval_path, 'r') as eval_file:
            first_key = '0'
            if first_key in eval_file:
                logits = eval_file[f"{first_key}/outputs/final/hit_filter/hit_logit"][0]
                preds = eval_file[f"{first_key}/preds/final/hit_filter/hit_on_valid_particle"][0]
                print(f"✓ Eval data loaded for event 0: {len(logits)} hits")
            else:
                print("✗ Event 0 not found in eval file")
                return
    except Exception as e:
        print(f"✗ Eval file loading failed: {e}")
        return
    
    # Test loading first event from raw data
    try:
        file_idx = file_indices[0]
        row_idx = row_indices[0]
        
        chunk = metadata['event_mapping']['chunk_summary'][file_idx]
        h5_file_path = Path(data_path) / chunk['h5_file']
        
        with h5py.File(h5_file_path, 'r') as f:
            num_hits = f['num_hits'][row_idx]
            num_tracks = f['num_tracks'][row_idx]
            hits_array = f['hits'][row_idx, :num_hits]
            tracks_array = f['tracks'][row_idx, :num_tracks]
            
        print(f"✓ Raw data loaded for event 0: {num_hits} hits, {num_tracks} tracks")
        
        # Check if shapes match
        if len(logits) == num_hits:
            print("✓ Shapes match between eval and raw data")
        else:
            print(f"✗ Shape mismatch: eval={len(logits)}, raw={num_hits}")
            
    except Exception as e:
        print(f"✗ Raw data loading failed: {e}")
        return
    
    print("Basic functionality test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()
