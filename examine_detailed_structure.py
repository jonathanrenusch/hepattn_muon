#!/usr/bin/env python3
"""
Quick script to examine the exact structure and contents of the tracking predictions.
"""

import h5py
import numpy as np
import sys

def examine_detailed_structure():
    eval_path = "/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5"
    
    with h5py.File(eval_path, 'r') as f:
        print("DETAILED STRUCTURE EXAMINATION")
        print("=" * 60)
        
        # Look at first event
        event = f['0']
        
        print("Available data for event 0:")
        print("\n1. OUTPUTS (Raw logits):")
        print("  outputs/final/track_valid/track_logit:", event['outputs/final/track_valid/track_logit'][...])
        print("  Shape:", event['outputs/final/track_valid/track_logit'].shape)
        
        print("\n  outputs/final/track_hit_valid/track_hit_logit shape:", event['outputs/final/track_hit_valid/track_hit_logit'].shape)
        print("  Sample values:", event['outputs/final/track_hit_valid/track_hit_logit'][0, 0, :5])  # First track, first 5 hits
        
        print("\n2. PREDICTIONS (Post-processed):")
        print("  preds/final/track_valid/track_valid:", event['preds/final/track_valid/track_valid'][...])
        print("  preds/final/track_hit_valid/track_hit_valid shape:", event['preds/final/track_hit_valid/track_hit_valid'].shape)
        print("  First track hits (first 10):", event['preds/final/track_hit_valid/track_hit_valid'][0, 0, :10])
        
        print("\n3. REGRESSION PREDICTIONS:")
        for param in ['track_truthMuon_eta', 'track_truthMuon_phi', 'track_truthMuon_qpt']:
            values = event[f'preds/final/parameter_regression/{param}'][...]
            print(f"  {param}: {values} (shape: {values.shape})")
        
        print("\n4. CHECK FOR REGRESSION LOGITS/RAW VALUES:")
        param_group = event['outputs/final/parameter_regression']
        print("  Available regression outputs:")
        for key in param_group.keys():
            data = param_group[key][...]
            print(f"    {key}: {data} (shape: {data.shape})")

if __name__ == "__main__":
    examine_detailed_structure()