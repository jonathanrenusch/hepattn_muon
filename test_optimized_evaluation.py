#!/usr/bin/env python3
"""
Quick test script for the optimized evaluation to verify it works correctly.
This tests with a small number of events to ensure everything is functioning.
"""

import sys
import os
sys.path.append('/shared/tracking/hepattn_muon/src')

from hepattn.experiments.atlas_muon.evaluate_hit_filter_dataloader_optimized import AtlasMuonEvaluatorDataLoaderOptimized

def test_evaluation():
    print("=" * 60)
    print("TESTING OPTIMIZED EVALUATION")
    print("=" * 60)
    
    # Test with a small number of events
    test_params = {
        'eval_path': "/shared/tracking/data/1_5Mio_test_training/epoch=049-val_acc=0.99711_ml_test_data_150K_processed_eval.h5",
        'data_dir': "/shared/tracking/data/ml_test_data_150K_processed",
        'config_path': "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
        'output_dir': '/shared/tracking/hepattn_muon/test_evaluation_results',
        'max_events': 100  # Test with just 100 events
    }
    
    print("Test parameters:")
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    
    # Check if files exist
    for key in ['eval_path', 'data_dir', 'config_path']:
        path = test_params[key]
        if not os.path.exists(path):
            print(f"ERROR: {key} not found: {path}")
            return False
    
    print("\nStarting test evaluation...")
    
    try:
        # Create evaluator
        evaluator = AtlasMuonEvaluatorDataLoaderOptimized(**test_params)
        
        # Run optimized evaluation
        evaluator.run_evaluation_optimized(include_track_lengths=True)
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"Check results in: {test_params['output_dir']}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation()
    sys.exit(0 if success else 1)
