#!/usr/bin/env python3
"""
Test script to validate the Optuna hyperparameter optimization setup.

This script runs a single trial to ensure the optimization framework works correctly.
"""

import yaml
import tempfile
import os
from pathlib import Path

from hepattn.experiments.atlas_muon.optuna_tune import (
    suggest_hyperparameters, 
    create_config_from_trial, 
    OptimizedWrapperModule
)


def test_config_generation():
    """Test that config generation works correctly."""
    print("Testing config generation...")
    
    # Load base config
    base_config_path = "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/smallCuts/optuna_base_config.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create a mock trial with some hyperparameters
    class MockTrial:
        def __init__(self):
            self.number = 0
        
        def suggest_categorical(self, name, choices):
            # Return the first choice for testing
            return choices[0]
    
    trial = MockTrial()
    
    # Get suggested hyperparameters
    trial_params = suggest_hyperparameters(trial)
    print(f"Suggested hyperparameters: {trial_params}")
    
    # Create config from trial
    config = create_config_from_trial(base_config, trial_params)
    
    # Validate that key parameters were updated
    assert config["model"]["model"]["init_args"]["dim"] == trial_params["dim"]
    assert config["model"]["model"]["init_args"]["encoder"]["init_args"]["num_layers"] == trial_params["num_encoder_layers"]
    assert config["model"]["model"]["init_args"]["decoder"]["num_decoder_layers"] == trial_params["num_decoder_layers"]
    
    print("✓ Config generation test passed!")
    return config


def test_config_file_creation():
    """Test that temporary config files can be created and loaded."""
    print("Testing config file creation...")
    
    # Generate a test config
    config = test_config_generation()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name
    
    try:
        # Try to load it back
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify it's the same
        assert loaded_config["model"]["model"]["init_args"]["dim"] == config["model"]["model"]["init_args"]["dim"]
        print("✓ Config file creation test passed!")
        
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_wrapper_module():
    """Test that the OptimizedWrapperModule can be instantiated."""
    print("Testing OptimizedWrapperModule...")
    
    # This is a basic test - we can't fully test without a complete model
    # but we can verify the class can be imported and has the expected methods
    assert hasattr(OptimizedWrapperModule, 'compute_raw_losses')
    assert hasattr(OptimizedWrapperModule, 'validation_step')
    assert hasattr(OptimizedWrapperModule, 'log_custom_metrics')
    
    print("✓ OptimizedWrapperModule test passed!")


def main():
    """Run all tests."""
    print("Running Optuna setup tests...\n")
    
    try:
        test_config_generation()
        print()
        
        test_config_file_creation()
        print()
        
        test_wrapper_module()
        print()
        
        print("🎉 All tests passed! The Optuna setup appears to be working correctly.")
        print("\nYou can now run the optimization with:")
        print("python optuna_tune.py --gpu 0 --n-trials 10")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())