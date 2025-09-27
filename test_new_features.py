#!/usr/bin/env python3
"""
Test script to verify the new functionality in prep_events_multiprocess.py
"""

import subprocess
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_help_output():
    """Test that the help output shows the new arguments"""
    try:
        result = subprocess.run([
            sys.executable, 
            'src/hepattn/experiments/atlas_muon/prep_events_multiprocess.py', 
            '--help'
        ], capture_output=True, text=True, cwd='/shared/tracking/hepattn_muon')
        
        print("Help output:")
        print(result.stdout)
        
        # Check that our new arguments are present
        assert '--no-NSM' in result.stdout, "Missing --no-NSM argument"
        assert '--no-RPC' in result.stdout, "Missing --no-RPC argument"
        
        # Check that old arguments are removed
        assert '--pt_threshold' not in result.stdout, "Old --pt_threshold argument still present"
        assert '--eta_threshold' not in result.stdout, "Old --eta_threshold argument still present"
        assert '--num_hits_threshold' not in result.stdout, "Old --num_hits_threshold argument still present"
        
        print("✓ Help output test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Help output test failed: {e}")
        return False

def test_import():
    """Test that the module can be imported without errors"""
    try:
        from hepattn.experiments.atlas_muon.prep_events_multiprocess import ParallelRootFilter
        print("✓ Import test passed!")
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing new functionality in prep_events_multiprocess.py")
    print("=" * 60)
    
    success = True
    success &= test_import()
    success &= test_help_output()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)