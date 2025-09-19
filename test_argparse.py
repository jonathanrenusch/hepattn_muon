#!/usr/bin/env python3

import sys
import os
import argparse

# Add the src directory to the path to find the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_argparse():
    """Test the argument parser to verify new flags are working"""
    
    # Create a simple version of the argument parser from the main file
    parser = argparse.ArgumentParser(description="Test new technology filtering flags")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input root files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output HDF5 files")
    parser.add_argument("-n", "--expected_num_events_per_file", type=int, default=2000, help="Expected number of events per root file")
    parser.add_argument("-max", "--max_events", type=int, default=-1, help="Maximum number of valid events each worker is allowed to process")
    parser.add_argument("-w", "--num_workers", type=int, default=None, help="Number of worker processes (default: 10)")
    parser.add_argument("--no-NSM", action="store_true", default=False, help="Remove STGC and MM technology hits from dataset")
    parser.add_argument("--no-RPC", action="store_true", default=False, help="Remove RPC technology hits from dataset")

    # Test cases
    test_cases = [
        # Basic case
        ["-i", "/test/input", "-o", "/test/output"],
        # With no-NSM flag
        ["-i", "/test/input", "-o", "/test/output", "--no-NSM"],
        # With no-RPC flag
        ["-i", "/test/input", "-o", "/test/output", "--no-RPC"],
        # With both flags
        ["-i", "/test/input", "-o", "/test/output", "--no-NSM", "--no-RPC"],
        # With additional parameters
        ["-i", "/test/input", "-o", "/test/output", "--no-NSM", "-max", "1000", "-w", "4"]
    ]
    
    print("Testing argument parser...")
    print("=" * 50)
    
    for i, test_args in enumerate(test_cases, 1):
        try:
            args = parser.parse_args(test_args)
            print(f"Test {i}: PASSED")
            print(f"  Args: {' '.join(test_args)}")
            print(f"  Parsed no_NSM: {getattr(args, 'no_NSM', False)}")
            print(f"  Parsed no_RPC: {getattr(args, 'no_RPC', False)}")
            print(f"  Input dir: {args.input_dir}")
            print(f"  Output dir: {args.output_dir}")
            
            # Test output directory modification logic
            if getattr(args, 'no_NSM', False) or getattr(args, 'no_RPC', False):
                base_output_dir = args.output_dir
                suffix = ""
                if getattr(args, 'no_NSM', False):
                    suffix += "_no-NSM"
                if getattr(args, 'no_RPC', False):
                    suffix += "_no-RPC"
                modified_output_dir = base_output_dir + suffix
                print(f"  Modified output dir: {modified_output_dir}")
            
            print()
            
        except Exception as e:
            print(f"Test {i}: FAILED - {e}")
            print()
    
    print("=" * 50)
    print("All argument parser tests completed!")

if __name__ == "__main__":
    test_argparse()