#!/usr/bin/env python3
"""
Debug script to check what's happening with the data processing.
This will help identify why no events are being saved.
"""

import sys
import os
import numpy as np
import yaml
from pathlib import Path

def check_processed_data(output_dir):
    """Check the output of processed data to understand what went wrong"""
    output_path = Path(output_dir)
    
    print(f"Checking directory: {output_path}")
    print("=" * 60)
    
    # Check if directory exists
    if not output_path.exists():
        print(f"❌ Output directory {output_path} does not exist!")
        return
    
    # List contents
    print("Directory contents:")
    for item in output_path.iterdir():
        print(f"  {item.name}")
    
    # Check for metadata file
    metadata_file = output_path / 'metadata.yaml'
    if metadata_file.exists():
        print(f"\n✓ Found metadata.yaml")
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        
        print(f"Processing summary:")
        summary = metadata.get('processing_summary', {})
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n❌ No metadata.yaml found!")
    
    # Check for index files
    file_indices_path = output_path / 'event_file_indices.npy'
    row_indices_path = output_path / 'event_row_indices.npy'
    
    if file_indices_path.exists():
        file_indices = np.load(file_indices_path)
        print(f"\n✓ Found event_file_indices.npy with {len(file_indices)} entries")
        if len(file_indices) > 0:
            print(f"  First 10 file indices: {file_indices[:10]}")
        else:
            print(f"  ❌ File indices array is empty!")
    else:
        print(f"\n❌ No event_file_indices.npy found!")
    
    if row_indices_path.exists():
        row_indices = np.load(row_indices_path)
        print(f"\n✓ Found event_row_indices.npy with {len(row_indices)} entries")
        if len(row_indices) > 0:
            print(f"  First 10 row indices: {row_indices[:10]}")
        else:
            print(f"  ❌ Row indices array is empty!")
    else:
        print(f"\n❌ No event_row_indices.npy found!")
    
    # Check data directory
    data_dir = output_path / 'data'
    if data_dir.exists():
        print(f"\n✓ Found data directory")
        h5_files = list(data_dir.glob('*.h5'))
        print(f"  Number of H5 files: {len(h5_files)}")
        if len(h5_files) > 0:
            print(f"  H5 files:")
            for h5_file in h5_files[:5]:  # Show first 5
                print(f"    {h5_file.name}")
            if len(h5_files) > 5:
                print(f"    ... and {len(h5_files) - 5} more")
        else:
            print(f"  ❌ No H5 files found in data directory!")
    else:
        print(f"\n❌ No data directory found!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_data.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    check_processed_data(output_dir)