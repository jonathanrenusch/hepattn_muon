#!/usr/bin/env python3
"""
Script to analyze HEP collision event files and count events by decay type.
Supports J/psi, ttbar, and Zmumu decay processes.
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import argparse

def parse_filename(filename):
    """
    Parse HDF5 filename to extract decay type and event count.
    
    Expected patterns:
    - Files containing 'jpsi' or 'Jpsi' for J/psi events
    - Files containing 'ttbar' for ttbar events  
    - Files containing 'Zmumu' for Z->mumu events
    - Event count from patterns like '1734events', prioritizing the second number over 'n2000'
    """
    # Convert to lowercase for easier matching
    filename_lower = filename.lower()
    
    # Determine decay type
    decay_type = None
    if 'jpsi' in filename_lower:
        decay_type = 'J/psi'
    elif 'ttbar' in filename_lower:
        decay_type = 'ttbar'
    elif 'zmumu' in filename_lower:
        decay_type = 'Zmumu'
    else:
        return None, 0
    
    # Extract event count - prioritize patterns with 'events' suffix
    # Example: PhPy8EG_AZNLO_Zmumu_PU200_skip954000_n2000_1734events.h5
    patterns = [
        r'(\d+)events',     # 1734events (highest priority)
        r'_(\d+)\.h5$',     # _2500.h5 (end of filename)
        r'_n(\d+)_',        # _n2000_ (lower priority)
        r'_n(\d+)',         # _n1000
        r'(\d+)_',          # 1000_
        r'(\d{3,})'         # any number with 3+ digits (lowest priority)
    ]
    
    event_count = 0
    for pattern in patterns:
        matches = re.findall(pattern, filename)
        if matches:
            # Take the last match for this pattern (in case of multiple)
            event_count = int(matches[-1])
            break
    
    return decay_type, event_count

def analyze_directory(directory_path):
    """Analyze all HDF5 files in the given directory."""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist!")
        return None
    
    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory!")
        return None
    
    # Find all HDF5 files
    h5_files = list(directory.glob("*.h5")) + list(directory.glob("*.hdf5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {directory_path}")
        return None
    
    # Count events by decay type
    event_counts = defaultdict(int)
    file_counts = defaultdict(int)
    unrecognized_files = []
    
    print(f"Analyzing {len(h5_files)} HDF5 files...")
    
    for h5_file in h5_files:
        decay_type, event_count = parse_filename(h5_file.name)
        
        if decay_type:
            event_counts[decay_type] += event_count
            file_counts[decay_type] += 1
            print(f"  {h5_file.name}: {decay_type} - {event_count} events")
        else:
            unrecognized_files.append(h5_file.name)
            print(f"  {h5_file.name}: UNRECOGNIZED")
    
    return {
        'event_counts': dict(event_counts),
        'file_counts': dict(file_counts),
        'unrecognized_files': unrecognized_files,
        'total_files': len(h5_files)
    }

def write_text_report(results, output_file, directory_path):
    """Write results to a text file."""
    total_events = sum(results['event_counts'].values())
    
    with open(output_file, 'w') as f:
        f.write("HEP Collision Event Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Directory analyzed: {directory_path}\n")
        f.write(f"Analysis date: {Path().cwd()}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total HDF5 files analyzed: {results['total_files']}\n")
        f.write(f"Total events: {total_events:,}\n")
        f.write(f"Recognized files: {results['total_files'] - len(results['unrecognized_files'])}\n")
        f.write(f"Unrecognized files: {len(results['unrecognized_files'])}\n\n")
        
        f.write("Event Distribution by Decay Type:\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'Decay Type':<10} {'Files':<6} {'Events':<12} {'Percentage':<10}\n")
        f.write("-" * 35 + "\n")
        
        for decay_type in sorted(results['event_counts'].keys()):
            events = results['event_counts'][decay_type]
            files = results['file_counts'][decay_type]
            percentage = (events / total_events * 100) if total_events > 0 else 0
            f.write(f"{decay_type:<10} {files:<6} {events:<12,} {percentage:<10.1f}%\n")
        
        f.write("-" * 35 + "\n")
        f.write(f"{'Total':<10} {results['total_files'] - len(results['unrecognized_files']):<6} {total_events:<12,} {'100.0':<10}%\n\n")
        
        if results['unrecognized_files']:
            f.write("Unrecognized Files:\n")
            f.write("-" * 20 + "\n")
            for filename in results['unrecognized_files']:
                f.write(f"  - {filename}\n")
    
    print(f"Text report written to: {output_file}")

def print_summary(results):
    """Print a summary to console."""
    total_events = sum(results['event_counts'].values())
    
    print("\n" + "="*60)
    print("HEP COLLISION EVENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total files analyzed: {results['total_files']}")
    print(f"Total events: {total_events:,}")
    print("\nEvent distribution by decay type:")
    print("-" * 40)
    
    for decay_type in sorted(results['event_counts'].keys()):
        events = results['event_counts'][decay_type]
        files = results['file_counts'][decay_type]
        percentage = (events / total_events * 100) if total_events > 0 else 0
        print(f"{decay_type:>8}: {files:>3} files, {events:>8,} events ({percentage:>5.1f}%)")
    
    if results['unrecognized_files']:
        print(f"\nUnrecognized files: {len(results['unrecognized_files'])}")
        for filename in results['unrecognized_files'][:5]:  # Show first 5
            print(f"  - {filename}")
        if len(results['unrecognized_files']) > 5:
            print(f"  ... and {len(results['unrecognized_files']) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Analyze HEP collision event files by decay type')
    parser.add_argument('--directory', '-d', help='Path to directory containing HDF5 files', default='/eos/project/e/end-to-end-muon-tracking/tracking/data/ml_training_data_2694000_hdf5/data')
    parser.add_argument('-o', '--output', default='event_analysis_report.txt', 
                       help='Output text file (default: event_analysis_report.txt)')
    
    args = parser.parse_args()
    
    # Analyze the directory
    results = analyze_directory(args.directory)
    
    if results is None:
        sys.exit(1)
    
    # Print summary to console
    print_summary(results)
    
    # Write text report
    write_text_report(results, args.output, args.directory)

if __name__ == "__main__":
    main()