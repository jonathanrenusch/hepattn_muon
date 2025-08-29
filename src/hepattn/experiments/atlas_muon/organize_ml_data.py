#!/usr/bin/env python3
"""
Script to organize HEP ROOT files for machine learning training.
Splits files into train/validation/test sets with equal representation of each physics process.

Usage:
    python organize_ml_data.py

The script will:
1. Identify files by physics process (Jpsi, ttbar, Zmumu)
2. Create train/validation/test directories
3. Move files maintaining 80%/10%/10% split with equal flavor distribution
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

def identify_file_flavor(filename):
    """Identify the physics process type from filename."""
    if filename.startswith('P8B_A14_CTEQ6L1_Jpsi1S_mu6mu6_PU200_'):
        return 'jpsi'
    elif filename.startswith('PhPy8EG_A14_ttbar_hdamp258p75_dil_PU200_'):
        return 'ttbar'
    elif filename.startswith('PhPy8EG_AZNLO_Zmumu_PU200_'):
        return 'zmumu'
    else:
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Organize HEP ROOT files for machine learning training.')
    parser.add_argument('--source-dir', '-s', type=str,
                        default='/eos/user/j/jorenusc/tracking/data_gen/PU200FullData_preMDTTrack',
                        help='Source directory containing the ROOT files')
    parser.add_argument('--dry-run', action='store_true',
                        help="Don't move files; just print actions")
    parser.add_argument('--events-per-file', '-e', type=int, default=2500,
                        help='Number of events per ROOT file (integrated into target directory names)')
    args = parser.parse_args()

    # Source directory containing the ROOT files
    source_dir = Path(args.source_dir)
    dry_run = args.dry_run
    events_per_file = args.events_per_file
    
    # We'll compute which files go into each split first so we can
    # determine the total number of events per split and embed that in
    # the directory names (events_per_file * number_of_files_in_split).
    
    # Get all ROOT files and organize by flavor
    files_by_flavor = defaultdict(list)
    
    for filename in os.listdir(source_dir):
        if filename.endswith('.root'):
            flavor = identify_file_flavor(filename)
            if flavor:
                files_by_flavor[flavor].append(filename)
            else:
                print(f"Warning: Unknown file pattern: {filename}")
    
    # Print summary of discovered files
    print("\nFile inventory:")
    total_files = 0
    for flavor, files in files_by_flavor.items():
        print(f"  {flavor.upper()}: {len(files)} files")
        total_files += len(files)
    print(f"  TOTAL: {total_files} files")
    
    # Verify we have equal numbers of each flavor
    flavor_counts = [len(files) for files in files_by_flavor.values()]
    if len(set(flavor_counts)) != 1:
        print("Warning: Unequal number of files per flavor!")
        for flavor, files in files_by_flavor.items():
            print(f"  {flavor}: {len(files)} files")
    
    # Set random seed for reproducible splits
    random.seed(42)

    # First pass: compute per-flavor split lists and planned counts
    splits_by_flavor = {}
    planned_counts = {'training': 0, 'validation': 0, 'test': 0}

    for flavor, files in files_by_flavor.items():
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)

        n_files = len(shuffled_files)
        n_train = int(n_files * 0.9)  # 90% for training
        n_val = int(n_files * 0.05)    # 5% for validation
        n_test = n_files - n_train - n_val  # Remaining for test (handles rounding)

        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train:n_train + n_val]
        test_files = shuffled_files[n_train + n_val:]

        splits_by_flavor[flavor] = {
            'training': train_files,
            'validation': val_files,
            'test': test_files,
        }

        planned_counts['training'] += len(train_files)
        planned_counts['validation'] += len(val_files)
        planned_counts['test'] += len(test_files)

        print(f"\n{flavor.upper()} split:")
        print(f"  Training: {len(train_files)} files")
        print(f"  Validation: {len(val_files)} files")
        print(f"  Test: {len(test_files)} files")

    # Compute total events per split and set target directories accordingly
    total_events = {k: events_per_file * v for k, v in planned_counts.items()}

    train_dir = source_dir.parent / f'ml_training_data_{total_events["training"]}'
    val_dir = source_dir.parent / f'ml_validation_data_{total_events["validation"]}'
    test_dir = source_dir.parent / f'ml_test_data_{total_events["test"]}'

    # Create target directories (skip creation in dry-run)
    for directory in [train_dir, val_dir, test_dir]:
        if not dry_run:
            directory.mkdir(exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"(dry-run) Would create directory: {directory}")

    # Second pass: perform moves using the precomputed lists
    split_summary = defaultdict(lambda: defaultdict(int))

    for flavor, parts in splits_by_flavor.items():
        for split_name, file_list in parts.items():
            target_dir = {'training': train_dir, 'validation': val_dir, 'test': test_dir}[split_name]
            for filename in file_list:
                source_path = source_dir / filename
                target_path = target_dir / filename
                try:
                    if dry_run:
                        print(f"(dry-run) Would move {source_path} -> {target_path}")
                    else:
                        shutil.move(str(source_path), str(target_path))
                    split_summary[split_name][flavor] += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
    
    # Print final summary
    print("\n" + "="*60)
    print("DATA ORGANIZATION COMPLETE")
    print("="*60)
    
    for split in ['training', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        total_in_split = 0
        for flavor in ['jpsi', 'ttbar', 'zmumu']:
            count = split_summary[split][flavor]
            print(f"  {flavor.upper()}: {count} files")
            total_in_split += count
        print(f"  TOTAL: {total_in_split} files")
    
    print(f"\nDirectories created:")
    print(f"  Training data: {train_dir}")
    print(f"  Validation data: {val_dir}")
    print(f"  Test data: {test_dir}")
    
    # Verify original directory
    remaining_files = [f for f in os.listdir(source_dir) if f.endswith('.root')]
    print(f"\nRemaining files in original directory: {len(remaining_files)}")
    if remaining_files:
        print("WARNING: Some ROOT files remain in the original directory!")
        for f in remaining_files[:5]:  # Show first 5
            print(f"  {f}")
        if len(remaining_files) > 5:
            print(f"  ... and {len(remaining_files) - 5} more")

    # Helper to derive number of .root entries in a directory
    def count_root_files(dir_path: Path) -> int:
        if not dir_path.exists():
            return 0
        try:
            return sum(1 for p in dir_path.iterdir() if p.is_file() and p.name.endswith('.root'))
        except Exception:
            # In case of permission/IO errors, return 0
            return 0

    # Derive counts from the directories themselves (more reliable than counters if external changes occurred)
    derived_train = count_root_files(train_dir)
    derived_val = count_root_files(val_dir)
    derived_test = count_root_files(test_dir)

    print('\nDerived file counts from target directories:')
    print(f'  Training dir ({train_dir}): {derived_train} .root files')
    print(f'  Validation dir ({val_dir}): {derived_val} .root files')
    print(f'  Test dir ({test_dir}): {derived_test} .root files')

if __name__ == "__main__":
    main()
