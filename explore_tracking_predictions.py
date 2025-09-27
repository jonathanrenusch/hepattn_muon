#!/usr/bin/env python3
"""
Script to explore the structure of tracking model predictions.
This will help us understand what data is available in the HDF5 evaluation file.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def explore_h5_structure(file_path, max_depth=2, max_events=3):
    """Recursively explore HDF5 file structure."""
    
    def print_item(name, item, depth=0):
        indent = "  " * depth
        if isinstance(item, h5py.Group):
            print(f"{indent}{name}/ (Group) - {len(item.keys())} items")
            if depth < max_depth:
                # Only show first few keys if there are many
                keys = list(item.keys())
                if len(keys) > 5:
                    keys = keys[:3] + ['...']
                for i, key in enumerate(keys):
                    if key == '...':
                        print(f"{indent}  ... ({len(item.keys()) - 3} more items)")
                        break
                    print_item(key, item[key], depth + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}{name} (Dataset) - Shape: {item.shape}, Dtype: {item.dtype}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"Exploring: {file_path}")
        print(f"Total events in file: {len(f.keys())}")
        print("=" * 80)
        
        # Only examine first few events
        event_keys = list(f.keys())[:max_events]
        for i, key in enumerate(event_keys):
            print_item(key, f[key])
            if i < len(event_keys) - 1:
                print()
        
        if len(f.keys()) > max_events:
            print(f"... and {len(f.keys()) - max_events} more events")

def examine_sample_event(file_path, event_id=None):
    """Examine a specific event in detail."""
    with h5py.File(file_path, 'r') as f:
        # Get first available event if none specified
        if event_id is None:
            event_keys = list(f.keys())
            if event_keys:
                event_id = event_keys[0]
            else:
                print("No events found in file!")
                return
        
        print(f"\nDetailed examination of event: {event_id}")
        print("=" * 50)
        
        if event_id not in f:
            print(f"Event {event_id} not found!")
            available_events = list(f.keys())[:5]  # Show first 5
            print(f"Available events (first 5): {available_events}")
            return
        
        event_group = f[event_id]
        
        def examine_group(group, group_name="", depth=0):
            indent = "  " * depth
            print(f"{indent}{group_name}:")
            
            keys = list(group.keys())
            # Limit to first 5 items per group
            if len(keys) > 5:
                keys = keys[:3]
                show_more = True
            else:
                show_more = False
            
            for key in keys:
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(f"{indent}  {key}/ (Group) - {len(item.keys())} items")
                    if depth < 2:  # Limit recursion
                        examine_group(item, key, depth + 1)
                elif isinstance(item, h5py.Dataset):
                    data = item[...]
                    print(f"{indent}  {key}: shape={item.shape}, dtype={item.dtype}")
                    
                    # Show statistics for numerical data
                    if np.issubdtype(item.dtype, np.number) and data.size > 0:
                        print(f"{indent}    Stats: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
                    
                    # Show sample values for small datasets only
                    if data.size <= 5:
                        print(f"{indent}    Values: {data}")
            
            if show_more:
                print(f"{indent}  ... and {len(group.keys()) - 3} more items")
        
        examine_group(event_group, event_id)

def check_for_logits(file_path):
    """Look for logit/raw output data in the predictions."""
    print("\nSearching for logits and raw outputs...")
    print("=" * 50)
    
    logit_keywords = ['logit', 'raw', 'before_sigmoid', 'before_softmax', 'logits']
    
    def search_logits(group, path=""):
        found_logits = []
        for key in group.keys():
            current_path = f"{path}/{key}" if path else key
            item = group[key]
            
            if isinstance(item, h5py.Dataset):
                # Check if name suggests logits
                if any(keyword in key.lower() for keyword in logit_keywords):
                    found_logits.append(current_path)
                    print(f"Potential logits found: {current_path} - Shape: {item.shape}")
            elif isinstance(item, h5py.Group):
                found_logits.extend(search_logits(item, current_path))
        
        return found_logits
    
    with h5py.File(file_path, 'r') as f:
        # Check first few events
        event_keys = list(f.keys())[:3]
        all_logits = []
        
        for event_key in event_keys:
            print(f"\nChecking event {event_key}:")
            logits_in_event = search_logits(f[event_key])
            all_logits.extend(logits_in_event)
        
        return list(set(all_logits))  # Remove duplicates

def analyze_prediction_tasks(file_path):
    """Analyze the three main prediction tasks."""
    print("\nAnalyzing prediction tasks...")
    print("=" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # Get first event for analysis
        event_keys = list(f.keys())
        if not event_keys:
            print("No events found!")
            return
        
        first_event = f[event_keys[0]]
        print(f"Analyzing first event: {event_keys[0]}")
        
        # Look for the three main tasks
        task_patterns = {
            'track_hit_valid': ['track_hit_valid', 'hit_mask', 'hit_assignment'],
            'track_valid': ['track_valid', 'object_valid', 'track_classification'],
            'regression': ['regression', 'parameter', 'eta', 'phi', 'qpt', 'pt']
        }
        
        def find_task_outputs(group, path=""):
            task_outputs = {task: [] for task in task_patterns.keys()}
            
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Dataset):
                    # Check which task this might belong to
                    for task, patterns in task_patterns.items():
                        if any(pattern in key.lower() for pattern in patterns):
                            task_outputs[task].append((current_path, item.shape, item.dtype))
                elif isinstance(item, h5py.Group):
                    sub_outputs = find_task_outputs(item, current_path)
                    for task in task_outputs:
                        task_outputs[task].extend(sub_outputs[task])
            
            return task_outputs
        
        task_outputs = find_task_outputs(first_event)
        
        for task, outputs in task_outputs.items():
            print(f"\nTask: {task}")
            if outputs:
                for path, shape, dtype in outputs:
                    print(f"  {path}: shape={shape}, dtype={dtype}")
            else:
                print(f"  No outputs found for {task}")

def main():
    # Path to the evaluation file
    eval_path = "/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5"
    
    if not Path(eval_path).exists():
        print(f"Error: File not found: {eval_path}")
        sys.exit(1)
    
    print("TRACKING MODEL PREDICTION EXPLORATION")
    print("=" * 80)
    
    # 1. Explore overall structure
    explore_h5_structure(eval_path, max_depth=2, max_events=2)
    
    # 2. Examine a sample event in detail
    examine_sample_event(eval_path)
    
    # 3. Look for logits
    logits_found = check_for_logits(eval_path)
    if logits_found:
        print(f"\nSummary - Logits found at: {logits_found}")
    else:
        print("\nSummary - No obvious logits found. May need to check raw model outputs.")
    
    # 4. Analyze prediction tasks
    analyze_prediction_tasks(eval_path)
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("Use this information to understand what data is available for evaluation.")

if __name__ == "__main__":
    main()