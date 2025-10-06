#!/usr/bin/env python3
"""
Deep analysis to find the real source of 40% vs 90% baseline filter discrepancy
within the already-filtered dataset.

Since both datasets show similar track distributions (96% with 2 tracks),
the difference must come from subtle differences in how tracks are selected
or filtered within events.
"""

import numpy as np
import h5py
import yaml
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

def analyze_hit_filter_approach(data_dir, config_path, max_events=500):
    """Simulate the hit filter script's approach to track selection."""
    
    print("=" * 80)
    print("SIMULATING HIT FILTER SCRIPT APPROACH")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    inputs = data_config.get('inputs', {})
    targets = data_config.get('targets', {})
    
    # Setup data module without particle limit
    data_module = AtlasMuonDataModule(
        train_dir=str(data_dir),
        val_dir=str(data_dir), 
        test_dir=str(data_dir),
        num_workers=1,
        num_train=1,
        num_val=1,
        num_test=max_events,
        batch_size=1,
        inputs=inputs,
        targets=targets,
        # No event_max_num_particles limit - this is key!
    )
    
    data_module.setup("test")
    dataloader = data_module.test_dataloader(shuffle=False)
    
    # Collect all hits and reconstruct tracks like hit filter script
    all_event_ids = []
    all_particle_ids = []
    all_particle_pts = []
    all_particle_etas = []
    all_station_indices = []
    all_true_labels = []
    
    print("Collecting data like hit filter script...")
    
    for batch_idx, (inputs_batch, targets_batch) in enumerate(tqdm(dataloader)):
        if batch_idx >= max_events:
            break
            
        # Extract hit-level information
        station_indices = inputs_batch["hit_spacePoint_stationIndex"][0].numpy()
        true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(bool)
        
        # Get particle information for each hit
        # This requires mapping hits to particles - complex reconstruction
        event_id = batch_idx
        
        # For simplicity, focus on true hits only
        true_hit_mask = true_labels
        if not np.any(true_hit_mask):
            continue
            
        # Store hit-level data
        n_hits = len(station_indices)
        all_event_ids.extend([event_id] * n_hits)
        all_station_indices.extend(station_indices)
        all_true_labels.extend(true_labels)
        
        # Placeholder particle data - this is where the complexity lies
        # In the real hit filter script, this comes from complex data processing
        all_particle_ids.extend([0] * n_hits)  # Simplified
        all_particle_pts.extend([10.0] * n_hits)  # Simplified
        all_particle_etas.extend([1.0] * n_hits)  # Simplified
    
    # Convert to arrays
    all_event_ids = np.array(all_event_ids)
    all_particle_ids = np.array(all_particle_ids) 
    all_particle_pts = np.array(all_particle_pts)
    all_particle_etas = np.array(all_particle_etas)
    all_station_indices = np.array(all_station_indices)
    all_true_labels = np.array(all_true_labels)
    
    print(f"Collected {len(all_true_labels):,} hits from {max_events} events")
    print(f"True hits: {np.sum(all_true_labels):,}")
    
    # Now simulate the baseline filtering like hit filter script
    # Get unique (event_id, particle_id) combinations for tracks
    true_hit_mask = all_true_labels
    valid_combinations = np.unique(
        np.column_stack([
            all_event_ids[true_hit_mask],
            all_particle_ids[true_hit_mask]
        ]), axis=0
    )
    
    print(f"Found {len(valid_combinations)} unique tracks from true hits")
    
    # Apply baseline filtering to each track
    baseline_stats = {
        'total_tracks_checked': 0,
        'tracks_passed_all_cuts': 0
    }
    
    for event_id, particle_id in valid_combinations:
        baseline_stats['total_tracks_checked'] += 1
        
        # Get hits for this track
        track_mask = (
            (all_event_ids == event_id) & 
            (all_particle_ids == particle_id) & 
            true_hit_mask
        )
        
        track_hits = np.sum(track_mask)
        if track_hits < 9:
            continue
            
        # Get station info for this track
        track_stations = all_station_indices[track_mask]
        unique_stations, station_counts = np.unique(track_stations, return_counts=True)
        
        if len(unique_stations) >= 3 and np.sum(station_counts >= 3) >= 3:
            baseline_stats['tracks_passed_all_cuts'] += 1
    
    pass_rate = baseline_stats['tracks_passed_all_cuts'] / max(1, baseline_stats['total_tracks_checked']) * 100
    
    print(f"\nHIT FILTER APPROACH RESULTS:")
    print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
    print(f"  Tracks passed baseline: {baseline_stats['tracks_passed_all_cuts']}")
    print(f"  Baseline pass rate: {pass_rate:.1f}%")
    
    return baseline_stats


def analyze_task1_approach(data_dir, config_path, max_events=500):
    """Simulate the task1 script's approach to track selection."""
    
    print("\n" + "=" * 80)
    print("SIMULATING TASK1 SCRIPT APPROACH")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    inputs = data_config.get('inputs', {})
    targets = data_config.get('targets', {})
    
    # Setup data module WITH particle limit like task1
    data_module = AtlasMuonDataModule(
        train_dir=str(data_dir),
        val_dir=str(data_dir),
        test_dir=str(data_dir),
        num_workers=1,
        num_train=1,
        num_val=1, 
        num_test=max_events,
        batch_size=1,
        event_max_num_particles=2,  # This is the key difference!
        inputs=inputs,
        targets=targets,
    )
    
    data_module.setup("test")
    dataloader = data_module.test_dataloader(shuffle=False)
    
    baseline_stats = {
        'total_tracks_checked': 0,
        'tracks_passed_all_cuts': 0
    }
    
    print("Processing events like task1 script...")
    
    for batch_idx, (inputs_batch, targets_batch) in enumerate(tqdm(dataloader)):
        if batch_idx >= max_events:
            break
            
        # Get particle validity
        particle_valid = targets_batch['particle_valid'][0].numpy()
        num_valid = int(particle_valid.sum())
        
        if num_valid == 0:
            continue
            
        # Get station indices and hit assignments
        station_indices = inputs_batch["hit_spacePoint_stationIndex"][0].numpy()
        
        if 'particle_hit_valid' in targets_batch:
            hit_assignments = targets_batch['particle_hit_valid'][0].numpy()
            
            # Process each valid particle/track
            for track_idx in range(num_valid):
                baseline_stats['total_tracks_checked'] += 1
                
                track_hits = hit_assignments[track_idx]
                track_mask = track_hits.astype(bool)
                
                # Apply baseline criteria
                total_hits = np.sum(track_hits)
                if total_hits < 9:
                    continue
                    
                # Get kinematic info if available
                eta_ok = True
                pt_ok = True
                
                if 'particle_truthMuon_eta' in targets_batch:
                    eta = targets_batch['particle_truthMuon_eta'][0, track_idx].item()
                    if abs(eta) < 0.1 or abs(eta) > 2.7:
                        eta_ok = False
                        
                if 'particle_truthMuon_pt' in targets_batch:
                    pt = targets_batch['particle_truthMuon_pt'][0, track_idx].item() 
                    if pt < 3.0:
                        pt_ok = False
                
                if not (eta_ok and pt_ok):
                    continue
                    
                # Check station requirements
                track_stations = station_indices[track_mask]
                unique_stations, station_counts = np.unique(track_stations, return_counts=True)
                
                if len(unique_stations) >= 3 and np.sum(station_counts >= 3) >= 3:
                    baseline_stats['tracks_passed_all_cuts'] += 1
    
    pass_rate = baseline_stats['tracks_passed_all_cuts'] / max(1, baseline_stats['total_tracks_checked']) * 100
    
    print(f"\nTASK1 APPROACH RESULTS:")
    print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
    print(f"  Tracks passed baseline: {baseline_stats['tracks_passed_all_cuts']}")
    print(f"  Baseline pass rate: {pass_rate:.1f}%")
    
    return baseline_stats


def main():
    data_dir = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    config_path = "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"
    max_events = 1000
    
    print(f"Deep analysis of baseline filter discrepancy")
    print(f"Data directory: {data_dir}")
    print(f"Max events: {max_events}")
    
    hit_filter_stats = analyze_hit_filter_approach(data_dir, config_path, max_events)
    task1_stats = analyze_task1_approach(data_dir, config_path, max_events)
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    hit_filter_rate = hit_filter_stats['tracks_passed_all_cuts'] / max(1, hit_filter_stats['total_tracks_checked']) * 100
    task1_rate = task1_stats['tracks_passed_all_cuts'] / max(1, task1_stats['total_tracks_checked']) * 100
    
    print(f"Hit filter approach: {hit_filter_rate:.1f}% ({hit_filter_stats['tracks_passed_all_cuts']}/{hit_filter_stats['total_tracks_checked']})")
    print(f"Task1 approach: {task1_rate:.1f}% ({task1_stats['tracks_passed_all_cuts']}/{task1_stats['total_tracks_checked']})")
    print(f"Difference: {task1_rate - hit_filter_rate:.1f} percentage points")
    
    print(f"\nPOSSIBLE CAUSES OF DIFFERENCE:")
    print(f"1. event_max_num_particles=2 in task1 vs unlimited in hit filter")
    print(f"2. Different track selection criteria")
    print(f"3. Different particle-to-hit mapping logic")
    print(f"4. Different handling of multi-track events")


if __name__ == "__main__":
    main()