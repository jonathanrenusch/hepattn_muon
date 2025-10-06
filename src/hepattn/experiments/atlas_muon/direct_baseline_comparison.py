#!/usr/bin/env python3
"""
Direct comparison of baseline filtering between hit filter and task1 approaches
on the EXACT SAME EVENTS to identify where the discrepancy comes from.

Since baseline filtering uses ground truth values, both approaches should give
identical results if they're analyzing the same tracks correctly.
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

def compare_baseline_filtering_exact_same_events(data_dir, config_path, max_events=100):
    """Compare baseline filtering on exactly the same events using both approaches."""
    
    print("=" * 80)
    print("DIRECT COMPARISON: SAME EVENTS, SAME TRACKS")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    inputs = data_config.get('inputs', {})
    targets = data_config.get('targets', {})
    
    # Setup BOTH data modules with IDENTICAL parameters
    print("Setting up data modules...")
    
    # Hit filter approach (no particle limit)
    hit_filter_module = AtlasMuonDataModule(
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
        # No event_max_num_particles - this allows all tracks
    )
    
    # Task1 approach (with particle limit)
    task1_module = AtlasMuonDataModule(
        train_dir=str(data_dir),
        val_dir=str(data_dir),
        test_dir=str(data_dir),
        num_workers=1,
        num_train=1,
        num_val=1,
        num_test=max_events,
        batch_size=1,
        event_max_num_particles=2,  # This is the key difference
        inputs=inputs,
        targets=targets,
    )
    
    hit_filter_module.setup("test")
    task1_module.setup("test")
    
    hit_filter_loader = hit_filter_module.test_dataloader(shuffle=False)
    task1_loader = task1_module.test_dataloader(shuffle=False)
    
    print("Analyzing same events with both approaches...")
    
    discrepancies = []
    
    # Process both loaders in parallel
    for event_idx, ((hf_inputs, hf_targets), (t1_inputs, t1_targets)) in enumerate(
        zip(hit_filter_loader, task1_loader)
    ):
        if event_idx >= max_events:
            break
            
        print(f"\nEvent {event_idx}:")
        print("-" * 40)
        
        # Analyze hit filter approach tracks
        hf_tracks = analyze_hit_filter_tracks(hf_inputs, hf_targets, event_idx)
        
        # Analyze task1 approach tracks  
        t1_tracks = analyze_task1_tracks(t1_inputs, t1_targets, event_idx)
        
        # Compare results
        compare_track_results(hf_tracks, t1_tracks, event_idx, discrepancies)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF DISCREPANCIES")
    print("=" * 80)
    
    if discrepancies:
        for disc in discrepancies:
            print(f"Event {disc['event']}: {disc['description']}")
    else:
        print("No discrepancies found - both approaches give identical results!")

def analyze_hit_filter_tracks(inputs, targets, event_idx):
    """Analyze tracks using hit filter approach."""
    
    station_indices = inputs["hit_spacePoint_stationIndex"][0].numpy()
    
    # Reconstruct tracks from hit_on_valid_particle
    if "hit_on_valid_particle" in targets:
        true_labels = targets["hit_on_valid_particle"][0].numpy().astype(bool)
        
        # For simplicity, assume each event has tracks with sequential IDs
        # This is where the hit filter approach gets complex - it needs to reconstruct
        # which hits belong to which tracks
        
        # Count valid hits per "track" (this is oversimplified)
        num_true_hits = np.sum(true_labels)
        
        if num_true_hits == 0:
            return {"tracks": [], "approach": "hit_filter"}
        
        # Simplified: assume all true hits belong to one track
        # In reality, hit filter approach reconstructs tracks from hit-level data
        tracks = []
        if num_true_hits >= 9:  # Only if enough hits
            track = {
                "hits": num_true_hits,
                "stations": station_indices[true_labels],
                "eta": 1.0,  # Placeholder - would need reconstruction
                "pt": 10.0,  # Placeholder - would need reconstruction  
                "passes_baseline": check_baseline_simplified(num_true_hits, station_indices[true_labels])
            }
            tracks.append(track)
        
        return {"tracks": tracks, "approach": "hit_filter", "total_hits": len(station_indices)}
    
    return {"tracks": [], "approach": "hit_filter", "total_hits": len(station_indices)}

def analyze_task1_tracks(inputs, targets, event_idx):
    """Analyze tracks using task1 approach."""
    
    station_indices = inputs["hit_spacePoint_stationIndex"][0].numpy()
    
    tracks = []
    
    if "particle_valid" in targets and "particle_hit_valid" in targets:
        particle_valid = targets["particle_valid"][0].numpy()
        hit_assignments = targets["particle_hit_valid"][0].numpy()
        
        # Get kinematic info if available
        etas = targets.get("particle_truthMuon_eta", [None])[0].numpy() if "particle_truthMuon_eta" in targets else [1.0, 1.0]
        pts = targets.get("particle_truthMuon_pt", [None])[0].numpy() if "particle_truthMuon_pt" in targets else [10.0, 10.0]
        
        for track_idx in range(len(particle_valid)):
            if not particle_valid[track_idx]:
                continue
                
            track_hits = hit_assignments[track_idx]
            track_mask = track_hits.astype(bool)
            
            num_hits = np.sum(track_hits)
            track_stations = station_indices[track_mask]
            
            eta = etas[track_idx] if len(etas) > track_idx else 1.0
            pt = pts[track_idx] if len(pts) > track_idx else 10.0
            
            track = {
                "hits": num_hits,
                "stations": track_stations,
                "eta": eta,
                "pt": pt,
                "passes_baseline": check_baseline_full(num_hits, track_stations, eta, pt)
            }
            tracks.append(track)
    
    return {"tracks": tracks, "approach": "task1", "total_hits": len(station_indices)}

def check_baseline_simplified(num_hits, stations):
    """Simplified baseline check (just hits and stations)."""
    if num_hits < 9:
        return False
    
    unique_stations, station_counts = np.unique(stations, return_counts=True)
    return len(unique_stations) >= 3 and np.sum(station_counts >= 3) >= 3

def check_baseline_full(num_hits, stations, eta, pt):
    """Full baseline check including kinematics."""
    if num_hits < 9:
        return False
    if abs(eta) < 0.1 or abs(eta) > 2.7:
        return False
    if pt < 3.0:
        return False
    
    unique_stations, station_counts = np.unique(stations, return_counts=True)
    return len(unique_stations) >= 3 and np.sum(station_counts >= 3) >= 3

def compare_track_results(hf_tracks, t1_tracks, event_idx, discrepancies):
    """Compare results between hit filter and task1 approaches."""
    
    hf_track_count = len(hf_tracks["tracks"])
    t1_track_count = len(t1_tracks["tracks"])
    
    hf_passed = sum(1 for t in hf_tracks["tracks"] if t["passes_baseline"])
    t1_passed = sum(1 for t in t1_tracks["tracks"] if t["passes_baseline"])
    
    print(f"  Hit Filter: {hf_track_count} tracks, {hf_passed} passed baseline")
    print(f"  Task1:      {t1_track_count} tracks, {t1_passed} passed baseline")
    print(f"  Total hits: HF={hf_tracks['total_hits']}, T1={t1_tracks['total_hits']}")
    
    # Check for discrepancies
    if hf_track_count != t1_track_count:
        discrepancies.append({
            "event": event_idx,
            "description": f"Different track counts: HF={hf_track_count}, T1={t1_track_count}"
        })
    
    if hf_passed != t1_passed:
        discrepancies.append({
            "event": event_idx, 
            "description": f"Different pass rates: HF={hf_passed}/{hf_track_count}, T1={t1_passed}/{t1_track_count}"
        })
    
    # Detailed track comparison
    for i, (hf_track, t1_track) in enumerate(zip(hf_tracks["tracks"], t1_tracks["tracks"])):
        if hf_track["passes_baseline"] != t1_track["passes_baseline"]:
            discrepancies.append({
                "event": event_idx,
                "description": f"Track {i} baseline mismatch: HF={hf_track['passes_baseline']}, T1={t1_track['passes_baseline']}"
            })


def main():
    data_dir = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    config_path = "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"
    max_events = 50
    
    print(f"Direct comparison of baseline filtering approaches")
    print(f"Data directory: {data_dir}")
    print(f"Max events: {max_events}")
    
    compare_baseline_filtering_exact_same_events(data_dir, config_path, max_events)


if __name__ == "__main__":
    main()