#!/usr/bin/env python3
"""
Compare station indices for true hits between two datasets

This script loads 100 random events from both datasets and compares:
1. Station indices for true hits per track
2. Length of station indices arrays
3. Unique station indices per track
4. Side-by-side comparison of track content

Datasets:
- Dataset 1: /scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600
- Dataset 2: /home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts
"""

import os
import sys
import numpy as np
from pathlib import Path
import random

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('/shared/tracking/hepattn_muon/src')
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

def setup_data_module(data_dir, max_events=100):
    """Setup data module for a specific dataset."""
    data_module = AtlasMuonDataModule(
        train_dir=data_dir,
        val_dir=data_dir,
        test_dir=data_dir,
        num_workers=1,
        num_train=1,
        num_val=1,
        num_test=max_events,
        batch_size=1,
        event_max_num_particles=2,
        inputs={
            'hit': [
                'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
                'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
                'spacePoint_time', 'spacePoint_driftR',
                'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
                'spacePoint_channel', 'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
                'spacePoint_stationIndex', 'spacePoint_technology',
                'r', 's', 'theta', 'phi'
            ]
        },
        targets={
            'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi', 'truthMuon_qpt']
        }
    )
    
    data_module.setup("test")
    return data_module.test_dataloader(shuffle=False)

def analyze_track_station_indices(batch, dataset_name):
    """Analyze station indices for true hits in a batch."""
    inputs, targets = batch
    
    # Remove batch dimension
    true_station_index = inputs["hit_spacePoint_stationIndex"][0]
    true_particle_valid = targets['particle_valid'][0]
    true_hit_assignments = targets['particle_hit_valid'][0]
    
    valid_particles = true_particle_valid.numpy()
    num_valid = int(valid_particles.sum())
    
    track_data = []
    
    for track_idx in range(num_valid):
        true_hits = true_hit_assignments[track_idx].numpy()
        track_mask = true_hits.astype(bool)
        
        # Get station indices for true hits
        track_station_indices = true_station_index[track_mask].numpy()
        
        # Get true track parameters
        eta_tensor = targets["particle_truthMuon_eta"][0, track_idx]
        phi_tensor = targets["particle_truthMuon_phi"][0, track_idx]
        pt_tensor = targets["particle_truthMuon_pt"][0, track_idx]
        
        true_eta = eta_tensor.item() if eta_tensor.numel() == 1 else eta_tensor[0].item()
        true_phi = phi_tensor.item() if phi_tensor.numel() == 1 else phi_tensor[0].item()
        true_pt = pt_tensor.item() if pt_tensor.numel() == 1 else pt_tensor[0].item()
        
        track_info = {
            'track_idx': track_idx,
            'num_hits': int(np.sum(true_hits)),
            'station_indices': track_station_indices,
            'unique_stations': np.unique(track_station_indices),
            'num_unique_stations': len(np.unique(track_station_indices)),
            'pt': true_pt,
            'eta': true_eta,
            'phi': true_phi,
            'dataset': dataset_name
        }
        
        track_data.append(track_info)
    
    return track_data

def main():
    print("=" * 80)
    print("COMPARING STATION INDICES BETWEEN TWO DATASETS")
    print("=" * 80)
    
    # Dataset paths
    dataset1_path = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    dataset2_path = "/scratch/ml_test_data_156000_hdf5"
    
    print(f"FILTERED dataset: {dataset1_path}")
    print(f"UNFILTERED dataset: {dataset2_path}")
    print()
    
    # Setup data modules
    print("Setting up data modules...")
    dataloader1 = setup_data_module(dataset1_path, max_events=100)
    dataloader2 = setup_data_module(dataset2_path, max_events=100)
    
    dataset1_size = len(dataloader1.dataset)
    dataset2_size = len(dataloader2.dataset)
    
    print(f"FILTERED dataset size: {dataset1_size}")
    print(f"UNFILTERED dataset size: {dataset2_size}")
    
    # Use same random seed for reproducible comparison
    random.seed(42)
    np.random.seed(42)
    
    # Select same 100 random indices for both datasets
    max_events = min(100, dataset1_size, dataset2_size)
    random_indices = sorted(random.sample(range(min(dataset1_size, dataset2_size)), max_events))
    
    print(f"Comparing {max_events} events with indices: {random_indices[:10]}{'...' if len(random_indices) > 10 else ''}")
    print()
    
    comparison_results = []
    
    print("Processing events...")
    for i, event_idx in enumerate(random_indices):
        if i % 10 == 0:
            print(f"Processing event {i+1}/{max_events} (index {event_idx})")
        
        # Get batches from both datasets
        batch1 = dataloader1.dataset[event_idx]
        batch2 = dataloader2.dataset[event_idx]
        
        # Analyze both batches
        tracks1 = analyze_track_station_indices(batch1, "Dataset1")
        tracks2 = analyze_track_station_indices(batch2, "Dataset2")
        
        # Store comparison
        event_comparison = {
            'event_idx': event_idx,
            'dataset1_tracks': tracks1,
            'dataset2_tracks': tracks2,
            'num_tracks_dataset1': len(tracks1),
            'num_tracks_dataset2': len(tracks2)
        }
        
        comparison_results.append(event_comparison)
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 80)
    
    # Summary statistics
    total_tracks_d1 = sum(len(comp['dataset1_tracks']) for comp in comparison_results)
    total_tracks_d2 = sum(len(comp['dataset2_tracks']) for comp in comparison_results)
    
    print(f"Total tracks found:")
    print(f"  FILTERED: {total_tracks_d1}")
    print(f"  UNFILTERED: {total_tracks_d2}")
    print()
    
    # Detailed event-by-event comparison
    print("EVENT-BY-EVENT COMPARISON (first 20 events):")
    print("-" * 80)
    
    for i, comp in enumerate(comparison_results[:20]):
        event_idx = comp['event_idx']
        tracks1 = comp['dataset1_tracks']
        tracks2 = comp['dataset2_tracks']
        
        print(f"\nEvent {event_idx}:")
        print(f"  FILTERED: {len(tracks1)} tracks | UNFILTERED: {len(tracks2)} tracks")
        
        # Compare tracks if both exist
        max_tracks = max(len(tracks1), len(tracks2))
        
        for track_idx in range(max_tracks):
            track1 = tracks1[track_idx] if track_idx < len(tracks1) else None
            track2 = tracks2[track_idx] if track_idx < len(tracks2) else None
            
            print(f"    Track {track_idx}:")
            
            if track1 and track2:
                # Both tracks exist - detailed comparison
                print(f"      Dataset1 (FILTERED): {track1['num_hits']:2d} hits, {track1['num_unique_stations']:2d} stations, pt={track1['pt']:6.2f}")
                print(f"      Dataset2 (UNFILTERED): {track2['num_hits']:2d} hits, {track2['num_unique_stations']:2d} stations, pt={track2['pt']:6.2f}")
                print(f"      Station indices match: {np.array_equal(track1['station_indices'], track2['station_indices'])}")
                print(f"      Unique stations match: {np.array_equal(track1['unique_stations'], track2['unique_stations'])}")
                
                # Always show station indices for comparison
                print(f"      FILTERED   stations: {sorted(track1['unique_stations'])}")
                print(f"      UNFILTERED stations: {sorted(track2['unique_stations'])}")
                
                # Check baseline filtering criteria for each
                # Baseline: >=3 stations with >=3 hits each
                station_counts_1 = {}
                station_counts_2 = {}
                
                for station in track1['station_indices']:
                    station_counts_1[station] = station_counts_1.get(station, 0) + 1
                
                for station in track2['station_indices']:
                    station_counts_2[station] = station_counts_2.get(station, 0) + 1
                
                # Count stations with >=3 hits
                stations_with_3plus_hits_1 = sum(1 for count in station_counts_1.values() if count >= 3)
                stations_with_3plus_hits_2 = sum(1 for count in station_counts_2.values() if count >= 3)
                
                baseline_pass_1 = (track1['num_hits'] >= 9 and 
                                 abs(track1['eta']) >= 0.1 and abs(track1['eta']) <= 2.7 and
                                 track1['pt'] >= 3.0 and
                                 len(track1['unique_stations']) >= 3 and
                                 stations_with_3plus_hits_1 >= 3)
                
                baseline_pass_2 = (track2['num_hits'] >= 9 and 
                                 abs(track2['eta']) >= 0.1 and abs(track2['eta']) <= 2.7 and
                                 track2['pt'] >= 3.0 and
                                 len(track2['unique_stations']) >= 3 and
                                 stations_with_3plus_hits_2 >= 3)
                
                print(f"      FILTERED   baseline pass: {baseline_pass_1} (stations with >=3 hits: {stations_with_3plus_hits_1})")
                print(f"      UNFILTERED baseline pass: {baseline_pass_2} (stations with >=3 hits: {stations_with_3plus_hits_2})")
                
                if baseline_pass_1 != baseline_pass_2:
                    print(f"      *** BASELINE FILTERING DIFFERENCE! ***")
                    print(f"          FILTERED hits per station: {dict(sorted(station_counts_1.items()))}")
                    print(f"          UNFILTERED hits per station: {dict(sorted(station_counts_2.items()))}")
                
                print()  # Extra spacing for readability
            
            elif track1:
                # Only dataset1 has this track
                print(f"      FILTERED: {track1['num_hits']:2d} hits, {track1['num_unique_stations']:2d} stations, pt={track1['pt']:6.2f}")
                print(f"      UNFILTERED: NO TRACK")
                print(f"      FILTERED stations: {sorted(track1['unique_stations'])}")
                
            elif track2:
                # Only dataset2 has this track
                print(f"      FILTERED: NO TRACK")
                print(f"      UNFILTERED: {track2['num_hits']:2d} hits, {track2['num_unique_stations']:2d} stations, pt={track2['pt']:6.2f}")
                print(f"      UNFILTERED stations: {sorted(track2['unique_stations'])}")
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    
    # Collect all track data
    all_tracks_d1 = []
    all_tracks_d2 = []
    
    for comp in comparison_results:
        all_tracks_d1.extend(comp['dataset1_tracks'])
        all_tracks_d2.extend(comp['dataset2_tracks'])
    
    print(f"Total tracks analyzed:")
    print(f"  FILTERED: {len(all_tracks_d1)}")
    print(f"  UNFILTERED: {len(all_tracks_d2)}")
    print()
    
    if all_tracks_d1:
        hits_d1 = [track['num_hits'] for track in all_tracks_d1]
        stations_d1 = [track['num_unique_stations'] for track in all_tracks_d1]
        
        print(f"FILTERED statistics:")
        print(f"  Hits per track: mean={np.mean(hits_d1):.1f}, std={np.std(hits_d1):.1f}, range=[{min(hits_d1)}, {max(hits_d1)}]")
        print(f"  Stations per track: mean={np.mean(stations_d1):.1f}, std={np.std(stations_d1):.1f}, range=[{min(stations_d1)}, {max(stations_d1)}]")
    
    if all_tracks_d2:
        hits_d2 = [track['num_hits'] for track in all_tracks_d2]
        stations_d2 = [track['num_unique_stations'] for track in all_tracks_d2]
        
        print(f"UNFILTERED statistics:")
        print(f"  Hits per track: mean={np.mean(hits_d2):.1f}, std={np.std(hits_d2):.1f}, range=[{min(hits_d2)}, {max(hits_d2)}]")
        print(f"  Stations per track: mean={np.mean(stations_d2):.1f}, std={np.std(stations_d2):.1f}, range=[{min(stations_d2)}, {max(stations_d2)}]")
    
    # Event matching analysis
    print("\n" + "=" * 80)
    print("EVENT MATCHING ANALYSIS")
    print("=" * 80)
    
    events_with_same_track_count = 0
    events_with_different_track_count = 0
    perfect_matches = 0
    
    for comp in comparison_results:
        if comp['num_tracks_dataset1'] == comp['num_tracks_dataset2']:
            events_with_same_track_count += 1
            
            # Check if tracks are identical
            if comp['num_tracks_dataset1'] > 0:
                tracks_match = True
                for i in range(comp['num_tracks_dataset1']):
                    track1 = comp['dataset1_tracks'][i]
                    track2 = comp['dataset2_tracks'][i]
                    if not np.array_equal(track1['station_indices'], track2['station_indices']):
                        tracks_match = False
                        break
                
                if tracks_match:
                    perfect_matches += 1
        else:
            events_with_different_track_count += 1
    
    print(f"Events with same track count: {events_with_same_track_count}/{len(comparison_results)} ({events_with_same_track_count/len(comparison_results)*100:.1f}%)")
    print(f"Events with different track count: {events_with_different_track_count}/{len(comparison_results)} ({events_with_different_track_count/len(comparison_results)*100:.1f}%)")
    print(f"Perfect matches (same tracks, same stations): {perfect_matches}/{len(comparison_results)} ({perfect_matches/len(comparison_results)*100:.1f}%)")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()