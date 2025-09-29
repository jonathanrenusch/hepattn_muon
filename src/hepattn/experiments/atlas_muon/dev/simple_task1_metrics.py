#!/usr/bin/env python3
"""
Simple script to calculate Task 1 metrics without plots.
Calculates double matching efficiency, fake rate, and efficiency using:
- track_hit_valid from predictions 
- particle_hit_valid from targets
- particle_valid from targets  
- track_valid from predictions
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

class SimpleTask1Evaluator:
    """Simple evaluator for key Task 1 metrics."""
    
    def __init__(self, eval_path, data_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        
        print(f"Simple Task 1 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Max events: {max_events}")
        
    def setup_data_module(self):
        """Setup the data module for loading truth information."""
        print("Setting up data module...")
        
        self.data_module = AtlasMuonDataModule(
            train_dir=self.data_dir,
            val_dir=self.data_dir,
            test_dir=self.data_dir,
            num_workers=100,
            num_train=1,
            num_val=1,
            num_test=self.max_events if self.max_events else -1,
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
        
        self.data_module.setup("test")
        self.test_dataloader = self.data_module.test_dataloader()
        print(f"Data module setup complete. Test dataset size: {len(self.test_dataloader.dataset)}")
        
    def calculate_metrics(self):
        """Calculate metrics on-the-fly without storing all data."""
        print("Calculating metrics...")
        
        # Metrics storage
        track_double_matches = []  # Per track: 1 if both eff>=50% and pur>=50%, 0 otherwise
        track_efficiencies = []    # Per track efficiency
        track_purities = []        # Per track purity
        fake_tracks = []           # Per track: 1 if predicted valid but no true particle, 0 otherwise
        true_tracks = []           # Per track: 1 if true particle exists, 0 otherwise
        predicted_tracks = []      # Per track: 1 if predicted as valid track, 0 otherwise
        
        # Baseline filtering statistics
        baseline_stats = {
            'total_tracks_checked': 0,
            'tracks_failed_min_hits': 0,
            'tracks_failed_eta_cuts': 0,
            'tracks_failed_pt_cuts': 0,
            'tracks_failed_station_cuts': 0,
            'tracks_passed_all_cuts': 0
        }
        
        # Baseline track metrics (for tracks that pass all baseline cuts)
        baseline_track_double_matches = []
        baseline_track_efficiencies = []
        baseline_track_purities = []
        
        # Event-level tracking for baseline fake rate calculation
        events_with_baseline_tracks = []  # Track which events have at least one baseline track
        baseline_event_track_predictions = []  # All track predictions from events with baseline tracks
        baseline_event_track_truth = []        # All track truth from events with baseline tracks
        
        with h5py.File(self.eval_path, 'r') as pred_file:
            event_count = 0
            
            for batch in tqdm(self.test_dataloader, desc="Processing events"):
                if self.max_events and event_count >= self.max_events:
                    break
                    
                event_id = str(event_count)
                
                if event_id not in pred_file:
                    print(f"Warning: Event {event_id} not found in predictions file")
                    event_count += 1
                    continue
                
                # Get truth information from batch
                inputs, targets = batch
                pred_group = pred_file[event_id]
                
                # Get predictions
                hit_track_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                
                # Get truth
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
                
                # Track whether this event has any baseline tracks
                event_has_baseline_track = False
                event_track_predictions = []  # Track predictions for this event (both tracks)
                event_track_truth = []        # Track truth for this event (both tracks)
                
                # Process both potential tracks (max 2 tracks per event)
                for track_idx in range(2):
                    # Get track validity predictions and truth
                    predicted_track_valid = bool(track_valid_pred[0, track_idx])

                    true_particle_exists = bool(true_particle_valid[track_idx])
                    
                    # Store event-level track data
                    event_track_predictions.append(1 if predicted_track_valid else 0)
                    event_track_truth.append(1 if true_particle_exists else 0)
                    
                    predicted_tracks.append(1 if predicted_track_valid else 0)
                    true_tracks.append(1 if true_particle_exists else 0)
                    
                    # Calculate fake rate: predicted valid but no true particle
                    is_fake = predicted_track_valid and not true_particle_exists
                    fake_tracks.append(1 if is_fake else 0)
                    
                    # For hit assignment metrics, only consider cases where we have both predictions and truth
                    if true_particle_exists:
                        baseline_stats['total_tracks_checked'] += 1
                        
                        # Get hit assignments
                        pred_hits = hit_track_pred[0, track_idx]  # Shape: (num_hits,)
                        true_hits = true_hit_assignments[track_idx].numpy()  # Shape: (num_hits,)
                        
                        # Handle shape mismatch
                        # min_hits = min(pred_hits.shape[0], true_hits.shape[0])
                        # if pred_hits.shape[0] != true_hits.shape[0]:
                        #     pred_hits = pred_hits[:min_hits]
                        #     true_hits = true_hits[:min_hits]
                        
                        # Convert to boolean arrays
                        pred_hits = pred_hits.astype(bool)
                        true_hits = true_hits.astype(bool)
                        
                        # Apply baseline filtering criteria
                        passes_baseline = True
                        
                        # Pre-filter 1: tracks must have at least 9 hits total
                        total_true_hits = true_hits.sum()
                        if total_true_hits < 9:
                            baseline_stats['tracks_failed_min_hits'] += 1
                            passes_baseline = False
                        
                        # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7
                        if passes_baseline:
                            track_eta = targets["particle_truthMuon_eta"][0, track_idx].item()
                            if np.abs(track_eta) < 0.1 or np.abs(track_eta) > 2.7:
                                baseline_stats['tracks_failed_eta_cuts'] += 1
                                passes_baseline = False
                        
                        # Pre-filter 3: pt threshold >= 3.0 GeV
                        if passes_baseline:
                            track_pt = targets["particle_truthMuon_pt"][0, track_idx].item()
                            if track_pt < 5.0:
                                baseline_stats['tracks_failed_pt_cuts'] += 1
                                passes_baseline = False
                        
                        # Pre-filter 4: station requirements (at least 3 different stations, 3 stations with >=3 hits each)
                        if passes_baseline:
                            # Get station indices for this track
                            true_station_index = inputs["hit_spacePoint_stationIndex"][0]  # Shape: (num_hits,)
                            
                            # # Handle shape mismatch for station indices  
                            # min_station_hits = min(true_station_index.shape[0], true_hits.shape[0])
                            # if true_station_index.shape[0] != true_hits.shape[0]:
                            #     true_station_index = true_station_index[:min_station_hits]
                            #     # Also need to truncate true_hits for station analysis
                            #     track_true_hits = true_hits[:min_station_hits]
                            # else:
                            #     track_true_hits = true_hits
                            
                            # Get station indices for hits belonging to this track
                            track_stations = true_station_index
                            unique_stations, station_counts = np.unique(track_stations, return_counts=True)
                            
                            # Check station requirements:
                            # 1. At least 3 different stations
                            if len(unique_stations) < 3:
                                baseline_stats['tracks_failed_station_cuts'] += 1
                                passes_baseline = False
                            else:
                                # 2. At least 3 stations must have at least 3 hits each
                                n_good_stations = np.sum(station_counts >= 3)
                                if n_good_stations < 3:
                                    baseline_stats['tracks_failed_station_cuts'] += 1
                                    passes_baseline = False
                        
                        if passes_baseline:
                            baseline_stats['tracks_passed_all_cuts'] += 1
                            event_has_baseline_track = True  # Mark this event as having a baseline track
                            
                            # Note: Individual baseline track-level metrics are not tracked here
                            # We'll calculate them at the event level later
                        
                        # Calculate track-level efficiency and purity
                        total_pred_hits = pred_hits.sum()
                        total_correct_hits = (pred_hits & true_hits).sum()
                        
                        # Efficiency: correctly predicted hits / total true hits
                        if total_true_hits > 0:
                            efficiency = total_correct_hits / total_true_hits
                        else:
                            efficiency = 0.0
                            
                        # Purity: correctly predicted hits / total predicted hits  
                        if total_pred_hits > 0:
                            purity = total_correct_hits / total_pred_hits
                        else:
                            purity = 0.0
                        
                        track_efficiencies.append(efficiency)
                        track_purities.append(purity)
                        
                        # Double matching: both efficiency >= 50% AND purity >= 50%
                        double_match = 1 if (efficiency >= 0.5 and purity >= 0.5) else 0
                        track_double_matches.append(double_match)
                        
                        # Store baseline track metrics separately
                        if passes_baseline:
                            baseline_track_efficiencies.append(efficiency)
                            baseline_track_purities.append(purity)
                            baseline_track_double_matches.append(double_match)
                
                # After processing both tracks in this event, check if any track passed baseline
                if event_has_baseline_track:
                    # Store all track predictions/truth from this event for baseline fake rate calculation
                    baseline_event_track_predictions.extend(event_track_predictions)
                    baseline_event_track_truth.extend(event_track_truth)
                
                event_count += 1
        
        # Calculate average metrics
        avg_efficiency = np.mean(track_efficiencies) if track_efficiencies else 0.0
        avg_purity = np.mean(track_purities) if track_purities else 0.0
        avg_double_matching = np.mean(track_double_matches) if track_double_matches else 0.0
        
        # Calculate baseline track metrics
        baseline_avg_efficiency = np.mean(baseline_track_efficiencies) if baseline_track_efficiencies else 0.0
        baseline_avg_purity = np.mean(baseline_track_purities) if baseline_track_purities else 0.0
        baseline_avg_double_matching = np.mean(baseline_track_double_matches) if baseline_track_double_matches else 0.0
        
        # Calculate fake rate: fake tracks / total predicted tracks  
        total_predicted = sum(predicted_tracks)
        total_fakes = sum(fake_tracks)
        fake_rate = total_fakes / total_predicted if total_predicted > 0 else 0.0
        
        # Calculate track finding efficiency: predicted valid tracks / true particles
        total_true_particles = sum(true_tracks)
        total_found = sum(1 for pred, true in zip(predicted_tracks, true_tracks) if pred and true)
        track_finding_efficiency = total_found / total_true_particles if total_true_particles > 0 else 0.0
        
        # Calculate baseline track-level metrics
        baseline_total_predicted = sum(baseline_event_track_predictions) if baseline_event_track_predictions else 0
        baseline_total_true_particles = sum(baseline_event_track_truth) if baseline_event_track_truth else 0
        
        # Calculate baseline fake rate: fakes among tracks from events with baseline tracks
        baseline_total_fakes = sum(1 for pred, true in zip(baseline_event_track_predictions, baseline_event_track_truth) 
                                  if pred and not true) if baseline_event_track_predictions and baseline_event_track_truth else 0
        baseline_fake_rate = baseline_total_fakes / baseline_total_predicted if baseline_total_predicted > 0 else 0.0
        
        # Calculate baseline track finding efficiency: found tracks among tracks from events with baseline tracks  
        baseline_total_found = sum(1 for pred, true in zip(baseline_event_track_predictions, baseline_event_track_truth) 
                                  if pred and true) if baseline_event_track_predictions and baseline_event_track_truth else 0
        baseline_track_finding_efficiency = baseline_total_found / baseline_total_true_particles if baseline_total_true_particles > 0 else 0.0
        
        # Print results
        print("\n" + "="*70)
        print("TASK 1 SIMPLE METRICS RESULTS")
        print("="*70)
        print(f"Total events processed: {event_count}")
        print(f"Total tracks analyzed: {len(track_efficiencies)}")
        print(f"Total predicted tracks: {total_predicted}")
        print(f"Total true particles: {total_true_particles}")
        print(f"Total fake tracks: {total_fakes}")
        print()
        
        # Print baseline filtering statistics
        print("BASELINE FILTERING STATISTICS:")
        print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
        print(f"  Failed minimum hits (>=9): {baseline_stats['tracks_failed_min_hits']} ({baseline_stats['tracks_failed_min_hits']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats['tracks_failed_eta_cuts']} ({baseline_stats['tracks_failed_eta_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed pt cuts (pt >= 3.0 GeV): {baseline_stats['tracks_failed_pt_cuts']} ({baseline_stats['tracks_failed_pt_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed station cuts (>=3 stations, 3 stations with >=3 hits): {baseline_stats['tracks_failed_station_cuts']} ({baseline_stats['tracks_failed_station_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Tracks passing all cuts: {baseline_stats['tracks_passed_all_cuts']} ({baseline_stats['tracks_passed_all_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print()
        
        print("ALL TRACKS - HIT ASSIGNMENT METRICS:")
        print(f"  Average hit assignment efficiency: {avg_efficiency:.4f}")
        print(f"  Average hit assignment purity: {avg_purity:.4f}")
        print(f"  Average double matching efficiency: {avg_double_matching:.4f}")
        print()
        
        print("BASELINE TRACKS ONLY - HIT ASSIGNMENT METRICS:")
        print(f"  Number of baseline tracks: {len(baseline_track_efficiencies)}")
        print(f"  Average hit assignment efficiency: {baseline_avg_efficiency:.4f}")
        print(f"  Average hit assignment purity: {baseline_avg_purity:.4f}")
        print(f"  Average double matching efficiency: {baseline_avg_double_matching:.4f}")
        print()
        
        print("ALL TRACKS - TRACK-LEVEL METRICS:")
        print(f"  Fake rate: {fake_rate:.4f} ({total_fakes}/{total_predicted})")
        print(f"  Track finding efficiency: {track_finding_efficiency:.4f} ({total_found}/{total_true_particles})")
        print()
        
        print("BASELINE TRACKS ONLY - TRACK-LEVEL METRICS:")
        print(f"  Number of baseline tracks considered: {baseline_total_true_particles}")
        print(f"  Baseline fake rate: {baseline_fake_rate:.4f} ({baseline_total_fakes}/{baseline_total_predicted})")
        print(f"  Baseline track finding efficiency: {baseline_track_finding_efficiency:.4f} ({baseline_total_found}/{baseline_total_true_particles})")
        print()
        
        # Additional statistics
        print("ALL TRACKS - ADDITIONAL STATISTICS:")
        if track_efficiencies:
            print(f"  Hit efficiency std: {np.std(track_efficiencies):.4f}")
            print(f"  Hit efficiency min/max: {np.min(track_efficiencies):.4f} / {np.max(track_efficiencies):.4f}")
        if track_purities:
            print(f"  Hit purity std: {np.std(track_purities):.4f}")
            print(f"  Hit purity min/max: {np.min(track_purities):.4f} / {np.max(track_purities):.4f}")
        print()
        
        print("BASELINE TRACKS - ADDITIONAL STATISTICS:")
        if baseline_track_efficiencies:
            print(f"  Hit efficiency std: {np.std(baseline_track_efficiencies):.4f}")
            print(f"  Hit efficiency min/max: {np.min(baseline_track_efficiencies):.4f} / {np.max(baseline_track_efficiencies):.4f}")
        if baseline_track_purities:
            print(f"  Hit purity std: {np.std(baseline_track_purities):.4f}")
            print(f"  Hit purity min/max: {np.min(baseline_track_purities):.4f} / {np.max(baseline_track_purities):.4f}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Simple Task 1 metrics calculation')
    parser.add_argument('--eval_path', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                       help='Path to processed test data directory')
    parser.add_argument('--max_events', "-m", type=int, default=10000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate paths
    eval_path = Path(args.eval_path)
    data_dir = Path(args.data_dir)
    
    if not eval_path.exists():
        print(f"Error: Evaluation file does not exist: {eval_path}")
        return
        
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    # Create evaluator and run
    evaluator = SimpleTask1Evaluator(
        eval_path=str(eval_path),
        data_dir=str(data_dir), 
        max_events=args.max_events
    )
    
    evaluator.setup_data_module()
    evaluator.calculate_metrics()


if __name__ == "__main__":
    main()