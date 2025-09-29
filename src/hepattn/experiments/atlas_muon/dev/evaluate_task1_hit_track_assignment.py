#!/usr/bin/env python3
"""
Evaluation script for Task 1: Hit-Track Assignment (track_hit_valid)

This script evaluates the performance of the hit-track assignment task by:
1. Creating efficiency and purity plots over pt, eta, phi (using true values)
2. Creating ROC curves using the logits

The evaluation uses filtering functionality similar to the hit filter evaluation
to allow comparison with baseline performance in different detector regions.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import argparse
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial
import traceback

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True, 
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'errorbar.capsize': 3
})

class Task1HitTrackEvaluator:
    """Evaluator for hit-track assignment task."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.config_path = config_path
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task1_evaluation_{timestamp}"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.predictions = []
        self.logits = []
        self.true_assignments = []
        self.track_info = []  # pt, eta, phi for each track
        self.hit_info = []    # hit information
        self.station_indices = []  # station index for each track
        
        print(f"Task 1 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {timestamp}")
        
    def setup_data_module(self):
        """Setup the data module for loading truth information."""
        print("Setting up data module...")
        
        # Create a minimal config for the data module
        self.data_module = AtlasMuonDataModule(
            train_dir=self.data_dir,
            val_dir=self.data_dir,
            test_dir=self.data_dir,
            num_workers=100,
            num_train=1,  # Set to 1 instead of 0 to avoid error
            num_val=1,    # Set to 1 instead of 0 to avoid error
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
        
        # Only setup test stage to avoid creating unnecessary train/val datasets
        self.data_module.setup("test")
        self.test_dataloader = self.data_module.test_dataloader()
        print(f"Data module setup complete. Test dataset size: {len(self.test_dataloader.dataset)}")
        
    def collect_data(self):
        """Collect predictions and truth data."""
        print("Collecting data from predictions and truth...")
        
        with h5py.File(self.eval_path, 'r') as pred_file:
            event_count = 0
            
            for batch in tqdm(self.test_dataloader, desc="Processing events"):
                if self.max_events and event_count >= self.max_events:
                    break
                    
                # Get event ID (assuming sequential)
                event_id = str(event_count)
                
                if event_id not in pred_file:
                    print(f"Warning: Event {event_id} not found in predictions file")
                    event_count += 1
                    continue
                
                # Get truth information from batch
                inputs, targets = batch

                true_station_index = inputs["hit_spacePoint_stationIndex"][0]        
                # Store the data
                
                self.station_indices.append(true_station_index)
                # Get predictions and logits
                pred_group = pred_file[event_id]
                
                # Hit-track assignment predictions and logits
                hit_track_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                hit_track_logits = pred_group['outputs/final/track_hit_valid/track_hit_logit'][...]  # Shape: (1, 2, num_hits)
                
                # Track validity predictions  
                track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                
                # Regression predictions (for track parameters)
                # track_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                # track_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                # track_qpt = pred_group['preds/final/parameter_regression/track_truthMuon_qpt'][...]  # Shape: (1, 2)
                
                
                
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
                
                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.numpy()
                num_valid = valid_particles.sum()
                # print(num_valid)
                
                if num_valid == 0:
                    event_count += 1
                    continue
                
                for track_idx in range(num_valid):
                    # Only process if BOTH model predicted a valid track AND there's a true valid particle
                    # if true_particle_valid[track_idx]:
                        
                    true_hits = true_hit_assignments[track_idx].numpy()  # Shape: (num_hits_truth,)
                    pred_hits = hit_track_pred[0, track_idx]  # Shape: (num_hits_pred,)
                    logit_hits = hit_track_logits[0, track_idx]  # Shape: (num_hits_pred,)
                    
                    # Handle shape mismatch by taking minimum length
                    min_hits = min(pred_hits.shape[0], true_hits.shape[0])
                    if pred_hits.shape[0] != true_hits.shape[0]:
                        print(f"Warning: Shape mismatch in event {event_count}, track {track_idx}: pred_hits {pred_hits.shape} vs true_hits {true_hits.shape}, using first {min_hits} hits")
                        pred_hits = pred_hits[:min_hits]
                        true_hits = true_hits[:min_hits]
                        logit_hits = logit_hits[:min_hits]
                    
                    # Get true track parameters
                    true_eta = targets["particle_truthMuon_eta"][0, track_idx]
                    true_phi = targets["particle_truthMuon_phi"][0, track_idx]
                    true_pt = targets["particle_truthMuon_pt"][0, track_idx]
                    
                    self.predictions.append(pred_hits)
                    self.logits.append(logit_hits)
                    self.true_assignments.append(true_hits)
                    self.track_info.append({
                        'pt': true_pt,
                            'eta': true_eta, 
                            'phi': true_phi,
                            'event_id': event_count,
                            'track_id': track_idx
                        })
                
                event_count += 1
                
        print(f"Collected data for {len(self.predictions)} tracks from {event_count} events")
        
    def create_baseline_track_filter(self):
        """
        Create filter masks for baseline evaluation that includes:
        - All tracks meeting baseline requirements (baseline category)
        - All tracks NOT meeting baseline requirements (rejected category)
        
        Baseline requirements:
          * |eta| >= 0.1 and |eta| <= 2.7 (detector acceptance region)
          * pt >= 3 GeV (minimum pt threshold)
          * At least 9 total hits (inferred from hit assignment arrays)
        
        This approach allows comparison between high-quality and low-quality tracks
        in the hit-track assignment task.
        
        Returns:
            baseline_mask: Boolean array for tracks in baseline evaluation
            rejected_mask: Boolean array for tracks in rejected evaluation
            stats: Dictionary with detailed filtering statistics
        """
        print("Creating baseline track filter for Task 1 (eta cuts, pt cuts, hit counts)...")
        print("Strategy: Separate tracks into baseline-qualified vs rejected categories")
        
        if len(self.track_info) == 0:
            print("Warning: No track data available for filtering")
            return np.array([]), np.array([]), {}
        
        # Track statistics for detailed reporting
        stats = {
            'total_tracks_checked': 0,
            'tracks_failed_min_hits': 0,
            'tracks_failed_eta_cuts': 0,
            'tracks_failed_pt_cuts': 0,
            'tracks_failed_station_cuts': 0,  # Not used in Task 1, but kept for consistency
            'tracks_passed_all_cuts': 0
        }
        
        # Process all tracks
        print(f"Processing {len(self.track_info)} tracks...")
        
        baseline_qualified_tracks = set()
        
        for track_idx in tqdm(range(len(self.track_info)), desc="Processing tracks"):
            stats['total_tracks_checked'] += 1
            
            track_info = self.track_info[track_idx]
            
            # Get track kinematic properties
            track_pt = track_info['pt']
            track_eta = track_info['eta']
            
            # Get hit information for this track
            hit_mask = self.true_assignments[track_idx]
            # pred_hits = self.predictions[track_idx]
            
            # Pre-filter 1: tracks must have at least 9 hits total
            total_hits = np.sum(hit_mask)
            if total_hits < 9:
                stats['tracks_failed_min_hits'] += 1
                continue
            
            # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7
            if np.abs(track_eta) < 0.1 or np.abs(track_eta) > 2.7:
                stats['tracks_failed_eta_cuts'] += 1
                continue
                
            # Pre-filter 3: pt threshold >= 3 GeV
            if track_pt < 3.0:
                stats['tracks_failed_pt_cuts'] += 1
                continue
            
            unique_stations, station_counts = np.unique(self.station_indices[track_idx], return_counts=True)
            # Check station requirements:
            # 1. At least 3 different stations
            if len(unique_stations) < 3:
                stats['tracks_failed_station_cuts'] += 1
                continue
                
            # 2. Each station must have at least 3 hits
            # if not np.all(station_counts >= 3):
            #     stats['tracks_failed_station_cuts'] += 1
            #     continue

            # 2. Each station must have at least 3 hits
            n_good_stations = np.sum(station_counts >= 3)
            if n_good_stations < 3:
                stats['tracks_failed_station_cuts'] += 1

                continue
            
            # Track passed all criteria
            baseline_qualified_tracks.add(track_idx)
            stats['tracks_passed_all_cuts'] += 1
        
        # Print detailed statistics
        print(f"Baseline filtering results:")
        print(f"  Total tracks checked: {stats['total_tracks_checked']}")
        print(f"  Failed minimum hits (>=9): {stats['tracks_failed_min_hits']} ({stats['tracks_failed_min_hits']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {stats['tracks_failed_eta_cuts']} ({stats['tracks_failed_eta_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed pt cuts (pt >= 3 GeV): {stats['tracks_failed_pt_cuts']} ({stats['tracks_failed_pt_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed station cuts (at least 3 stations and three stations with three hits): {stats['tracks_failed_station_cuts']} ({stats['tracks_failed_station_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Tracks passing all cuts: {stats['tracks_passed_all_cuts']} ({stats['tracks_passed_all_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        
        # Create masks for tracks to include in baseline and rejected evaluations
        total_tracks = len(self.track_info)
        baseline_mask = np.zeros(total_tracks, dtype=bool)
        rejected_mask = np.zeros(total_tracks, dtype=bool)
        
        # Create sets for baseline and rejected tracks
        all_track_indices = set(range(total_tracks))
        rejected_track_set = all_track_indices - baseline_qualified_tracks
        
        print(f"Creating masks for {len(baseline_qualified_tracks)} baseline tracks and {len(rejected_track_set)} rejected tracks...")
        
        # Set the appropriate masks
        for track_idx in baseline_qualified_tracks:
            baseline_mask[track_idx] = True
            
        for track_idx in rejected_track_set:
            rejected_mask[track_idx] = True
        
        # Calculate statistics for both categories
        baseline_track_count = np.sum(baseline_mask)
        rejected_track_count = np.sum(rejected_mask)
        
        print(f"  Baseline tracks: {baseline_track_count:,} / {total_tracks:,} ({baseline_track_count/total_tracks*100:.1f}%)")
        print(f"  Rejected tracks: {rejected_track_count:,} / {total_tracks:,} ({rejected_track_count/total_tracks*100:.1f}%)")
        
        # Additional statistics for the summary
        stats['baseline_track_count'] = baseline_track_count
        stats['rejected_track_count'] = rejected_track_count
        stats['total_tracks'] = total_tracks
        stats['rejected_tracks'] = len(rejected_track_set)
        
        return baseline_mask, rejected_mask, stats
    
    def _backup_original_data(self):
        """Backup the original data before applying any filters."""
        self._original_predictions = [pred.copy() for pred in self.predictions]
        self._original_logits = [logit.copy() for logit in self.logits]
        self._original_true_assignments = [truth.copy() for truth in self.true_assignments]
        self._original_track_info = [track.copy() for track in self.track_info]
        self._original_hit_info = [hit.copy() for hit in self.hit_info] if self.hit_info else []
    
    def _restore_original_data(self):
        """Restore the original unfiltered data."""
        self.predictions = [pred.copy() for pred in self._original_predictions]
        self.logits = [logit.copy() for logit in self._original_logits]
        self.true_assignments = [truth.copy() for truth in self._original_true_assignments]
        self.track_info = [track.copy() for track in self._original_track_info]
        self.hit_info = [hit.copy() for hit in self._original_hit_info] if hasattr(self, '_original_hit_info') else []
    
    def _apply_track_filter(self, track_mask):
        """Apply a boolean mask to filter the track data arrays."""
        filtered_indices = np.where(track_mask)[0]
        self.predictions = [self.predictions[i] for i in filtered_indices]
        self.logits = [self.logits[i] for i in filtered_indices]
        self.true_assignments = [self.true_assignments[i] for i in filtered_indices]
        self.track_info = [self.track_info[i] for i in filtered_indices]
        if self.hit_info:
            self.hit_info = [self.hit_info[i] for i in filtered_indices]
    
    def run_evaluation_with_categories(self):
        """Run evaluation for all tracks, baseline tracks, and rejected tracks."""
        print("=" * 80)
        print("TASK 1: HIT-TRACK ASSIGNMENT EVALUATION WITH CATEGORIES")
        print("=" * 80)
        
        # Setup and collect data ONCE (efficiency optimization)
        print("PHASE 1: Data Collection (done once for all categories)")
        print("-" * 50)
        self.setup_data_module()
        self.collect_data()
        
        if len(self.predictions) == 0:
            print("Error: No data collected. Check file paths and data format.")
            return
            
        print(f"✓ Data collection complete: {len(self.predictions)} tracks loaded")
        print("✓ Using backup/restore pattern to avoid reloading data for each category")
        
        # Backup original data
        self._backup_original_data()
        
        # Create baseline filter
        baseline_mask, rejected_mask, filter_stats = self.create_baseline_track_filter()
        
        # Create subdirectories for each category
        all_tracks_dir = self.output_dir / "all_tracks"
        baseline_dir = self.output_dir / "baseline_tracks"
        rejected_dir = self.output_dir / "rejected_tracks"
        
        for subdir in [all_tracks_dir, baseline_dir, rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreated subdirectories:")
        print(f"  All tracks: {all_tracks_dir}")
        print(f"  Baseline tracks: {baseline_dir}")
        print(f"  Rejected tracks: {rejected_dir}")
        
        # Dictionary to store results from each category
        results = {}
        
        # 1. Evaluate all tracks
        print("\n" + "="*50)
        print("EVALUATING ALL TRACKS")
        print("="*50)
        self._restore_original_data()
        results['all_tracks'] = self._run_single_evaluation("all_tracks", all_tracks_dir)
        
        # 2. Evaluate baseline tracks
        print("\n" + "="*50)
        print("EVALUATING BASELINE TRACKS")
        print("="*50)
        self._restore_original_data()
        self._apply_track_filter(baseline_mask)
        results['baseline_tracks'] = self._run_single_evaluation("baseline_tracks", baseline_dir)
        
        # 3. Evaluate rejected tracks
        print("\n" + "="*50)
        print("EVALUATING REJECTED TRACKS")
        print("="*50)
        self._restore_original_data()
        self._apply_track_filter(rejected_mask)
        results['rejected_tracks'] = self._run_single_evaluation("rejected_tracks", rejected_dir)
        
        # Write comprehensive summary
        self._write_comparative_summary(results, filter_stats)
        
        print(f"\nTask 1 evaluation with categories complete. Results saved to {self.output_dir}")
        print(f"Summary files:")
        print(f"  Comparative summary: {self.output_dir / 'task1_comparative_summary.txt'}")
        print(f"  Individual summaries in each subdirectory")
    
    def _run_single_evaluation(self, category_name, output_subdir):
        """Run evaluation for a single category of tracks."""
        if len(self.predictions) == 0:
            print(f"Warning: No tracks in {category_name} category")
            return {}
        
        print(f"Processing {len(self.predictions)} tracks in {category_name} category...")
        print(f"Saving results to: {output_subdir}")
        
        # Create plots
        print("Generating plots...")
        
        # Efficiency/purity vs kinematic variables
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_efficiency_purity_vs_variable(variable, output_subdir)
            except Exception as e:
                print(f"Error creating efficiency/purity vs {variable} plot: {e}")
        
        # Individual working point efficiency/purity plots (combined)
        working_points = [0.5, 0.80, 0.85, 0.95]
        for variable in ['pt', 'eta', 'phi']:
            for wp in working_points:
                try:
                    self.plot_working_point_efficiency_purity_vs_variable(variable, wp, output_subdir)
                except Exception as e:
                    print(f"Error creating efficiency/purity plot for WP {wp:.2f} vs {variable}: {e}")
        
        # Double matching efficiency vs kinematic variables (original - 50% thresholds)
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_double_matching_efficiency_vs_variable(variable, output_subdir)
            except Exception as e:
                print(f"Error creating {variable} double matching efficiency plots: {e}")
        
        # Double matching efficiency with working points
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_double_matching_efficiency_vs_variable_with_working_points(variable, working_points, output_subdir)
            except Exception as e:
                print(f"Error creating {variable} double matching efficiency with working points plots: {e}")
        
        # ROC curve
        try:
            roc_auc = self.plot_roc_curve(output_subdir)
        except Exception as e:
            print(f"Error creating ROC curve: {e}")
            roc_auc = None
        
        # Calculate overall metrics
        try:
            overall_efficiency, overall_purity = self.calculate_overall_efficiency_purity()
            overall_double_matching = self.calculate_overall_double_matching_efficiency()
            
            print(f"\n{category_name.upper()} METRICS:")
            print(f"  Overall Hit Assignment Efficiency: {overall_efficiency:.4f}")
            print(f"  Overall Hit Assignment Purity: {overall_purity:.4f}")
            print(f"  Overall Double Matching Efficiency: {overall_double_matching:.4f}")
        except Exception as e:
            print(f"Error calculating overall metrics: {e}")
            overall_efficiency, overall_purity, overall_double_matching = None, None, None
        
        # Calculate working point metrics
        try:
            working_point_metrics = self.calculate_working_point_metrics([0.5, 0.80, 0.85, 0.95])
            print(f"\n{category_name.upper()} WORKING POINT METRICS:")
            for wp, metrics in working_point_metrics.items():
                if metrics['purity'] is not None:
                    print(f"  WP {wp:.2f}: threshold={metrics['threshold']:.4f}, "
                          f"achieved_eff={metrics['achieved_efficiency']:.4f}, "
                          f"purity={metrics['purity']:.4f}")
                else:
                    print(f"  WP {wp:.2f}: Could not achieve target efficiency")
        except Exception as e:
            print(f"Error calculating working point metrics: {e}")
            working_point_metrics = {}
        
        # Calculate average metrics by variable
        avg_metrics = {}
        for variable in ['pt', 'eta', 'phi']:
            try:
                avg_eff, avg_pur, avg_dm = self.calculate_average_metrics_by_variable(variable)
                avg_metrics[variable] = {
                    'avg_efficiency': avg_eff,
                    'avg_purity': avg_pur, 
                    'avg_double_matching': avg_dm
                }
                print(f"  Average {variable} efficiency: {avg_eff:.4f}, purity: {avg_pur:.4f}, double matching: {avg_dm:.4f}")
            except Exception as e:
                print(f"Error calculating average metrics for {variable}: {e}")
                avg_metrics[variable] = {'avg_efficiency': None, 'avg_purity': None, 'avg_double_matching': None}
        
        # Write individual summary for this category
        try:
            individual_summary_path = output_subdir / f'{category_name}_summary.txt'
            self._write_individual_summary(individual_summary_path, category_name, roc_auc, 
                                         overall_efficiency, overall_purity, overall_double_matching, 
                                         avg_metrics, working_point_metrics)
        except Exception as e:
            print(f"Error writing individual summary: {e}")
        
        return {
            'roc_auc': roc_auc,
            'overall_efficiency': overall_efficiency,
            'overall_purity': overall_purity,
            'overall_double_matching': overall_double_matching,
            'avg_metrics': avg_metrics,
            'working_point_metrics': working_point_metrics,
            'num_tracks': len(self.predictions)
        }
    
    def _write_comparative_summary(self, results, filter_stats):
        """Write comprehensive summary comparing all categories."""
        summary_path = self.output_dir / 'task1_comparative_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK 1: HIT-TRACK ASSIGNMENT EVALUATION - COMPARATIVE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Max events processed: {self.max_events}\n\n")
            
            # Write filtering statistics
            f.write("BASELINE FILTERING STATISTICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total tracks checked: {filter_stats.get('total_tracks_checked', 0):,}\n")
            f.write(f"Failed minimum hits (>=9): {filter_stats.get('tracks_failed_min_hits', 0):,}\n")
            f.write(f"Failed eta cuts (0.1 <= |eta| <= 2.7): {filter_stats.get('tracks_failed_eta_cuts', 0):,}\n")
            f.write(f"Failed pt cuts (pt >= 3 GeV): {filter_stats.get('tracks_failed_pt_cuts', 0):,}\n")
            f.write(f"Failed station cuts (at least 3 stations and 3 stations with 3+ hits): {filter_stats.get('tracks_failed_station_cuts', 0):,}\n")
            f.write(f"Tracks passing all cuts: {filter_stats.get('tracks_passed_all_cuts', 0):,}\n")
            
            # Calculate and display filtering percentages
            total_checked = filter_stats.get('total_tracks_checked', 0)
            if total_checked > 0:
                min_hits_pct = (filter_stats.get('tracks_failed_min_hits', 0) / total_checked) * 100
                eta_cuts_pct = (filter_stats.get('tracks_failed_eta_cuts', 0) / total_checked) * 100
                pt_cuts_pct = (filter_stats.get('tracks_failed_pt_cuts', 0) / total_checked) * 100
                station_cuts_pct = (filter_stats.get('tracks_failed_station_cuts', 0) / total_checked) * 100
                baseline_percentage = (filter_stats.get('tracks_passed_all_cuts', 0) / total_checked) * 100
                rejected_percentage = 100 - baseline_percentage
                
                f.write(f"\nDetailed Cut Statistics:\n")
                f.write(f"  Failed minimum hits: {min_hits_pct:.1f}%\n")
                f.write(f"  Failed eta cuts: {eta_cuts_pct:.1f}%\n")
                f.write(f"  Failed pt cuts: {pt_cuts_pct:.1f}%\n")
                f.write(f"  Failed station cuts: {station_cuts_pct:.1f}%\n")
                f.write(f"  Passed all cuts (baseline): {baseline_percentage:.1f}%\n")
                f.write(f"  Rejected (failed any cut): {rejected_percentage:.1f}%\n")
                
                f.write(f"\nBaseline Filtering Criteria:\n")
                f.write(f"  Minimum hits: >= 9 hits\n")
                f.write(f"  Eta acceptance: 0.1 <= |eta| <= 2.7\n")
                f.write(f"  Minimum pt: >= 3.0 GeV\n")
                f.write(f"  Station requirements: >= 3 stations with >= 3 stations having >= 3 hits each\n")
            f.write("\n")
            
            # Write track counts per category
            f.write("TRACKS ANALYZED PER CATEGORY\n")
            f.write("-" * 30 + "\n")
            for category in ['all_tracks', 'baseline_tracks', 'rejected_tracks']:
                if category in results:
                    num_tracks = results[category].get('num_tracks', 0)
                    f.write(f"{category.replace('_', ' ').title()}: {num_tracks:,} tracks\n")
                else:
                    f.write(f"{category.replace('_', ' ').title()}: 0 tracks\n")
            f.write("\n")
            
            # Write comparative metrics
            f.write("COMPARATIVE METRICS\n")
            f.write("-" * 20 + "\n")
            
            categories = ['all_tracks', 'baseline_tracks', 'rejected_tracks']
            metrics = ['roc_auc', 'overall_efficiency', 'overall_purity', 'overall_double_matching']
            
            # Header
            f.write(f"{'Category':<20}")
            for metric in metrics:
                f.write(f"{metric.replace('_', ' ').title():<25}")
            f.write(f"{'Num Tracks':<15}\n")
            f.write("-" * 120 + "\n")
            
            # Data rows
            for category in categories:
                if category in results:
                    result = results[category]
                    f.write(f"{category.replace('_', ' ').title():<20}")
                    for metric in metrics:
                        value = result.get(metric)
                        if value is not None:
                            f.write(f"{value:<25.4f}")
                        else:
                            f.write(f"{'N/A':<25}")
                    f.write(f"{result.get('num_tracks', 0):<15}\n")
                else:
                    f.write(f"{category.replace('_', ' ').title():<20}")
                    for metric in metrics:
                        f.write(f"{'N/A':<25}")
                    f.write(f"{'0':<15}\n")
            
            f.write("\n")
            
            # Working point metrics comparison
            f.write("WORKING POINT METRICS COMPARISON\n")
            f.write("-" * 35 + "\n")

            working_points = [0.5, 0.80, 0.85, 0.95]
            for wp in working_points:
                f.write(f"\nWorking Point {wp:.2f}:\n")
                f.write(f"{'Category':<20}{'Threshold':<15}{'Achieved Eff':<15}{'Purity':<15}\n")
                f.write("-" * 65 + "\n")
                
                for category in categories:
                    if category in results:
                        wp_metrics = results[category].get('working_point_metrics', {})
                        if wp in wp_metrics and wp_metrics[wp]['purity'] is not None:
                            threshold = wp_metrics[wp]['threshold']
                            achieved_eff = wp_metrics[wp]['achieved_efficiency']
                            purity = wp_metrics[wp]['purity']
                            f.write(f"{category.replace('_', ' ').title():<20}{threshold:<15.4f}{achieved_eff:<15.4f}{purity:<15.4f}\n")
                        else:
                            f.write(f"{category.replace('_', ' ').title():<20}{'N/A':<15}{'N/A':<15}{'N/A':<15}\n")
            
            f.write("\n")
            
            # Detailed metrics by variable
            for category in categories:
                if category not in results:
                    continue
                    
                f.write(f"{category.replace('_', ' ').upper()} - DETAILED METRICS BY VARIABLE\n")
                f.write("-" * 50 + "\n")
                
                result = results[category]
                avg_metrics = result.get('avg_metrics', {})
                
                for variable in ['pt', 'eta', 'phi']:
                    if variable in avg_metrics:
                        metrics_data = avg_metrics[variable]
                        f.write(f"{variable.upper()}:\n")
                        f.write(f"  Average Efficiency: {metrics_data.get('avg_efficiency', 'N/A'):.4f}\n")
                        f.write(f"  Average Purity: {metrics_data.get('avg_purity', 'N/A'):.4f}\n")
                        f.write(f"  Average Double Matching: {metrics_data.get('avg_double_matching', 'N/A'):.4f}\n")
                
                f.write("\n")
        
        print(f"Comparative summary written to {summary_path}")
    
    def _write_individual_summary(self, summary_path, category_name, roc_auc, overall_efficiency, overall_purity, overall_double_matching, avg_metrics, working_point_metrics=None):
        """Write individual summary for a single category."""
        with open(summary_path, 'w') as f:
            f.write(f"TASK 1: HIT-TRACK ASSIGNMENT EVALUATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name.replace('_', ' ').title()}\n\n")
            
            # Track count prominently displayed
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of tracks analyzed: {len(self.predictions):,}\n")
            f.write(f"Max events limit: {self.max_events}\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n" if roc_auc is not None else "ROC AUC: N/A\n")
            f.write(f"Overall Hit Assignment Efficiency: {overall_efficiency:.4f}\n" if overall_efficiency is not None else "Overall Hit Assignment Efficiency: N/A\n")
            f.write(f"Overall Hit Assignment Purity: {overall_purity:.4f}\n" if overall_purity is not None else "Overall Hit Assignment Purity: N/A\n")
            f.write(f"Overall Double Matching Efficiency: {overall_double_matching:.4f}\n" if overall_double_matching is not None else "Overall Double Matching Efficiency: N/A\n")
            f.write("\n")
            
            # Working point metrics
            if working_point_metrics:
                f.write("WORKING POINT METRICS\n")
                f.write("-" * 25 + "\n")
                for wp, metrics in working_point_metrics.items():
                    if metrics['purity'] is not None:
                        f.write(f"Working Point {wp:.2f}:\n")
                        f.write(f"  Threshold: {metrics['threshold']:.4f}\n")
                        f.write(f"  Target Efficiency: {metrics['efficiency']:.4f}\n")
                        f.write(f"  Achieved Efficiency: {metrics['achieved_efficiency']:.4f}\n")
                        f.write(f"  Purity: {metrics['purity']:.4f}\n")
                    else:
                        f.write(f"Working Point {wp:.2f}: Could not achieve target efficiency\n")
                f.write("\n")
            
            # Average metrics by variable
            f.write("AVERAGE METRICS BY KINEMATIC VARIABLE\n")
            f.write("-" * 40 + "\n")
            for variable in ['pt', 'eta', 'phi']:
                if variable in avg_metrics:
                    metrics_data = avg_metrics[variable]
                    f.write(f"{variable.upper()}:\n")
                    f.write(f"  Average Efficiency: {metrics_data.get('avg_efficiency', 'N/A'):.4f}\n" if metrics_data.get('avg_efficiency') is not None else f"  Average Efficiency: N/A\n")
                    f.write(f"  Average Purity: {metrics_data.get('avg_purity', 'N/A'):.4f}\n" if metrics_data.get('avg_purity') is not None else f"  Average Purity: N/A\n")
                    f.write(f"  Average Double Matching: {metrics_data.get('avg_double_matching', 'N/A'):.4f}\n" if metrics_data.get('avg_double_matching') is not None else f"  Average Double Matching: N/A\n")
                    f.write("\n")
        
        print(f"Individual summary for {category_name} written to {summary_path}")
        
    def calculate_efficiency_purity_by_variable(self, variable='pt', bins=None):
        """Calculate efficiency and purity binned by a kinematic variable."""
        
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Extract the variable values
        var_values = np.array([track[variable] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        # bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        efficiencies = []
        purities = []
        bin_centers = []
        eff_errors = []
        pur_errors = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get predictions and truth for this binf
            # for pred in self.predictions: # these are proper tracks 
            #     print(pred)

            bin_predictions = [self.predictions[j] for j in range(len(self.predictions)) if mask[j]]
            bin_truth = [self.true_assignments[j] for j in range(len(self.true_assignments)) if mask[j]]
            
            # print(bin_truth)
            # print(bin_prediction)
            # Calculate efficiency and purity

            total_true_hits = sum(truth.sum() for truth in bin_truth)
            total_pred_hits = sum(pred.sum() for pred in bin_predictions) 
            total_correct_hits = sum((pred & truth).sum() for pred, truth in zip(bin_predictions, bin_truth))
            efficiencies = [(pred & truth).sum() / truth.sum() for pred, truth in zip(bin_predictions, bin_truth)]
            # print("efficiencies", efficiencies)
            purities = [(pred & truth).sum() / pred.sum() for pred, truth in zip(bin_predictions, bin_truth)]
            # print("purities", purities)
            if total_true_hits > 0:
                efficiency = np.mean(efficiencies)
                # print("efficiency", efficiency)
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_true_hits)
            else:
                efficiency = 0
                eff_error = 0
                
            if total_pred_hits > 0:
                purity = np.mean(purities)
                # print("purity", purity)
                pur_error = np.sqrt(purity * (1 - purity) / total_pred_hits)
            else:
                purity = 0
                pur_error = 0
            
            efficiencies.append(efficiency)
            purities.append(purity)
            eff_errors.append(eff_error)
            pur_errors.append(pur_error)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(efficiencies), np.array(purities), np.array(eff_errors), np.array(pur_errors)
    
    def calculate_double_matching_efficiency_by_variable(self, variable='pt', bins=None):
        """Calculate double matching efficiency binned by a kinematic variable.
        
        Double matching efficiency counts a track as +1 if both efficiency >= 50% AND purity >= 50%
        for that track, otherwise it counts as 0.
        """
        
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Extract the variable values
        var_values = np.array([track[variable] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        
        double_matching_efficiencies = []
        bin_centers = []
        dm_errors = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get predictions and truth for this bin
            bin_predictions = [self.predictions[j] for j in range(len(self.predictions)) if mask[j]]
            bin_truth = [self.true_assignments[j] for j in range(len(self.true_assignments)) if mask[j]]
            
            # Calculate double matching efficiency for each track in the bin
            double_match_count = 0
            total_tracks_in_bin = len(bin_predictions)
            
            for pred, truth in zip(bin_predictions, bin_truth):
                # Calculate efficiency and purity for this individual track
                true_hits = truth.sum()
                
                pred_hits = pred.sum()
                correct_hits = (pred & truth).sum()
                
                # Calculate track-level efficiency and purity
                track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
                # print("track_efficiency", track_efficiency)
                track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
                # print("track_purity", track_purity)
                
                # Count as double match if both >= 50%
                if track_efficiency >= 0.5 and track_purity >= 0.5:
                    double_match_count += 1
            
            # Calculate double matching efficiency for this bin
            if total_tracks_in_bin > 0:
                double_matching_eff = double_match_count / total_tracks_in_bin
                # Binomial error
                dm_error = np.sqrt(double_matching_eff * (1 - double_matching_eff) / total_tracks_in_bin)
            else:
                double_matching_eff = 0
                dm_error = 0
            
            double_matching_efficiencies.append(double_matching_eff)
            dm_errors.append(dm_error)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(double_matching_efficiencies), np.array(dm_errors)

    def calculate_double_matching_efficiency_by_variable_with_working_points(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95], bins=None):
        """Calculate double matching efficiency binned by a kinematic variable using efficiency working points.
        
        Double matching efficiency counts a track as +1 if the track's efficiency is >= any of the working point thresholds
        AND the track's purity >= 50%, otherwise it counts as 0.
        
        Args:
            variable: 'pt', 'eta', or 'phi'
            working_points: List of efficiency thresholds to consider (e.g., [0.80, 0.85, 0.95])
            bins: Bins for the variable
            
        Returns:
            Dictionary with results for each working point
        """
        
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Extract the variable values
        var_values = np.array([track[variable] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        
        # Results for each working point
        results = {}
        for wp in working_points:
            results[wp] = {
                'bin_centers': [],
                'double_matching_efficiencies': [],
                'dm_errors': []
            }
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get predictions and truth for this bin
            bin_predictions = [self.predictions[j] for j in range(len(self.predictions)) if mask[j]]
            bin_truth = [self.true_assignments[j] for j in range(len(self.true_assignments)) if mask[j]]
            
            bin_center = (bins[i] + bins[i+1]) / 2
            total_tracks_in_bin = len(bin_predictions)
            
            # Calculate double matching efficiency for each working point
            for wp in working_points:
                double_match_count = 0
                
                for pred, truth in zip(bin_predictions, bin_truth):
                    # Calculate efficiency and purity for this individual track
                    true_hits = truth.sum()
                    pred_hits = pred.sum()
                    correct_hits = (pred & truth).sum()
                    
                    # Calculate track-level efficiency and purity
                    track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
                    track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
                    
                    # Count as double match if efficiency >= working point threshold AND purity >= 50%
                    if track_efficiency >= wp and track_purity >= 0.5:
                        double_match_count += 1
                
                # Calculate double matching efficiency for this bin and working point
                if total_tracks_in_bin > 0:
                    double_matching_eff = double_match_count / total_tracks_in_bin
                    # Binomial error
                    dm_error = np.sqrt(double_matching_eff * (1 - double_matching_eff) / total_tracks_in_bin)
                else:
                    double_matching_eff = 0
                    dm_error = 0
                
                results[wp]['bin_centers'].append(bin_center)
                results[wp]['double_matching_efficiencies'].append(double_matching_eff)
                results[wp]['dm_errors'].append(dm_error)
        
        # Convert lists to arrays
        for wp in working_points:
            results[wp]['bin_centers'] = np.array(results[wp]['bin_centers'])
            results[wp]['double_matching_efficiencies'] = np.array(results[wp]['double_matching_efficiencies'])
            results[wp]['dm_errors'] = np.array(results[wp]['dm_errors'])
        
        return results

    def plot_double_matching_efficiency_vs_variable_with_working_points(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95], output_subdir=None):
        """Plot double matching efficiency vs a kinematic variable for different working points."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        wp_results = self.calculate_double_matching_efficiency_by_variable_with_working_points(variable, working_points, bins)
        
        # Create separate plots for each working point
        for wp in working_points:
            bin_centers = wp_results[wp]['bin_centers']
            double_matching_effs = wp_results[wp]['double_matching_efficiencies']
            dm_errors = wp_results[wp]['dm_errors']
            
            if len(bin_centers) == 0:
                print(f"Warning: No data points for {variable} double matching efficiency plot (WP {wp:.2f})")
                continue
            
            # Create the plot with step style and error bands (baseline style)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Reconstruct bins for step plotting
            for i, (bin_center, dm_val, dm_err) in enumerate(zip(bin_centers, double_matching_effs, dm_errors)):
                # Find corresponding bin edges
                bin_idx = np.searchsorted(bins[:-1] + (bins[1:] - bins[:-1])/2, bin_center)
                lhs = bins[bin_idx]
                rhs = bins[bin_idx + 1]
                color = 'green'
                
                # Create error band
                if dm_err > 0:
                    point_in_range = np.linspace(lhs, rhs, 100)
                    y_upper = min(dm_val + dm_err, 1.0)  # Cap at 1.0
                    y_lower = max(dm_val - dm_err, 0.0)  # Floor at 0.0
                    ax.fill_between(point_in_range, y_upper, y_lower, 
                                   color=color, alpha=0.3, 
                                   label="binomial err - Double Matching Eff" if i == 0 else "")
                
                # Step plot
                ax.step([lhs, rhs], [dm_val, dm_val], 
                       color=color, linewidth=2.5,
                       label=f"Double Matching Efficiency (WP {wp:.2f})" if i == 0 else "")
            
            ax.set_ylabel('Double Matching Efficiency')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Double Matching Efficiency vs {variable.capitalize()}\n(Efficiency ≥ {wp:.0%} AND Purity ≥ 50%)')
            
            # Set x-axis labels
            if variable == 'pt':
                ax.set_xlabel('$p_T$ [GeV]')
            elif variable == 'eta':
                ax.set_xlabel('$\\eta$')
            elif variable == 'phi':
                ax.set_xlabel('$\\phi$ [rad]')
            
            plt.tight_layout()
            
            # Save the plot
            if output_subdir:
                output_dir = output_subdir  # output_subdir is already a Path object
            else:
                output_dir = self.output_dir
                
            output_path = output_dir / f'double_matching_efficiency_wp{wp:.2f}_vs_{variable}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved double matching efficiency (WP {wp:.2f}) vs {variable} plot to {output_path}")
    
    def plot_efficiency_purity_vs_variable(self, variable='pt', output_subdir=None):
        """Plot efficiency and purity vs a kinematic variable."""
        
        # Define bins same as in calculate_efficiency_purity_by_variable (matching filter evaluation ranges)
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3 (matching filter eval)
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi (matching filter eval)
        
        bin_centers, efficiencies, purities, eff_errors, pur_errors = self.calculate_efficiency_purity_by_variable(variable, bins)
        
        if len(bin_centers) == 0:
            print(f"Warning: No data points for {variable} plots")
            return
        
        # Create the plot with step style and error bands
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Efficiency plot with step style and error bands
        for i, (lhs, rhs, eff_val, eff_err) in enumerate(zip(bins[:-1], bins[1:], efficiencies, eff_errors)):
            color = 'blue'
            
            # Create error band
            if eff_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(eff_val + eff_err, 1.0)  # Cap at 1.0
                y_lower = max(eff_val - eff_err, 0.0)  # Floor at 0.0
                ax1.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Efficiency" if i == 0 else "")
            
            # Step plot
            ax1.step([lhs, rhs], [eff_val, eff_val], 
                   color=color, linewidth=2.5,
                   label="Efficiency" if i == 0 else "")
        
        ax1.set_ylabel('Hit Assignment Efficiency')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Hit Assignment Efficiency vs {variable.capitalize()}')
        
        # Purity plot with step style and error bands
        for i, (lhs, rhs, pur_val, pur_err) in enumerate(zip(bins[:-1], bins[1:], purities, pur_errors)):
            color = 'red'
            
            # Create error band
            if pur_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(pur_val + pur_err, 1.0)  # Cap at 1.0
                y_lower = max(pur_val - pur_err, 0.0)  # Floor at 0.0
                ax2.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Purity" if i == 0 else "")
            
            # Step plot
            ax2.step([lhs, rhs], [pur_val, pur_val], 
                   color=color, linewidth=2.5,
                   label="Purity" if i == 0 else "")
        
        ax2.set_ylabel('Hit Assignment Purity')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Hit Assignment Purity vs {variable.capitalize()}')
        
        # Set x-axis labels
        if variable == 'pt':
            ax2.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax2.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax2.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir  # output_subdir is already a Path object
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'efficiency_purity_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved efficiency/purity vs {variable} plot to {output_path}")
    
    def plot_double_matching_efficiency_vs_variable(self, variable='pt', output_subdir=None):
        """Plot double matching efficiency vs a kinematic variable."""
        
        # Define bins same as in calculate_double_matching_efficiency_by_variable
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        bin_centers, double_matching_effs, dm_errors = self.calculate_double_matching_efficiency_by_variable(variable, bins)
        
        if len(bin_centers) == 0:
            print(f"Warning: No data points for {variable} double matching efficiency plot")
            return
        
        # Create the plot with step style and error bands
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Double matching efficiency plot with step style and error bands
        for i, (lhs, rhs, dm_val, dm_err) in enumerate(zip(bins[:-1], bins[1:], double_matching_effs, dm_errors)):
            color = 'green'
            
            # Create error band
            if dm_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(dm_val + dm_err, 1.0)  # Cap at 1.0
                y_lower = max(dm_val - dm_err, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Double Matching Eff" if i == 0 else "")
            
            # Step plot
            ax.step([lhs, rhs], [dm_val, dm_val], 
                   color=color, linewidth=2.5,
                   label="Double Matching Efficiency" if i == 0 else "")
        
        ax.set_ylabel('Double Matching Efficiency')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Double Matching Efficiency vs {variable.capitalize()}\n(Both Efficiency ≥ 50% AND Purity ≥ 50%)')
        
        # Set x-axis labels
        if variable == 'pt':
            ax.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir  # output_subdir is already a Path object
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'double_matching_efficiency_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved double matching efficiency vs {variable} plot to {output_path}")
    
    def plot_roc_curve(self, output_subdir=None):
        """Plot ROC curve using hit assignment logits."""
        
        if len(self.logits) == 0:
            print("Warning: No logits available for ROC curve")
            return
        
        # Flatten all logits and truth labels
        all_logits = []
        all_truth = []
        
        for logit, truth in zip(self.logits, self.true_assignments):
            all_logits.extend(logit)
            all_truth.extend(truth)
        
        all_logits = np.array(all_logits)
        all_truth = np.array(all_truth, dtype=bool)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_truth, all_logits)
        roc_auc = auc(fpr, tpr)
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Hit-Track Assignment')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir  # output_subdir is already a Path object
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / 'roc_curve_hit_track_assignment.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curve to {output_path}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return roc_auc
    
    def calculate_overall_efficiency_purity(self):
        """Calculate overall efficiency and purity across all tracks."""
        if len(self.predictions) == 0:
            return 0.0, 0.0
        
        # Calculate overall metrics across all tracks
        total_true_hits = sum(truth.sum() for truth in self.true_assignments)
        total_pred_hits = sum(pred.sum() for pred in self.predictions)
        total_correct_hits = sum((pred & truth).sum() for pred, truth in zip(self.predictions, self.true_assignments))
        
        # Calculate efficiency and purity
        efficiency = total_correct_hits / total_true_hits if total_true_hits > 0 else 0.0
        purity = total_correct_hits / total_pred_hits if total_pred_hits > 0 else 0.0
        
        return efficiency, purity
    
    def calculate_overall_double_matching_efficiency(self):
        """Calculate overall double matching efficiency across all tracks.
        
        Returns the fraction of tracks where both efficiency >= 50% AND purity >= 50%.
        """
        if len(self.predictions) == 0:
            return 0.0
        
        double_match_count = 0
        total_tracks = len(self.predictions)
        
        for pred, truth in zip(self.predictions, self.true_assignments):
            # Calculate efficiency and purity for this individual track
            true_hits = truth.sum()
            pred_hits = pred.sum()
            correct_hits = (pred & truth).sum()
            
            # Calculate track-level efficiency and purity
            track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
            track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
            
            # Count as double match if both >= 50%
            if track_efficiency >= 0.5 and track_purity >= 0.5:
                double_match_count += 1
        
        # Calculate overall double matching efficiency
        return double_match_count / total_tracks if total_tracks > 0 else 0.0

    def calculate_working_point_metrics(self, working_points=[0.5, 0.80, 0.85, 0.95]):
        """Calculate efficiency and purity at specific working points using logits.
        
        Args:
            working_points: List of target efficiency values
            
        Returns:
            Dictionary with working point results
        """
        if len(self.logits) == 0 or len(self.true_assignments) == 0:
            print("Warning: No logits or truth data available for working point calculation")
            return {}
        
        # Flatten all logits and truth labels
        all_logits = []
        all_truth = []
        
        for logit, truth in zip(self.logits, self.true_assignments):
            all_logits.extend(logit)
            all_truth.extend(truth)
        
        all_logits = np.array(all_logits)
        all_truth = np.array(all_truth, dtype=bool)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_truth, all_logits)
        
        working_point_results = {}
        
        for working_point in working_points:
            # Find threshold that gives the desired efficiency (recall)
            target_efficiency = working_point
            
            # Find the threshold that achieves the target efficiency
            if not np.any(tpr >= target_efficiency):
                print(f"Warning: Cannot achieve efficiency {target_efficiency}")
                working_point_results[working_point] = {
                    'threshold': None,
                    'efficiency': None,
                    'purity': None,
                    'achieved_efficiency': None
                }
                continue
            
            threshold = thresholds[tpr >= target_efficiency][0]
            
            # Apply threshold to get predictions
            cut_predictions = all_logits >= threshold
            
            # Calculate overall purity for this working point
            total_true_positives = np.sum(all_truth & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            total_true_hits = np.sum(all_truth)
            
            achieved_efficiency = total_true_positives / total_true_hits if total_true_hits > 0 else 0.0
            purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
            
            working_point_results[working_point] = {
                'threshold': threshold,
                'efficiency': target_efficiency,
                'purity': purity,
                'achieved_efficiency': achieved_efficiency
            }
            
            print(f"Working point {working_point:.2f}: threshold={threshold:.4f}, "
                  f"achieved_eff={achieved_efficiency:.4f}, purity={purity:.4f}")
        
        return working_point_results

    def calculate_working_point_metrics_by_variable(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95], bins=None):
        """Calculate working point efficiency and purity binned by a kinematic variable.
        
        Args:
            variable: Kinematic variable ('pt', 'eta', 'phi')
            working_points: List of target efficiency values
            bins: Custom bin edges (if None, uses default binning)
            
        Returns:
            Dictionary with bin centers and metrics for each working point
        """
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Extract the variable values
        var_values = np.array([track[variable] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        
        results = {}
        bin_centers = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get logits and truth for this bin
            bin_logits = [self.logits[j] for j in range(len(self.logits)) if mask[j]]
            bin_truth = [self.true_assignments[j] for j in range(len(self.true_assignments)) if mask[j]]
            
            # Flatten logits and truth for this bin
            flat_logits = []
            flat_truth = []
            
            for logit, truth in zip(bin_logits, bin_truth):
                flat_logits.extend(logit)
                flat_truth.extend(truth)
            
            flat_logits = np.array(flat_logits)
            flat_truth = np.array(flat_truth, dtype=bool)
            
            if len(flat_logits) == 0 or len(flat_truth) == 0:
                continue
            
            # Calculate ROC curve for this bin
            fpr, tpr, thresholds = roc_curve(flat_truth, flat_logits)
            
            bin_center = (bins[i] + bins[i+1]) / 2
            bin_centers.append(bin_center)
            
            bin_results = {}
            
            for working_point in working_points:
                target_efficiency = working_point
                
                # Find the threshold that achieves the target efficiency
                if not np.any(tpr >= target_efficiency):
                    bin_results[working_point] = {
                        'threshold': None,
                        'efficiency': None,
                        'purity': None,
                        'achieved_efficiency': None
                    }
                    continue
                
                threshold = thresholds[tpr >= target_efficiency][0]
                
                # Apply threshold to get predictions
                cut_predictions = flat_logits >= threshold
                
                # Calculate metrics for this working point in this bin
                total_true_positives = np.sum(flat_truth & cut_predictions)
                total_predicted_positives = np.sum(cut_predictions)
                total_true_hits = np.sum(flat_truth)
                
                achieved_efficiency = total_true_positives / total_true_hits if total_true_hits > 0 else 0.0
                purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
                
                bin_results[working_point] = {
                    'threshold': threshold,
                    'efficiency': target_efficiency,
                    'purity': purity,
                    'achieved_efficiency': achieved_efficiency
                }
            
            results[bin_center] = bin_results
        
        return np.array(bin_centers), results

    def plot_working_point_comparison_vs_variable(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95], output_subdir=None):
        """Plot comparison between prediction-based and working point-based efficiency/purity vs a kinematic variable."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get traditional prediction-based metrics
        bin_centers_pred, efficiencies_pred, purities_pred, eff_errors_pred, pur_errors_pred = self.calculate_efficiency_purity_by_variable(variable, bins)
        
        # Get working point-based metrics
        bin_centers_wp, wp_results = self.calculate_working_point_metrics_by_variable(variable, working_points, bins)
        
        if len(bin_centers_pred) == 0 and len(bin_centers_wp) == 0:
            print(f"Warning: No data points for {variable} working point comparison plots")
            return
        
        # Create the comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Traditional efficiency plot (top left)
        if len(bin_centers_pred) > 0:
            for i, (lhs, rhs, eff_val, eff_err) in enumerate(zip(bins[:-1], bins[1:], efficiencies_pred, eff_errors_pred)):
                color = 'blue'
                
                # Error band
                if eff_err > 0:
                    point_in_range = np.linspace(lhs, rhs, 100)
                    y_upper = min(eff_val + eff_err, 1.0)
                    y_lower = max(eff_val - eff_err, 0.0)
                    ax1.fill_between(point_in_range, y_upper, y_lower, 
                                   color=color, alpha=0.3, 
                                   label="binomial err - Traditional" if i == 0 else "")
                
                # Step plot
                ax1.step([lhs, rhs], [eff_val, eff_val], 
                       color=color, linewidth=2.5,
                       label="Traditional Predictions" if i == 0 else "")
        
        ax1.set_ylabel('Hit Assignment Efficiency')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Traditional Efficiency vs {variable.capitalize()}')
        
        # Traditional purity plot (top right)
        if len(bin_centers_pred) > 0:
            for i, (lhs, rhs, pur_val, pur_err) in enumerate(zip(bins[:-1], bins[1:], purities_pred, pur_errors_pred)):
                color = 'red'
                
                # Error band
                if pur_err > 0:
                    point_in_range = np.linspace(lhs, rhs, 100)
                    y_upper = min(pur_val + pur_err, 1.0)
                    y_lower = max(pur_val - pur_err, 0.0)
                    ax2.fill_between(point_in_range, y_upper, y_lower, 
                                   color=color, alpha=0.3, 
                                   label="binomial err - Traditional" if i == 0 else "")
                
                # Step plot
                ax2.step([lhs, rhs], [pur_val, pur_val], 
                       color=color, linewidth=2.5,
                       label="Traditional Predictions" if i == 0 else "")
        
        ax2.set_ylabel('Hit Assignment Purity')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Traditional Purity vs {variable.capitalize()}')
        
        # Working point efficiency plot (bottom left)
        colors_wp_eff = ['darkgreen', 'orange', 'purple']
        for wp_idx, working_point in enumerate(working_points):
            wp_efficiencies = []
            wp_bin_centers = []
            
            for bin_center in bin_centers_wp:
                if bin_center in wp_results and working_point in wp_results[bin_center]:
                    achieved_eff = wp_results[bin_center][working_point]['achieved_efficiency']
                    if achieved_eff is not None:
                        wp_efficiencies.append(achieved_eff)
                        wp_bin_centers.append(bin_center)
            
            if len(wp_efficiencies) > 0:
                ax3.plot(wp_bin_centers, wp_efficiencies, 
                        color=colors_wp_eff[wp_idx], marker='o', linewidth=2, markersize=4,
                        label=f'WP {working_point:.2f}')
        
        ax3.set_ylabel('Achieved Efficiency')
        ax3.set_ylim(0, 1.1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title(f'Working Point Efficiency vs {variable.capitalize()}')
        
        # Working point purity plot (bottom right)
        colors_wp_pur = ['darkgreen', 'orange', 'purple']
        for wp_idx, working_point in enumerate(working_points):
            wp_purities = []
            wp_bin_centers = []
            
            for bin_center in bin_centers_wp:
                if bin_center in wp_results and working_point in wp_results[bin_center]:
                    purity = wp_results[bin_center][working_point]['purity']
                    if purity is not None:
                        wp_purities.append(purity)
                        wp_bin_centers.append(bin_center)
            
            if len(wp_purities) > 0:
                ax4.plot(wp_bin_centers, wp_purities, 
                        color=colors_wp_pur[wp_idx], marker='s', linewidth=2, markersize=4,
                        label=f'WP {working_point:.2f}')
        
        ax4.set_ylabel('Purity at Working Point')
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title(f'Working Point Purity vs {variable.capitalize()}')
        
        # Set x-axis labels for bottom plots
        for ax in [ax3, ax4]:
            if variable == 'pt':
                ax.set_xlabel('$p_T$ [GeV]')
            elif variable == 'eta':
                ax.set_xlabel('$\\eta$')
            elif variable == 'phi':
                ax.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'working_point_comparison_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved working point comparison vs {variable} plot to {output_path}")
    
    def plot_working_point_efficiency_vs_variable(self, variable='pt', working_point=0.80, output_subdir=None):
        """Plot efficiency for a specific working point vs a kinematic variable using baseline plot style."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get working point-based metrics
        bin_centers_wp, wp_results = self.calculate_working_point_metrics_by_variable(variable, [working_point], bins)
        
        if len(bin_centers_wp) == 0:
            print(f"Warning: No data points for {variable} working point {working_point} efficiency plot")
            return
        
        # Extract efficiencies for this working point
        wp_efficiencies = []
        wp_eff_errors = []
        valid_bins = []
        
        for i, bin_center in enumerate(bin_centers_wp):
            if bin_center in wp_results and working_point in wp_results[bin_center]:
                achieved_eff = wp_results[bin_center][working_point]['achieved_efficiency']
                if achieved_eff is not None:
                    wp_efficiencies.append(achieved_eff)
                    # Calculate binomial error (approximate)
                    # For efficiency, error = sqrt(eff * (1-eff) / N)
                    # We'll use a simplified error estimate
                    eff_err = np.sqrt(achieved_eff * (1 - achieved_eff) / max(1, len(self.track_info) // len(bins)))
                    wp_eff_errors.append(eff_err)
                    valid_bins.append(i)
        
        if len(wp_efficiencies) == 0:
            print(f"Warning: No valid efficiency data for working point {working_point}")
            return
        
        # Create the plot with baseline style (step plot with error bands)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Efficiency plot with step style and error bands
        for i, (bin_idx, eff_val, eff_err) in enumerate(zip(valid_bins, wp_efficiencies, wp_eff_errors)):
            lhs = bins[bin_idx]
            rhs = bins[bin_idx + 1]
            color = 'blue'
            
            # Create error band
            if eff_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(eff_val + eff_err, 1.0)  # Cap at 1.0
                y_lower = max(eff_val - eff_err, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Efficiency" if i == 0 else "")
            
            # Step plot
            ax.step([lhs, rhs], [eff_val, eff_val], 
                   color=color, linewidth=2.5,
                   label=f"Efficiency (WP {working_point:.2f})" if i == 0 else "")
        
        ax.set_ylabel('Hit Assignment Efficiency')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Hit Assignment Efficiency vs {variable.capitalize()} (Working Point {working_point:.2f})')
        
        # Set x-axis labels
        if variable == 'pt':
            ax.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'efficiency_wp{working_point:.2f}_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved working point {working_point:.2f} efficiency vs {variable} plot to {output_path}")
    
    def plot_working_point_purity_vs_variable(self, variable='pt', working_point=0.80, output_subdir=None):
        """Plot purity for a specific working point vs a kinematic variable using baseline plot style."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get working point-based metrics
        bin_centers_wp, wp_results = self.calculate_working_point_metrics_by_variable(variable, [working_point], bins)
        
        if len(bin_centers_wp) == 0:
            print(f"Warning: No data points for {variable} working point {working_point} purity plot")
            return
        
        # Extract purities for this working point
        wp_purities = []
        wp_pur_errors = []
        valid_bins = []
        
        for i, bin_center in enumerate(bin_centers_wp):
            if bin_center in wp_results and working_point in wp_results[bin_center]:
                purity = wp_results[bin_center][working_point]['purity']
                if purity is not None:
                    wp_purities.append(purity)
                    # Calculate binomial error (approximate)
                    # For purity, error = sqrt(pur * (1-pur) / N)
                    # We'll use a simplified error estimate
                    pur_err = np.sqrt(purity * (1 - purity) / max(1, len(self.track_info) // len(bins)))
                    wp_pur_errors.append(pur_err)
                    valid_bins.append(i)
        
        if len(wp_purities) == 0:
            print(f"Warning: No valid purity data for working point {working_point}")
            return
        
        # Create the plot with baseline style (step plot with error bands)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Purity plot with step style and error bands
        for i, (bin_idx, pur_val, pur_err) in enumerate(zip(valid_bins, wp_purities, wp_pur_errors)):
            lhs = bins[bin_idx]
            rhs = bins[bin_idx + 1]
            color = 'red'
            
            # Create error band
            if pur_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(pur_val + pur_err, 1.0)  # Cap at 1.0
                y_lower = max(pur_val - pur_err, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Purity" if i == 0 else "")
            
            # Step plot
            ax.step([lhs, rhs], [pur_val, pur_val], 
                   color=color, linewidth=2.5,
                   label=f"Purity (WP {working_point:.2f})" if i == 0 else "")
        
        ax.set_ylabel('Hit Assignment Purity')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Hit Assignment Purity vs {variable.capitalize()} (Working Point {working_point:.2f})')
        
        # Set x-axis labels
        if variable == 'pt':
            ax.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'purity_wp{working_point:.2f}_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved working point {working_point:.2f} purity vs {variable} plot to {output_path}")
    
    def plot_working_point_efficiency_purity_vs_variable(self, variable='pt', working_point=0.80, output_subdir=None):
        """Plot efficiency and purity for a specific working point vs a kinematic variable in combined plot (like baseline)."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get working point-based metrics
        bin_centers_wp, wp_results = self.calculate_working_point_metrics_by_variable(variable, [working_point], bins)
        
        if len(bin_centers_wp) == 0:
            print(f"Warning: No data points for {variable} working point {working_point} efficiency/purity plot")
            return
        
        # Extract efficiencies and purities for this working point
        wp_efficiencies = []
        wp_eff_errors = []
        wp_purities = []
        wp_pur_errors = []
        valid_bins = []
        
        for i, bin_center in enumerate(bin_centers_wp):
            if bin_center in wp_results and working_point in wp_results[bin_center]:
                achieved_eff = wp_results[bin_center][working_point]['achieved_efficiency']
                purity = wp_results[bin_center][working_point]['purity']
                
                if achieved_eff is not None and purity is not None:
                    wp_efficiencies.append(achieved_eff)
                    wp_purities.append(purity)
                    
                    # Calculate binomial errors (approximate)
                    eff_err = np.sqrt(achieved_eff * (1 - achieved_eff) / max(1, len(self.track_info) // len(bins)))
                    pur_err = np.sqrt(purity * (1 - purity) / max(1, len(self.track_info) // len(bins)))
                    
                    wp_eff_errors.append(eff_err)
                    wp_pur_errors.append(pur_err)
                    valid_bins.append(i)
        
        if len(wp_efficiencies) == 0:
            print(f"Warning: No valid efficiency/purity data for working point {working_point}")
            return
        
        # Create the plot with baseline style (two subplots, step plots with error bands)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Efficiency plot with step style and error bands
        for i, (bin_idx, eff_val, eff_err) in enumerate(zip(valid_bins, wp_efficiencies, wp_eff_errors)):
            lhs = bins[bin_idx]
            rhs = bins[bin_idx + 1]
            color = 'blue'
            
            # Create error band
            if eff_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(eff_val + eff_err, 1.0)  # Cap at 1.0
                y_lower = max(eff_val - eff_err, 0.0)  # Floor at 0.0
                ax1.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Efficiency" if i == 0 else "")
            
            # Step plot
            ax1.step([lhs, rhs], [eff_val, eff_val], 
                   color=color, linewidth=2.5,
                   label=f"Efficiency (WP {working_point:.2f})" if i == 0 else "")
        
        ax1.set_ylabel('Hit Assignment Efficiency')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Hit Assignment Efficiency vs {variable.capitalize()} (Working Point {working_point:.2f})')
        
        # Purity plot with step style and error bands
        for i, (bin_idx, pur_val, pur_err) in enumerate(zip(valid_bins, wp_purities, wp_pur_errors)):
            lhs = bins[bin_idx]
            rhs = bins[bin_idx + 1]
            color = 'red'
            
            # Create error band
            if pur_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(pur_val + pur_err, 1.0)  # Cap at 1.0
                y_lower = max(pur_val - pur_err, 0.0)  # Floor at 0.0
                ax2.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Purity" if i == 0 else "")
            
            # Step plot
            ax2.step([lhs, rhs], [pur_val, pur_val], 
                   color=color, linewidth=2.5,
                   label=f"Purity (WP {working_point:.2f})" if i == 0 else "")
        
        ax2.set_ylabel('Hit Assignment Purity')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Hit Assignment Purity vs {variable.capitalize()} (Working Point {working_point:.2f})')
        
        # Set x-axis labels
        if variable == 'pt':
            ax2.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax2.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax2.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = output_subdir
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'efficiency_purity_wp{working_point:.2f}_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved working point {working_point:.2f} efficiency/purity vs {variable} plot to {output_path}")
    
    def calculate_average_metrics_by_variable(self, variable='pt'):
        """Calculate average efficiency, purity, and double matching efficiency for a variable."""
        # Define bins same as other methods
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get metrics for each bin
        bin_centers, efficiencies, purities, eff_errors, pur_errors = self.calculate_efficiency_purity_by_variable(variable, bins)
        _, double_matching_effs, dm_errors = self.calculate_double_matching_efficiency_by_variable(variable, bins)
        
        # Calculate averages (weighted by number of tracks in each bin)
        var_values = np.array([track[variable] for track in self.track_info])
        bin_indices = np.digitize(var_values, bins) - 1
        
        # Calculate weights for each bin
        weights = []
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            weights.append(mask.sum())
        
        weights = np.array(weights)
        total_weight = weights.sum()
        
        if total_weight > 0 and len(efficiencies) > 0:
            # Only include bins that have data
            valid_indices = weights > 0
            avg_efficiency = np.average(efficiencies, weights=weights[valid_indices]) if np.any(valid_indices) else 0.0
            avg_purity = np.average(purities, weights=weights[valid_indices]) if np.any(valid_indices) else 0.0
            avg_double_matching = np.average(double_matching_effs, weights=weights[valid_indices]) if np.any(valid_indices) else 0.0
        else:
            avg_efficiency = 0.0
            avg_purity = 0.0
            avg_double_matching = 0.0
        
        return avg_efficiency, avg_purity, avg_double_matching

    def calculate_average_working_point_metrics_by_variable(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95]):
        """Calculate average efficiency and purity for working points for a variable."""
        # Define bins same as other methods
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get working point metrics for each bin
        bin_centers_wp, wp_results = self.calculate_working_point_metrics_by_variable(variable, working_points, bins)
        
        # Calculate weights for each bin
        var_values = np.array([track[variable] for track in self.track_info])
        bin_indices = np.digitize(var_values, bins) - 1
        
        weights = []
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            weights.append(mask.sum())
        
        weights = np.array(weights)
        total_weight = weights.sum()
        
        # Calculate averages for each working point
        avg_wp_metrics = {}
        for wp in working_points:
            wp_efficiencies = []
            wp_purities = []
            valid_weights = []
            
            for i, bin_center in enumerate(bin_centers_wp):
                if bin_center in wp_results and wp in wp_results[bin_center]:
                    achieved_eff = wp_results[bin_center][wp]['achieved_efficiency']
                    purity = wp_results[bin_center][wp]['purity']
                    
                    if achieved_eff is not None and purity is not None:
                        wp_efficiencies.append(achieved_eff)
                        wp_purities.append(purity)
                        # Find corresponding weight
                        bin_idx = np.searchsorted(bins[:-1] + (bins[1:] - bins[:-1])/2, bin_center)
                        if bin_idx < len(weights):
                            valid_weights.append(weights[bin_idx])
                        else:
                            valid_weights.append(1)  # fallback weight
            
            if len(wp_efficiencies) > 0 and sum(valid_weights) > 0:
                avg_efficiency = np.average(wp_efficiencies, weights=valid_weights)
                avg_purity = np.average(wp_purities, weights=valid_weights)
            else:
                avg_efficiency = 0.0
                avg_purity = 0.0
            
            avg_wp_metrics[wp] = {
                'efficiency': avg_efficiency,
                'purity': avg_purity
            }
        
        return avg_wp_metrics

    def calculate_average_double_matching_with_working_points_by_variable(self, variable='pt', working_points=[0.5, 0.80, 0.85, 0.95]):
        """Calculate average double matching efficiency with working points for a variable."""
        # Define bins same as other methods
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Get double matching metrics with working points
        wp_results = self.calculate_double_matching_efficiency_by_variable_with_working_points(variable, working_points, bins)
        
        # Calculate weights for each bin
        var_values = np.array([track[variable] for track in self.track_info])
        bin_indices = np.digitize(var_values, bins) - 1
        
        weights = []
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            weights.append(mask.sum())
        
        weights = np.array(weights)
        
        # Calculate averages for each working point
        avg_dm_metrics = {}
        for wp in working_points:
            if wp in wp_results:
                dm_efficiencies = wp_results[wp]['double_matching_efficiencies']
                bin_centers = wp_results[wp]['bin_centers']
                
                # Match weights to bins
                valid_weights = []
                for bin_center in bin_centers:
                    bin_idx = np.searchsorted(bins[:-1] + (bins[1:] - bins[:-1])/2, bin_center)
                    if bin_idx < len(weights):
                        valid_weights.append(weights[bin_idx])
                    else:
                        valid_weights.append(1)  # fallback weight
                
                if len(dm_efficiencies) > 0 and sum(valid_weights) > 0:
                    avg_double_matching = np.average(dm_efficiencies, weights=valid_weights)
                else:
                    avg_double_matching = 0.0
            else:
                avg_double_matching = 0.0
            
            avg_dm_metrics[wp] = avg_double_matching
        
        return avg_dm_metrics
    
    def run_evaluation(self):
        """Run the complete evaluation for Task 1."""
        print("=" * 80)
        print("TASK 1: HIT-TRACK ASSIGNMENT EVALUATION")
        print("=" * 80)
        
        # Setup and collect data
        self.setup_data_module()
        self.collect_data()
        
        if len(self.predictions) == 0:
            print("Error: No data collected. Check file paths and data format.")
            return
        
        # Create plots
        print("\nGenerating plots...")
        
        # Efficiency/purity vs kinematic variables
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_efficiency_purity_vs_variable(variable)
            except Exception as e:
                print(f"Error creating {variable} plots: {e}")
        
        # Individual working point efficiency/purity plots (combined)
        working_points = [0.5, 0.80, 0.85, 0.95]
        for variable in ['pt', 'eta', 'phi']:
            for wp in working_points:
                try:
                    self.plot_working_point_efficiency_purity_vs_variable(variable, wp)
                except Exception as e:
                    print(f"Error creating efficiency/purity plot for WP {wp:.2f} vs {variable}: {e}")
        
        # Double matching efficiency vs kinematic variables (original - 50% thresholds)
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_double_matching_efficiency_vs_variable(variable)
            except Exception as e:
                print(f"Error creating {variable} double matching efficiency plots: {e}")
        
        # Double matching efficiency with working points
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_double_matching_efficiency_vs_variable_with_working_points(variable, working_points)
            except Exception as e:
                print(f"Error creating {variable} double matching efficiency with working points plots: {e}")
        
        # ROC curve
        try:
            roc_auc = self.plot_roc_curve()
        except Exception as e:
            print(f"Error creating ROC curve: {e}")
            roc_auc = None
        
        # Calculate overall efficiency and purity
        try:
            overall_efficiency, overall_purity = self.calculate_overall_efficiency_purity()
            print(f"\nOverall Hit Assignment Efficiency: {overall_efficiency:.4f}")
            print(f"Overall Hit Assignment Purity: {overall_purity:.4f}")
        except Exception as e:
            print(f"Error calculating overall metrics: {e}")
            overall_efficiency, overall_purity = None, None
        
        # Calculate overall double matching efficiency
        try:
            overall_double_matching = self.calculate_overall_double_matching_efficiency()
            print(f"Overall Double Matching Efficiency: {overall_double_matching:.4f}")
        except Exception as e:
            print(f"Error calculating overall double matching efficiency: {e}")
            overall_double_matching = None
        
        # Calculate working point metrics
        try:
            working_point_metrics = self.calculate_working_point_metrics([0.5, 0.80, 0.85, 0.95])
            print(f"\nWORKING POINT METRICS:")
            for wp, metrics in working_point_metrics.items():
                if metrics['purity'] is not None:
                    print(f"  WP {wp:.2f}: threshold={metrics['threshold']:.4f}, "
                          f"achieved_eff={metrics['achieved_efficiency']:.4f}, "
                          f"purity={metrics['purity']:.4f}")
                else:
                    print(f"  WP {wp:.2f}: Could not achieve target efficiency")
        except Exception as e:
            print(f"Error calculating working point metrics: {e}")
            working_point_metrics = {}
        
        # Calculate average metrics by variable
        avg_metrics = {}
        avg_wp_metrics = {}
        avg_dm_wp_metrics = {}
        working_points = [0.5, 0.6, 0.80, 0.85, 0.95]
        
        for variable in ['pt', 'eta', 'phi']:
            try:
                avg_eff, avg_pur, avg_dm = self.calculate_average_metrics_by_variable(variable)
                avg_metrics[variable] = {
                    'efficiency': avg_eff,
                    'purity': avg_pur,
                    'double_matching': avg_dm
                }
                print(f"\nAverage metrics for {variable}:")
                print(f"  Efficiency: {avg_eff:.4f}")
                print(f"  Purity: {avg_pur:.4f}")
                print(f"  Double Matching: {avg_dm:.4f}")
            except Exception as e:
                print(f"Error calculating average metrics for {variable}: {e}")
                avg_metrics[variable] = {'efficiency': None, 'purity': None, 'double_matching': None}
            
            # Calculate average working point metrics
            try:
                avg_wp_var = self.calculate_average_working_point_metrics_by_variable(variable, working_points)
                avg_wp_metrics[variable] = avg_wp_var
                print(f"\nAverage working point metrics for {variable}:")
                for wp, metrics in avg_wp_var.items():
                    print(f"  WP {wp:.2f}: efficiency={metrics['efficiency']:.4f}, purity={metrics['purity']:.4f}")
            except Exception as e:
                print(f"Error calculating average working point metrics for {variable}: {e}")
                avg_wp_metrics[variable] = {}
            
            # Calculate average double matching with working points
            try:
                avg_dm_var = self.calculate_average_double_matching_with_working_points_by_variable(variable, working_points)
                avg_dm_wp_metrics[variable] = avg_dm_var
                print(f"\nAverage double matching with working points for {variable}:")
                for wp, dm_eff in avg_dm_var.items():
                    print(f"  WP {wp:.2f}: double_matching={dm_eff:.4f}")
            except Exception as e:
                print(f"Error calculating average double matching with working points for {variable}: {e}")
                avg_dm_wp_metrics[variable] = {}
        
        # Write summary
        self.write_summary(roc_auc, overall_efficiency, overall_purity, overall_double_matching, 
                          avg_metrics, working_point_metrics, avg_wp_metrics, avg_dm_wp_metrics)
        
        print(f"\nTask 1 evaluation complete. Results saved to {self.output_dir}")
    
    def write_summary(self, roc_auc, overall_efficiency=None, overall_purity=None, overall_double_matching=None, 
                     avg_metrics=None, working_point_metrics=None, avg_wp_metrics=None, avg_dm_wp_metrics=None):
        """Write evaluation summary."""
        summary_path = self.output_dir / 'task1_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("TASK 1: HIT-TRACK ASSIGNMENT EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Number of tracks processed: {len(self.predictions)}\n")
            f.write(f"Total number of hits: {sum(len(pred) for pred in self.predictions)}\n")
            f.write(f"Total true hits: {sum(truth.sum() for truth in self.true_assignments)}\n")
            f.write(f"Total predicted hits: {sum(pred.sum() for pred in self.predictions)}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            if roc_auc is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            
            # Add overall efficiency and purity metrics
            if overall_efficiency is not None:
                f.write(f"Overall Hit Assignment Efficiency: {overall_efficiency:.4f}\n")
            if overall_purity is not None:
                f.write(f"Overall Hit Assignment Purity: {overall_purity:.4f}\n")
            if overall_double_matching is not None:
                f.write(f"Overall Double Matching Efficiency: {overall_double_matching:.4f}\n")
            
            # Add working point metrics
            if working_point_metrics:
                f.write(f"\nWORKING POINT METRICS\n")
                f.write("-" * 25 + "\n")
                for wp, metrics in working_point_metrics.items():
                    if metrics['purity'] is not None:
                        f.write(f"Working Point {wp:.2f}:\n")
                        f.write(f"  Threshold: {metrics['threshold']:.4f}\n")
                        f.write(f"  Target Efficiency: {metrics['efficiency']:.4f}\n")
                        f.write(f"  Achieved Efficiency: {metrics['achieved_efficiency']:.4f}\n")
                        f.write(f"  Purity: {metrics['purity']:.4f}\n")
                    else:
                        f.write(f"Working Point {wp:.2f}: Could not achieve target efficiency\n")
            
            # Add average metrics by variable
            if avg_metrics:
                f.write(f"\nAVERAGE METRICS BY KINEMATIC VARIABLES\n")
                f.write("-" * 40 + "\n")
                for variable, metrics in avg_metrics.items():
                    f.write(f"\n{variable.upper()} Averages:\n")
                    if metrics['efficiency'] is not None:
                        f.write(f"  Average Efficiency: {metrics['efficiency']:.4f}\n")
                    if metrics['purity'] is not None:
                        f.write(f"  Average Purity: {metrics['purity']:.4f}\n")
                    if metrics['double_matching'] is not None:
                        f.write(f"  Average Double Matching Efficiency: {metrics['double_matching']:.4f}\n")
            
            # Add average working point metrics by variable
            if avg_wp_metrics:
                f.write(f"\nAVERAGE WORKING POINT METRICS BY KINEMATIC VARIABLES\n")
                f.write("-" * 50 + "\n")
                for variable, wp_metrics in avg_wp_metrics.items():
                    if wp_metrics:
                        f.write(f"\n{variable.upper()} Working Point Averages:\n")
                        for wp, metrics in wp_metrics.items():
                            f.write(f"  WP {wp:.2f}:\n")
                            f.write(f"    Average Efficiency: {metrics['efficiency']:.4f}\n")
                            f.write(f"    Average Purity: {metrics['purity']:.4f}\n")
            
            # Add average double matching with working points by variable
            if avg_dm_wp_metrics:
                f.write(f"\nAVERAGE DOUBLE MATCHING WITH WORKING POINTS BY KINEMATIC VARIABLES\n")
                f.write("-" * 65 + "\n")
                for variable, dm_wp_metrics in avg_dm_wp_metrics.items():
                    if dm_wp_metrics:
                        f.write(f"\n{variable.upper()} Double Matching Working Point Averages:\n")
                        for wp, dm_eff in dm_wp_metrics.items():
                            f.write(f"  WP {wp:.2f}: Average Double Matching Efficiency = {dm_eff:.4f}\n")
            
            f.write(f"\nGenerated at: {datetime.now()}\n")
        
        print(f"Summary written to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 1: Hit-Track Assignment')
    parser.add_argument('--eval_path', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                    #    default="/scratch/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                    #    default="/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                    #    default="/shared/tracking/hepattn_muon/src/logs/TRK-ATLAS-Muon-smallModel_20250922-T170248/ckpts/epoch=008-val_loss=0.68485_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, 
                    #    default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_tracking_NGT_small2track_regression_inference.yaml",
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_tracking_NGT_small2track_regression_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task1_hit_track_assignment',
                       help='Base output directory for plots and results (timestamp will be added automatically)')
    parser.add_argument('--max_events', "-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    parser.add_argument('--categories', action='store_true',
                       help='Run evaluation with baseline categories (all tracks, baseline tracks, rejected tracks)')
    
    args = parser.parse_args()
    
    print("Task 1: Hit-Track Assignment Evaluation")
    print("=" * 50)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    print(f"Categories mode: {args.categories}")
    
    try:
        evaluator = Task1HitTrackEvaluator(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        if args.categories:
            evaluator.run_evaluation_with_categories()
        else:
            evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()