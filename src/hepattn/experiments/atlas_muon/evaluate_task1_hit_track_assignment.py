#!/usr/bin/env python3
"""
Evaluation script for Task 1: Hit-Track Assignment (hit_track_valid) with Categories

This script evaluates the performance of the hit-track assignment task by:
1. Creating efficiency and purity plots over pt, eta, phi (using true values)
2. Creating ROC curves using the track validity logits
3. Analyzing the performance with three categories: all tracks, baseline tracks, rejected tracks
4. Applying baseline filtering criteria from Tasks 2 and 3

Based on lessons learned from Task 2 and Task 3 evaluation improvements.
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
    """Evaluator for hit-track assignment task with baseline filtering and categories."""
    
    def __init__(self, eval_path, data_dir, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task1_evaluation_{timestamp}"
        
        # Create output directory and subdirectories for categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks"
        self.rejected_dir = self.output_dir / "rejected_tracks"
        
        for subdir in [self.all_tracks_dir, self.baseline_dir, self.rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Task 1 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max events: {max_events}")
        
    def setup_data_module(self):
        """Setup the data module for loading truth information."""
        print("Setting up data module...")
        
        self.data_module = AtlasMuonDataModule(
            train_dir=self.data_dir,
            val_dir=self.data_dir,
            test_dir=self.data_dir,
            num_workers=1,  # Reduced to avoid threading issues
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

    def collect_and_process_data(self):
        """Collect and process data on-the-fly, applying baseline filtering."""
        print("Collecting and processing data with baseline filtering...")
        
        # Data storage for all three categories
        all_data = {
            'predictions': [], 'logits': [], 'true_assignments': [], 
            'track_info': [], 'station_indices': []
        }
        baseline_data = {
            'predictions': [], 'logits': [], 'true_assignments': [], 
            'track_info': [], 'station_indices': []
        }
        rejected_data = {
            'predictions': [], 'logits': [], 'true_assignments': [], 
            'track_info': [], 'station_indices': []
        }
        
        # Baseline filtering statistics
        baseline_stats = {
            'total_tracks_checked': 0,
            'tracks_failed_min_hits': 0,
            'tracks_failed_eta_cuts': 0,
            'tracks_failed_pt_cuts': 0,
            'tracks_failed_station_cuts': 0,
            'tracks_passed_all_cuts': 0
        }
        
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
                true_station_index = inputs["hit_spacePoint_stationIndex"][0]
                
                # Get predictions and logits
                pred_group = pred_file[event_id]
                
                # Hit-track assignment predictions and logits
                hit_track_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]
                hit_track_logits = pred_group['outputs/final/track_hit_valid/track_hit_logit'][...]
                
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]
                true_hit_assignments = targets['particle_hit_valid'][0]
                
                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.numpy()
                num_valid = valid_particles.sum()
                
                if num_valid == 0:
                    event_count += 1
                    continue
                
                for track_idx in range(num_valid):
                    true_hits = true_hit_assignments[track_idx].numpy()
                    pred_hits = hit_track_pred[0, track_idx]
                    logit_hits = hit_track_logits[0, track_idx]
                    
                    # Handle shape mismatch by taking minimum length
                    min_hits = min(pred_hits.shape[0], true_hits.shape[0])
                    if pred_hits.shape[0] != true_hits.shape[0]:
                        pred_hits = pred_hits[:min_hits]
                        true_hits = true_hits[:min_hits]
                        logit_hits = logit_hits[:min_hits]
                    
                    # Get true track parameters
                    true_eta = targets["particle_truthMuon_eta"][0, track_idx]
                    true_phi = targets["particle_truthMuon_phi"][0, track_idx]
                    true_pt = targets["particle_truthMuon_pt"][0, track_idx]
                    
                    track_info = {
                        'pt': true_pt,
                        'eta': true_eta, 
                        'phi': true_phi,
                        'event_id': event_count,
                        'track_id': track_idx
                    }
                    
                    # Add to all tracks
                    all_data['predictions'].append(pred_hits)
                    all_data['logits'].append(logit_hits)
                    all_data['true_assignments'].append(true_hits)
                    all_data['track_info'].append(track_info)
                    all_data['station_indices'].append(true_station_index)
                    
                    # Apply baseline filtering
                    baseline_stats['total_tracks_checked'] += 1
                    
                    # Check baseline criteria
                    total_hits = np.sum(true_hits)
                    if total_hits < 9:
                        baseline_stats['tracks_failed_min_hits'] += 1
                        rejected_data['predictions'].append(pred_hits)
                        rejected_data['logits'].append(logit_hits)
                        rejected_data['true_assignments'].append(true_hits)
                        rejected_data['track_info'].append(track_info)
                        rejected_data['station_indices'].append(true_station_index)
                        continue
                    
                    if np.abs(true_eta) < 0.1 or np.abs(true_eta) > 2.7:
                        baseline_stats['tracks_failed_eta_cuts'] += 1
                        rejected_data['predictions'].append(pred_hits)
                        rejected_data['logits'].append(logit_hits)
                        rejected_data['true_assignments'].append(true_hits)
                        rejected_data['track_info'].append(track_info)
                        rejected_data['station_indices'].append(true_station_index)
                        continue
                        
                    if true_pt < 3.0:
                        baseline_stats['tracks_failed_pt_cuts'] += 1
                        rejected_data['predictions'].append(pred_hits)
                        rejected_data['logits'].append(logit_hits)
                        rejected_data['true_assignments'].append(true_hits)
                        rejected_data['track_info'].append(track_info)
                        rejected_data['station_indices'].append(true_station_index)
                        continue
                    
                    # Station requirements check
                    unique_stations, station_counts = np.unique(true_station_index, return_counts=True)
                    if len(unique_stations) < 3:
                        baseline_stats['tracks_failed_station_cuts'] += 1
                        rejected_data['predictions'].append(pred_hits)
                        rejected_data['logits'].append(logit_hits)
                        rejected_data['true_assignments'].append(true_hits)
                        rejected_data['track_info'].append(track_info)
                        rejected_data['station_indices'].append(true_station_index)
                        continue
                        
                    n_good_stations = np.sum(station_counts >= 3)
                    if n_good_stations < 3:
                        baseline_stats['tracks_failed_station_cuts'] += 1
                        rejected_data['predictions'].append(pred_hits)
                        rejected_data['logits'].append(logit_hits)
                        rejected_data['true_assignments'].append(true_hits)
                        rejected_data['track_info'].append(track_info)
                        rejected_data['station_indices'].append(true_station_index)
                        continue
                    
                    # Track passed all criteria - add to baseline
                    baseline_stats['tracks_passed_all_cuts'] += 1
                    baseline_data['predictions'].append(pred_hits)
                    baseline_data['logits'].append(logit_hits)
                    baseline_data['true_assignments'].append(true_hits)
                    baseline_data['track_info'].append(track_info)
                    baseline_data['station_indices'].append(true_station_index)
                
                event_count += 1
        
        print(f"\nData collection complete!")
        print(f"Total events processed: {event_count}")
        print(f"")
        print(f"Baseline Filtering Statistics:")
        print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
        print(f"  Failed minimum hits (>=9): {baseline_stats['tracks_failed_min_hits']} ({baseline_stats['tracks_failed_min_hits']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats['tracks_failed_eta_cuts']} ({baseline_stats['tracks_failed_eta_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed pt cuts (pt >= 3.0 GeV): {baseline_stats['tracks_failed_pt_cuts']} ({baseline_stats['tracks_failed_pt_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed station cuts: {baseline_stats['tracks_failed_station_cuts']} ({baseline_stats['tracks_failed_station_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Tracks passing all cuts: {baseline_stats['tracks_passed_all_cuts']} ({baseline_stats['tracks_passed_all_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"")
        print(f"Category Statistics:")
        print(f"  All tracks: {len(all_data['predictions'])}")
        print(f"  Baseline tracks: {len(baseline_data['predictions'])}")
        print(f"  Rejected tracks: {len(rejected_data['predictions'])}")
        
        return all_data, baseline_data, rejected_data, baseline_stats

    def run_evaluation_with_categories(self):
        """Run evaluation for all tracks, baseline tracks, and rejected tracks."""
        print("=" * 80)
        print("TASK 1: HIT-TRACK ASSIGNMENT EVALUATION WITH CATEGORIES")
        print("=" * 80)
        
        # Setup and collect data
        self.setup_data_module()
        all_data, baseline_data, rejected_data, baseline_stats = self.collect_and_process_data()
        
        if len(all_data['predictions']) == 0:
            print("Error: No data collected. Check file paths and data format.")
            return
            
        # Dictionary to store results from each category
        results = {}
        
        # 1. Evaluate all tracks
        print("\n" + "="*50)
        print("EVALUATING ALL TRACKS")
        print("="*50)
        self._set_data(all_data)
        results['all_tracks'] = self._run_single_evaluation("all_tracks", self.all_tracks_dir)
        
        # 2. Evaluate baseline tracks
        print("\n" + "="*50)
        print("EVALUATING BASELINE TRACKS")
        print("="*50)
        self._set_data(baseline_data)
        results['baseline_tracks'] = self._run_single_evaluation("baseline_tracks", self.baseline_dir)
        
        # 3. Evaluate rejected tracks
        print("\n" + "="*50)
        print("EVALUATING REJECTED TRACKS")
        print("="*50)
        self._set_data(rejected_data)
        results['rejected_tracks'] = self._run_single_evaluation("rejected_tracks", self.rejected_dir)
        
        # Write comprehensive summary
        self._write_comparative_summary(results, baseline_stats)
        
        print(f"\nTask 1 evaluation with categories complete. Results saved to {self.output_dir}")

    def _set_data(self, data_dict):
        """Set the evaluator's data attributes from a data dictionary."""
        self.predictions = data_dict['predictions']
        self.logits = data_dict['logits']
        self.true_assignments = data_dict['true_assignments']
        self.track_info = data_dict['track_info']
        self.station_indices = data_dict['station_indices']

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
        
        # Double matching efficiency vs kinematic variables
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_double_matching_efficiency_vs_variable(variable, output_subdir)
            except Exception as e:
                print(f"Error creating {variable} double matching efficiency plots: {e}")
        
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
        
        # Use overall metrics for all variables (they are consistent across PT/ETA/PHI)
        avg_metrics = {}
        for variable in ['pt', 'eta', 'phi']:
            avg_metrics[variable] = {
                'avg_efficiency': overall_efficiency,
                'avg_purity': overall_purity, 
                'avg_double_matching': overall_double_matching
            }
        
        # Write individual summary for this category
        try:
            individual_summary_path = output_subdir / f'{category_name}_summary.txt'
            self._write_individual_summary(individual_summary_path, category_name, roc_auc, 
                                         overall_efficiency, overall_purity, overall_double_matching, 
                                         avg_metrics)
        except Exception as e:
            print(f"Error writing individual summary: {e}")
        
        return {
            'roc_auc': roc_auc,
            'overall_efficiency': overall_efficiency,
            'overall_purity': overall_purity,
            'overall_double_matching': overall_double_matching,
            'avg_metrics': avg_metrics,
            'num_tracks': len(self.predictions)
        }

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
        
        efficiencies = []
        purities = []
        bin_centers = []
        eff_errors = []
        pur_errors = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get predictions and truth for this bin
            bin_predictions = [self.predictions[j] for j in range(len(self.predictions)) if mask[j]]
            bin_truth = [self.true_assignments[j] for j in range(len(self.true_assignments)) if mask[j]]
            
            # Calculate efficiency and purity
            total_true_hits = sum(truth.sum() for truth in bin_truth)
            total_pred_hits = sum(pred.sum() for pred in bin_predictions) 
            total_correct_hits = sum((pred & truth).sum() for pred, truth in zip(bin_predictions, bin_truth))
            
            if total_true_hits > 0:
                efficiency = total_correct_hits / total_true_hits
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_true_hits)
            else:
                efficiency = 0
                eff_error = 0
                
            if total_pred_hits > 0:
                purity = total_correct_hits / total_pred_hits
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
                track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
                
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

    def plot_efficiency_purity_vs_variable(self, variable='pt', output_subdir=None):
        """Plot efficiency and purity vs a kinematic variable."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
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
        output_path = output_subdir / f'efficiency_purity_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {variable.capitalize()} efficiency/purity vs {variable} plot to {output_path}")

    def plot_double_matching_efficiency_vs_variable(self, variable='pt', output_subdir=None):
        """Plot double matching efficiency vs a kinematic variable."""
        
        # Define bins
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
        output_path = output_subdir / f'double_matching_efficiency_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {variable.capitalize()} double matching efficiency vs {variable} plot to {output_path}")

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
        output_path = output_subdir / 'roc_curve_hit_track_assignment.png'
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

    def _write_individual_summary(self, summary_path, category_name, roc_auc, overall_efficiency, overall_purity, overall_double_matching, avg_metrics):
        """Write individual summary for a single category."""
        with open(summary_path, 'w') as f:
            f.write(f"TASK 1: HIT-TRACK ASSIGNMENT EVALUATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name.replace('_', ' ').title()}\n\n")
            
            # Track count prominently displayed
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of tracks analyzed: {len(self.predictions):,}\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n" if roc_auc is not None else "ROC AUC: N/A\n")
            f.write(f"Overall Hit Assignment Efficiency: {overall_efficiency:.4f}\n" if overall_efficiency is not None else "Overall Hit Assignment Efficiency: N/A\n")
            f.write(f"Overall Hit Assignment Purity: {overall_purity:.4f}\n" if overall_purity is not None else "Overall Hit Assignment Purity: N/A\n")
            f.write(f"Overall Double Matching Efficiency: {overall_double_matching:.4f}\n" if overall_double_matching is not None else "Overall Double Matching Efficiency: N/A\n")
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
            f.write(f"Failed station cuts: {filter_stats.get('tracks_failed_station_cuts', 0):,}\n")
            f.write(f"Tracks passing all cuts: {filter_stats.get('tracks_passed_all_cuts', 0):,}\n")
            
            # Write track counts per category
            f.write("\nTRACKS ANALYZED PER CATEGORY\n")
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 1: Hit-Track Assignment with Categories')
    parser.add_argument('--eval_path', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                       help='Path to processed test data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task1_hit_track',
                       help='Base output directory for plots and results')
    parser.add_argument('--max_events', "-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("Task 1: Hit-Track Assignment Evaluation with Categories")
    print("=" * 60)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    
    try:
        evaluator = Task1HitTrackEvaluator(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        evaluator.run_evaluation_with_categories()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()