#!/usr/bin/env python3
"""
Evaluation script for Task 2: Track Validity Classification (track_valid)

This script evaluates the performance of the track validity classification task with:
1. Three categories: all tracks, baseline tracks, rejected tracks
2. On-the-fly data processing (no large memory storage)
3. ROC curves using track validity logits
4. Efficiency and fake rate plots over pt, eta, phi
5. Baseline filtering criteria from Task 1

Based on lessons learned from simple_task1_metrics.py
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

class Task2TrackValidityEvaluator:
    """Evaluator for track validity classification task with baseline filtering."""
    
    def __init__(self, eval_path, data_dir, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task2_evaluation_{timestamp}"
        
        # Create output directory and subdirectories for categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks" 
        self.rejected_dir = self.output_dir / "rejected_tracks"
        
        for subdir in [self.all_tracks_dir, self.baseline_dir, self.rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Task 2 Evaluator initialized")
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
            num_workers=4,  # Reduced from 100
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
            'logits': [], 'true_validity': [], 'predictions': [],
            'track_pts': [], 'track_etas': [], 'track_phis': []
        }
        baseline_data = {
            'logits': [], 'true_validity': [], 'predictions': [],
            'track_pts': [], 'track_etas': [], 'track_phis': []
        }
        rejected_data = {
            'logits': [], 'true_validity': [], 'predictions': [],
            'track_pts': [], 'track_etas': [], 'track_phis': []
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
                pred_group = pred_file[event_id]
                
                # Get predictions
                track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                track_valid_logits = pred_group['outputs/final/track_valid/track_logit'][...]  # Shape: (1, 2)
                track_hit_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                
                # Get predicted track parameters for binning (available for both real and fake tracks)
                pred_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                pred_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                
                # Check if pt predictions are available
                has_pt_pred = 'preds/final/parameter_regression/track_truthMuon_pt' in pred_group
                if has_pt_pred:
                    pred_pt = pred_group['preds/final/parameter_regression/track_truthMuon_pt'][...]  # Shape: (1, 2)
                else:
                    pred_pt = None
                
                # Get truth
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
                
                # First pass: check if this event contains any baseline tracks
                event_has_baseline_track = False
                event_tracks = []
                
                # Process both potential tracks (max 2 tracks per event)
                for track_idx in range(2):
                    # Get track validity predictions and truth
                    predicted_track_valid = bool(track_valid_pred[0, track_idx])
                    true_particle_exists = bool(true_particle_valid[track_idx])
                    track_logit = float(track_valid_logits[0, track_idx])
                    
                    # Use predicted kinematic parameters for binning (works for both real and fake tracks)
                    track_eta = pred_eta[0, track_idx].item()
                    track_phi = pred_phi[0, track_idx].item()
                    track_pt = pred_pt[0, track_idx].item() if has_pt_pred else 0.0
                    
                    # Store track info temporarily
                    track_info = {
                        'predicted_track_valid': predicted_track_valid,
                        'true_particle_exists': true_particle_exists,
                        'track_logit': track_logit,
                        'track_pt': track_pt,
                        'track_eta': track_eta,
                        'track_phi': track_phi
                    }
                    
                    # Apply baseline filtering (only for tracks with true particles)
                    passes_baseline = False
                    if true_particle_exists:
                        baseline_stats['total_tracks_checked'] += 1
                        passes_baseline = True
                        
                        # Get hit assignments for baseline filtering
                        true_hits = true_hit_assignments[track_idx].numpy().astype(bool)
                        
                        # Get truth parameters for baseline filtering criteria
                        truth_pt = targets["particle_truthMuon_pt"][0, track_idx].item()
                        truth_eta = targets["particle_truthMuon_eta"][0, track_idx].item()
                        
                        # Pre-filter 1: tracks must have at least 9 hits total
                        total_true_hits = true_hits.sum()
                        if total_true_hits < 9:
                            baseline_stats['tracks_failed_min_hits'] += 1
                            passes_baseline = False
                        
                        # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7 (use truth eta)
                        if passes_baseline and (np.abs(truth_eta) < 0.1 or np.abs(truth_eta) > 2.7):
                            baseline_stats['tracks_failed_eta_cuts'] += 1
                            passes_baseline = False
                        
                        # Pre-filter 3: pt threshold >= 3.0 GeV (use truth pt)
                        if passes_baseline and truth_pt < 5.0:
                            baseline_stats['tracks_failed_pt_cuts'] += 1
                            passes_baseline = False
                        
                        # Pre-filter 4: station requirements
                        if passes_baseline:
                            true_station_index = inputs["hit_spacePoint_stationIndex"][0]
                            track_stations = true_station_index[true_hits]
                            unique_stations, station_counts = np.unique(track_stations, return_counts=True)
                            
                            # At least 3 different stations and 3 stations with >=3 hits each
                            if len(unique_stations) < 3:
                                baseline_stats['tracks_failed_station_cuts'] += 1
                                passes_baseline = False
                            else:
                                n_good_stations = np.sum(station_counts >= 3)
                                if n_good_stations < 3:
                                    baseline_stats['tracks_failed_station_cuts'] += 1
                                    passes_baseline = False
                        
                        if passes_baseline:
                            baseline_stats['tracks_passed_all_cuts'] += 1
                            event_has_baseline_track = True
                    
                    track_info['passes_baseline'] = passes_baseline
                    event_tracks.append(track_info)
                
                # Second pass: assign tracks to categories based on event-level information
                for track_info in event_tracks:
                    predicted_track_valid = track_info['predicted_track_valid']
                    true_particle_exists = track_info['true_particle_exists']
                    track_logit = track_info['track_logit']
                    track_pt = track_info['track_pt']
                    track_eta = track_info['track_eta']
                    track_phi = track_info['track_phi']
                    passes_baseline = track_info['passes_baseline']
                    
                    # Add to all tracks category
                    all_data['logits'].append(track_logit)
                    all_data['true_validity'].append(true_particle_exists)
                    all_data['predictions'].append(predicted_track_valid)
                    all_data['track_pts'].append(track_pt)
                    all_data['track_etas'].append(track_eta)
                    all_data['track_phis'].append(track_phi)
                    
                    # Add to appropriate category
                    if true_particle_exists and passes_baseline:
                        # Baseline tracks (true particles that pass filtering)
                        baseline_data['logits'].append(track_logit)
                        baseline_data['true_validity'].append(true_particle_exists)
                        baseline_data['predictions'].append(predicted_track_valid)
                        baseline_data['track_pts'].append(track_pt)
                        baseline_data['track_etas'].append(track_eta)
                        baseline_data['track_phis'].append(track_phi)
                    elif true_particle_exists and not passes_baseline:
                        # Rejected tracks (true particles that fail filtering)
                        rejected_data['logits'].append(track_logit)
                        rejected_data['true_validity'].append(true_particle_exists)
                        rejected_data['predictions'].append(predicted_track_valid)
                        rejected_data['track_pts'].append(track_pt)
                        rejected_data['track_etas'].append(track_eta)
                        rejected_data['track_phis'].append(track_phi)
                    elif not true_particle_exists:
                        # Fake tracks: add to rejected category, and to baseline if event has baseline tracks
                        rejected_data['logits'].append(track_logit)
                        rejected_data['true_validity'].append(true_particle_exists)
                        rejected_data['predictions'].append(predicted_track_valid)
                        rejected_data['track_pts'].append(track_pt)
                        rejected_data['track_etas'].append(track_eta)
                        rejected_data['track_phis'].append(track_phi)
                        
                        # Also add fake tracks to baseline category if event has baseline tracks
                        # This allows ROC curve calculation for baseline category
                        if event_has_baseline_track:
                            baseline_data['logits'].append(track_logit)
                            baseline_data['true_validity'].append(true_particle_exists)  # False
                            baseline_data['predictions'].append(predicted_track_valid)
                            baseline_data['track_pts'].append(track_pt)
                            baseline_data['track_etas'].append(track_eta)
                            baseline_data['track_phis'].append(track_phi)
                
                event_count += 1
        
        # Convert to numpy arrays
        for data_dict in [all_data, baseline_data, rejected_data]:
            for key in data_dict:
                data_dict[key] = np.array(data_dict[key])
        
        print(f"\nData collection complete!")
        print(f"Total events processed: {event_count}")
        
        # Print baseline filtering statistics
        print(f"\nBaseline Filtering Statistics:")
        print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
        print(f"  Failed minimum hits (>=9): {baseline_stats['tracks_failed_min_hits']} ({baseline_stats['tracks_failed_min_hits']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats['tracks_failed_eta_cuts']} ({baseline_stats['tracks_failed_eta_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed pt cuts (pt >= 3.0 GeV): {baseline_stats['tracks_failed_pt_cuts']} ({baseline_stats['tracks_failed_pt_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed station cuts: {baseline_stats['tracks_failed_station_cuts']} ({baseline_stats['tracks_failed_station_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Tracks passing all cuts: {baseline_stats['tracks_passed_all_cuts']} ({baseline_stats['tracks_passed_all_cuts']/baseline_stats['total_tracks_checked']*100:.1f}%)")
        
        # Print category statistics
        print(f"\nCategory Statistics:")
        print(f"  All tracks: {len(all_data['logits'])}")
        print(f"  Baseline tracks: {len(baseline_data['logits'])}")
        print(f"  Rejected tracks: {len(rejected_data['logits'])}")
        
        return all_data, baseline_data, rejected_data, baseline_stats
    
    def calculate_efficiency_fakerate_by_variable(self, data, variable='pt', bins=None):
        """Calculate efficiency and fake rate binned by a kinematic variable."""
        
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 200, 21)  # 20 bins
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins
        
        # Extract the variable values
        if variable == 'pt':
            var_values = data['track_pts']
        elif variable == 'eta':
            var_values = data['track_etas']
        elif variable == 'phi':
            var_values = data['track_phis']
        
        predictions = data['predictions']
        true_validity = data['true_validity']
        
        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        efficiencies = []
        fake_rates = []
        bin_centers = []
        eff_errors = []
        fake_errors = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            
            if mask.sum() == 0:
                continue
                
            # Get predictions and truth for this bin
            bin_predictions = predictions[mask]
            bin_truth = true_validity[mask]
            n_total = len(bin_predictions)
            
            # Calculate efficiency and fake rate
            n_true = bin_truth.sum()  # Number of true tracks
            n_false = (~bin_truth).sum()  # Number of fake tracks
            
            # True positives: correctly identified real tracks
            true_positives = (bin_predictions & bin_truth).sum()
            # False positives: incorrectly identified fake tracks as real
            false_positives = (bin_predictions & ~bin_truth).sum()
            # False negatives: missed real tracks
            false_negatives = (~bin_predictions & bin_truth).sum()
            
            # Efficiency = TP / (TP + FN) = TP / n_true
            if n_true > 0:
                efficiency = true_positives / n_true
                eff_error = np.sqrt(efficiency * (1 - efficiency) / n_true)
            else:
                efficiency = 0.0
                eff_error = 0.0
            
            # Fake rate = FP / Total Predictions = false positives / all predictions
            if n_total > 0:
                fake_rate = false_positives / n_total
                fake_error = np.sqrt(fake_rate * (1 - fake_rate) / n_total)
            else:
                fake_rate = 0.0
                fake_error = 0.0
            
            efficiencies.append(efficiency)
            fake_rates.append(fake_rate)
            eff_errors.append(eff_error)
            fake_errors.append(fake_error)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(efficiencies), np.array(fake_rates), np.array(eff_errors), np.array(fake_errors)
    
    def plot_efficiency_fakerate_vs_variable(self, data, variable='pt', output_dir=None, category_name=""):
        """Plot efficiency and fake rate vs a kinematic variable."""
        
        # Define bins
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)
        
        bin_centers, efficiencies, fake_rates, eff_errors, fake_errors = self.calculate_efficiency_fakerate_by_variable(data, variable, bins)
        
        if len(bin_centers) == 0:
            print(f"Warning: No data points for {variable} plots in {category_name}")
            return
        
        # Create the plot with step style and error bands
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Efficiency plot
        for i, (lhs, rhs, eff_val, eff_err) in enumerate(zip(bins[:-1], bins[1:], efficiencies, eff_errors)):
            color = 'blue'
            
            # Create error band
            if eff_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(eff_val + eff_err, 1.0)
                y_lower = max(eff_val - eff_err, 0.0)
                ax1.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Efficiency" if i == 0 else "")
            
            # Step plot
            ax1.step([lhs, rhs], [eff_val, eff_val], 
                   color=color, linewidth=2.5,
                   label="Efficiency" if i == 0 else "")
        
        ax1.set_ylabel('Track Validity Efficiency')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Track Validity Efficiency vs {variable.capitalize()} - {category_name}')
        
        # Fake rate plot
        for i, (lhs, rhs, fake_val, fake_err) in enumerate(zip(bins[:-1], bins[1:], fake_rates, fake_errors)):
            color = 'red'
            
            # Create error band
            if fake_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(fake_val + fake_err, 1.0)
                y_lower = max(fake_val - fake_err, 0.0)
                ax2.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label="binomial err - Fake Rate" if i == 0 else "")
            
            # Step plot
            ax2.step([lhs, rhs], [fake_val, fake_val], 
                   color=color, linewidth=2.5,
                   label="Fake Rate" if i == 0 else "")
        
        ax2.set_ylabel('Track Validity Fake Rate')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Track Validity Fake Rate vs {variable.capitalize()} - {category_name}')
        
        # Set x-axis labels
        if variable == 'pt':
            ax2.set_xlabel('$p_T$ [GeV]')
        elif variable == 'eta':
            ax2.set_xlabel('$\\eta$')
        elif variable == 'phi':
            ax2.set_xlabel('$\\phi$ [rad]')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f'track_validity_efficiency_fakerate_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {category_name} efficiency/fake rate vs {variable} plot to {output_path}")
    
    def plot_roc_curve(self, data, output_dir=None, category_name=""):
        """Plot ROC curve using track validity logits."""
        
        logits = data['logits']
        true_validity = data['true_validity']
        
        if len(logits) == 0:
            print(f"Warning: No logits available for ROC curve in {category_name}")
            return None
        
        # Check if we have both positive and negative examples
        n_positive = true_validity.sum()
        n_negative = (~true_validity).sum()
        
        if n_positive == 0 or n_negative == 0:
            print(f"Warning: Cannot create ROC curve for {category_name} - need both positive and negative examples")
            print(f"  Positive examples: {n_positive}, Negative examples: {n_negative}")
            return None
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_validity, logits)
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
        plt.title(f'ROC Curve for Track Validity Classification - {category_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_path = output_dir / 'roc_curve_track_validity.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {category_name} ROC curve to {output_path} (AUC: {roc_auc:.4f})")
        
        return roc_auc
    
    def plot_logit_distributions(self, data, output_dir=None, category_name=""):
        """Plot distributions of logits for true and fake tracks."""
        
        logits = data['logits']
        true_validity = data['true_validity']
        
        if len(logits) == 0:
            print(f"Warning: No logits available for distributions in {category_name}")
            return
        
        true_track_logits = logits[true_validity]
        fake_track_logits = logits[~true_validity]
        
        plt.figure(figsize=(10, 6))
        
        if len(fake_track_logits) > 0:
            plt.hist(fake_track_logits, bins=50, alpha=0.7, 
                    label=f'Fake tracks (n={len(fake_track_logits)})', 
                    color='red', density=False)
        
        if len(true_track_logits) > 0:
            plt.hist(true_track_logits, bins=50, alpha=0.7, 
                    label=f'True tracks (n={len(true_track_logits)})', 
                    color='blue', density=False)
        
        plt.xlabel('Track Validity Logit')
        plt.ylabel('Count')
        plt.title(f'Distribution of Track Validity Logits - {category_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_path = output_dir / 'track_validity_logit_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {category_name} logit distributions to {output_path}")
    
    def write_category_summary(self, data, output_dir, category_name, roc_auc):
        """Write evaluation summary for a category."""
        summary_path = output_dir / f'{category_name.lower().replace(" ", "_")}_summary.txt'
        
        logits = data['logits']
        predictions = data['predictions']
        true_validity = data['true_validity']
        
        if len(predictions) == 0:
            print(f"Warning: No data for {category_name} summary")
            return
        
        # Calculate overall statistics
        n_total = len(predictions)
        n_true_tracks = true_validity.sum()
        n_fake_tracks = n_total - n_true_tracks
        n_pred_valid = predictions.sum()
        n_pred_invalid = n_total - n_pred_valid
        
        # Calculate confusion matrix elements
        true_positives = (predictions & true_validity).sum()
        false_positives = (predictions & ~true_validity).sum()
        true_negatives = (~predictions & ~true_validity).sum()
        false_negatives = (~predictions & true_validity).sum()
        
        # Calculate metrics
        efficiency = true_positives / n_true_tracks if n_true_tracks > 0 else 0  # TP / (TP + FN)
        fake_rate = false_positives / n_total if n_total > 0 else 0  # FP / Total predictions
        
        with open(summary_path, 'w') as f:
            f.write(f"TASK 2: TRACK VALIDITY CLASSIFICATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total tracks analyzed: {n_total:,}\n")
            f.write(f"True valid tracks: {n_true_tracks:,}\n")
            f.write(f"True invalid tracks: {n_fake_tracks:,}\n")
            f.write(f"Predicted valid tracks: {n_pred_valid:,}\n")
            f.write(f"Predicted invalid tracks: {n_pred_invalid:,}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write(f"True Positives: {true_positives:,}\n")
            f.write(f"False Positives: {false_positives:,}\n")
            f.write(f"True Negatives: {true_negatives:,}\n")
            f.write(f"False Negatives: {false_negatives:,}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Efficiency: {efficiency:.4f}\n")
            f.write(f"Fake Rate: {fake_rate:.4f}\n")
            
            if roc_auc is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            else:
                f.write("ROC AUC: N/A (insufficient data)\n")
            
            f.write(f"\nGenerated at: {datetime.now()}\n")
        
        print(f"Summary for {category_name} written to {summary_path}")
    
    def write_comparative_summary(self, all_results, baseline_stats):
        """Write comprehensive summary comparing all categories."""
        summary_path = self.output_dir / 'task2_comparative_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK 2: TRACK VALIDITY CLASSIFICATION - COMPARATIVE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Max events processed: {self.max_events}\n\n")
            
            # Write filtering statistics
            f.write("BASELINE FILTERING STATISTICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total tracks checked: {baseline_stats.get('total_tracks_checked', 0):,}\n")
            f.write(f"Failed minimum hits (>=9): {baseline_stats.get('tracks_failed_min_hits', 0):,}\n")
            f.write(f"Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats.get('tracks_failed_eta_cuts', 0):,}\n")
            f.write(f"Failed pt cuts (pt >= 3.0 GeV): {baseline_stats.get('tracks_failed_pt_cuts', 0):,}\n")
            f.write(f"Failed station cuts: {baseline_stats.get('tracks_failed_station_cuts', 0):,}\n")
            f.write(f"Tracks passing all cuts: {baseline_stats.get('tracks_passed_all_cuts', 0):,}\n\n")
            
            # Write comparative metrics
            f.write("COMPARATIVE METRICS\n")
            f.write("-" * 20 + "\n")
            
            categories = ['All Tracks', 'Baseline Tracks', 'Rejected Tracks']
            f.write(f"{'Category':<20}{'Num Tracks':<15}{'Efficiency':<12}{'Fake Rate':<12}{'ROC AUC':<10}\n")
            f.write("-" * 70 + "\n")
            
            for category in categories:
                if category in all_results:
                    result = all_results[category]
                    f.write(f"{category:<20}{result['num_tracks']:<15}{result['efficiency']:<12.4f}{result['fake_rate']:<12.4f}")
                    if result['roc_auc'] is not None:
                        f.write(f"{result['roc_auc']:<10.4f}\n")
                    else:
                        f.write(f"{'N/A':<10}\n")
                else:
                    f.write(f"{category:<20}{'0':<15}{'N/A':<12}{'N/A':<12}{'N/A':<10}\n")
            
            f.write("\n")
        
        print(f"Comparative summary written to {summary_path}")
    
    def run_evaluation_with_categories(self):
        """Run evaluation for all categories."""
        print("=" * 80)
        print("TASK 2: TRACK VALIDITY CLASSIFICATION WITH CATEGORIES")
        print("=" * 80)
        
        # Setup and collect data
        self.setup_data_module()
        all_data, baseline_data, rejected_data, baseline_stats = self.collect_and_process_data()
        
        # Store results for comparative summary
        all_results = {}
        
        # Process each category
        categories = [
            ("All Tracks", all_data, self.all_tracks_dir),
            ("Baseline Tracks", baseline_data, self.baseline_dir),
            ("Rejected Tracks", rejected_data, self.rejected_dir)
        ]
        
        for category_name, data, output_dir in categories:
            print(f"\n" + "="*50)
            print(f"EVALUATING {category_name.upper()}")
            print("="*50)
            
            if len(data['logits']) == 0:
                print(f"Warning: No data for {category_name}")
                continue
            
            # Generate plots
            print("Generating plots...")
            
            # ROC curve
            try:
                roc_auc = self.plot_roc_curve(data, output_dir, category_name)
            except Exception as e:
                print(f"Error creating ROC curve: {e}")
                roc_auc = None
            
            # Logit distributions
            try:
                self.plot_logit_distributions(data, output_dir, category_name)
            except Exception as e:
                print(f"Error creating logit distributions: {e}")
            
            # Efficiency/fake rate vs kinematic variables
            for variable in ['pt', 'eta', 'phi']:
                try:
                    self.plot_efficiency_fakerate_vs_variable(data, variable, output_dir, category_name)
                except Exception as e:
                    print(f"Error creating efficiency/fake rate vs {variable} plot: {e}")
            
            # Calculate summary statistics
            predictions = data['predictions']
            true_validity = data['true_validity']
            
            if len(predictions) > 0:
                n_total = len(predictions)
                n_true_tracks = true_validity.sum()
                n_fake_tracks = n_total - n_true_tracks
                n_pred_valid = predictions.sum()
                
                true_positives = (predictions & true_validity).sum()
                false_positives = (predictions & ~true_validity).sum()
                true_negatives = (~predictions & ~true_validity).sum()
                
                efficiency = true_positives / n_true_tracks if n_true_tracks > 0 else 0  # TP / (TP + FN)
                fake_rate = false_positives / n_total if n_total > 0 else 0  # FP / Total predictions
                
                all_results[category_name] = {
                    'num_tracks': n_total,
                    'efficiency': efficiency,
                    'fake_rate': fake_rate,
                    'roc_auc': roc_auc
                }
                
                print(f"\n{category_name.upper()} METRICS:")
                print(f"  Total tracks: {n_total:,}")
                print(f"  Efficiency: {efficiency:.4f}")
                print(f"  Fake rate: {fake_rate:.4f}")
                if roc_auc is not None:
                    print(f"  ROC AUC: {roc_auc:.4f}")
            
            # Write individual summary
            try:
                self.write_category_summary(data, output_dir, category_name, roc_auc)
            except Exception as e:
                print(f"Error writing summary for {category_name}: {e}")
        
        # Write comparative summary
        self.write_comparative_summary(all_results, baseline_stats)
        
        print(f"\nTask 2 evaluation with categories complete. Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 2: Track Validity Classification')
    parser.add_argument('--eval_path', type=str, 
                    #    default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                       default="/scratch/epoch=139-val_loss=0.70832_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                    #    default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                       default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600",
                       help='Path to processed test data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task2_track_validity',
                       help='Base output directory for plots and results')
    parser.add_argument('--max_events', "-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("Task 2: Track Validity Classification Evaluation with Categories")
    print("=" * 70)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    
    try:
        evaluator = Task2TrackValidityEvaluator(
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