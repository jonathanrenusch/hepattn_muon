#!/usr/bin/env python3
"""
Evaluation script for Task 2: Track Validity Classification (track_valid)

This script evaluates the performance of the track validity classification task by:
1. Creating ROC curves using the track validity logits
2. Creating efficiency and fake rate plots over pt, eta, phi (using true values)
3. Analyzing the performance in different detector regions

FILTERING CRITERION: Optionally filters tracks based on minimum number of hits assigned.
Use --min_hits 9 to apply the 9-hit minimum criterion for both fake rate and true rate tracks.
Default behavior (--min_hits 0) applies no filtering.

The evaluation is inspired by the hit filter evaluation approach.
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
    """Evaluator for track validity classification task."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None, min_hits=0):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.max_events = max_events
        self.min_hits = min_hits  # Minimum number of hits required per track
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.predictions = []       # Boolean predictions for track validity
        self.logits = []           # Raw logits for track validity
        self.true_validity = []    # True track validity
        self.track_info = []       # pt, eta, phi for each track
        self.hit_assignments = []  # Track-hit assignments for each track (for hit count filtering)
        self.has_pt_predictions = False  # Whether pt predictions are available
        
        print(f"Task 2 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum hits per track: {min_hits} {'(no filtering)' if min_hits == 0 else f'({min_hits}-hit cut applied)'}")
        
    def setup_data_module(self):
        """Setup the data module for loading truth information."""
        print("Setting up data module...")
        
        # Create a minimal config for the data module
        self.data_module = AtlasMuonDataModule(
            train_dir=self.data_dir,
            val_dir=self.data_dir,
            test_dir=self.data_dir,
            num_workers=1,
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
            
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                if self.max_events and event_count >= self.max_events:
                    break
                    
                # Get event ID (assuming sequential)
                event_id = str(event_count)
                
                if event_id not in pred_file:
                    print(f"Warning: Event {event_id} not found in predictions file")
                    event_count += 1
                    continue
                
                # Get predictions and logits
                pred_group = pred_file[event_id]
                
                # Track validity predictions and logits
                track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                track_valid_logits = pred_group['outputs/final/track_valid/track_logit'][...]  # Shape: (1, 2)
                
                # Track-hit assignment predictions (for hit count filtering)
                track_hit_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                
                # Get predicted track parameters for binning
                pred_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                pred_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                
                # Check if pt predictions are available
                has_pt_pred = 'preds/final/parameter_regression/track_truthMuon_pt' in pred_group
                if has_pt_pred:
                    pred_pt = pred_group['preds/final/parameter_regression/track_truthMuon_pt'][...]  # Shape: (1, 2)
                else:
                    # Try qpt and convert to pt
                    pred_pt = None
                    has_pt_pred = False
                
                # Get truth information from batch
                inputs, targets = batch
                
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                
                # Convert to numpy for easier handling
                valid_particles = true_particle_valid.numpy()
                
                # For this event, store the data for each track slot
                batch_size, num_tracks = track_valid_pred.shape
                
                for track_idx in range(num_tracks):
                    # Process ALL track slots to get proper positive/negative examples
                    # Get prediction and logit for this track slot
                    pred_prob = track_valid_pred[0, track_idx]  # This might be a probability
                    logit_valid = track_valid_logits[0, track_idx]
                    
                    # Convert probability to boolean prediction (threshold at 0.5)
                    pred_valid = bool(pred_prob > 0.5) if isinstance(pred_prob, (float, np.floating)) else bool(pred_prob)
                    
                    # Get truth validity - directly from particle_valid array
                    true_valid = bool(valid_particles[track_idx]) if track_idx < len(valid_particles) else False
                    
                    # Store the prediction data for ALL track slots
                    self.predictions.append(pred_valid)
                    self.logits.append(logit_valid)
                    self.true_validity.append(true_valid)
                    
                    # Store hit assignments for hit count filtering
                    hit_assignments = track_hit_pred[0, track_idx]  # Get hits for this track
                    self.hit_assignments.append(hit_assignments)
                    
                    # Use predicted values for binning (for both real and fake tracks)
                    pred_eta_val = pred_eta[0, track_idx]
                    pred_phi_val = pred_phi[0, track_idx]
                    pred_pt_val = pred_pt[0, track_idx] if has_pt_pred else 0.0
                    
                    self.track_info.append({
                        'pt': pred_pt_val,
                        'eta': pred_eta_val, 
                        'phi': pred_phi_val,
                        'event_id': event_count,
                        'track_id': track_idx,
                        'has_pt': has_pt_pred
                    })
                
                event_count += 1
                
                # Set pt availability flag based on first event
                if event_count == 1:
                    self.has_pt_predictions = has_pt_pred
                
        print(f"Collected data for {len(self.predictions)} track slots from {event_count} events")
        print(f"PT predictions available: {self.has_pt_predictions}")
        
        # Convert to numpy arrays for easier handling
        self.predictions = np.array(self.predictions)
        self.logits = np.array(self.logits)
        self.true_validity = np.array(self.true_validity)
        # Note: hit_assignments remains as list since each track may have different number of hits
        
        # Debug information
        print(f"Prediction range: {self.predictions.min():.3f} to {self.predictions.max():.3f}")
        print(f"Logit range: {self.logits.min():.3f} to {self.logits.max():.3f}")
        print(f"True validity distribution: {self.true_validity.sum()} true, {(~self.true_validity).sum()} false")
        print(f"Predicted validity distribution: {self.predictions.sum()} true, {(~self.predictions).sum()} false")
        
        # Print some statistics
        n_true_tracks = self.true_validity.sum()
        n_pred_tracks = self.predictions.sum()
        n_correct = (self.predictions & self.true_validity).sum()
        
        print(f"True tracks: {n_true_tracks}")
        print(f"Predicted tracks: {n_pred_tracks}")
        print(f"Correctly predicted tracks: {n_correct}")
        print(f"Overall accuracy: {(self.predictions == self.true_validity).mean():.3f}")
        
    def calculate_efficiency_fakerate_by_variable(self, variable='pt', bins=None):
        """Calculate efficiency and fake rate binned by a kinematic variable.
        
        Only considers tracks that have at least self.min_hits hits assigned (if min_hits > 0).
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
        
        # Apply minimum hit criterion if enabled
        if self.min_hits > 0:
            hit_counts = np.array([np.sum(hits) for hits in self.hit_assignments])
            min_hit_mask = hit_counts >= self.min_hits
            
            print(f"Applying {self.min_hits}-hit minimum criterion: {min_hit_mask.sum()}/{len(min_hit_mask)} tracks pass ({min_hit_mask.sum()/len(min_hit_mask)*100:.1f}%)")
            
            # Filter all data based on hit count criterion
            var_values = var_values[min_hit_mask]
            predictions_filtered = self.predictions[min_hit_mask]
            true_validity_filtered = self.true_validity[min_hit_mask]
        else:
            print(f"No hit filtering applied - using all {len(var_values)} tracks")
            predictions_filtered = self.predictions
            true_validity_filtered = self.true_validity
        
        if len(var_values) == 0:
            if self.min_hits > 0:
                print(f"Warning: No tracks pass the {self.min_hits}-hit minimum criterion")
            else:
                print("Warning: No tracks available for evaluation")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
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
            bin_predictions = predictions_filtered[mask]
            bin_truth = true_validity_filtered[mask]
            
            # Calculate efficiency and fake rate
            n_true = bin_truth.sum()  # True positives (real tracks)
            n_false = (~bin_truth).sum()  # True negatives (fake tracks)
            n_correct = (bin_predictions & bin_truth).sum()  # Correctly identified real tracks
            n_false_pos = (bin_predictions & ~bin_truth).sum()  # Incorrectly identified fake tracks as real
            
            # Efficiency = TP / (TP + FN) = correct real tracks / all real tracks
            if n_true > 0:
                efficiency = n_correct / n_true
                eff_error = np.sqrt(efficiency * (1 - efficiency) / n_true)
            else:
                efficiency = 0
                eff_error = 0
            
            # Fake rate = FP / (FP + TN) = false positives / all fake tracks
            if n_false > 0:
                fake_rate = n_false_pos / n_false
                fake_error = np.sqrt(fake_rate * (1 - fake_rate) / n_false)
            else:
                fake_rate = 0
                fake_error = 0
            
            efficiencies.append(efficiency)
            fake_rates.append(fake_rate)
            eff_errors.append(eff_error)
            fake_errors.append(fake_error)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(efficiencies), np.array(fake_rates), np.array(eff_errors), np.array(fake_errors)
    
    def plot_efficiency_fakerate_vs_variable(self, variable='pt', output_subdir=None):
        """Plot efficiency and fake rate vs a kinematic variable."""
        
        # Check if pt is requested but not available
        if variable == 'pt' and not self.has_pt_predictions:
            print(f"Warning: PT predictions not available, skipping {variable} plot")
            return
        
        # Define bins same as in calculate_efficiency_fakerate_by_variable (matching filter evaluation ranges)
        if variable == 'pt':
            bins = np.linspace(0, 200, 21)  # 20 bins: 0 to 200 GeV linear
        elif variable == 'eta':
            bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3 (matching filter eval)
        elif variable == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi (matching filter eval)
        
        bin_centers, efficiencies, fake_rates, eff_errors, fake_errors = self.calculate_efficiency_fakerate_by_variable(variable, bins)
        
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
        
        ax1.set_ylabel('Track Validity Efficiency')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Track Validity Efficiency vs {variable.capitalize()}')
        
        # Fake rate plot with step style and error bands
        for i, (lhs, rhs, fake_val, fake_err) in enumerate(zip(bins[:-1], bins[1:], fake_rates, fake_errors)):
            color = 'red'
            
            # Create error band
            if fake_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(fake_val + fake_err, 1.0)  # Cap at 1.0
                y_lower = max(fake_val - fake_err, 0.0)  # Floor at 0.0
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
        ax2.set_ylabel('Track Validity Fake Rate')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Track Validity Fake Rate vs {variable.capitalize()}')
        
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
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'track_validity_efficiency_fakerate_vs_{variable}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved track validity efficiency/fake rate vs {variable} plot to {output_path}")
    
    def plot_roc_curve(self, output_subdir=None):
        """Plot ROC curve using track validity logits.
        
        Only considers tracks that have at least self.min_hits hits assigned (if min_hits > 0).
        """
        
        if len(self.logits) == 0:
            print("Warning: No logits available for ROC curve")
            return None
        
        # Apply minimum hit criterion if enabled
        if self.min_hits > 0:
            hit_counts = np.array([np.sum(hits) for hits in self.hit_assignments])
            min_hit_mask = hit_counts >= self.min_hits
            
            print(f"ROC curve: Applying {self.min_hits}-hit minimum criterion: {min_hit_mask.sum()}/{len(min_hit_mask)} tracks pass ({min_hit_mask.sum()/len(min_hit_mask)*100:.1f}%)")
            
            # Filter data
            logits_filtered = self.logits[min_hit_mask]
            true_validity_filtered = self.true_validity[min_hit_mask]
        else:
            print(f"ROC curve: No hit filtering applied - using all {len(self.logits)} tracks")
            logits_filtered = self.logits
            true_validity_filtered = self.true_validity
        
        if len(logits_filtered) == 0:
            if self.min_hits > 0:
                print(f"Warning: No tracks pass the {self.min_hits}-hit minimum criterion for ROC curve")
            else:
                print("Warning: No tracks available for ROC curve")
            return None
        
        # Check if we have both positive and negative examples
        n_positive = true_validity_filtered.sum()
        n_negative = (~true_validity_filtered).sum()
        
        if n_positive == 0:
            print("Warning: No positive examples for ROC curve")
            return None
        
        if n_negative == 0:
            print("Warning: No negative examples for ROC curve - all tracks are valid")
            print("This could indicate:")
            print("1. Pre-filtered dataset with only valid tracks (expected)")
            print("2. Data collection issue excluding negative examples (check logic)")
            if self.min_hits > 0:
                print(f"Total tracks processed (with >={self.min_hits} hits): {len(true_validity_filtered)}")
            else:
                print(f"Total tracks processed: {len(true_validity_filtered)}")
            print(f"True positives: {n_positive}")
            
            # Create a plot indicating the situation
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, f'No negative examples found\n({n_positive} positive examples)\n\nThis may indicate:\n• Pre-filtered dataset\n• Data collection bias', 
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Track Validity Classification')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            if output_subdir:
                output_dir = self.output_dir / output_subdir
                output_dir.mkdir(exist_ok=True)
            else:
                output_dir = self.output_dir
                
            output_path = output_dir / 'roc_curve_track_validity.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved ROC curve (no negatives) to {output_path}")
            return None  # Return None instead of 1.0 to indicate undefined AUC
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_validity_filtered, logits_filtered)
        roc_auc = auc(fpr, tpr)
        
        # Debug: Check if we need to flip the logits
        print(f"Initial AUC: {roc_auc:.4f}")
        if roc_auc < 0.5:
            print("AUC < 0.5, trying flipped logits...")
            fpr_flip, tpr_flip, thresholds_flip = roc_curve(true_validity_filtered, -logits_filtered)
            roc_auc_flip = auc(fpr_flip, tpr_flip)
            print(f"Flipped AUC: {roc_auc_flip:.4f}")
            
            if roc_auc_flip > roc_auc:
                print("Using flipped logits for ROC curve")
                fpr, tpr, thresholds = fpr_flip, tpr_flip, thresholds_flip
                roc_auc = roc_auc_flip
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Track Validity Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if output_subdir:
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / 'roc_curve_track_validity.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curve to {output_path}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return roc_auc
    
    def plot_logit_distributions(self, output_subdir=None):
        """Plot distributions of logits for true and fake tracks.
        
        Only considers tracks that have at least self.min_hits hits assigned (if min_hits > 0).
        """
        
        # Apply minimum hit criterion if enabled
        if self.min_hits > 0:
            hit_counts = np.array([np.sum(hits) for hits in self.hit_assignments])
            min_hit_mask = hit_counts >= self.min_hits
            
            print(f"Logit distributions: Applying {self.min_hits}-hit minimum criterion: {min_hit_mask.sum()}/{len(min_hit_mask)} tracks pass ({min_hit_mask.sum()/len(min_hit_mask)*100:.1f}%)")
            
            # Filter data
            logits_filtered = self.logits[min_hit_mask]
            true_validity_filtered = self.true_validity[min_hit_mask]
        else:
            print(f"Logit distributions: No hit filtering applied - using all {len(self.logits)} tracks")
            logits_filtered = self.logits
            true_validity_filtered = self.true_validity
        
        if len(logits_filtered) == 0:
            if self.min_hits > 0:
                print(f"Warning: No tracks pass the {self.min_hits}-hit minimum criterion for logit distributions")
            else:
                print("Warning: No tracks available for logit distributions")
            return
        
        true_track_logits = logits_filtered[true_validity_filtered]
        fake_track_logits = logits_filtered[~true_validity_filtered]
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(fake_track_logits, bins=50, alpha=0.7, label=f'Fake tracks (n={len(fake_track_logits)})', 
                color='red', density=False)
        plt.hist(true_track_logits, bins=50, alpha=0.7, label=f'True tracks (n={len(true_track_logits)})', 
                color='blue', density=False)
        
        plt.xlabel('Track Validity Logit')
        plt.ylabel('Count')
        plt.title('Distribution of Track Validity Logits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if output_subdir:
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / 'track_validity_logit_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved logit distributions to {output_path}")
    
    def run_evaluation(self):
        """Run the complete evaluation for Task 2."""
        print("=" * 80)
        print("TASK 2: TRACK VALIDITY CLASSIFICATION EVALUATION")
        print("=" * 80)
        
        # Setup and collect data
        self.setup_data_module()
        self.collect_data()
        
        if len(self.predictions) == 0:
            print("Error: No data collected. Check file paths and data format.")
            return
        
        # Create plots
        print("\nGenerating plots...")
        
        # ROC curve
        try:
            roc_auc = self.plot_roc_curve()
        except Exception as e:
            print(f"Error creating ROC curve: {e}")
            roc_auc = None
        
        # Logit distributions
        try:
            self.plot_logit_distributions()
        except Exception as e:
            print(f"Error creating logit distributions: {e}")
        
        # Efficiency/purity vs kinematic variables
        variables_to_plot = ['eta', 'phi']
        if self.has_pt_predictions:
            variables_to_plot.insert(0, 'pt')  # Add pt first if available
        
        for variable in variables_to_plot:
            try:
                self.plot_efficiency_fakerate_vs_variable(variable)
            except Exception as e:
                print(f"Error creating {variable} plots: {e}")
        
        # Write summary
        self.write_summary(roc_auc)
        
        print(f"\nTask 2 evaluation complete. Results saved to {self.output_dir}")
    
    def calculate_averaged_fake_rates(self):
        """Calculate averaged fake rates across different kinematic variables."""
        averaged_fake_rates = {}
        
        # Variables to calculate averaged fake rates for
        variables_to_check = ['eta', 'phi']
        if self.has_pt_predictions:
            variables_to_check.insert(0, 'pt')
        
        for variable in variables_to_check:
            try:
                bin_centers, efficiencies, fake_rates, eff_errors, fake_errors = self.calculate_efficiency_fakerate_by_variable(variable)
                
                if len(fake_rates) > 0:
                    # Calculate weighted average (by number of tracks in each bin)
                    # For now, use simple average since we don't have bin populations
                    avg_fake_rate = np.mean(fake_rates)
                    std_fake_rate = np.std(fake_rates)
                    averaged_fake_rates[variable] = {
                        'mean': avg_fake_rate,
                        'std': std_fake_rate,
                        'n_bins': len(fake_rates)
                    }
                else:
                    averaged_fake_rates[variable] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'n_bins': 0
                    }
            except Exception as e:
                print(f"Error calculating averaged fake rate for {variable}: {e}")
                averaged_fake_rates[variable] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'n_bins': 0
                }
        
        return averaged_fake_rates
    
    def write_summary(self, roc_auc):
        """Write evaluation summary."""
        summary_path = self.output_dir / 'task2_summary.txt'
        
        # Calculate averaged fake rates across kinematic variables
        averaged_fake_rates = self.calculate_averaged_fake_rates()
        
        # Apply minimum hit criterion if enabled
        if self.min_hits > 0:
            hit_counts = np.array([np.sum(hits) for hits in self.hit_assignments])
            min_hit_mask = hit_counts >= self.min_hits
        else:
            min_hit_mask = np.ones(len(self.predictions), dtype=bool)  # Include all tracks
        
        # Calculate statistics both before and after filtering
        total_tracks_before = len(self.predictions)
        tracks_after_filter = min_hit_mask.sum()
        
        # Overall statistics (before filtering)
        n_true_tracks_before = self.true_validity.sum()
        n_pred_tracks_before = self.predictions.sum()
        n_correct_before = (self.predictions & self.true_validity).sum()
        accuracy_before = (self.predictions == self.true_validity).mean()
        
        # Statistics after 9-hit filtering
        predictions_filtered = self.predictions[min_hit_mask]
        true_validity_filtered = self.true_validity[min_hit_mask]
        
        if len(predictions_filtered) > 0:
            n_true_tracks = true_validity_filtered.sum()
            n_pred_tracks = predictions_filtered.sum()
            n_correct = (predictions_filtered & true_validity_filtered).sum()
            accuracy = (predictions_filtered == true_validity_filtered).mean()
            
            if n_true_tracks > 0:
                efficiency = n_correct / n_true_tracks
            else:
                efficiency = 0
                
            if n_pred_tracks > 0:
                purity = n_correct / n_pred_tracks
            else:
                purity = 0
            
            # Compute overall fake rate: false positives / all fake tracks
            n_false_tracks = len(predictions_filtered) - n_true_tracks
            n_false_pos = (predictions_filtered & ~true_validity_filtered).sum()
            if n_false_tracks > 0:
                overall_fake_rate = n_false_pos / n_false_tracks
            else:
                overall_fake_rate = 0
        else:
            n_true_tracks = n_pred_tracks = n_correct = 0
            accuracy = efficiency = purity = overall_fake_rate = 0
        
        with open(summary_path, 'w') as f:
            f.write("TASK 2: TRACK VALIDITY CLASSIFICATION EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            if self.min_hits > 0:
                f.write(f"FILTERING CRITERION: Minimum {self.min_hits} hits assigned per track\n\n")
            else:
                f.write(f"FILTERING CRITERION: No hit filtering applied\n\n")
            
            if self.min_hits > 0:
                f.write("BEFORE HIT FILTERING:\n")
                f.write(f"  Number of track slots processed: {total_tracks_before}\n")
                f.write(f"  True valid tracks: {n_true_tracks_before}\n")
                f.write(f"  Predicted valid tracks: {n_pred_tracks_before}\n")
                f.write(f"  Overall accuracy: {accuracy_before:.4f}\n\n")
                
                f.write(f"AFTER {self.min_hits}-HIT FILTERING:\n")
                f.write(f"  Tracks passing filter: {tracks_after_filter} / {total_tracks_before} ({tracks_after_filter/total_tracks_before*100:.1f}%)\n")
            else:
                f.write("EVALUATION RESULTS (NO FILTERING):\n")
                f.write(f"  Number of track slots processed: {total_tracks_before}\n")
            
            f.write(f"  True valid tracks: {n_true_tracks}\n")
            f.write(f"  True invalid tracks: {len(predictions_filtered) - n_true_tracks}\n")
            f.write(f"  Predicted valid tracks: {n_pred_tracks}\n")
            f.write(f"  Predicted invalid tracks: {len(predictions_filtered) - n_pred_tracks}\n")
            f.write(f"  Correctly predicted: {n_correct}\n")
            
            if len(predictions_filtered) > 0:
                f.write(f"  True Positives (correct valid): {(predictions_filtered & true_validity_filtered).sum()}\n")
                f.write(f"  True Negatives (correct invalid): {(~predictions_filtered & ~true_validity_filtered).sum()}\n")
                f.write(f"  False Positives (incorrect valid): {(predictions_filtered & ~true_validity_filtered).sum()}\n")
                f.write(f"  False Negatives (incorrect invalid): {(~predictions_filtered & true_validity_filtered).sum()}\n")
            
            f.write(f"  Overall accuracy: {accuracy:.4f}\n")
            f.write(f"  Overall efficiency: {efficiency:.4f}\n")
            f.write(f"  Overall purity: {purity:.4f}\n")
            f.write(f"  Overall fake rate: {overall_fake_rate:.4f}\n")
            
            # Add averaged fake rates across kinematic bins
            f.write(f"\nAVERAGED FAKE RATES ACROSS KINEMATIC BINS:\n")
            for variable, stats in averaged_fake_rates.items():
                if stats['n_bins'] > 0:
                    f.write(f"  {variable.capitalize()} bins: {stats['mean']:.4f} ± {stats['std']:.4f} (across {stats['n_bins']} bins)\n")
                else:
                    f.write(f"  {variable.capitalize()} bins: No data available\n")
            
            if roc_auc is not None:
                f.write(f"\n  ROC AUC: {roc_auc:.4f}\n")
            else:
                f.write("\n  ROC AUC: Not applicable (no negative examples)\n")
            
            f.write(f"\nGenerated at: {datetime.now()}\n")
        
        print(f"Summary written to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 2: Track Validity Classification')
    parser.add_argument('--eval_path', type=str, 
                       default="/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, 
                       default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_tracking_NGT_small2track_regression_inference.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task2_track_validity',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events',"-m", type=int, default=10000,
                       help='Maximum number of events to process (for testing)')
    parser.add_argument('--min_hits', type=int, default=0,
                       help='Minimum number of hits required per track (0 = no filtering, 9 = apply 9-hit cut)')
    
    args = parser.parse_args()
    
    print("Task 2: Track Validity Classification Evaluation")
    print("=" * 60)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    print(f"Minimum hits per track: {args.min_hits} {'(no filtering)' if args.min_hits == 0 else f'({args.min_hits}-hit cut)'}")
    
    try:
        evaluator = Task2TrackValidityEvaluator(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events,
            min_hits=args.min_hits
        )
        
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()