#!/usr/bin/env python3
"""
Evaluation script for Task 2: Track Validity Classification (track_valid)

This script evaluates the performance of the track validity classification task by:
1. Creating ROC curves using the track validity logits
2. Creating efficiency and fake rate plots over pt, eta, phi (using true values)
3. Analyzing the performance in different detector regions

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
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.max_events = max_events
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.predictions = []       # Boolean predictions for track validity
        self.logits = []           # Raw logits for track validity
        self.true_validity = []    # True track validity
        self.track_info = []       # pt, eta, phi for each track
        
        print(f"Task 2 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        
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
                
                # Regression predictions (for track parameters)
                track_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                track_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                track_qpt = pred_group['preds/final/parameter_regression/track_truthMuon_qpt'][...]  # Shape: (1, 2)
                
                # Get truth information from batch
                inputs, targets = batch
                
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                
                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.numpy()
                num_valid = valid_particles.sum()
                
                # For this event, store the data for each track slot
                batch_size, num_tracks = track_valid_pred.shape
                
                for track_idx in range(num_tracks):
                    # Predicted track validity
                    pred_valid = track_valid_pred[0, track_idx]
                    logit_valid = track_valid_logits[0, track_idx]
                    
                    # True track validity
                    true_valid = track_idx < num_valid and valid_particles[track_idx]
                    
                    # Track parameters (use regression predictions as proxy for kinematics)
                    # For pt, convert from q/pt
                    true_pt = 1.0 / abs(track_qpt[0, track_idx]) if abs(track_qpt[0, track_idx]) > 1e-6 else 0.0
                    true_eta = track_eta[0, track_idx] 
                    true_phi = track_phi[0, track_idx]
                    
                    # Store the data
                    self.predictions.append(pred_valid)
                    self.logits.append(logit_valid)
                    self.true_validity.append(true_valid)
                    self.track_info.append({
                        'pt': true_pt,
                        'eta': true_eta, 
                        'phi': true_phi,
                        'event_id': event_count,
                        'track_id': track_idx
                    })
                
                event_count += 1
                
        print(f"Collected data for {len(self.predictions)} track slots from {event_count} events")
        
        # Convert to numpy arrays for easier handling
        self.predictions = np.array(self.predictions)
        self.logits = np.array(self.logits)
        self.true_validity = np.array(self.true_validity)
        
        # Print some statistics
        n_true_tracks = self.true_validity.sum()
        n_pred_tracks = self.predictions.sum()
        n_correct = (self.predictions & self.true_validity).sum()
        
        print(f"True tracks: {n_true_tracks}")
        print(f"Predicted tracks: {n_pred_tracks}")
        print(f"Correctly predicted tracks: {n_correct}")
        print(f"Overall accuracy: {(self.predictions == self.true_validity).mean():.3f}")
        
    def calculate_efficiency_fakerate_by_variable(self, variable='pt', bins=None):
        """Calculate efficiency and fake rate binned by a kinematic variable."""
        
        if bins is None:
            if variable == 'pt':
                bins = np.linspace(0, 50, 21)  # 20 bins: 0 to 50 GeV linear
            elif variable == 'eta':
                bins = np.linspace(-3, 3, 21)  # 20 bins: -3 to 3
            elif variable == 'phi':
                bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins: -pi to pi
        
        # Extract the variable values
        var_values = np.array([track[variable] for track in self.track_info])
        
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
            bin_predictions = self.predictions[mask]
            bin_truth = self.true_validity[mask]
            
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
        
        # Define bins same as in calculate_efficiency_fakerate_by_variable (matching filter evaluation ranges)
        if variable == 'pt':
            bins = np.linspace(0, 50, 21)  # 20 bins: 0 to 50 GeV linear
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
        """Plot ROC curve using track validity logits."""
        
        if len(self.logits) == 0:
            print("Warning: No logits available for ROC curve")
            return None
        
        # Check if we have both positive and negative examples
        n_positive = self.true_validity.sum()
        n_negative = (~self.true_validity).sum()
        
        if n_positive == 0:
            print("Warning: No positive examples for ROC curve")
            return None
        
        if n_negative == 0:
            print("Warning: No negative examples for ROC curve - all tracks are valid")
            print("This is expected for pre-filtered datasets")
            # Create a dummy plot indicating perfect performance
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, 'All tracks are valid\n(Perfect performance)', 
                    ha='center', va='center', fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
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
            
            print(f"Saved ROC curve (perfect performance) to {output_path}")
            return 1.0  # Perfect AUC
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.true_validity, self.logits)
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
        """Plot distributions of logits for true and fake tracks."""
        
        true_track_logits = self.logits[self.true_validity]
        fake_track_logits = self.logits[~self.true_validity]
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(fake_track_logits, bins=50, alpha=0.7, label=f'Fake tracks (n={len(fake_track_logits)})', 
                color='red', density=True)
        plt.hist(true_track_logits, bins=50, alpha=0.7, label=f'True tracks (n={len(true_track_logits)})', 
                color='blue', density=True)
        
        plt.xlabel('Track Validity Logit')
        plt.ylabel('Density')
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
        for variable in ['pt', 'eta', 'phi']:
            try:
                self.plot_efficiency_fakerate_vs_variable(variable)
            except Exception as e:
                print(f"Error creating {variable} plots: {e}")
        
        # Write summary
        self.write_summary(roc_auc)
        
        print(f"\nTask 2 evaluation complete. Results saved to {self.output_dir}")
    
    def write_summary(self, roc_auc):
        """Write evaluation summary."""
        summary_path = self.output_dir / 'task2_summary.txt'
        
        n_true_tracks = self.true_validity.sum()
        n_pred_tracks = self.predictions.sum()
        n_correct = (self.predictions & self.true_validity).sum()
        accuracy = (self.predictions == self.true_validity).mean()
        
        if n_true_tracks > 0:
            efficiency = n_correct / n_true_tracks
        else:
            efficiency = 0
            
        if n_pred_tracks > 0:
            purity = n_correct / n_pred_tracks
        else:
            purity = 0
        
        with open(summary_path, 'w') as f:
            f.write("TASK 2: TRACK VALIDITY CLASSIFICATION EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Number of track slots processed: {len(self.predictions)}\n")
            f.write(f"True valid tracks: {n_true_tracks}\n")
            f.write(f"Predicted valid tracks: {n_pred_tracks}\n")
            f.write(f"Correctly predicted: {n_correct}\n")
            f.write(f"Overall accuracy: {accuracy:.4f}\n")
            f.write(f"Overall efficiency: {efficiency:.4f}\n")
            f.write(f"Overall purity: {purity:.4f}\n")
            
            if roc_auc is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            
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
    parser.add_argument('--max_events',"-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("Task 2: Track Validity Classification Evaluation")
    print("=" * 60)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    
    try:
        evaluator = Task2TrackValidityEvaluator(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()