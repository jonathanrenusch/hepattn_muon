#!/usr/bin/env python3
"""
Evaluation script for Task 3: Regression Outputs (parameter_regression)

This script evaluates the performance of the regression outputs by:
1. Creating truth-normalized residual plots for eta, phi, and q/pt
2. Creating correlation plots between predictions and truth
3. Analyzing the performance in different detector regions and kinematic ranges

The evaluation follows similar patterns to the hit filter evaluation.
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
from scipy import stats
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

class Task3RegressionEvaluator:
    """Evaluator for regression outputs task."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.max_events = max_events
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage for regression parameters
        self.predictions = {}  # Dictionary with keys: 'phi', 'eta', 'pt', 'q'
        self.truth = {}        # Dictionary with keys: 'phi', 'eta', 'pt', 'q'
        self.track_info = []   # Additional track information
        self.statistics = {}   # Dictionary to store mean and RMS statistics
        self.normalized_residuals = {}  # Store truth-normalized residuals per parameter
        
        # Initialize storage
        for param in ['phi', 'eta', 'pt', 'q']:
            self.predictions[param] = []
            self.truth[param] = []
            self.normalized_residuals[param] = np.array([])

        print(f"Task 3 Evaluator initialized")
        print(f"Evaluation file: {self.eval_path}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
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
                
                # Get predictions
                pred_group = pred_file[event_id]
                
                # Track validity predictions (to filter out invalid tracks)
                track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                
                # Regression predictions
                pred_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                pred_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                pred_pt = pred_group['preds/final/parameter_regression/track_truthMuon_pt'][...]  # Shape: (1, 2)
                pred_q = pred_group['preds/final/parameter_regression/track_truthMuon_q'][...]  # Shape: (1, 2)
                
                # Get truth information from batch
                inputs, targets = batch
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)

                
                valid_particles = true_particle_valid.numpy()
                num_valid = valid_particles.sum()
                
                # For each valid track, store the regression results
                batch_size, num_tracks = pred_phi.shape
                # print(pred_phi.shape, pred_eta.shape, pred_pt.shape, pred_q.shape, track_valid_pred.shape)
                for track_idx in range(num_valid):
                    # Only process if this track is predicted as valid and we have truth
                    if valid_particles[track_idx]:
                        
                        # Predicted parameters
                        pred_phi_val = pred_phi[0, track_idx]
                        pred_eta_val = pred_eta[0, track_idx]
                        pred_pt_val = pred_pt[0, track_idx]
                        pred_q_val = pred_q[0, track_idx]
                        # Truth parameters
                        # print("Shape: targets[particle_truthMuon_eta]", targets["particle_truthMuon_eta"])
                        truth_phi_val = targets["particle_truthMuon_phi"][0, track_idx]
                        truth_eta_val = targets["particle_truthMuon_eta"][0, track_idx]
                        truth_pt_val = targets["particle_truthMuon_pt"][0, track_idx]
                        truth_q_val = targets["particle_truthMuon_q"][0, track_idx]
                        
                        # Store the data
                        self.predictions['phi'].append(pred_phi_val)
                        self.predictions['eta'].append(pred_eta_val)
                        self.predictions['pt'].append(pred_pt_val)
                        self.predictions['q'].append(pred_q_val)
                        
                        self.truth['phi'].append(truth_phi_val)
                        self.truth['eta'].append(truth_eta_val)
                        self.truth['pt'].append(truth_pt_val)
                        self.truth['q'].append(truth_q_val)
                        
                        # Calculate pt from qpt for additional information
                        
                        self.track_info.append({
                            'pt': truth_pt_val,
                            'eta': truth_eta_val, 
                            'phi': truth_phi_val,
                            'q': truth_q_val,
                            'event_id': event_count,
                            'track_id': track_idx
                        })
                
                event_count += 1
                
        # Convert to numpy arrays
        for param in ['phi', 'eta', 'pt', 'q']:
            self.predictions[param] = np.array(self.predictions[param])
            self.truth[param] = np.array(self.truth[param])
        
        print(f"Collected data for {len(self.track_info)} tracks from {event_count} events")
        
        # Calculate and store statistics
        for param in ['phi', 'eta', 'pt', 'q']:
            if len(self.predictions[param]) > 0:
                # Raw residuals
                residuals = self.predictions[param] - self.truth[param]
                print(residuals)
                print("std:", np.std(residuals))
                
                # For charge (q), use absolute residuals instead of normalized residuals
                if param == 'q':
                    # For charge, we use absolute residuals since truth is discrete (-1 or 1)
                    clean_norm_residuals = residuals  # No normalization for discrete charge
                else:
                    # Compute truth-normalized residuals using absolute truth to avoid sign flips
                    truth_abs = np.abs(self.truth[param])
                    # Avoid division by zero by replacing very small values with NaN
                    eps = 1e-12
                    denom = np.where(truth_abs > eps, truth_abs, np.nan)
                    norm_residuals = residuals / denom
                    
                    clean_norm_residuals = residuals / truth_abs 
                    # Filter out non-finite values introduced by division
                    # finite_mask = np.isfinite(norm_residuals)
                    # clean_norm_residuals = norm_residuals[finite_mask]

                # Store normalized residuals for later plotting and consistent stats
                self.normalized_residuals[param] = clean_norm_residuals

                if clean_norm_residuals.size > 0:
                    mean_residual = np.mean(residuals)
                    std_residual = np.std(residuals)
                    num_tracks = clean_norm_residuals.size
                else:
                    mean_residual = np.nan
                    std_residual = np.nan
                    num_tracks = 0

                # Save statistics (these are for truth-normalized residuals to match plots)
                self.statistics[param] = {
                    'mean_residual': mean_residual,
                    'std_residual': std_residual,
                    'num_tracks': num_tracks
                }
                print(f"{param}: Mean normalized residual = {mean_residual:.4f}, STD = {std_residual:.4f}")
            else:
                self.statistics[param] = {
                    'mean_residual': np.nan,
                    'std_residual': np.nan,
                    'num_tracks': 0
                }
        
    def plot_residuals(self, param, output_subdir=None):
        """Plot residual distributions using step histogram style."""
        
        if len(self.predictions[param]) == 0:
            print(f"Warning: No data for {param} residuals")
            return
        
        # Use precomputed truth-normalized residuals (consistent with summary)
        clean_norm_residuals = self.normalized_residuals.get(param, np.array([]))
        if clean_norm_residuals.size == 0:
            print(f"Warning: No finite normalized residuals for {param}")
            return
        
        # Create step histogram plot
        plt.figure(figsize=(10, 6))
        
        # Define bins for the histogram
        bins = np.linspace(np.percentile(clean_norm_residuals, 1), 
                          np.percentile(clean_norm_residuals, 99), 50)
        
        # Calculate histogram
        counts, bin_edges = np.histogram(clean_norm_residuals, bins=bins, density=False)

        # Create step histogram with thinner lines
        plt.step(bin_edges[:-1], counts, where='post', linewidth=1, color='blue',
                 label=f'{param.capitalize()} Normalized Residuals')

        # Add vertical lines for statistics
        # Use precomputed statistics for consistency with summary
        mean_val = self.statistics[param]['mean_residual']
        std_val = self.statistics[param]['std_residual']
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.6f}')
        plt.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1,
                   label='Perfect agreement')

        # Formatting
        if param == 'q':
            plt.xlabel(f'{param.capitalize()} Absolute Residual (Pred - Truth)')
            plt.title(f'{param.capitalize()} Absolute Residual Distribution')
        else:
            plt.xlabel(f'{param.capitalize()} Normalized Residual (Pred - Truth)/|Truth|')
            plt.title(f'{param.capitalize()} Truth-Normalized Residual Distribution')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text box with statistics (use precomputed values for consistency)
        stats_text = f'Mean: {mean_val:.6f}\nSTD: {std_val:.6f}\nN: {len(clean_norm_residuals)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save plot
        if output_subdir:
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir

        output_path = output_dir / f'{param}_residuals.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved {param} residual plot to {output_path}")
        
    def plot_overlaid_distributions(self, param, output_subdir=None):
        """Plot overlaid distribution plots comparing model predictions with ground truth."""
        
        if len(self.predictions[param]) == 0:
            print(f"Warning: No data for {param} distribution plots")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Calculate appropriate bins with more resolution
        all_values = np.concatenate([self.predictions[param], self.truth[param]])
        
        # Set axis ranges to match filter evaluation
        if param == 'eta':
            bins = np.linspace(-3, 3, 100)  # More bins for better resolution
            plt.xlim(-3, 3)
        elif param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 100)
            plt.xlim(-np.pi, np.pi)
        elif param == 'pt':
            # For pt, use a reasonable range around the data
            p1, p99 = np.percentile(all_values, [1, 99])
            bins = np.linspace(p1, p99, 100)
        elif param == 'q':
            # For charge, use discrete bins centered around -1, 0, 1
            bins = np.linspace(-1.5, 1.5, 7)  # Creates bins: -1.5, -1, -0.5, 0, 0.5, 1, 1.5
            plt.xlim(-1.5, 1.5)
        else:
            bins = np.linspace(np.percentile(all_values, 1), np.percentile(all_values, 99), 100)

        # Plot both distributions with transparency (alpha) to make them properly overlaid
        plt.hist(self.truth[param], bins=bins, alpha=0.6, density=False, 
                label='Ground Truth', color='blue', histtype='stepfilled')
        plt.hist(self.predictions[param], bins=bins, alpha=0.6, density=False, 
                label='Model Predictions', color='red', histtype='stepfilled')
        
        plt.xlabel(f'{param.capitalize()}')
        plt.ylabel('Count')
        plt.title(f'{param.capitalize()} Distribution: Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if output_subdir:
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'{param}_distribution_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {param} distribution comparison to {output_path}")
    
    def plot_residuals_vs_kinematics(self, param, kinematic_var='pt', output_subdir=None):
        """Plot residuals vs kinematic variables."""
        
        if len(self.predictions[param]) == 0:
            print(f"Warning: No data for {param} vs {kinematic_var} plot")
            return
        
        # For charge, use absolute residuals instead of normalized residuals
        if param == 'q':
            residuals = self.predictions[param] - self.truth[param]  # Absolute residuals for charge
        else:
            residuals = (self.predictions[param] - self.truth[param]) / self.truth[param]  # Normalized residuals
        
        # Get kinematic variable values
        if kinematic_var == 'pt':
            kin_values = np.array([track['pt'] for track in self.track_info])
            # bins = np.logspace(0, 2, 11)  # 1 to 100 GeV
            bins = np.linspace(0, 200, 100)  # 1 to 100 GeV
            xlabel = '$p_T$ [GeV]'
            xscale = 'log'
        elif kinematic_var == 'eta':
            kin_values = np.array([track['eta'] for track in self.track_info])
            bins = np.linspace(-3, 3, 100)
            xlabel = '$\\eta$'
            xscale = 'linear'
        elif kinematic_var == 'phi':
            kin_values = np.array([track['phi'] for track in self.track_info])
            bins = np.linspace(-np.pi, np.pi, 100)
            xlabel = '$\\phi$ [rad]'
            xscale = 'linear'

        # Calculate mean and STD in bins
        bin_indices = np.digitize(kin_values, bins)
        # bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        bin_centers = []
        mean_residuals = []
        std_residuals = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            if mask.sum() > 0:
                bin_residuals = residuals[mask]
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                mean_residuals.append(np.mean(bin_residuals))
                std_residuals.append(np.std(bin_residuals))
        
        if len(bin_centers) == 0:
            print(f"Warning: No data points for {param} vs {kinematic_var}")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Mean residuals
        ax1.plot(bin_centers, mean_residuals, 'o-', color='blue', label='Mean residual')
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel(f'Mean {param} Residual')
        ax1.set_xscale(xscale)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title(f'{param.capitalize()} Mean Residual vs {kinematic_var.capitalize()}')
        
        # RMS residuals
        ax2.plot(bin_centers, std_residuals, 'o-', color='red', label='STD residual')
        ax2.set_ylabel(f'STD {param} Residual')
        ax2.set_xlabel(xlabel)
        ax2.set_xscale(xscale)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title(f'{param.capitalize()} STD Residual vs {kinematic_var.capitalize()}')
        
        plt.tight_layout()
        
        # Save the plot
        if output_subdir:
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.output_dir
            
        output_path = output_dir / f'regression_residuals_{param}_vs_{kinematic_var}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {param} vs {kinematic_var} residual plot to {output_path}")
    
    def run_evaluation(self):
        """Run the complete evaluation for Task 3."""
        print("=" * 80)
        print("TASK 3: REGRESSION EVALUATION")
        print("=" * 80)
        
        # Setup and collect data
        self.setup_data_module()
        self.collect_data()
        
        if len(self.track_info) == 0:
            print("Error: No data collected. Check file paths and data format.")
            return
        
        # Create plots
        print("\nGenerating plots...")
        
        # Residual plots for each parameter (step histogram style)
        for param in ['phi', 'eta', 'pt', 'q']:
            try:
                self.plot_residuals(param)
                print(f"✓ Created step histogram residual plot for {param}")
            except Exception as e:
                print(f"Error creating {param} residual plots: {e}")
        
        # Overlaid distribution plots
        for param in ['phi', 'eta', 'pt', 'q']:
            try:
                self.plot_overlaid_distributions(param)
                print(f"✓ Created overlaid distribution plot for {param}")
            except Exception as e:
                print(f"Error creating {param} distribution plots: {e}")
        
        # Remove old residuals vs kinematic plots as requested
        # (User requested "remove all the stuff you did for that task")
        
        # Write summary
        self.write_summary()
        
        print(f"\nTask 3 evaluation complete. Results saved to {self.output_dir}")
    
    def write_summary(self):
        """Write evaluation summary."""
        summary_path = self.output_dir / 'task3_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("TASK 3: REGRESSION EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Number of tracks processed: {len(self.track_info)}\n\n")
            
            f.write("REGRESSION STATISTICS:\n")
            f.write("-" * 25 + "\n")
            for param in ['phi', 'eta', 'pt', 'q']:
                stats = self.statistics.get(param, {})
                if stats.get('num_tracks', 0) > 0:
                    f.write(f"{param.upper()}:\n")
                    if param == 'q':
                        f.write(f"  Mean absolute residual: {stats['mean_residual']:.6f}\n")
                        f.write(f"  STD absolute residual:  {stats['std_residual']:.6f}\n")
                    else:
                        f.write(f"  Mean normalized residual: {stats['mean_residual']:.6f}\n")
                        f.write(f"  STD normalized residual:  {stats['std_residual']:.6f}\n")
                    f.write(f"  Number of tracks: {stats['num_tracks']}\n\n")
                else:
                    f.write(f"{param.upper()}: No data available\n\n")
            
            f.write("PLOTS GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("• Step histogram residual plots for phi, eta, pt, q\n")
            f.write("• Overlaid distribution comparisons for phi, eta, pt, q\n")
            f.write("• Truth-normalized residual analysis (absolute residuals for charge q)\n\n")
            
            f.write(f"Generated at: {datetime.now()}\n")
        
        print(f"Summary written to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 3: Regression Outputs')
    parser.add_argument('--eval_path', type=str, 
                    #    default="/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                    #    default="/scratch/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                    #    default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, 
                    #    default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_tracking_NGT_small2track_regression_inference.yaml",
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_tracking_NGT_small2track_regression_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task3_regression',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events', "-m", type=int, default=5000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("Task 3: Regression Evaluation")
    print("=" * 40)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    
    try:
        evaluator = Task3RegressionEvaluator(
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