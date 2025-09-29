#!/usr/bin/env python3
"""
Evaluation script for Task 3: Regression Outputs (parameter_regression) with Categories

This script evaluates the performance of the regression outputs by:
1. Creating truth-normalized residual plots for eta, phi, pt, and q
2. Creating correlation plots between predictions and truth
3. Analyzing the performance with three categories: all tracks, baseline tracks, rejected tracks
4. Applying baseline filtering criteria from Task 1

Based on lessons learned from Task 1 and Task 2 evaluation improvements.
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
    """Evaluator for regression outputs task with baseline filtering and categories."""
    
    def __init__(self, eval_path, data_dir, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task3_evaluation_{timestamp}"
        
        # Create output directory and subdirectories for categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks" 
        self.rejected_dir = self.output_dir / "rejected_tracks"
        
        for subdir in [self.all_tracks_dir, self.baseline_dir, self.rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Task 3 Evaluator initialized")
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
            num_workers=1,  # Reduced to 1 to avoid threading issues
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
        all_data = {'phi': [], 'eta': [], 'pt': [], 'q': [], 'phi_truth': [], 'eta_truth': [], 'pt_truth': [], 'q_truth': []}
        baseline_data = {'phi': [], 'eta': [], 'pt': [], 'q': [], 'phi_truth': [], 'eta_truth': [], 'pt_truth': [], 'q_truth': []}
        rejected_data = {'phi': [], 'eta': [], 'pt': [], 'q': [], 'phi_truth': [], 'eta_truth': [], 'pt_truth': [], 'q_truth': []}
        
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
                
                # Get regression predictions
                pred_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                pred_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                pred_pt = pred_group['preds/final/parameter_regression/track_truthMuon_pt'][...]  # Shape: (1, 2)
                pred_q = pred_group['preds/final/parameter_regression/track_truthMuon_q'][...]  # Shape: (1, 2)
                
                # Get track validity and hit assignments for baseline filtering
                track_hit_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                
                # Get truth
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
                
                # Process both potential tracks (max 2 tracks per event)
                for track_idx in range(2):
                    true_particle_exists = bool(true_particle_valid[track_idx])
                    
                    # Only process tracks with true particles
                    if not true_particle_exists:
                        continue
                    
                    # Get predicted parameters
                    pred_phi_val = float(pred_phi[0, track_idx])
                    pred_eta_val = float(pred_eta[0, track_idx])
                    pred_pt_val = float(pred_pt[0, track_idx])
                    pred_q_val = float(pred_q[0, track_idx])
                    
                    # Get truth parameters
                    truth_phi_val = targets["particle_truthMuon_phi"][0, track_idx].item()
                    truth_eta_val = targets["particle_truthMuon_eta"][0, track_idx].item()
                    truth_pt_val = targets["particle_truthMuon_pt"][0, track_idx].item()
                    truth_q_val = targets["particle_truthMuon_q"][0, track_idx].item()
                    
                    # Add to all tracks category
                    all_data['phi'].append(pred_phi_val)
                    all_data['eta'].append(pred_eta_val)
                    all_data['pt'].append(pred_pt_val)
                    all_data['q'].append(pred_q_val)
                    all_data['phi_truth'].append(truth_phi_val)
                    all_data['eta_truth'].append(truth_eta_val)
                    all_data['pt_truth'].append(truth_pt_val)
                    all_data['q_truth'].append(truth_q_val)
                    
                    # Apply baseline filtering
                    baseline_stats['total_tracks_checked'] += 1
                    passes_baseline = True
                    
                    # Get hit assignments for baseline filtering
                    true_hits = true_hit_assignments[track_idx].numpy().astype(bool)
                    
                    # Pre-filter 1: tracks must have at least 9 hits total
                    total_true_hits = true_hits.sum()
                    if total_true_hits < 9:
                        baseline_stats['tracks_failed_min_hits'] += 1
                        passes_baseline = False
                    
                    # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7
                    if passes_baseline and (np.abs(truth_eta_val) < 0.1 or np.abs(truth_eta_val) > 2.7):
                        baseline_stats['tracks_failed_eta_cuts'] += 1
                        passes_baseline = False
                    
                    # Pre-filter 3: pt threshold >= 3.0 GeV
                    if passes_baseline and truth_pt_val < 3.0:
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
                    
                    # Add to appropriate category
                    if passes_baseline:
                        # Baseline tracks (tracks that pass filtering)
                        baseline_data['phi'].append(pred_phi_val)
                        baseline_data['eta'].append(pred_eta_val)
                        baseline_data['pt'].append(pred_pt_val)
                        baseline_data['q'].append(pred_q_val)
                        baseline_data['phi_truth'].append(truth_phi_val)
                        baseline_data['eta_truth'].append(truth_eta_val)
                        baseline_data['pt_truth'].append(truth_pt_val)
                        baseline_data['q_truth'].append(truth_q_val)
                    else:
                        # Rejected tracks (tracks that fail filtering)
                        rejected_data['phi'].append(pred_phi_val)
                        rejected_data['eta'].append(pred_eta_val)
                        rejected_data['pt'].append(pred_pt_val)
                        rejected_data['q'].append(pred_q_val)
                        rejected_data['phi_truth'].append(truth_phi_val)
                        rejected_data['eta_truth'].append(truth_eta_val)
                        rejected_data['pt_truth'].append(truth_pt_val)
                        rejected_data['q_truth'].append(truth_q_val)
                
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
        print(f"  All tracks: {len(all_data['phi'])}")
        print(f"  Baseline tracks: {len(baseline_data['phi'])}")
        print(f"  Rejected tracks: {len(rejected_data['phi'])}")
        
        return all_data, baseline_data, rejected_data, baseline_stats
    
    def calculate_statistics(self, data, category_name):
        """Calculate regression statistics for a category."""
        statistics = {}
        
        params = ['phi', 'eta', 'pt', 'q']
        for param in params:
            if len(data[param]) > 0:
                predictions = data[param]
                truth = data[param + '_truth']
                
                # Raw residuals
                residuals = predictions - truth
                
                # For charge (q), use absolute residuals instead of normalized residuals
                if param == 'q':
                    # For charge, we use absolute residuals since truth is discrete (-1 or 1)
                    clean_norm_residuals = residuals  # No normalization for discrete charge
                else:
                    # Compute truth-normalized residuals using absolute truth to avoid sign flips
                    truth_abs = np.abs(truth)
                    # Avoid division by zero by replacing very small values with NaN
                    eps = 1e-12
                    denom = np.where(truth_abs > eps, truth_abs, np.nan)
                    clean_norm_residuals = residuals / denom
                    # Filter out non-finite values
                    clean_norm_residuals = clean_norm_residuals[np.isfinite(clean_norm_residuals)]

                if clean_norm_residuals.size > 0:
                    mean_residual = np.mean(clean_norm_residuals)
                    std_residual = np.std(clean_norm_residuals)
                    num_tracks = clean_norm_residuals.size
                else:
                    mean_residual = np.nan
                    std_residual = np.nan
                    num_tracks = 0

                statistics[param] = {
                    'mean_residual': mean_residual,
                    'std_residual': std_residual,
                    'num_tracks': num_tracks,
                    'normalized_residuals': clean_norm_residuals
                }
                print(f"{category_name} - {param}: Mean normalized residual = {mean_residual:.6f}, STD = {std_residual:.6f}")
            else:
                statistics[param] = {
                    'mean_residual': np.nan,
                    'std_residual': np.nan,
                    'num_tracks': 0,
                    'normalized_residuals': np.array([])
                }
        
        return statistics
    
    def plot_residuals(self, data, param, output_dir, category_name, statistics):
        """Plot residual distributions using step histogram style."""
        
        if len(data[param]) == 0:
            print(f"Warning: No data for {param} residuals in {category_name}")
            return
        
        # Use precomputed normalized residuals from statistics
        clean_norm_residuals = statistics[param]['normalized_residuals']
        if clean_norm_residuals.size == 0:
            print(f"Warning: No finite normalized residuals for {param} in {category_name}")
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
        mean_val = statistics[param]['mean_residual']
        std_val = statistics[param]['std_residual']
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.6f}')
        plt.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1,
                   label='Perfect agreement')

        # Formatting
        if param == 'q':
            plt.xlabel(f'{param.capitalize()} Absolute Residual (Pred - Truth)')
            plt.title(f'{param.capitalize()} Absolute Residual Distribution - {category_name}')
        else:
            plt.xlabel(f'{param.capitalize()} Normalized Residual (Pred - Truth)/|Truth|')
            plt.title(f'{param.capitalize()} Truth-Normalized Residual Distribution - {category_name}')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text box with statistics
        stats_text = f'Mean: {mean_val:.6f}\nSTD: {std_val:.6f}\nN: {len(clean_norm_residuals)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save plot
        output_path = output_dir / f'{param}_residuals.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved {category_name} {param} residual plot to {output_path}")
        
    def plot_overlaid_distributions(self, data, param, output_dir, category_name):
        """Plot overlaid distribution plots comparing model predictions with ground truth."""
        
        if len(data[param]) == 0:
            print(f"Warning: No data for {param} distribution plots in {category_name}")
            return
        
        plt.figure(figsize=(10, 6))
        
        predictions = data[param]
        truth = data[param + '_truth']
        
        # Calculate appropriate bins with more resolution
        all_values = np.concatenate([predictions, truth])
        
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
        plt.hist(truth, bins=bins, alpha=0.6, density=False, 
                label='Ground Truth', color='blue', histtype='stepfilled')
        plt.hist(predictions, bins=bins, alpha=0.6, density=False, 
                label='Model Predictions', color='red', histtype='stepfilled')
        
        plt.xlabel(f'{param.capitalize()}')
        plt.ylabel('Count')
        plt.title(f'{param.capitalize()} Distribution: Predictions vs Ground Truth - {category_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'{param}_distribution_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {category_name} {param} distribution comparison to {output_path}")
    
    def write_category_summary(self, data, statistics, output_dir, category_name):
        """Write evaluation summary for a category."""
        summary_path = output_dir / f'{category_name.lower().replace(" ", "_")}_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write(f"TASK 3: REGRESSION EVALUATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total tracks analyzed: {len(data['phi']):,}\n\n")
            
            f.write("REGRESSION STATISTICS\n")
            f.write("-" * 25 + "\n")
            for param in ['phi', 'eta', 'pt', 'q']:
                stats = statistics.get(param, {})
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
            
            f.write(f"\nGenerated at: {datetime.now()}\n")
        
        print(f"Summary for {category_name} written to {summary_path}")
    
    def write_comparative_summary(self, all_results, baseline_stats):
        """Write comprehensive summary comparing all categories."""
        summary_path = self.output_dir / 'task3_comparative_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK 3: REGRESSION EVALUATION - COMPARATIVE SUMMARY\n")
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
            
            # Write comparative metrics for each parameter
            params = ['phi', 'eta', 'pt', 'q']
            categories = ['All Tracks', 'Baseline Tracks', 'Rejected Tracks']
            
            for param in params:
                f.write(f"{param.upper()} COMPARATIVE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Category':<20}{'Num Tracks':<15}{'Mean Residual':<15}{'STD Residual':<15}\n")
                f.write("-" * 65 + "\n")
                
                for category in categories:
                    if category in all_results:
                        result = all_results[category]
                        if param in result:
                            stats = result[param]
                            f.write(f"{category:<20}{stats['num_tracks']:<15}{stats['mean_residual']:<15.6f}{stats['std_residual']:<15.6f}\n")
                        else:
                            f.write(f"{category:<20}{'0':<15}{'N/A':<15}{'N/A':<15}\n")
                    else:
                        f.write(f"{category:<20}{'0':<15}{'N/A':<15}{'N/A':<15}\n")
                f.write("\n")
        
        print(f"Comparative summary written to {summary_path}")
    
    def run_evaluation_with_categories(self):
        """Run evaluation for all categories."""
        print("=" * 80)
        print("TASK 3: REGRESSION EVALUATION WITH CATEGORIES")
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
            
            if len(data['phi']) == 0:
                print(f"Warning: No data for {category_name}")
                continue
            
            # Calculate statistics
            statistics = self.calculate_statistics(data, category_name)
            all_results[category_name] = statistics
            
            # Generate plots
            print("Generating plots...")
            
            # Residual plots for each parameter
            for param in ['phi', 'eta', 'pt', 'q']:
                try:
                    self.plot_residuals(data, param, output_dir, category_name, statistics)
                except Exception as e:
                    print(f"Error creating {param} residual plots: {e}")
            
            # Overlaid distribution plots
            for param in ['phi', 'eta', 'pt', 'q']:
                try:
                    self.plot_overlaid_distributions(data, param, output_dir, category_name)
                except Exception as e:
                    print(f"Error creating {param} distribution plots: {e}")
            
            # Write individual summary
            try:
                self.write_category_summary(data, statistics, output_dir, category_name)
            except Exception as e:
                print(f"Error writing summary for {category_name}: {e}")
        
        # Write comparative summary
        self.write_comparative_summary(all_results, baseline_stats)
        
        print(f"\nTask 3 evaluation with categories complete. Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Task 3: Regression Outputs with Categories')
    parser.add_argument('--eval_path', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
                       help='Path to processed test data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results/task3_regression',
                       help='Base output directory for plots and results')
    parser.add_argument('--max_events', "-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("Task 3: Regression Evaluation with Categories")
    print("=" * 50)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    
    try:
        evaluator = Task3RegressionEvaluator(
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