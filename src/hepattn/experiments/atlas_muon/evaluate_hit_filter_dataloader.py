#!/usr/bin/env python3
"""
Evaluation script for ATLAS muon hit filtering model using DataLoader approach.
This version uses the AtlasMuonDataModule for proper multi-worker data loading.

PERFORMANCE OPTIMIZATIONS:
- Memory-efficient data structures with proper dtypes
- Pre-allocated arrays and vectorized operations  
- Reduced redundant ROC calculations
- Optimized plotting with selective output
- Proper signal handling for background execution
- Memory monitoring and garbage collection
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import yaml
import warnings
from scipy.stats import binned_statistic
import traceback
from datetime import datetime
import pandas as pd
import gc
import signal
import sys
import os

# Memory monitoring
import psutil

# Import the data module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings('ignore')

# Configure signal handling for proper background execution
def signal_handler(signum, frame):
    print(f'\nReceived signal {signum}. Cleaning up and exiting...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Global configuration constants
DEFAULT_WORKING_POINTS = [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]

# Set matplotlib backend and style
plt.switch_backend('Agg')
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.grid': True, 
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'errorbar.capsize': 4
})


class AtlasMuonEvaluatorDataLoader:
    """Evaluation class for ATLAS muon hit filtering using DataLoader."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = Path(eval_path)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_events = max_events
        
        # Define sensor technologies mapping
        self.technology_mapping = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 4, "MM": 5}
        self.technology_names = list(self.technology_mapping.keys())
        
        # Initialize data module
        self.setup_data_module()
        
        # Storage for all collected data
        self.all_logits = None
        self.all_true_labels = None
        self.all_particle_pts = None
        self.all_particle_ids = None
        self.all_event_ids = None
    
    def setup_data_module(self):
        """Initialize the AtlasMuonDataModule with proper configuration."""
        print("Setting up data module...")
        
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract inputs and targets from config
        data_config = config.get('data', {})
        inputs = data_config.get('inputs', {})
        targets = data_config.get('targets', {})
        
        # Create fresh copies to avoid corruption
        inputs_eval = {k: list(v) for k, v in inputs.items()}
        targets_eval = {k: list(v) for k, v in targets.items()}
        
        # Set evaluation parameters
        # When max_events is None or -1, use all available events
        num_test_events = self.max_events if self.max_events is not None and self.max_events != -1 else -1
        
        # Initialize data module following the working example
        # Note: Even for test-only evaluation, we need to set num_train > 0
        self.data_module = AtlasMuonDataModule(
            train_dir=str(self.data_dir),
            val_dir=str(self.data_dir),
            test_dir=str(self.data_dir),
            num_workers=10,  # Use many workers for maximum speed
            num_train=abs(num_test_events) if num_test_events != -1 else 1000,  # Set to positive value
            num_val=abs(num_test_events) if num_test_events != -1 else 1000,   # Set to positive value
            num_test=num_test_events,  # -1 means use all available events
            batch_size=1,
            inputs=inputs_eval,
            targets=targets_eval,
            pin_memory=True,
        )
        
        # Setup the data module
        self.data_module.setup(stage='test')
        self.test_dataloader = self.data_module.test_dataloader()
        
        # print(f"DataLoader setup complete with 100 workers, processing {num_test_events} events")
    
    def collect_data(self): # verified: (check!)
        """Collect all data for analysis using the DataLoader with memory optimizations."""
        print("Collecting data from all events using DataLoader (memory optimized)...")
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # First, let's check what's in the evaluation file
        with h5py.File(self.eval_path, 'r') as eval_file:
            eval_keys = list(eval_file.keys())
            print(f"Evaluation file contains {len(eval_keys)} events")
        
        # Pre-allocate storage with estimated sizes (more memory efficient)
        estimated_hits = min(self.max_events * 1000 if self.max_events and self.max_events > 0 else 10000000, 50000000)
        
        # Use memory-efficient dtypes
        try:
            # Pre-allocate arrays with conservative size estimates
            all_logits = np.zeros(estimated_hits, dtype=np.float32)
            all_true_labels = np.zeros(estimated_hits, dtype=np.bool_)
            all_particle_pts = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_eta = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_phi = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_ids = np.zeros(estimated_hits, dtype=np.int32)
            all_particle_technology = np.zeros(estimated_hits, dtype=np.int8)
            all_event_ids = np.zeros(estimated_hits, dtype=np.int32)
            
            current_idx = 0
            events_processed = 0
            events_attempted = 0
            
        except MemoryError:
            print("ERROR: Not enough memory to pre-allocate arrays. Using list-based approach.")
            # Fall back to lists if pre-allocation fails
            all_logits = []
            all_true_labels = []
            all_particle_pts = []
            all_particle_eta = []
            all_particle_phi = []
            all_particle_ids = []
            all_particle_technology = []
            all_event_ids = []
            current_idx = None
        
        try:
            with h5py.File(self.eval_path, 'r') as eval_file:
                for batch_idx, batch_data in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                    events_attempted += 1
                    
                    # Only break if max_events is explicitly set (not None or -1)
                    if self.max_events is not None and self.max_events > 0 and events_processed >= self.max_events:
                        break
                    
                    # Memory monitoring every 1000 events
                    if events_processed % 1000 == 0 and events_processed > 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        print(f"Processed {events_processed} events. Memory: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
                        
                        # Force garbage collection if memory usage is high
                        if current_memory > 8000:  # 8GB threshold
                            gc.collect()
                    
                    try:
                        # Unpack batch data
                        if len(batch_data) == 2:
                            inputs_batch, targets_batch = batch_data
                        else:
                            print(f"Unexpected batch data structure: {len(batch_data)} elements")
                            continue
                        
                        # Extract event index
                        if "sample_id" not in targets_batch:
                            print(f"Warning: sample_id not found in targets, skipping batch {batch_idx}")
                            continue
                        
                        event_idx = targets_batch["sample_id"][0].item()
                        
                        # Load predictions for this event
                        if str(event_idx) not in eval_file:
                            continue
                        
                        # Get hit logits from evaluation file
                        hit_logits = eval_file[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)
                        
                        # Get truth labels from DataLoader
                        if "hit_on_valid_particle" not in targets_batch:
                            continue
                        
                        true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(np.bool_)
                        hit_particle_ids = inputs_batch["plotting_spacePoint_truthLink"][0].numpy().astype(np.int32)
                        hit_technologies = inputs_batch["hit_spacePoint_technology"][0].numpy().astype(np.int8)
                        
                        # Verify shapes match
                        n_hits = len(hit_logits)
                        if n_hits != len(true_labels) or n_hits != len(hit_particle_ids):
                            print(f"Warning: Shape mismatch in event {event_idx}")
                            continue
                        
                        # Get particle pt values
                        if "particle_truthMuon_pt" not in targets_batch:
                            continue
                        
                        particle_pts = targets_batch["particle_truthMuon_pt"][0].numpy().astype(np.float32)
                        particle_etas = targets_batch["particle_truthMuon_eta"][0].numpy().astype(np.float32)
                        particle_phis = targets_batch["particle_truthMuon_phi"][0].numpy().astype(np.float32)

                        # Map hits to particle pt values (vectorized for speed)
                        hit_pts = np.full(n_hits, -1.0, dtype=np.float32)  # Default for noise hits
                        hit_etas = np.full(n_hits, -1.0, dtype=np.float32)
                        hit_phis = np.full(n_hits, -1.0, dtype=np.float32)
                        
                        # Vectorized mapping using advanced indexing
                        unique_particle_ids = np.unique(hit_particle_ids)
                        valid_particle_ids = unique_particle_ids[unique_particle_ids >= 0]  # Skip -1 (noise)
                        
                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_pts):  # Safety check
                                hit_mask = hit_particle_ids == particle_id
                                hit_pts[hit_mask] = particle_pts[idx]
                                hit_etas[hit_mask] = particle_etas[idx]
                                hit_phis[hit_mask] = particle_phis[idx]

                        # Store data efficiently
                        if current_idx is not None:  # Pre-allocated arrays
                            if current_idx + n_hits >= len(all_logits):
                                # Resize arrays if needed (rare case)
                                new_size = max(len(all_logits) * 2, current_idx + n_hits + 100000)
                                all_logits = np.resize(all_logits, new_size)
                                all_true_labels = np.resize(all_true_labels, new_size)
                                all_particle_pts = np.resize(all_particle_pts, new_size)
                                all_particle_eta = np.resize(all_particle_eta, new_size)
                                all_particle_phi = np.resize(all_particle_phi, new_size)
                                all_particle_ids = np.resize(all_particle_ids, new_size)
                                all_particle_technology = np.resize(all_particle_technology, new_size)
                                all_event_ids = np.resize(all_event_ids, new_size)
                            
                            # Copy data to pre-allocated arrays
                            all_logits[current_idx:current_idx+n_hits] = hit_logits
                            all_true_labels[current_idx:current_idx+n_hits] = true_labels
                            all_particle_pts[current_idx:current_idx+n_hits] = hit_pts
                            all_particle_eta[current_idx:current_idx+n_hits] = hit_etas
                            all_particle_phi[current_idx:current_idx+n_hits] = hit_phis
                            all_particle_ids[current_idx:current_idx+n_hits] = hit_particle_ids
                            all_particle_technology[current_idx:current_idx+n_hits] = hit_technologies
                            all_event_ids[current_idx:current_idx+n_hits] = event_idx
                            current_idx += n_hits
                        else:
                            # Fall back to list append
                            all_logits.append(hit_logits)
                            all_true_labels.append(true_labels)
                            all_particle_pts.append(hit_pts)
                            all_particle_eta.append(hit_etas)
                            all_particle_phi.append(hit_phis)
                            all_particle_ids.append(hit_particle_ids)
                            all_particle_technology.append(hit_technologies)
                            all_event_ids.append(np.full(n_hits, event_idx, dtype=np.int32))
                        
                        events_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error during data collection: {e}")
            traceback.print_exc()
            return False
        
        print(f"\nDataLoader provided {events_attempted} batches, successfully processed {events_processed} events")
        
        if events_processed == 0:
            print("ERROR: No events were successfully processed!")
            return False
        
        # Convert to final numpy arrays
        if current_idx is not None:  # Pre-allocated arrays
            self.all_logits = all_logits[:current_idx]
            self.all_true_labels = all_true_labels[:current_idx]
            self.all_particle_pts = all_particle_pts[:current_idx]
            self.all_particle_etas = all_particle_eta[:current_idx]
            self.all_particle_phis = all_particle_phi[:current_idx]
            self.all_particle_ids = all_particle_ids[:current_idx]
            self.all_particle_technology = all_particle_technology[:current_idx]
            self.all_event_ids = all_event_ids[:current_idx]
        else:
            # Concatenate lists
            self.all_logits = np.concatenate(all_logits) if all_logits else np.array([])
            self.all_true_labels = np.concatenate(all_true_labels) if all_true_labels else np.array([], dtype=bool)
            self.all_particle_pts = np.concatenate(all_particle_pts) if all_particle_pts else np.array([])
            self.all_particle_etas = np.concatenate(all_particle_eta) if all_particle_eta else np.array([])
            self.all_particle_phis = np.concatenate(all_particle_phi) if all_particle_phi else np.array([])
            self.all_particle_ids = np.concatenate(all_particle_ids) if all_particle_ids else np.array([])
            self.all_particle_technology = np.concatenate(all_particle_technology) if all_particle_technology else np.array([])
            self.all_event_ids = np.concatenate(all_event_ids) if all_event_ids else np.array([])
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nData collection complete! Final memory: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
        print(f"Events processed: {events_processed}")
        print(f"Total hits: {len(self.all_logits):,}")
        print(f"True hits: {np.sum(self.all_true_labels):,}")
        print(f"Noise hits: {np.sum(~self.all_true_labels):,}")
        print(f"Valid particle hits (pt > 0): {np.sum(self.all_particle_pts > 0):,}")

        # Print pt statistics for valid particles
        valid_pt_mask = self.all_particle_pts > 0
        if np.any(valid_pt_mask):
            valid_pts = self.all_particle_pts[valid_pt_mask]
            print("\nPT statistics for valid particles:")
            print(f"  Min: {np.min(valid_pts):.1f} GeV")
            print(f"  Max: {np.max(valid_pts):.1f} GeV")
            print(f"  Mean: {np.mean(valid_pts):.1f} GeV")
            print(f"  Median: {np.median(valid_pts):.1f} GeV")
        
        # Force garbage collection
        gc.collect()
        
        return True
    
    def plot_roc_curve(self): # verified: (check!)
        """Generate ROC curve with AUC score."""
        print("Generating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='green', lw=0.8, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=0.8, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ATLAS Muon Hit Filter')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.1)
        
        # Save plot
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to {output_path}")
        print(f"AUC Score: {roc_auc:.4f}")
        
        return roc_auc
    
    def calculate_efficiency_by_pt(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_pt and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_pts = self.all_particle_pts[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_pts = self.all_particle_pts
        
        fpr, tpr, thresholds = roc_curve(true_labels, logits)
        
        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point
        
        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None
        
        threshold = thresholds[tpr >= target_efficiency][0]
        
        # Apply threshold to get predictions
        cut_predictions = logits >= threshold
        
        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)
        
        if total_predicted_positives > 0:
            overall_purity = total_true_positives / total_predicted_positives
        else:
            overall_purity = 0.0
        
        # Define pt bins
        pt_min, pt_max = 5.0, 200.0
        pt_bins = np.linspace(pt_min, pt_max, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []
        
        for i in range(len(pt_bins) - 1):
            pt_mask = (particle_pts >= pt_bins[i]) & (particle_pts < pt_bins[i+1])

            if not np.any(pt_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue
            
            bin_true_labels = true_labels[pt_mask]
            bin_predictions = cut_predictions[pt_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)
            
            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0
            
            efficiencies.append(efficiency)
            eff_errors.append(eff_error)
        
        return pt_centers, np.array(efficiencies), np.array(eff_errors), overall_purity
    
    def binomial_err(self, p, n):
        """Calculate binomial error"""
        return ((p*(1-p))/n)**0.5
    
    def plot_efficiency_vs_pt(self, working_points=None, skip_individual_plots=True):
        """
        Plot efficiency vs pT for different working points with overall purity in legend
        OPTIMIZATION: Skip individual plots by default to reduce file I/O
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        
        print("Generating efficiency plots (optimized)...")
        
        # Only create the main combined plots, skip individual plots by default
        self._plot_efficiency_vs_pt_general(working_points, skip_individual_plots)
        
        if not skip_individual_plots:
            # Only create technology-specific plots if explicitly requested
            self._plot_efficiency_vs_pt_by_technology(working_points)
    
    def _plot_efficiency_vs_pt_general(self, working_points, skip_individual_plots=True):
        """Create general efficiency vs pT plots (all technologies combined) - OPTIMIZED"""
        # Cache ROC calculation to avoid redundant computations
        if not hasattr(self, '_cached_roc'):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        # Prepare data for all working points (vectorized)
        results_dict = {}
        pt_min, pt_max = 5.0, 200.0
        pt_bins = np.linspace(pt_min, pt_max, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        print("Computing efficiency for all working points (vectorized)...")
        for wp in tqdm(working_points, desc="Working points"):
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                continue
            
            threshold = thresholds[tpr >= wp][0]
            cut_predictions = self.all_logits >= threshold
            
            # Calculate overall purity
            total_true_positives = np.sum(self.all_true_labels & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            
            if total_predicted_positives > 0:
                overall_purity = total_true_positives / total_predicted_positives
            else:
                overall_purity = 0.0
            
            # Vectorized efficiency calculation using scipy.stats.binned_statistic
            true_labels_array = self.all_true_labels.astype(float)
            predictions_array = cut_predictions.astype(float)
            
            # Mask for valid particles (avoid noise)
            valid_mask = self.all_particle_pts > 0
            
            if not np.any(valid_mask):
                continue
            
            valid_pts = self.all_particle_pts[valid_mask]
            valid_true = true_labels_array[valid_mask]
            valid_pred = predictions_array[valid_mask]
            
            # Calculate efficiency in bins using vectorized operations
            true_positive_counts, _, _ = binned_statistic(valid_pts, valid_true * valid_pred, 
                                                        statistic='sum', bins=pt_bins)
            total_positive_counts, _, _ = binned_statistic(valid_pts, valid_true, 
                                                         statistic='sum', bins=pt_bins)
            
            # Calculate efficiency and errors
            efficiencies = np.divide(true_positive_counts, total_positive_counts, 
                                   out=np.zeros_like(true_positive_counts), 
                                   where=total_positive_counts!=0)
            
            # Binomial errors (vectorized)
            eff_errors = np.sqrt(np.divide(efficiencies * (1 - efficiencies), total_positive_counts,
                                         out=np.zeros_like(efficiencies),
                                         where=total_positive_counts!=0))
            
            results_dict[f'WP {wp:.3f}'] = {
                'pt_bins': pt_bins,
                'pt_centers': pt_centers,
                'efficiency': efficiencies,
                'efficiency_err': eff_errors,
                'overall_purity': overall_purity,
                'counts': total_positive_counts.astype(int)
            }
        
        # Create the main combined plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color schemes for different working points
        colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
        
        for i, (wp_name, wp_data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Plot efficiency with step plot and error bands
            self._plot_metric_with_errors(ax, wp_data['pt_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
        
        # Format efficiency plot
        ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
        ax.set_title('ATLAS Muon Hit Filter - All Technologies', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_ylim([0.85, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"General efficiency vs pT plot saved to: {output_path}")
        
        # Only create individual plots if explicitly requested
        if not skip_individual_plots:
            self._plot_individual_working_points_pt(results_dict)
    
    def _plot_efficiency_vs_pt_by_technology(self, working_points):
        """Create efficiency vs pT plots for each sensor technology"""
        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency plots for {tech_name} technology...")
            
            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value
            
            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue
            
            print(f"Found {np.sum(tech_mask)} hits for {tech_name} technology")
            
            # Prepare data for all working points
            results_dict = {}
            
            for wp in working_points:
                pt_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_pt(wp, tech_mask)
                
                if pt_centers is None:
                    continue
                
                # Create pt bins from centers (approximate)
                pt_bins = np.zeros(len(pt_centers) + 1)
                if len(pt_centers) > 1:
                    bin_width = pt_centers[1] - pt_centers[0]
                    pt_bins[0] = pt_centers[0] - bin_width/2
                    for i in range(len(pt_centers)):
                        pt_bins[i+1] = pt_centers[i] + bin_width/2
                else:
                    pt_bins = np.array([0, 200])  # fallback
                
                results_dict[f'WP {wp:.3f}'] = {
                    'pt_bins': pt_bins,
                    'pt_centers': pt_centers,
                    'efficiency': efficiencies,
                    'efficiency_err': eff_errors,
                    'overall_purity': overall_purity,
                    'counts': None
                }
            
            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue
            
            # Create the main combined plot for this technology
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Color schemes for different working points
            colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
            
            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]
                
                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors(ax, wp_data['pt_bins'], wp_data['efficiency'], 
                                            wp_data['efficiency_err'], wp_data['counts'],
                                            color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            # Format efficiency plot
            ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} Technology', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"efficiency_vs_pt_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{tech_name} efficiency vs pT plot saved to: {output_path}")
            
            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_pt_technology(results_dict, tech_name)
    
    def _plot_metric_with_errors(self, ax, pt_bins, values, errors, counts, color, label, metric_type):
        """Helper function to plot metrics with error bands and step plots"""
        for i in range(len(pt_bins) - 1):
            lhs, rhs = pt_bins[i], pt_bins[i + 1]
            value = values[i]
            error = errors[i] if errors is not None else 0
            
            # Create error band
            if error > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                if metric_type == 'track_length':
                    # Don't cap values for track length plots
                    y_upper = value + error
                    y_lower = max(value - error, 0.0)  # Only floor at 0.0
                else:
                    # Cap efficiency values between 0 and 1
                    y_upper = min(value + error, 1.0)  # Cap at 1.0
                    y_lower = max(value - error, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label=f"binomial err - {label}" if i == 0 else "")
            
            # Step plot
            ax.step([lhs, rhs], [value, value], 
                   color=color, linewidth=2.5,
                   label=label if i == 0 else "")
    
    def _plot_individual_working_points_pt(self, results_dict):
        """Create separate efficiency plots for each working point (pt)"""
        # Create subdirectory for organizing plots
        efficiency_dir = self.output_dir / "efficiency_plots"
        efficiency_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            self._plot_metric_with_errors(ax, wp_data['pt_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'coral', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {wp_name}', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            
            plt.tight_layout()
            
            output_path = efficiency_dir / f"efficiency_vs_pt_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual pt working point plots saved to: {efficiency_dir}")
    
    def _plot_individual_working_points_pt_technology(self, results_dict, tech_name):
        """Create separate efficiency plots for each working point for a specific technology (pt)"""
        # Create subdirectory for organizing plots
        efficiency_dir = self.output_dir / f"efficiency_plots_{tech_name}"
        efficiency_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            self._plot_metric_with_errors(ax, wp_data['pt_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'coral', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} - {wp_name}', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            
            plt.tight_layout()
            
            output_path = efficiency_dir / f"efficiency_vs_pt_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual {tech_name} pt working point plots saved to: {efficiency_dir}")
    
    def plot_track_lengths(self):
        """
        Plot track lengths (number of hits per track) binned by pT, eta, and phi
        """
        print("Generating track length plots...")
        
        # Create track_lengths subdirectory
        track_lengths_dir = self.output_dir / "track_lengths"
        track_lengths_dir.mkdir(exist_ok=True)
        
        # Calculate track lengths for each unique (event_id, particle_id) combination
        unique_tracks = {}
        track_properties = {}
        
        # Group hits by (event_id, particle_id) to count hits per track
        # Only consider true hits for track length calculation
        for i in range(len(self.all_event_ids)):
            # Skip if this is not a true hit
            if not self.all_true_labels[i]:
                continue
                
            event_id = int(self.all_event_ids[i])
            particle_id = int(self.all_particle_ids[i])
            track_key = (event_id, particle_id)
            
            if track_key not in unique_tracks:
                unique_tracks[track_key] = 0
                # Store the first occurrence properties for this track
                track_properties[track_key] = {
                    'pt': self.all_particle_pts[i],
                    'eta': self.all_particle_etas[i], 
                    'phi': self.all_particle_phis[i]
                }
            
            unique_tracks[track_key] += 1
        
        # Extract track lengths and properties
        track_lengths = list(unique_tracks.values())
        track_pts = [track_properties[key]['pt'] for key in unique_tracks.keys()]
        track_etas = [track_properties[key]['eta'] for key in unique_tracks.keys()]
        track_phis = [track_properties[key]['phi'] for key in unique_tracks.keys()]
        
        track_lengths = np.array(track_lengths)
        track_pts = np.array(track_pts)
        track_etas = np.array(track_etas)
        track_phis = np.array(track_phis)
        
        print(f"Found {len(track_lengths)} unique tracks")
        print(f"Track length statistics: min={np.min(track_lengths)}, max={np.max(track_lengths)}, mean={np.mean(track_lengths):.1f}")
        
        # Plot track length vs pT
        self._plot_track_length_vs_pt(track_lengths, track_pts, track_lengths_dir)
        
        # Plot track length vs eta
        self._plot_track_length_vs_eta(track_lengths, track_etas, track_lengths_dir)
        
        # Plot track length vs phi
        self._plot_track_length_vs_phi(track_lengths, track_phis, track_lengths_dir)
    
    def _plot_track_length_vs_pt(self, track_lengths, track_pts, output_dir):
        """Plot average track length vs pT with binomial errors"""
        # Define pT bins (linear scale) - 0 to 200 GeV
        pt_bins = np.linspace(0, 200, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate average track length in each pT bin
        avg_lengths = []
        std_lengths = []
        
        for i in range(len(pt_bins) - 1):
            mask = (track_pts >= pt_bins[i]) & (track_pts < pt_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)
        
        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors(ax, pt_bins, avg_lengths, std_lengths, None,
                                    'royalblue', 'Average Track Length', 'track_length')
        
        ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Average Track Length [hits]', fontsize=14)
        ax.set_title('ATLAS Muon Track Length vs $p_T$', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_xlim([0, 200])
        
        plt.tight_layout()
        
        output_path = output_dir / "track_length_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Track length vs pT plot saved to: {output_path}")
    
    def _plot_track_length_vs_eta(self, track_lengths, track_etas, output_dir):
        """Plot average track length vs eta with binomial errors"""
        # Define eta bins - same as efficiency plots
        eta_bins = np.linspace(-2.7, 2.7, 21)  # 20 bins
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
        
        # Calculate average track length in each eta bin
        avg_lengths = []
        std_lengths = []
        
        for i in range(len(eta_bins) - 1):
            mask = (track_etas >= eta_bins[i]) & (track_etas < eta_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)
        
        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors_eta_phi(ax, eta_bins, avg_lengths, std_lengths, None,
                                            'forestgreen', 'Average Track Length', 'track_length')
        
        ax.set_xlabel('Truth Muon $\\eta$', fontsize=14)
        ax.set_ylabel('Average Track Length [hits]', fontsize=14)
        ax.set_title('ATLAS Muon Track Length vs $\\eta$', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_xlim([-2.7, 2.7])
        
        plt.tight_layout()
        
        output_path = output_dir / "track_length_vs_eta.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Track length vs eta plot saved to: {output_path}")
    
    def _plot_track_length_vs_phi(self, track_lengths, track_phis, output_dir):
        """Plot average track length vs phi with binomial errors"""
        # Define phi bins - same as efficiency plots
        phi_bins = np.linspace(-3.2, 3.2, 21)  # 20 bins
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
        
        # Calculate average track length in each phi bin
        avg_lengths = []
        std_lengths = []
        
        for i in range(len(phi_bins) - 1):
            mask = (track_phis >= phi_bins[i]) & (track_phis < phi_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)
        
        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors_eta_phi(ax, phi_bins, avg_lengths, std_lengths, None,
                                            'purple', 'Average Track Length', 'track_length')
        
        ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
        ax.set_ylabel('Average Track Length [hits]', fontsize=14)
        ax.set_title('ATLAS Muon Track Length vs $\\phi$', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_xlim([-3.2, 3.2])
        
        plt.tight_layout()
        
        output_path = output_dir / "track_length_vs_phi.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Track length vs phi plot saved to: {output_path}")

    def plot_working_point_performance(self, working_points=None):
        """
        Plot average purity for different working points with detailed track statistics.
        
        ULTRA-OPTIMIZATIONS APPLIED:
        1. Use cached ROC curve
        2. Pre-compute all thresholds and predictions in vectorized operations
        3. Use ultra-fast pandas groupby operations for track statistics
        4. Minimize memory allocations and copies
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        
        print("Generating working point performance plot (ultra-optimized)...")
        
        # Use cached ROC curve
        if not hasattr(self, '_cached_roc'):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        avg_purities = []
        avg_purity_errors = []
        
        # Pre-calculate all thresholds and predictions (vectorized)
        print("Pre-calculating thresholds and predictions (vectorized)...")
        thresholds_dict = {}
        predictions_dict = {}
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                avg_purities.append(0.0)
                avg_purity_errors.append(0.0)
                thresholds_dict[wp] = None
                predictions_dict[wp] = None
                continue
            
            threshold = thresholds[tpr >= wp][0]
            predictions = self.all_logits >= threshold
            thresholds_dict[wp] = threshold
            predictions_dict[wp] = predictions
            
            # Calculate overall purity (vectorized)
            total_true_positives = np.sum(self.all_true_labels & predictions)
            total_predicted_positives = np.sum(predictions)
            
            if total_predicted_positives > 0:
                purity = total_true_positives / total_predicted_positives
                purity_error = np.sqrt(purity * (1 - purity) / total_predicted_positives)
            else:
                purity = 0.0
                purity_error = 0.0
            
            avg_purities.append(purity)
            avg_purity_errors.append(purity_error)
        
        # Calculate detailed track statistics for all working points at once (ultra-fast)
        track_statistics = self._calculate_track_statistics_ultra_fast_optimized(working_points, predictions_dict)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(working_points, avg_purities, yerr=avg_purity_errors,
                    marker='o', capsize=4, linewidth=2, markersize=8,
                    color='darkred', label='Average Purity')
        
        plt.xlabel('Working Point (Target Efficiency)')
        plt.ylabel('Achieved Average Purity')
        plt.title('Working Point Performance - ATLAS Muon Hit Filter')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.4, 1.05)  # Zoom in on y-axis as requested
        
        # Save plot
        output_path = self.output_dir / "working_point_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Working point performance plot saved to {output_path}")
        
        # Save detailed statistics
        self._save_working_point_statistics(working_points, avg_purities, avg_purity_errors, track_statistics, output_path)
        
        # Print performance summary
        print("\nWorking Point Performance Summary:")
        for wp, purity, error in zip(working_points, avg_purities, avg_purity_errors):
            print(f"  WP {wp:.3f}: Purity = {purity:.4f} ± {error:.4f}")
    
    def _calculate_track_statistics_ultra_fast_optimized(self, working_points, predictions_dict):
        """Ultra-optimized pandas-based calculation of track statistics for all working points."""
        print("Calculating track statistics (ultra-fast optimized)...")
        
        # Pre-filter data to reduce memory usage and computation time
        valid_particle_mask = self.all_particle_ids >= 0  # Remove noise hits
        
        if not np.any(valid_particle_mask):
            # Return empty statistics if no valid particles
            return {wp: {'total_tracks': 0, 'tracks_with_few_hits': 0, 
                        'tracks_completely_lost': 0, 'events_analyzed': 0} 
                   for wp in working_points}
        
        # Create a DataFrame only with valid particles for maximum speed
        df = pd.DataFrame({
            'event_id': self.all_event_ids[valid_particle_mask],
            'particle_id': self.all_particle_ids[valid_particle_mask],
            'true_label': self.all_true_labels[valid_particle_mask],
            'original_idx': np.where(valid_particle_mask)[0]  # Track original indices efficiently
        })
        
        # Group by track and get track-level true hit counts (ultra-fast)
        track_info = df.groupby(['event_id', 'particle_id'], sort=False)['true_label'].sum()
        
        # Only keep tracks that have at least one true hit
        valid_tracks = track_info[track_info > 0].index
        valid_tracks_set = set(valid_tracks)
        
        print(f"Found {len(valid_tracks_set)} valid tracks to analyze")
        
        track_statistics = {}
        
        for wp in working_points:
            if predictions_dict[wp] is None:
                track_statistics[wp] = {
                    'total_tracks': 0,
                    'tracks_with_few_hits': 0,
                    'tracks_completely_lost': 0,
                    'events_analyzed': len(np.unique(self.all_event_ids))
                }
                continue
            
            # Add predictions to dataframe using original indices (vectorized)
            df_wp = df.copy()
            df_wp['predicted'] = predictions_dict[wp][df_wp['original_idx']]
            
            # Group by track and count predicted hits (ultra-fast)
            track_predictions = df_wp.groupby(['event_id', 'particle_id'], sort=False)['predicted'].sum()
            
            # Filter to only valid tracks using set intersection (ultra-fast)
            valid_track_predictions = track_predictions.loc[track_predictions.index.intersection(valid_tracks)]
            
            # Calculate statistics using vectorized pandas operations
            total_tracks = len(valid_track_predictions)
            if total_tracks > 0:
                tracks_completely_lost = int((valid_track_predictions == 0).sum())
                tracks_with_few_hits = int(((valid_track_predictions > 0) & (valid_track_predictions < 3)).sum())
            else:
                tracks_completely_lost = 0
                tracks_with_few_hits = 0
            
            track_statistics[wp] = {
                'total_tracks': total_tracks,
                'tracks_with_few_hits': tracks_with_few_hits,
                'tracks_completely_lost': tracks_completely_lost,
                'events_analyzed': len(np.unique(self.all_event_ids))
            }
            
            if total_tracks > 0:
                print(f"  WP {wp:.3f}: {total_tracks} tracks, {tracks_completely_lost} lost ({tracks_completely_lost/total_tracks*100:.1f}%)")
        
        return track_statistics

    def _calculate_track_statistics_vectorized(self, working_points, predictions_dict):
        """Vectorized calculation of track statistics for all working points."""
        print("Calculating track statistics (vectorized)...")
        
        # Pre-compute masks and data structures
        unique_events = np.unique(self.all_event_ids)
        events_analyzed = len(unique_events)
        
        # Initialize results dictionary
        track_statistics = {}
        
        # Create event-to-index mapping for faster lookup
        event_to_indices = {}
        for i, event_id in enumerate(unique_events):
            event_mask = self.all_event_ids == event_id
            event_to_indices[event_id] = np.where(event_mask)[0]
        
        for wp in working_points:
            if predictions_dict[wp] is None:
                track_statistics[wp] = {
                    'total_tracks': 0,
                    'tracks_with_few_hits': 0,
                    'tracks_completely_lost': 0,
                    'events_analyzed': events_analyzed
                }
                continue
            
            predictions = predictions_dict[wp]
            
            total_tracks = 0
            tracks_with_few_hits = 0
            tracks_completely_lost = 0
            
            # Process events in batches for better memory efficiency
            for event_id in unique_events:
                event_indices = event_to_indices[event_id]
                
                # Get data for this event using pre-computed indices
                event_particle_ids = self.all_particle_ids[event_indices]
                event_predictions = predictions[event_indices]
                event_true_labels = self.all_true_labels[event_indices]
                
                # Get unique particles (tracks) excluding noise (-1)
                unique_particles = np.unique(event_particle_ids)
                unique_particles = unique_particles[unique_particles >= 0]
                
                # Vectorized processing of particles in this event
                for particle_id in unique_particles:
                    particle_mask = event_particle_ids == particle_id
                    particle_true_hits = event_true_labels[particle_mask]
                    
                    # Skip if no true hits for this particle
                    if np.sum(particle_true_hits) == 0:
                        continue
                    
                    total_tracks += 1
                    
                    # Count predicted hits for this track
                    num_predicted_hits = np.sum(event_predictions[particle_mask])
                    
                    if num_predicted_hits == 0:
                        tracks_completely_lost += 1
                    elif num_predicted_hits < 3:
                        tracks_with_few_hits += 1
            
            track_statistics[wp] = {
                'total_tracks': total_tracks,
                'tracks_with_few_hits': tracks_with_few_hits,
                'tracks_completely_lost': tracks_completely_lost,
                'events_analyzed': events_analyzed
            }
            
            print(f"  WP {wp:.3f}: {total_tracks} tracks processed")
        
        return track_statistics

    def _calculate_technology_statistics(self):
        """Calculate technology distribution in the truth labels."""
        tech_stats = {}
        total_true_hits = np.sum(self.all_true_labels)
        
        # Calculate statistics for each technology
        for tech_name, tech_value in self.technology_mapping.items():
            # Create mask for this technology
            tech_mask = self.all_particle_technology == tech_value
            
            # Count true hits for this technology
            tech_true_hits = np.sum(self.all_true_labels & tech_mask)
            total_tech_hits = np.sum(tech_mask)
            
            # Calculate percentage of total true hits
            percentage_of_true = (tech_true_hits / total_true_hits * 100) if total_true_hits > 0 else 0.0
            
            tech_stats[tech_name] = {
                'true_hits': int(tech_true_hits),
                'total_hits': int(total_tech_hits),
                'percentage_of_true_hits': percentage_of_true
            }
        
        return tech_stats

    def _calculate_track_statistics_per_working_point(self, working_point, threshold, predictions):
        """Calculate detailed track statistics for a specific working point."""
        # Get unique events
        unique_events = np.unique(self.all_event_ids)
        
        total_tracks = 0
        tracks_with_few_hits = 0  # tracks with < 3 hits after filtering
        tracks_completely_lost = 0  # tracks with 0 hits after filtering
        events_analyzed = len(unique_events)
        
        for event_id in unique_events:
            # Get hits for this event
            event_mask = self.all_event_ids == event_id
            event_particle_ids = self.all_particle_ids[event_mask]
            event_predictions = predictions[event_mask]
            event_true_labels = self.all_true_labels[event_mask]
            
            # Get unique particle IDs (tracks) in this event, excluding noise (-1)
            unique_particles = np.unique(event_particle_ids)
            unique_particles = unique_particles[unique_particles >= 0]  # Remove noise hits
            
            for particle_id in unique_particles:
                # Check if this particle has any true hits (is a valid track)
                particle_mask = event_particle_ids == particle_id
                particle_true_hits = event_true_labels[particle_mask]
                
                if np.sum(particle_true_hits) == 0:
                    continue  # Skip if no true hits for this particle
                
                total_tracks += 1
                
                # Count predicted hits for this track
                particle_predicted_hits = event_predictions[particle_mask]
                num_predicted_hits = np.sum(particle_predicted_hits)
                
                if num_predicted_hits == 0:
                    tracks_completely_lost += 1
                elif num_predicted_hits < 3:
                    tracks_with_few_hits += 1
        
        return {
            'total_tracks': total_tracks,
            'tracks_with_few_hits': tracks_with_few_hits,
            'tracks_completely_lost': tracks_completely_lost,
            'events_analyzed': events_analyzed
        }
    
    def _save_working_point_statistics(self, working_points, purities, purity_errors, track_statistics, output_plot_path):
        """
        Save comprehensive working point statistics to a text file.
        
        Parameters:
        -----------
        working_points : list
            List of working point values
        purities : list
            List of purity values for each working point
        purity_errors : list
            List of purity error values for each working point
        track_statistics : dict
            Dictionary containing track statistics for each working point
        output_plot_path : Path
            Path where the plot is saved, used to determine where to save the statistics file
        """
        from datetime import datetime
        
        # Determine output directory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_plot_path:
            # Save the txt file in the same directory as the plot
            plot_path = Path(output_plot_path)
            output_dir = plot_path.parent
            filename = output_dir / f"working_point_statistics_{timestamp}.txt"
        else:
            # Default to current directory if no plot path provided
            filename = f"working_point_statistics_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WORKING POINT PERFORMANCE STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events analyzed: {self.max_events}\n\n")
            
            # Overall summary
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            for wp, purity, error in zip(working_points, purities, purity_errors):
                f.write(f"Working Point {wp:.3f}: Purity = {purity:.4f} ± {error:.4f}\n")
            f.write("\n")
            
            # Technology distribution in truth labels
            tech_stats = self._calculate_technology_statistics()
            f.write("TECHNOLOGY DISTRIBUTION IN TRUTH LABELS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Technology':<12} {'True Hits':<12} {'Total Hits':<12} {'% of True':<12}\n")
            f.write("-" * 50 + "\n")
            
            for tech_name, stats in tech_stats.items():
                f.write(f"{tech_name:<12} {stats['true_hits']:<12,} {stats['total_hits']:<12,} {stats['percentage_of_true_hits']:<12.1f}%\n")
            
            f.write(f"\nTotal true hits across all technologies: {np.sum(self.all_true_labels):,}\n")
            f.write(f"Total hits across all technologies: {len(self.all_true_labels):,}\n")
            f.write(f"Overall true hit rate: {(np.sum(self.all_true_labels) / len(self.all_true_labels) * 100):.1f}%\n\n")
            
            # Detailed statistics for each working point
            f.write("DETAILED TRACK STATISTICS BY WORKING POINT:\n")
            f.write("=" * 60 + "\n")
            
            for wp in working_points:
                stats = track_statistics[wp]
                f.write(f"\nWorking Point {wp:.3f}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Events analyzed: {stats['events_analyzed']}\n")
                f.write(f"Total valid tracks: {stats['total_tracks']}\n")
                
                if stats['total_tracks'] > 0:
                    # Tracks completely lost
                    lost_percentage = (stats['tracks_completely_lost'] / stats['total_tracks']) * 100
                    f.write(f"Tracks completely lost (0 hits): {stats['tracks_completely_lost']} ({lost_percentage:.2f}%)\n")
                    
                    # Tracks with few hits
                    few_hits_percentage = (stats['tracks_with_few_hits'] / stats['total_tracks']) * 100
                    f.write(f"Tracks with <3 hits: {stats['tracks_with_few_hits']} ({few_hits_percentage:.2f}%)\n")
                    
                    # Tracks with ≥3 hits (good tracks)
                    good_tracks = stats['total_tracks'] - stats['tracks_completely_lost'] - stats['tracks_with_few_hits']
                    good_tracks_percentage = (good_tracks / stats['total_tracks']) * 100
                    f.write(f"Tracks with ≥3 hits: {good_tracks} ({good_tracks_percentage:.2f}%)\n")
                    
                    # Track survival rate
                    survived_tracks = stats['total_tracks'] - stats['tracks_completely_lost']
                    survival_rate = (survived_tracks / stats['total_tracks']) * 100
                    f.write(f"Track survival rate: {survival_rate:.2f}%\n")
                else:
                    f.write("No valid tracks found\n")
            
            # Comparison table
            f.write("\n" + "=" * 80 + "\n")
            f.write("COMPARISON TABLE:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'WP':<6} {'Purity':<8} {'Total':<7} {'Lost':<6} {'Lost%':<7} {'<3hits':<7} {'<3hits%':<8} {'≥3hits':<7} {'≥3hits%':<8}\n")
            f.write("-" * 80 + "\n")
            
            for wp in working_points:
                stats = track_statistics[wp]
                purity = next(p for w, p in zip(working_points, purities) if w == wp)
                
                if stats['total_tracks'] > 0:
                    lost_pct = (stats['tracks_completely_lost'] / stats['total_tracks']) * 100
                    few_hits_pct = (stats['tracks_with_few_hits'] / stats['total_tracks']) * 100
                    good_tracks = stats['total_tracks'] - stats['tracks_completely_lost'] - stats['tracks_with_few_hits']
                    good_pct = (good_tracks / stats['total_tracks']) * 100
                    
                    f.write(f"{wp:<6.3f} {purity:<8.4f} {stats['total_tracks']:<7} {stats['tracks_completely_lost']:<6} "
                           f"{lost_pct:<7.1f} {stats['tracks_with_few_hits']:<7} {few_hits_pct:<8.1f} "
                           f"{good_tracks:<7} {good_pct:<8.1f}\n")
                else:
                    f.write(f"{wp:<6.3f} {purity:<8.4f} {stats['total_tracks']:<7} -      -       -       -        -       -\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("LEGEND:\n")
            f.write("WP      = Working Point (target efficiency)\n")
            f.write("Purity  = Hit filter purity (precision)\n")
            f.write("Total   = Total number of valid tracks\n")
            f.write("Lost    = Tracks with 0 predicted hits\n")
            f.write("Lost%   = Percentage of tracks completely lost\n")
            f.write("<3hits  = Tracks with 1-2 predicted hits\n")
            f.write("<3hits% = Percentage of tracks with <3 hits\n")
            f.write("≥3hits  = Tracks with 3 or more predicted hits\n")
            f.write("≥3hits% = Percentage of tracks with ≥3 hits\n")
            f.write("\nTECHNOLOGY CODES:\n")
            for tech_name, tech_value in self.technology_mapping.items():
                f.write(f"{tech_name:<6} = {tech_value}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nWorking point statistics saved to: {filename}")
        
        # Print summary to console
        if len(working_points) > 0:
            best_wp_idx = np.argmax(purities)
            best_wp = working_points[best_wp_idx]
            best_stats = track_statistics[best_wp]
            
            print(f"\nSummary:")
            print(f"  - Best working point: {best_wp:.3f} (purity: {purities[best_wp_idx]:.3f})")
            if best_stats['total_tracks'] > 0:
                survival_rate = ((best_stats['total_tracks'] - best_stats['tracks_completely_lost']) / best_stats['total_tracks']) * 100
                print(f"  - Track survival rate at best WP: {survival_rate:.1f}%")
                good_tracks = best_stats['total_tracks'] - best_stats['tracks_completely_lost'] - best_stats['tracks_with_few_hits']
                good_rate = (good_tracks / best_stats['total_tracks']) * 100
                print(f"  - Tracks with ≥3 hits at best WP: {good_rate:.1f}%")
    
    def calculate_efficiency_by_eta(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_eta and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_etas = self.all_particle_etas[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_etas = self.all_particle_etas
        
        fpr, tpr, thresholds = roc_curve(true_labels, logits)
        
        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point
        
        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None
        
        threshold = thresholds[tpr >= target_efficiency][0]
        
        # Apply threshold to get predictions
        cut_predictions = logits >= threshold
        
        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)
        
        if total_predicted_positives > 0:
            overall_purity = total_true_positives / total_predicted_positives
        else:
            overall_purity = 0.0
        
        # Define eta bins
        eta_min, eta_max = -2.7, 2.7
        eta_bins = np.linspace(eta_min, eta_max, 21)  # 20 bins
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
        
        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []
        
        for i in range(len(eta_bins) - 1):
            eta_mask = (particle_etas >= eta_bins[i]) & (particle_etas < eta_bins[i+1])

            if not np.any(eta_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue
            
            bin_true_labels = true_labels[eta_mask]
            bin_predictions = cut_predictions[eta_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)
            
            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0
            
            efficiencies.append(efficiency)
            eff_errors.append(eff_error)
        
        return eta_centers, np.array(efficiencies), np.array(eff_errors), overall_purity
    
    def calculate_efficiency_by_phi(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_phi and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_phis = self.all_particle_phis[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_phis = self.all_particle_phis
        
        fpr, tpr, thresholds = roc_curve(true_labels, logits)
        
        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point
        
        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None
        
        threshold = thresholds[tpr >= target_efficiency][0]
        
        # Apply threshold to get predictions
        cut_predictions = logits >= threshold
        
        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)
        
        if total_predicted_positives > 0:
            overall_purity = total_true_positives / total_predicted_positives
        else:
            overall_purity = 0.0
        
        # Define phi bins
        phi_min, phi_max = -3.2, 3.2
        phi_bins = np.linspace(phi_min, phi_max, 21)  # 20 bins
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
        
        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []
        
        for i in range(len(phi_bins) - 1):
            phi_mask = (particle_phis >= phi_bins[i]) & (particle_phis < phi_bins[i+1])

            if not np.any(phi_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue
            
            bin_true_labels = true_labels[phi_mask]
            bin_predictions = cut_predictions[phi_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)
            
            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0
            
            efficiencies.append(efficiency)
            eff_errors.append(eff_error)
        
        return phi_centers, np.array(efficiencies), np.array(eff_errors), overall_purity
    
    def plot_efficiency_vs_eta(self, working_points=None):
        """
        Plot efficiency vs eta for different working points with overall purity in legend
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs eta plots...")
        
        # First, create the general (all technologies) plots
        self._plot_efficiency_vs_eta_general(working_points)
        
        # Then, create technology-specific plots
        self._plot_efficiency_vs_eta_by_technology(working_points)
    
    def _plot_efficiency_vs_eta_general(self, working_points, skip_individual_plots=True):
        """Create general efficiency vs eta plots (all technologies combined) - OPTIMIZED"""
        # Use cached ROC calculation  
        if not hasattr(self, '_cached_roc'):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        # Prepare data for all working points (vectorized)
        results_dict = {}
        eta_min, eta_max = -2.7, 2.7
        eta_bins = np.linspace(eta_min, eta_max, 21)  # 20 bins
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                continue
            
            threshold = thresholds[tpr >= wp][0]
            cut_predictions = self.all_logits >= threshold
            
            # Calculate overall purity
            total_true_positives = np.sum(self.all_true_labels & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            
            if total_predicted_positives > 0:
                overall_purity = total_true_positives / total_predicted_positives
            else:
                overall_purity = 0.0
            
            # Vectorized efficiency calculation
            true_labels_array = self.all_true_labels.astype(float)
            predictions_array = cut_predictions.astype(float)
            
            # Mask for valid particles (avoid noise)
            valid_mask = self.all_particle_etas > -999  # Basic validity check
            
            if not np.any(valid_mask):
                continue
            
            valid_etas = self.all_particle_etas[valid_mask]
            valid_true = true_labels_array[valid_mask]
            valid_pred = predictions_array[valid_mask]
            
            # Calculate efficiency in bins using vectorized operations
            true_positive_counts, _, _ = binned_statistic(valid_etas, valid_true * valid_pred, 
                                                        statistic='sum', bins=eta_bins)
            total_positive_counts, _, _ = binned_statistic(valid_etas, valid_true, 
                                                         statistic='sum', bins=eta_bins)
            
            # Calculate efficiency and errors
            efficiencies = np.divide(true_positive_counts, total_positive_counts, 
                                   out=np.zeros_like(true_positive_counts), 
                                   where=total_positive_counts!=0)
            
            # Binomial errors (vectorized)
            eff_errors = np.sqrt(np.divide(efficiencies * (1 - efficiencies), total_positive_counts,
                                         out=np.zeros_like(efficiencies),
                                         where=total_positive_counts!=0))
            
            results_dict[f'WP {wp:.3f}'] = {
                'eta_bins': eta_bins,
                'eta_centers': eta_centers,
                'efficiency': efficiencies,
                'efficiency_err': eff_errors,
                'overall_purity': overall_purity,
                'counts': total_positive_counts.astype(int)
            }
        
        # Create the main combined plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color schemes for different working points
        colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
        
        for i, (wp_name, wp_data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Plot efficiency with step plot and error bands
            self._plot_metric_with_errors_eta_phi(ax, wp_data['eta_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
        
        # Format efficiency plot
        ax.set_xlabel('Truth Muon $\\eta$', fontsize=14)
        ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
        ax.set_title('ATLAS Muon Hit Filter - All Technologies', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_ylim([0.85, 1.05])
        ax.set_xlim([-2.7, 2.7])
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_vs_eta.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"General efficiency vs eta plot saved to: {output_path}")
        
        # Only create individual plots if explicitly requested
        if not skip_individual_plots:
            self._plot_individual_working_points_eta(results_dict)
    
    def _plot_efficiency_vs_phi_general(self, working_points, skip_individual_plots=True):
        """Create general efficiency vs phi plots (all technologies combined) - OPTIMIZED"""
        # Use cached ROC calculation  
        if not hasattr(self, '_cached_roc'):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        # Prepare data for all working points (vectorized)
        results_dict = {}
        phi_min, phi_max = -3.2, 3.2
        phi_bins = np.linspace(phi_min, phi_max, 21)  # 20 bins
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                continue
            
            threshold = thresholds[tpr >= wp][0]
            cut_predictions = self.all_logits >= threshold
            
            # Calculate overall purity
            total_true_positives = np.sum(self.all_true_labels & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            
            if total_predicted_positives > 0:
                overall_purity = total_true_positives / total_predicted_positives
            else:
                overall_purity = 0.0
            
            # Vectorized efficiency calculation
            true_labels_array = self.all_true_labels.astype(float)
            predictions_array = cut_predictions.astype(float)
            
            # Mask for valid particles (avoid noise)
            valid_mask = self.all_particle_phis > -999  # Basic validity check
            
            if not np.any(valid_mask):
                continue
            
            valid_phis = self.all_particle_phis[valid_mask]
            valid_true = true_labels_array[valid_mask]
            valid_pred = predictions_array[valid_mask]
            
            # Calculate efficiency in bins using vectorized operations
            true_positive_counts, _, _ = binned_statistic(valid_phis, valid_true * valid_pred, 
                                                        statistic='sum', bins=phi_bins)
            total_positive_counts, _, _ = binned_statistic(valid_phis, valid_true, 
                                                         statistic='sum', bins=phi_bins)
            
            # Calculate efficiency and errors
            efficiencies = np.divide(true_positive_counts, total_positive_counts, 
                                   out=np.zeros_like(true_positive_counts), 
                                   where=total_positive_counts!=0)
            
            # Binomial errors (vectorized)
            eff_errors = np.sqrt(np.divide(efficiencies * (1 - efficiencies), total_positive_counts,
                                         out=np.zeros_like(efficiencies),
                                         where=total_positive_counts!=0))
            
            results_dict[f'WP {wp:.3f}'] = {
                'phi_bins': phi_bins,
                'phi_centers': phi_centers,
                'efficiency': efficiencies,
                'efficiency_err': eff_errors,
                'overall_purity': overall_purity,
                'counts': total_positive_counts.astype(int)
            }
        
        # Create the main combined plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color schemes for different working points
        colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
        
        for i, (wp_name, wp_data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Plot efficiency with step plot and error bands
            self._plot_metric_with_errors_eta_phi(ax, wp_data['phi_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
        
        # Format efficiency plot
        ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
        ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
        ax.set_title('ATLAS Muon Hit Filter - All Technologies', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_ylim([0.85, 1.05])
        ax.set_xlim([-3.2, 3.2])
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_vs_phi.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"General efficiency vs phi plot saved to: {output_path}")
        
        # Only create individual plots if explicitly requested
        if not skip_individual_plots:
            self._plot_individual_working_points_phi(results_dict)
    
    def _plot_efficiency_vs_eta_by_technology(self, working_points):
        """Create efficiency vs eta plots for each sensor technology"""
        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency vs eta plots for {tech_name} technology...")
            
            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value
            
            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue
            
            # Prepare data for all working points
            results_dict = {}
            
            for wp in working_points:
                eta_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_eta(wp, tech_mask)
                
                if eta_centers is None:
                    continue
                
                # Create eta bins from centers (approximate)
                eta_bins = np.zeros(len(eta_centers) + 1)
                if len(eta_centers) > 1:
                    bin_width = eta_centers[1] - eta_centers[0]
                    eta_bins[0] = eta_centers[0] - bin_width/2
                    for i in range(len(eta_centers)):
                        eta_bins[i+1] = eta_centers[i] + bin_width/2
                else:
                    eta_bins = np.array([-2.7, 2.7])  # fallback
                
                results_dict[f'WP {wp:.3f}'] = {
                    'eta_bins': eta_bins,
                    'eta_centers': eta_centers,
                    'efficiency': efficiencies,
                    'efficiency_err': eff_errors,
                    'overall_purity': overall_purity,
                    'counts': None
                }
            
            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue
            
            # Create the main combined plot for this technology
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Color schemes for different working points
            colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
            
            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]
                
                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors_eta_phi(ax, wp_data['eta_bins'], wp_data['efficiency'], 
                                            wp_data['efficiency_err'], wp_data['counts'],
                                            color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            # Format efficiency plot
            ax.set_xlabel('Truth Muon $\\eta$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} Technology', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"efficiency_vs_eta_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{tech_name} efficiency vs eta plot saved to: {output_path}")
            
            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_eta_technology(results_dict, tech_name)
    
    def plot_efficiency_vs_phi(self, working_points=None):
        """
        Plot efficiency vs phi for different working points with overall purity in legend
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs phi plots...")
        
        # First, create the general (all technologies) plots
        self._plot_efficiency_vs_phi_general(working_points)
        
        # Then, create technology-specific plots
        self._plot_efficiency_vs_phi_by_technology(working_points)
    
    def _plot_efficiency_vs_phi_general(self, working_points, skip_individual_plots=True):
        """Create general efficiency vs phi plots (all technologies combined) - OPTIMIZED"""
        # Use cached ROC calculation  
        if not hasattr(self, '_cached_roc'):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        # Prepare data for all working points (vectorized)
        results_dict = {}
        phi_min, phi_max = -3.2, 3.2
        phi_bins = np.linspace(phi_min, phi_max, 21)  # 20 bins
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                continue
            
            threshold = thresholds[tpr >= wp][0]
            cut_predictions = self.all_logits >= threshold
            
            # Calculate overall purity
            total_true_positives = np.sum(self.all_true_labels & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            
            if total_predicted_positives > 0:
                overall_purity = total_true_positives / total_predicted_positives
            else:
                overall_purity = 0.0
            
            # Vectorized efficiency calculation
            true_labels_array = self.all_true_labels.astype(float)
            predictions_array = cut_predictions.astype(float)
            
            # Mask for valid particles (avoid noise)
            valid_mask = self.all_particle_phis > -999  # Basic validity check
            
            if not np.any(valid_mask):
                continue
            
            valid_phis = self.all_particle_phis[valid_mask]
            valid_true = true_labels_array[valid_mask]
            valid_pred = predictions_array[valid_mask]
            
            # Calculate efficiency in bins using vectorized operations
            true_positive_counts, _, _ = binned_statistic(valid_phis, valid_true * valid_pred, 
                                                        statistic='sum', bins=phi_bins)
            total_positive_counts, _, _ = binned_statistic(valid_phis, valid_true, 
                                                         statistic='sum', bins=phi_bins)
            
            # Calculate efficiency and errors
            efficiencies = np.divide(true_positive_counts, total_positive_counts, 
                                   out=np.zeros_like(true_positive_counts), 
                                   where=total_positive_counts!=0)
            
            # Binomial errors (vectorized)
            eff_errors = np.sqrt(np.divide(efficiencies * (1 - efficiencies), total_positive_counts,
                                         out=np.zeros_like(efficiencies),
                                         where=total_positive_counts!=0))
            
            results_dict[f'WP {wp:.3f}'] = {
                'phi_bins': phi_bins,
                'phi_centers': phi_centers,
                'efficiency': efficiencies,
                'efficiency_err': eff_errors,
                'overall_purity': overall_purity,
                'counts': total_positive_counts.astype(int)
            }
        
        # Create the main combined plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color schemes for different working points
        colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
        
        for i, (wp_name, wp_data) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Plot efficiency with step plot and error bands
            self._plot_metric_with_errors_eta_phi(ax, wp_data['phi_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
        
        # Format efficiency plot
        ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
        ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
        ax.set_title('ATLAS Muon Hit Filter - All Technologies', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.legend()
        ax.set_ylim([0.85, 1.05])
        ax.set_xlim([-3.2, 3.2])
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_vs_phi.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"General efficiency vs phi plot saved to: {output_path}")
        
        # Only create individual plots if explicitly requested
        if not skip_individual_plots:
            self._plot_individual_working_points_phi(results_dict)
    
    def _plot_efficiency_vs_phi_by_technology(self, working_points):
        """Create efficiency vs phi plots for each sensor technology"""
        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency vs phi plots for {tech_name} technology...")
            
            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value
            
            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue
            
            # Prepare data for all working points
            results_dict = {}
            
            for wp in working_points:
                phi_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_phi(wp, tech_mask)
                
                if phi_centers is None:
                    continue
                
                # Create phi bins from centers (approximate)
                phi_bins = np.zeros(len(phi_centers) + 1)
                if len(phi_centers) > 1:
                    bin_width = phi_centers[1] - phi_centers[0]
                    phi_bins[0] = phi_centers[0] - bin_width/2
                    for i in range(len(phi_centers)):
                        phi_bins[i+1] = phi_centers[i] + bin_width/2
                else:
                    phi_bins = np.array([-3.2, 3.2])  # fallback
                
                results_dict[f'WP {wp:.3f}'] = {
                    'phi_bins': phi_bins,
                    'phi_centers': phi_centers,
                    'efficiency': efficiencies,
                    'efficiency_err': eff_errors,
                    'overall_purity': overall_purity,
                    'counts': None
                }
            
            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue
            
            # Create the main combined plot for this technology
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Color schemes for different working points
            colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
            
            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]
                
                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors_eta_phi(ax, wp_data['phi_bins'], wp_data['efficiency'], 
                                            wp_data['efficiency_err'], wp_data['counts'],
                                            color, f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            # Format efficiency plot
            ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} Technology', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"efficiency_vs_phi_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{tech_name} efficiency vs phi plot saved to: {output_path}")
            
            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_phi_technology(results_dict, tech_name)
    
    def _plot_metric_with_errors_eta_phi(self, ax, bins, values, errors, counts, color, label, metric_type):
        """Helper function to plot metrics with error bands and step plots for eta/phi"""
        for i in range(len(bins) - 1):
            lhs, rhs = bins[i], bins[i + 1]
            value = values[i]
            error = errors[i] if errors is not None else 0
            
            # Create error band
            if error > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                if metric_type == 'track_length':
                    # Don't cap values for track length plots
                    y_upper = value + error
                    y_lower = max(value - error, 0.0)  # Only floor at 0.0
                else:
                    # Cap efficiency values between 0 and 1
                    y_upper = min(value + error, 1.0)  # Cap at 1.0
                    y_lower = max(value - error, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, 
                               color=color, alpha=0.3, 
                               label=f"binomial err - {label}" if i == 0 else "")
            
            # Step plot
            ax.step([lhs, rhs], [value, value], 
                   color=color, linewidth=2.5,
                   label=label if i == 0 else "")
    
    def _plot_individual_working_points_eta(self, results_dict):
        """Create separate efficiency plots for each working point (eta)"""
        # Create subdirectory for organizing plots
        efficiency_dir = self.output_dir / "efficiency_plots"
        efficiency_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            self._plot_metric_with_errors_eta_phi(ax, wp_data['eta_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'coral', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $\\eta$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {wp_name}', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])
            
            plt.tight_layout()
            
            output_path = efficiency_dir / f"efficiency_vs_eta_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual eta working point plots saved to: {efficiency_dir}")
    
    def _plot_individual_working_points_eta_technology(self, results_dict, tech_name):
        """Create individual eta plots for each working point and technology"""
        # Create efficiency_plots_<technology> subdirectory (unified for all coordinates)
        efficiency_plots_dir = self.output_dir / f"efficiency_plots_{tech_name}"
        efficiency_plots_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            self._plot_metric_with_errors_eta_phi(ax, wp_data['eta_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'royalblue', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $\\eta$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} Technology - {wp_name}', loc='left', fontsize=12)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])
            
            plt.tight_layout()
            
            output_path = efficiency_plots_dir / f"efficiency_vs_eta_{tech_name}_{wp_name.replace(' ', '_').replace('.', '_')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual {tech_name} eta efficiency plots saved to: {efficiency_plots_dir}")
    
    def _plot_individual_working_points_phi(self, results_dict):
        """Create separate efficiency plots for each working point (phi)"""
        # Create subdirectory for organizing plots
        efficiency_dir = self.output_dir / "efficiency_plots"
        efficiency_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            self._plot_metric_with_errors_eta_phi(ax, wp_data['phi_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'coral', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {wp_name}', loc='left', fontsize=14)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])
            
            plt.tight_layout()
            
            output_path = efficiency_dir / f"efficiency_vs_phi_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual phi working point plots saved to: {efficiency_dir}")
    
    def _plot_individual_working_points_phi_technology(self, results_dict, tech_name):
        """Create individual phi plots for each working point and technology"""
        # Create efficiency_plots_<technology> subdirectory (unified for all coordinates)
        efficiency_plots_dir = self.output_dir / f"efficiency_plots_{tech_name}"
        efficiency_plots_dir.mkdir(exist_ok=True)
        
        for wp_name, wp_data in results_dict.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            self._plot_metric_with_errors_eta_phi(ax, wp_data['phi_bins'], wp_data['efficiency'], 
                                        wp_data['efficiency_err'], wp_data['counts'],
                                        'royalblue', f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})", 'efficiency')
            
            ax.set_xlabel('Truth Muon $\\phi$', fontsize=14)
            ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
            ax.set_title(f'ATLAS Muon Hit Filter - {tech_name} Technology - {wp_name}', loc='left', fontsize=12)
            ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])
            
            plt.tight_layout()
            
            output_path = efficiency_plots_dir / f"efficiency_vs_phi_{tech_name}_{wp_name.replace(' ', '_').replace('.', '_')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual {tech_name} phi efficiency plots saved to: {efficiency_plots_dir}")
    
    def run_full_evaluation(self, skip_individual_plots=True, skip_technology_plots=True, skip_eta_phi_plots=True):
        """
        Run complete evaluation pipeline with configurable plotting options.
        
        Parameters:
        -----------
        skip_individual_plots : bool, default True
            Skip individual working point plots to save time/space
        skip_technology_plots : bool, default True  
            Skip technology-specific plots to save time/space
        skip_eta_phi_plots : bool, default True
            Skip eta and phi binned plots to save time/space
        """
        print("Starting full evaluation of ATLAS muon hit filter (optimized)...")
        
        # Monitor memory throughout
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        print(f"Starting memory usage: {start_memory:.1f} MB")
        
        # Collect data
        if not self.collect_data():
            print("Data collection failed, aborting evaluation")
            return
        
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after data collection: {current_memory:.1f} MB (+{current_memory-start_memory:.1f} MB)")
        
        # Generate core plots (always generated)
        print("\n=== Generating core evaluation plots ===")
        
        # ROC curve
        roc_auc = self.plot_roc_curve()
        gc.collect()
        
        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots)
        gc.collect()
        
        # Working point performance
        self.plot_working_point_performance()
        gc.collect()
        
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after core plots: {current_memory:.1f} MB")
        
        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots)
            gc.collect()
            
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots)  
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after eta/phi plots: {current_memory:.1f} MB")
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")
        
        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS)
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after technology plots: {current_memory:.1f} MB")
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")
        
        # Track lengths (lightweight, always include)
        print("\n=== Generating track length plots ===")
        self.plot_track_lengths()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nEvaluation complete! Final memory: {final_memory:.1f} MB")
        print(f"Peak memory usage: +{final_memory-start_memory:.1f} MB")
        print(f"Results saved to {self.output_dir}")
        print(f"Final AUC Score: {roc_auc:.4f}")
        
        # Write summary file
        self._write_evaluation_summary(roc_auc, skip_individual_plots, skip_technology_plots, skip_eta_phi_plots)
    
    def _write_evaluation_summary(self, roc_auc, skip_individual_plots, skip_technology_plots, skip_eta_phi_plots):
        """Write a summary of the evaluation run."""
        summary_path = self.output_dir / "evaluation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ATLAS MUON HIT FILTER EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events processed: {self.max_events}\n")
            f.write(f"Total hits analyzed: {len(self.all_logits):,}\n")
            f.write(f"True hits: {np.sum(self.all_true_labels):,}\n")
            f.write(f"AUC Score: {roc_auc:.4f}\n\n")
            
            f.write("PLOTS GENERATED:\n")
            f.write("- ROC curve\n")
            f.write("- Efficiency vs pT (main)\n")
            f.write("- Working point performance\n")
            f.write("- Track lengths\n")
            
            if not skip_eta_phi_plots:
                f.write("- Efficiency vs eta\n")
                f.write("- Efficiency vs phi\n")
            else:
                f.write("- Efficiency vs eta/phi (SKIPPED)\n")
                
            if not skip_technology_plots:
                f.write("- Technology-specific plots\n")
            else:
                f.write("- Technology-specific plots (SKIPPED)\n")
                
            if not skip_individual_plots:
                f.write("- Individual working point plots\n")
            else:
                f.write("- Individual working point plots (SKIPPED)\n")
            
            f.write(f"\nOutputs saved to: {self.output_dir}\n")
        
        print(f"Evaluation summary saved to: {summary_path}")
    
    def plot_efficiency_vs_eta(self, working_points=None, skip_individual_plots=True):
        """Plot efficiency vs eta - optimized version"""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs eta plots (optimized)...")
        
        # Only create the general plots, skip individual by default
        self._plot_efficiency_vs_eta_general(working_points, skip_individual_plots)
        
        if not skip_individual_plots:
            self._plot_efficiency_vs_eta_by_technology(working_points)
    
    def plot_efficiency_vs_phi(self, working_points=None, skip_individual_plots=True):
        """Plot efficiency vs phi - optimized version"""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs phi plots (optimized)...")
        
        # Only create the general plots, skip individual by default
        self._plot_efficiency_vs_phi_general(working_points, skip_individual_plots)
        
        if not skip_individual_plots:
            self._plot_efficiency_vs_phi_by_technology(working_points)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ATLAS muon hit filter using DataLoader (OPTIMIZED)')
    parser.add_argument('--eval_path', type=str, default="/scratch/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, default="/scratch/ml_test_data_156000_hdf5",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, default="./hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events', '-m', type=int, default=-1,
                       help='Maximum number of events to process (for testing)')
    
    # Performance and output control options
    parser.add_argument('--include-individual-plots', action='store_true',
                       help='Generate individual working point plots (slower, more files)')
    parser.add_argument('--include-tech', action='store_true', 
                       help='Generate technology-specific plots (slower)')
    parser.add_argument('--include-eta-phi', action='store_true',
                       help='Generate eta and phi binned plots (slower)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: skip all optional plots (equivalent to default)')
    
    args = parser.parse_args()
    
    # Handle fast mode
    if args.fast:
        args.include_individual_plots = False
        args.include_tech = False
        args.include_eta_phi = False
    
    print("=" * 80)
    print("ATLAS MUON HIT FILTER EVALUATION (OPTIMIZED)")
    print("=" * 80)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Include individual plots: {args.include_individual_plots}")
    print(f"Include technology plots: {args.include_tech}")
    print(f"Include eta/phi plots: {args.include_eta_phi}")
    print("=" * 80)
    
    # Enable stdout buffering for nohup
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    try:
        # Create evaluator and run
        evaluator = AtlasMuonEvaluatorDataLoader(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        evaluator.run_full_evaluation(
            skip_individual_plots=not args.include_individual_plots,
            skip_technology_plots=not args.include_tech,
            skip_eta_phi_plots=not args.include_eta_phi
        )
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Evaluation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
