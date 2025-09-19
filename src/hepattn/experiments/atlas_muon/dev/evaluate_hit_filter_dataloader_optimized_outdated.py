#!/usr/bin/env python3
"""
OPTIMIZED Evaluation script for ATLAS muon hit filtering model using DataLoader approach.

PERFORMANCE OPTIMIZATIONS:
- Memory-efficient data structures with proper dtypes
- Pre-allocated arrays and vectorized operations  
- Reduced redundant ROC calculations with caching
- Optimized plotting with selective output
- Proper signal handling for background execution
- Memory monitoring and garbage collection
- Configurable plot generation to save time/space
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
try:
    import psutil
except ImportError:
    # Fallback if psutil is not available
    class FakePsutil:
        def Process(self):
            return self
        def memory_info(self):
            class MemInfo:
                rss = 0
            return MemInfo()
    psutil = FakePsutil()

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


class AtlasMuonEvaluatorDataLoaderOptimized:
    """OPTIMIZED Evaluation class for ATLAS muon hit filtering using DataLoader."""
    
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
        
        # Storage for all collected data (will be populated efficiently)
        self.all_logits = None
        self.all_true_labels = None
        self.all_particle_pts = None
        self.all_particle_etas = None
        self.all_particle_phis = None
        self.all_particle_ids = None
        self.all_particle_technology = None
        self.all_event_ids = None
        
        # Cache for expensive computations
        self._cached_roc = None
    
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
        num_test_events = self.max_events if self.max_events is not None and self.max_events != -1 else -1
        
        # Initialize data module with reduced workers for memory efficiency
        self.data_module = AtlasMuonDataModule(
            train_dir=str(self.data_dir),
            val_dir=str(self.data_dir),
            test_dir=str(self.data_dir),
            num_workers=min(4, os.cpu_count() or 4),  # Limit workers to reduce memory pressure
            num_train=abs(num_test_events) if num_test_events != -1 else 1000,
            num_val=abs(num_test_events) if num_test_events != -1 else 1000,
            num_test=num_test_events,
            batch_size=1,
            inputs=inputs_eval,
            targets=targets_eval,
            pin_memory=False,  # Disable pin_memory to reduce memory usage
        )
        
        # Setup the data module
        self.data_module.setup(stage='test')
        self.test_dataloader = self.data_module.test_dataloader()
        
        print(f"DataLoader setup complete, processing {num_test_events if num_test_events > 0 else 'all'} events")
    
    def collect_data(self):
        """Collect all data for analysis using memory-optimized approach."""
        print("Collecting data from all events using DataLoader (MEMORY OPTIMIZED)...")
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Check evaluation file
        with h5py.File(self.eval_path, 'r') as eval_file:
            eval_keys = list(eval_file.keys())
            print(f"Evaluation file contains {len(eval_keys)} events")
        
        # Pre-allocate storage with estimated sizes
        estimated_hits = min(self.max_events * 1000 if self.max_events and self.max_events > 0 else 10000000, 50000000)
        
        try:
            # Pre-allocate arrays with memory-efficient dtypes
            all_logits = np.zeros(estimated_hits, dtype=np.float32)
            all_true_labels = np.zeros(estimated_hits, dtype=np.bool_)
            all_particle_pts = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_etas = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_phis = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_ids = np.zeros(estimated_hits, dtype=np.int32)
            all_particle_technology = np.zeros(estimated_hits, dtype=np.int8)
            all_event_ids = np.zeros(estimated_hits, dtype=np.int32)
            
            current_idx = 0
            events_processed = 0
            events_attempted = 0
            use_prealloc = True
            
            print(f"Pre-allocated arrays for {estimated_hits:,} hits")
            
        except MemoryError:
            print("WARNING: Not enough memory for pre-allocation. Using list-based approach.")
            all_logits = []
            all_true_labels = []
            all_particle_pts = []
            all_particle_etas = []
            all_particle_phis = []
            all_particle_ids = []
            all_particle_technology = []
            all_event_ids = []
            current_idx = None
            use_prealloc = False
        
        try:
            with h5py.File(self.eval_path, 'r') as eval_file:
                for batch_idx, batch_data in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                    events_attempted += 1
                    
                    # Break if max_events reached
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
                        if len(batch_data) != 2:
                            continue
                        
                        inputs_batch, targets_batch = batch_data
                        
                        # Extract event index
                        if "sample_id" not in targets_batch:
                            continue
                        
                        event_idx = targets_batch["sample_id"][0].item()
                        
                        # Check if event exists in evaluation file
                        if str(event_idx) not in eval_file:
                            continue
                        
                        # Get data from evaluation file and dataloader
                        hit_logits = eval_file[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)
                        
                        if "hit_on_valid_particle" not in targets_batch:
                            continue
                        
                        true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(np.bool_)
                        hit_particle_ids = inputs_batch["plotting_spacePoint_truthLink"][0].numpy().astype(np.int32)
                        hit_technologies = inputs_batch["hit_spacePoint_technology"][0].numpy().astype(np.int8)
                        
                        # Verify shapes match
                        n_hits = len(hit_logits)
                        if n_hits != len(true_labels) or n_hits != len(hit_particle_ids):
                            continue
                        
                        # Get particle properties
                        required_fields = ["particle_truthMuon_pt", "particle_truthMuon_eta", "particle_truthMuon_phi"]
                        if not all(field in targets_batch for field in required_fields):
                            continue
                        
                        particle_pts = targets_batch["particle_truthMuon_pt"][0].numpy().astype(np.float32)
                        particle_etas = targets_batch["particle_truthMuon_eta"][0].numpy().astype(np.float32)
                        particle_phis = targets_batch["particle_truthMuon_phi"][0].numpy().astype(np.float32)

                        # Map hits to particle properties (vectorized)
                        hit_pts = np.full(n_hits, -1.0, dtype=np.float32)
                        hit_etas = np.full(n_hits, -1.0, dtype=np.float32)
                        hit_phis = np.full(n_hits, -1.0, dtype=np.float32)
                        
                        # Efficient vectorized mapping
                        unique_particle_ids = np.unique(hit_particle_ids)
                        valid_particle_ids = unique_particle_ids[unique_particle_ids >= 0]
                        
                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_pts):
                                hit_mask = hit_particle_ids == particle_id
                                hit_pts[hit_mask] = particle_pts[idx]
                                hit_etas[hit_mask] = particle_etas[idx]
                                hit_phis[hit_mask] = particle_phis[idx]

                        # Store data efficiently
                        if use_prealloc and current_idx is not None:
                            # Resize arrays if needed
                            if current_idx + n_hits >= len(all_logits):
                                new_size = max(len(all_logits) * 2, current_idx + n_hits + 100000)
                                print(f"Resizing arrays to {new_size:,} elements")
                                all_logits = np.resize(all_logits, new_size)
                                all_true_labels = np.resize(all_true_labels, new_size)
                                all_particle_pts = np.resize(all_particle_pts, new_size)
                                all_particle_etas = np.resize(all_particle_etas, new_size)
                                all_particle_phis = np.resize(all_particle_phis, new_size)
                                all_particle_ids = np.resize(all_particle_ids, new_size)
                                all_particle_technology = np.resize(all_particle_technology, new_size)
                                all_event_ids = np.resize(all_event_ids, new_size)
                            
                            # Copy data to pre-allocated arrays
                            all_logits[current_idx:current_idx+n_hits] = hit_logits
                            all_true_labels[current_idx:current_idx+n_hits] = true_labels
                            all_particle_pts[current_idx:current_idx+n_hits] = hit_pts
                            all_particle_etas[current_idx:current_idx+n_hits] = hit_etas
                            all_particle_phis[current_idx:current_idx+n_hits] = hit_phis
                            all_particle_ids[current_idx:current_idx+n_hits] = hit_particle_ids
                            all_particle_technology[current_idx:current_idx+n_hits] = hit_technologies
                            all_event_ids[current_idx:current_idx+n_hits] = event_idx
                            current_idx += n_hits
                        else:
                            # Fall back to list append
                            all_logits.append(hit_logits)
                            all_true_labels.append(true_labels)
                            all_particle_pts.append(hit_pts)
                            all_particle_etas.append(hit_etas)
                            all_particle_phis.append(hit_phis)
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
        if use_prealloc and current_idx is not None:
            self.all_logits = all_logits[:current_idx]
            self.all_true_labels = all_true_labels[:current_idx]
            self.all_particle_pts = all_particle_pts[:current_idx]
            self.all_particle_etas = all_particle_etas[:current_idx]
            self.all_particle_phis = all_particle_phis[:current_idx]
            self.all_particle_ids = all_particle_ids[:current_idx]
            self.all_particle_technology = all_particle_technology[:current_idx]
            self.all_event_ids = all_event_ids[:current_idx]
        else:
            # Concatenate lists
            self.all_logits = np.concatenate(all_logits) if all_logits else np.array([])
            self.all_true_labels = np.concatenate(all_true_labels) if all_true_labels else np.array([], dtype=bool)
            self.all_particle_pts = np.concatenate(all_particle_pts) if all_particle_pts else np.array([])
            self.all_particle_etas = np.concatenate(all_particle_etas) if all_particle_etas else np.array([])
            self.all_particle_phis = np.concatenate(all_particle_phis) if all_particle_phis else np.array([])
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
    
    def plot_roc_curve(self):
        """Generate ROC curve with AUC score."""
        print("Generating ROC curve...")
        
        # Calculate and cache ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        self._cached_roc = (fpr, tpr, thresholds)  # Cache for reuse
        
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
    
    def plot_efficiency_vs_pt_optimized(self, working_points=None):
        """Optimized efficiency vs pT plotting with vectorized operations."""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        
        print("Generating efficiency vs pT plots (OPTIMIZED)...")
        
        # Use cached ROC curve
        if self._cached_roc is None:
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        # Define bins
        pt_min, pt_max = 5.0, 200.0
        pt_bins = np.linspace(pt_min, pt_max, 21)  # 20 bins
        
        # Pre-compute all working point data
        results = {}
        
        print("Computing efficiency for all working points (vectorized)...")
        for wp in tqdm(working_points, desc="Working points"):
            # Find threshold
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                continue
            
            threshold = thresholds[tpr >= wp][0]
            cut_predictions = self.all_logits >= threshold
            
            # Calculate overall purity
            total_true_positives = np.sum(self.all_true_labels & cut_predictions)
            total_predicted_positives = np.sum(cut_predictions)
            overall_purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
            
            # Vectorized efficiency calculation using scipy.stats.binned_statistic
            valid_mask = self.all_particle_pts > 0  # Only valid particles
            
            if not np.any(valid_mask):
                continue
            
            valid_pts = self.all_particle_pts[valid_mask]
            valid_true = self.all_true_labels[valid_mask].astype(float)
            valid_pred = cut_predictions[valid_mask].astype(float)
            
            # Calculate efficiency in bins
            true_positive_counts, _, _ = binned_statistic(valid_pts, valid_true * valid_pred, 
                                                        statistic='sum', bins=pt_bins)
            total_positive_counts, _, _ = binned_statistic(valid_pts, valid_true, 
                                                         statistic='sum', bins=pt_bins)
            
            # Calculate efficiency and errors
            efficiencies = np.divide(true_positive_counts, total_positive_counts, 
                                   out=np.zeros_like(true_positive_counts), 
                                   where=total_positive_counts!=0)
            
            eff_errors = np.sqrt(np.divide(efficiencies * (1 - efficiencies), total_positive_counts,
                                         out=np.zeros_like(efficiencies),
                                         where=total_positive_counts!=0))
            
            results[f'WP {wp:.3f}'] = {
                'efficiency': efficiencies,
                'efficiency_err': eff_errors,
                'overall_purity': overall_purity
            }
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        colors = ['coral', 'royalblue', 'forestgreen', 'purple', 'orange', 'brown']
        
        for i, (wp_name, wp_data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            # Step plot with error bands
            for j in range(len(pt_bins) - 1):
                lhs, rhs = pt_bins[j], pt_bins[j + 1]
                value = wp_data['efficiency'][j]
                error = wp_data['efficiency_err'][j]
                
                # Error band
                if error > 0:
                    point_range = np.linspace(lhs, rhs, 100)
                    y_upper = min(value + error, 1.0)
                    y_lower = max(value - error, 0.0)
                    ax.fill_between(point_range, y_upper, y_lower, 
                                   color=color, alpha=0.3)
                
                # Step plot
                ax.step([lhs, rhs], [value, value], 
                       color=color, linewidth=2.5,
                       label=f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})" if j == 0 else "")
        
        ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Hit Filter Efficiency', fontsize=14)
        ax.set_title('ATLAS Muon Hit Filter - All Technologies', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.legend()
        ax.set_ylim([0.85, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Efficiency vs pT plot saved to: {output_path}")
        gc.collect()
    
    def plot_working_point_performance_optimized(self, working_points=None):
        """Ultra-optimized working point performance analysis."""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        
        print("Generating working point performance plot (ULTRA-OPTIMIZED)...")
        
        # Use cached ROC curve
        if self._cached_roc is None:
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)
        
        fpr, tpr, thresholds = self._cached_roc
        
        avg_purities = []
        avg_purity_errors = []
        
        # Pre-calculate all thresholds and predictions (vectorized)
        print("Pre-calculating thresholds and predictions...")
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                avg_purities.append(0.0)
                avg_purity_errors.append(0.0)
                continue
            
            threshold = thresholds[tpr >= wp][0]
            predictions = self.all_logits >= threshold
            
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
        plt.ylim(0.4, 1.05)
        
        # Save plot
        output_path = self.output_dir / "working_point_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Working point performance plot saved to {output_path}")
        
        # Print performance summary
        print("\nWorking Point Performance Summary:")
        for wp, purity, error in zip(working_points, avg_purities, avg_purity_errors):
            print(f"  WP {wp:.3f}: Purity = {purity:.4f} ± {error:.4f}")
        
        gc.collect()
    
    def plot_track_lengths_lightweight(self):
        """Lightweight track length analysis."""
        print("Generating track length plots (lightweight)...")
        
        # Calculate track lengths for valid tracks only
        valid_mask = (self.all_true_labels) & (self.all_particle_ids >= 0)
        
        if not np.any(valid_mask):
            print("No valid tracks found for track length analysis")
            return
        
        # Use pandas for fast groupby operations
        df = pd.DataFrame({
            'event_id': self.all_event_ids[valid_mask],
            'particle_id': self.all_particle_ids[valid_mask],
            'pt': self.all_particle_pts[valid_mask]
        })
        
        # Group by track and count hits
        track_stats = df.groupby(['event_id', 'particle_id']).agg({
            'pt': 'first'  # Get PT for each track
        }).reset_index()
        track_stats['track_length'] = df.groupby(['event_id', 'particle_id']).size().values
        
        print(f"Found {len(track_stats)} unique tracks")
        print(f"Track length statistics: min={track_stats['track_length'].min()}, max={track_stats['track_length'].max()}, mean={track_stats['track_length'].mean():.1f}")
        
        # Plot track length vs pT
        pt_bins = np.linspace(0, 200, 21)
        
        avg_lengths = []
        std_lengths = []
        
        for i in range(len(pt_bins) - 1):
            mask = (track_stats['pt'] >= pt_bins[i]) & (track_stats['pt'] < pt_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_stats.loc[mask, 'track_length']
                avg_lengths.append(lengths_in_bin.mean())
                std_lengths.append(lengths_in_bin.std() / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Step plot with error bars
        for i in range(len(pt_bins) - 1):
            lhs, rhs = pt_bins[i], pt_bins[i + 1]
            value = avg_lengths[i]
            error = std_lengths[i]
            
            # Error band
            if error > 0:
                point_range = np.linspace(lhs, rhs, 100)
                y_upper = value + error
                y_lower = max(value - error, 0.0)
                ax.fill_between(point_range, y_upper, y_lower, 
                               color='royalblue', alpha=0.3)
            
            # Step plot
            ax.step([lhs, rhs], [value, value], 
                   color='royalblue', linewidth=2.5,
                   label='Average Track Length' if i == 0 else "")
        
        ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Average Track Length [hits]', fontsize=14)
        ax.set_title('ATLAS Muon Track Length vs $p_T$', loc='left', fontsize=14)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        ax.legend()
        ax.set_xlim([0, 200])
        
        plt.tight_layout()
        
        output_path = self.output_dir / "track_length_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Track length vs pT plot saved to: {output_path}")
        gc.collect()
    
    def run_evaluation_optimized(self, include_track_lengths=True):
        """Run optimized evaluation pipeline with only essential plots."""
        print("Starting OPTIMIZED evaluation of ATLAS muon hit filter...")
        print("Only generating essential plots for maximum speed and minimum memory usage.")
        
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
        
        # Generate essential plots only
        print("\n=== Generating essential evaluation plots ===")
        
        # ROC curve (essential)
        roc_auc = self.plot_roc_curve()
        gc.collect()
        
        # Efficiency vs pT (essential)
        self.plot_efficiency_vs_pt_optimized()
        gc.collect()
        
        # Working point performance (essential)
        self.plot_working_point_performance_optimized()
        gc.collect()
        
        # Track lengths (optional but lightweight)
        if include_track_lengths:
            self.plot_track_lengths_lightweight()
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nOptimized evaluation complete! Final memory: {final_memory:.1f} MB")
        print(f"Peak memory usage: +{final_memory-start_memory:.1f} MB")
        print(f"Results saved to {self.output_dir}")
        print(f"Final AUC Score: {roc_auc:.4f}")
        
        # Write summary file
        self._write_evaluation_summary(roc_auc)
    
    def _write_evaluation_summary(self, roc_auc):
        """Write a summary of the evaluation run."""
        summary_path = self.output_dir / "evaluation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ATLAS MUON HIT FILTER EVALUATION SUMMARY (OPTIMIZED)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events processed: {self.max_events}\n")
            f.write(f"Total hits analyzed: {len(self.all_logits):,}\n")
            f.write(f"True hits: {np.sum(self.all_true_labels):,}\n")
            f.write(f"AUC Score: {roc_auc:.4f}\n\n")
            
            f.write("PLOTS GENERATED (OPTIMIZED SET):\n")
            f.write("- ROC curve\n")
            f.write("- Efficiency vs pT (main)\n")
            f.write("- Working point performance\n")
            f.write("- Track lengths vs pT\n\n")
            
            f.write("OPTIMIZATION FEATURES:\n")
            f.write("- Memory-efficient data structures\n")
            f.write("- Pre-allocated arrays with proper dtypes\n")
            f.write("- Cached ROC calculations\n")
            f.write("- Vectorized binning operations\n")
            f.write("- Reduced plot generation\n")
            f.write("- Garbage collection management\n\n")
            
            f.write(f"Outputs saved to: {self.output_dir}\n")
        
        print(f"Evaluation summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='OPTIMIZED ATLAS muon hit filter evaluation')
    parser.add_argument('--eval_path', type=str, 
                       default="/scratch/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/scratch/ml_test_data_156000_hdf5",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, 
                       default="./hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_optimized',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events', '-m', type=int, default=-1,
                       help='Maximum number of events to process (for testing)')
    parser.add_argument('--skip-track-lengths', action='store_true',
                       help='Skip track length plots for maximum speed')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ATLAS MUON HIT FILTER EVALUATION (ULTRA-OPTIMIZED)")
    print("=" * 80)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Include track lengths: {not args.skip_track_lengths}")
    print("=" * 80)
    print("OPTIMIZATIONS: Memory-efficient data structures, cached computations,")
    print("               vectorized operations, selective plotting")
    print("=" * 80)
    
    # Enable stdout buffering for nohup
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        # Python < 3.7 compatibility
        pass
    
    try:
        # Create evaluator and run
        evaluator = AtlasMuonEvaluatorDataLoaderOptimized(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        evaluator.run_evaluation_optimized(
            include_track_lengths=not args.skip_track_lengths
        )
        
        print("\n" + "=" * 80)
        print("OPTIMIZED EVALUATION COMPLETED SUCCESSFULLY")
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
