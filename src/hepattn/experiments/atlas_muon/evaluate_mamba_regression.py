#!/usr/bin/env python3
"""
Evaluation script for Mamba Track Regression Model

This script evaluates the performance of the Mamba-based track parameter regression by:
1. Creating residual distribution plots for eta, phi, pt
2. Creating distribution comparison plots (predictions vs truth)
3. Creating pT resolution plot: |pT_pred - pT_truth| / pT_truth
4. Creating precision (STD) plots binned over eta, phi, pt for each target variable
5. Creating charge classification accuracy plots vs eta, phi, pt
6. Computing charge classification AUC
7. Analyzing performance across three categories: all tracks, baseline tracks, rejected tracks

All plots follow ATLAS publication style conventions (without ATLAS label).

Baseline filtering criteria:
- >= 9 hits per track
- |eta| in [0.1, 2.7]
- pt >= 5.0 GeV
- >= 3 unique stations
- >= 3 stations with >= 3 hits each

Usage:
    python evaluate_mamba_regression.py \\
        --pred_path /path/to/predictions.h5 \\
        --data_dir /path/to/test_data \\
        --output_dir ./evaluation_output \\
        --max_tracks 10000
"""

import os 
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
import h5py

# Import atlasify for ATLAS style
try:
    import atlasify
    ATLASIFY_AVAILABLE = True
except ImportError:
    ATLASIFY_AVAILABLE = False
    print("Warning: atlasify not available, using default matplotlib style")

# Import sklearn for AUC calculation
from sklearn.metrics import roc_auc_score, roc_curve

# Import PerTrackAtlasMuonDataset for loading track data
from hepattn.experiments.atlas_muon.data_per_track import PerTrackAtlasMuonDataset

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3
})


def angular_difference(phi1, phi2):
    """Compute angular difference handling periodicity at ±π."""
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


class MambaRegressionEvaluator:
    """Evaluator for Mamba track regression model outputs."""
    
    def __init__(self, pred_path, data_dir, output_dir, max_tracks=None):
        self.pred_path = Path(pred_path)
        self.data_dir = Path(data_dir)
        self.max_tracks = max_tracks
        self.station_filtering_available = True  # Will be set to False if loading fails
        
        # Create timestamped output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"mamba_regression_eval_{timestamp}"
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Category subdirectories
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks"
        self.rejected_dir = self.output_dir / "rejected_tracks"
        
        # Plot type subdirectories within each category
        for cat_dir in [self.all_tracks_dir, self.baseline_dir, self.rejected_dir]:
            (cat_dir / "distributions").mkdir(parents=True, exist_ok=True)
            (cat_dir / "residuals").mkdir(parents=True, exist_ok=True)
            (cat_dir / "precision").mkdir(parents=True, exist_ok=True)
            (cat_dir / "resolution").mkdir(parents=True, exist_ok=True)
            (cat_dir / "charge_classification").mkdir(parents=True, exist_ok=True)
        
        print(f"Mamba Regression Evaluator initialized")
        print(f"Prediction file: {pred_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max tracks: {max_tracks if max_tracks else 'all'}")
    
    def load_predictions_and_data(self):
        """Load predictions from HDF5 file and station indices from test data.
        
        Expected prediction structure:
        - sample_id/preds/final/mamba_regression/{eta, phi, pt, charge, charge_prob}
        - sample_id/targets/{eta, phi, pt, charge}
        
        Also loads station indices from the original test data for baseline filtering.
        """
        print("Loading predictions from HDF5 file...")
        
        with h5py.File(self.pred_path, 'r') as f:
            all_eta_pred = []
            all_phi_pred = []
            all_pt_pred = []
            all_charge_pred = []
            all_charge_prob = []
            
            all_eta_truth = []
            all_phi_truth = []
            all_pt_truth = []
            all_charge_truth = []
            
            all_sample_ids = []
            
            # Get all event keys (numeric indices)
            event_keys = sorted([k for k in f.keys() if k.isdigit()], key=int)
            
            if self.max_tracks:
                event_keys = event_keys[:self.max_tracks]
            
            print(f"Found {len(event_keys)} events in prediction file")
            
            # Determine the structure by examining first event
            first_event = f[event_keys[0]]
            
            # Check for layer/task structure vs flat structure
            preds_group = first_event.get('preds', None)
            has_layer_structure = False
            preds_path = None
            
            if preds_group is not None and len(preds_group.keys()) > 0:
                first_layer = list(preds_group.keys())[0]
                if isinstance(preds_group[first_layer], h5py.Group):
                    # Check if it's layer/task structure
                    first_task = list(preds_group[first_layer].keys())[0] if len(preds_group[first_layer].keys()) > 0 else None
                    if first_task and isinstance(preds_group[first_layer][first_task], h5py.Group):
                        has_layer_structure = True
                        preds_path = f'preds/{first_layer}/{first_task}'
                        print(f"Detected layer/task structure: {preds_path}")
            
            for event_key in tqdm(event_keys, desc="Loading events"):
                event_group = f[event_key]
                
                # Load predictions
                if has_layer_structure and preds_path:
                    try:
                        preds = event_group[preds_path]
                        if 'eta' in preds:
                            all_eta_pred.append(float(preds['eta'][...].flatten()[0]))
                        if 'phi' in preds:
                            all_phi_pred.append(float(preds['phi'][...].flatten()[0]))
                        if 'pt' in preds:
                            all_pt_pred.append(float(preds['pt'][...].flatten()[0]))
                        if 'charge' in preds:
                            all_charge_pred.append(float(preds['charge'][...].flatten()[0]))
                        if 'charge_prob' in preds:
                            all_charge_prob.append(float(preds['charge_prob'][...].flatten()[0]))
                        all_sample_ids.append(int(event_key))
                    except KeyError:
                        continue
                else:
                    # Flat structure fallback
                    if 'preds' in event_group:
                        preds = event_group['preds']
                        if 'eta' in preds:
                            all_eta_pred.append(float(preds['eta'][...].flatten()[0]))
                        if 'phi' in preds:
                            all_phi_pred.append(float(preds['phi'][...].flatten()[0]))
                        if 'pt' in preds:
                            all_pt_pred.append(float(preds['pt'][...].flatten()[0]))
                        if 'charge' in preds:
                            all_charge_pred.append(float(preds['charge'][...].flatten()[0]))
                        all_sample_ids.append(int(event_key))
                
                # Load targets
                if 'targets' in event_group:
                    targets = event_group['targets']
                    if 'eta' in targets:
                        all_eta_truth.append(float(targets['eta'][...].flatten()[0]))
                    if 'phi' in targets:
                        all_phi_truth.append(float(targets['phi'][...].flatten()[0]))
                    if 'pt' in targets:
                        all_pt_truth.append(float(targets['pt'][...].flatten()[0]))
                    if 'charge' in targets:
                        all_charge_truth.append(float(targets['charge'][...].flatten()[0]))
        
        # Convert to numpy arrays
        self.data = {
            'eta_pred': np.array(all_eta_pred),
            'phi_pred': np.array(all_phi_pred),
            'pt_pred': np.array(all_pt_pred),
            'charge_pred': np.array(all_charge_pred),
            'charge_prob': np.array(all_charge_prob) if all_charge_prob else np.array(all_charge_pred),
            'eta_truth': np.array(all_eta_truth),
            'phi_truth': np.array(all_phi_truth),
            'pt_truth': np.array(all_pt_truth),
            'charge_truth': np.array(all_charge_truth),
            'sample_ids': np.array(all_sample_ids),
        }
        
        print(f"Loaded {len(self.data['eta_pred'])} tracks with predictions")
        print(f"Loaded {len(self.data['eta_truth'])} tracks with targets")
        
        # Load station indices from test data for baseline filtering
        print("\nLoading station indices from test data for baseline filtering...")
        self._load_station_indices()
        
        return self.data
    
    def _load_station_indices(self):
        """Load station indices using PerTrackAtlasMuonDataset.
        
        Uses the PerTrackAtlasMuonDataset class which handles track index
        creation/caching automatically and provides access to hit data.
        """
        n_tracks = len(self.data['sample_ids'])
        
        # Default inputs/targets needed for dataset initialization
        inputs = {
            'hit': [
                'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
                'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
                'spacePoint_time', 'spacePoint_driftR',
                'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
                'spacePoint_channel', 'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
                'spacePoint_stationIndex', 'spacePoint_technology',
                'r', 's', 'theta', 'phi'
            ]
        }
        targets = {
            'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi', 'truthMuon_qpt']
        }
        
        # Create PerTrackAtlasMuonDataset - this will build/load track index automatically
        print("Creating PerTrackAtlasMuonDataset to load track data...")
        per_track_dataset = PerTrackAtlasMuonDataset(
            dirpath=str(self.data_dir),
            inputs=inputs,
            targets=targets,
            num_events=-1,  # Load all events
            min_hits_per_track=1,  # We want all tracks, filtering done later
            max_hits_per_track=600,
            event_max_num_particles=2,
        )
        
        print(f"PerTrackAtlasMuonDataset has {len(per_track_dataset)} tracks")
        print(f"Predictions file has {n_tracks} tracks")
        
        # Verify track counts match
        if len(per_track_dataset) != n_tracks:
            print(f"WARNING: Track count mismatch! Dataset has {len(per_track_dataset)}, predictions have {n_tracks}")
            print("Will use min of both counts")
        
        sample_ids = self.data['sample_ids']
        station_indices = []
        num_hits_per_track = []
        
        print("Loading station indices from PerTrackAtlasMuonDataset...")
        for sample_id in tqdm(sample_ids, desc="Loading station indices"):
            if sample_id < len(per_track_dataset):
                # Get track data from the dataset
                track_data = per_track_dataset[sample_id]
                
                # Get station indices from hit features
                # The hit_fields include 'spacePoint_stationIndex' at a specific position
                hit_features = track_data['hit_features'].numpy()  # (num_hits, num_features)
                num_hits = track_data['num_hits']
                
                # Find the station index position in hit_fields
                hit_fields = per_track_dataset.hit_fields
                if 'spacePoint_stationIndex' in hit_fields:
                    station_idx_pos = hit_fields.index('spacePoint_stationIndex')
                    track_stations = hit_features[:, station_idx_pos]
                    # Filter out padding (station index 0 or negative might be padding)
                    track_stations = track_stations[track_stations > 0]
                else:
                    track_stations = np.array([])
                
                station_indices.append(track_stations)
                num_hits_per_track.append(num_hits)
            else:
                station_indices.append(np.array([]))
                num_hits_per_track.append(0)
        
        self.data['station_indices'] = station_indices
        self.data['num_hits'] = np.array(num_hits_per_track)
        
        print(f"Loaded station indices for {len(station_indices)} tracks")

    def apply_baseline_filtering(self):
        """Apply baseline filtering and split data into categories.
        
        Baseline criteria:
        1. >= 9 hits per track
        2. |eta| in [0.1, 2.7]  (excludes very forward/central region)
        3. pt >= 5.0 GeV  (minimum pt cut)
        4. >= 3 unique stations
        5. >= 3 stations with >= 3 hits each
        """
        print("Applying baseline filtering...")
        
        n_tracks = len(self.data['eta_pred'])
        
        # Initialize filter statistics
        self.filter_stats = {
            'total_tracks': n_tracks,
            'failed_min_hits': 0,
            'failed_eta_cuts': 0,
            'failed_pt_cuts': 0,
            'failed_station_cuts': 0,
            'passed_all_cuts': 0,
        }
        
        # Create baseline mask - start with all True
        baseline_mask = np.ones(n_tracks, dtype=bool)
        
        eta_truth = self.data['eta_truth']
        pt_truth = self.data['pt_truth']
        num_hits = self.data['num_hits']
        station_indices = self.data['station_indices']
        
        # Apply cuts track by track (needed for station filtering)
        for i in range(n_tracks):
            # 1. Check minimum hits
            if num_hits[i] < 9:
                self.filter_stats['failed_min_hits'] += 1
                baseline_mask[i] = False
                continue
            
            # 2. Check eta cuts (|eta| in [0.1, 2.7])
            if np.abs(eta_truth[i]) < 0.1 or np.abs(eta_truth[i]) > 2.7:
                self.filter_stats['failed_eta_cuts'] += 1
                baseline_mask[i] = False
                continue
            
            # 3. Check pt cuts (pt >= 5.0 GeV)
            if pt_truth[i] < 5.0:
                self.filter_stats['failed_pt_cuts'] += 1
                baseline_mask[i] = False
                continue
            
            # 4 & 5. Check station requirements
            track_stations = station_indices[i]
            if len(track_stations) == 0:
                self.filter_stats['failed_station_cuts'] += 1
                baseline_mask[i] = False
                continue
            
            unique_stations = np.unique(track_stations)
            
            # Need at least 3 unique stations
            if len(unique_stations) < 3:
                self.filter_stats['failed_station_cuts'] += 1
                baseline_mask[i] = False
                continue
            
            # Need at least 3 stations with >= 3 hits each
            station_counts = {}
            for station in track_stations:
                station_counts[station] = station_counts.get(station, 0) + 1
            
            n_good_stations = sum(1 for count in station_counts.values() if count >= 3)
            if n_good_stations < 3:
                self.filter_stats['failed_station_cuts'] += 1
                baseline_mask[i] = False
                continue
            
            # Track passed all cuts
            self.filter_stats['passed_all_cuts'] += 1
        
        # Split data into categories
        array_keys = ['eta_pred', 'phi_pred', 'pt_pred', 'charge_pred', 'charge_prob',
                      'eta_truth', 'phi_truth', 'pt_truth', 'charge_truth', 'sample_ids', 'num_hits']
        
        self.all_data = {k: self.data[k] for k in array_keys if k in self.data}
        self.baseline_data = {k: self.data[k][baseline_mask] for k in array_keys if k in self.data}
        self.rejected_data = {k: self.data[k][~baseline_mask] for k in array_keys if k in self.data}
        
        print(f"\nBaseline Filtering Statistics:")
        print(f"  Total tracks: {self.filter_stats['total_tracks']:,}")
        print(f"  Failed min hits (>=9): {self.filter_stats['failed_min_hits']:,}")
        print(f"  Failed eta cuts (|eta| in [0.1, 2.7]): {self.filter_stats['failed_eta_cuts']:,}")
        print(f"  Failed pt cuts (pt >= 5.0 GeV): {self.filter_stats['failed_pt_cuts']:,}")
        print(f"  Failed station cuts (>=3 stations with >=3 hits): {self.filter_stats['failed_station_cuts']:,}")
        print(f"  Passed all cuts (baseline): {self.filter_stats['passed_all_cuts']:,}")
        print(f"  Rejected: {n_tracks - self.filter_stats['passed_all_cuts']:,}")
        
        return self.all_data, self.baseline_data, self.rejected_data
    
    def save_plot(self, fig, output_dir, filename):
        """Save plot in both PNG and PDF formats."""
        png_path = output_dir / f"{filename}.png"
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        return png_path, pdf_path
    
    def apply_atlas_style(self, ax, subtext=None):
        """Apply ATLAS-style formatting without ATLAS label."""
        if ATLASIFY_AVAILABLE:
            atlasify.atlasify(
                atlas=False,  # No ATLAS label
                subtext=subtext if subtext else "",
                font_size=14,
                sub_font_size=11,
                label_font_size=14
            )
    
    def plot_distribution_comparison(self, data, param, output_dir, category_name):
        """Plot overlaid distribution comparing predictions with truth."""
        pred_key = f'{param}_pred'
        truth_key = f'{param}_truth'
        
        if len(data[pred_key]) == 0:
            print(f"Warning: No data for {param} distribution in {category_name}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        predictions = data[pred_key]
        truth = data[truth_key]
        
        # Set bins based on parameter
        if param == 'eta':
            bins = np.linspace(-3, 3, 60)
        elif param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 60)
        elif param == 'pt':
            bins = np.linspace(0, 200, 60)
        elif param == 'charge':
            bins = np.linspace(-1.5, 1.5, 7)
        
        # Plot histograms (not normalized - show actual counts)
        ax.hist(truth, bins=bins, alpha=0.6,
                label='Ground Truth', color='royalblue', histtype='stepfilled')
        ax.hist(predictions, bins=bins, alpha=0.6,
                label='Model Predictions', color='red', histtype='stepfilled')
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]', 'charge': 'Charge'}
        ax.set_xlabel(param_labels.get(param, param), fontsize=14)
        ax.set_ylabel('Tracks / Bin', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        self.apply_atlas_style(ax, subtext=f"{category_name}\nN = {len(predictions):,}")
        
        self.save_plot(fig, output_dir / "distributions", f"{param}_distribution_comparison")
        print(f"  Saved {param} distribution comparison")
    
    def plot_residuals(self, data, param, output_dir, category_name):
        """Plot residual distribution."""
        pred_key = f'{param}_pred'
        truth_key = f'{param}_truth'
        
        if len(data[pred_key]) == 0:
            print(f"Warning: No data for {param} residuals in {category_name}")
            return
        
        predictions = data[pred_key]
        truth = data[truth_key]
        
        # Calculate residuals (handle phi periodicity)
        if param == 'phi':
            residuals = angular_difference(predictions, truth)
        else:
            residuals = predictions - truth
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        # Set bins
        p1, p99 = np.percentile(residuals, [1, 99])
        bins = np.linspace(p1, p99, 60)
        
        # Plot histogram
        ax.hist(residuals, bins=bins, alpha=0.7, color='royalblue',
                histtype='stepfilled', edgecolor='black', linewidth=0.5)
        
        # Add vertical lines (no labels in legend - stats go to subtext)
        ax.axvline(mean_res, color='red', linestyle='--', linewidth=2)
        ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$ [rad]', 'pt': r'$p_T$ [GeV]', 'charge': 'Charge'}
        ax.set_xlabel(f'{param_labels.get(param, param)} Residual (Pred - Truth)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Include stats in the ATLAS style subtext
        stats_subtext = (f"{category_name}\n"
                        f"Mean: {mean_res:.4f}, STD: {std_res:.4f}\n"
                        f"N = {len(residuals):,}")
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, output_dir / "residuals", f"{param}_residuals")
        print(f"  Saved {param} residuals plot")
        
        return {'mean': mean_res, 'std': std_res, 'n': len(residuals)}
    
    def plot_pt_resolution_vs_variable(self, data, bin_param, output_dir, category_name):
        """Plot pT resolution: |pT_pred - pT_truth| / pT_truth binned over a variable."""
        if len(data['pt_pred']) == 0:
            print(f"Warning: No data for pT resolution vs {bin_param} in {category_name}")
            return
        
        pt_pred = data['pt_pred']
        pt_truth = data['pt_truth']
        bin_values = data[f'{bin_param}_truth']
        
        # Filter to pT range [5, 200] GeV
        mask = (pt_truth >= 5) & (pt_truth <= 200)
        pt_pred_filtered = pt_pred[mask]
        pt_truth_filtered = pt_truth[mask]
        bin_values_filtered = bin_values[mask]
        
        if len(pt_pred_filtered) == 0:
            print(f"Warning: No tracks in pT range [5, 200] GeV for {category_name}")
            return
        
        # Calculate resolution for each track
        resolution = np.abs(pt_pred_filtered - pt_truth_filtered) / pt_truth_filtered
        
        # Define bins (use 20 bins for better resolution)
        if bin_param == 'eta':
            bins = np.linspace(-2.7, 2.7, 21)
        elif bin_param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)
        elif bin_param == 'pt':
            bins = np.linspace(5, 200, 21)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate mean and std resolution in each bin
        mean_resolutions = []
        mean_errors = []
        n_per_bin = []
        
        for i in range(len(bins) - 1):
            bin_mask = (bin_values_filtered >= bins[i]) & (bin_values_filtered < bins[i+1])
            bin_resolution = resolution[bin_mask]
            
            if len(bin_resolution) > 2:
                mean_res = np.mean(bin_resolution)
                # Standard error of the mean
                std_err = np.std(bin_resolution) / np.sqrt(len(bin_resolution))
                mean_resolutions.append(mean_res)
                mean_errors.append(std_err)
                n_per_bin.append(len(bin_resolution))
            else:
                mean_resolutions.append(np.nan)
                mean_errors.append(np.nan)
                n_per_bin.append(0)
        
        mean_resolutions = np.array(mean_resolutions)
        mean_errors = np.array(mean_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with error bars and horizontal bars for each bin
        valid = ~np.isnan(mean_resolutions)
        ax.errorbar(bin_centers[valid], mean_resolutions[valid], yerr=mean_errors[valid],
                    fmt='o', color='royalblue', markersize=6, capsize=4,
                    label=r'$p_T$ Resolution', zorder=3)
        
        # Add horizontal bars for each bin (discrete bins, no connecting line)
        for i, (is_valid, res_val) in enumerate(zip(valid, mean_resolutions)):
            if is_valid:
                ax.hlines(res_val, bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Calculate overall statistics
        mean_res = np.mean(resolution)
        median_res = np.median(resolution)
        std_res = np.std(resolution)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]'}
        
        ax.set_xlabel(f'Truth {param_labels.get(bin_param, bin_param)}', fontsize=14)
        ax.set_ylabel(r'$p_T$ Resolution: $|p_T^{pred} - p_T^{truth}| / p_T^{truth}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Include stats in the ATLAS style subtext
        stats_subtext = (f"{category_name}\n" + r"$p_T \in [5, 200]$ GeV" + 
                        f"\nMean: {mean_res:.4f}, Median: {median_res:.4f}\nN = {len(resolution):,}")
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, output_dir / "resolution", f"pt_resolution_vs_{bin_param}")
        print(f"  Saved pT resolution vs {bin_param}")
        
        return {'mean': mean_res, 'median': median_res, 'std': std_res, 'n': len(resolution)}
    
    def plot_precision_vs_variable(self, data, target_param, bin_param, output_dir, category_name):
        """Plot precision (STD of residuals) binned over a variable."""
        pred_key = f'{target_param}_pred'
        truth_key = f'{target_param}_truth'
        bin_truth_key = f'{bin_param}_truth'
        
        if len(data[pred_key]) == 0:
            print(f"Warning: No data for {target_param} precision vs {bin_param} in {category_name}")
            return
        
        predictions = data[pred_key]
        truth = data[truth_key]
        bin_values = data[bin_truth_key]
        
        # Calculate residuals
        if target_param == 'phi':
            residuals = angular_difference(predictions, truth)
        else:
            residuals = predictions - truth
        
        # Define bins based on bin_param (use 20 bins for better resolution)
        if bin_param == 'eta':
            bins = np.linspace(-2.7, 2.7, 21)
        elif bin_param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)
        elif bin_param == 'pt':
            bins = np.linspace(5, 200, 21)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate STD in each bin
        stds = []
        std_errors = []
        n_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_values >= bins[i]) & (bin_values < bins[i+1])
            bin_residuals = residuals[mask]
            
            if len(bin_residuals) > 2:
                std = np.std(bin_residuals)
                # Error on STD estimate: std / sqrt(2*(n-1))
                std_err = std / np.sqrt(2 * (len(bin_residuals) - 1))
                stds.append(std)
                std_errors.append(std_err)
                n_per_bin.append(len(bin_residuals))
            else:
                stds.append(np.nan)
                std_errors.append(np.nan)
                n_per_bin.append(0)
        
        stds = np.array(stds)
        std_errors = np.array(std_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with error bars and horizontal bars for each bin
        valid = ~np.isnan(stds)
        ax.errorbar(bin_centers[valid], stds[valid], yerr=std_errors[valid],
                    fmt='o', color='royalblue', markersize=6, capsize=4,
                    label=f'{target_param} precision', zorder=3)
        
        # Add horizontal bars for each bin (discrete bins, no connecting line)
        for i, (is_valid, std_val) in enumerate(zip(valid, stds)):
            if is_valid:
                ax.hlines(std_val, bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]'}
        target_labels = {'eta': r'$\sigma(\eta_{pred} - \eta_{truth})$',
                         'phi': r'$\sigma(\phi_{pred} - \phi_{truth})$ [rad]',
                         'pt': r'$\sigma(p_T^{pred} - p_T^{truth})$ [GeV]'}
        
        ax.set_xlabel(f'Truth {param_labels.get(bin_param, bin_param)}', fontsize=14)
        ax.set_ylabel(target_labels.get(target_param, f'{target_param} STD'), fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Calculate unbinned mean and median from raw residuals (not from bin STDs)
        unbinned_std = np.std(residuals)
        unbinned_mean_abs_res = np.mean(np.abs(residuals))
        unbinned_median_abs_res = np.median(np.abs(residuals))
        
        # Include stats in the ATLAS style subtext (using unbinned statistics)
        stats_subtext = (f"{category_name}\n"
                        f"Unbinned STD: {unbinned_std:.4f}\n"
                        f"N = {len(residuals):,}")
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, output_dir / "precision", f"{target_param}_precision_vs_{bin_param}")
        print(f"  Saved {target_param} precision vs {bin_param}")
    
    def plot_charge_accuracy_vs_variable(self, data, bin_param, output_dir, category_name):
        """Plot charge classification accuracy binned over a variable."""
        if len(data['charge_pred']) == 0:
            print(f"Warning: No data for charge accuracy vs {bin_param} in {category_name}")
            return
        
        charge_pred = data['charge_pred']
        charge_truth = data['charge_truth']
        bin_values = data[f'{bin_param}_truth']
        
        # Convert charge predictions to binary: >0 -> 1, <=0 -> 0
        # This matches the BCE truth format (0/1)
        charge_pred_binary = (charge_pred > 0).astype(float)
        
        # Also handle if truth is in -1/+1 format (convert to 0/1)
        if np.min(charge_truth) < 0:
            charge_truth_binary = (charge_truth > 0).astype(float)
        else:
            charge_truth_binary = charge_truth
        
        # Calculate correct predictions
        correct = (charge_pred_binary == charge_truth_binary).astype(float)
        
        # Define bins (use 20 bins for better resolution)
        if bin_param == 'eta':
            bins = np.linspace(-2.7, 2.7, 21)
        elif bin_param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)
        elif bin_param == 'pt':
            bins = np.linspace(5, 200, 21)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate accuracy in each bin
        accuracies = []
        acc_errors = []
        n_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_values >= bins[i]) & (bin_values < bins[i+1])
            bin_correct = correct[mask]
            
            if len(bin_correct) > 0:
                acc = np.mean(bin_correct)
                # Binomial error: sqrt(p*(1-p)/n)
                acc_err = np.sqrt(acc * (1 - acc) / len(bin_correct)) if len(bin_correct) > 1 else 0
                accuracies.append(acc)
                acc_errors.append(acc_err)
                n_per_bin.append(len(bin_correct))
            else:
                accuracies.append(np.nan)
                acc_errors.append(np.nan)
                n_per_bin.append(0)
        
        accuracies = np.array(accuracies)
        acc_errors = np.array(acc_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with error bars and horizontal bars for each bin
        valid = ~np.isnan(accuracies)
        ax.errorbar(bin_centers[valid], accuracies[valid], yerr=acc_errors[valid],
                    fmt='o', color='royalblue', markersize=6, capsize=4,
                    zorder=3)
        
        # Add horizontal bars for each bin (discrete bins, no connecting line)
        for i, (is_valid, acc_val) in enumerate(zip(valid, accuracies)):
            if is_valid:
                ax.hlines(acc_val, bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Reference line at 0.5 (random)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]'}
        
        ax.set_xlabel(f'Truth {param_labels.get(bin_param, bin_param)}', fontsize=14)
        ax.set_ylabel('Charge Classification Accuracy', fontsize=14)
        ax.set_ylim([0.4, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Calculate unbinned accuracy only (not binned mean/median)
        unbinned_accuracy = np.mean(correct)
        
        stats_subtext = (f"{category_name}\n"
                        f"Accuracy: {unbinned_accuracy:.4f}\n"
                        f"N = {len(correct):,}")
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, output_dir / "charge_classification", f"charge_accuracy_vs_{bin_param}")
        print(f"  Saved charge accuracy vs {bin_param}")
    
    def calculate_charge_auc(self, data, category_name):
        """Calculate AUC for charge classification using probability scores."""
        if len(data.get('charge_prob', [])) == 0 and len(data.get('charge_pred', [])) == 0:
            return None
        
        # Use charge_prob if available (continuous probability), else use charge_pred
        charge_score = data.get('charge_prob', data['charge_pred'])
        charge_truth = data['charge_truth']
        
        # Convert truth to binary: positive charge (1) -> 1, negative charge (-1 or 0) -> 0
        # Note: charge_truth might be in 0/1 format (for BCE) or -1/+1 format
        if np.min(charge_truth) < 0:
            # -1/+1 format
            truth_binary = (charge_truth > 0).astype(int)
        else:
            # Already 0/1 format
            truth_binary = charge_truth.astype(int)
        
        # Use charge probability/score (higher = more likely positive)
        try:
            auc = roc_auc_score(truth_binary, charge_score)
            print(f"  {category_name} Charge AUC: {auc:.4f}")
            return auc
        except Exception as e:
            print(f"  Warning: Could not calculate AUC for {category_name}: {e}")
            return None
    
    def write_statistics(self, all_stats, baseline_stats, rejected_stats):
        """Write statistics summary to text file."""
        stats_path = self.output_dir / "evaluation_statistics.txt"
        
        with open(stats_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MAMBA TRACK REGRESSION EVALUATION STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction file: {self.pred_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Filtering statistics
            f.write("-" * 40 + "\n")
            f.write("BASELINE FILTERING STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total tracks: {self.filter_stats['total_tracks']:,}\n")
            f.write(f"Failed min hits (>=9 required): {self.filter_stats['failed_min_hits']:,}\n")
            f.write(f"Failed eta cuts (|eta| not in [0.1, 2.7]): {self.filter_stats['failed_eta_cuts']:,}\n")
            f.write(f"Failed pt cuts (pt < 5.0 GeV): {self.filter_stats['failed_pt_cuts']:,}\n")
            f.write(f"Failed station cuts (>=3 stations with >=3 hits): {self.filter_stats['failed_station_cuts']:,}\n")
            f.write(f"Passed all cuts (baseline): {self.filter_stats['passed_all_cuts']:,}\n")
            f.write(f"Rejected: {self.filter_stats['total_tracks'] - self.filter_stats['passed_all_cuts']:,}\n\n")
            
            # Baseline filtering criteria description
            f.write("-" * 40 + "\n")
            f.write("BASELINE FILTERING CRITERIA\n")
            f.write("-" * 40 + "\n")
            f.write("1. >= 9 hits per track\n")
            f.write("2. |eta| in [0.1, 2.7]\n")
            f.write("3. pt >= 5.0 GeV\n")
            f.write("4. >= 3 unique stations\n")
            f.write("5. >= 3 stations with >= 3 hits each\n\n")
            
            # Per-category statistics
            for cat_name, cat_stats in [("ALL TRACKS", all_stats),
                                         ("BASELINE TRACKS", baseline_stats),
                                         ("REJECTED TRACKS", rejected_stats)]:
                f.write("-" * 40 + "\n")
                f.write(f"{cat_name}\n")
                f.write("-" * 40 + "\n")
                
                if cat_stats is None:
                    f.write("No data available\n\n")
                    continue
                
                f.write(f"Number of tracks: {cat_stats.get('n_tracks', 'N/A'):,}\n\n")
                
                # Residual statistics
                for param in ['eta', 'phi', 'pt']:
                    if f'{param}_residuals' in cat_stats:
                        res = cat_stats[f'{param}_residuals']
                        f.write(f"{param.upper()} Residuals:\n")
                        f.write(f"  Mean: {res['mean']:.6f}\n")
                        f.write(f"  STD:  {res['std']:.6f}\n")
                        f.write(f"  N:    {res['n']:,}\n\n")
                
                # pT resolution
                if 'pt_resolution' in cat_stats:
                    res = cat_stats['pt_resolution']
                    f.write(f"pT Resolution (|pred-truth|/truth):\n")
                    f.write(f"  Mean:   {res['mean']:.6f}\n")
                    f.write(f"  Median: {res['median']:.6f}\n")
                    f.write(f"  STD:    {res['std']:.6f}\n")
                    f.write(f"  N:      {res['n']:,}\n\n")
                
                # Charge AUC
                if 'charge_auc' in cat_stats:
                    f.write(f"Charge Classification AUC: {cat_stats['charge_auc']:.4f}\n\n")
                
                # Charge accuracy
                if 'charge_accuracy' in cat_stats:
                    f.write(f"Charge Classification Accuracy: {cat_stats['charge_accuracy']:.4f}\n\n")
        
        print(f"Statistics written to {stats_path}")
    
    def evaluate_category(self, data, output_dir, category_name):
        """Run full evaluation for a single category."""
        print(f"\n{'='*50}")
        print(f"EVALUATING: {category_name}")
        print(f"{'='*50}")
        
        if len(data['eta_pred']) == 0:
            print(f"Warning: No data for {category_name}")
            return None
        
        stats = {'n_tracks': len(data['eta_pred'])}
        
        # 1. Distribution comparison plots
        print("\nGenerating distribution comparison plots...")
        for param in ['eta', 'phi', 'pt', 'charge']:
            self.plot_distribution_comparison(data, param, output_dir, category_name)
        
        # 2. Residual plots
        print("\nGenerating residual plots...")
        for param in ['eta', 'phi', 'pt']:
            res_stats = self.plot_residuals(data, param, output_dir, category_name)
            if res_stats:
                stats[f'{param}_residuals'] = res_stats
        
        # 3. pT resolution plots (binned over pt, eta, phi)
        print("\nGenerating pT resolution plots...")
        for bin_param in ['pt', 'eta', 'phi']:
            pt_res_stats = self.plot_pt_resolution_vs_variable(data, bin_param, output_dir, category_name)
            if pt_res_stats and bin_param == 'pt':
                stats['pt_resolution'] = pt_res_stats
        
        # 4. Precision plots (9 combinations)
        print("\nGenerating precision plots...")
        for target_param in ['eta', 'phi', 'pt']:
            for bin_param in ['eta', 'phi', 'pt']:
                self.plot_precision_vs_variable(data, target_param, bin_param, output_dir, category_name)
        
        # 5. Charge classification accuracy plots
        print("\nGenerating charge classification accuracy plots...")
        for bin_param in ['eta', 'phi', 'pt']:
            self.plot_charge_accuracy_vs_variable(data, bin_param, output_dir, category_name)
        
        # 6. Charge AUC
        print("\nCalculating charge classification AUC...")
        auc = self.calculate_charge_auc(data, category_name)
        if auc is not None:
            stats['charge_auc'] = auc
        
        # Overall charge accuracy
        if len(data['charge_pred']) > 0:
            # Convert predictions to 0/1 format
            charge_pred_binary = (data['charge_pred'] > 0).astype(float)
            # Handle truth format (may be 0/1 or -1/+1)
            charge_truth = data['charge_truth']
            if np.min(charge_truth) < 0:
                charge_truth_binary = (charge_truth > 0).astype(float)
            else:
                charge_truth_binary = charge_truth
            stats['charge_accuracy'] = np.mean(charge_pred_binary == charge_truth_binary)
        
        return stats
    
    def run_evaluation(self):
        """Run complete evaluation."""
        print("\n" + "=" * 80)
        print("MAMBA TRACK REGRESSION EVALUATION")
        print("=" * 80)
        
        # Load predictions and test data (for station indices)
        self.load_predictions_and_data()
        
        # Apply filtering
        all_data, baseline_data, rejected_data = self.apply_baseline_filtering()
        
        # Evaluate each category
        all_stats = self.evaluate_category(all_data, self.all_tracks_dir, "All Tracks")
        baseline_stats = self.evaluate_category(baseline_data, self.baseline_dir, "Baseline Tracks")
        rejected_stats = self.evaluate_category(rejected_data, self.rejected_dir, "Rejected Tracks")
        
        # Write statistics
        print("\n" + "=" * 50)
        print("Writing statistics summary...")
        self.write_statistics(all_stats, baseline_stats, rejected_stats)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mamba Track Regression Model')
    parser.add_argument('--pred_path', '-p', type=str, required=True,
                        help='Path to predictions HDF5 file')
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                        help='Path to test data directory (for station indices)')
    parser.add_argument('--output_dir', '-o', type=str, default='./mamba_evaluation_results',
                        help='Base output directory for plots and results')
    parser.add_argument('--max_tracks', '-m', type=int, default=None,
                        help='Maximum number of tracks to process (default: all)')
    
    args = parser.parse_args()
    
    try:
        evaluator = MambaRegressionEvaluator(
            pred_path=args.pred_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_tracks=args.max_tracks
        )
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
