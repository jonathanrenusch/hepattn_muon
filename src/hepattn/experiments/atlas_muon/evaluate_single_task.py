#!/usr/bin/env python3
"""
Standalone Evaluation Script for Single-Task Mamba Models

This script evaluates predictions from separate single-task models (eta, phi, pt, charge)
by loading data directly from the HDF5 prediction files (which contain both predictions 
and targets). This avoids the complexity of matching with dataset iteration.

Each prediction file has structure:
    sample_id/preds/final/mamba_{task}/{parameter(s)}
    sample_id/targets/{eta, phi, pt, charge}

Usage:
    python evaluate_single_task.py \
        --eta_pred_path /path/to/eta_predictions.h5 \
        --phi_pred_path /path/to/phi_predictions.h5 \
        --pt_pred_path /path/to/pt_predictions.h5 \
        --charge_pred_path /path/to/charge_predictions.h5 \
        --output_dir ./single_task_evaluation \
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


def angular_difference_abs(phi1, phi2):
    """Compute absolute angular difference handling periodicity.
    
    This matches the training metric calculation:
    Returns absolute angular difference in range [0, π].
    """
    diff = np.abs(phi1 - phi2)
    return np.minimum(diff, 2 * np.pi - diff)


def angular_difference_signed(phi1, phi2):
    """Compute signed angular difference handling periodicity at ±π.
    
    Returns signed angular difference in range [-π, π].
    Used for residual distributions where we want to see bias direction.
    """
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


class SingleTaskEvaluator:
    """Evaluator for single-task Mamba models.
    
    Loads predictions directly from HDF5 files without dataset matching.
    """
    
    def __init__(self, pred_paths, output_dir, max_tracks=None):
        """Initialize evaluator.
        
        Args:
            pred_paths: dict with keys 'eta', 'phi', 'pt', 'charge' -> file paths
            output_dir: Output directory for plots and statistics
            max_tracks: Maximum number of tracks to process (None = all)
        """
        self.pred_paths = pred_paths
        self.max_tracks = max_tracks
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"single_task_eval_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for plot types
        (self.output_dir / "distributions").mkdir(exist_ok=True)
        (self.output_dir / "residuals").mkdir(exist_ok=True)
        (self.output_dir / "precision").mkdir(exist_ok=True)
        (self.output_dir / "resolution").mkdir(exist_ok=True)
        (self.output_dir / "charge_classification").mkdir(exist_ok=True)
        
        print(f"Single-Task Evaluator initialized")
        print(f"Output directory: {self.output_dir}")
        for task, path in pred_paths.items():
            if path:
                print(f"  {task}: {path}")
        print(f"Max tracks: {max_tracks if max_tracks else 'all'}")
    
    def load_predictions(self):
        """Load predictions and targets directly from HDF5 files."""
        print("\nLoading predictions from HDF5 files...")
        
        # Storage
        all_eta_pred = []
        all_phi_pred = []
        all_pt_pred = []
        all_charge_pred = []
        all_charge_prob = []
        
        all_eta_truth = []
        all_phi_truth = []
        all_pt_truth = []
        all_charge_truth = []
        
        # Determine which file to use for sample iteration and targets
        # (any file will work since they all have the same samples)
        primary_task = None
        primary_path = None
        for task in ['eta', 'phi', 'pt', 'charge']:
            if self.pred_paths.get(task):
                primary_task = task
                primary_path = self.pred_paths[task]
                break
        
        if primary_path is None:
            raise ValueError("No prediction files provided!")
        
        # Open all available prediction files
        pred_files = {}
        preds_paths = {}  # task -> path within file
        
        for task, path in self.pred_paths.items():
            if path:
                pred_files[task] = h5py.File(path, 'r')
        
        # Get sample IDs from primary file
        primary_file = pred_files[primary_task]
        sample_ids = sorted([k for k in primary_file.keys() if k.isdigit()], key=int)
        print(f"Found {len(sample_ids)} samples in {primary_task} file")
        
        # Detect prediction structure from first sample
        first_sample = primary_file[sample_ids[0]]
        preds_group = first_sample.get('preds', None)
        
        if preds_group is not None and 'final' in preds_group:
            # Structure: preds/final/mamba_{task}/{params}
            for task in pred_files.keys():
                task_file = pred_files[task]
                sample_preds = task_file[sample_ids[0]]['preds']['final']
                # Find the task-specific group (e.g., mamba_eta, mamba_phi, etc.)
                for key in sample_preds.keys():
                    if task in key.lower():
                        preds_paths[task] = f'preds/final/{key}'
                        print(f"  {task} prediction path: {preds_paths[task]}")
                        break
        
        # Limit samples if max_tracks specified
        if self.max_tracks:
            sample_ids = sample_ids[:self.max_tracks]
        
        # Load data from each sample
        for sample_id in tqdm(sample_ids, desc="Loading samples"):
            try:
                # Load targets from primary file
                targets = primary_file[sample_id]['targets']
                eta_truth = float(targets['eta'][...].flatten()[0])
                phi_truth = float(targets['phi'][...].flatten()[0])
                pt_truth = float(targets['pt'][...].flatten()[0])
                charge_truth = float(targets['charge'][...].flatten()[0])
                
                all_eta_truth.append(eta_truth)
                all_phi_truth.append(phi_truth)
                all_pt_truth.append(pt_truth)
                all_charge_truth.append(charge_truth)
                
                # Load eta prediction
                if 'eta' in pred_files and 'eta' in preds_paths:
                    preds = pred_files['eta'][sample_id][preds_paths['eta']]
                    all_eta_pred.append(float(preds['eta'][...].flatten()[0]))
                else:
                    all_eta_pred.append(np.nan)
                
                # Load phi prediction
                if 'phi' in pred_files and 'phi' in preds_paths:
                    preds = pred_files['phi'][sample_id][preds_paths['phi']]
                    all_phi_pred.append(float(preds['phi'][...].flatten()[0]))
                else:
                    all_phi_pred.append(np.nan)
                
                # Load pt prediction
                if 'pt' in pred_files and 'pt' in preds_paths:
                    preds = pred_files['pt'][sample_id][preds_paths['pt']]
                    all_pt_pred.append(float(preds['pt'][...].flatten()[0]))
                else:
                    all_pt_pred.append(np.nan)
                
                # Load charge prediction
                if 'charge' in pred_files and 'charge' in preds_paths:
                    preds = pred_files['charge'][sample_id][preds_paths['charge']]
                    all_charge_pred.append(float(preds['charge'][...].flatten()[0]))
                    if 'charge_prob' in preds:
                        all_charge_prob.append(float(preds['charge_prob'][...].flatten()[0]))
                    else:
                        all_charge_prob.append(np.nan)
                else:
                    all_charge_pred.append(np.nan)
                    all_charge_prob.append(np.nan)
                    
            except (KeyError, IndexError) as e:
                continue
        
        # Close files
        for f in pred_files.values():
            f.close()
        
        # Convert to numpy arrays
        self.data = {
            'eta_pred': np.array(all_eta_pred),
            'phi_pred': np.array(all_phi_pred),
            'pt_pred': np.array(all_pt_pred),
            'charge_pred': np.array(all_charge_pred),
            'charge_prob': np.array(all_charge_prob),
            'eta_truth': np.array(all_eta_truth),
            'phi_truth': np.array(all_phi_truth),
            'pt_truth': np.array(all_pt_truth),
            'charge_truth': np.array(all_charge_truth),
        }
        
        print(f"\nLoaded {len(self.data['eta_truth']):,} tracks")
        print(f"  Eta predictions: {np.sum(~np.isnan(self.data['eta_pred'])):,}")
        print(f"  Phi predictions: {np.sum(~np.isnan(self.data['phi_pred'])):,}")
        print(f"  Pt predictions: {np.sum(~np.isnan(self.data['pt_pred'])):,}")
        print(f"  Charge predictions: {np.sum(~np.isnan(self.data['charge_pred'])):,}")
        
        return self.data
    
    def apply_atlas_style(self, ax, subtext=None):
        """Apply ATLAS-style formatting without ATLAS label."""
        if ATLASIFY_AVAILABLE:
            atlasify.atlasify(
                atlas=False,
                subtext=subtext if subtext else "",
                font_size=14,
                sub_font_size=11,
                label_font_size=14
            )
    
    def save_plot(self, fig, subdir, filename):
        """Save plot in both PNG and PDF formats."""
        output_dir = self.output_dir / subdir
        png_path = output_dir / f"{filename}.png"
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        return png_path, pdf_path
    
    def plot_distribution_comparison(self, param):
        """Plot overlaid distribution comparing predictions with truth."""
        pred_key = f'{param}_pred'
        truth_key = f'{param}_truth'
        
        predictions = self.data[pred_key]
        truth = self.data[truth_key]
        
        # Filter NaNs
        valid = ~np.isnan(predictions)
        predictions = predictions[valid]
        truth = truth[valid]
        
        if len(predictions) == 0:
            print(f"Warning: No valid data for {param} distribution")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set bins based on parameter
        if param == 'eta':
            bins = np.linspace(-3, 3, 60)
        elif param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 60)
        elif param == 'pt':
            bins = np.linspace(0, 200, 60)
        elif param == 'charge':
            bins = np.linspace(-1.5, 1.5, 7)
        
        # Plot histograms
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
        
        self.apply_atlas_style(ax, subtext=f"N = {len(predictions):,}")
        
        self.save_plot(fig, "distributions", f"{param}_distribution_comparison")
        print(f"  Saved {param} distribution comparison")
    
    def plot_residuals(self, param):
        """Plot residual distribution."""
        pred_key = f'{param}_pred'
        truth_key = f'{param}_truth'
        
        predictions = self.data[pred_key]
        truth = self.data[truth_key]
        
        # Filter NaNs
        valid = ~np.isnan(predictions)
        predictions = predictions[valid]
        truth = truth[valid]
        
        if len(predictions) == 0:
            print(f"Warning: No valid data for {param} residuals")
            return None
        
        # Calculate residuals (handle phi periodicity)
        if param == 'phi':
            # Use absolute angular difference for STD (matches training)
            residuals_abs = angular_difference_abs(predictions, truth)
            # Use signed for distribution plot to see bias direction
            residuals = angular_difference_signed(predictions, truth)
            # STD computed on absolute (matching training)
            std_res = np.std(residuals_abs)
            mean_res = np.mean(residuals)  # Mean on signed to see bias
        else:
            residuals = predictions - truth
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set bins
        p1, p99 = np.percentile(residuals, [1, 99])
        bins = np.linspace(p1, p99, 60)
        
        # Plot histogram
        ax.hist(residuals, bins=bins, alpha=0.7, color='royalblue',
                histtype='stepfilled', edgecolor='black', linewidth=0.5)
        
        # Add vertical lines
        ax.axvline(mean_res, color='red', linestyle='--', linewidth=2)
        ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$ [rad]', 'pt': r'$p_T$ [GeV]'}
        ax.set_xlabel(f'{param_labels.get(param, param)} Residual (Pred - Truth)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        stats_subtext = f"Mean: {mean_res:.4f}, STD: {std_res:.4f}\nN = {len(residuals):,}"
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, "residuals", f"{param}_residuals")
        print(f"  Saved {param} residuals plot")
        
        return {'mean': mean_res, 'std': std_res, 'n': len(residuals)}
    
    def plot_pt_resolution_vs_variable(self, bin_param):
        """Plot pT resolution binned over a variable."""
        pt_pred = self.data['pt_pred']
        pt_truth = self.data['pt_truth']
        bin_values = self.data[f'{bin_param}_truth']
        
        # Filter valid predictions and pT range [5, 200] GeV
        valid = ~np.isnan(pt_pred) & (pt_truth >= 5) & (pt_truth <= 200)
        pt_pred = pt_pred[valid]
        pt_truth = pt_truth[valid]
        bin_values = bin_values[valid]
        
        if len(pt_pred) == 0:
            print(f"Warning: No valid data for pT resolution vs {bin_param}")
            return None
        
        # Calculate resolution
        resolution = np.abs(pt_pred - pt_truth) / pt_truth
        
        # Define bins
        if bin_param == 'eta':
            bins = np.linspace(-2.7, 2.7, 21)
        elif bin_param == 'phi':
            bins = np.linspace(-np.pi, np.pi, 21)
        elif bin_param == 'pt':
            bins = np.linspace(5, 200, 21)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate mean and std in each bin
        mean_resolutions = []
        mean_errors = []
        
        for i in range(len(bins) - 1):
            mask = (bin_values >= bins[i]) & (bin_values < bins[i+1])
            bin_res = resolution[mask]
            
            if len(bin_res) > 2:
                mean_resolutions.append(np.mean(bin_res))
                mean_errors.append(np.std(bin_res) / np.sqrt(len(bin_res)))
            else:
                mean_resolutions.append(np.nan)
                mean_errors.append(np.nan)
        
        mean_resolutions = np.array(mean_resolutions)
        mean_errors = np.array(mean_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot
        valid_bins = ~np.isnan(mean_resolutions)
        ax.errorbar(bin_centers[valid_bins], mean_resolutions[valid_bins], 
                    yerr=mean_errors[valid_bins],
                    fmt='o', color='royalblue', markersize=6, capsize=4, zorder=3)
        
        # Add horizontal bars for bins
        for i, (is_valid, res_val) in enumerate(zip(valid_bins, mean_resolutions)):
            if is_valid:
                ax.hlines(res_val, bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Overall statistics
        mean_res = np.mean(resolution)
        median_res = np.median(resolution)
        std_res = np.std(resolution)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]'}
        ax.set_xlabel(f'Truth {param_labels.get(bin_param, bin_param)}', fontsize=14)
        ax.set_ylabel(r'$p_T$ Resolution: $|p_T^{pred} - p_T^{truth}| / p_T^{truth}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        stats_subtext = (r"$p_T \in [5, 200]$ GeV" + 
                        f"\nMean: {mean_res:.4f}, Median: {median_res:.4f}\nN = {len(resolution):,}")
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, "resolution", f"pt_resolution_vs_{bin_param}")
        print(f"  Saved pT resolution vs {bin_param}")
        
        return {'mean': mean_res, 'median': median_res, 'std': std_res, 'n': len(resolution)}
    
    def plot_precision_vs_variable(self, target_param, bin_param):
        """Plot precision (STD of residuals) binned over a variable."""
        pred_key = f'{target_param}_pred'
        truth_key = f'{target_param}_truth'
        bin_truth_key = f'{bin_param}_truth'
        
        predictions = self.data[pred_key]
        truth = self.data[truth_key]
        bin_values = self.data[bin_truth_key]
        
        # Filter valid
        valid = ~np.isnan(predictions)
        predictions = predictions[valid]
        truth = truth[valid]
        bin_values = bin_values[valid]
        
        if len(predictions) == 0:
            print(f"Warning: No valid data for {target_param} precision vs {bin_param}")
            return
        
        # Calculate residuals
        if target_param == 'phi':
            # Use absolute angular difference for STD (matches training)
            residuals = angular_difference_abs(predictions, truth)
        else:
            residuals = predictions - truth
        
        # Define bins
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
        
        for i in range(len(bins) - 1):
            mask = (bin_values >= bins[i]) & (bin_values < bins[i+1])
            bin_res = residuals[mask]
            
            if len(bin_res) > 2:
                std = np.std(bin_res)
                std_err = std / np.sqrt(2 * (len(bin_res) - 1))
                stds.append(std)
                std_errors.append(std_err)
            else:
                stds.append(np.nan)
                std_errors.append(np.nan)
        
        stds = np.array(stds)
        std_errors = np.array(std_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot
        valid_bins = ~np.isnan(stds)
        ax.errorbar(bin_centers[valid_bins], stds[valid_bins], yerr=std_errors[valid_bins],
                    fmt='o', color='royalblue', markersize=6, capsize=4, zorder=3)
        
        # Add horizontal bars
        for i, (is_valid, std_val) in enumerate(zip(valid_bins, stds)):
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
        
        unbinned_std = np.std(residuals)
        stats_subtext = f"Unbinned STD: {unbinned_std:.4f}\nN = {len(residuals):,}"
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, "precision", f"{target_param}_precision_vs_{bin_param}")
        print(f"  Saved {target_param} precision vs {bin_param}")
    
    def plot_charge_accuracy_vs_variable(self, bin_param):
        """Plot charge classification accuracy binned over a variable."""
        charge_pred = self.data['charge_pred']
        charge_truth = self.data['charge_truth']
        bin_values = self.data[f'{bin_param}_truth']
        
        # Filter valid charge predictions
        valid = ~np.isnan(charge_pred)
        
        if valid.sum() == 0:
            print(f"Warning: No valid charge predictions for charge accuracy vs {bin_param} - skipping")
            return
        
        charge_pred = charge_pred[valid]
        charge_truth = charge_truth[valid]
        bin_values = bin_values[valid]
        
        # Convert to binary (predictions are -1/+1, truth is 0/1)
        charge_pred_binary = (charge_pred > 0).astype(float)
        if np.min(charge_truth) < 0:
            charge_truth_binary = (charge_truth > 0).astype(float)
        else:
            charge_truth_binary = charge_truth
        
        # Calculate correct predictions
        correct = (charge_pred_binary == charge_truth_binary).astype(float)
        
        # Define bins
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
        
        for i in range(len(bins) - 1):
            mask = (bin_values >= bins[i]) & (bin_values < bins[i+1])
            bin_correct = correct[mask]
            
            if len(bin_correct) > 0:
                acc = np.mean(bin_correct)
                acc_err = np.sqrt(acc * (1 - acc) / len(bin_correct)) if len(bin_correct) > 1 else 0
                accuracies.append(acc)
                acc_errors.append(acc_err)
            else:
                accuracies.append(np.nan)
                acc_errors.append(np.nan)
        
        accuracies = np.array(accuracies)
        acc_errors = np.array(acc_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot
        valid_bins = ~np.isnan(accuracies)
        ax.errorbar(bin_centers[valid_bins], accuracies[valid_bins], yerr=acc_errors[valid_bins],
                    fmt='o', color='royalblue', markersize=6, capsize=4, zorder=3)
        
        # Add horizontal bars
        for i, (is_valid, acc_val) in enumerate(zip(valid_bins, accuracies)):
            if is_valid:
                ax.hlines(acc_val, bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Reference line at 0.5
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        
        # Labels
        param_labels = {'eta': r'$\eta$', 'phi': r'$\phi$', 'pt': r'$p_T$ [GeV]'}
        ax.set_xlabel(f'Truth {param_labels.get(bin_param, bin_param)}', fontsize=14)
        ax.set_ylabel('Charge Classification Accuracy', fontsize=14)
        ax.set_ylim([0.4, 1.05])
        ax.grid(True, alpha=0.3)
        
        unbinned_accuracy = np.mean(correct)
        stats_subtext = f"Accuracy: {unbinned_accuracy:.4f}\nN = {len(correct):,}"
        self.apply_atlas_style(ax, subtext=stats_subtext)
        
        self.save_plot(fig, "charge_classification", f"charge_accuracy_vs_{bin_param}")
        print(f"  Saved charge accuracy vs {bin_param}")
    
    def calculate_charge_auc(self):
        """Calculate AUC for charge classification."""
        charge_prob = self.data['charge_prob']
        charge_truth = self.data['charge_truth']
        
        # Filter valid
        valid = ~np.isnan(charge_prob)
        charge_prob = charge_prob[valid]
        charge_truth = charge_truth[valid]
        
        if len(charge_prob) == 0:
            return None
        
        # Convert truth to binary
        if np.min(charge_truth) < 0:
            truth_binary = (charge_truth > 0).astype(int)
        else:
            truth_binary = charge_truth.astype(int)
        
        try:
            auc = roc_auc_score(truth_binary, charge_prob)
            print(f"  Charge AUC: {auc:.4f}")
            return auc
        except Exception as e:
            print(f"  Warning: Could not calculate AUC: {e}")
            return None
    
    def compute_training_compatible_metrics(self):
        """Compute metrics exactly as training does for comparison.
        
        This ensures we can directly compare validation metrics during training
        with evaluation metrics from this script.
        
        Training computes:
        - std_eta: std(eta_pred - eta_truth)
        - std_phi: std(angular_difference_abs(phi_pred, phi_truth))
        - std_pt: std(pt_pred - pt_truth)
        """
        metrics = {}
        
        # Eta (signed residuals)
        eta_pred = self.data['eta_pred']
        eta_truth = self.data['eta_truth']
        valid_eta = ~np.isnan(eta_pred)
        if valid_eta.sum() > 0:
            eta_residuals = eta_pred[valid_eta] - eta_truth[valid_eta]
            metrics['std_eta'] = np.std(eta_residuals)
            metrics['mae_eta'] = np.mean(np.abs(eta_residuals))
            metrics['n_eta'] = valid_eta.sum()
        
        # Phi (absolute angular difference - matches training)
        phi_pred = self.data['phi_pred']
        phi_truth = self.data['phi_truth']
        valid_phi = ~np.isnan(phi_pred)
        if valid_phi.sum() > 0:
            phi_residuals = angular_difference_abs(phi_pred[valid_phi], phi_truth[valid_phi])
            metrics['std_phi'] = np.std(phi_residuals)
            metrics['mae_phi'] = np.mean(phi_residuals)  # MAE = mean since it's already absolute
            metrics['n_phi'] = valid_phi.sum()
        
        # Pt (signed residuals)
        pt_pred = self.data['pt_pred']
        pt_truth = self.data['pt_truth']
        valid_pt = ~np.isnan(pt_pred)
        if valid_pt.sum() > 0:
            pt_residuals = pt_pred[valid_pt] - pt_truth[valid_pt]
            metrics['std_pt'] = np.std(pt_residuals)
            metrics['mae_pt'] = np.mean(np.abs(pt_residuals))
            metrics['n_pt'] = valid_pt.sum()
            
            # Relative resolution (matching training)
            rel_res = np.abs(pt_residuals) / (np.abs(pt_truth[valid_pt]) + 1e-8)
            metrics['rel_res_pt'] = np.mean(rel_res)
        
        return metrics
    
    def write_statistics(self, stats):
        """Write statistics summary to text file."""
        stats_path = self.output_dir / "evaluation_statistics.txt"
        
        with open(stats_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SINGLE-TASK MAMBA EVALUATION STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("Prediction files:\n")
            for task, path in self.pred_paths.items():
                if path:
                    f.write(f"  {task}: {path}\n")
            f.write("\n")
            
            f.write("-" * 40 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of tracks: {stats.get('n_tracks', 'N/A'):,}\n\n")
            
            # Residual statistics
            for param in ['eta', 'phi', 'pt']:
                if f'{param}_residuals' in stats:
                    res = stats[f'{param}_residuals']
                    f.write(f"{param.upper()} Residuals:\n")
                    f.write(f"  Mean: {res['mean']:.6f}\n")
                    f.write(f"  STD:  {res['std']:.6f}\n")
                    f.write(f"  N:    {res['n']:,}\n\n")
            
            # pT resolution
            if 'pt_resolution' in stats:
                res = stats['pt_resolution']
                f.write(f"pT Resolution (|pred-truth|/truth):\n")
                f.write(f"  Mean:   {res['mean']:.6f}\n")
                f.write(f"  Median: {res['median']:.6f}\n")
                f.write(f"  STD:    {res['std']:.6f}\n")
                f.write(f"  N:      {res['n']:,}\n\n")
            
            # Charge metrics
            if 'charge_auc' in stats:
                f.write(f"Charge Classification AUC: {stats['charge_auc']:.4f}\n")
            if 'charge_accuracy' in stats:
                f.write(f"Charge Classification Accuracy: {stats['charge_accuracy']:.4f}\n")
            
            # Training-compatible metrics section
            f.write("\n" + "-" * 40 + "\n")
            f.write("TRAINING-COMPATIBLE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write("These metrics match the training validation metrics exactly:\n")
            f.write("(Same formulas used during training)\n\n")
            
            if 'training_metrics' in stats:
                tm = stats['training_metrics']
                if 'std_eta' in tm:
                    f.write(f"std_eta:  {tm['std_eta']:.6f}  (training uses signed residuals)\n")
                if 'std_phi' in tm:
                    f.write(f"std_phi:  {tm['std_phi']:.6f}  (training uses |angular_diff|)\n")
                if 'std_pt' in tm:
                    f.write(f"std_pt:   {tm['std_pt']:.6f}  (training uses signed residuals)\n")
                f.write("\n")
                if 'mae_eta' in tm:
                    f.write(f"mae_eta:  {tm['mae_eta']:.6f}\n")
                if 'mae_phi' in tm:
                    f.write(f"mae_phi:  {tm['mae_phi']:.6f}\n")
                if 'mae_pt' in tm:
                    f.write(f"mae_pt:   {tm['mae_pt']:.6f}\n")
                if 'rel_res_pt' in tm:
                    f.write(f"rel_res_pt: {tm['rel_res_pt']:.6f}\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write("COMPARISON WITH TRAINING VALIDATION\n")
            f.write("-" * 40 + "\n")
            f.write("If these differ significantly from training validation metrics,\n")
            f.write("possible causes include:\n")
            f.write("  - Test dataset differs from validation dataset\n")
            f.write("  - Different pT range in data\n")
            f.write("  - Batch averaging vs full dataset statistics\n")
        
        print(f"\nStatistics written to {stats_path}")
    
    def run_evaluation(self):
        """Run complete evaluation."""
        print("\n" + "=" * 80)
        print("SINGLE-TASK MAMBA EVALUATION")
        print("=" * 80)
        
        # Load predictions
        self.load_predictions()
        
        stats = {'n_tracks': len(self.data['eta_truth'])}
        
        # 1. Distribution comparison plots
        print("\nGenerating distribution comparison plots...")
        for param in ['eta', 'phi', 'pt', 'charge']:
            self.plot_distribution_comparison(param)
        
        # 2. Residual plots
        print("\nGenerating residual plots...")
        for param in ['eta', 'phi', 'pt']:
            res_stats = self.plot_residuals(param)
            if res_stats:
                stats[f'{param}_residuals'] = res_stats
        
        # 3. pT resolution plots
        print("\nGenerating pT resolution plots...")
        for bin_param in ['pt', 'eta', 'phi']:
            pt_res_stats = self.plot_pt_resolution_vs_variable(bin_param)
            if pt_res_stats and bin_param == 'pt':
                stats['pt_resolution'] = pt_res_stats
        
        # 4. Precision plots (9 combinations)
        print("\nGenerating precision plots...")
        for target_param in ['eta', 'phi', 'pt']:
            for bin_param in ['eta', 'phi', 'pt']:
                self.plot_precision_vs_variable(target_param, bin_param)
        
        # 5. Charge accuracy plots
        print("\nGenerating charge classification accuracy plots...")
        n_valid_charge = np.sum(~np.isnan(self.data['charge_pred']))
        if n_valid_charge > 0:
            print(f"  Found {n_valid_charge:,} valid charge predictions")
            for bin_param in ['eta', 'phi', 'pt']:
                self.plot_charge_accuracy_vs_variable(bin_param)
        else:
            print("  Warning: No charge predictions provided - skipping charge accuracy plots")
        
        # 6. Charge AUC
        if n_valid_charge > 0:
            print("\nCalculating charge classification AUC...")
            auc = self.calculate_charge_auc()
            if auc is not None:
                stats['charge_auc'] = auc
        
        # 7. Overall charge accuracy
        charge_pred = self.data['charge_pred']
        charge_truth = self.data['charge_truth']
        valid = ~np.isnan(charge_pred)
        if valid.sum() > 0:
            pred_binary = (charge_pred[valid] > 0).astype(float)
            if np.min(charge_truth[valid]) < 0:
                truth_binary = (charge_truth[valid] > 0).astype(float)
            else:
                truth_binary = charge_truth[valid]
            stats['charge_accuracy'] = np.mean(pred_binary == truth_binary)
            print(f"  Charge Accuracy: {stats['charge_accuracy']:.4f}")
        
        # 8. Compute training-compatible metrics
        print("\nComputing training-compatible metrics...")
        training_metrics = self.compute_training_compatible_metrics()
        stats['training_metrics'] = training_metrics
        
        # Print training-compatible metrics to console
        print("\n" + "=" * 50)
        print("TRAINING-COMPATIBLE METRICS (for direct comparison)")
        print("=" * 50)
        if 'std_eta' in training_metrics:
            print(f"  std_eta:  {training_metrics['std_eta']:.6f}")
        if 'std_phi' in training_metrics:
            print(f"  std_phi:  {training_metrics['std_phi']:.6f}")
        if 'std_pt' in training_metrics:
            print(f"  std_pt:   {training_metrics['std_pt']:.2f}")
        
        # Write statistics
        print("\n" + "=" * 50)
        print("Writing statistics summary...")
        self.write_statistics(stats)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Single-Task Mamba Models')
    
    parser.add_argument('--eta_pred_path', type=str, default=None,
                        help='Path to eta predictions HDF5 file')
    parser.add_argument('--phi_pred_path', type=str, default=None,
                        help='Path to phi predictions HDF5 file')
    parser.add_argument('--pt_pred_path', type=str, default=None,
                        help='Path to pt predictions HDF5 file')
    parser.add_argument('--charge_pred_path', type=str, default=None,
                        help='Path to charge predictions HDF5 file')
    parser.add_argument('--output_dir', '-o', type=str, default='./single_task_evaluation',
                        help='Output directory for plots and results')
    parser.add_argument('--max_tracks', '-m', type=int, default=None,
                        help='Maximum number of tracks to process (default: all)')
    
    args = parser.parse_args()
    
    pred_paths = {
        'eta': args.eta_pred_path,
        'phi': args.phi_pred_path,
        'pt': args.pt_pred_path,
        'charge': args.charge_pred_path,
    }
    
    if not any(p is not None for p in pred_paths.values()):
        print("Error: Must specify at least one prediction file.")
        sys.exit(1)
    
    try:
        evaluator = SingleTaskEvaluator(
            pred_paths=pred_paths,
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
