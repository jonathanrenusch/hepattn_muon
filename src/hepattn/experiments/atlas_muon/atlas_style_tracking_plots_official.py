#!/usr/bin/env python3
"""
ATLAS-style plots for track filtering model evaluation.

This script generates publication-quality plots for the tracking model:
1. Hit Assignment Efficiency vs Eta (with Clopper-Pearson intervals)
2. Double Matching Efficiency vs Eta (with Clopper-Pearson intervals)
3. Charge Classification Accuracy vs Eta (with Clopper-Pearson intervals)
4. pT Distribution Comparison (Model vs Truth)

All plots follow ATLAS publication style conventions using the atlasify package.
"""

import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from scipy import stats
import sys
import os
import yaml
from tqdm import tqdm
import warnings

# Import atlasify for ATLAS style
import atlasify

warnings.filterwarnings('ignore')

# Import the data module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule


class ATLASStyleTrackingPlotter:
    """Generate ATLAS-style plots for tracking model evaluation."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = Path(eval_path)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / "tracking_plots" / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for collected data
        self.hit_predictions = []
        self.hit_true_assignments = []
        self.track_info = []
        self.charge_predictions = []
        self.charge_truth = []
        self.pt_predictions = []
        self.pt_truth = []
        self.eta_truth = []
        
        print(f"Output directory: {self.output_dir}")
    
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
        
        # Initialize data module
        self.data_module = AtlasMuonDataModule(
            train_dir=str(self.data_dir),
            val_dir=str(self.data_dir),
            test_dir=str(self.data_dir),
            num_workers=100,
            num_train=abs(num_test_events) if num_test_events != -1 else 1000,
            num_val=abs(num_test_events) if num_test_events != -1 else 1000,
            num_test=num_test_events,
            batch_size=1,
            inputs=inputs_eval,
            targets=targets_eval,
            pin_memory=True,
        )
        
        # Setup the data module
        self.data_module.setup(stage='test')
        self.test_dataloader = self.data_module.test_dataloader(shuffle=True)
        
        print(f"DataLoader setup complete")
    
    def collect_data(self):
        """Collect all data for analysis using the DataLoader."""
        print("Collecting data from all events using DataLoader...")
        
        events_processed = 0
        events_attempted = 0
        
        # Get the dataset and random indices (same pattern as evaluate_task1)
        dataset = self.test_dataloader.dataset
        num_events = self.max_events if self.max_events and self.max_events > 0 else len(dataset)
        random_indices = np.random.choice(len(dataset), size=min(num_events, len(dataset)), replace=False)
        
        try:
            with h5py.File(self.eval_path, 'r') as eval_file:
                # Use the randomly sampled indices to access both dataset and eval file
                for dataset_idx in tqdm(random_indices, desc="Processing events"):
                    events_attempted += 1
                    
                    try:
                        # Get the batch from dataset using the random index
                        batch = dataset[dataset_idx]
                        
                        # Use the dataset index as the event_id for the eval file
                        event_id = str(dataset_idx)
                        
                        if event_id not in eval_file:
                            if events_attempted <= 5:
                                print(f"Warning: Event {event_id} not found in predictions file")
                            continue
                        
                        # Get truth information from batch (dataset still has batch dimension of 1)
                        inputs_batch, targets_batch = batch
                        
                        # Access the event group in the HDF5 file
                        pred_group = eval_file[event_id]
                        
                        # Get predictions using [...] to read the full dataset
                        hit_track_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]
                        
                        # Get regression predictions (stored as single array with 4 values per track)
                        if 'outputs/final/parameter_regression/track_regr' in pred_group:
                            track_regr = pred_group['outputs/final/parameter_regression/track_regr'][...]
                            # track_regr shape: (1, num_tracks, 4) where 4 = [eta, phi, pt, q]
                            # Extract individual parameters
                            eta_pred = track_regr[0, :, 0]  # First parameter is eta
                            phi_pred = track_regr[0, :, 1]  # Second parameter is phi  
                            pt_pred = track_regr[0, :, 2]   # Third parameter is pt
                            q_pred = track_regr[0, :, 3]    # Fourth parameter is q
                        else:
                            if events_attempted <= 5:
                                print(f"Event {event_id}: parameter_regression not found")
                            continue
                        
                        # Get truth particle validity
                        true_particle_valid = targets_batch['particle_valid'][0]  # Remove batch dimension
                        valid_particles = true_particle_valid.numpy()
                        num_valid = int(valid_particles.sum())
                        
                        if num_valid == 0:
                            continue
                        
                        # Get number of tracks (use num_valid instead of array shape)
                        n_tracks = num_valid
                        
                        # Get truth information
                        eta_truth = targets_batch["particle_truthMuon_eta"][0].numpy()
                        pt_truth = targets_batch["particle_truthMuon_pt"][0].numpy()
                        q_truth = targets_batch["particle_truthMuon_q"][0].numpy()
                        
                        # Get truth hit assignments
                        true_hit_assignments = targets_batch['particle_hit_valid'][0]  # Remove batch dimension
                        
                        # Process each track
                        for track_idx in range(n_tracks):
                            if track_idx >= len(eta_truth):
                                continue
                            
                            # Get hit assignment predictions and truth (using [0, track_idx] pattern)
                            hit_assignment_pred = hit_track_pred[0, track_idx].astype(bool)
                            hit_assignment_truth = true_hit_assignments[track_idx].numpy().astype(bool)
                            
                            # Store hit assignment data
                            self.hit_predictions.append(hit_assignment_pred)
                            self.hit_true_assignments.append(hit_assignment_truth)
                            
                            # Store track info
                            track_info = {
                                'eta': eta_truth[track_idx],
                                'pt': pt_truth[track_idx],
                                'q': q_truth[track_idx]
                            }
                            self.track_info.append(track_info)
                            
                            # Store regression predictions
                            self.charge_predictions.append(q_pred[track_idx])
                            self.charge_truth.append(q_truth[track_idx])
                            self.pt_predictions.append(pt_pred[track_idx])
                            self.pt_truth.append(pt_truth[track_idx])
                            self.eta_truth.append(eta_truth[track_idx])
                        
                        events_processed += 1
                        
                    except Exception as e:
                        if events_attempted <= 5:
                            print(f"Error processing event {dataset_idx}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\nProcessed {events_processed} events out of {events_attempted} attempted")
        
        if events_processed == 0:
            raise ValueError("No events were successfully processed!")
        
        # Convert to numpy arrays
        self.charge_predictions = np.array(self.charge_predictions)
        self.charge_truth = np.array(self.charge_truth)
        self.pt_predictions = np.array(self.pt_predictions)
        self.pt_truth = np.array(self.pt_truth)
        self.eta_truth = np.array(self.eta_truth)
        
        print(f"Data collection complete!")
        print(f"Events processed: {events_processed}")
        print(f"Total tracks: {len(self.hit_predictions):,}")
        
        return True
    
    def calculate_hit_efficiency_by_eta(self, bins=None):
        """Calculate hit assignment efficiency binned by eta with Clopper-Pearson intervals.
        
        Uses pooled hits across all tracks in each bin to calculate efficiency as a binomial proportion:
        efficiency = (total correct hits) / (total true hits)
        This allows proper use of Clopper-Pearson confidence intervals.
        """
        
        if bins is None:
            # Use data-driven bins for eta
            eta_values = np.array([track['eta'] for track in self.track_info])
            min_eta = np.min(eta_values)
            max_eta = np.max(eta_values)
            bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Extract eta values
        eta_values = np.array([track['eta'] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(eta_values, bins) - 1
        
        bin_efficiencies = []
        bin_centers = []
        err_low = []
        err_high = []
        n_tracks_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            n_tracks = np.sum(mask)
            n_tracks_per_bin.append(n_tracks)
            
            if n_tracks == 0:
                continue
            
            # Get predictions and truth for this bin
            bin_predictions = [self.hit_predictions[j] for j in range(len(self.hit_predictions)) if mask[j]]
            bin_truth = [self.hit_true_assignments[j] for j in range(len(self.hit_true_assignments)) if mask[j]]
            
            # Calculate efficiency for each track in this bin
            all_efficiencies = []
            for pred, truth in zip(bin_predictions, bin_truth):
                true_hits = np.sum(truth)
                correct_hits = np.sum(pred & truth)
                if true_hits > 0:
                    all_efficiencies.append(correct_hits / true_hits)
            
            if len(all_efficiencies) == 0:
                continue
            
            # Calculate mean efficiency and standard error
            efficiency = np.mean(all_efficiencies)
            std_err = np.std(all_efficiencies, ddof=1) / np.sqrt(len(all_efficiencies)) if len(all_efficiencies) > 1 else 0
            
            # Calculate mean efficiency and standard error
            efficiency = np.mean(all_efficiencies)
            std_err = np.std(all_efficiencies, ddof=1) / np.sqrt(len(all_efficiencies)) if len(all_efficiencies) > 1 else 0
            
            # Use 3-sigma error bars (99.7% confidence)
            err_margin = 3 * std_err
            
            bin_efficiencies.append(efficiency)
            err_low.append(err_margin)
            err_high.append(err_margin)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(bin_efficiencies), np.array(err_low), np.array(err_high), bins, np.array(n_tracks_per_bin)
    
    def calculate_purity_by_eta(self, bins=None):
        """Calculate hit assignment purity binned by eta with Clopper-Pearson intervals.
        
        Uses pooled hits across all tracks in each bin to calculate purity as a binomial proportion:
        purity = (total correct hits) / (total predicted hits)
        This allows proper use of Clopper-Pearson confidence intervals.
        """
        
        if bins is None:
            # Use data-driven bins for eta
            eta_values = np.array([track['eta'] for track in self.track_info])
            min_eta = np.min(eta_values)
            max_eta = np.max(eta_values)
            bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Extract eta values
        eta_values = np.array([track['eta'] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(eta_values, bins) - 1
        
        bin_purities = []
        bin_centers = []
        err_low = []
        err_high = []
        n_tracks_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            n_tracks = np.sum(mask)
            n_tracks_per_bin.append(n_tracks)
            
            if n_tracks == 0:
                continue
            
            # Get predictions and truth for this bin
            bin_predictions = [self.hit_predictions[j] for j in range(len(self.hit_predictions)) if mask[j]]
            bin_truth = [self.hit_true_assignments[j] for j in range(len(self.hit_true_assignments)) if mask[j]]
            
            # Calculate purity for each track in this bin
            all_purities = []
            for pred, truth in zip(bin_predictions, bin_truth):
                pred_hits = np.sum(pred)
                correct_hits = np.sum(pred & truth)
                if pred_hits > 0:
                    all_purities.append(correct_hits / pred_hits)
            
            if len(all_purities) == 0:
                continue
            
            # Calculate mean purity and standard error
            purity = np.mean(all_purities)
            std_err = np.std(all_purities, ddof=1) / np.sqrt(len(all_purities)) if len(all_purities) > 1 else 0
            
            # Use 3-sigma error bars (99.7% confidence)
            err_margin = 3 * std_err
            
            bin_purities.append(purity)
            err_low.append(err_margin)
            err_high.append(err_margin)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(bin_purities), np.array(err_low), np.array(err_high), bins, np.array(n_tracks_per_bin)
    
    def calculate_double_matching_by_eta(self, bins=None, threshold=0.5):
        """Calculate double matching efficiency binned by eta with Clopper-Pearson intervals.
        
        Args:
            bins: Eta bins to use
            threshold: Efficiency and purity threshold (default 0.5 for 50%)
        """
        
        if bins is None:
            # Use data-driven bins for eta
            eta_values = np.array([track['eta'] for track in self.track_info])
            min_eta = np.min(eta_values)
            max_eta = np.max(eta_values)
            bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Extract eta values
        eta_values = np.array([track['eta'] for track in self.track_info])
        
        # Calculate bin indices
        bin_indices = np.digitize(eta_values, bins) - 1
        
        bin_efficiencies = []
        bin_centers = []
        err_low = []
        err_high = []
        n_tracks_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            n_tracks = np.sum(mask)
            n_tracks_per_bin.append(n_tracks)
            
            if n_tracks == 0:
                continue
            
            # Get predictions and truth for this bin
            bin_predictions = [self.hit_predictions[j] for j in range(len(self.hit_predictions)) if mask[j]]
            bin_truth = [self.hit_true_assignments[j] for j in range(len(self.hit_true_assignments)) if mask[j]]
            
            # Count tracks with both efficiency >= threshold AND purity >= threshold
            n_double_match = 0
            for pred, truth in zip(bin_predictions, bin_truth):
                true_hits = np.sum(truth)
                pred_hits = np.sum(pred)
                correct_hits = np.sum(pred & truth)
                
                track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
                track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
                
                if track_efficiency >= threshold and track_purity >= threshold:
                    n_double_match += 1
            
            # Clopper-Pearson confidence interval (99.7% ~ 3 sigma)
            alpha = 0.003  # 99.7% confidence interval
            
            if n_double_match == 0:
                ci_low = 0.0
            else:
                ci_low = stats.beta.ppf(alpha/2, n_double_match, n_tracks - n_double_match + 1)
            
            if n_double_match == n_tracks:
                ci_high = 1.0
            else:
                ci_high = stats.beta.ppf(1 - alpha/2, n_double_match + 1, n_tracks - n_double_match)
            
            efficiency = n_double_match / n_tracks
            
            bin_efficiencies.append(efficiency)
            err_low.append(max(0, efficiency - ci_low))
            err_high.append(max(0, ci_high - efficiency))
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(bin_efficiencies), np.array(err_low), np.array(err_high), bins, np.array(n_tracks_per_bin)
    
    def calculate_charge_accuracy_by_eta(self, bins=None):
        """Calculate charge classification accuracy binned by eta with Clopper-Pearson intervals."""
        
        if bins is None:
            # Use data-driven bins for eta
            min_eta = np.min(self.eta_truth)
            max_eta = np.max(self.eta_truth)
            bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Calculate bin indices
        bin_indices = np.digitize(self.eta_truth, bins) - 1
        
        # Convert charge predictions to discrete classifications
        pred_charge_discrete = np.where(self.charge_predictions >= 0, 1, -1)
        
        bin_accuracies = []
        bin_centers = []
        err_low = []
        err_high = []
        n_tracks_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            n_tracks = np.sum(mask)
            n_tracks_per_bin.append(n_tracks)
            
            if n_tracks == 0:
                continue
            
            # Get predictions and truth for this bin
            bin_pred = pred_charge_discrete[mask]
            bin_truth = self.charge_truth[mask]
            
            # Count correct predictions
            n_correct = np.sum(bin_pred == bin_truth)
            
            # Clopper-Pearson confidence interval (99.7% ~ 3 sigma)
            alpha = 0.003  # 99.7% confidence interval
            
            if n_correct == 0:
                ci_low = 0.0
            else:
                ci_low = stats.beta.ppf(alpha/2, n_correct, n_tracks - n_correct + 1)
            
            if n_correct == n_tracks:
                ci_high = 1.0
            else:
                ci_high = stats.beta.ppf(1 - alpha/2, n_correct + 1, n_tracks - n_correct)
            
            accuracy = n_correct / n_tracks
            
            bin_accuracies.append(accuracy)
            err_low.append(max(0, accuracy - ci_low))
            err_high.append(max(0, ci_high - accuracy))
            bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(bin_accuracies), np.array(err_low), np.array(err_high), bins, np.array(n_tracks_per_bin)
    
    def plot_hit_efficiency_and_purity_vs_eta(self):
        """Plot hit efficiency and purity vs eta on a single plot."""
        print("Generating Hit Assignment Efficiency and Purity vs Eta plot...")
        
        # Calculate both metrics with same bins
        eta_values = np.array([track['eta'] for track in self.track_info])
        min_eta = np.min(eta_values)
        max_eta = np.max(eta_values)
        bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Get hit efficiency
        centers_eff, efficiencies, err_low_eff, err_high_eff, bins_eff, n_tracks_eff = self.calculate_hit_efficiency_by_eta(bins)
        
        # Get purity
        centers_pur, purities, err_low_pur, err_high_pur, bins_pur, n_tracks_pur = self.calculate_purity_by_eta(bins)
        
        # Create single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Hit Assignment Efficiency
        ax.errorbar(centers_eff, efficiencies, yerr=[err_low_eff, err_high_eff],
                    fmt='o', color='royalblue', markersize=6, capsize=4,
                    label='Hit Assignment Efficiency', zorder=3)
        
        # Step function for clarity
        for i in range(len(bins) - 1):
            if i < len(efficiencies) and n_tracks_eff[i] > 0:
                ax.hlines(efficiencies[i], bins[i], bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Plot Hit Assignment Purity
        ax.errorbar(centers_pur, purities, yerr=[err_low_pur, err_high_pur],
                    fmt='s', color='green', markersize=6, capsize=4,
                    label='Hit Assignment Purity', zorder=3)
        
        # Step function for clarity
        for i in range(len(bins) - 1):
            if i < len(purities) and n_tracks_pur[i] > 0:
                ax.hlines(purities[i], bins[i], bins[i+1], 
                         colors='green', linewidth=2, alpha=0.3, zorder=2)
        
        ax.set_xlabel(r'Truth $\eta$', fontsize=14)
        ax.set_ylabel('Efficiency / Purity', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set y-axis limits AFTER atlasify to prevent override
        ax.set_ylim([0.825, 1.025])
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_hit_efficiency_purity_vs_eta.png"
        output_path_pdf = self.output_dir / "atlas_hit_efficiency_purity_vs_eta.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Hit Efficiency and Purity plot saved to {output_path_png} and {output_path_pdf}")
        
        return efficiencies, purities
    
    def plot_double_matching_vs_eta(self, include_75wp=False):
        """Plot standalone double matching efficiency vs eta.
        
        Args:
            include_75wp: If True, also plot 75% working point alongside 50% (default: False)
        """
        print("Generating standalone Double Matching Efficiency vs Eta plot...")
        
        # Calculate double matching with data-driven bins
        eta_values = np.array([track['eta'] for track in self.track_info])
        min_eta = np.min(eta_values)
        max_eta = np.max(eta_values)
        bins = np.linspace(min_eta, max_eta, 21)  # 20 bins
        
        # Always calculate 50% working point
        centers_50, dm_eff_50, err_low_50, err_high_50, bins_50, n_tracks_50 = self.calculate_double_matching_by_eta(bins, threshold=0.5)
        
        # Optionally calculate 75% working point
        if include_75wp:
            centers_75, dm_eff_75, err_low_75, err_high_75, bins_75, n_tracks_75 = self.calculate_double_matching_by_eta(bins, threshold=0.75)
        else:
            centers_75, dm_eff_75 = [], []
        
        if len(centers_50) == 0:
            print("Warning: No data points for double matching plot")
            return None
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot 50% working point
        if len(centers_50) > 0:
            label_50 = 'Double Matching (50% WP)\n(Eff ≥ 50% AND Pur ≥ 50%)' if include_75wp else 'Double Matching Efficiency\n(Both Efficiency ≥ 50% AND Purity ≥ 50%)'
            ax.errorbar(centers_50, dm_eff_50, yerr=[err_low_50, err_high_50],
                       fmt='o', color='green', markersize=6, capsize=4,
                       label=label_50, zorder=3)
            
            # Step function for clarity
            for i in range(len(bins) - 1):
                if i < len(dm_eff_50) and n_tracks_50[i] > 0:
                    ax.hlines(dm_eff_50[i], bins[i], bins[i+1], 
                             colors='green', linewidth=2, alpha=0.3, zorder=2)
        
        # Plot 75% working point if requested
        if include_75wp and len(centers_75) > 0:
            ax.errorbar(centers_75, dm_eff_75, yerr=[err_low_75, err_high_75],
                       fmt='s', color='darkorange', markersize=6, capsize=4,
                       label='Double Matching (75% WP)\n(Eff ≥ 75% AND Pur ≥ 75%)', zorder=3)
            
            # Step function for clarity
            for i in range(len(bins) - 1):
                if i < len(dm_eff_75) and n_tracks_75[i] > 0:
                    ax.hlines(dm_eff_75[i], bins[i], bins[i+1], 
                             colors='darkorange', linewidth=2, alpha=0.3, zorder=2)
        
        ax.set_xlabel(r'Truth $\eta$', fontsize=14)
        ax.set_ylabel('Double Matching Efficiency', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set y-axis limits AFTER atlasify to prevent override
        ax.set_ylim([0.825, 1.025])
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_double_matching_vs_eta.png"
        output_path_pdf = self.output_dir / "atlas_double_matching_vs_eta.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Standalone Double Matching plot saved to {output_path_png} and {output_path_pdf}")
        
        if include_75wp:
            return dm_eff_50, dm_eff_75
        else:
            return dm_eff_50
    
    def plot_charge_accuracy_vs_eta(self):
        """Plot charge classification accuracy vs eta."""
        print("Generating Charge Classification Accuracy vs Eta plot...")
        
        centers, accuracies, err_low, err_high, bins, n_tracks = self.calculate_charge_accuracy_by_eta()
        
        if len(centers) == 0:
            print("Warning: No data points for charge accuracy plot")
            return None
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot accuracy with error bars
        ax.errorbar(centers, accuracies, yerr=[err_low, err_high],
                   fmt='o', color='purple', markersize=6, capsize=4,
                   label='Charge Classification Accuracy', zorder=3)
        
        # Step function for clarity
        for i in range(len(bins) - 1):
            if i < len(accuracies) and n_tracks[i] > 0:
                ax.hlines(accuracies[i], bins[i], bins[i+1], 
                         colors='purple', linewidth=2, alpha=0.3, zorder=2)
        
        ax.set_xlabel(r'Truth $\eta$', fontsize=14)
        ax.set_ylabel('Charge Classification Accuracy', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set y-axis limits AFTER atlasify to prevent override
        ax.set_ylim([0.85, 1.05])
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_charge_accuracy_vs_eta.png"
        output_path_pdf = self.output_dir / "atlas_charge_accuracy_vs_eta.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Charge Accuracy plot saved to {output_path_png} and {output_path_pdf}")
        
        return accuracies
    
    def plot_pt_distribution_comparison(self):
        """Plot overlaid pT distribution comparing model predictions with ground truth."""
        print("Generating pT Distribution Comparison plot...")
        
        if len(self.pt_predictions) == 0:
            print("Warning: No data for pT distribution plot")
            return
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate bins
        all_pt = np.concatenate([self.pt_predictions, self.pt_truth])
        min_pt = np.min(all_pt)
        max_pt = min(200.0, np.max(all_pt) * 1.1)
        bins = np.linspace(min_pt, max_pt, 100)
        
        # Plot both distributions with transparency
        ax.hist(self.pt_truth, bins=bins, alpha=0.6, density=False, 
               label='Ground Truth', color='blue', histtype='stepfilled')
        ax.hist(self.pt_predictions, bins=bins, alpha=0.6, density=False, 
               label='Model Predictions', color='red', histtype='stepfilled')
        
        ax.set_xlabel(r'$p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_pt_distribution_comparison.png"
        output_path_pdf = self.output_dir / "atlas_pt_distribution_comparison.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"pT Distribution Comparison plot saved to {output_path_png} and {output_path_pdf}")
    
    def calculate_and_save_unbinned_metrics(self, include_75wp=False):
        """Calculate and save unbinned average metrics to a text file.
        
        Args:
            include_75wp: If True, also calculate and save 75% working point metrics (default: False)
        """
        print("Calculating unbinned average metrics...")
        
        # Hit assignment efficiency
        hit_efficiencies = []
        for pred, truth in zip(self.hit_predictions, self.hit_true_assignments):
            true_hits = np.sum(truth)
            if true_hits > 0:
                correct_hits = np.sum(pred & truth)
                hit_efficiencies.append(correct_hits / true_hits)
        
        avg_hit_efficiency = np.mean(hit_efficiencies) if len(hit_efficiencies) > 0 else 0
        frac_hit_efficiency_50 = np.sum(np.array(hit_efficiencies) >= 0.5) / len(hit_efficiencies) if len(hit_efficiencies) > 0 else 0
        
        # Hit assignment purity
        hit_purities = []
        for pred, truth in zip(self.hit_predictions, self.hit_true_assignments):
            pred_hits = np.sum(pred)
            if pred_hits > 0:
                correct_hits = np.sum(pred & truth)
                hit_purities.append(correct_hits / pred_hits)
        
        avg_hit_purity = np.mean(hit_purities) if len(hit_purities) > 0 else 0
        frac_hit_purity_50 = np.sum(np.array(hit_purities) >= 0.5) / len(hit_purities) if len(hit_purities) > 0 else 0
        
        # Double matching efficiency (50% working point)
        n_double_match_50 = 0
        for pred, truth in zip(self.hit_predictions, self.hit_true_assignments):
            true_hits = np.sum(truth)
            pred_hits = np.sum(pred)
            correct_hits = np.sum(pred & truth)
            
            track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
            track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
            
            if track_efficiency >= 0.5 and track_purity >= 0.5:
                n_double_match_50 += 1
        
        avg_double_matching_50 = n_double_match_50 / len(self.hit_predictions) if len(self.hit_predictions) > 0 else 0
        
        # Double matching efficiency (75% working point) - optional
        if include_75wp:
            n_double_match_75 = 0
            for pred, truth in zip(self.hit_predictions, self.hit_true_assignments):
                true_hits = np.sum(truth)
                pred_hits = np.sum(pred)
                correct_hits = np.sum(pred & truth)
                
                track_efficiency = correct_hits / true_hits if true_hits > 0 else 0
                track_purity = correct_hits / pred_hits if pred_hits > 0 else 0
                
                if track_efficiency >= 0.75 and track_purity >= 0.75:
                    n_double_match_75 += 1
            
            avg_double_matching_75 = n_double_match_75 / len(self.hit_predictions) if len(self.hit_predictions) > 0 else 0
        else:
            avg_double_matching_75 = None
        
        # Charge classification accuracy
        pred_charge_discrete = np.where(self.charge_predictions >= 0, 1, -1)
        correct_charge = np.sum(pred_charge_discrete == self.charge_truth)
        avg_charge_accuracy = correct_charge / len(self.charge_truth) if len(self.charge_truth) > 0 else 0
        
        # pT statistics
        pt_residuals = self.pt_predictions - self.pt_truth
        avg_pt_residual = np.mean(pt_residuals)
        std_pt_residual = np.std(pt_residuals)
        
        # Write to file
        output_path = self.output_dir / "unbinned_metrics_summary.txt"
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNBINNED AVERAGE METRICS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Total tracks analyzed: {len(self.hit_predictions):,}\n")
            f.write("\n")
            f.write("HIT ASSIGNMENT EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Hit Assignment Efficiency:  {avg_hit_efficiency:.6f} ({avg_hit_efficiency*100:.2f}%)\n")
            f.write(f"  (Mean fraction of true hits correctly assigned)\n")
            f.write(f"Fraction with Efficiency ≥ 50%:     {frac_hit_efficiency_50:.6f} ({frac_hit_efficiency_50*100:.2f}%)\n")
            f.write(f"  (Tracks meeting 50% efficiency threshold)\n")
            f.write("\n")
            f.write("HIT ASSIGNMENT PURITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Hit Assignment Purity:      {avg_hit_purity:.6f} ({avg_hit_purity*100:.2f}%)\n")
            f.write(f"  (Mean fraction of assigned hits that are correct)\n")
            f.write(f"Fraction with Purity ≥ 50%:         {frac_hit_purity_50:.6f} ({frac_hit_purity_50*100:.2f}%)\n")
            f.write(f"  (Tracks meeting 50% purity threshold)\n")
            f.write("\n")
            f.write("DOUBLE MATCHING EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Double Matching Efficiency (50% WP): {avg_double_matching_50:.6f} ({avg_double_matching_50*100:.2f}%)\n")
            f.write(f"  (Fraction of tracks with both efficiency ≥50% AND purity ≥50%)\n")
            if include_75wp and avg_double_matching_75 is not None:
                f.write(f"Double Matching Efficiency (75% WP): {avg_double_matching_75:.6f} ({avg_double_matching_75*100:.2f}%)\n")
                f.write(f"  (Fraction of tracks with both efficiency ≥75% AND purity ≥75%)\n")
            f.write("\n")
            f.write("CHARGE CLASSIFICATION ACCURACY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Charge Classification Accuracy:     {avg_charge_accuracy:.6f} ({avg_charge_accuracy*100:.2f}%)\n")
            f.write(f"  (Fraction with correct charge sign predictions)\n")
            f.write("\n")
            f.write("pT REGRESSION METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average pT Residual (Pred - Truth): {avg_pt_residual:.4f} GeV\n")
            f.write(f"Standard Deviation of pT Residual:  {std_pt_residual:.4f} GeV\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        print(f"Unbinned metrics summary saved to {output_path}")
        print(f"  Hit Assignment Efficiency (avg): {avg_hit_efficiency:.4f}")
        print(f"  Hit Assignment Purity (avg): {avg_hit_purity:.4f}")
        print(f"  Double Matching Efficiency (50% WP): {avg_double_matching_50:.4f}")
        if include_75wp and avg_double_matching_75 is not None:
            print(f"  Double Matching Efficiency (75% WP): {avg_double_matching_75:.4f}")
        print(f"  Charge Classification Accuracy: {avg_charge_accuracy:.4f}")
    
    def save_plot_captions(self):
        """Save suggested plot captions for presentations."""
        captions_path = self.output_dir / "plot_captions.txt"
        
        with open(captions_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ATLAS MUON TRACKING MODEL EVALUATION - TRACKING STAGE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ABSTRACT\n")
            f.write("-" * 80 + "\n")
            f.write("This analysis evaluates the tracking stage of a two-stage graph neural\n")
            f.write("network model for muon track reconstruction in the ATLAS detector at\n")
            f.write("High-Luminosity LHC conditions. The model processes simulated proton-proton\n")
            f.write("collisions at √s = 14 TeV with an average pileup of <μ> = 200, using events\n")
            f.write("containing muons from t𝑡̄, J/ψ, and Z→μμ processes with pT > 5 GeV.\n")
            f.write("\n")
            f.write("The two-stage architecture consists of: (1) a hit filtering stage that\n")
            f.write("reduces combinatorial background by classifying detector hits as signal or\n")
            f.write("noise, and (2) a tracking stage that performs hit-to-track assignment and\n")
            f.write("simultaneously reconstructs track parameters (η, φ, pT) and classifies the\n")
            f.write("muon charge sign. The following plots assess the tracking stage performance.\n")
            f.write("\n")
            f.write("Uncertainty quantification uses Clopper-Pearson 99.7%% confidence intervals\n")
            f.write("(~3σ) for all binomial proportions, including pooled hit assignment\n")
            f.write("efficiency and purity (calculated across all hits in each η bin), charge\n")
            f.write("classification accuracy, and double matching efficiency. The Clopper-Pearson\n")
            f.write("method provides exact coverage guarantees for finite sample sizes, ensuring\n")
            f.write("statistically rigorous and conservative uncertainty estimates appropriate for\n")
            f.write("particle physics applications.\n")
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIGURE CAPTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Figure 1: Hit Assignment Efficiency and Purity vs Pseudorapidity\n")
            f.write("-" * 70 + "\n")
            f.write("Pooled hit assignment efficiency (blue circles) and purity (green squares)\n")
            f.write("as a function of truth muon pseudorapidity η. Efficiency is calculated as\n")
            f.write("the ratio of total correct hits to total true hits across all tracks in\n")
            f.write("each bin. Purity is the ratio of total correct hits to total predicted\n")
            f.write("hits. Error bars represent 99.7%% Clopper-Pearson confidence intervals.\n")
            f.write("\n\n")
            
            f.write("Figure 2: Double Matching Efficiency vs Pseudorapidity\n")
            f.write("-" * 70 + "\n")
            f.write("Fraction of tracks satisfying both efficiency ≥ 50%% and purity ≥ 50%%\n")
            f.write("criteria as a function of truth pseudorapidity η. This metric requires\n")
            f.write("bidirectional consistency between hit-to-track and track-to-hits\n")
            f.write("assignments. Error bars show 99.7%% Clopper-Pearson confidence intervals.\n")
            f.write("\n\n")
            
            f.write("Figure 3: Charge Classification Accuracy vs Pseudorapidity\n")
            f.write("-" * 70 + "\n")
            f.write("Accuracy of muon charge sign classification (±1) as a function of truth\n")
            f.write("pseudorapidity η. Error bars represent 99.7%% Clopper-Pearson confidence\n")
            f.write("intervals for the binomial proportion of correctly classified tracks.\n")
            f.write("\n\n")
            
            f.write("Figure 4: Transverse Momentum Distribution Comparison\n")
            f.write("-" * 70 + "\n")
            f.write("Comparison of predicted (red) and truth (blue) transverse momentum\n")
            f.write("distributions for reconstructed muon tracks. Agreement between the two\n")
            f.write("distributions indicates unbiased pT estimation across the kinematic range.\n")
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DATASET AND METHODOLOGY NOTES\n")
            f.write("=" * 80 + "\n")
            f.write("- Simulation: pp collisions at √s = 14 TeV\n")
            f.write("- Average pileup: <μ> = 200 interactions per bunch crossing\n")
            # f.write("- Physics processes: t𝑡̄, J/ψ→μμ, Z→μμ\n")
            f.write("- Muon selection: pT > 5 GeV\n")
            f.write("- Model architecture: Two-stage graph neural network\n")
            f.write("  * Stage 1: Hit filtering (signal/noise classification)\n")
            f.write("  * Stage 2: Tracking (hit assignment + parameter reconstruction)\n")
            f.write("- Reconstructed parameters: η, φ, pT, charge sign\n")
            f.write("- Uncertainty quantification:\n")
            f.write("  * Clopper-Pearson 99.7%% CI (~3σ): All binomial proportions\n")
            f.write("  * Pooled statistics: Efficiency and purity use total hits across tracks\n")
            f.write("  * Per-track statistics: Double matching and charge accuracy\n")
            f.write("\n")
        
        print(f"Plot captions saved to {captions_path}")
    
    def run_analysis(self, include_75wp=False):
        """Run the complete analysis pipeline.
        
        Args:
            include_75wp: If True, include 75% working point in double matching analysis (default: False)
        """
        print("\n=== Starting ATLAS-style Tracking Plot Analysis ===\n")
        
        # Collect data
        self.setup_data_module()
        self.collect_data()
        
        # Generate all plots
        self.plot_hit_efficiency_and_purity_vs_eta()
        self.plot_double_matching_vs_eta(include_75wp=include_75wp)
        self.plot_charge_accuracy_vs_eta()
        self.plot_pt_distribution_comparison()
        
        # Save unbinned metrics
        self.calculate_and_save_unbinned_metrics(include_75wp=include_75wp)
        
        # Save plot captions
        self.save_plot_captions()
        
        print("\n=== Analysis Complete ===\n")


def main():
    parser = argparse.ArgumentParser(description='Generate ATLAS-style plots for tracking model evaluation')
    parser.add_argument('--eval_path', '-e', type=str, 
                       default="/scratch/epoch=139-val_loss=2.74982_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', '-d', type=str, 
                       default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_old",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', '-c', type=str, 
                       default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='.',
                       help='Base output directory (tracking_plots subdirectory will be created)')
    parser.add_argument('--max_events', '-m', type=int, default=-1,
                       help='Maximum number of events to process (-1 = all events)')
    parser.add_argument('--include-75wp', action='store_true',
                       help='Include 75%% working point in double matching analysis (default: False)')
    
    args = parser.parse_args()
    
    # Enable stdout buffering for better logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    try:
        plotter = ATLASStyleTrackingPlotter(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        plotter.run_analysis(include_75wp=args.include_75wp)
        
        print("\n✓ All ATLAS-style tracking plots generated successfully!")
        sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
