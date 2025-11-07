#!/usr/bin/env python3
"""
Independent script to generate ATLAS-style plots for hit filtering model evaluation.
This script creates:
1. ROC curve with AUC
2. Rejection rate vs. Purity plot
3. Hit Efficiency vs. Hit Purity plot

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
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule, AtlasMuonDataset
from hepattn.experiments.atlas_muon.data_vis.track_visualizer_h5_MDTGeometry import h5TrackVisualizerMDTGeometry


class ATLASStylePlotter:
    """Generate ATLAS-style plots for hit filtering evaluation."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = Path(eval_path)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / "CTD_plots" / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for collected data
        self.all_logits = None
        self.all_true_labels = None
        self.all_particle_ids = None
        self.all_event_ids = None
        self.all_particle_pts = None
        self.all_particle_etas = None
        self.all_particle_phis = None
        
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
        
        # Pre-allocate storage with estimated sizes
        estimated_hits = min(self.max_events * 7000 if self.max_events and self.max_events > 0 else 10000000, 50000000)
        
        # Use memory-efficient dtypes
        all_logits = np.zeros(estimated_hits, dtype=np.float32)
        all_true_labels = np.zeros(estimated_hits, dtype=bool)
        all_particle_ids = np.zeros(estimated_hits, dtype=np.int32)
        all_event_ids = np.zeros(estimated_hits, dtype=np.int32)
        all_particle_pts = np.zeros(estimated_hits, dtype=np.float32)
        all_particle_etas = np.zeros(estimated_hits, dtype=np.float32)
        all_particle_phis = np.zeros(estimated_hits, dtype=np.float32)
        
        current_idx = 0
        events_processed = 0
        events_attempted = 0
        
        try:
            with h5py.File(self.eval_path, 'r') as eval_file:
                for batch_idx, batch_data in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                    events_attempted += 1
                    
                    try:
                        # Unpack batch data
                        if len(batch_data) == 2:
                            inputs_batch, targets_batch = batch_data
                        else:
                            print(f"Warning: Unexpected batch data structure: {len(batch_data)} elements")
                            continue
                        
                        # Extract event index using sample_id
                        if "sample_id" not in targets_batch:
                            print(f"Warning: sample_id not found in targets, skipping batch {batch_idx}")
                            continue
                        
                        event_idx = targets_batch["sample_id"][0].item()
                        
                        # Load predictions for this event from evaluation file
                        if str(event_idx) not in eval_file:
                            print(f"Warning: event {event_idx} not found in evaluation file")
                            continue
                        
                        # Get hit logits from evaluation file (following the structure from main evaluation script)
                        hit_logits = eval_file[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)
                        
                        # Get truth labels from DataLoader
                        if "hit_on_valid_particle" not in targets_batch:
                            print(f"Warning: hit_on_valid_particle not found in targets, skipping batch {batch_idx}")
                            continue
                        
                        true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(bool)
                        
                        # Get particle IDs and pT values for particle reconstruction efficiency
                        hit_particle_ids = inputs_batch["plotting_spacePoint_truthLink"][0].numpy().astype(np.int32)
                        
                        # Get particle pt values
                       
                        particle_pts = targets_batch["particle_truthMuon_pt"][0].numpy().astype(np.float32)
                        
                        # Map hits to particle pt values
                        hit_pts = np.full(len(hit_logits), -1.0, dtype=np.float32)  # Default for noise hits
                        unique_particle_ids = np.unique(hit_particle_ids)
                        valid_particle_ids = unique_particle_ids[unique_particle_ids >= 0]  # Skip -1 (noise)
                        
                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_pts):  # Safety check
                                hit_mask = hit_particle_ids == particle_id
                                hit_pts[hit_mask] = particle_pts[idx]
                       
                        
                        # Get particle eta values
                        particle_etas = targets_batch["particle_truthMuon_eta"][0].numpy().astype(np.float32)
                        
                        # Map hits to particle eta values
                        hit_etas = np.full(len(hit_logits), -999.0, dtype=np.float32)  # Default for noise hits
                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_etas):
                                hit_mask = hit_particle_ids == particle_id
                                hit_etas[hit_mask] = particle_etas[idx]
                        
                        # Get particle phi values
                        particle_phis = targets_batch["particle_truthMuon_phi"][0].numpy().astype(np.float32)
                        
                        # Map hits to particle phi values
                        hit_phis = np.full(len(hit_logits), -999.0, dtype=np.float32)  # Default for noise hits
                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_phis):
                                hit_mask = hit_particle_ids == particle_id
                                hit_phis[hit_mask] = particle_phis[idx]

                        
                        # Verify shapes match
                        n_hits = len(hit_logits)
                        if n_hits != len(true_labels):
                            print(f"Warning: Shape mismatch in event {event_idx}: logits={n_hits}, labels={len(true_labels)}")
                            continue
                        
                        # Resize arrays if needed
                        if current_idx + n_hits > len(all_logits):
                            new_size = max(len(all_logits) * 2, current_idx + n_hits)
                            all_logits = np.resize(all_logits, new_size)
                            all_true_labels = np.resize(all_true_labels, new_size)
                            all_particle_ids = np.resize(all_particle_ids, new_size)
                            all_event_ids = np.resize(all_event_ids, new_size)
                            all_particle_pts = np.resize(all_particle_pts, new_size)
                            all_particle_etas = np.resize(all_particle_etas, new_size)
                            all_particle_phis = np.resize(all_particle_phis, new_size)
                        
                        # Store data
                        all_logits[current_idx:current_idx + n_hits] = hit_logits
                        all_true_labels[current_idx:current_idx + n_hits] = true_labels
                        all_particle_ids[current_idx:current_idx + n_hits] = hit_particle_ids
                        all_event_ids[current_idx:current_idx + n_hits] = event_idx
                        all_particle_pts[current_idx:current_idx + n_hits] = hit_pts
                        all_particle_etas[current_idx:current_idx + n_hits] = hit_etas
                        all_particle_phis[current_idx:current_idx + n_hits] = hit_phis
                        
                        current_idx += n_hits
                        events_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\nDataLoader provided {events_attempted} batches, successfully processed {events_processed} events")
        
        if events_processed == 0:
            raise ValueError("No events were successfully processed!")
        
        # Trim arrays to actual size
        self.all_logits = all_logits[:current_idx]
        self.all_true_labels = all_true_labels[:current_idx]
        self.all_particle_ids = all_particle_ids[:current_idx]
        self.all_event_ids = all_event_ids[:current_idx]
        self.all_particle_pts = all_particle_pts[:current_idx]
        self.all_particle_etas = all_particle_etas[:current_idx]
        self.all_particle_phis = all_particle_phis[:current_idx]
        
        print(f"Data collection complete!")
        print(f"Events processed: {events_processed}")
        print(f"Total hits: {len(self.all_logits):,}")
        print(f"True hits: {np.sum(self.all_true_labels):,}")
        print(f"Noise hits: {np.sum(~self.all_true_labels):,}")
        
        return True
    
    def plot_atlas_roc_curve(self):
        """Generate ATLAS-style ROC curve with AUC score."""
        print("Generating ATLAS-style ROC curve...")
        
        # Safety checks
        if len(self.all_logits) == 0:
            raise ValueError("No data available for ROC curve!")
        
        n_true = np.sum(self.all_true_labels)
        n_false = np.sum(~self.all_true_labels)
        
        if n_true == 0 or n_false == 0:
            raise ValueError("Need both true and false labels for ROC curve!")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        roc_auc = auc(fpr, tpr)
        
        print(f"Data statistics: {len(self.all_logits)} hits, {n_true} true, {n_false} false")
        print(f"AUC Score: {roc_auc:.5f}")
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='royalblue', lw=0.8, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=0.8, linestyle='--', 
                label='Random classifier')
        
        # Set axis limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate (Efficiency)', fontsize=14)
        
        # Add legend
        ax.legend(loc="lower right", fontsize=12)
        
        # Add grid
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
        output_path_png = self.output_dir / "atlas_roc_curve.png"
        output_path_pdf = self.output_dir / "atlas_roc_curve.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"ATLAS-style ROC curve saved to {output_path_png} and {output_path_pdf}")
        
        return roc_auc
    
    def plot_atlas_rejection_vs_purity(self):
        """Generate ATLAS-style Rejection rate vs. Purity plot."""
        print("Generating ATLAS-style Rejection vs. Purity plot...")
        
        # Safety checks
        if len(self.all_logits) == 0:
            raise ValueError("No data available for rejection vs purity plot!")
        
        n_true = np.sum(self.all_true_labels)
        n_false = np.sum(~self.all_true_labels)
        
        if n_true == 0 or n_false == 0:
            raise ValueError("Need both true and false labels for rejection vs purity plot!")
        
        # Calculate ROC curve to get thresholds
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Define working points (efficiency points) to sweep
        working_points = np.arange(0.95, 0.998, 0.0001)
        
        purities = []
        rejections = []
        valid_wps = []
        
        print(f"Sweeping {len(working_points)} working points from {working_points[0]:.3f} to {working_points[-1]:.3f}...")
        
        for wp in tqdm(working_points, desc="Calculating rejection and purity"):
            # Find threshold that gives this efficiency (TPR)
            if not np.any(tpr >= wp):
                continue
            
            # Get threshold for this working point
            threshold = thresholds[tpr >= wp][0]
            
            # Apply threshold to get predictions
            predictions = self.all_logits >= threshold
            
            # Calculate purity (precision): TP / (TP + FP) = TP / total_predicted_positive
            true_positives = np.sum(self.all_true_labels & predictions)
            total_predicted_positives = np.sum(predictions)
            
            if total_predicted_positives > 0:
                purity = true_positives / total_predicted_positives
            else:
                continue
            
            # Calculate rejection rate: TN / (TN + FP) = TN / total_negatives
            true_negatives = np.sum(~self.all_true_labels & ~predictions)
            total_negatives = np.sum(~self.all_true_labels)
            
            if total_negatives > 0:
                rejection = true_negatives / total_negatives
            else:
                continue
            
            purities.append(purity)
            rejections.append(rejection)
            valid_wps.append(wp)
        
        purities = np.array(purities)
        rejections = np.array(rejections)
        valid_wps = np.array(valid_wps)
        
        print(f"Generated {len(valid_wps)} valid points")
        print(f"Purity range: [{np.min(purities):.4f}, {np.max(purities):.4f}]")
        print(f"Rejection range: [{np.min(rejections):.4f}, {np.max(rejections):.4f}]")
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot rejection vs purity with color gradient by working point
        scatter = ax.scatter(purities, rejections, c=valid_wps, cmap='viridis', 
                           s=20, alpha=0.7, edgecolors='none')
        
        # Add colorbar for working points
        cbar = plt.colorbar(scatter, ax=ax, label='Efficiency (Working Point)')
        cbar.ax.tick_params(labelsize=12)
        
        # Add marker for 0.99 working point (the one used in analysis)
        # Find the point closest to 0.99
        wp_target = 0.99
        idx_099 = np.argmin(np.abs(valid_wps - wp_target))
        purity_099 = purities[idx_099]
        rejection_099 = rejections[idx_099]
        actual_wp_099 = valid_wps[idx_099]
        
        # Plot a cross marker at the 0.99 working point
        ax.scatter(purity_099, rejection_099, marker='x', s=200, linewidth=0.8, 
                  color='red', zorder=5, label=f'Working Point {actual_wp_099:.4f}')
        
        print(f"0.99 working point: Purity={purity_099:.4f}, Rejection={rejection_099:.4f}")
        
       
        ax.set_xlabel('Hit Purity (Precision)', fontsize=14)
        ax.set_ylabel('Background Rejection Rate', fontsize=14)
        
        # Add legend for the 0.99 marker
        ax.legend(loc='lower right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )

         # Set axis limits and labels
        ax.set_xlim([np.min(purities) - 0.05, np.max(purities) + 0.05])
        ax.set_ylim([np.min(rejections) - 0.005, np.max(rejections) + 0.005])
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_rejection_vs_purity.png"
        output_path_pdf = self.output_dir / "atlas_rejection_vs_purity.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"ATLAS-style Rejection vs. Purity plot saved to {output_path_png} and {output_path_pdf}")
        
        # Save data to CSV
        csv_path = self.output_dir / "rejection_purity_data.csv"
        import pandas as pd
        df = pd.DataFrame({
            'working_point': valid_wps,
            'purity': purities,
            'rejection': rejections
        })
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        return purities, rejections, valid_wps
    
    def plot_atlas_efficiency_vs_purity(self):
        """Generate ATLAS-style Hit Efficiency vs. Hit Purity plot."""
        print("Generating ATLAS-style Hit Efficiency vs. Hit Purity plot...")
        
        # Safety checks
        if len(self.all_logits) == 0:
            raise ValueError("No data available for efficiency vs purity plot!")
        
        n_true = np.sum(self.all_true_labels)
        n_false = np.sum(~self.all_true_labels)
        
        if n_true == 0 or n_false == 0:
            raise ValueError("Need both true and false labels for efficiency vs purity plot!")
        
        # Calculate ROC curve to get thresholds
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Define working points (efficiency points) to sweep - same as rejection vs purity
        working_points = np.arange(0.95, 0.998, 0.0001)
        
        purities = []
        efficiencies = []
        valid_wps = []
        
        print(f"Sweeping {len(working_points)} working points from {working_points[0]:.3f} to {working_points[-1]:.3f}...")
        
        for wp in tqdm(working_points, desc="Calculating efficiency and purity"):
            # Find threshold that gives this efficiency (TPR)
            if not np.any(tpr >= wp):
                continue
            
            # Get threshold for this working point
            threshold = thresholds[tpr >= wp][0]
            
            # Apply threshold to get predictions
            predictions = self.all_logits >= threshold
            
            # Calculate purity (precision): TP / (TP + FP) = TP / total_predicted_positive
            true_positives = np.sum(self.all_true_labels & predictions)
            total_predicted_positives = np.sum(predictions)
            
            if total_predicted_positives > 0:
                purity = true_positives / total_predicted_positives
            else:
                continue
            
            # Calculate efficiency (recall/TPR): TP / (TP + FN) = TP / total_true_positives
            total_true_positives = np.sum(self.all_true_labels)
            
            if total_true_positives > 0:
                efficiency = true_positives / total_true_positives
            else:
                continue
            
            purities.append(purity)
            efficiencies.append(efficiency)
            valid_wps.append(wp)
        
        purities = np.array(purities)
        efficiencies = np.array(efficiencies)
        valid_wps = np.array(valid_wps)
        
        print(f"Generated {len(valid_wps)} valid points")
        print(f"Purity range: [{np.min(purities):.4f}, {np.max(purities):.4f}]")
        print(f"Efficiency range: [{np.min(efficiencies):.4f}, {np.max(efficiencies):.4f}]")
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot purity vs efficiency as a line plot
        # Sort by efficiency (now on x-axis) for proper line plotting
        sort_idx = np.argsort(efficiencies)
        sorted_eff = efficiencies[sort_idx]
        sorted_pur = purities[sort_idx]
        sorted_wps = valid_wps[sort_idx]
        
        # Plot as a simple line (efficiency on x-axis, purity on y-axis)
        ax.plot(sorted_eff, sorted_pur, 'royalblue', linewidth=0.8, zorder=2)
        
        # Add marker for 0.99 working point (the one used in analysis)
        # Find the point closest to 0.99
        wp_target = 0.99
        idx_099 = np.argmin(np.abs(valid_wps - wp_target))
        purity_099 = purities[idx_099]
        efficiency_099 = efficiencies[idx_099]
        actual_wp_099 = valid_wps[idx_099]
        
        # Plot a cross marker at the 0.99 working point
        ax.scatter(efficiency_099, purity_099, marker='x', s=200, linewidth=0.8, 
                  color='red', zorder=5, label=f'Working Point {actual_wp_099:.4f}')
        
        print(f"0.99 working point: Efficiency={efficiency_099:.4f}, Purity={purity_099:.4f}")
        
        # Set axis labels (limits will be set after atlasify)
        ax.set_xlabel('Hit Efficiency (Recall)', fontsize=14)
        ax.set_ylabel('Hit Purity (Precision)', fontsize=14)
        
        # Add legend for the 0.99 marker
        ax.legend(loc='lower right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set axis limits AFTER atlasify to prevent override
        ax.set_xlim([np.min(efficiencies) - 0.005, np.max(efficiencies) + 0.005])
        ax.set_ylim([np.min(purities) - 0.1, np.max(purities) + 0.3])
        # ax.set_xlim([np.min(efficiencies) - 0.005, np.max(efficiencies) + 0.005])
        # ax.set_ylim([-0.1, 1.1])
        
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_efficiency_vs_purity.png"
        output_path_pdf = self.output_dir / "atlas_efficiency_vs_purity.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"ATLAS-style Hit Efficiency vs. Purity plot saved to {output_path_png} and {output_path_pdf}")
        
        # Save data to CSV
        csv_path = self.output_dir / "efficiency_purity_data.csv"
        import pandas as pd
        df = pd.DataFrame({
            'working_point': valid_wps,
            'purity': purities,
            'efficiency': efficiencies
        })
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        return purities, efficiencies, valid_wps
    
    def plot_atlas_background_efficiency_inverse(self):
        """Generate ATLAS-style 1/Background Efficiency vs. Signal Efficiency plot."""
        print("Generating ATLAS-style 1/Background Efficiency vs. Signal Efficiency plot...")
        
        # Safety checks
        if len(self.all_logits) == 0:
            raise ValueError("No data available for background efficiency plot!")
        
        n_true = np.sum(self.all_true_labels)
        n_false = np.sum(~self.all_true_labels)
        
        if n_true == 0 or n_false == 0:
            raise ValueError("Need both true and false labels for background efficiency plot!")
        
        # Calculate ROC curve to get thresholds
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Define working points (efficiency points) to sweep
        working_points = np.arange(0.95, 0.998, 0.0001)
        
        signal_efficiencies = []
        inverse_bkg_efficiencies = []
        valid_wps = []
        
        print(f"Sweeping {len(working_points)} working points from {working_points[0]:.3f} to {working_points[-1]:.3f}...")
        
        for wp in tqdm(working_points, desc="Calculating signal and background efficiencies"):
            # Find threshold that gives this efficiency (TPR)
            if not np.any(tpr >= wp):
                continue
            
            # Get threshold for this working point
            threshold = thresholds[tpr >= wp][0]
            
            # Apply threshold to get predictions
            predictions = self.all_logits >= threshold
            
            # Calculate signal efficiency (TPR): TP / (TP + FN)
            true_positives = np.sum(self.all_true_labels & predictions)
            total_true_positives = np.sum(self.all_true_labels)
            
            if total_true_positives > 0:
                signal_eff = true_positives / total_true_positives
            else:
                continue
            
            # Calculate background efficiency: FP / (FP + TN) = FP / total_negatives
            false_positives = np.sum(~self.all_true_labels & predictions)
            total_negatives = np.sum(~self.all_true_labels)
            
            if total_negatives > 0:
                bkg_eff = false_positives / total_negatives
            else:
                continue
            
            # Calculate inverse background efficiency (1/eff_bkg)
            if bkg_eff > 0:
                inverse_bkg_eff = 1.0 / bkg_eff
            else:
                continue  # Skip points with zero background efficiency
            
            signal_efficiencies.append(signal_eff)
            inverse_bkg_efficiencies.append(inverse_bkg_eff)
            valid_wps.append(wp)
        
        signal_efficiencies = np.array(signal_efficiencies)
        inverse_bkg_efficiencies = np.array(inverse_bkg_efficiencies)
        valid_wps = np.array(valid_wps)
        
        print(f"Generated {len(valid_wps)} valid points")
        print(f"Signal Efficiency range: [{np.min(signal_efficiencies):.4f}, {np.max(signal_efficiencies):.4f}]")
        print(f"1/Background Efficiency range: [{np.min(inverse_bkg_efficiencies):.2f}, {np.max(inverse_bkg_efficiencies):.2f}]")
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot 1/bkg_eff vs signal_eff as a line
        ax.plot(signal_efficiencies, inverse_bkg_efficiencies, 'royalblue', linewidth=0.8, zorder=2)
        
        # Add marker for 0.99 working point
        wp_target = 0.99
        idx_099 = np.argmin(np.abs(valid_wps - wp_target))
        signal_eff_099 = signal_efficiencies[idx_099]
        inverse_bkg_eff_099 = inverse_bkg_efficiencies[idx_099]
        actual_wp_099 = valid_wps[idx_099]
        
        # Plot a cross marker at the 0.99 working point
        ax.scatter(signal_eff_099, inverse_bkg_eff_099, marker='x', s=200, linewidth=0.8, 
                  color='red', zorder=5, label=f'Working Point {actual_wp_099:.4f}')
        
        print(f"0.99 working point: Signal Eff={signal_eff_099:.4f}, 1/Bkg Eff={inverse_bkg_eff_099:.2f}")
        
        # Set axis labels (limits will be set after atlasify)
        ax.set_xlabel('Hit Efficiency (Recall)', fontsize=14)
        ax.set_ylabel(r'1 / Background Efficiency', fontsize=14)
        
        # Add legend for the 0.99 marker
        ax.legend(loc='upper right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set axis limits AFTER atlasify to prevent override
        ax.set_xlim([np.min(signal_efficiencies) - 0.005, np.max(signal_efficiencies) + 0.005])
        # Use log scale for y-axis to better show the range
        ax.set_yscale('log')
        ax.set_ylim([np.min(inverse_bkg_efficiencies) * 0.8, np.max(inverse_bkg_efficiencies) * 1.2])
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_inverse_background_efficiency.png"
        output_path_pdf = self.output_dir / "atlas_inverse_background_efficiency.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"ATLAS-style 1/Background Efficiency plot saved to {output_path_png} and {output_path_pdf}")
        
        # Save data to CSV
        csv_path = self.output_dir / "inverse_background_efficiency_data.csv"
        import pandas as pd
        df = pd.DataFrame({
            'working_point': valid_wps,
            'signal_efficiency': signal_efficiencies,
            'inverse_background_efficiency': inverse_bkg_efficiencies
        })
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        return signal_efficiencies, inverse_bkg_efficiencies, valid_wps
    
    def plot_particle_reconstruction_efficiency(self):
        """
        Generate ATLAS-style particle reconstruction efficiency plot.
        A particle is considered reconstructable after filtering if it retains at least 3 true hits.
        Uses Clopper-Pearson confidence intervals for uncertainty.
        """
        print("Generating ATLAS-style Particle Reconstruction Efficiency plot...")
        
        # Safety checks
        if len(self.all_logits) == 0:
            raise ValueError("No data available for particle reconstruction efficiency plot!")
        
        if self.all_particle_ids is None or self.all_particle_pts is None:
            raise ValueError("Particle ID and pT data not available!")
        
        # Calculate ROC curve to get threshold for 0.99 efficiency
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Find threshold for 0.99 efficiency
        wp_target = 0.99
        if not np.any(tpr >= wp_target):
            print(f"Warning: Cannot achieve efficiency {wp_target}, using maximum achievable")
            threshold = thresholds[-1]
        else:
            threshold = thresholds[tpr >= wp_target][0]
        
        # Apply threshold to get predictions
        predictions = self.all_logits >= threshold
        
        # Get unique particles (only true hits from real particles, not noise)
        true_hit_mask = self.all_true_labels
        valid_particle_mask = true_hit_mask & (self.all_particle_pts > 0)
        
        # Get unique (event_id, particle_id) combinations
        unique_particles = np.unique(
            np.column_stack([
                self.all_event_ids[valid_particle_mask],
                self.all_particle_ids[valid_particle_mask]
            ]), axis=0
        )
        
        print(f"Found {len(unique_particles)} unique particles")
        
        # For each particle, count:
        # 1. Total true hits
        # 2. True hits that pass the filter (true AND predicted as true)
        particle_data = []
        
        for event_id, particle_id in tqdm(unique_particles, desc="Processing particles"):
            # Get mask for this particle's true hits
            particle_mask = (
                (self.all_event_ids == event_id) &
                (self.all_particle_ids == particle_id) &
                true_hit_mask
            )
            
            # Count total true hits for this particle
            n_true_hits = np.sum(particle_mask)
            
            # Count true hits that pass the filter
            n_kept_hits = np.sum(particle_mask & predictions)
            
            # Get particle pT (from any hit of this particle)
            particle_pt = self.all_particle_pts[particle_mask][0]
            
            # Particle is reconstructable if it has at least 3 kept hits
            is_reconstructable = n_kept_hits >= 3
            
            particle_data.append({
                'pt': particle_pt,
                'n_true_hits': n_true_hits,
                'n_kept_hits': n_kept_hits,
                'is_reconstructable': is_reconstructable
            })
        
        # Convert to arrays
        particle_pts = np.array([p['pt'] for p in particle_data])
        is_reconstructable = np.array([p['is_reconstructable'] for p in particle_data])
        
        print(f"Total particles: {len(particle_pts)}")
        print(f"Reconstructable particles (>=3 hits): {np.sum(is_reconstructable)}")
        print(f"Reconstruction efficiency: {np.sum(is_reconstructable)/len(is_reconstructable)*100:.2f}%")
        
        # Define pT bins (0 to 200 GeV)
        pt_bins = np.linspace(0, 200, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate efficiency in each bin with Clopper-Pearson confidence intervals
        efficiencies = []
        err_low = []
        err_high = []
        n_particles_per_bin = []
        
        for i in range(len(pt_bins) - 1):
            pt_mask = (particle_pts >= pt_bins[i]) & (particle_pts < pt_bins[i+1])
            n_total = np.sum(pt_mask)
            n_reconstructable = np.sum(is_reconstructable[pt_mask])
            
            n_particles_per_bin.append(n_total)
            
            if n_total > 0:
                efficiency = n_reconstructable / n_total
                
                # Clopper-Pearson confidence interval (95% confidence level)
                # Using beta distribution quantiles
                alpha = 0.05  # 95% confidence
                
                if n_reconstructable == 0:
                    ci_low = 0.0
                else:
                    ci_low = stats.beta.ppf(alpha/2, n_reconstructable, n_total - n_reconstructable + 1)
                
                if n_reconstructable == n_total:
                    ci_high = 1.0
                else:
                    ci_high = stats.beta.ppf(1 - alpha/2, n_reconstructable + 1, n_total - n_reconstructable)
                
                efficiencies.append(efficiency)
                err_low.append(efficiency - ci_low)
                err_high.append(ci_high - efficiency)
            else:
                efficiencies.append(0.0)
                err_low.append(0.0)
                err_high.append(0.0)
        
        efficiencies = np.array(efficiencies)
        err_low = np.array(err_low)
        err_high = np.array(err_high)
        n_particles_per_bin = np.array(n_particles_per_bin)
        
        # Create ATLAS-style plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot efficiency with error bars
        ax.errorbar(pt_centers, efficiencies, yerr=[err_low, err_high],
                   fmt='o', color='royalblue', markersize=6, capsize=4,
                   label='Particle Reconstruction Efficiency', zorder=3)
        
        # Also plot as step function for clarity
        for i in range(len(pt_bins) - 1):
            if n_particles_per_bin[i] > 0:
                ax.hlines(efficiencies[i], pt_bins[i], pt_bins[i+1], 
                         colors='royalblue', linewidth=2, alpha=0.3, zorder=2)
        
        # Set axis limits and labels
        ax.set_xlim([0, 200])
        ax.set_ylim([0.95, 1.02])
        ax.set_xlabel('Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Particle Reconstruction Efficiency', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV" + f"\nWorking Point {wp_target:.2f}",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save plot in both PNG and PDF formats
        output_path_png = self.output_dir / "atlas_particle_reconstruction_efficiency.png"
        output_path_pdf = self.output_dir / "atlas_particle_reconstruction_efficiency.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"ATLAS-style Particle Reconstruction Efficiency plot saved to {output_path_png} and {output_path_pdf}")
        
        # Save data to CSV
        csv_path = self.output_dir / "particle_reconstruction_efficiency_data.csv"
        import pandas as pd
        df = pd.DataFrame({
            'pt_bin_center': pt_centers,
            'pt_bin_low': pt_bins[:-1],
            'pt_bin_high': pt_bins[1:],
            'efficiency': efficiencies,
            'error_low': err_low,
            'error_high': err_high,
            'n_particles': n_particles_per_bin
        })
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        return efficiencies, err_low, err_high, pt_centers
    
    def plot_truth_pt_distribution(self):
        """Generate ATLAS-style truth pT distribution plot with logarithmic scale."""
        print("Generating truth pT distribution plot...")
        
        # Use already collected data
        if self.all_particle_pts is None:
            print("Warning: No pT data collected")
            return None
        
        # Filter out noise hits (pt < 0) and get unique particle pT values
        valid_mask = (self.all_particle_pts > 0) & (self.all_true_labels)
        pt_values = self.all_particle_pts[valid_mask]
        
        # Get unique particles (event_id, particle_id) to avoid counting same particle multiple times
        event_particle_pairs = np.column_stack([
            self.all_event_ids[valid_mask],
            self.all_particle_ids[valid_mask]
        ])
        unique_pairs, unique_indices = np.unique(event_particle_pairs, axis=0, return_index=True)
        pt_values = pt_values[unique_indices]
        
        print(f"Collected {len(pt_values)} unique particle pT values")
        
        # Create histogram with 100 bins
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(0, 200, 101)  # 100 bins from 0 to 200 GeV
        counts, bin_edges, patches = ax.hist(pt_values, bins=bins, histtype='stepfilled', 
                                            color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Move legend higher to avoid the spike around 10 GeV
        # Calculate y-position: max count * 10 (one order of magnitude higher)
        max_count = np.max(counts)
        legend_y = max_count * 10
        
        # Set axis labels
        ax.set_xlabel(r'Truth Muon $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Adjust y-limits to give legend breathing room
        current_ylim = ax.get_ylim()
        ax.set_ylim([current_ylim[0], max(current_ylim[1], legend_y * 2)])
        
        # Save plot
        output_path_png = self.output_dir / "atlas_truth_pt_distribution.png"
        output_path_pdf = self.output_dir / "atlas_truth_pt_distribution.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Truth pT distribution plot saved to {output_path_png} and {output_path_pdf}")
        
        return pt_values
    
    def plot_truth_eta_distribution(self):
        """Generate ATLAS-style truth eta distribution plot with logarithmic scale."""
        print("Generating truth eta distribution plot...")
        
        # Use already collected data
        if self.all_particle_etas is None:
            print("Warning: No eta data collected")
            return None
        
        # Filter out noise hits (eta == -999) and get unique particle eta values
        valid_mask = (self.all_particle_etas != -999.0) & (self.all_true_labels)
        eta_values = self.all_particle_etas[valid_mask]
        
        # Get unique particles (event_id, particle_id) to avoid counting same particle multiple times
        event_particle_pairs = np.column_stack([
            self.all_event_ids[valid_mask],
            self.all_particle_ids[valid_mask]
        ])
        unique_pairs, unique_indices = np.unique(event_particle_pairs, axis=0, return_index=True)
        eta_values = eta_values[unique_indices]
        
        print(f"Collected {len(eta_values)} unique particle eta values")
        
        # Create histogram with 100 bins
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(-2.7, 2.7, 101)  # 100 bins from -2.7 to 2.7
        counts, bin_edges, patches = ax.hist(eta_values, bins=bins, histtype='stepfilled', 
                                            color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set axis labels
        ax.set_xlabel(r'Truth Muon $\eta$', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Set x-axis limits to [-2.7, 2.7]
        ax.set_xlim([-2.7, 2.7])
        
        # Save plot
        output_path_png = self.output_dir / "atlas_truth_eta_distribution.png"
        output_path_pdf = self.output_dir / "atlas_truth_eta_distribution.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Truth eta distribution plot saved to {output_path_png} and {output_path_pdf}")
        
        return eta_values
    
    def plot_truth_phi_distribution(self):
        """Generate ATLAS-style truth phi distribution plot with logarithmic scale."""
        print("Generating truth phi distribution plot...")
        
        # Use already collected data
        if self.all_particle_phis is None:
            print("Warning: No phi data collected")
            return None
        
        # Filter out noise hits (phi == -999) and get unique particle phi values
        valid_mask = (self.all_particle_phis != -999.0) & (self.all_true_labels)
        phi_values = self.all_particle_phis[valid_mask]
        
        # Get unique particles (event_id, particle_id) to avoid counting same particle multiple times
        event_particle_pairs = np.column_stack([
            self.all_event_ids[valid_mask],
            self.all_particle_ids[valid_mask]
        ])
        unique_pairs, unique_indices = np.unique(event_particle_pairs, axis=0, return_index=True)
        phi_values = phi_values[unique_indices]
        
        print(f"Collected {len(phi_values)} unique particle phi values")
        
        # Create histogram with 100 bins
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(-np.pi, np.pi, 101)  # 100 bins from -pi to pi
        counts, bin_edges, patches = ax.hist(phi_values, bins=bins, histtype='stepfilled', 
                                            color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set axis labels
        ax.set_xlabel(r'Truth Muon $\phi$', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save plot
        output_path_png = self.output_dir / "atlas_truth_phi_distribution.png"
        output_path_pdf = self.output_dir / "atlas_truth_phi_distribution.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Truth phi distribution plot saved to {output_path_png} and {output_path_pdf}")
        
        return phi_values
    
    def plot_logit_distribution(self):
        """Generate ATLAS-style logit distribution plots comparing signal and background hits.
        Creates both linear and logarithmic versions."""
        print("Generating logit distribution plots...")
        
        # Use already collected data
        if self.all_logits is None or self.all_true_labels is None:
            print("ERROR: No logit data available")
            return
        
        # Separate logits for signal hits and background hits
        signal_hit_logits = self.all_logits[self.all_true_labels]
        background_hit_logits = self.all_logits[~self.all_true_labels]
        
        print(f"Signal hit logits: {len(signal_hit_logits):,}")
        print(f"Background hit logits: {len(background_hit_logits):,}")
        
        # Define x-axis range and bins
        x_min, x_max = -100, 100
        bins = np.linspace(x_min, x_max, 201)  # 200 bins
        
        # ===== LINEAR SCALE PLOT =====
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both distributions with transparency (same style as pT comparison)
        ax.hist(background_hit_logits, bins=bins, alpha=0.6, density=False, 
               label='Background Hits', color='blue', histtype='stepfilled', range=(x_min, x_max))
        ax.hist(signal_hit_logits, bins=bins, alpha=0.6, density=False, 
               label='Signal Hits', color='red', histtype='stepfilled', range=(x_min, x_max))
        
        # Set axis labels and limits
        ax.set_xlabel(r'Model Output (Logit)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim([x_min, x_max])
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save linear plot
        output_path_png = self.output_dir / "atlas_logit_distribution.png"
        output_path_pdf = self.output_dir / "atlas_logit_distribution.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Linear logit distribution plot saved to {output_path_png} and {output_path_pdf}")
        
        # ===== LOGARITHMIC SCALE PLOT =====
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both distributions with transparency (same style as pT comparison)
        ax.hist(background_hit_logits, bins=bins, alpha=0.6, density=False, 
               label='Background Hits', color='blue', histtype='stepfilled', range=(x_min, x_max))
        ax.hist(signal_hit_logits, bins=bins, alpha=0.6, density=False, 
               label='Signal Hits', color='red', histtype='stepfilled', range=(x_min, x_max))
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set axis labels and limits
        ax.set_xlabel(r'Model Output (Logit)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim([x_min, x_max])
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Add grid (for both major and minor ticks on log scale)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save logarithmic plot
        output_path_png_log = self.output_dir / "atlas_logit_distribution_log.png"
        output_path_pdf_log = self.output_dir / "atlas_logit_distribution_log.pdf"
        plt.savefig(output_path_png_log, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf_log, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Logarithmic logit distribution plot saved to {output_path_png_log} and {output_path_pdf_log}")
        
        # ===== SIGMOID-TRANSFORMED LINEAR SCALE PLOT =====
        # Apply sigmoid transformation: sigmoid(x) = 1 / (1 + exp(-x))
        signal_hit_probs = 1.0 / (1.0 + np.exp(-signal_hit_logits))
        background_hit_probs = 1.0 / (1.0 + np.exp(-background_hit_logits))
        
        # Define bins for probabilities (0 to 1)
        prob_bins = np.linspace(0, 1, 201)  # 200 bins
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both distributions with transparency
        ax.hist(background_hit_probs, bins=prob_bins, alpha=0.6, density=False, 
               label='Background Hits', color='blue', histtype='stepfilled', range=(0, 1))
        ax.hist(signal_hit_probs, bins=prob_bins, alpha=0.6, density=False, 
               label='Signal Hits', color='red', histtype='stepfilled', range=(0, 1))
        
        # Set axis labels and limits
        ax.set_xlabel(r'Model Output (Sigmoid Probability)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim([0, 1])
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save sigmoid linear plot
        output_path_png_sigmoid = self.output_dir / "atlas_logit_distribution_sigmoid.png"
        output_path_pdf_sigmoid = self.output_dir / "atlas_logit_distribution_sigmoid.pdf"
        plt.savefig(output_path_png_sigmoid, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf_sigmoid, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Sigmoid logit distribution plot saved to {output_path_png_sigmoid} and {output_path_pdf_sigmoid}")
        
        # ===== SIGMOID-TRANSFORMED LOGARITHMIC SCALE PLOT =====
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both distributions with transparency
        ax.hist(background_hit_probs, bins=prob_bins, alpha=0.6, density=False, 
               label='Background Hits', color='blue', histtype='stepfilled', range=(0, 1))
        ax.hist(signal_hit_probs, bins=prob_bins, alpha=0.6, density=False, 
               label='Signal Hits', color='red', histtype='stepfilled', range=(0, 1))
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set axis labels and limits
        ax.set_xlabel(r'Model Output (Sigmoid Probability)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim([0, 1])
        
        # Add legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Add grid (for both major and minor ticks on log scale)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        
        # Apply ATLAS style
        atlasify.atlasify(
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + r"$\langle\mu\rangle$ = 200" + "\n" + r"Muon $p_T \geq 5$ GeV",
            font_size=14,
            sub_font_size=11,
            label_font_size=14
        )
        
        # Save sigmoid logarithmic plot
        output_path_png_sigmoid_log = self.output_dir / "atlas_logit_distribution_sigmoid_log.png"
        output_path_pdf_sigmoid_log = self.output_dir / "atlas_logit_distribution_sigmoid_log.pdf"
        plt.savefig(output_path_png_sigmoid_log, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf_sigmoid_log, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Sigmoid logarithmic logit distribution plot saved to {output_path_png_sigmoid_log} and {output_path_pdf_sigmoid_log}")
        
        # Print summary statistics
        print(f"Signal hit logits - Mean: {np.mean(signal_hit_logits):.4f}, Std: {np.std(signal_hit_logits):.4f}")
        print(f"Background hit logits - Mean: {np.mean(background_hit_logits):.4f}, Std: {np.std(background_hit_logits):.4f}")
        
        return signal_hit_logits, background_hit_logits
    
    def save_working_point_metrics(self):
        """
        Save key metrics for working points 0.99 and 0.995 to a text file.
        Includes rejection rate, purity, and overall particle reconstruction efficiency.
        """
        print("Saving working point metrics...")
        
        # Calculate ROC curve to get thresholds
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Working points to analyze
        working_points = [0.99, 0.995]
        
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("WORKING POINT METRICS SUMMARY")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        for wp_target in working_points:
            output_lines.append(f"\n{'='*80}")
            output_lines.append(f"Working Point: {wp_target:.3f}")
            output_lines.append(f"{'='*80}")
            
            # Find threshold for this working point
            if not np.any(tpr >= wp_target):
                output_lines.append(f"Warning: Cannot achieve efficiency {wp_target}")
                continue
            
            threshold = thresholds[tpr >= wp_target][0]
            actual_efficiency = tpr[tpr >= wp_target][0]
            
            # Apply threshold to get predictions
            predictions = self.all_logits >= threshold
            
            # Calculate hit-level metrics
            true_positives = np.sum(self.all_true_labels & predictions)
            false_positives = np.sum(~self.all_true_labels & predictions)
            true_negatives = np.sum(~self.all_true_labels & ~predictions)
            false_negatives = np.sum(self.all_true_labels & ~predictions)
            
            # Purity (Precision): TP / (TP + FP)
            total_predicted_positives = true_positives + false_positives
            purity = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
            
            # Rejection Rate: TN / (TN + FP)
            total_negatives = true_negatives + false_positives
            rejection = true_negatives / total_negatives if total_negatives > 0 else 0.0
            
            # Calculate particle reconstruction efficiency
            true_hit_mask = self.all_true_labels
            valid_particle_mask = true_hit_mask & (self.all_particle_pts > 0)
            
            unique_particles = np.unique(
                np.column_stack([
                    self.all_event_ids[valid_particle_mask],
                    self.all_particle_ids[valid_particle_mask]
                ]), axis=0
            )
            
            n_reconstructable = 0
            for event_id, particle_id in unique_particles:
                particle_mask = (
                    (self.all_event_ids == event_id) &
                    (self.all_particle_ids == particle_id) &
                    true_hit_mask
                )
                n_kept_hits = np.sum(particle_mask & predictions)
                if n_kept_hits >= 3:
                    n_reconstructable += 1
            
            particle_recon_efficiency = n_reconstructable / len(unique_particles) if len(unique_particles) > 0 else 0.0
            
            # Output metrics
            output_lines.append(f"\nThreshold: {threshold:.6f}")
            output_lines.append(f"Actual Hit Efficiency: {actual_efficiency:.6f}")
            output_lines.append("")
            output_lines.append("Hit-Level Metrics:")
            output_lines.append(f"  Purity (Precision):        {purity:.6f} ({purity*100:.2f}%)")
            output_lines.append(f"  Background Rejection Rate: {rejection:.6f} ({rejection*100:.2f}%)")
            output_lines.append(f"  True Positives:            {true_positives:,}")
            output_lines.append(f"  False Positives:           {false_positives:,}")
            output_lines.append(f"  True Negatives:            {true_negatives:,}")
            output_lines.append(f"  False Negatives:           {false_negatives:,}")
            output_lines.append("")
            output_lines.append("Particle-Level Metrics:")
            output_lines.append(f"  Total Particles:                    {len(unique_particles)}")
            output_lines.append(f"  Reconstructable Particles (≥3 hits): {n_reconstructable}")
            output_lines.append(f"  Particle Reconstruction Efficiency: {particle_recon_efficiency:.6f} ({particle_recon_efficiency*100:.2f}%)")
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append("END OF METRICS SUMMARY")
        output_lines.append(f"{'='*80}")
        
        # Save to file
        output_path = self.output_dir / "working_point_metrics.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        print(f"Working point metrics saved to {output_path}")
        
        # Also print to console
        print('\n'.join(output_lines))
        
        return output_path
    
    def plot_event_display(self, event_index):
        """
        Generate ATLAS-style event display plots showing before and after hit filtering.
        Creates 2D projections (X-Y, Z-Y, Z-X) in pyramid layout, matching the original visualizer.
        
        Parameters:
        -----------
        event_index : int
            The event index to visualize
        """
        print(f"\n--- Generating ATLAS-Style Event Display for Event {event_index} ---")
        
        # Load config to get inputs and targets
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_config = config.get('data', {})
        inputs = {k: list(v) for k, v in data_config.get('inputs', {}).items()}
        targets = {k: list(v) for k, v in data_config.get('targets', {}).items()}
        
        # Create dataset WITHOUT hit filtering (raw data)
        print("Creating dataset for raw data (before filtering)...")
        dataset_raw = AtlasMuonDataset(
            dirpath=str(self.data_dir),
            inputs=inputs,
            targets=targets,
            hit_eval_path=None  # No filtering
        )
        
        # Get the raw event data
        raw_inputs, raw_targets = dataset_raw[event_index]
        
        # Extract hit coordinates (convert to mm)
        all_high_x = raw_inputs["hit_spacePoint_globEdgeHighX"][0].numpy() * 1000
        all_high_y = raw_inputs["hit_spacePoint_globEdgeHighY"][0].numpy() * 1000
        all_high_z = raw_inputs["hit_spacePoint_globEdgeHighZ"][0].numpy() * 1000
        all_low_x = raw_inputs["hit_spacePoint_globEdgeLowX"][0].numpy() * 1000
        all_low_y = raw_inputs["hit_spacePoint_globEdgeLowY"][0].numpy() * 1000
        all_low_z = raw_inputs["hit_spacePoint_globEdgeLowZ"][0].numpy() * 1000
        
        # Get track assignments
        num_particles = np.sum(raw_targets["particle_valid"].numpy())
        truth_links = raw_targets["particle_hit_valid"][0][:num_particles, :].numpy()
        truthMuon_phi = raw_targets["particle_truthMuon_phi"][0][:num_particles].numpy()
        truthMuon_eta = raw_targets["particle_truthMuon_eta"][0][:num_particles].numpy()
        
        # Create track ID array
        all_truth = np.full(len(all_high_x), -1, dtype=int)  # Default to -1 for background
        for particle_id, truth_link in enumerate(truth_links):
            indices = np.where(truth_link)[0]
            all_truth[indices] = particle_id
        
        # Read the logits from the evaluation file and apply threshold
        print("Reading filter logits from evaluation file...")
        with h5py.File(self.eval_path, 'r') as eval_file:
            if str(event_index) not in eval_file:
                print(f"Warning: Event {event_index} not found in evaluation file")
                return
                
            event_group = eval_file[str(event_index)]
            
            if 'outputs/final/hit_filter/hit_logit' not in event_group:
                print(f"Warning: No logits found for event {event_index}")
                return
                
            logits = event_group['outputs/final/hit_filter/hit_logit'][0]
            
            # Apply threshold of -2.5
            filter_mask = logits > -2.640625
            
            n_total = len(filter_mask)
            n_passed = np.sum(filter_mask)
            n_rejected = n_total - n_passed
            
            print(f"Applied threshold -1.5 to logits:")
            print(f"  Total hits: {n_total}")
            print(f"  Passed filter: {n_passed} ({100*n_passed/n_total:.2f}%)")
            print(f"  Rejected: {n_rejected} ({100*n_rejected/n_total:.2f}%)")
        
        # Track colors (same as visualizer)
        track_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        
        # Plot BEFORE filtering
        print(f"\nGenerating 'Before Filtering' plot...")
        save_path_before = self.output_dir / f"event_{event_index}_before_filtering.png"
        
        fig_before = self._create_event_display_plot(
            all_high_x, all_high_y, all_high_z,
            all_low_x, all_low_y, all_low_z,
            all_truth, truthMuon_phi, truthMuon_eta,
            event_index, track_colors,
            title_suffix="Before Hit Filtering",
            info_text=f"{n_total} hits"
        )
        
        save_path_before_pdf = self.output_dir / f"event_{event_index}_before_filtering.pdf"
        fig_before.savefig(save_path_before, dpi=300, bbox_inches='tight')
        fig_before.savefig(save_path_before_pdf, format='pdf', bbox_inches='tight')
        plt.close(fig_before)
        print(f"✓ Before filtering plot saved to {save_path_before} and {save_path_before_pdf}")
        
        # Plot AFTER filtering
        print(f"\nGenerating 'After Filtering' plot...")
        save_path_after = self.output_dir / f"event_{event_index}_after_filtering.png"
        save_path_after_pdf = self.output_dir / f"event_{event_index}_after_filtering.pdf"
        
        # Apply filter mask to all arrays
        all_high_x_filt = all_high_x[filter_mask]
        all_high_y_filt = all_high_y[filter_mask]
        all_high_z_filt = all_high_z[filter_mask]
        all_low_x_filt = all_low_x[filter_mask]
        all_low_y_filt = all_low_y[filter_mask]
        all_low_z_filt = all_low_z[filter_mask]
        all_truth_filt = all_truth[filter_mask]
        
        fig_after = self._create_event_display_plot(
            all_high_x_filt, all_high_y_filt, all_high_z_filt,
            all_low_x_filt, all_low_y_filt, all_low_z_filt,
            all_truth_filt, truthMuon_phi, truthMuon_eta,
            event_index, track_colors,
            title_suffix="After Hit Filtering",
            info_text=f"{n_passed}/{n_total} hits ({100*n_passed/n_total:.1f}%)"
        )
        
        fig_after.savefig(save_path_after, dpi=300, bbox_inches='tight')
        fig_after.savefig(save_path_after_pdf, format='pdf', bbox_inches='tight')
        plt.close(fig_after)
        print(f"✓ After filtering plot saved to {save_path_after} and {save_path_after_pdf}")
        
        # Plot COMBINED side-by-side
        print(f"\nGenerating combined side-by-side plot...")
        save_path_combined = self.output_dir / f"event_{event_index}_combined.png"
        save_path_combined_pdf = self.output_dir / f"event_{event_index}_combined.pdf"
        
        fig_combined = self._create_combined_event_display(
            all_high_x, all_high_y, all_high_z,
            all_low_x, all_low_y, all_low_z,
            all_truth, truthMuon_phi, truthMuon_eta,
            filter_mask,
            event_index, track_colors,
            n_total, n_passed
        )
        
        fig_combined.savefig(save_path_combined, dpi=300, bbox_inches='tight')
        fig_combined.savefig(save_path_combined_pdf, format='pdf', bbox_inches='tight')
        plt.close(fig_combined)
        print(f"✓ Combined plot saved to {save_path_combined} and {save_path_combined_pdf}")
        
        # Force cleanup
        plt.close('all')
        import gc
        gc.collect()
        
        # Save event display info
        event_info_path = self.output_dir / f"event_{event_index}_info.txt"
        unique_tracks = np.unique(all_truth_filt[all_truth_filt >= 0])
        with open(event_info_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVENT {event_index} DISPLAY INFORMATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total hits in event: {n_total}\n")
            f.write(f"Hits passing filter (logits > -1.5): {n_passed} ({100*n_passed/n_total:.2f}%)\n")
            f.write(f"Hits rejected: {n_rejected} ({100*n_rejected/n_total:.2f}%)\n\n")
            f.write(f"Filter threshold: -1.5 (on logits)\n\n")
            f.write("Tracks in filtered event:\n")
            for track_id in unique_tracks:
                track_hits = np.sum(all_truth_filt == track_id)
                f.write(f"  Track {track_id}: {track_hits} hits\n")
            background_hits = np.sum(all_truth_filt == -1)
            if background_hits > 0:
                f.write(f"  Background: {background_hits} hits\n")
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Event info saved to {event_info_path}")
    
    def _create_event_display_plot(self, all_high_x, all_high_y, all_high_z,
                                   all_low_x, all_low_y, all_low_z,
                                   all_truth, truthMuon_phi, truthMuon_eta,
                                   event_number, track_colors,
                                   title_suffix="", info_text=""):
        """
        Create event display with X-Y transverse plane projection only.
        Shows view from above the detector with ATLAS styling.
        """
        # Separate background and track hits
        background_mask = all_truth == -1
        track_mask = all_truth >= 0
        
        # Create single square figure for X-Y projection
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot background hits
        if np.sum(background_mask) > 0:
            for low_x, low_y, high_x, high_y in zip(
                all_low_x[background_mask], all_low_y[background_mask],
                all_high_x[background_mask], all_high_y[background_mask]
            ):
                ax.plot([low_x, high_x], [low_y, high_y], color="gray", alpha=0.2, linewidth=0.3)
        
        # Plot track hits
        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            
            for i, track_id in enumerate(unique_tracks):
                track_color = track_colors[i % len(track_colors)]
                track_points = all_truth == track_id
                
                # Plot detector elements
                for x_low, y_low, x_high, y_high in zip(
                    all_low_x[track_points], all_low_y[track_points],
                    all_high_x[track_points], all_high_y[track_points]
                ):
                    ax.plot([x_low, x_high], [y_low, y_high], color=track_color, alpha=0.9, linewidth=0.3)
                
                # Plot truth trajectory lines (dashed)
                if i < len(truthMuon_phi):
                    phi = truthMuon_phi[i]
                    line_length = 14000
                    x1 = line_length * np.cos(phi)
                    y1 = line_length * np.sin(phi)
                    ax.plot([0, x1], [0, y1], color=track_color, linewidth=0.9, 
                           alpha=0.9, linestyle='--', label=f"Track {track_id}" if i < 5 else "")
        
        # Format plot with equal aspect ratio
        ax.set_xlabel("X [mm]", fontsize=14)
        ax.set_ylabel("Y [mm]", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        
        # Set same range for both axes to make it square
        max_range = 14000
        
        
        # Apply ATLAS style
        import atlasify
        atlasify.atlasify(
            axes=ax,
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + 
                    r"$\langle\mu\rangle$ = 200" + "\n" +
                    f"Event {event_number}" + "\n" +
                    title_suffix + "\n" + info_text,
            font_size=17,
            sub_font_size=14,
            label_font_size=17
        )
        
        # Create simple legend
        legend_elements = []
        if np.sum(background_mask) > 0:
            legend_elements.append(plt.Line2D([0], [0], color="gray", linewidth=2, alpha=0.4, label="Background"))
        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            for i, track_id in enumerate(unique_tracks[:5]):  # Limit legend to first 5 tracks
                track_color = track_colors[i % len(track_colors)]
                legend_elements.append(plt.Line2D([0], [0], color=track_color, linewidth=2, 
                                                 alpha=0.9, label=f"Track {track_id}"))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        
        plt.tight_layout()
        
        return fig

    def _create_combined_event_display(self, all_high_x, all_high_y, all_high_z,
                                       all_low_x, all_low_y, all_low_z,
                                       all_truth, truthMuon_phi, truthMuon_eta,
                                       filter_mask, event_number, track_colors,
                                       n_total, n_passed):
        """
        Create side-by-side event display showing before and after hit filtering.
        Uses shared Y-axis for direct comparison.
        """
        # Create figure with two subplots side by side
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        
        # --- LEFT PANEL: BEFORE FILTERING ---
        background_mask = all_truth == -1
        track_mask = all_truth >= 0
        
        # Plot background hits
        if np.sum(background_mask) > 0:
            for low_x, low_y, high_x, high_y in zip(
                all_low_x[background_mask], all_low_y[background_mask],
                all_high_x[background_mask], all_high_y[background_mask]
            ):
                ax_before.plot([low_x, high_x], [low_y, high_y], color="gray", alpha=0.2, linewidth=0.3)
        
        # Plot track hits
        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            
            for i, track_id in enumerate(unique_tracks):
                track_color = track_colors[i % len(track_colors)]
                track_points = all_truth == track_id
                
                # Plot detector elements
                for x_low, y_low, x_high, y_high in zip(
                    all_low_x[track_points], all_low_y[track_points],
                    all_high_x[track_points], all_high_y[track_points]
                ):
                    ax_before.plot([x_low, x_high], [y_low, y_high], color=track_color, alpha=0.9, linewidth=0.3)
                
                # Plot truth trajectory lines
                if i < len(truthMuon_phi):
                    phi = truthMuon_phi[i]
                    line_length = 14000
                    x1 = line_length * np.cos(phi)
                    y1 = line_length * np.sin(phi)
                    ax_before.plot([0, x1], [0, y1], color=track_color, linewidth=0.9, 
                                  alpha=0.9, linestyle='--')
        
        # Format left panel
        ax_before.set_xlabel("X [mm]", fontsize=14)
        ax_before.set_ylabel("Y [mm]", fontsize=14)
        ax_before.grid(True, alpha=0.3)
        ax_before.set_aspect("equal", adjustable="box")
        # ax_before.set_title("Before Hit Filtering", fontsize=16, pad=10)
        
        # --- RIGHT PANEL: AFTER FILTERING ---
        # Apply filter mask
        all_high_x_filt = all_high_x[filter_mask]
        all_high_y_filt = all_high_y[filter_mask]
        all_low_x_filt = all_low_x[filter_mask]
        all_low_y_filt = all_low_y[filter_mask]
        all_truth_filt = all_truth[filter_mask]
        
        background_mask_filt = all_truth_filt == -1
        track_mask_filt = all_truth_filt >= 0
        
        # Plot background hits
        if np.sum(background_mask_filt) > 0:
            for low_x, low_y, high_x, high_y in zip(
                all_low_x_filt[background_mask_filt], all_low_y_filt[background_mask_filt],
                all_high_x_filt[background_mask_filt], all_high_y_filt[background_mask_filt]
            ):
                ax_after.plot([low_x, high_x], [low_y, high_y], color="gray", alpha=0.2, linewidth=0.3)
        
        # Plot track hits
        if np.sum(track_mask_filt) > 0:
            unique_tracks_filt = np.unique(all_truth_filt[track_mask_filt])
            
            for i, track_id in enumerate(unique_tracks_filt):
                track_color = track_colors[i % len(track_colors)]
                track_points = all_truth_filt == track_id
                
                # Plot detector elements
                for x_low, y_low, x_high, y_high in zip(
                    all_low_x_filt[track_points], all_low_y_filt[track_points],
                    all_high_x_filt[track_points], all_high_y_filt[track_points]
                ):
                    ax_after.plot([x_low, x_high], [y_low, y_high], color=track_color, alpha=0.9, linewidth=0.3)
                
                # Plot truth trajectory lines
                if i < len(truthMuon_phi):
                    phi = truthMuon_phi[i]
                    line_length = 14000
                    x1 = line_length * np.cos(phi)
                    y1 = line_length * np.sin(phi)
                    ax_after.plot([0, x1], [0, y1], color=track_color, linewidth=0.9, 
                                 alpha=0.9, linestyle='--')
        
        # Format right panel
        ax_after.set_xlabel("X [mm]", fontsize=14)
        ax_after.grid(True, alpha=0.3)
        ax_after.set_aspect("equal", adjustable="box")
        # ax_after.set_title(f"After Hit Filtering ({n_passed}/{n_total} hits, {100*n_passed/n_total:.1f}%)", 
        #                   fontsize=16, pad=10)
        
       
        
        # Apply ATLAS style to both panels
        import atlasify
        atlasify.atlasify(
            axes=ax_before,
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + 
                    r"$\langle\mu\rangle$ = 200" + "\n" +
                    f"Event {event_number}" + "\n" +
                    f"Before Hit Filtering: {n_total} hits",
            font_size=17,
            sub_font_size=14,
            label_font_size=17
        )
        
        atlasify.atlasify(
            axes=ax_after,
            atlas="Simulation Preliminary",
            subtext=r"$\sqrt{s}$ = 14 TeV" + "\n" + 
                    r"$\langle\mu\rangle$ = 200" + "\n" +
                    f"Event {event_number}" + "\n" +
                    f"After Hit Filtering ({n_passed}/{n_total} hits, {100*n_passed/n_total:.1f}%)",
            font_size=17,
            sub_font_size=14,
            label_font_size=17
        )
        
         # Set same range for both panels
        max_range = 14000
        ax_before.set_xlim([-max_range, max_range])
        ax_before.set_ylim([-max_range, max_range])
        ax_after.set_xlim([-max_range, max_range])
        ax_after.set_ylim([-max_range, max_range])

        # Create combined legend (only on left panel to avoid duplication)
        legend_elements = []
        if np.sum(background_mask) > 0:
            legend_elements.append(plt.Line2D([0], [0], color="gray", linewidth=2, alpha=0.4, label="Background"))
        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            for i, track_id in enumerate(unique_tracks[:5]):
                track_color = track_colors[i % len(track_colors)]
                legend_elements.append(plt.Line2D([0], [0], color=track_color, linewidth=2, 
                                                 alpha=0.9, label=f"Track {track_id}"))
        
        if legend_elements:
            ax_before.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        
        return fig

    def save_plot_captions(self, event_display_index=None):
        """Save suggested plot captions for presentations and publications."""
        captions_path = self.output_dir / "plot_captions.txt"
        
        with open(captions_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ATLAS MUON TRACKING MODEL EVALUATION - HIT FILTERING STAGE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("ABSTRACT\n")
            f.write("-" * 80 + "\n")
            f.write("This analysis evaluates the hit filtering stage of a two-stage graph neural\n")
            f.write("network model for muon track reconstruction in the ATLAS detector at\n")
            f.write("High-Luminosity LHC conditions. The model processes simulated proton-proton\n")
            f.write("collisions at √s = 14 TeV with an average pileup of <μ> = 200, using events\n")
            f.write("containing muons from t𝑡̄, J/ψ, and Z→μμ processes with pT > 5 GeV.\n")
            f.write("\n")
            f.write("The two-stage architecture consists of: (1) a hit filtering stage that\n")
            f.write("reduces combinatorial background by classifying detector hits as signal or\n")
            f.write("noise, and (2) a tracking stage that performs hit-to-track assignment and\n")
            f.write("simultaneously reconstructs track parameters (η, φ, pT) and classifies the\n")
            f.write("muon charge sign. The following plots assess the hit filtering stage,\n")
            f.write("which serves to reduce data volume by >99%% while retaining signal hits.\n")
            f.write("\n")
            f.write("Uncertainty quantification for binomial proportions (hit classification)\n")
            f.write("uses Clopper-Pearson 95%% confidence intervals, which provide exact coverage\n")
            f.write("guarantees for finite sample sizes, ensuring statistically rigorous\n")
            f.write("uncertainty estimates.\n")
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIGURE CAPTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Figure 1: ROC Curve for Hit Classification\n")
            f.write("-" * 70 + "\n")
            f.write("Receiver operating characteristic curve showing true positive rate\n")
            f.write("(signal efficiency) versus false positive rate for hit classification.\n")
            f.write("The area under the curve (AUC) quantifies overall discrimination between\n")
            f.write("signal and background hits.\n")
            f.write("\n\n")
            
            f.write("Figure 2: Background Rejection vs. Hit Purity\n")
            f.write("-" * 70 + "\n")
            f.write("Background rejection rate as a function of hit purity (precision) for\n")
            f.write("varying signal efficiency thresholds (indicated by color scale). The red\n")
            f.write("cross marks the 99%% efficiency working point used in the analysis.\n")
            f.write("\n\n")
            
            f.write("Figure 3: Hit Efficiency vs. Hit Purity\n")
            f.write("-" * 70 + "\n")
            f.write("Hit efficiency (recall) versus hit purity (precision) as classification\n")
            f.write("threshold varies. The red cross indicates the 99%% efficiency operating\n")
            f.write("point selected to balance signal retention and background suppression.\n")
            f.write("\n\n")
            
            f.write("Figure 4: Inverse Background Efficiency vs. Hit Efficiency\n")
            f.write("-" * 70 + "\n")
            f.write("Signal efficiency versus inverse background efficiency (1/ε_background),\n")
            f.write("displayed on a logarithmic scale to emphasize background rejection\n")
            f.write("capability. The red cross marks the 99%% efficiency working point.\n")
            f.write("\n\n")
            
            f.write("Figure 5: Truth Muon pT Distribution\n")
            f.write("-" * 70 + "\n")
            f.write("Distribution of truth muon transverse momentum (pT) in the evaluation\n")
            f.write("dataset, shown on a logarithmic scale. This distribution reflects the\n")
            f.write("combined contributions from t𝑡̄, J/ψ→μμ, and Z→μμ processes.\n")
            f.write("\n\n")
            
            f.write("Figure 6: Truth Muon η Distribution\n")
            f.write("-" * 70 + "\n")
            f.write("Distribution of truth muon pseudorapidity (η) in the evaluation dataset,\n")
            f.write("shown on a logarithmic scale. The distribution spans the muon spectrometer\n")
            f.write("acceptance range.\n")
            f.write("\n\n")
            
            f.write("Figure 7: Truth Muon φ Distribution\n")
            f.write("-" * 70 + "\n")
            f.write("Distribution of truth muon azimuthal angle (φ) in the evaluation dataset,\n")
            f.write("shown on a logarithmic scale. A uniform distribution is expected in φ.\n")
            f.write("\n\n")
            
            if event_display_index is not None:
                f.write(f"Figure 8: Event {event_display_index} Display - Before Filtering\n")
                f.write("-" * 70 + "\n")
                f.write(f"Transverse (X-Y) projection of all reconstructed hits in event\n")
                f.write(f"{event_display_index} before hit filtering. Colored lines represent true\n")
                f.write("muon track trajectories, while gray points show background hits from\n")
                f.write("detector noise and pileup interactions.\n")
                f.write("\n\n")
                
                f.write(f"Figure 9: Event {event_display_index} Display - After Filtering\n")
                f.write("-" * 70 + "\n")
                f.write(f"Transverse (X-Y) projection of event {event_display_index} after applying\n")
                f.write("the hit filter at 99%% signal efficiency. Background suppression is evident,\n")
                f.write("while the muon track structure (colored lines) is preserved.\n")
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
            f.write("- Operating point: 99%% signal efficiency chosen to optimize downstream\n")
            f.write("  tracking performance\n")
            f.write("- Uncertainty quantification: Clopper-Pearson 95%% confidence intervals\n")
            f.write("  for binomial proportions\n")
            f.write("\n")
        
        print(f"Plot captions saved to {captions_path}")
    
    def run_analysis(self, event_display_index=None):
        """Run complete analysis and generate all plots."""
        print("=" * 80)
        print("ATLAS-STYLE PLOT GENERATION")
        print("=" * 80)
        print(f"Evaluation file: {self.eval_path}")
        print(f"Data directory: {self.data_dir}")
        print(f"Config file: {self.config_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max events: {self.max_events if self.max_events and self.max_events > 0 else 'ALL'}")
        print("=" * 80)
        
        # Setup data module
        self.setup_data_module()
        
        # Collect data
        if not self.collect_data():
            print("ERROR: Failed to collect data!")
            return False
        
        # Generate ATLAS-style ROC curve
        try:
            roc_auc = self.plot_atlas_roc_curve()
        except Exception as e:
            print(f"ERROR generating ROC curve: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Generate ATLAS-style Rejection vs Purity plot
        try:
            purities, rejections, wps = self.plot_atlas_rejection_vs_purity()
        except Exception as e:
            print(f"ERROR generating Rejection vs Purity plot: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Generate ATLAS-style Hit Efficiency vs Hit Purity plot
        try:
            purities_eff, efficiencies, wps_eff = self.plot_atlas_efficiency_vs_purity()
        except Exception as e:
            print(f"ERROR generating Hit Efficiency vs Purity plot: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Generate ATLAS-style 1/Background Efficiency vs Signal Efficiency plot
        try:
            signal_effs, inv_bkg_effs, wps_bkg = self.plot_atlas_background_efficiency_inverse()
        except Exception as e:
            print(f"ERROR generating 1/Background Efficiency plot: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Generate ATLAS-style Particle Reconstruction Efficiency plot
        # try:
        #     eff_vals, err_low, err_high, pt_centers = self.plot_particle_reconstruction_efficiency()
        # except Exception as e:
        #     print(f"ERROR generating Particle Reconstruction Efficiency plot: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return False
        
        # Generate truth distribution plots
        try:
            self.plot_truth_pt_distribution()
        except Exception as e:
            print(f"ERROR generating truth pT distribution plot: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole analysis
        
        try:
            self.plot_truth_eta_distribution()
        except Exception as e:
            print(f"ERROR generating truth eta distribution plot: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole analysis
        
        try:
            self.plot_truth_phi_distribution()
        except Exception as e:
            print(f"ERROR generating truth phi distribution plot: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole analysis
        
        # Generate logit distribution plot
        try:
            self.plot_logit_distribution()
        except Exception as e:
            print(f"ERROR generating logit distribution plot: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole analysis
        
        # Save working point metrics summary
        try:
            self.save_working_point_metrics()
        except Exception as e:
            print(f"ERROR saving working point metrics: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Generate event display if requested
        if event_display_index is not None:
            try:
                self.plot_event_display(event_display_index)
            except Exception as e:
                print(f"ERROR generating event display: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole analysis if event display fails
        
        # Save plot captions
        try:
            self.save_plot_captions(event_display_index=event_display_index)
        except Exception as e:
            print(f"ERROR saving plot captions: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole analysis if captions fail
        
        print("=" * 80)
        print("ANALYSIS COMPLETE!")
        print(f"All plots saved to: {self.output_dir}")
        print("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate ATLAS-style plots for muon hit filter evaluation')
    parser.add_argument('--eval_path', '-e', type=str, 
                       default="/scratch/epoch=039-val_loss=0.00399_ml_test_data_156000_hdf5_eval.h5",
                    #    default="/scratch/epoch=023-val_loss=0.00482_ml_test_data_156000_hdf5_no-NSW_no-RPC_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', '-d', type=str, 
                    #    default="/scratch/ml_test_data_156000_hdf5_no-NSW_no-RPC",
                       default="/scratch/ml_test_data_156000_hdf5",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', '-c', type=str, 
                       default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='.',
                       help='Base output directory (CTD_plots subdirectory will be created)')
    parser.add_argument('--max_events', '-m', type=int, default=-1,
                       help='Maximum number of events to process (-1 = all events)')
    parser.add_argument('--event_display', type=int, default=50100,
                       help='Event index for which to generate event display plots (before/after filtering)')
    
    args = parser.parse_args()
    
    # Enable stdout buffering for better logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    try:
        plotter = ATLASStylePlotter(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        success = plotter.run_analysis(event_display_index=args.event_display)
        
        if success:
            print("\n✓ All ATLAS-style plots generated successfully!")
            sys.exit(0)
        else:
            print("\n✗ Analysis failed!")
            sys.exit(1)
            
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
