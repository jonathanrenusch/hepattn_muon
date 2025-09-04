#!/usr/bin/env python3
"""
Evaluation script for ATLAS muon hit filtering model using DataLoader approach.
This version uses the AtlasMuonDataModule for proper multi-worker data loading.
"""

import argparse
import h5py
import numpy as np
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

# Import the data module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings('ignore')

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
        self.technology_mapping = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 3, "MM": 5}
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
        """Collect all data for analysis using the DataLoader."""
        print("Collecting data from all events using DataLoader...")
        
        # First, let's check what's in the evaluation file
        with h5py.File(self.eval_path, 'r') as eval_file:
            eval_keys = list(eval_file.keys())
            print(f"Evaluation file contains {len(eval_keys)} events")
        
        # Storage for collected data
        all_logits = []
        all_true_labels = []
        all_particle_pts = []
        all_particle_eta = []
        all_particle_phi = []
        all_particle_ids = []
        all_particle_technology = []
        all_event_ids = []
        events_processed = 0
        events_attempted = 0
        
        try:
            with h5py.File(self.eval_path, 'r') as eval_file:
                for batch_idx, batch_data in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                    events_attempted += 1
                    
                    # Only break if max_events is explicitly set (not None or -1)
                    if self.max_events is not None and self.max_events > 0 and events_processed >= self.max_events:
                        break
                    
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
                            print(f"Available target keys: {list(targets_batch.keys())}")
                            continue
                        
                        event_idx = targets_batch["sample_id"][0].item()
                        
                        # Load predictions for this event
                        if str(event_idx) not in eval_file:
                            raise Exception(f"Event {event_idx} not found in evaluation file")
                            # Skip events not found in eval file
                            continue
                        
                        # Get hit logits from evaluation file
                        hit_logits = eval_file[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0]
                        
                        # Get truth labels from DataLoader
                        if "hit_on_valid_particle" not in targets_batch:
                            print(f"Warning: hit_on_valid_particle not found in targets for event {event_idx}")
                            print(f"Available target keys: {list(targets_batch.keys())}")
                            continue
                        
                        true_labels = targets_batch["hit_on_valid_particle"][0].numpy()

                        # Get spacePoint_truthLink from inputs
                        hit_particle_ids = inputs_batch["plotting_spacePoint_truthLink"][0].numpy()

                        hit_technologies = inputs_batch["hit_spacePoint_technology"][0].numpy()
                        # Verify shapes match
                        if len(hit_logits) != len(true_labels) or len(hit_logits) != len(hit_particle_ids):
                            print(f"Warning: Shape mismatch in event {event_idx}")
                            print(f"  Logits: {len(hit_logits)}, Labels: {len(true_labels)}, IDs: {len(hit_particle_ids)}")
                            continue
                        
                        # Get particle pt values
                        if "particle_truthMuon_pt" not in targets_batch:
                            print(f"Warning: particle_truthMuon_pt not found in targets for event {event_idx}")
                            continue
                        
                        particle_pts = targets_batch["particle_truthMuon_pt"][0].numpy()
                        particle_etas = targets_batch["particle_truthMuon_eta"][0].numpy()
                        particle_phis = targets_batch["particle_truthMuon_phi"][0].numpy()

                        particle_valid = targets_batch["particle_valid"][0].numpy()
                        # print(particle_valid)
                        num_valid_particles = np.sum(particle_valid.astype(bool))
                        # Map hits to particle pt values
                        hit_pts = np.full(len(hit_logits), -1.0)  # Default for noise hits
                        hit_etas = np.full(len(hit_logits), -1.0)  # Default for noise hits
                        hit_phis = np.full(len(hit_logits), -1.0)  # Default for noise hits
                        # print(np.unique(hit_particle_ids))
                        for idx, hit_particle_id in enumerate(np.unique(hit_particle_ids)[1:]): # Skip -1 (noise)
                            hit_mask = hit_particle_ids == hit_particle_id
                            hit_pts[hit_mask] = particle_pts[idx]
                            hit_etas[hit_mask] = particle_etas[idx]
                            hit_phis[hit_mask] = particle_phis[idx]


                        # for i, particle_id in enumerate(hit_particle_ids):
                        #     if particle_id >= 0:
                        #         hit_pts[i] = particle_pts[particle_id]
                        
                        # Store data
                        all_logits.extend(hit_logits)
                        all_true_labels.extend(true_labels)
                        all_particle_pts.extend(hit_pts)
                        all_particle_eta.extend(hit_etas)
                        all_particle_phi.extend(hit_phis)
                        all_particle_technology.extend(hit_technologies)
                        all_particle_ids.extend(hit_particle_ids)
                        all_event_ids.extend([event_idx] * len(hit_logits))
                        
                        events_processed += 1
                        
                        # Print progress every 1000 events
                        # if events_processed % 1000 == 0:
                        #     print(f"Processed {events_processed} events...")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        print("Traceback:")
                        traceback.print_exc()
                        continue
        
        except Exception as e:
            print(f"Error during data collection: {e}")
            print("Traceback:")
            traceback.print_exc()
            return False
        
        print(f"\nDataLoader provided {events_attempted} batches, successfully processed {events_processed} events")
        
        if events_processed == 0:
            print("ERROR: No events were successfully processed!")
            print("This could be due to:")
            print("1. Event index mismatch between DataLoader and evaluation file")
            print("2. Missing required fields in the data")
            print("3. DataLoader configuration issues")
            return False
        
        # Convert to numpy arrays
        self.all_logits = np.array(all_logits)
        self.all_true_labels = np.array(all_true_labels, dtype=bool)
        self.all_particle_pts = np.array(all_particle_pts)
        self.all_particle_etas = np.array(all_particle_eta)
        self.all_particle_phis = np.array(all_particle_phi)
        self.all_particle_technology = np.array(all_particle_technology)
        self.all_particle_ids = np.array(all_particle_ids)
        self.all_event_ids = np.array(all_event_ids)
        
        print(f"\nData collection complete!")
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
    
    def plot_efficiency_vs_pt(self, working_points=[0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]):
        """
        Plot efficiency vs pT for different working points with overall purity in legend
        """
        print("Generating efficiency plots...")
        
        # First, create the general (all technologies) plots
        self._plot_efficiency_vs_pt_general(working_points)
        
        # Then, create technology-specific plots
        self._plot_efficiency_vs_pt_by_technology(working_points)
    
    def _plot_efficiency_vs_pt_general(self, working_points):
        """Create general efficiency vs pT plots (all technologies combined)"""
        # Prepare data for all working points
        results_dict = {}
        
        for wp in working_points:
            pt_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_pt(wp)
            
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
        
        # Create individual plots for each working point
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

    def plot_working_point_performance(self, working_points=[0.96, 0.97, 0.98, 0.99, 0.995]):
        """Plot average purity for different working points with detailed track statistics."""
        print("Generating working point performance plot...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        avg_purities = []
        avg_purity_errors = []
        track_statistics = {}
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                avg_purities.append(0.0)
                avg_purity_errors.append(0.0)
                track_statistics[wp] = {
                    'total_tracks': 0,
                    'tracks_with_few_hits': 0,
                    'tracks_completely_lost': 0,
                    'events_analyzed': 0
                }
                continue
            
            threshold = thresholds[tpr >= wp][0]
            predictions = self.all_logits >= threshold
            
            # Calculate overall purity
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
            
            # Calculate detailed track statistics per event
            track_stats = self._calculate_track_statistics_per_working_point(wp, threshold, predictions)
            track_statistics[wp] = track_stats
        
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
    
    def plot_efficiency_vs_eta(self, working_points=[0.96, 0.97, 0.98, 0.99, 0.995]):
        """
        Plot efficiency vs eta for different working points with overall purity in legend
        """
        print("Generating efficiency vs eta plots...")
        
        # First, create the general (all technologies) plots
        self._plot_efficiency_vs_eta_general(working_points)
        
        # Then, create technology-specific plots
        self._plot_efficiency_vs_eta_by_technology(working_points)
    
    def _plot_efficiency_vs_eta_general(self, working_points):
        """Create general efficiency vs eta plots (all technologies combined)"""
        # Prepare data for all working points
        results_dict = {}
        
        for wp in working_points:
            eta_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_eta(wp)
            
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
        
        # Create individual plots for eta
        self._plot_individual_working_points_eta(results_dict)
    
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
    
    def plot_efficiency_vs_phi(self, working_points=[0.96, 0.97, 0.98, 0.99, 0.995]):
        """
        Plot efficiency vs phi for different working points with overall purity in legend
        """
        print("Generating efficiency vs phi plots...")
        
        # First, create the general (all technologies) plots
        self._plot_efficiency_vs_phi_general(working_points)
        
        # Then, create technology-specific plots
        self._plot_efficiency_vs_phi_by_technology(working_points)
    
    def _plot_efficiency_vs_phi_general(self, working_points):
        """Create general efficiency vs phi plots (all technologies combined)"""
        # Prepare data for all working points
        results_dict = {}
        
        for wp in working_points:
            phi_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_phi(wp)
            
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
        
        # Create individual plots for phi
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
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("Starting full evaluation of ATLAS muon hit filter...")
        
        # Collect data
        if not self.collect_data():
            print("Data collection failed, aborting evaluation")
            return
        
        # Generate all plots
        roc_auc = self.plot_roc_curve()
        self.plot_efficiency_vs_pt()
        self.plot_efficiency_vs_eta()
        self.plot_efficiency_vs_phi()
        self.plot_track_lengths()
        self.plot_working_point_performance()
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")
        print(f"Final AUC Score: {roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ATLAS muon hit filter using DataLoader')
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
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = AtlasMuonEvaluatorDataLoader(
        eval_path=args.eval_path,
        data_dir=args.data_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        max_events=args.max_events
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
