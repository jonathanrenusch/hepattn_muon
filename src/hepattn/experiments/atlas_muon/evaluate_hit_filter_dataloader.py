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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        
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
            num_workers=50,  # Use many workers for maximum speed
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
    
    def collect_data(self):
        """Collect all data for analysis using the DataLoader."""
        print("Collecting data from all events using DataLoader...")
        
        # First, let's check what's in the evaluation file
        with h5py.File(self.eval_path, 'r') as eval_file:
            eval_keys = list(eval_file.keys())
            print(f"Evaluation file contains {len(eval_keys)} events")
            print(f"First few event keys: {eval_keys[:10]}")
        
        # Storage for collected data
        all_logits = []
        all_true_labels = []
        all_particle_pts = []
        all_particle_ids = []
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
                        truth_link_key = None
                        for key in inputs_batch.keys():
                            if 'truthLink' in key or 'spacePoint_truthLink' in key:
                                truth_link_key = key
                                break
                        
                        if truth_link_key is None:
                            print(f"Warning: spacePoint_truthLink not found in inputs for event {event_idx}")
                            print(f"Available input keys: {list(inputs_batch.keys())}")
                            continue
                        
                        hit_particle_ids = inputs_batch[truth_link_key][0].numpy()
                        
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
                        
                        if "particle_valid" in targets_batch:
                            particle_valid = targets_batch["particle_valid"][0].numpy()
                            num_valid_particles = np.sum(particle_valid.astype(bool))
                        else:
                            # Assume all particles are valid
                            num_valid_particles = len(particle_pts)
                            particle_valid = np.ones(len(particle_pts), dtype=bool)
                        
                        # Map hits to particle pt values
                        hit_pts = np.full(len(hit_logits), -1.0)  # Default for noise hits
                        
                        for i, particle_id in enumerate(hit_particle_ids):
                            # Convert particle_id to integer and check if it's valid
                            particle_id = int(particle_id)
                            if particle_id >= 0 and particle_id < num_valid_particles:
                                if particle_valid[particle_id]:
                                    hit_pts[i] = particle_pts[particle_id]
                        
                        # Store data
                        all_logits.extend(hit_logits)
                        all_true_labels.extend(true_labels)
                        all_particle_pts.extend(hit_pts)
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
    
    def plot_roc_curve(self):
        """Generate ROC curve with AUC score."""
        print("Generating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ATLAS Muon Hit Filter')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to {output_path}")
        print(f"AUC Score: {roc_auc:.4f}")
        
        return roc_auc
    
    def calculate_efficiency_purity_by_pt(self, working_point):
        """Calculate efficiency and purity binned by truthMuon_pt."""
        # Calculate ROC curve to determine threshold for target efficiency
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point
        
        # Find the threshold that achieves the target efficiency
        valid_indices = tpr >= target_efficiency
        if not np.any(valid_indices):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None, None
        
        threshold = thresholds[tpr >= target_efficiency][0]
        
        # Apply threshold to get predictions
        predictions = self.all_logits >= threshold
        
        # Only consider hits from valid particles (pt > 0)
        valid_particle_mask = self.all_particle_pts > 0
        
        if not np.any(valid_particle_mask):
            print("Warning: No valid particles found")
            return None, None, None, None, None
        
        # Filter data to valid particles only
        valid_logits = self.all_logits[valid_particle_mask]
        valid_true_labels = self.all_true_labels[valid_particle_mask]
        valid_particle_pts = self.all_particle_pts[valid_particle_mask]
        valid_predictions = predictions[valid_particle_mask]
        
        # Define pt bins
        pt_min, pt_max = 1.0, 100.0
        pt_bins = np.linspace(pt_min, pt_max, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate efficiency and purity for each bin
        efficiencies = []
        purities = []
        eff_errors = []
        pur_errors = []
        
        for i in range(len(pt_bins) - 1):
            pt_mask = (valid_particle_pts >= pt_bins[i]) & (valid_particle_pts < pt_bins[i+1])
            
            if not np.any(pt_mask):
                efficiencies.append(0.0)
                purities.append(0.0)
                eff_errors.append(0.0)
                pur_errors.append(0.0)
                continue
            
            bin_true_labels = valid_true_labels[pt_mask]
            bin_predictions = valid_predictions[pt_mask]
            
            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            false_negatives = np.sum(bin_true_labels & ~bin_predictions)
            total_positives = true_positives + false_negatives
            
            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0
            
            # Calculate purity (precision)
            false_positives = np.sum(~bin_true_labels & bin_predictions)
            total_predicted_positives = true_positives + false_positives
            
            if total_predicted_positives > 0:
                purity = true_positives / total_predicted_positives
                # Binomial error for purity
                pur_error = np.sqrt(purity * (1 - purity) / total_predicted_positives)
            else:
                purity = 0.0
                pur_error = 0.0
            
            efficiencies.append(efficiency)
            purities.append(purity)
            eff_errors.append(eff_error)
            pur_errors.append(pur_error)
        
        return pt_centers, np.array(efficiencies), np.array(purities), np.array(eff_errors), np.array(pur_errors)
    
    def plot_efficiency_purity_vs_pt(self, working_points=[0.96, 0.97, 0.98, 0.99, 0.995]):
        """Plot efficiency and purity vs pt for different working points."""
        print("Generating efficiency and purity plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, wp in enumerate(working_points):
            pt_centers, efficiencies, purities, eff_errors, pur_errors = self.calculate_efficiency_purity_by_pt(wp)
            
            if pt_centers is None:
                continue
            
            color = colors[i % len(colors)]
            label = f'WP {wp:.3f}'
            
            # Plot efficiency
            ax1.errorbar(pt_centers, efficiencies, yerr=eff_errors, 
                        label=label, color=color, marker='o', capsize=4, linewidth=2)
            
            # Plot purity
            ax2.errorbar(pt_centers, purities, yerr=pur_errors, 
                        label=label, color=color, marker='s', capsize=4, linewidth=2)
        
        # Format efficiency plot
        ax1.set_xlabel('Truth Muon $p_T$ [GeV]')
        ax1.set_ylabel('Efficiency (Recall)')
        ax1.set_title('Hit Filter Efficiency vs $p_T$')
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format purity plot
        ax2.set_xlabel('Truth Muon $p_T$ [GeV]')
        ax2.set_ylabel('Purity (Precision)')
        ax2.set_title('Hit Filter Purity vs $p_T$')
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "efficiency_purity_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Efficiency and purity plots saved to {output_path}")
    
    def plot_working_point_performance(self, working_points=[0.96, 0.97, 0.98, 0.99, 0.995]):
        """Plot average recall/efficiency for different working points."""
        print("Generating working point performance plot...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        
        avg_recalls = []
        avg_recall_errors = []
        
        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                avg_recalls.append(0.0)
                avg_recall_errors.append(0.0)
                continue
            
            threshold = thresholds[tpr >= wp][0]
            predictions = self.all_logits >= threshold
            
            # Only consider hits from valid particles
            valid_particle_mask = self.all_particle_pts > 0
            
            if not np.any(valid_particle_mask):
                avg_recalls.append(0.0)
                avg_recall_errors.append(0.0)
                continue
            
            valid_true_labels = self.all_true_labels[valid_particle_mask]
            valid_predictions = predictions[valid_particle_mask]
            
            # Calculate overall recall
            true_positives = np.sum(valid_true_labels & valid_predictions)
            total_positives = np.sum(valid_true_labels)
            
            if total_positives > 0:
                recall = true_positives / total_positives
                recall_error = np.sqrt(recall * (1 - recall) / total_positives)
            else:
                recall = 0.0
                recall_error = 0.0
            
            avg_recalls.append(recall)
            avg_recall_errors.append(recall_error)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(working_points, avg_recalls, yerr=avg_recall_errors,
                    marker='o', capsize=4, linewidth=2, markersize=8,
                    color='darkblue', label='Average Recall')
        
        plt.xlabel('Working Point (Target Efficiency)')
        plt.ylabel('Achieved Average Recall')
        plt.title('Working Point Performance - ATLAS Muon Hit Filter')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.0, 1.05)
        
        # Save plot
        output_path = self.output_dir / "working_point_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Working point performance plot saved to {output_path}")
        
        # Print performance summary
        print("\nWorking Point Performance Summary:")
        for wp, recall, error in zip(working_points, avg_recalls, avg_recall_errors):
            print(f"  WP {wp:.3f}: Recall = {recall:.4f} ± {error:.4f}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("Starting full evaluation of ATLAS muon hit filter...")
        
        # Collect data
        if not self.collect_data():
            print("Data collection failed, aborting evaluation")
            return
        
        # Generate all plots
        roc_auc = self.plot_roc_curve()
        self.plot_efficiency_purity_vs_pt()
        self.plot_working_point_performance()
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")
        print(f"Final AUC Score: {roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ATLAS muon hit filter using DataLoader')
    parser.add_argument('--eval_path', type=str, required=True,
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events', type=int, default=-1,
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
