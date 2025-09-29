#!/usr/bin/env python3
"""
Simplified evaluation script for ATLAS muon hit filtering using direct data loading.
Generates ROC curves, efficiency/purity plots binned by truthMuon_pt with improved styling.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import warnings
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

class AtlasMuonEvaluatorSimple:
    """Simplified evaluator for ATLAS muon hit filtering performance."""
    
    def __init__(self, eval_path, data_path, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        
        # Load metadata
        with open(self.data_path / 'metadata.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        self.hit_features = self.metadata['hit_features']
        self.track_features = self.metadata['track_features']
        
        # Load efficient index arrays
        self.file_indices = np.load(self.data_path / 'event_file_indices.npy')
        self.row_indices = np.load(self.data_path / 'event_row_indices.npy')
        
        # Working points for efficiency analysis
        self.working_points = [0.96, 0.97, 0.98, 0.99, 0.995]
        
        print(f"Initialized evaluator with {len(self.file_indices)} events")
        if self.max_events:
            print(f"Will process only first {self.max_events} events")
    
    def load_event_data(self, idx):
        """Load a single event's truth data and particle info efficiently."""
        file_idx = self.file_indices[idx]
        row_idx = self.row_indices[idx]
        
        chunk = self.metadata['event_mapping']['chunk_summary'][file_idx]
        h5_file_path = self.data_path / chunk['h5_file']
        
        try:
            with h5py.File(h5_file_path, 'r') as f:
                num_hits = f['num_hits'][row_idx]
                num_tracks = f['num_tracks'][row_idx]
                
                hits_array = f['hits'][row_idx, :num_hits]
                tracks_array = f['tracks'][row_idx, :num_tracks]
                
        except Exception as e:
            raise RuntimeError(f"Failed to load event {idx} from {h5_file_path}: {e}")
        
        # Extract only needed data
        truth_link_idx = self.hit_features.index('spacePoint_truthLink')
        hit_particle_ids = hits_array[:, truth_link_idx].astype(int)
        
        # Get truth labels
        true_labels = hit_particle_ids >= 0
        
        # Get particle pt values
        pt_idx = self.track_features.index('truthMuon_pt')
        particle_pts = tracks_array[:, pt_idx]
        
        # Map hits to particle pt values
        hit_pts = np.full(len(hit_particle_ids), -1.0)  # Default for noise hits
        
        for i, particle_id in enumerate(hit_particle_ids):
            if particle_id >= 0 and particle_id < len(particle_pts):
                hit_pts[i] = particle_pts[particle_id]
        
        return true_labels, hit_particle_ids, hit_pts
    
    def collect_data(self):
        """Collect all data for analysis using direct loading."""
        print("Collecting data from all events...")
        
        # Storage for collected data
        all_logits = []
        all_true_labels = []
        all_particle_pts = []
        all_particle_ids = []
        all_event_ids = []
        
        # Determine number of events to process
        num_events = len(self.file_indices)
        if self.max_events:
            num_events = min(num_events, self.max_events)
        
        with h5py.File(self.eval_path, 'r') as eval_file:
            for idx in tqdm(range(num_events), desc="Processing events"):
                try:
                    # Load predictions
                    if str(idx) not in eval_file:
                        print(f"Warning: Event {idx} not found in eval file, skipping")
                        continue
                    
                    hit_logits = eval_file[f"{idx}/outputs/final/hit_filter/hit_logit"][0]
                    
                    # Load truth data
                    true_labels, hit_particle_ids, hit_pts = self.load_event_data(idx)
                    
                    # Verify shapes match
                    if len(hit_logits) != len(true_labels):
                        print(f"Warning: Shape mismatch in event {idx}, skipping")
                        continue
                    
                    # Store data
                    all_logits.extend(hit_logits)
                    all_true_labels.extend(true_labels)
                    all_particle_pts.extend(hit_pts)
                    all_particle_ids.extend(hit_particle_ids)
                    all_event_ids.extend([idx] * len(hit_logits))
                    
                except Exception as e:
                    print(f"Error processing event {idx}: {e}")
                    continue
        
        # Convert to numpy arrays
        self.all_logits = np.array(all_logits)
        self.all_true_labels = np.array(all_true_labels, dtype=bool)
        self.all_particle_pts = np.array(all_particle_pts)
        self.all_particle_ids = np.array(all_particle_ids)
        self.all_event_ids = np.array(all_event_ids)
        
        print(f"Collected data from {len(self.all_logits)} hits")
        print(f"True hits: {np.sum(self.all_true_labels)}")
        print(f"Noise hits: {np.sum(~self.all_true_labels)}")
        print(f"Valid particle hits (pt > 0): {np.sum(self.all_particle_pts > 0)}")
        
        # Print pt statistics for valid particles
        valid_pt_mask = self.all_particle_pts > 0
        if np.any(valid_pt_mask):
            valid_pts = self.all_particle_pts[valid_pt_mask]
            print("PT statistics for valid particles:")
            print(f"  Min: {np.min(valid_pts):.1f} GeV")
            print(f"  Max: {np.max(valid_pts):.1f} GeV")
            print(f"  Mean: {np.mean(valid_pts):.1f} GeV")
            print(f"  Median: {np.median(valid_pts):.1f} GeV")
    
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
            print(f"Warning: Cannot achieve efficiency {target_efficiency}, using highest available")
            threshold = thresholds[-1]
        else:
            threshold = thresholds[valid_indices][0]
        
        print(f"Using threshold {threshold:.4f} for target efficiency {target_efficiency}")
        achieved_eff = tpr[valid_indices][0] if np.any(valid_indices) else tpr[-1]
        print(f"Overall efficiency achieved: {achieved_eff:.3f}")
        
        # Apply threshold to get predictions
        predictions = self.all_logits >= threshold
        
        # Get pt range for valid particles
        valid_pt_mask = self.all_particle_pts > 0
        if np.any(valid_pt_mask):
            min_pt = np.min(self.all_particle_pts[valid_pt_mask])
            max_pt = np.max(self.all_particle_pts[valid_pt_mask])
            print(f"PT range for valid particles: {min_pt:.1f} to {max_pt:.1f} GeV")
            
            # Define pt bins based on actual data range
            pt_bins = np.linspace(min_pt, max_pt, 21)
        else:
            print("No valid particles found!")
            return None
        
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        efficiencies = []
        purities = []
        efficiency_errors = []
        purity_errors = []
        tracks_lost_percentages = []
        
        for i in range(len(pt_bins) - 1):
            # Get hits in this pt bin (only consider true muon hits)
            pt_mask = ((self.all_particle_pts >= pt_bins[i]) & 
                      (self.all_particle_pts < pt_bins[i+1]) & 
                      self.all_true_labels)
            
            if np.sum(pt_mask) == 0:
                efficiencies.append(0)
                purities.append(0)
                efficiency_errors.append(0)
                purity_errors.append(0)
                tracks_lost_percentages.append(0)
                continue
            
            # Calculate efficiency (recall)
            true_hits_in_bin = np.sum(pt_mask)
            predicted_true_in_bin = np.sum(predictions[pt_mask])
            efficiency = predicted_true_in_bin / true_hits_in_bin if true_hits_in_bin > 0 else 0
            
            # Calculate purity (precision) - include noise in the same spatial region
            pt_mask_all = ((self.all_particle_pts >= pt_bins[i]) & 
                          (self.all_particle_pts < pt_bins[i+1]))
            
            predicted_in_bin = np.sum(predictions[pt_mask_all])
            actual_true_predicted_in_bin = np.sum(predictions[pt_mask_all] & self.all_true_labels[pt_mask_all])
            purity = actual_true_predicted_in_bin / predicted_in_bin if predicted_in_bin > 0 else 0
            
            # Calculate track loss percentage
            unique_particles = np.unique(self.all_particle_ids[pt_mask])
            tracks_lost = 0
            total_tracks = 0
            
            for particle_id in unique_particles:
                if particle_id < 0:  # Skip noise
                    continue
                particle_hits_mask = (self.all_particle_ids == particle_id) & pt_mask
                particle_predicted_hits = np.sum(predictions[particle_hits_mask])
                total_tracks += 1
                if particle_predicted_hits < 3:
                    tracks_lost += 1
            
            tracks_lost_pct = (tracks_lost / total_tracks * 100) if total_tracks > 0 else 0
            
            # Calculate binomial uncertainties
            efficiency_error = np.sqrt(efficiency * (1 - efficiency) / true_hits_in_bin) if true_hits_in_bin > 0 else 0
            purity_error = np.sqrt(purity * (1 - purity) / predicted_in_bin) if predicted_in_bin > 0 else 0
            
            efficiencies.append(efficiency)
            purities.append(purity)
            efficiency_errors.append(efficiency_error)
            purity_errors.append(purity_error)
            tracks_lost_percentages.append(tracks_lost_pct)
        
        return (pt_centers, np.array(efficiencies), np.array(purities), 
                np.array(efficiency_errors), np.array(purity_errors), 
                np.array(tracks_lost_percentages))
    
    def plot_efficiency_purity_by_pt(self):
        """Plot efficiency and purity vs pt for all working points with improved styling."""
        print("Generating efficiency and purity plots...")
        
        # Create output directories
        (self.output_dir / "efficiency_plots").mkdir(exist_ok=True)
        (self.output_dir / "purity_plots").mkdir(exist_ok=True)
        
        for wp in self.working_points:
            print(f"Processing working point {wp}...")
            
            # Calculate metrics
            result = self.calculate_efficiency_purity_by_pt(wp)
            if result is None:
                continue
                
            (pt_centers, efficiencies, purities, 
             eff_errors, pur_errors, tracks_lost) = result
            
            # Calculate average metrics
            valid_mask = efficiencies > 0
            avg_efficiency = np.mean(efficiencies[valid_mask]) if np.any(valid_mask) else 0
            avg_purity = np.mean(purities[valid_mask]) if np.any(valid_mask) else 0
            avg_tracks_lost = np.mean(tracks_lost[valid_mask]) if np.any(valid_mask) else 0
            
            # Plot efficiency with improved styling
            plt.figure(figsize=(10, 7))
            
            # Create color scheme similar to the attached plot
            color = '#1f77b4'  # Blue color
            
            # Plot with filled error bands
            plt.fill_between(pt_centers, efficiencies - eff_errors, efficiencies + eff_errors,
                           alpha=0.3, color=color, label='_nolegend_')
            
            # Plot the main line with error bars
            plt.errorbar(pt_centers, efficiencies, yerr=eff_errors, 
                        marker='o', capsize=4, markersize=6,
                        color=color, linewidth=2, linestyle='-',
                        label=f'Efficiency (Target: {wp})')
            
            plt.xlabel('Particle $p_T^{True}$ [GeV]', fontsize=16)
            plt.ylabel('Efficiency', fontsize=16)
            plt.title(f'Hit Filter Efficiency vs $p_T$ (Target Efficiency: {wp})', fontsize=16)
            
            # Set axis limits and grid
            plt.xlim([0, np.max(pt_centers) * 1.05])
            plt.ylim([0.85, 1.02])  # Similar to the attached plot
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add horizontal line at target efficiency
            plt.axhline(y=wp, color='red', linestyle='--', alpha=0.7, 
                       label=f'Target: {wp}')
            
            # Add text box with statistics
            textstr = f'Avg Efficiency: {avg_efficiency:.3f}\nTracks Lost: {avg_tracks_lost:.1f}%'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "efficiency_plots" / f"efficiency_target_{wp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot purity with improved styling
            plt.figure(figsize=(10, 7))
            
            # Use a different color for purity
            color = '#ff7f0e'  # Orange color
            
            # Plot with filled error bands
            plt.fill_between(pt_centers, purities - pur_errors, purities + pur_errors,
                           alpha=0.3, color=color, label='_nolegend_')
            
            # Plot the main line with error bars
            plt.errorbar(pt_centers, purities, yerr=pur_errors, 
                        marker='s', capsize=4, markersize=6,
                        color=color, linewidth=2, linestyle='-',
                        label=f'Purity (Target Eff: {wp})')
            
            plt.xlabel('Particle $p_T^{True}$ [GeV]', fontsize=16)
            plt.ylabel('Purity', fontsize=16)
            plt.title(f'Hit Filter Purity vs $p_T$ (Target Efficiency: {wp})', fontsize=16)
            
            # Set axis limits and grid
            plt.xlim([0, np.max(pt_centers) * 1.05])
            plt.ylim([0, 1.05])
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add text box with statistics
            textstr = f'Avg Purity: {avg_purity:.3f}\nTracks Lost: {avg_tracks_lost:.1f}%'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "purity_plots" / f"purity_target_{wp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Avg Efficiency: {avg_efficiency:.3f}")
            print(f"  Avg Purity: {avg_purity:.3f}")
            print(f"  Avg Tracks Lost: {avg_tracks_lost:.1f}%")
    
    def generate_summary_report(self):
        """Generate a summary report with key metrics."""
        print("Generating summary report...")
        
        report_lines = ["ATLAS Muon Hit Filter Evaluation Report", "=" * 50, ""]
        
        # Overall statistics
        total_hits = len(self.all_logits)
        true_hits = np.sum(self.all_true_labels)
        noise_hits = total_hits - true_hits
        
        report_lines.extend([
            f"Total hits analyzed: {total_hits:,}",
            f"True muon hits: {true_hits:,} ({true_hits/total_hits*100:.1f}%)",
            f"Noise hits: {noise_hits:,} ({noise_hits/total_hits*100:.1f}%)",
            ""
        ])
        
        # ROC AUC
        fpr, tpr, thresholds = roc_curve(self.all_true_labels, self.all_logits)
        roc_auc = auc(fpr, tpr)
        report_lines.append(f"ROC AUC Score: {roc_auc:.4f}")
        report_lines.append("")
        
        # Working point summary
        report_lines.append("Target Efficiency Summary:")
        report_lines.append("-" * 30)
        
        for wp in self.working_points:
            # Find threshold for target efficiency
            target_efficiency = wp
            valid_indices = tpr >= target_efficiency
            if not np.any(valid_indices):
                threshold = thresholds[-1]
                achieved_efficiency = tpr[-1]
            else:
                threshold = thresholds[valid_indices][0]
                achieved_efficiency = tpr[valid_indices][0]
            
            # Apply threshold
            predictions = self.all_logits >= threshold
            
            # Overall metrics
            tp = np.sum(predictions & self.all_true_labels)
            fp = np.sum(predictions & ~self.all_true_labels)
            fn = np.sum(~predictions & self.all_true_labels)
            
            efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0
            purity = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            report_lines.extend([
                f"Target Efficiency {wp}:",
                f"  Threshold used: {threshold:.4f}",
                f"  Achieved Efficiency: {efficiency:.3f}",
                f"  Overall Purity: {purity:.3f}",
                ""
            ])
        
        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to {report_path}")
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("Starting full evaluation pipeline...")
        
        # Collect data
        self.collect_data()
        
        # Generate plots
        self.plot_roc_curve()
        self.plot_efficiency_purity_by_pt()
        
        # Generate summary
        self.generate_summary_report()
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ATLAS muon hit filter performance")
    parser.add_argument("--eval-path", type=str, 
                       default="/shared/tracking/hepattn_muon/src/logs/ATLAS-Muon-6H100-600K_20250831-T195418/ckpts/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5",
                       help="Path to the evaluation HDF5 file")
    parser.add_argument("--data-path", type=str, 
                       default="/scratch/ml_test_data_156000_hdf5",
                       help="Path to the raw data directory")
    parser.add_argument("--output-dir", type=str, 
                       default="./evaluation_results",
                       help="Output directory for plots and reports")
    parser.add_argument("--max-events", type=int, default=None,
                       help="Maximum number of events to process (default: all)")
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = AtlasMuonEvaluatorSimple(
        eval_path=args.eval_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_events=args.max_events
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
