#!/usr/bin/env python3
"""
Script to analyze station occupation statistics for ATLAS Muon tracking data.
This script analyzes:
1. Average number of unique station indices per track as a function of eta, phi, pt
2. Number of true hits per track as a function of eta, phi, pt
3. Average number of true/background/total hits per station as a function of eta, phi, pt
4. Overall statistics for unbinned data
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import h5py
import numpy as np
from tqdm import tqdm
import torch

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data_vis.h5_config import DEFAULT_TREE_NAME, H5_FILEPATH, HIT_EVAL_FILEPATH
from .data import AtlasMuonDataModule, AtlasMuonDataset


def create_output_directory(base_output_dir: str) -> Path:
    """Create timestamped output directory for the analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"station_occupation_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_inputs_targets_from_config(config_path):
    """Load input and target configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config.get("data", {})
    inputs = data_cfg.get("inputs", {})
    targets = data_cfg.get("targets", {})
    return inputs, targets


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    return "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_")).rstrip()


class StationOccupationAnalyzer:
    """Analyzer for station occupation statistics."""
    
    def __init__(self, datamodule: AtlasMuonDataModule, dataset: AtlasMuonDataset, 
                 num_events: int = -1):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        datamodule : AtlasMuonDataModule
            Data module for loading batches
        dataset : AtlasMuonDataset
            Dataset for direct event access
        num_events : int
            Number of events to analyze (-1 for all)
        """
        self.datamodule = datamodule
        self.dataset = dataset
        self.num_events = num_events if num_events > 0 else len(dataset)
        
        # Define bins for eta, phi, pt
        self.eta_bins = np.linspace(-3, 3, 21)  # 20 bins
        self.phi_bins = np.linspace(-np.pi, np.pi, 21)  # 20 bins
        self.pt_bins = np.linspace(0, 200, 21)  # 20 bins
        
        # Storage for unbinned statistics
        self.unbinned_stats = {
            'num_unique_stations': [],
            'num_true_hits': [],
            'num_background_hits': [],
            'num_total_hits': [],
        }
        
    def analyze(self) -> Dict:
        """
        Run the complete analysis.
        
        Returns:
        --------
        Dict containing all analysis results
        """
        print(f"\n{'='*80}")
        print("Starting Station Occupation Analysis")
        print(f"Analyzing {self.num_events} events")
        print(f"{'='*80}\n")
        
        # Initialize data structures for binned statistics
        binned_data = self._initialize_binned_data()
        
        # Get dataloader
        test_dataloader = self.datamodule.test_dataloader(shuffle=True)
        
        # Process events
        for event_idx, batch in enumerate(tqdm(test_dataloader, 
                                                desc="Processing events",
                                                total=self.num_events)):
            if event_idx >= self.num_events:
                break
                
            self._process_event(event_idx, batch, binned_data)
            
            # Periodic garbage collection to prevent memory issues
            if event_idx % 1000 == 0:
                import gc
                gc.collect()
        
        # Compute final statistics
        results = self._compute_statistics(binned_data)
        
        return results
    
    def _initialize_binned_data(self) -> Dict:
        """Initialize data structures for binned statistics."""
        n_eta = len(self.eta_bins) - 1
        n_phi = len(self.phi_bins) - 1
        n_pt = len(self.pt_bins) - 1
        
        return {
            'eta': {
                'unique_stations': [[] for _ in range(n_eta)],
                'true_hits_per_track': [[] for _ in range(n_eta)],
                'true_hits_per_station': [[] for _ in range(n_eta)],
                'background_hits_per_station': [[] for _ in range(n_eta)],
                'total_hits_per_station': [[] for _ in range(n_eta)],
            },
            'phi': {
                'unique_stations': [[] for _ in range(n_phi)],
                'true_hits_per_track': [[] for _ in range(n_phi)],
                'true_hits_per_station': [[] for _ in range(n_phi)],
                'background_hits_per_station': [[] for _ in range(n_phi)],
                'total_hits_per_station': [[] for _ in range(n_phi)],
            },
            'pt': {
                'unique_stations': [[] for _ in range(n_pt)],
                'true_hits_per_track': [[] for _ in range(n_pt)],
                'true_hits_per_station': [[] for _ in range(n_pt)],
                'background_hits_per_station': [[] for _ in range(n_pt)],
                'total_hits_per_station': [[] for _ in range(n_pt)],
            },
        }
    
    def _process_event(self, event_idx: int, batch: Tuple, binned_data: Dict):
        """Process a single event and update statistics."""
        inputs, targets = batch
        
        # Extract data (remove batch dimension)
        hit_valid = targets["hit_valid"][0]
        particle_valid = targets["particle_valid"][0]
        particle_hit_valid = targets["particle_hit_valid"][0]  # Shape: [num_particles, num_hits]
        hit_on_valid_particle = targets["hit_on_valid_particle"][0]
        
        # Get station indices
        station_indices = inputs["hit_spacePoint_stationIndex"][0][hit_valid]
        
        # Get particle properties
        particle_eta = targets["particle_truthMuon_eta"][0]
        particle_phi = targets["particle_truthMuon_phi"][0]
        particle_pt = targets["particle_truthMuon_pt"][0]
        
        # Process each valid particle/track
        num_valid_particles = particle_valid.sum().item()
        
        for particle_idx in range(num_valid_particles):
            # Get hits for this particle
            hits_mask = particle_hit_valid[particle_idx]
            
            if hits_mask.sum() == 0:
                continue
            
            # Get station indices for this track
            track_station_indices = station_indices[hits_mask]
            unique_stations = torch.unique(track_station_indices)
            num_unique_stations = len(unique_stations)
            num_true_hits = hits_mask.sum().item()
            
            # Calculate hits per station for this track
            true_hits_per_station = []
            background_hits_per_station = []
            total_hits_per_station = []
            
            for station_idx in unique_stations:
                station_mask = station_indices == station_idx
                
                # True hits in this station from this track
                true_in_station = (hits_mask & station_mask).sum().item()
                true_hits_per_station.append(true_in_station)
                
                # Background hits in this station (hits not from this track)
                background_in_station = ((~hits_mask) & station_mask).sum().item()
                background_hits_per_station.append(background_in_station)
                
                # Total hits in this station
                total_in_station = station_mask.sum().item()
                total_hits_per_station.append(total_in_station)
            
            # Get particle kinematics
            eta = particle_eta[particle_idx].item()
            phi = particle_phi[particle_idx].item()
            pt = particle_pt[particle_idx].item()
            
            # Skip if any kinematic value is NaN
            if np.isnan(eta) or np.isnan(phi) or np.isnan(pt):
                continue
            
            # Update unbinned statistics
            self.unbinned_stats['num_unique_stations'].append(num_unique_stations)
            self.unbinned_stats['num_true_hits'].append(num_true_hits)
            self.unbinned_stats['num_background_hits'].extend(background_hits_per_station)
            self.unbinned_stats['num_total_hits'].extend(total_hits_per_station)
            
            # Find bins for this particle
            eta_bin = np.digitize(eta, self.eta_bins) - 1
            phi_bin = np.digitize(phi, self.phi_bins) - 1
            pt_bin = np.digitize(pt, self.pt_bins) - 1
            
            # Update binned statistics for eta
            if 0 <= eta_bin < len(self.eta_bins) - 1:
                binned_data['eta']['unique_stations'][eta_bin].append(num_unique_stations)
                binned_data['eta']['true_hits_per_track'][eta_bin].append(num_true_hits)
                binned_data['eta']['true_hits_per_station'][eta_bin].extend(true_hits_per_station)
                binned_data['eta']['background_hits_per_station'][eta_bin].extend(background_hits_per_station)
                binned_data['eta']['total_hits_per_station'][eta_bin].extend(total_hits_per_station)
            
            # Update binned statistics for phi
            if 0 <= phi_bin < len(self.phi_bins) - 1:
                binned_data['phi']['unique_stations'][phi_bin].append(num_unique_stations)
                binned_data['phi']['true_hits_per_track'][phi_bin].append(num_true_hits)
                binned_data['phi']['true_hits_per_station'][phi_bin].extend(true_hits_per_station)
                binned_data['phi']['background_hits_per_station'][phi_bin].extend(background_hits_per_station)
                binned_data['phi']['total_hits_per_station'][phi_bin].extend(total_hits_per_station)
            
            # Update binned statistics for pt
            if 0 <= pt_bin < len(self.pt_bins) - 1:
                binned_data['pt']['unique_stations'][pt_bin].append(num_unique_stations)
                binned_data['pt']['true_hits_per_track'][pt_bin].append(num_true_hits)
                binned_data['pt']['true_hits_per_station'][pt_bin].extend(true_hits_per_station)
                binned_data['pt']['background_hits_per_station'][pt_bin].extend(background_hits_per_station)
                binned_data['pt']['total_hits_per_station'][pt_bin].extend(total_hits_per_station)
    
    def _compute_statistics(self, binned_data: Dict) -> Dict:
        """Compute final statistics from accumulated data."""
        results = {
            'unbinned': {},
            'binned': {'eta': {}, 'phi': {}, 'pt': {}},
        }
        
        # Compute unbinned statistics
        for key, values in self.unbinned_stats.items():
            if len(values) > 0:
                results['unbinned'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            else:
                results['unbinned'][key] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
                }
        
        # Compute binned statistics
        for var_name in ['eta', 'phi', 'pt']:
            for metric_name in binned_data[var_name].keys():
                means = []
                stds = []
                for bin_data in binned_data[var_name][metric_name]:
                    if len(bin_data) > 0:
                        means.append(np.mean(bin_data))
                        stds.append(np.std(bin_data))
                    else:
                        means.append(0.0)
                        stds.append(0.0)
                
                results['binned'][var_name][metric_name] = {
                    'means': np.array(means),
                    'stds': np.array(stds),
                }
        
        return results
    
    def save_text_summary(self, results: Dict, output_path: Path):
        """Save text summary of unbinned statistics."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATION OCCUPATION ANALYSIS - UNBINNED STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Number of events analyzed: {self.num_events}\n\n")
            
            for key, stats in results['unbinned'].items():
                f.write(f"{key.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std Dev: {stats['std']:.4f}\n")
                f.write(f"  Median: {stats['median']:.4f}\n")
                f.write(f"  Min: {stats['min']:.4f}\n")
                f.write(f"  Max: {stats['max']:.4f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"Saved text summary to {output_path}")
    
    def plot_results(self, results: Dict, output_dir: Path):
        """Generate all plots."""
        print("\nGenerating plots...")
        
        # Plot 1: Unique stations per track vs eta, phi, pt
        self._plot_unique_stations(results, output_dir)
        
        # Plot 2: True hits per track vs eta, phi, pt
        self._plot_true_hits_per_track(results, output_dir)
        
        # Plot 3: Hits per station (3 panes: true, background, total) vs eta, phi, pt
        self._plot_hits_per_station(results, output_dir)
        
        # Plot 4: Distribution histograms with mean and std lines
        self._plot_distribution_histograms(output_dir)
        
        # Plot 5: Distribution histograms for MDT hits only
        self._plot_distribution_histograms_mdt(output_dir)
        
        print("All plots generated successfully!")
    
    def _plot_distribution_histograms(self, output_dir: Path):
        """Plot distributions of true hits with mean and std deviation lines."""
        # Collect statistics per event and per station
        avg_true_hits_per_station_per_event = []
        total_hits_per_station = []
        background_hits_per_station_with_true = []
        tracks_per_station = []
        
        # Get dataloader to recalculate statistics
        test_dataloader = self.datamodule.test_dataloader(shuffle=False)
        
        print("  Computing station statistics for distributions...")
        for event_idx, batch in enumerate(tqdm(test_dataloader, 
                                                desc="Processing for distributions",
                                                total=self.num_events)):
            if event_idx >= self.num_events:
                break
                
            inputs, targets = batch
            
            # Extract data (remove batch dimension)
            hit_valid = targets["hit_valid"][0]
            particle_valid = targets["particle_valid"][0]
            particle_hit_valid = targets["particle_hit_valid"][0]
            
            # Get station indices
            station_indices = inputs["hit_spacePoint_stationIndex"][0][hit_valid]
            
            # Get all unique stations in this event
            unique_stations = torch.unique(station_indices)
            
            # For each station, calculate statistics
            event_true_hits_per_station = []
            for station_idx in unique_stations:
                station_mask = station_indices == station_idx
                
                # Total hits in this station
                total_in_station = station_mask.sum().item()
                total_hits_per_station.append(total_in_station)
                
                # True hits in this station (from any valid particle)
                true_in_station = 0
                num_tracks_in_station = 0
                num_valid_particles = particle_valid.sum().item()
                for particle_idx in range(num_valid_particles):
                    hits_mask = particle_hit_valid[particle_idx]
                    true_hits_from_track = (hits_mask & station_mask).sum().item()
                    if true_hits_from_track > 0:
                        num_tracks_in_station += 1
                    true_in_station += true_hits_from_track
                
                # Record number of tracks per station
                tracks_per_station.append(num_tracks_in_station)
                
                # Only consider stations that contain true hits
                if true_in_station > 0:
                    event_true_hits_per_station.append(true_in_station)
                    
                    # Background hits in this station
                    background_in_station = total_in_station - true_in_station
                    background_hits_per_station_with_true.append(background_in_station)
            
            # Average true hits per station for this event (only stations with true hits)
            if len(event_true_hits_per_station) > 0:
                avg_true_hits_per_station_per_event.append(np.mean(event_true_hits_per_station))
        
        # Create figure with 2 rows and 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        
        # Row 1, Plot 1: Distribution of true hits per station per track (not averaged)
        ax = axes[0, 0]
        # We need to collect all true hits per station values (not averaged per track)
        all_true_hits_per_station = []
        
        # Recalculate to get all individual station hit counts
        test_dataloader = self.datamodule.test_dataloader(shuffle=False)
        for event_idx, batch in enumerate(test_dataloader):
            if event_idx >= self.num_events:
                break
                
            inputs, targets = batch
            hit_valid = targets["hit_valid"][0]
            particle_valid = targets["particle_valid"][0]
            particle_hit_valid = targets["particle_hit_valid"][0]
            station_indices = inputs["hit_spacePoint_stationIndex"][0][hit_valid]
            num_valid_particles = particle_valid.sum().item()
            
            for particle_idx in range(num_valid_particles):
                hits_mask = particle_hit_valid[particle_idx]
                if hits_mask.sum() == 0:
                    continue
                
                track_station_indices = station_indices[hits_mask]
                unique_stations = torch.unique(track_station_indices)
                
                for station_idx in unique_stations:
                    station_mask = station_indices == station_idx
                    true_in_station = (hits_mask & station_mask).sum().item()
                    all_true_hits_per_station.append(true_in_station)
        
        if len(all_true_hits_per_station) > 0:
            data = np.array(all_true_hits_per_station)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of True Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of True Hits per Station per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 2: Distribution of average true hits per station per event
        ax = axes[0, 1]
        if len(avg_true_hits_per_station_per_event) > 0:
            data = np.array(avg_true_hits_per_station_per_event)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create bins that align with integer values
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.linspace(min_val, max_val, min(50, max_val - min_val + 1))
            
            ax.hist(data, bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Average Number of True Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Avg True Hits per Station per Event\n(Stations with True Hits Only)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 3: Distribution of total true hits per track
        ax = axes[0, 2]
        if len(self.unbinned_stats['num_true_hits']) > 0:
            data = np.array(self.unbinned_stats['num_true_hits'])
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of True Hits per Track', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Total True Hits per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 4: Distribution of number of unique stations per track
        ax = axes[0, 3]
        if len(self.unbinned_stats['num_unique_stations']) > 0:
            data = np.array(self.unbinned_stats['num_unique_stations'])
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Unique Stations per Track', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Unique Stations per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 1: Distribution of total hits per station
        ax = axes[1, 0]
        if len(total_hits_per_station) > 0:
            data = np.array(total_hits_per_station)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='dodgerblue', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Total Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Total Hits per Station', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 2: Distribution of background hits in stations with true hits
        ax = axes[1, 1]
        if len(background_hits_per_station_with_true) > 0:
            data = np.array(background_hits_per_station_with_true)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='salmon', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Background Hits', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Background Hits per Station\n(Stations with True Hits Only)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 3: Distribution of tracks per station
        ax = axes[1, 2]
        if len(tracks_per_station) > 0:
            data = np.array(tracks_per_station)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5  # Center bins on integers
            
            ax.hist(data, bins=bins, color='gold', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Tracks per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Tracks per Station', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 4: Empty for now (can be used for future plots)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / 'true_hits_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def _plot_distribution_histograms_mdt(self, output_dir: Path):
        """Plot distributions for MDT (Monitored Drift Tubes) hits only (technology index 0)."""
        # Collect statistics per event and per station for MDT hits only
        avg_true_hits_per_station_per_event_mdt = []
        total_hits_per_station_mdt = []
        background_hits_per_station_with_true_mdt = []
        tracks_per_station_mdt = []
        num_true_hits_per_track_mdt = []
        num_unique_stations_per_track_mdt = []
        
        # Get dataloader to recalculate statistics
        test_dataloader = self.datamodule.test_dataloader(shuffle=False)
        
        print("  Computing MDT station statistics for distributions...")
        for event_idx, batch in enumerate(tqdm(test_dataloader, 
                                                desc="Processing MDT for distributions",
                                                total=self.num_events)):
            if event_idx >= self.num_events:
                break
                
            inputs, targets = batch
            
            # Extract data (remove batch dimension)
            hit_valid = targets["hit_valid"][0]
            particle_valid = targets["particle_valid"][0]
            particle_hit_valid = targets["particle_hit_valid"][0]
            
            # Get technology indices and filter for MDT (index 0)
            hit_technology = inputs["hit_spacePoint_technology"][0][hit_valid]
            mdt_mask = hit_technology == 0
            
            # Get station indices for MDT hits only
            station_indices = inputs["hit_spacePoint_stationIndex"][0][hit_valid]
            station_indices_mdt = station_indices[mdt_mask]
            
            if len(station_indices_mdt) == 0:
                continue
            
            # Get all unique MDT stations in this event
            unique_stations_mdt = torch.unique(station_indices_mdt)
            
            # For each MDT station, calculate statistics
            event_true_hits_per_station_mdt = []
            for station_idx in unique_stations_mdt:
                station_mask = station_indices == station_idx
                station_mask_mdt = mdt_mask & station_mask
                
                # Total MDT hits in this station
                total_in_station = station_mask_mdt.sum().item()
                total_hits_per_station_mdt.append(total_in_station)
                
                # True MDT hits in this station (from any valid particle)
                true_in_station = 0
                num_tracks_in_station = 0
                num_valid_particles = particle_valid.sum().item()
                for particle_idx in range(num_valid_particles):
                    hits_mask = particle_hit_valid[particle_idx]
                    true_hits_from_track = (hits_mask & station_mask_mdt).sum().item()
                    if true_hits_from_track > 0:
                        num_tracks_in_station += 1
                    true_in_station += true_hits_from_track
                
                # Record number of tracks per MDT station
                tracks_per_station_mdt.append(num_tracks_in_station)
                
                # Only consider stations that contain true hits
                if true_in_station > 0:
                    event_true_hits_per_station_mdt.append(true_in_station)
                    
                    # Background hits in this station
                    background_in_station = total_in_station - true_in_station
                    background_hits_per_station_with_true_mdt.append(background_in_station)
            
            # Average true MDT hits per station for this event (only stations with true hits)
            if len(event_true_hits_per_station_mdt) > 0:
                avg_true_hits_per_station_per_event_mdt.append(np.mean(event_true_hits_per_station_mdt))
            
            # Process each valid particle/track for MDT hits
            num_valid_particles = particle_valid.sum().item()
            for particle_idx in range(num_valid_particles):
                # Get hits for this particle
                hits_mask = particle_hit_valid[particle_idx]
                
                # Filter for MDT hits only
                hits_mask_mdt = hits_mask & mdt_mask
                
                if hits_mask_mdt.sum() == 0:
                    continue
                
                # Get MDT station indices for this track
                track_station_indices_mdt = station_indices[hits_mask_mdt]
                unique_stations = torch.unique(track_station_indices_mdt)
                num_unique_stations = len(unique_stations)
                num_true_hits = hits_mask_mdt.sum().item()
                
                num_true_hits_per_track_mdt.append(num_true_hits)
                num_unique_stations_per_track_mdt.append(num_unique_stations)
        
        # Create figure with 2 rows and 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        
        # Row 1, Plot 1: Distribution of true MDT hits per station per track (not averaged)
        ax = axes[0, 0]
        all_true_hits_per_station_mdt = []
        
        # Recalculate to get all individual MDT station hit counts
        test_dataloader = self.datamodule.test_dataloader(shuffle=False)
        for event_idx, batch in enumerate(test_dataloader):
            if event_idx >= self.num_events:
                break
                
            inputs, targets = batch
            hit_valid = targets["hit_valid"][0]
            particle_valid = targets["particle_valid"][0]
            particle_hit_valid = targets["particle_hit_valid"][0]
            
            hit_technology = inputs["hit_spacePoint_technology"][0][hit_valid]
            mdt_mask = hit_technology == 0
            
            station_indices = inputs["hit_spacePoint_stationIndex"][0][hit_valid]
            num_valid_particles = particle_valid.sum().item()
            
            for particle_idx in range(num_valid_particles):
                hits_mask = particle_hit_valid[particle_idx]
                hits_mask_mdt = hits_mask & mdt_mask
                
                if hits_mask_mdt.sum() == 0:
                    continue
                
                track_station_indices_mdt = station_indices[hits_mask_mdt]
                unique_stations = torch.unique(track_station_indices_mdt)
                
                for station_idx in unique_stations:
                    station_mask = station_indices == station_idx
                    station_mask_mdt = mdt_mask & station_mask
                    true_in_station = (hits_mask & station_mask_mdt).sum().item()
                    all_true_hits_per_station_mdt.append(true_in_station)
        
        if len(all_true_hits_per_station_mdt) > 0:
            data = np.array(all_true_hits_per_station_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Create integer bins
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of True MDT Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of True MDT Hits per Station per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 2: Distribution of average true MDT hits per station per event
        ax = axes[0, 1]
        if len(avg_true_hits_per_station_per_event_mdt) > 0:
            data = np.array(avg_true_hits_per_station_per_event_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.linspace(min_val, max_val, min(50, max_val - min_val + 1))
            
            ax.hist(data, bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Average Number of True MDT Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Avg True MDT Hits per Station per Event\n(Stations with True Hits Only)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 3: Distribution of total true MDT hits per track
        ax = axes[0, 2]
        if len(num_true_hits_per_track_mdt) > 0:
            data = np.array(num_true_hits_per_track_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of True MDT Hits per Track', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Total True MDT Hits per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 1, Plot 4: Distribution of number of unique MDT stations per track
        ax = axes[0, 3]
        if len(num_unique_stations_per_track_mdt) > 0:
            data = np.array(num_unique_stations_per_track_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Unique MDT Stations per Track', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Unique MDT Stations per Track', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 1: Distribution of total MDT hits per station
        ax = axes[1, 0]
        if len(total_hits_per_station_mdt) > 0:
            data = np.array(total_hits_per_station_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='dodgerblue', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Total MDT Hits per Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Total MDT Hits per Station', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 2: Distribution of background MDT hits in stations with true hits
        ax = axes[1, 1]
        if len(background_hits_per_station_with_true_mdt) > 0:
            data = np.array(background_hits_per_station_with_true_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='salmon', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Background MDT Hits', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Background MDT Hits per Station\n(Stations with True Hits Only)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 3: Distribution of tracks per MDT station
        ax = axes[1, 2]
        if len(tracks_per_station_mdt) > 0:
            data = np.array(tracks_per_station_mdt)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            min_val = int(np.floor(data.min()))
            max_val = int(np.ceil(data.max()))
            bins = np.arange(min_val, max_val + 2) - 0.5
            
            ax.hist(data, bins=bins, color='gold', edgecolor='black', alpha=0.7)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2, label=f'Mean ± Std')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2)
            ax.axvline(mean_val - 2*std_val, color='green', linestyle=':', linewidth=2, label=f'Mean ± 2×Std')
            ax.axvline(mean_val + 2*std_val, color='green', linestyle=':', linewidth=2)
            
            ax.set_xlabel('Number of Tracks per MDT Station', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Tracks per MDT Station', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
        
        # Row 2, Plot 4: Empty for now
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / 'true_hits_distributions_MDT.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def _plot_unique_stations(self, results: Dict, output_dir: Path):
        """Plot average number of unique stations per track."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        variables = [
            ('eta', self.eta_bins, r'$\eta$'),
            ('phi', self.phi_bins, r'$\phi$ [rad]'),
            ('pt', self.pt_bins, r'$p_T$ [GeV]')
        ]
        
        for ax, (var_name, bins, label) in zip(axes, variables):
            means = results['binned'][var_name]['unique_stations']['means']
            
            # Use histogram-style step plot
            color = 'blue'
            for i, (lhs, rhs, mean_val) in enumerate(zip(bins[:-1], bins[1:], means)):
                # Step plot
                ax.step([lhs, rhs], [mean_val, mean_val], 
                       color=color, linewidth=2.5,
                       label="Unique Stations" if i == 0 else "")
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Avg. Number of Unique Stations', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Unique Stations vs {label}', fontsize=13)
            ax.legend(loc='best')
        
        plt.tight_layout()
        output_path = output_dir / 'unique_stations_per_track.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def _plot_true_hits_per_track(self, results: Dict, output_dir: Path):
        """Plot number of true hits per track."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        variables = [
            ('eta', self.eta_bins, r'$\eta$'),
            ('phi', self.phi_bins, r'$\phi$ [rad]'),
            ('pt', self.pt_bins, r'$p_T$ [GeV]')
        ]
        
        for ax, (var_name, bins, label) in zip(axes, variables):
            means = results['binned'][var_name]['true_hits_per_track']['means']
            
            # Use histogram-style step plot
            color = 'green'
            for i, (lhs, rhs, mean_val) in enumerate(zip(bins[:-1], bins[1:], means)):
                # Step plot
                ax.step([lhs, rhs], [mean_val, mean_val], 
                       color=color, linewidth=2.5,
                       label="True Hits per Track" if i == 0 else "")
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Avg. Number of True Hits', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'True Hits per Track vs {label}', fontsize=13)
            ax.legend(loc='best')
        
        plt.tight_layout()
        output_path = output_dir / 'true_hits_per_track.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def _plot_hits_per_station(self, results: Dict, output_dir: Path):
        """Plot hits per station (true, background, total) for each variable."""
        variables = [
            ('eta', self.eta_bins, r'$\eta$'),
            ('phi', self.phi_bins, r'$\phi$ [rad]'),
            ('pt', self.pt_bins, r'$p_T$ [GeV]')
        ]
        
        for var_name, bins, label in variables:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # True hits per station
            means_true = results['binned'][var_name]['true_hits_per_station']['means']
            color_true = 'green'
            for i, (lhs, rhs, mean_val) in enumerate(zip(bins[:-1], bins[1:], means_true)):
                axes[0].step([lhs, rhs], [mean_val, mean_val], 
                           color=color_true, linewidth=2.5,
                           label="True Hits" if i == 0 else "")
            axes[0].set_xlabel(label, fontsize=12)
            axes[0].set_ylabel('Avg. True Hits per Station', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('True Hits per Station', fontsize=13)
            axes[0].legend(loc='best')
            
            # Background hits per station
            means_bg = results['binned'][var_name]['background_hits_per_station']['means']
            color_bg = 'red'
            for i, (lhs, rhs, mean_val) in enumerate(zip(bins[:-1], bins[1:], means_bg)):
                axes[1].step([lhs, rhs], [mean_val, mean_val], 
                           color=color_bg, linewidth=2.5,
                           label="Background Hits" if i == 0 else "")
            axes[1].set_xlabel(label, fontsize=12)
            axes[1].set_ylabel('Avg. Background Hits per Station', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Background Hits per Station', fontsize=13)
            axes[1].legend(loc='best')
            
            # Total hits per station
            means_total = results['binned'][var_name]['total_hits_per_station']['means']
            color_total = 'blue'
            for i, (lhs, rhs, mean_val) in enumerate(zip(bins[:-1], bins[1:], means_total)):
                axes[2].step([lhs, rhs], [mean_val, mean_val], 
                           color=color_total, linewidth=2.5,
                           label="Total Hits" if i == 0 else "")
            axes[2].set_xlabel(label, fontsize=12)
            axes[2].set_ylabel('Avg. Total Hits per Station', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Total Hits per Station', fontsize=13)
            axes[2].legend(loc='best')
            
            plt.tight_layout()
            output_path = output_dir / f'hits_per_station_vs_{var_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path}")


def main() -> None:
    """Main function to run the station occupation analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze station occupation statistics for ATLAS Muon tracking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./station_occupation_analysis",
        help="Base directory for output (default: ./station_occupation_analysis)",
    )
    parser.add_argument(
        "--num-events",
        "-n",
        type=int,
        default=-1,
        help="Number of events to analyze (-1 for all available)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
        help="Path to configuration YAML file",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    inputs, targets = load_inputs_targets_from_config(args.config_path)
    
    # Create fresh copies for the datamodule
    inputs_copy = {k: list(v) for k, v in inputs.items()}
    targets_copy = {k: list(v) for k, v in targets.items()}
    
    # Initialize datamodule
    print("\nInitializing data module...")
    datamodule = AtlasMuonDataModule(
        train_dir=H5_FILEPATH,
        val_dir=H5_FILEPATH,
        test_dir=H5_FILEPATH,
        num_workers=10,
        num_train=-1,
        num_val=-1,
        num_test=-1,
        batch_size=1,
        inputs=inputs_copy,
        targets=targets_copy,
        hit_eval_train=HIT_EVAL_FILEPATH,
        hit_eval_val=HIT_EVAL_FILEPATH,
        hit_eval_test=HIT_EVAL_FILEPATH,
    )
    
    # Initialize dataset
    dataset = AtlasMuonDataset(
        dirpath=H5_FILEPATH,
        inputs=inputs,
        targets=targets,
        hit_eval_path=HIT_EVAL_FILEPATH,
    )
    
    datamodule.setup("test")
    
    # Create analyzer
    analyzer = StationOccupationAnalyzer(
        datamodule=datamodule,
        dataset=dataset,
        num_events=args.num_events,
    )
    
    # Run analysis
    results = analyzer.analyze()
    
    # Save text summary
    text_output_path = output_dir / "station_occupation_summary.txt"
    analyzer.save_text_summary(results, text_output_path)
    
    # Generate plots
    analyzer.plot_results(results, output_dir)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
