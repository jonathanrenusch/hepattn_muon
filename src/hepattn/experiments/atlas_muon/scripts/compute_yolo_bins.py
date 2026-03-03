#!/usr/bin/env python3
"""Compute YOLO-style classification bins for track parameter regression.

This script iterates over the entire training dataset to:
1. Compute min/max/statistics for pt, eta, phi distributions
2. Compute innermost hit seed precision for eta and phi (std of residual)
3. Create quantile-based binning arrays for pt (equal sample counts per bin)
4. Create uniform binning arrays for eta and phi
5. Save all bin arrays and statistics to the dataset directory
6. Generate histogram plots for visualization

Usage:
    cd hepattn_muon/src
    pixi run python hepattn/experiments/atlas_muon/scripts/compute_yolo_bins.py \
        --data_dir /scratch/ml_training_data_2694000_regression_true_hits_only \
        --output_dir /scratch/ml_training_data_2694000_regression_true_hits_only/bins

The bins will be saved as:
    - pt_bins_{n}.npy: Quantile-based bin edges for pt
    - eta_bins_{n}.npy: Uniform bin edges for eta
    - phi_bins_{n}.npy: Uniform bin edges for phi
    - statistics.txt: Summary statistics including innermost hit precision
    - histograms/: Directory containing visualization plots
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


def angular_difference(phi_pred: np.ndarray, phi_true: np.ndarray) -> np.ndarray:
    """Compute signed angular difference handling periodicity.
    
    Returns difference in [-pi, pi] range.
    """
    diff = phi_pred - phi_true
    return np.arctan2(np.sin(diff), np.cos(diff))


def load_metadata(data_dir: Path) -> dict:
    """Load dataset metadata."""
    with open(data_dir / 'metadata.yaml', 'r') as f:
        return yaml.safe_load(f)


def iterate_tracks(data_dir: Path, metadata: dict, max_events: int = -1):
    """Generator that yields track data from all events.
    
    Yields dictionaries with:
        - pt: truth pT
        - eta: truth eta
        - phi: truth phi
        - charge: truth charge
        - innermost_hit_eta: eta of innermost hit (smallest r)
        - innermost_hit_phi: phi of innermost hit (smallest r)
    """
    file_indices = np.load(data_dir / 'event_file_indices.npy')
    row_indices = np.load(data_dir / 'event_row_indices.npy')
    
    hit_features = metadata['hit_features']
    track_features = metadata['track_features']
    
    # Find indices for features we need
    posX_idx = hit_features.index('spacePoint_globPosX')
    posY_idx = hit_features.index('spacePoint_globPosY')
    posZ_idx = hit_features.index('spacePoint_globPosZ')
    truth_link_idx = hit_features.index('spacePoint_truthLink')
    
    pt_idx = track_features.index('truthMuon_pt')
    eta_idx = track_features.index('truthMuon_eta')
    phi_idx = track_features.index('truthMuon_phi')
    charge_idx = track_features.index('truthMuon_q')
    
    # Get chunk info
    chunk_summary = metadata['event_mapping']['chunk_summary']
    
    num_events = len(row_indices) if max_events == -1 else min(max_events, len(row_indices))
    
    # Cache for open file handles per chunk
    current_chunk_idx = -1
    current_h5 = None
    
    for event_idx in tqdm(range(num_events), desc="Processing events"):
        file_idx = file_indices[event_idx]
        row_idx = row_indices[event_idx]
        
        # Open new file if needed
        if file_idx != current_chunk_idx:
            if current_h5 is not None:
                current_h5.close()
            h5_path = data_dir / chunk_summary[file_idx]['h5_file']
            current_h5 = h5py.File(h5_path, 'r')
            current_chunk_idx = file_idx
        
        # Load event data
        num_hits = current_h5['num_hits'][row_idx]
        num_tracks = current_h5['num_tracks'][row_idx]
        
        if num_tracks == 0:
            continue
            
        hits_array = current_h5['hits'][row_idx, :num_hits]
        tracks_array = current_h5['tracks'][row_idx, :num_tracks]
        
        # Get hit positions (in mm, will convert to m like in data.py for consistency)
        posX = hits_array[:, posX_idx] * 0.001  # mm -> m scale as in data.py
        posY = hits_array[:, posY_idx] * 0.001
        posZ = hits_array[:, posZ_idx] * 0.001
        truth_links = hits_array[:, truth_link_idx].astype(int)
        
        # Compute derived hit quantities
        r = np.sqrt(posX**2 + posY**2)
        s = np.sqrt(posX**2 + posY**2 + posZ**2)
        hit_theta = np.arccos(np.clip(posZ / s, -1, 1))
        hit_phi = np.arctan2(posY, posX)
        hit_eta = -np.log(np.tan(hit_theta / 2.0))
        
        # Process each track
        for track_idx in range(num_tracks):
            pt = tracks_array[track_idx, pt_idx]
            eta = tracks_array[track_idx, eta_idx]
            phi = tracks_array[track_idx, phi_idx]
            charge = tracks_array[track_idx, charge_idx]
            
            # Find hits belonging to this track
            # In this dataset, truth_links contain particle indices
            track_hit_mask = truth_links == track_idx
            
            if not np.any(track_hit_mask):
                continue
            
            # Find innermost hit (smallest r)
            track_r = r[track_hit_mask]
            track_eta = hit_eta[track_hit_mask]
            track_phi = hit_phi[track_hit_mask]
            
            innermost_idx = np.argmin(track_r)
            innermost_eta = track_eta[innermost_idx]
            innermost_phi = track_phi[innermost_idx]
            
            yield {
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'charge': charge,
                'innermost_hit_eta': innermost_eta,
                'innermost_hit_phi': innermost_phi,
                'num_hits': np.sum(track_hit_mask),
            }
    
    if current_h5 is not None:
        current_h5.close()


def compute_quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute quantile-based bin edges for equal sample counts per bin.
    
    Returns array of bin edges of length (num_bins + 1).
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(values, percentiles)
    return bin_edges


def compute_log_bins(min_val: float, max_val: float, num_bins: int) -> np.ndarray:
    """Compute logarithmically-spaced bin edges.
    
    This provides finer resolution at low values and coarser resolution at high values,
    which is appropriate for pt where resolution typically scales with pt.
    
    Returns array of bin edges of length (num_bins + 1).
    """
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_edges = np.linspace(log_min, log_max, num_bins + 1)
    return np.exp(log_edges)


def compute_uniform_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute uniform bin edges.
    
    Returns array of bin edges of length (num_bins + 1).
    """
    return np.linspace(np.min(values), np.max(values), num_bins + 1)


def compute_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Compute bin centers from bin edges."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def create_histograms(all_pt, all_eta, all_phi, all_charge, 
                      eta_residuals, phi_residuals, output_dir, all_inv_pt=None):
    """Create and save histogram plots."""
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING HISTOGRAM PLOTS")
    print("="*60)
    
    # 1. PT distribution (log scale on y-axis)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_pt, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('pT [GeV]', fontsize=12)
    ax.set_ylabel('Count (log scale)', fontsize=12)
    ax.set_title(f'pT Distribution (N = {len(all_pt):,} tracks)', fontsize=14)
    ax.axvline(np.median(all_pt), color='red', linestyle='--', label=f'Median: {np.median(all_pt):.1f} GeV')
    ax.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / 'pt_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'pt_distribution.png'}")
    
    # 2. Log PT distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.log(all_pt), bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('log(pT) [log GeV]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'log(pT) Distribution (N = {len(all_pt):,} tracks)', fontsize=14)
    plt.tight_layout()
    plt.savefig(hist_dir / 'log_pt_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'log_pt_distribution.png'}")
    
    # 3. Eta distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_eta, bins=100, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('η', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'η Distribution (N = {len(all_eta):,} tracks)', fontsize=14)
    ax.axvline(np.mean(all_eta), color='red', linestyle='--', label=f'Mean: {np.mean(all_eta):.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / 'eta_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'eta_distribution.png'}")
    
    # 4. Phi distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_phi, bins=100, color='darkorange', edgecolor='black', alpha=0.7)
    ax.set_xlabel('φ [rad]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'φ Distribution (N = {len(all_phi):,} tracks)', fontsize=14)
    plt.tight_layout()
    plt.savefig(hist_dir / 'phi_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'phi_distribution.png'}")
    
    # 5. Charge distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    charges, counts = np.unique(all_charge, return_counts=True)
    ax.bar(charges, counts, color=['crimson', 'steelblue'], edgecolor='black', width=0.5)
    ax.set_xlabel('Charge', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Charge Distribution (N = {len(all_charge):,} tracks)', fontsize=14)
    ax.set_xticks([-1, 1])
    for c, cnt in zip(charges, counts):
        ax.annotate(f'{cnt:,}\n({100*cnt/len(all_charge):.1f}%)', 
                   xy=(c, cnt), ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(hist_dir / 'charge_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'charge_distribution.png'}")
    
    # 6. Eta seed residuals (innermost hit - truth) - LOG SCALE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(eta_residuals * 1000, bins=100, color='purple', edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('η residual (innermost hit - truth) [mrad]', fontsize=12)
    ax.set_ylabel('Count (log scale)', fontsize=12)
    ax.set_title(f'Innermost Hit η Seed Precision\nStd: {np.std(eta_residuals)*1000:.2f} mrad', fontsize=14)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(hist_dir / 'eta_seed_residuals.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'eta_seed_residuals.png'}")
    
    # 7. Phi seed residuals (innermost hit - truth) - LOG SCALE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(phi_residuals * 1000, bins=100, color='teal', edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('φ residual (innermost hit - truth) [mrad]', fontsize=12)
    ax.set_ylabel('Count (log scale)', fontsize=12)
    ax.set_title(f'Innermost Hit φ Seed Precision\nStd: {np.std(phi_residuals)*1000:.2f} mrad', fontsize=14)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(hist_dir / 'phi_seed_residuals.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'phi_seed_residuals.png'}")
    
    # 8. 1/PT distribution (if provided)
    if all_inv_pt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_inv_pt, bins=100, color='darkred', edgecolor='black', alpha=0.7)
        ax.set_xlabel('1/pT [1/GeV]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'1/pT Distribution (N = {len(all_inv_pt):,} tracks)', fontsize=14)
        ax.axvline(np.median(all_inv_pt), color='blue', linestyle='--', 
                   label=f'Median: {np.median(all_inv_pt):.4f} 1/GeV')
        ax.legend()
        plt.tight_layout()
        plt.savefig(hist_dir / 'inv_pt_distribution.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / 'inv_pt_distribution.png'}")
        
        # 1/PT distribution (log y-axis)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_inv_pt, bins=100, color='darkred', edgecolor='black', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel('1/pT [1/GeV]', fontsize=12)
        ax.set_ylabel('Count (log scale)', fontsize=12)
        ax.set_title(f'1/pT Distribution - Log Scale (N = {len(all_inv_pt):,} tracks)', fontsize=14)
        ax.axvline(np.median(all_inv_pt), color='blue', linestyle='--', 
                   label=f'Median: {np.median(all_inv_pt):.4f} 1/GeV')
        ax.legend()
        plt.tight_layout()
        plt.savefig(hist_dir / 'inv_pt_distribution_logy.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / 'inv_pt_distribution_logy.png'}")
        
        # 1/PT vs PT comparison side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].hist(all_pt, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('pT [GeV]', fontsize=12)
        axes[0].set_ylabel('Count (log)', fontsize=12)
        axes[0].set_title('pT Distribution', fontsize=13)
        
        axes[1].hist(all_inv_pt, bins=100, color='darkred', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('1/pT [1/GeV]', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('1/pT Distribution (curvature proxy)', fontsize=13)
        
        plt.suptitle(f'pT vs 1/pT Distributions (N = {len(all_pt):,} tracks)', fontsize=14)
        plt.tight_layout()
        plt.savefig(hist_dir / 'pt_vs_inv_pt_comparison.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / 'pt_vs_inv_pt_comparison.png'}")
    
    # 9. Combined 2x2 summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PT (log y)
    axes[0, 0].hist(all_pt, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('pT [GeV]')
    axes[0, 0].set_ylabel('Count (log)')
    axes[0, 0].set_title('pT Distribution')
    
    # Eta
    axes[0, 1].hist(all_eta, bins=100, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('η')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('η Distribution')
    
    # Phi
    axes[1, 0].hist(all_phi, bins=100, color='darkorange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('φ [rad]')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('φ Distribution')
    
    # Seed residuals combined (log scale)
    axes[1, 1].hist(eta_residuals * 1000, bins=100, alpha=0.6, 
                    label=f'η (std={np.std(eta_residuals)*1000:.1f} mrad)', color='purple')
    axes[1, 1].hist(phi_residuals * 1000, bins=100, alpha=0.6, 
                    label=f'φ (std={np.std(phi_residuals)*1000:.1f} mrad)', color='teal')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Residual [mrad]')
    axes[1, 1].set_ylabel('Count (log)')
    axes[1, 1].set_title('Innermost Hit Seed Residuals')
    axes[1, 1].legend()
    
    plt.suptitle(f'ATLAS Muon Track Parameter Distributions (N = {len(all_pt):,} tracks)', fontsize=14)
    plt.tight_layout()
    plt.savefig(hist_dir / 'summary_distributions.png', dpi=150)
    plt.close()
    print(f"  Saved: {hist_dir / 'summary_distributions.png'}")
    
    print(f"\nAll histograms saved to: {hist_dir}")


def create_bin_resolution_plots(all_pt, all_eta, output_dir, pt_bin_counts, eta_bin_counts):
    """Create visualization plots showing bin boundaries on distributions.
    
    This shows vertical markers for each bin edge overlaid on the distribution
    histograms, giving an intuitive sense of the resolution at different bin counts.
    """
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING BIN RESOLUTION VISUALIZATION PLOTS")
    print("="*60)
    
    PT_MIN, PT_MAX = 5.0, 200.0
    ETA_MIN, ETA_MAX = -2.7, 2.7
    ETA_GAP_MIN, ETA_GAP_MAX = -0.1, 0.1
    
    # Filter PT for binning
    pt_for_binning = all_pt[(all_pt >= PT_MIN) & (all_pt <= PT_MAX)]
    
    # Create PT resolution plots for each bin count
    for num_bins in pt_bin_counts:
        pt_bins = compute_quantile_bins(pt_for_binning, num_bins)
        pt_bins[0] = PT_MIN
        pt_bins[-1] = PT_MAX
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot histogram
        counts, edges, _ = ax.hist(all_pt, bins=200, color='steelblue', 
                                   edgecolor='none', alpha=0.6, label='pT distribution')
        ax.set_yscale('log')
        
        # Add vertical lines for bin edges
        for i, edge in enumerate(pt_bins):
            alpha = 0.8 if i in [0, len(pt_bins)-1] else 0.4
            linewidth = 2 if i in [0, len(pt_bins)-1] else 0.8
            color = 'red' if i in [0, len(pt_bins)-1] else 'darkred'
            ax.axvline(edge, color=color, linestyle='-', alpha=alpha, linewidth=linewidth)
        
        # Compute bin width statistics
        bin_widths = np.diff(pt_bins)
        
        ax.set_xlabel('pT [GeV]', fontsize=12)
        ax.set_ylabel('Count (log scale)', fontsize=12)
        ax.set_title(f'pT Distribution with {num_bins} Quantile Bins\n'
                    f'Bin widths: min={np.min(bin_widths):.2f}, max={np.max(bin_widths):.2f}, '
                    f'median={np.median(bin_widths):.2f} GeV', fontsize=13)
        ax.set_xlim(0, PT_MAX + 10)
        
        # Add legend with bin info
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='steelblue', alpha=0.6, lw=10, label='pT distribution'),
            Line2D([0], [0], color='darkred', lw=1.5, label=f'{num_bins} quantile bin edges'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(hist_dir / f'pt_bins_{num_bins}_resolution.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / f'pt_bins_{num_bins}_resolution.png'}")
    
    # Create Eta resolution plots for each bin count
    for num_bins in eta_bin_counts:
        # Split bins between negative and positive eta regions
        eta_neg = all_eta[(all_eta >= ETA_MIN) & (all_eta <= ETA_GAP_MIN)]
        eta_pos = all_eta[(all_eta >= ETA_GAP_MAX) & (all_eta <= ETA_MAX)]
        
        neg_frac = len(eta_neg) / (len(eta_neg) + len(eta_pos))
        neg_bins = max(1, int(num_bins * neg_frac))
        pos_bins = num_bins - neg_bins
        
        eta_bins_neg = compute_quantile_bins(eta_neg, neg_bins)
        eta_bins_pos = compute_quantile_bins(eta_pos, pos_bins)
        
        eta_bins_neg[0] = ETA_MIN
        eta_bins_neg[-1] = ETA_GAP_MIN
        eta_bins_pos[0] = ETA_GAP_MAX
        eta_bins_pos[-1] = ETA_MAX
        
        eta_bins = np.concatenate([eta_bins_neg, eta_bins_pos])
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot histogram
        counts, edges, _ = ax.hist(all_eta, bins=200, color='forestgreen', 
                                   edgecolor='none', alpha=0.6)
        
        # Add vertical lines for bin edges (skip gap region for better visibility)
        for i, edge in enumerate(eta_bins):
            # Highlight boundaries and gap edges
            is_boundary = i in [0, len(eta_bins)-1]
            is_gap_edge = (np.abs(edge - ETA_GAP_MIN) < 0.001 or 
                          np.abs(edge - ETA_GAP_MAX) < 0.001)
            
            if is_boundary:
                ax.axvline(edge, color='red', linestyle='-', alpha=0.8, linewidth=2)
            elif is_gap_edge:
                ax.axvline(edge, color='orange', linestyle='--', alpha=0.8, linewidth=1.5)
            else:
                ax.axvline(edge, color='darkgreen', linestyle='-', alpha=0.3, linewidth=0.6)
        
        # Add shaded region for gap
        ax.axvspan(ETA_GAP_MIN, ETA_GAP_MAX, color='gray', alpha=0.3, label='Barrel gap')
        
        # Compute bin width statistics (excluding gap)
        bin_widths = np.diff(eta_bins)
        real_widths = bin_widths[bin_widths < 0.15]  # Exclude gap
        
        ax.set_xlabel('η', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'η Distribution with {num_bins} Quantile Bins ({neg_bins} neg + {pos_bins} pos)\n'
                    f'Bin widths (excl. gap): min={np.min(real_widths)*1000:.2f}, '
                    f'max={np.max(real_widths)*1000:.2f} mrad', fontsize=13)
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='forestgreen', alpha=0.6, label='η distribution'),
            Line2D([0], [0], color='darkgreen', lw=1, label=f'{num_bins}+1 quantile bin edges'),
            Patch(facecolor='gray', alpha=0.3, label='Barrel gap (no tracks)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(hist_dir / f'eta_bins_{num_bins}_resolution.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / f'eta_bins_{num_bins}_resolution.png'}")
    
    print(f"\nBin resolution plots saved to: {hist_dir}")


def create_log_pt_bin_resolution_plots(all_pt, output_dir, pt_bin_counts):
    """Create visualization plots showing logarithmic pt bin boundaries on distributions.
    
    Similar to quantile bins but with log-spaced bin edges, providing finer
    resolution at low pt and coarser resolution at high pt.
    """
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING LOGARITHMIC PT BIN RESOLUTION PLOTS")
    print("="*60)
    
    PT_MIN, PT_MAX = 5.0, 200.0
    
    for num_bins in pt_bin_counts:
        pt_bins = compute_log_bins(PT_MIN, PT_MAX, num_bins)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot histogram
        counts, edges, _ = ax.hist(all_pt, bins=200, color='steelblue', 
                                   edgecolor='none', alpha=0.6, label='pT distribution')
        ax.set_yscale('log')
        
        # Add vertical lines for bin edges
        for i, edge in enumerate(pt_bins):
            alpha = 0.8 if i in [0, len(pt_bins)-1] else 0.4
            linewidth = 2 if i in [0, len(pt_bins)-1] else 0.8
            color = 'darkgreen' if i in [0, len(pt_bins)-1] else 'green'
            ax.axvline(edge, color=color, linestyle='-', alpha=alpha, linewidth=linewidth)
        
        # Compute bin width statistics
        bin_widths = np.diff(pt_bins)
        
        ax.set_xlabel('pT [GeV]', fontsize=12)
        ax.set_ylabel('Count (log scale)', fontsize=12)
        ax.set_title(f'pT Distribution with {num_bins} Logarithmic Bins\n'
                    f'Bin widths: low pT={bin_widths[0]:.2f} GeV, high pT={bin_widths[-1]:.2f} GeV, '
                    f'ratio={bin_widths[-1]/bin_widths[0]:.1f}x', fontsize=13)
        ax.set_xlim(0, PT_MAX + 10)
        
        # Add legend with bin info
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='steelblue', alpha=0.6, lw=10, label='pT distribution'),
            Line2D([0], [0], color='green', lw=1.5, label=f'{num_bins} logarithmic bin edges'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(hist_dir / f'pt_bins_{num_bins}_log_resolution.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / f'pt_bins_{num_bins}_log_resolution.png'}")
    
    # Create comparison plot: quantile vs log for same bin count
    for num_bins in pt_bin_counts:
        pt_for_binning = all_pt[(all_pt >= PT_MIN) & (all_pt <= PT_MAX)]
        pt_bins_quantile = compute_quantile_bins(pt_for_binning, num_bins)
        pt_bins_quantile[0] = PT_MIN
        pt_bins_quantile[-1] = PT_MAX
        pt_bins_log = compute_log_bins(PT_MIN, PT_MAX, num_bins)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Left: Quantile bins
        axes[0].hist(all_pt, bins=200, color='steelblue', edgecolor='none', alpha=0.6)
        axes[0].set_yscale('log')
        for i, edge in enumerate(pt_bins_quantile):
            alpha = 0.8 if i in [0, len(pt_bins_quantile)-1] else 0.4
            axes[0].axvline(edge, color='darkred', linestyle='-', alpha=alpha, linewidth=0.8)
        widths_q = np.diff(pt_bins_quantile)
        axes[0].set_xlabel('pT [GeV]', fontsize=12)
        axes[0].set_ylabel('Count (log scale)', fontsize=12)
        axes[0].set_title(f'Quantile Bins ({num_bins})\n'
                         f'Width: {np.min(widths_q):.2f}-{np.max(widths_q):.2f} GeV', fontsize=13)
        axes[0].set_xlim(0, PT_MAX + 10)
        
        # Right: Log bins
        axes[1].hist(all_pt, bins=200, color='steelblue', edgecolor='none', alpha=0.6)
        axes[1].set_yscale('log')
        for i, edge in enumerate(pt_bins_log):
            alpha = 0.8 if i in [0, len(pt_bins_log)-1] else 0.4
            axes[1].axvline(edge, color='darkgreen', linestyle='-', alpha=alpha, linewidth=0.8)
        widths_l = np.diff(pt_bins_log)
        axes[1].set_xlabel('pT [GeV]', fontsize=12)
        axes[1].set_ylabel('Count (log scale)', fontsize=12)
        axes[1].set_title(f'Logarithmic Bins ({num_bins})\n'
                         f'Width: {widths_l[0]:.2f} (low) → {widths_l[-1]:.2f} (high) GeV', fontsize=13)
        axes[1].set_xlim(0, PT_MAX + 10)
        
        plt.suptitle(f'pT Binning Comparison: Quantile vs Logarithmic ({num_bins} bins)', fontsize=14)
        plt.tight_layout()
        plt.savefig(hist_dir / f'pt_bins_{num_bins}_quantile_vs_log.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / f'pt_bins_{num_bins}_quantile_vs_log.png'}")
    
    print(f"\nLogarithmic pT bin plots saved to: {hist_dir}")


def create_inv_pt_bin_resolution_plots(all_inv_pt, output_dir, inv_pt_bin_counts):
    """Create visualization plots showing 1/pt bin boundaries on distributions.
    
    The detector measures curvature ∝ 1/pt directly, so uniform binning in 1/pt
    should provide more natural resolution for the detector's measurement capability.
    """
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING 1/PT BIN RESOLUTION PLOTS")
    print("="*60)
    
    INV_PT_MIN = 1.0 / 200.0  # = 0.005 (from PT_MAX)
    INV_PT_MAX = 1.0 / 5.0    # = 0.2   (from PT_MIN)
    
    for num_bins in inv_pt_bin_counts:
        # Uniform bins in 1/pt space
        inv_pt_bins = np.linspace(INV_PT_MIN, INV_PT_MAX, num_bins + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Left: 1/pt distribution with bins
        counts, edges, _ = axes[0].hist(all_inv_pt, bins=200, color='darkred',
                                        edgecolor='none', alpha=0.6)
        for i, edge in enumerate(inv_pt_bins):
            alpha_val = 0.8 if i in [0, len(inv_pt_bins)-1] else 0.4
            linewidth = 2 if i in [0, len(inv_pt_bins)-1] else 0.8
            color = 'navy' if i in [0, len(inv_pt_bins)-1] else 'blue'
            axes[0].axvline(edge, color=color, linestyle='-', alpha=alpha_val, linewidth=linewidth)
        
        bin_widths = np.diff(inv_pt_bins)
        axes[0].set_xlabel('1/pT [1/GeV]', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'1/pT with {num_bins} Uniform Bins\n'
                         f'Bin width: {bin_widths[0]*1000:.2f} × 10⁻³ 1/GeV', fontsize=13)
        
        # Right: Same bins mapped back to pt space  
        pt_bin_edges = 1.0 / inv_pt_bins[::-1]  # Invert and reverse for increasing pt
        all_pt = 1.0 / all_inv_pt
        counts, edges, _ = axes[1].hist(all_pt, bins=200, color='steelblue',
                                        edgecolor='none', alpha=0.6)
        axes[1].set_yscale('log')
        for i, edge in enumerate(pt_bin_edges):
            alpha_val = 0.8 if i in [0, len(pt_bin_edges)-1] else 0.4
            linewidth = 2 if i in [0, len(pt_bin_edges)-1] else 0.8
            axes[1].axvline(edge, color='blue', linestyle='-', alpha=alpha_val, linewidth=linewidth)
        
        pt_widths = np.diff(pt_bin_edges)
        axes[1].set_xlabel('pT [GeV]', fontsize=12)
        axes[1].set_ylabel('Count (log scale)', fontsize=12)
        axes[1].set_title(f'Same {num_bins} bins mapped to pT space\n'
                         f'pT width: {np.min(pt_widths):.2f}-{np.max(pt_widths):.2f} GeV', fontsize=13)
        axes[1].set_xlim(0, 210)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='darkred', alpha=0.6, lw=10, label='Distribution'),
            Line2D([0], [0], color='blue', lw=1.5, label=f'{num_bins} uniform 1/pT bins'),
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle(f'1/pT Uniform Binning ({num_bins} bins): Curvature-Motivated', fontsize=14)
        plt.tight_layout()
        plt.savefig(hist_dir / f'inv_pt_bins_{num_bins}_resolution.png', dpi=150)
        plt.close()
        print(f"  Saved: {hist_dir / f'inv_pt_bins_{num_bins}_resolution.png'}")
    
    print(f"\n1/pT bin resolution plots saved to: {hist_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compute YOLO-style classification bins')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for bins (default: data_dir/bins)')
    parser.add_argument('--max_events', type=int, default=-1,
                        help='Maximum number of events to process (-1 for all)')
    parser.add_argument('--pt_bins', type=str, default='10,25,50,100,200,400',
                        help='Comma-separated list of pt bin counts')
    parser.add_argument('--eta_bins', type=str, default='50,100,500,1000,2000,6000,12000',
                        help='Comma-separated list of eta bin counts')
    parser.add_argument('--phi_bins', type=str, default='50,100,500,1000,2000,6000,12000',
                        help='Comma-separated list of phi bin counts')
    parser.add_argument('--pt_log_bins', action='store_true',
                        help='Use logarithmic binning for pt instead of quantile-based')
    parser.add_argument('--inv_pt_bins', type=str, default='10,25,50,100,200,400',
                        help='Comma-separated list of 1/pt bin counts (uniform in 1/pt space)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'bins'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pt_bin_counts = [int(x) for x in args.pt_bins.split(',')]
    eta_bin_counts = [int(x) for x in args.eta_bins.split(',')]
    phi_bin_counts = [int(x) for x in args.phi_bins.split(',')]
    inv_pt_bin_counts = [int(x) for x in args.inv_pt_bins.split(',')]
    
    print(f"Loading dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"PT bin counts: {pt_bin_counts}")
    print(f"1/PT bin counts: {inv_pt_bin_counts}")
    print(f"Eta bin counts: {eta_bin_counts}")
    print(f"Phi bin counts: {phi_bin_counts}")
    
    # Load metadata
    metadata = load_metadata(data_dir)
    print(f"Total events in dataset: {metadata['event_mapping']['total_events']}")
    
    # Collect all track data
    print("\nCollecting track data...")
    all_pt = []
    all_eta = []
    all_phi = []
    all_charge = []
    all_innermost_eta = []
    all_innermost_phi = []
    all_num_hits = []
    
    for track_data in iterate_tracks(data_dir, metadata, max_events=args.max_events):
        all_pt.append(track_data['pt'])
        all_eta.append(track_data['eta'])
        all_phi.append(track_data['phi'])
        all_charge.append(track_data['charge'])
        all_innermost_eta.append(track_data['innermost_hit_eta'])
        all_innermost_phi.append(track_data['innermost_hit_phi'])
        all_num_hits.append(track_data['num_hits'])
    
    # Convert to numpy arrays
    all_pt = np.array(all_pt)
    all_eta = np.array(all_eta)
    all_phi = np.array(all_phi)
    all_charge = np.array(all_charge)
    all_innermost_eta = np.array(all_innermost_eta)
    all_innermost_phi = np.array(all_innermost_phi)
    all_num_hits = np.array(all_num_hits)
    
    num_tracks = len(all_pt)
    print(f"\nTotal tracks collected: {num_tracks:,}")
    
    # Compute statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # PT statistics
    print(f"\nPT (GeV):")
    print(f"  Min: {np.min(all_pt):.4f}")
    print(f"  Max: {np.max(all_pt):.4f}")
    print(f"  Mean: {np.mean(all_pt):.4f}")
    print(f"  Std: {np.std(all_pt):.4f}")
    print(f"  Median: {np.median(all_pt):.4f}")
    print(f"  Log PT range: [{np.log(np.min(all_pt)):.4f}, {np.log(np.max(all_pt)):.4f}]")
    
    # 1/PT statistics
    all_inv_pt = 1.0 / all_pt
    print(f"\n1/PT (1/GeV):")
    print(f"  Min: {np.min(all_inv_pt):.6f}")
    print(f"  Max: {np.max(all_inv_pt):.6f}")
    print(f"  Mean: {np.mean(all_inv_pt):.6f}")
    print(f"  Std: {np.std(all_inv_pt):.6f}")
    print(f"  Median: {np.median(all_inv_pt):.6f}")
    
    # Eta statistics
    print(f"\nEta:")
    print(f"  Min: {np.min(all_eta):.4f}")
    print(f"  Max: {np.max(all_eta):.4f}")
    print(f"  Mean: {np.mean(all_eta):.4f}")
    print(f"  Std: {np.std(all_eta):.4f}")
    
    # Phi statistics
    print(f"\nPhi (rad):")
    print(f"  Min: {np.min(all_phi):.4f}")
    print(f"  Max: {np.max(all_phi):.4f}")
    print(f"  Mean: {np.mean(all_phi):.4f}")
    print(f"  Std: {np.std(all_phi):.4f}")
    
    # Charge statistics
    print(f"\nCharge:")
    pos_charge = np.sum(all_charge > 0)
    neg_charge = np.sum(all_charge < 0)
    print(f"  Positive: {pos_charge:,} ({100*pos_charge/num_tracks:.1f}%)")
    print(f"  Negative: {neg_charge:,} ({100*neg_charge/num_tracks:.1f}%)")
    
    # Hits per track statistics
    print(f"\nHits per track:")
    print(f"  Min: {np.min(all_num_hits)}")
    print(f"  Max: {np.max(all_num_hits)}")
    print(f"  Mean: {np.mean(all_num_hits):.2f}")
    print(f"  Median: {np.median(all_num_hits):.1f}")
    
    # Innermost hit seed precision (THIS IS KEY FOR THE SEED BIASED SOFTMAX)
    print("\n" + "="*60)
    print("INNERMOST HIT SEED PRECISION")
    print("="*60)
    
    # Eta seed precision
    eta_residuals = all_innermost_eta - all_eta
    eta_seed_std = np.std(eta_residuals)
    eta_seed_mae = np.mean(np.abs(eta_residuals))
    print(f"\nEta (innermost hit - truth):")
    print(f"  Std of residual: {eta_seed_std:.6f} rad = {eta_seed_std * 1000:.3f} mrad")
    print(f"  MAE: {eta_seed_mae:.6f} rad = {eta_seed_mae * 1000:.3f} mrad")
    print(f"  Mean bias: {np.mean(eta_residuals):.6f} rad")
    
    # Phi seed precision (with periodic handling)
    phi_residuals = angular_difference(all_innermost_phi, all_phi)
    phi_seed_std = np.std(phi_residuals)
    phi_seed_mae = np.mean(np.abs(phi_residuals))
    print(f"\nPhi (innermost hit - truth, periodic-aware):")
    print(f"  Std of residual: {phi_seed_std:.6f} rad = {phi_seed_std * 1000:.3f} mrad")
    print(f"  MAE: {phi_seed_mae:.6f} rad = {phi_seed_mae * 1000:.3f} mrad")
    print(f"  Mean bias: {np.mean(phi_residuals):.6f} rad")
    
    # Create histogram plots
    create_histograms(all_pt, all_eta, all_phi, all_charge, 
                      eta_residuals, phi_residuals, output_dir, all_inv_pt=all_inv_pt)
    
    # Create bin resolution visualization plots
    create_bin_resolution_plots(all_pt, all_eta, output_dir, pt_bin_counts, eta_bin_counts)
    
    # Create logarithmic pt bin resolution plots
    create_log_pt_bin_resolution_plots(all_pt, output_dir, pt_bin_counts)
    
    # Create 1/pt bin resolution plots
    create_inv_pt_bin_resolution_plots(all_inv_pt, output_dir, inv_pt_bin_counts)

    print("\n" + "="*60)
    print("CREATING BIN ARRAYS")
    print("="*60)
    print("="*60)
    
    # Hard clipping ranges
    PT_MIN, PT_MAX = 5.0, 200.0  # GeV
    ETA_MIN, ETA_MAX = -2.7, 2.7
    ETA_GAP_MIN, ETA_GAP_MAX = -0.1, 0.1  # Barrel gap with no tracks
    PHI_MIN, PHI_MAX = -np.pi, np.pi
    
    # Filter PT for binning (clip to valid range)
    pt_for_binning = all_pt[(all_pt >= PT_MIN) & (all_pt <= PT_MAX)]
    print(f"\nPT clipping: [{PT_MIN}, {PT_MAX}] GeV")
    print(f"  Tracks in range: {len(pt_for_binning):,} / {len(all_pt):,} ({100*len(pt_for_binning)/len(all_pt):.2f}%)")
    
    # Create PT bins - either logarithmic or quantile-based
    if args.pt_log_bins:
        print("\nPT bins (LOGARITHMIC, clipped):")
        for num_bins in pt_bin_counts:
            pt_bins = compute_log_bins(PT_MIN, PT_MAX, num_bins)
            pt_centers = compute_bin_centers(pt_bins)
            
            # Save bin edges and centers with _log suffix
            np.save(output_dir / f'pt_bins_{num_bins}_log.npy', pt_bins)
            np.save(output_dir / f'pt_bin_centers_{num_bins}_log.npy', pt_centers)
            
            # Compute bin widths for reporting
            bin_widths = np.diff(pt_bins)
            print(f"  {num_bins} bins: edges [{pt_bins[0]:.2f}, {pt_bins[-1]:.2f}] GeV, "
                  f"width range [{np.min(bin_widths):.3f}, {np.max(bin_widths):.3f}] GeV")
            print(f"    Low pt width: {bin_widths[0]:.3f} GeV, High pt width: {bin_widths[-1]:.3f} GeV")
    else:
        print("\nPT bins (quantile-based, clipped):")
        for num_bins in pt_bin_counts:
            pt_bins = compute_quantile_bins(pt_for_binning, num_bins)
            # Ensure exact boundaries
            pt_bins[0] = PT_MIN
            pt_bins[-1] = PT_MAX
            pt_centers = compute_bin_centers(pt_bins)
            
            # Save bin edges and centers
            np.save(output_dir / f'pt_bins_{num_bins}.npy', pt_bins)
            np.save(output_dir / f'pt_bin_centers_{num_bins}.npy', pt_centers)
            
            # Compute bin widths for reporting
            bin_widths = np.diff(pt_bins)
            print(f"  {num_bins} bins: edges [{pt_bins[0]:.2f}, {pt_bins[-1]:.2f}] GeV, "
                  f"width range [{np.min(bin_widths):.3f}, {np.max(bin_widths):.3f}] GeV")
    
    # Filter eta: remove barrel gap (-0.1, 0.1) where no tracks exist
    eta_for_binning = all_eta[(all_eta <= ETA_GAP_MIN) | (all_eta >= ETA_GAP_MAX)]
    print(f"\nEta range: [{ETA_MIN}, {ETA_MAX}] with gap [{ETA_GAP_MIN}, {ETA_GAP_MAX}] removed")
    print(f"  Tracks outside gap: {len(eta_for_binning):,} / {len(all_eta):,} ({100*len(eta_for_binning)/len(all_eta):.2f}%)")
    
    # Create Eta bins (quantile-based, with gap removed)
    # We create separate bins for negative and positive eta regions
    print("\nEta bins (quantile-based, gap removed):")
    for num_bins in eta_bin_counts:
        # Split bins between negative and positive eta regions
        eta_neg = all_eta[(all_eta >= ETA_MIN) & (all_eta <= ETA_GAP_MIN)]
        eta_pos = all_eta[(all_eta >= ETA_GAP_MAX) & (all_eta <= ETA_MAX)]
        
        # Allocate bins proportionally to track count in each region
        neg_frac = len(eta_neg) / (len(eta_neg) + len(eta_pos))
        neg_bins = max(1, int(num_bins * neg_frac))
        pos_bins = num_bins - neg_bins
        
        # Compute quantile bins for each region
        eta_bins_neg = compute_quantile_bins(eta_neg, neg_bins)
        eta_bins_pos = compute_quantile_bins(eta_pos, pos_bins)
        
        # Ensure exact boundaries
        eta_bins_neg[0] = ETA_MIN
        eta_bins_neg[-1] = ETA_GAP_MIN
        eta_bins_pos[0] = ETA_GAP_MAX
        eta_bins_pos[-1] = ETA_MAX
        
        # Concatenate (remove duplicate at boundary)
        eta_bins = np.concatenate([eta_bins_neg, eta_bins_pos])
        eta_centers = compute_bin_centers(eta_bins)
        
        np.save(output_dir / f'eta_bins_{num_bins}.npy', eta_bins)
        np.save(output_dir / f'eta_bin_centers_{num_bins}.npy', eta_centers)
        
        bin_widths = np.diff(eta_bins)
        # Report widths excluding the gap
        real_widths = bin_widths[bin_widths < 0.15]  # Exclude gap width
        print(f"  {num_bins} bins ({neg_bins} neg + {pos_bins} pos): edges [{eta_bins[0]:.4f}, {eta_bins[-1]:.4f}], "
              f"width range [{np.min(real_widths)*1000:.3f}, {np.max(real_widths)*1000:.3f}] mrad (excl. gap)")
    
    # Create Phi bins (uniform over [-pi, pi])
    print(f"\nPhi range: [{PHI_MIN:.4f}, {PHI_MAX:.4f}] (exactly -π to π)")
    print("\nPhi bins (uniform):")
    for num_bins in phi_bin_counts:
        # Use exact -pi to pi range
        phi_bins = np.linspace(PHI_MIN, PHI_MAX, num_bins + 1)
        phi_centers = compute_bin_centers(phi_bins)
        
        np.save(output_dir / f'phi_bins_{num_bins}.npy', phi_bins)
        np.save(output_dir / f'phi_bin_centers_{num_bins}.npy', phi_centers)
        
        bin_width = (phi_bins[-1] - phi_bins[0]) / num_bins
        print(f"  {num_bins} bins: edges [{phi_bins[0]:.4f}, {phi_bins[-1]:.4f}], "
              f"width {bin_width:.6f} rad = {bin_width * 1000:.3f} mrad")
    
    # Create 1/PT bins (uniform in 1/pt space)
    # The detector measures curvature proportional to 1/pt directly
    INV_PT_MIN = 1.0 / PT_MAX  # = 0.005 (at pt = 200 GeV)
    INV_PT_MAX = 1.0 / PT_MIN  # = 0.2   (at pt = 5 GeV)
    print(f"\n1/PT range: [{INV_PT_MIN:.6f}, {INV_PT_MAX:.6f}] (from PT [{PT_MIN}, {PT_MAX}])")
    print("\n1/PT bins (uniform in 1/pt space):")
    for num_bins in inv_pt_bin_counts:
        inv_pt_bins = np.linspace(INV_PT_MIN, INV_PT_MAX, num_bins + 1)
        inv_pt_centers = compute_bin_centers(inv_pt_bins)
        
        np.save(output_dir / f'inv_pt_bins_{num_bins}.npy', inv_pt_bins)
        np.save(output_dir / f'inv_pt_bin_centers_{num_bins}.npy', inv_pt_centers)
        
        bin_width = (INV_PT_MAX - INV_PT_MIN) / num_bins
        print(f"  {num_bins} bins: 1/pt edges [{inv_pt_bins[0]:.6f}, {inv_pt_bins[-1]:.6f}], "
              f"width {bin_width*1000:.4f}e-3 1/GeV")
        print(f"    -> pt resolution: {1/inv_pt_bins[1]:.1f}-{1/inv_pt_bins[0]:.1f} GeV (low pt), "
              f"{1/inv_pt_bins[-1]:.1f}-{1/inv_pt_bins[-2]:.1f} GeV (high pt)")
    
    # Save raw data arrays for histogram plotting
    print("\nSaving raw data arrays for plotting...")
    np.save(output_dir / 'all_pt.npy', all_pt)
    np.save(output_dir / 'all_eta.npy', all_eta)
    np.save(output_dir / 'all_phi.npy', all_phi)
    np.save(output_dir / 'all_charge.npy', all_charge)
    np.save(output_dir / 'all_inv_pt.npy', all_inv_pt)
    
    # Save statistics to text file
    stats_file = output_dir / 'statistics.txt'
    with open(stats_file, 'w') as f:
        f.write(f"YOLO Binning Statistics\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total tracks: {num_tracks:,}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"PT (GeV)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Min: {np.min(all_pt):.4f}\n")
        f.write(f"Max: {np.max(all_pt):.4f}\n")
        f.write(f"Mean: {np.mean(all_pt):.4f}\n")
        f.write(f"Std: {np.std(all_pt):.4f}\n")
        f.write(f"Median: {np.median(all_pt):.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Eta\n")
        f.write(f"{'='*60}\n")
        f.write(f"Min: {np.min(all_eta):.4f}\n")
        f.write(f"Max: {np.max(all_eta):.4f}\n")
        f.write(f"Mean: {np.mean(all_eta):.4f}\n")
        f.write(f"Std: {np.std(all_eta):.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Phi (rad)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Min: {np.min(all_phi):.4f}\n")
        f.write(f"Max: {np.max(all_phi):.4f}\n")
        f.write(f"Mean: {np.mean(all_phi):.4f}\n")
        f.write(f"Std: {np.std(all_phi):.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"INNERMOST HIT SEED PRECISION (KEY FOR SEED BIASED SOFTMAX)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Eta seed std: {eta_seed_std:.6f} rad = {eta_seed_std * 1000:.3f} mrad\n")
        f.write(f"Eta seed MAE: {eta_seed_mae:.6f} rad = {eta_seed_mae * 1000:.3f} mrad\n")
        f.write(f"Phi seed std: {phi_seed_std:.6f} rad = {phi_seed_std * 1000:.3f} mrad\n")
        f.write(f"Phi seed MAE: {phi_seed_mae:.6f} rad = {phi_seed_mae * 1000:.3f} mrad\n")
        f.write(f"\nRecommended sigma for SeedBiasedSoftmax:\n")
        f.write(f"  eta_sigma: {eta_seed_std:.4f} (1 std of innermost hit residual)\n")
        f.write(f"  phi_sigma: {phi_seed_std:.4f} (1 std of innermost hit residual)\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Charge Distribution\n")
        f.write(f"{'='*60}\n")
        f.write(f"Positive: {pos_charge:,} ({100*pos_charge/num_tracks:.1f}%)\n")
        f.write(f"Negative: {neg_charge:,} ({100*neg_charge/num_tracks:.1f}%)\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"All bin arrays saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
