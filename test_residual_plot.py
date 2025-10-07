#!/usr/bin/env python3
"""
Test script to verify the residual plotting fix
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_residual_plot():
    """Test the 3-sigma residual plotting logic"""
    
    # Generate test data with some outliers
    np.random.seed(42)
    
    # Normal data centered around 0.1 with some spread
    normal_data = np.random.normal(0.1, 0.05, 1000)
    
    # Add some outliers
    outliers = np.random.uniform(-0.5, 0.6, 50)
    
    # Combine data
    clean_norm_residuals = np.concatenate([normal_data, outliers])
    
    # Calculate mean and std for 3-sigma range
    mean_residual = np.mean(clean_norm_residuals)
    std_residual = np.std(clean_norm_residuals)
    
    print(f"Data statistics:")
    print(f"  Mean: {mean_residual:.6f}")
    print(f"  Std: {std_residual:.6f}")
    print(f"  Total points: {len(clean_norm_residuals)}")
    
    # Define range as mean ± 3*sigma
    lower_bound = mean_residual - 3 * std_residual
    upper_bound = mean_residual + 3 * std_residual
    
    print(f"  3-sigma range: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    # Create bins centered on mean, extending 3 sigma to each side
    n_bins = 50
    bins = np.linspace(lower_bound, upper_bound, n_bins + 1)
    
    # Clip outliers into edge bins
    clipped_residuals = np.clip(clean_norm_residuals, lower_bound, upper_bound)
    
    # Count clipped values
    n_total = len(clean_norm_residuals)
    n_clipped = np.sum((clean_norm_residuals < lower_bound) | (clean_norm_residuals > upper_bound))
    clipped_pct = 100 * n_clipped / n_total if n_total > 0 else 0
    
    print(f"  Clipped: {n_clipped} ({clipped_pct:.1f}%)")
    
    # Calculate histogram with clipped data
    counts, bin_edges = np.histogram(clipped_residuals, bins=bins, density=False)
    
    # Create step histogram plot
    plt.figure(figsize=(10, 6))
    
    # Create step histogram with thinner lines
    plt.step(bin_edges[:-1], counts, where='post', linewidth=1, color='blue',
             label=f'Test Normalized Residuals')

    # Add vertical lines for statistics
    plt.axvline(mean_residual, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_residual:.6f}')
    plt.axvline(mean_residual - std_residual, color='orange', linestyle=':', alpha=0.7,
               label=f'±1σ: {std_residual:.6f}')
    plt.axvline(mean_residual + std_residual, color='orange', linestyle=':', alpha=0.7)
    plt.axvline(mean_residual - 2*std_residual, color='yellow', linestyle=':', alpha=0.7,
               label=f'±2σ')
    plt.axvline(mean_residual + 2*std_residual, color='yellow', linestyle=':', alpha=0.7)
    plt.axvline(lower_bound, color='purple', linestyle='-', alpha=0.7,
               label=f'±3σ (plot range)')
    plt.axvline(upper_bound, color='purple', linestyle='-', alpha=0.7)
    plt.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1,
               label='Perfect agreement')

    # Formatting
    plt.xlabel('Test Normalized Residual')
    plt.title('Test Truth-Normalized Residual Distribution\n(Centered on mean ± 3σ, outliers clipped to edge bins)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text box with statistics
    stats_text = f'Mean: {mean_residual:.6f}\nSTD: {std_residual:.6f}\nN: {n_total}\nClipped: {n_clipped} ({clipped_pct:.1f}%)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot
    output_path = 'test_residual_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved test residual plot to {output_path}")
    print(f"Plot shows data centered on mean ({mean_residual:.6f}) ± 3σ range")
    print(f"Outliers beyond ±3σ ({n_clipped} points) are clipped into edge bins")

if __name__ == "__main__":
    test_residual_plot()