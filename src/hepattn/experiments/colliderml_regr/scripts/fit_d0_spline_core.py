#!/usr/bin/env python3
"""Fit a monotonic PCHIP spline to the d0 distribution from the core finetune dataset.

The d0 distribution on the core dataset (d0 in [-2.5, 2.5] mm) is extremely
peaked: 90% of tracks have |d0| < 0.023 mm.  A linear normalization to [-1,1]
wastes >99% of the output range on the tails.  This script fits a monotone
spline that maps d0 -> [0, 1] via the empirical CDF, giving ~48x better
resolution in the core where most tracks live.

The resulting spline config is used by ``SplineQuantileLoss`` in the training
pipeline.

Usage::

    python fit_d0_spline_core.py
    python fit_d0_spline_core.py --num-shards 200   # faster, fewer shards
    python fit_d0_spline_core.py --all               # use all 1000 shards

Output:
    config/NeurIPS_retraining/v2/core/splines/spline_d0.yaml
    config/NeurIPS_retraining/v2/core/splines/d0_spline_fit.png
    config/NeurIPS_retraining/v2/core/splines/d0_qq_uniform.png
    config/NeurIPS_retraining/v2/core/splines/d0_calibration.png
    config/NeurIPS_retraining/v2/core/splines/d0_pdf_overlay.png
    config/NeurIPS_retraining/v2/core/splines/d0_transformed_hist.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.colliderml_regr.spline import (
    evaluate_pchip_np as evaluate_pchip,
    fritsch_carlson_slopes_np as fritsch_carlson_slopes,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("/scratch/colliderml/p200_core_finetune")
D0_COL = 0  # d0 is column 0 in track_targets.npy

# Knot placement strategy for d0.
#
# The d0 distribution is extremely peaked at zero:
#   - 90% of mass in [-0.023, +0.023] within a [-2.5, +2.5] range
#   - CDF rises from ~0.05 to ~0.95 within a 0.046 mm window
#   - Long flat tails extending to ±2.5 mm
#
# Strategy:
#   1. 50 uniformly-spaced quantiles (q=0, 1/49, 2/49, ..., 1) form the
#      backbone — most will cluster near zero because that's where the mass is.
#   2. Dense tail quantiles capture the edges where the CDF is nearly flat.
#      We go down to 0.00005 and up to 0.99995 to resolve d0 near ±2.5.
#   3. Dense transition-region quantiles (CDF 0.005–0.07, 0.93–0.995)
#      bridge the sharp knee where the steep core meets the flat tails.
#      Without these, the PCHIP interpolant can introduce physical-space
#      errors of ~0.04 mm in the transition, which is significant for
#      a 0.1 mm precision target.
#
# After merging and deduplication, this typically yields ~100-120 knots.
NUM_CORE_KNOTS = 50

TAIL_QUANTILES = [
    # Lower extreme tail: d0 near -2.5 mm
    0.00005, 0.0001, 0.0002, 0.0005,
    # Lower tail: d0 ~ -2.0 to -0.3 mm
    0.001, 0.0015, 0.002, 0.003, 0.004, 0.005,
    # Lower transition (CDF 0.005-0.07): the critical knee region
    # This is where the CDF slope drops from ~40/mm (core) to ~0.01/mm (tail)
    # and where the previous fit had max errors of ~0.04 mm.
    0.006, 0.007, 0.008, 0.009, 0.01,
    0.011, 0.012, 0.013, 0.014, 0.015,
    0.016, 0.018, 0.02, 0.022, 0.025,
    0.028, 0.03, 0.033, 0.035, 0.04,
    0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
    # Upper transition (CDF 0.93-0.995): mirror of the lower knee
    0.93, 0.935, 0.94, 0.945, 0.95, 0.955,
    0.96, 0.965, 0.967, 0.97, 0.972, 0.975,
    0.978, 0.98, 0.982, 0.985, 0.987, 0.989,
    0.99, 0.991, 0.992, 0.993, 0.994, 0.995,
    # Upper tail: d0 ~ +0.3 to +2.0 mm
    0.996, 0.997, 0.998, 0.9985, 0.999,
    # Upper extreme tail: d0 near +2.5 mm
    0.9995, 0.9998, 0.9999, 0.99995,
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_d0(preprocessed_dir: Path, num_shards: int = -1) -> np.ndarray:
    """Load d0 values from preprocessed shards."""
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    chunks: list[np.ndarray] = []
    for sd in tqdm(shard_dirs, desc="Loading d0"):
        tgt_path = sd / "selected_tracks" / "track_targets.npy"
        if tgt_path.exists():
            arr = np.load(tgt_path)
            if len(arr) > 0:
                chunks.append(arr[:, D0_COL])

    if not chunks:
        raise ValueError(f"No targets found in {preprocessed_dir}")

    d0 = np.concatenate(chunks)
    print(f"Loaded {len(d0):,} d0 values from {len(shard_dirs)} shards")
    return d0


# ---------------------------------------------------------------------------
# Knot placement
# ---------------------------------------------------------------------------


def compute_knots(
    values: np.ndarray,
    num_core: int,
    tail_quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Place spline knots at quantile positions of the data.

    Returns (knot_x, knot_y) where knot_x are physical d0 values and
    knot_y are the corresponding CDF values in [0, 1].
    """
    core_q = np.linspace(0.0, 1.0, num_core)
    all_q = np.unique(np.concatenate([core_q, tail_quantiles]))
    all_q = np.clip(all_q, 0.0, 1.0)
    all_q = np.sort(all_q)

    knot_x = np.quantile(values, all_q)

    # Remove duplicate x-values (can happen in the steep core where many
    # quantiles map to nearly the same physical value)
    _, unique_idx = np.unique(knot_x, return_index=True)
    unique_idx = np.sort(unique_idx)
    knot_x = knot_x[unique_idx]
    knot_y = all_q[unique_idx]

    # Pin endpoints to exactly 0 and 1
    knot_y[0] = 0.0
    knot_y[-1] = 1.0

    return knot_x, knot_y


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_spline_fit(
    d0: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    output_dir: Path,
) -> None:
    """3-panel diagnostic: CDF comparison, transformed histogram, residuals."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"d0 Spline Fit — Core Dataset ({len(d0):,} tracks, {len(knot_x)} knots)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: CDF comparison
    ax = axes[0]
    sorted_d0 = np.sort(d0)
    ecdf_y = np.arange(1, len(sorted_d0) + 1) / len(sorted_d0)
    fine_x = np.linspace(knot_x[0], knot_x[-1], 5000)
    spline_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)

    ax.plot(sorted_d0[::100], ecdf_y[::100], "b-", alpha=0.4, lw=0.5,
            label="Empirical CDF")
    ax.plot(fine_x, spline_y, "r-", lw=2, label="PCHIP spline")
    ax.plot(knot_x, knot_y, "ko", ms=4, label=f"Knots (n={len(knot_x)})",
            zorder=5)
    ax.set_xlabel("d0 [mm]")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF vs Spline")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Transformed histogram (should be uniform)
    ax = axes[1]
    transformed = evaluate_pchip(d0, knot_x, knot_y, slopes)
    ax.hist(transformed, bins=100, density=True, alpha=0.7, color="steelblue",
            edgecolor="black", lw=0.3)
    ax.axhline(1.0, color="r", ls="--", lw=1.5, label="Ideal uniform")
    ax.set_xlabel("Transformed d0 (CDF space)")
    ax.set_ylabel("Density")
    ax.set_title("Transformed Histogram (should be flat)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Residuals
    ax = axes[2]
    spline_at_data = evaluate_pchip(sorted_d0, knot_x, knot_y, slopes)
    residuals = ecdf_y - spline_at_data
    # Subsample for plotting
    step = max(1, len(residuals) // 10000)
    ax.plot(sorted_d0[::step], residuals[::step], "b-", lw=0.5, alpha=0.6)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.fill_between(sorted_d0[::step], residuals[::step], alpha=0.2,
                     color="steelblue")
    ax.set_xlabel("d0 [mm]")
    ax.set_ylabel("Residual (ECDF − Spline)")
    ax.set_title("Fit Residuals")
    ax.grid(True, alpha=0.3)

    rmse = np.sqrt(np.mean(residuals**2))
    max_err = np.max(np.abs(residuals))
    ks_stat = max_err
    stats = f"RMSE: {rmse:.6f}\nMax |err|: {max_err:.6f}\nKS stat: {ks_stat:.6f}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            va="top", bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})

    plt.tight_layout()
    fig.savefig(output_dir / "d0_spline_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: d0_spline_fit.png  (RMSE={rmse:.6f}, KS={ks_stat:.6f})")


def plot_qq_uniform(
    d0: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    output_dir: Path,
) -> None:
    """Q-Q plot of transformed values vs Uniform(0,1)."""
    transformed = np.sort(evaluate_pchip(d0, knot_x, knot_y, slopes))
    n = len(transformed)
    theoretical = np.linspace(0, 1, n)

    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, n // 5000)
    ax.scatter(theoretical[::step], transformed[::step], s=1, alpha=0.3,
               color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Theoretical uniform quantiles")
    ax.set_ylabel("Empirical (transformed) quantiles")
    ax.set_title("Q-Q Plot: Spline-Transformed d0 vs Uniform(0,1)")

    ks_stat = np.max(np.abs(transformed - theoretical))
    ax.text(0.05, 0.92, f"KS = {ks_stat:.5f}", transform=ax.transAxes,
            fontsize=11, bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "d0_qq_uniform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: d0_qq_uniform.png  (KS={ks_stat:.5f})")


def plot_calibration(
    d0: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    output_dir: Path,
) -> None:
    """Calibration plot: expected vs observed quantile coverage."""
    transformed = evaluate_pchip(d0, knot_x, knot_y, slopes)
    q_levels = np.linspace(0.01, 0.99, 200)
    observed = np.array([np.mean(transformed <= q) for q in q_levels])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(q_levels, observed, "-", lw=2, color="tab:blue", label="d0 spline")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("Expected quantile level")
    ax.set_ylabel("Observed fraction below level")
    ax.set_title("d0 Spline Calibration")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    max_cal_err = np.max(np.abs(observed - q_levels))
    ax.text(0.05, 0.92, f"Max cal. error: {max_cal_err:.5f}",
            transform=ax.transAxes, fontsize=11,
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})

    plt.tight_layout()
    fig.savefig(output_dir / "d0_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: d0_calibration.png  (max cal err={max_cal_err:.5f})")


def plot_pdf_overlay(
    d0: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    output_dir: Path,
) -> None:
    """Spline derivative (≈ PDF) overlaid on empirical histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("d0 Spline PDF vs Histogram", fontsize=13, fontweight="bold")

    # Full range
    ax = axes[0]
    ax.hist(d0, bins=200, density=True, alpha=0.4, color="steelblue",
            edgecolor="none", label="Histogram")
    fine_x = np.linspace(knot_x[0], knot_x[-1], 10000)
    fine_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)
    dx = fine_x[1] - fine_x[0]
    deriv = np.gradient(fine_y, dx)
    ax.plot(fine_x, deriv, "r-", lw=2, label="Spline dCDF/dx")
    ax.set_xlabel("d0 [mm]")
    ax.set_ylabel("Density")
    ax.set_title("Full range")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Zoomed to core [-0.1, 0.1]
    ax = axes[1]
    core_mask = np.abs(d0) < 0.1
    ax.hist(d0[core_mask], bins=200, density=True, alpha=0.4, color="steelblue",
            edgecolor="none", label="Histogram (|d0|<0.1)")
    core_fine = (fine_x >= -0.1) & (fine_x <= 0.1)
    ax.plot(fine_x[core_fine], deriv[core_fine], "r-", lw=2,
            label="Spline dCDF/dx")
    ax.set_xlabel("d0 [mm]")
    ax.set_ylabel("Density")
    ax.set_title("Zoomed: |d0| < 0.1 mm (core)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "d0_pdf_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: d0_pdf_overlay.png")


def plot_transformed_hist(
    d0: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    output_dir: Path,
) -> None:
    """Detailed histogram of transformed values with uniformity metrics."""
    transformed = evaluate_pchip(d0, knot_x, knot_y, slopes)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts, edges, _ = ax.hist(transformed, bins=100, density=True, alpha=0.7,
                                color="steelblue", edgecolor="black", lw=0.3)
    ax.axhline(1.0, color="r", ls="--", lw=1.5, label="Ideal uniform density")
    ax.set_xlabel("Transformed d0 (CDF space)")
    ax.set_ylabel("Density")
    ax.set_title(f"Transformed d0 Histogram — {len(d0):,} tracks, 100 bins")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Uniformity metrics
    bin_counts_raw = np.histogram(transformed, bins=100)[0]
    expected = len(d0) / 100
    chi2 = np.sum((bin_counts_raw - expected) ** 2 / expected)
    max_dev = np.max(np.abs(counts - 1.0))
    stats = f"Chi² (100 bins): {chi2:.1f}\nMax |density − 1|: {max_dev:.4f}"
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "d0_transformed_hist.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: d0_transformed_hist.png  (chi²={chi2:.1f}, max_dev={max_dev:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit d0 spline for the core finetune dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preprocessed-dir", type=str,
        default=str(PREPROCESSED_DIR),
        help="Preprocessed memmap directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: config/NeurIPS_retraining/v2/core/splines)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=-1,
        help="Number of shards to load (-1 for all)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Load all shards (same as --num-shards -1)",
    )
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    if args.output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "config" / "NeurIPS_retraining" / "v2" / "core" / "splines"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_shards = -1 if args.all else args.num_shards

    print("=" * 70)
    print("d0 Spline Fitting — Core Finetune Dataset")
    print("=" * 70)
    print(f"  Data:       {preprocessed_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Shards:     {'all' if num_shards == -1 else num_shards}")
    print(f"  Core knots: {NUM_CORE_KNOTS}")
    print(f"  Tail Qs:    {len(TAIL_QUANTILES)}")
    print()

    # Load data
    d0 = load_d0(preprocessed_dir, num_shards=num_shards)

    # Print distribution summary
    print(f"\n  d0 distribution summary:")
    print(f"    range: [{d0.min():+.6f}, {d0.max():+.6f}] mm")
    print(f"    mean:  {d0.mean():+.6f}")
    print(f"    std:   {d0.std():.6f}")
    for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f"    p{int(p*100):02d}:   {np.quantile(d0, p):+.6f}")
    print()

    # Compute knots
    print("  Fitting spline knots...")
    knot_x, knot_y = compute_knots(d0, NUM_CORE_KNOTS, TAIL_QUANTILES)
    slopes = fritsch_carlson_slopes(knot_x, knot_y)
    print(f"    Total knots: {len(knot_x)}")
    print(f"    x range: [{knot_x[0]:+.8f}, {knot_x[-1]:+.8f}]")
    print(f"    Knots with |x| < 0.05: {np.sum(np.abs(knot_x) < 0.05)}")
    print(f"    Knots with |x| > 1.0:  {np.sum(np.abs(knot_x) > 1.0)}")
    print()

    # Verify monotonicity of slopes (should always be true with PCHIP)
    spline_at_knots = evaluate_pchip(knot_x, knot_x, knot_y, slopes)
    is_monotone = np.all(np.diff(spline_at_knots) >= 0)
    print(f"    Monotonicity check: {'PASS' if is_monotone else 'FAIL'}")
    if not is_monotone:
        print("    WARNING: spline is not monotone — check knot placement")

    # Save spline config
    config = {
        "name": "d0",
        "units": "mm",
        "knot_x": [float(v) for v in knot_x],
        "knot_y": [float(v) for v in knot_y],
        "num_tracks": int(len(d0)),
    }
    config_path = output_dir / "spline_d0.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"    Config saved: {config_path}")

    # Generate diagnostic plots
    print("\n  Generating diagnostic plots...")
    plot_spline_fit(d0, knot_x, knot_y, slopes, output_dir)
    plot_qq_uniform(d0, knot_x, knot_y, slopes, output_dir)
    plot_calibration(d0, knot_x, knot_y, slopes, output_dir)
    plot_pdf_overlay(d0, knot_x, knot_y, slopes, output_dir)
    plot_transformed_hist(d0, knot_x, knot_y, slopes, output_dir)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
