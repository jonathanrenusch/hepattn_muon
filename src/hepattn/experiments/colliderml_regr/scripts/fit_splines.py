#!/usr/bin/env python3
# ruff: noqa: TID252, PLR0915
"""Fit monotonic splines to ColliderML track parameter distributions.

This version reads from the **preprocessed memmap format** produced by
``preprocess_colliderml.py`` — much faster than reading raw parquet shards.

For each of the four non-angular track parameters (d0, z0, theta, qop) this
script:

1. Loads track targets from preprocessed ``shard_XXXX/selected_tracks/track_targets.npy``.
2. Places spline knots at quantile boundaries of the selected distribution.
3. Fits a monotone PCHIP spline mapping physical values → [0, 1].
4. Saves the knot tables to a YAML file for each parameter.
5. Produces diagnostic plots.

Usage::

    python fit_splines.py --preprocessed-dir /scratch/colliderml/p0_preprocessed \\
        --output-dir config/splines --num-shards -1
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
import matplotlib.pyplot as plt


# Target column indices in track_targets.npy: [d0, z0, phi, theta, qop]
TARGET_COLS = {"d0": 0, "z0": 1, "phi": 2, "theta": 3, "qop": 4}


# ---------------------------------------------------------------------------
# Data loading — preprocessed memmap
# ---------------------------------------------------------------------------


def load_targets_from_preprocessed(
    preprocessed_dir: Path,
    num_shards: int = -1,
) -> dict[str, np.ndarray]:
    """Load track targets and per-track hit counts from preprocessed memmap shards.

    Returns dict with keys d0, z0, phi, theta, qop, nhits, pt, eta.
    Derived quantities (pt, eta) are computed from theta and qop.
    """
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    all_targets = []
    all_nhits = []
    for sd in tqdm(shard_dirs, desc="Loading targets"):
        tgt_path = sd / "selected_tracks" / "track_targets.npy"
        off_path = sd / "selected_tracks" / "track_hit_offsets.npy"
        if tgt_path.exists():
            arr = np.load(tgt_path)  # (N, 5)
            if len(arr) > 0:
                all_targets.append(arr)
                if off_path.exists():
                    offsets = np.load(off_path)  # (N+1,)
                    nhits = np.diff(offsets).astype(np.int32)
                    all_nhits.append(nhits)

    if not all_targets:
        raise ValueError(f"No targets found in {preprocessed_dir}")

    targets = np.concatenate(all_targets, axis=0)  # (total, 5)
    nhits = np.concatenate(all_nhits, axis=0) if all_nhits else np.zeros(len(targets), dtype=np.int32)
    print(f"Loaded {len(targets):,} tracks from {len(shard_dirs)} shards")

    # Derive pT and eta from theta and qop
    theta_vals = targets[:, 3]
    qop_vals = targets[:, 4]
    p = np.where(np.abs(qop_vals) > 1e-12, 1.0 / np.abs(qop_vals), 0.0)
    pt = p * np.sin(theta_vals)
    eta = -np.log(np.tan(theta_vals / 2.0 + 1e-12))

    return {
        "d0": targets[:, 0],
        "z0": targets[:, 1],
        "phi": targets[:, 2],
        "theta": targets[:, 3],
        "qop": targets[:, 4],
        "nhits": nhits,
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Spline fitting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-parameter knot placement strategies
# ---------------------------------------------------------------------------

# Default tail quantiles (used for z0, theta, and as fallback)
_DEFAULT_TAIL_Q = [0.001, 0.005, 0.01, 0.02, 0.98, 0.99, 0.995, 0.999]

# d0: very peaked at zero — need much denser tails to resolve the edges
_D0_TAIL_Q = [
    0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007,
    0.01, 0.015, 0.02, 0.03,
    0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997,
    0.998, 0.999, 0.9995, 0.9999,
]

# qop: bimodal (positive/negative charge) — need denser middle where CDF
# transitions rapidly between the two charge peaks
_QOP_TAIL_Q = _DEFAULT_TAIL_Q  # keep the original tails
_QOP_EXTRA_Q = [
    0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.48, 0.49,
    0.51, 0.52, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65,
]

# Registry: param_name -> (num_core_knots, tail_quantiles, extra_quantiles)
PARAM_KNOT_CONFIG: dict[str, tuple[int, list[float], list[float]]] = {
    "d0":    (35, _D0_TAIL_Q,      []),
    "z0":    (25, _DEFAULT_TAIL_Q,  []),
    "theta": (25, _DEFAULT_TAIL_Q,  []),
    "qop":   (35, _QOP_TAIL_Q,      _QOP_EXTRA_Q),
}


def compute_quantile_knots(
    values: np.ndarray,
    num_knots: int = 25,
    tail_quantiles: list[float] | None = None,
    extra_quantiles: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute knot positions at quantile boundaries.

    Parameters
    ----------
    values : array
        1-D data samples.
    num_knots : int
        Number of uniformly-spaced core quantiles (0 to 1 inclusive).
    tail_quantiles : list[float] | None
        Additional quantile positions for the distribution tails.
    extra_quantiles : list[float] | None
        Any other extra quantile positions (e.g. densifying the middle).
    """
    if tail_quantiles is None:
        tail_quantiles = _DEFAULT_TAIL_Q
    if extra_quantiles is None:
        extra_quantiles = []

    core_q = np.linspace(0.0, 1.0, num_knots)
    all_q = np.unique(np.concatenate([core_q, tail_quantiles, extra_quantiles]))
    all_q = np.clip(all_q, 0.0, 1.0)
    all_q = np.sort(all_q)

    knot_x = np.quantile(values, all_q)

    # Remove duplicates
    _, unique_idx = np.unique(knot_x, return_index=True)
    unique_idx = np.sort(unique_idx)
    knot_x = knot_x[unique_idx]
    knot_y = all_q[unique_idx]
    knot_y[0] = 0.0
    knot_y[-1] = 1.0

    return knot_x, knot_y


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_spline_diagnostics(
    values: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    param_name: str,
    output_dir: Path,
    units: str = "",
) -> None:
    """Generate 3-panel diagnostic plot for a spline fit."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Spline fit: {param_name}", fontsize=14, fontweight="bold")

    unit_str = f" [{units}]" if units else ""

    # -- Panel 1: CDF comparison --
    ax = axes[0]
    sorted_vals = np.sort(values)
    ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fine_x = np.linspace(knot_x[0], knot_x[-1], 2000)
    spline_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)

    ax.plot(sorted_vals, ecdf_y, "b-", alpha=0.4, linewidth=0.5, label="Empirical CDF")
    ax.plot(fine_x, spline_y, "r-", linewidth=2, label="PCHIP spline")
    ax.plot(knot_x, knot_y, "ko", markersize=5, label=f"Knots (n={len(knot_x)})", zorder=5)
    ax.set_xlabel(f"{param_name}{unit_str}")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF vs Spline Fit")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Panel 2: Transformed histogram --
    ax = axes[1]
    transformed = evaluate_pchip(values, knot_x, knot_y, slopes)
    ax.hist(transformed, bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.5, label="Ideal uniform")
    ax.set_xlabel(f"Transformed {param_name}")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Transformed Values")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # -- Panel 3: Residuals --
    ax = axes[2]
    spline_at_data = evaluate_pchip(sorted_vals, knot_x, knot_y, slopes)
    residuals = ecdf_y - spline_at_data

    ax.plot(sorted_vals, residuals, "b-", linewidth=0.5, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.fill_between(sorted_vals, residuals, alpha=0.2, color="steelblue")
    ax.set_xlabel(f"{param_name}{unit_str}")
    ax.set_ylabel("Residual (ECDF - Spline)")
    ax.set_title("Fit Residuals")
    ax.grid(True, alpha=0.3)

    rmse = np.sqrt(np.mean(residuals**2))
    max_err = np.max(np.abs(residuals))
    stats_text = f"RMSE: {rmse:.6f}\nMax |err|: {max_err:.6f}\nN tracks: {len(values):,}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})

    plt.tight_layout()
    fig.savefig(output_dir / f"spline_fit_{param_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved: spline_fit_{param_name}.png  (RMSE={rmse:.6f}, max|err|={max_err:.6f})")


def plot_parameter_distributions(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot raw distributions of all parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Track Parameter Distributions (after selection)", fontsize=14, fontweight="bold")

    params = [
        ("d0", "mm"), ("z0", "mm"), ("phi", "rad"), ("theta", "rad"), ("qop", "e/GeV"),
    ]
    for idx, (name, units) in enumerate(params):
        ax = axes.flat[idx]
        vals = data[name]
        ax.hist(vals, bins=100, density=True, alpha=0.7, color="steelblue",
                edgecolor="black", linewidth=0.3)
        ax.set_xlabel(f"{name} [{units}]")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}  (u={np.mean(vals):.4f}, s={np.std(vals):.4f})")
        ax.grid(True, alpha=0.3)

    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: parameter_distributions.png")


def plot_selection_distributions(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot distributions of track selection variables (pT, eta, nhits).

    These show the kinematic profile of the tracks passing the selection cuts:
    primary, hard-scatter, charged, pT >= 0.5 GeV, |eta| <= 3, nhits >= 6,
    finite perigee parameters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Selected Track Distributions\n"
        r"(primary, hard-scatter, charged, $p_T \geq 0.5$ GeV, $|\eta| \leq 3$, $n_\mathrm{hits} \geq 6$)",
        fontsize=13, fontweight="bold",
    )

    # -- pT --
    ax = axes[0, 0]
    pt = data["pt"]
    ax.hist(pt, bins=np.linspace(0, 20, 120), density=True, alpha=0.7,
            color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title(f"Transverse momentum  (median={np.median(pt):.2f} GeV)")
    ax.axvline(0.5, color="r", ls="--", lw=1.5, label=r"$p_T$ cut = 0.5 GeV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- pT log scale --
    ax = axes[0, 1]
    logpt = np.log10(np.clip(pt, 1e-3, None))
    ax.hist(logpt, bins=100, density=True, alpha=0.7,
            color="darkorange", edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"$\log_{10}(p_T$ / GeV$)$")
    ax.set_ylabel("Density")
    ax.set_title(r"$\log_{10}(p_T)$ distribution")
    ax.axvline(np.log10(0.5), color="r", ls="--", lw=1.5, label="cut")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- eta --
    ax = axes[0, 2]
    eta = data["eta"]
    ax.hist(eta, bins=100, density=True, alpha=0.7,
            color="seagreen", edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Density")
    ax.set_title(f"Pseudorapidity  (mean={np.mean(eta):.3f})")
    ax.axvline(-3.0, color="r", ls="--", lw=1.5, label=r"$|\eta|$ cut = 3")
    ax.axvline(3.0, color="r", ls="--", lw=1.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- nhits --
    ax = axes[1, 0]
    nhits = data["nhits"]
    max_nh = min(int(np.percentile(nhits, 99.5)) + 5, 60)
    ax.hist(nhits, bins=np.arange(0, max_nh + 1) - 0.5, density=True, alpha=0.7,
            color="mediumpurple", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Number of hits per track")
    ax.set_ylabel("Density")
    ax.set_title(f"Hit multiplicity  (mean={np.mean(nhits):.1f}, max={np.max(nhits)})")
    ax.axvline(6, color="r", ls="--", lw=1.5, label="min hits cut = 6")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- pT vs eta 2D --
    ax = axes[1, 1]
    h = ax.hist2d(eta, np.clip(pt, 0, 15), bins=[80, 80],
                  cmap="viridis", norm=matplotlib.colors.LogNorm())
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$p_T$ [GeV]")
    ax.set_title(r"$p_T$ vs $\eta$")
    plt.colorbar(h[3], ax=ax, label="Counts")

    # -- nhits vs eta --
    ax = axes[1, 2]
    h2 = ax.hist2d(eta, nhits.astype(float), bins=[80, np.arange(0, max_nh + 1) - 0.5],
                   cmap="magma", norm=matplotlib.colors.LogNorm())
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Number of hits")
    ax.set_title(r"Hit multiplicity vs $\eta$")
    plt.colorbar(h2[3], ax=ax, label="Counts")

    plt.tight_layout()
    fig.savefig(output_dir / "selection_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: selection_distributions.png")


def plot_qq_uniform(
    data: dict[str, np.ndarray],
    spline_fits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Q-Q plot: transformed values vs theoretical uniform quantiles.

    If the spline perfectly captures the CDF, the Q-Q plot is a straight
    diagonal line.  Deviations reveal regions where the spline under/over-
    represents the empirical distribution.
    """
    params = [p for p in spline_fits]
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Q-Q Plot: Spline-Transformed vs Uniform(0,1)", fontsize=13, fontweight="bold")

    for ax, pname in zip(axes, params):
        knot_x, knot_y, slopes = spline_fits[pname]
        transformed = evaluate_pchip(data[pname], knot_x, knot_y, slopes)
        transformed_sorted = np.sort(transformed)
        n_pts = len(transformed_sorted)
        theoretical = np.linspace(0, 1, n_pts)

        # Subsample for plotting if too many points
        step = max(1, n_pts // 5000)
        ax.scatter(theoretical[::step], transformed_sorted[::step],
                   s=1, alpha=0.3, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Ideal")
        ax.set_xlabel("Theoretical uniform quantiles")
        ax.set_ylabel("Empirical (transformed) quantiles")
        ax.set_title(pname)

        # KS statistic
        ks_stat = np.max(np.abs(transformed_sorted - theoretical))
        ax.text(0.05, 0.92, f"KS = {ks_stat:.5f}", transform=ax.transAxes,
                fontsize=10, bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
        ax.legend(fontsize=9, loc="lower right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "eval_qq_uniform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: eval_qq_uniform.png")


def plot_spline_derivatives(
    data: dict[str, np.ndarray],
    spline_fits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plot spline derivatives (≈ estimated PDF) vs histogram density.

    The derivative dCDF/dx of the monotonic spline is an estimate of the PDF.
    Overlaying it on the empirical histogram validates whether the spline
    faithfully captures the shape of the distribution.
    """
    params = list(spline_fits.keys())
    units_map = {"d0": "mm", "z0": "mm", "theta": "rad", "qop": "e/GeV"}
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Spline Derivative (≈ PDF) vs Empirical Histogram", fontsize=13, fontweight="bold")

    for ax, pname in zip(axes, params):
        knot_x, knot_y, slopes = spline_fits[pname]
        vals = data[pname]
        unit_str = f" [{units_map.get(pname, '')}]"

        # Histogram
        ax.hist(vals, bins=150, density=True, alpha=0.4, color="steelblue",
                edgecolor="none", label="Histogram")

        # Numerical derivative of spline
        fine_x = np.linspace(knot_x[0], knot_x[-1], 5000)
        fine_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)
        dx = fine_x[1] - fine_x[0]
        deriv = np.gradient(fine_y, dx)
        ax.plot(fine_x, deriv, "r-", lw=2, label="Spline dCDF/dx")

        ax.set_xlabel(f"{pname}{unit_str}")
        ax.set_ylabel("Density")
        ax.set_title(pname)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "eval_spline_pdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: eval_spline_pdf.png")


def plot_calibration(
    data: dict[str, np.ndarray],
    spline_fits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Calibration plot: expected vs observed coverage at each quantile level.

    For a perfect CDF spline, the fraction of data below quantile q should
    be exactly q.  This plot checks that across 100 evenly spaced quantile
    levels.
    """
    params = list(spline_fits.keys())
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle("Calibration: Expected vs Observed Quantile Coverage", fontsize=13, fontweight="bold")

    q_levels = np.linspace(0.01, 0.99, 100)
    colors = {"d0": "tab:blue", "z0": "tab:orange", "theta": "tab:green", "qop": "tab:red"}

    for pname in params:
        knot_x, knot_y, slopes = spline_fits[pname]
        vals = data[pname]
        transformed = evaluate_pchip(vals, knot_x, knot_y, slopes)

        observed_fracs = np.array([np.mean(transformed <= q) for q in q_levels])
        ax.plot(q_levels, observed_fracs, "-", lw=1.8,
                color=colors.get(pname, "gray"), label=pname)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("Expected quantile level")
    ax.set_ylabel("Observed fraction below level")
    ax.set_title("Quantile Calibration (all 4 spline-fitted parameters)")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(output_dir / "eval_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: eval_calibration.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fit monotonic splines to ColliderML track parameter distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preprocessed-dir", type=str,
        default="/scratch/colliderml/p0_preprocessed",
        help="Preprocessed memmap directory (from preprocess_colliderml.py)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Output directory for spline configs and plots (default: auto-detect config/splines)",
    )
    parser.add_argument("--num-shards", type=int, default=-1,
                        help="Number of preprocessed shards to read (-1 for all)")
    parser.add_argument("--num-knots", type=int, default=25,
                        help="Number of core (equi-spaced) knots")

    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)

    if args.output_dir is None:
        # Auto-detect: relative to this script
        output_dir = Path(__file__).resolve().parent.parent / "config" / "splines"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Spline Fitting for Track Parameter Regression")
    print("=" * 70)
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Num shards:       {args.num_shards}")
    print(f"  Num knots:        {args.num_knots}")
    print()

    # Load data from preprocessed format
    data = load_targets_from_preprocessed(preprocessed_dir, num_shards=args.num_shards)

    if len(data["d0"]) == 0:
        print("ERROR: No tracks found. Check preprocessed directory.")
        return

    # Apply global filter cuts (same as dataset loading)
    n_before = len(data["d0"])
    keep = (
        (data["d0"] >= -1.0) & (data["d0"] <= 1.0)
        & (data["z0"] >= -150.0) & (data["z0"] <= 150.0)
    )
    for key in data:
        data[key] = data[key][keep]
    n_after = len(data["d0"])
    print(f"Global filter: |d0| <= 1 mm, |z0| <= 150 mm")
    print(f"  {n_before:,} -> {n_after:,} tracks ({n_before - n_after:,} removed, {(n_before - n_after)/n_before*100:.2f}%)")
    print()

    # Plot raw target distributions
    plot_parameter_distributions(data, output_dir)

    # Plot selection variable distributions (pT, eta, nhits, 2D correlations)
    plot_selection_distributions(data, output_dir)

    # Fit splines for d0, z0, theta, qop (NOT phi — phi uses circular loss)
    params_to_fit = {
        "d0": {"units": "mm"},
        "z0": {"units": "mm"},
        "theta": {"units": "rad"},
        "qop": {"units": "e/GeV"},
    }

    norm_ranges = {}
    spline_fits = {}   # pname → (knot_x, knot_y, slopes) for eval plots

    print("\n--- Fitting splines ---")
    for param_name, meta in params_to_fit.items():
        print(f"\n  {param_name}:")
        values = data[param_name]

        norm_ranges[param_name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
        }

        n_core, tail_q, extra_q = PARAM_KNOT_CONFIG.get(
            param_name, (args.num_knots, None, None)
        )
        # CLI --num-knots overrides the default core count only if
        # the param has no custom config entry
        if param_name not in PARAM_KNOT_CONFIG:
            n_core = args.num_knots
        knot_x, knot_y = compute_quantile_knots(
            values, num_knots=n_core,
            tail_quantiles=tail_q, extra_quantiles=extra_q,
        )
        slopes = fritsch_carlson_slopes(knot_x, knot_y)
        spline_fits[param_name] = (knot_x, knot_y, slopes)

        print(f"    Knots: {len(knot_x)}  (range: [{knot_x[0]:.4f}, {knot_x[-1]:.4f}])")

        spline_config = {
            "name": param_name,
            "units": meta["units"],
            "knot_x": [float(v) for v in knot_x],
            "knot_y": [float(v) for v in knot_y],
            "num_tracks": int(len(values)),
        }
        config_path = output_dir / f"spline_{param_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(spline_config, f, default_flow_style=False, sort_keys=False)
        print(f"    Config saved: {config_path}")

        plot_spline_diagnostics(
            values, knot_x, knot_y, slopes,
            param_name=param_name,
            output_dir=output_dir,
            units=meta["units"],
        )

    # Evaluation plots
    print("\n--- Generating evaluation plots ---")
    plot_qq_uniform(data, spline_fits, output_dir)
    plot_spline_derivatives(data, spline_fits, output_dir)
    plot_calibration(data, spline_fits, output_dir)

    # Save normalisation ranges (including phi)
    norm_path = output_dir / "normalisation_ranges.yaml"
    norm_ranges["phi"] = {
        "min": float(np.min(data["phi"])),
        "max": float(np.max(data["phi"])),
        "mean": float(np.mean(data["phi"])),
        "std": float(np.std(data["phi"])),
    }
    with open(norm_path, "w") as f:
        yaml.dump(norm_ranges, f, default_flow_style=False, sort_keys=False)
    print(f"\nNormalisation ranges saved: {norm_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for p, r in norm_ranges.items():
        print(f"  {p:8s}:  range=[{r['min']:.4f}, {r['max']:.4f}]  "
              f"mean={r['mean']:.4f}  std={r['std']:.4f}")
    print()


if __name__ == "__main__":
    main()
