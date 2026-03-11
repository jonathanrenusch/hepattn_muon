#!/usr/bin/env python3
"""Evaluate track regression predictions: precision and pull plots.

Reads the HDF5 file produced by ``RegressionPredictionWriter`` and generates:

1. **Precision plots** — σ(residual) vs η for each track parameter
   (d0, z0, phi, theta, qop), matching the style from
   ``evaluate_acts_tracking.py``.

2. **Pull plots** — (pred − truth) / σ(residual) histograms fitted with
   a Gaussian. Ideal pulls are N(0, 1).

Usage::

    python -m hepattn.experiments.colliderml_regr.evaluate_predictions \
        --predictions /path/to/test_predictions.h5 \
        --output-dir /path/to/eval_output

The testing dataset already contains preselected particles, so no additional
track selection is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ============================================================================
# Data loading
# ============================================================================

def load_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load predictions and targets from HDF5 file.

    Returns ``{"preds": {name: array}, "targets": {name: array}}``.
    """
    data: dict[str, dict[str, np.ndarray]] = {"preds": {}, "targets": {}}
    with h5py.File(path, "r") as f:
        for group in ("preds", "targets"):
            for name in f[group]:
                data[group][name] = f[group][name][:]
    return data


# ============================================================================
# Residuals
# ============================================================================

def compute_residuals(data: dict) -> dict[str, np.ndarray]:
    """Compute pred − truth residuals and truth eta for binning."""
    residuals: dict[str, np.ndarray] = {}
    for name in ["d0", "z0", "phi", "theta", "qop"]:
        residuals[name] = data["preds"][name] - data["targets"][name]

    # Wrap phi residual to [-π, π]
    residuals["phi"] = (residuals["phi"] + np.pi) % (2 * np.pi) - np.pi

    # Truth eta from truth theta
    theta_truth = data["targets"]["theta"]
    residuals["eta"] = -np.log(np.tan(theta_truth / 2.0 + 1e-12))

    return residuals


# ============================================================================
# Precision vs η
# ============================================================================

UNIT_SCALE = {
    "d0": 1.0,       # mm
    "z0": 1.0,       # mm
    "phi": 1e3,      # rad → mrad
    "theta": 1e3,    # rad → mrad
    "qop": 1.0,      # 1/GeV
}

PARAM_LABELS = {
    "d0": r"$\sigma(d_0)$ [mm]",
    "z0": r"$\sigma(z_0)$ [mm]",
    "phi": r"$\sigma(\phi)$ [mrad]",
    "theta": r"$\sigma(\theta)$ [mrad]",
    "qop": r"$\sigma(q/p)$ [1/GeV]",
}


def compute_precision_vs_eta(
    residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Compute binned precision (std of residual) vs η.

    Returns ``(eta_bins, {param: {eta_centers, std, std_err, count, unbinned_std}})``.
    """
    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta = residuals["eta"]
    result: dict[str, dict[str, np.ndarray]] = {}

    for name in ["d0", "z0", "phi", "theta", "qop"]:
        res = residuals[name]
        min_len = min(len(eta), len(res))
        e, r = eta[:min_len], res[:min_len]

        unbinned_std = float(np.std(r))

        centers, stds, counts = [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (e >= eta_bins[i]) & (e < eta_bins[i + 1])
            n = int(np.sum(mask))
            if n > 2:
                centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
                stds.append(np.std(r[mask]))
                counts.append(n)

        stds_arr = np.array(stds)
        counts_arr = np.array(counts, dtype=float)

        result[name] = {
            "eta_centers": np.array(centers),
            "std": stds_arr,
            "std_err": stds_arr / np.sqrt(2 * counts_arr),
            "count": counts_arr,
            "unbinned_std": unbinned_std,
        }

    return eta_bins, result


def _build_step_arrays(bin_edges: np.ndarray, values: np.ndarray):
    """Convert bin centers/values to step-plot compatible arrays."""
    x = np.repeat(bin_edges, 2)
    y = np.repeat(values, 2)
    y = np.concatenate([[y[0]], y, [y[-1]]])
    return x, y[: len(x)]


def plot_precision_vs_eta(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot precision (σ of residual) vs η for each parameter."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in precision_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        centers = data["eta_centers"]
        std = data["std"]
        std_err = data["std_err"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        # Build step arrays from bin edges that correspond to the non-empty bins
        bin_indices = np.searchsorted(eta_bins, centers, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(eta_bins) - 2)
        edges = np.append(eta_bins[bin_indices], eta_bins[bin_indices[-1] + 1])

        x, y = _build_step_arrays(edges, std * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                label=f"Precision (avg σ: {unbinned:.4f})")

        # Error band
        lo = np.maximum((std - std_err) * scale, 0)
        hi = (std + std_err) * scale
        x_fill = np.repeat(edges, 2)[1:-1]
        lo_fill = np.repeat(lo, 2)
        hi_fill = np.repeat(hi, 2)
        ax.fill_between(x_fill, lo_fill, hi_fill, alpha=0.25, color="steelblue",
                        label="Uncertainty")

        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$", fontsize=12)
        ax.set_ylabel(PARAM_LABELS.get(name, f"σ({name})"), fontsize=12)
        ax.set_title(f"{name.upper()} Precision vs $\\eta$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_precision_vs_eta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary plot with all parameters
    _plot_summary_precision(precision_data, eta_bins, output_dir)


def _plot_summary_precision(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
) -> None:
    """Summary 2×3 panel with all precision vs η curves."""
    params = [p for p in ["d0", "z0", "phi", "theta", "qop"] if p in precision_data]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        data = precision_data[name]
        centers = data["eta_centers"]
        std = data["std"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        bin_indices = np.searchsorted(eta_bins, centers, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(eta_bins) - 2)
        edges = np.append(eta_bins[bin_indices], eta_bins[bin_indices[-1] + 1])

        x, y = _build_step_arrays(edges, std * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2)
        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        ax.set_ylabel(PARAM_LABELS.get(name, f"σ({name})"))
        ax.set_title(f"{name} (avg σ: {unbinned:.4f})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Track Parameter Precision vs η", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_precision_vs_eta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Pull distributions
# ============================================================================

PULL_LABELS = {
    "d0": r"$d_0$ pull",
    "z0": r"$z_0$ pull",
    "phi": r"$\phi$ pull",
    "theta": r"$\theta$ pull",
    "qop": r"$q/p$ pull",
}


def compute_and_plot_pulls(
    residuals: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 100,
    pull_range: tuple[float, float] = (-5.0, 5.0),
) -> None:
    """Compute pseudo-pulls normalised by global σ and plot distributions.

    Pull = residual / σ(residual). For well-calibrated uncertainties this
    should be approximately N(0,1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pull_data: dict[str, np.ndarray] = {}
    for name in ["d0", "z0", "phi", "theta", "qop"]:
        res = residuals[name]
        sigma = np.std(res)
        if sigma > 0:
            pull_data[name] = res / sigma
        else:
            pull_data[name] = np.zeros_like(res)

    # Individual pull plots
    for name, pulls in pull_data.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        finite = pulls[np.isfinite(pulls)]
        clipped = finite[(finite >= pull_range[0]) & (finite <= pull_range[1])]

        counts, bin_edges, _ = ax.hist(
            clipped, bins=n_bins, range=pull_range, density=True,
            alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.5,
            label="Pseudo-pulls",
        )

        # Fit Gaussian
        mu, sigma = norm.fit(clipped)
        x_fit = np.linspace(pull_range[0], pull_range[1], 300)
        ax.plot(x_fit, norm.pdf(x_fit, mu, sigma), "r-", linewidth=2,
                label=f"Gaussian fit: μ={mu:.3f}, σ={sigma:.3f}")

        # Reference N(0,1)
        ax.plot(x_fit, norm.pdf(x_fit, 0, 1), "k--", linewidth=1.5,
                alpha=0.5, label="N(0, 1) reference")

        ax.set_xlabel(PULL_LABELS.get(name, f"{name} pull"), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{name.upper()} Pull Distribution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_pull.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary pull plot
    _plot_summary_pulls(pull_data, pull_range, n_bins, output_dir)


def _plot_summary_pulls(
    pull_data: dict[str, np.ndarray],
    pull_range: tuple[float, float],
    n_bins: int,
    output_dir: Path,
) -> None:
    """Summary panel with all pull distributions."""
    params = [p for p in ["d0", "z0", "phi", "theta", "qop"] if p in pull_data]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        pulls = pull_data[name]
        finite = pulls[np.isfinite(pulls)]
        clipped = finite[(finite >= pull_range[0]) & (finite <= pull_range[1])]

        ax.hist(clipped, bins=n_bins, range=pull_range, density=True,
                alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.5)

        mu, sigma = norm.fit(clipped)
        x_fit = np.linspace(pull_range[0], pull_range[1], 200)
        ax.plot(x_fit, norm.pdf(x_fit, mu, sigma), "r-", linewidth=1.5)
        ax.plot(x_fit, norm.pdf(x_fit, 0, 1), "k--", linewidth=1, alpha=0.5)

        ax.set_xlabel(PULL_LABELS.get(name, name))
        ax.set_title(f"{name} (μ={mu:.2f}, σ={sigma:.2f})")
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Pull Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_pulls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate track regression predictions: precision and pull plots"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to test_predictions.h5 written by RegressionPredictionWriter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots (default: same dir as predictions file)",
    )
    parser.add_argument("--eta-min", type=float, default=-3.0)
    parser.add_argument("--eta-max", type=float, default=3.0)
    parser.add_argument("--n-eta-bins", type=int, default=30)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "eval_plots"

    print(f"Loading predictions from {pred_path}")
    data = load_predictions(pred_path)

    n_tracks = len(data["preds"]["d0"])
    print(f"  Loaded {n_tracks:,} tracks")

    # Compute residuals
    residuals = compute_residuals(data)

    # Precision vs eta
    print("Computing precision vs η ...")
    precision_dir = output_dir / "precision"
    eta_bins, precision_data = compute_precision_vs_eta(
        residuals, eta_range=(args.eta_min, args.eta_max), n_eta_bins=args.n_eta_bins,
    )
    plot_precision_vs_eta(precision_data, eta_bins, precision_dir)
    print(f"  Precision plots saved to {precision_dir}")

    # Print summary
    print("\n  Parameter precisions (unbinned σ):")
    for name, pdata in precision_data.items():
        scale = UNIT_SCALE.get(name, 1.0)
        unit = {1.0: "", 1e3: " mrad"}.get(scale, "")
        print(f"    {name:6s}: σ = {pdata['unbinned_std'] * scale:.5f}{unit}")

    # Pull distributions
    print("\nComputing pull distributions ...")
    pull_dir = output_dir / "pulls"
    compute_and_plot_pulls(residuals, pull_dir)
    print(f"  Pull plots saved to {pull_dir}")

    print(f"\nAll evaluation plots saved to {output_dir}")


if __name__ == "__main__":
    main()
