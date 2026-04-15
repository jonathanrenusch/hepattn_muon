#!/usr/bin/env python3
"""Quantile alignment diagnostics for track regression predictions.

Assesses whether the quantile loss is correctly calibrated by comparing
predicted quantile levels against empirical coverage on the test set.

Generates:
- Truth distributions with predicted quantile positions (vertical lines)
- Quantile calibration curves (observed vs expected coverage)
- PIT (probability integral transform) histograms
- Quantile interval width vs η profiles
- Quantile interval width vs pT profiles

Usage::

    python -m hepattn.experiments.colliderml_regr.evaluate_quantile_alignment \
        --predictions /path/to/test_predictions.h5 \
        --output-dir /path/to/eval_output \
        [--data-dir /scratch/colliderml/p200_preprocessed_plus_qcd/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    FULL_RANGE_PARAMS,
    PARAM_VALUE_LABELS,
    PARAMS,
    UNIT_SCALE,
    load_predictions,
)


# ============================================================================
# Colours for quantile levels
# ============================================================================

# Symmetric pairs share colours; the median is distinct.
QUANTILE_COLORS = {
    0.05: "#1b9e77",
    0.10: "#d95f02",
    0.25: "#7570b3",
    0.50: "#e7298a",
    0.75: "#7570b3",
    0.90: "#d95f02",
    0.95: "#1b9e77",
}

QUANTILE_LINESTYLES = {
    0.05: "--",
    0.10: "--",
    0.25: "--",
    0.50: "-",
    0.75: ":",
    0.90: ":",
    0.95: ":",
}


def _quantile_color(tau: float) -> str:
    closest = min(QUANTILE_COLORS, key=lambda k: abs(k - tau))
    if abs(closest - tau) < 0.02:
        return QUANTILE_COLORS[closest]
    return "gray"


def _quantile_ls(tau: float) -> str:
    closest = min(QUANTILE_LINESTYLES, key=lambda k: abs(k - tau))
    if abs(closest - tau) < 0.02:
        return QUANTILE_LINESTYLES[closest]
    return "--" if tau < 0.5 else (":" if tau > 0.5 else "-")


# ============================================================================
# 1. Truth distribution with quantile vertical lines (log y, step histogram)
# ============================================================================

def plot_truth_distribution_with_quantiles(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 120,
) -> None:
    """Plot truth distributions (log-y step histogram) with vertical lines at
    the mean predicted quantile position for each quantile level."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name]
        q = quantiles[name]           # (N, Q)
        levels = quantile_levels[name]  # (Q,)

        # Determine bin range
        if name in FULL_RANGE_PARAMS:
            lo, hi = float(np.min(t)), float(np.max(t))
        else:
            lo = float(np.percentile(t, 0.1))
            hi = float(np.percentile(t, 99.9))
        bins = np.linspace(lo, hi, n_bins + 1)

        # Step histogram of truth
        counts, _ = np.histogram(t, bins=bins)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        # Draw as step function
        ax.step(bin_centres, counts, where="mid", color="black", linewidth=1.5, label="Truth")
        ax.set_yscale("log")
        ymin = max(1, counts[counts > 0].min() * 0.5) if np.any(counts > 0) else 1
        ymax = counts.max() * 5
        ax.set_ylim(ymin, ymax)

        # Vertical lines at mean quantile prediction per level
        for j, tau in enumerate(levels):
            q_mean = float(np.mean(q[:, j]))
            color = _quantile_color(tau)
            ls = _quantile_ls(tau)
            ax.axvline(q_mean, color=color, linestyle=ls, linewidth=1.4,
                       label=f"$\\tau={tau:.2f}$  (mean={q_mean:.4g})")

        ax.set_xlabel(PARAM_VALUE_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Truth Distribution with Mean Predicted Quantile Positions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "truth_dist_quantile_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved truth_dist_quantile_lines.png")


# ============================================================================
# 2. Per-track quantile overlay on truth (sampled subset, zoomed)
# ============================================================================

def plot_quantile_spread_samples(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_samples: int = 200,
    seed: int = 42,
) -> None:
    """For a random subset of tracks, draw horizontal error bars spanning the
    quantile spread overlaid on the truth value.  Gives a visual sense of
    per-track uncertainty calibration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        return

    rng = np.random.default_rng(seed)
    n_total = len(next(iter(targets.values())))
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    idx.sort()

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name][idx]
        q = quantiles[name][idx]       # (n_samples, Q)
        levels = quantile_levels[name]

        # Sort by truth value for readability
        order = np.argsort(t)
        t = t[order]
        q = q[order]
        y_pos = np.arange(len(t))

        # Find widest symmetric pair for error bars
        median_idx = np.argmin(np.abs(levels - 0.5))
        q_med = q[:, median_idx]

        # Colour tracks by whether truth is inside the widest interval
        lo_idx, hi_idx = 0, len(levels) - 1
        inside = (t >= q[:, lo_idx]) & (t <= q[:, hi_idx])

        # Draw interval and truth
        ax.hlines(y_pos, q[:, lo_idx], q[:, hi_idx], colors=np.where(inside, "steelblue", "salmon"),
                  alpha=0.5, linewidth=0.8)
        ax.scatter(t, y_pos, s=4, color="black", zorder=3, label="Truth")
        ax.scatter(q_med, y_pos, s=3, color="steelblue", marker="|", zorder=2, label="Median pred")

        coverage = inside.mean()
        expected = levels[hi_idx] - levels[lo_idx]
        ax.set_xlabel(PARAM_VALUE_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Track index (sorted by truth)", fontsize=9)
        ax.set_title(f"{name.upper()} — [{levels[lo_idx]:.0%},{levels[hi_idx]:.0%}] "
                     f"coverage: {coverage:.1%} (expected {expected:.0%})", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Quantile Spread per Track (random subset)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "quantile_spread_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved quantile_spread_samples.png")


# ============================================================================
# 3. Quantile calibration curve (observed coverage vs expected)
# ============================================================================

def plot_quantile_calibration(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot observed quantile coverage vs nominal level for each parameter.

    A perfectly calibrated model sits on the diagonal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name]              # (N,)
        q = quantiles[name]            # (N, Q)
        levels = quantile_levels[name]  # (Q,)

        observed = np.array([np.mean(t <= q[:, j]) for j in range(len(levels))])

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        ax.plot(levels, observed, "o-", color="steelblue", markersize=5, linewidth=1.5, label=name.upper())

        # Annotate deviations
        for j, (lev, obs) in enumerate(zip(levels, observed)):
            delta = obs - lev
            ax.annotate(f"{delta:+.1%}", (lev, obs), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("Nominal quantile level $\\tau$", fontsize=11)
        ax.set_ylabel("Observed coverage $P(y \\leq \\hat{q}_\\tau)$", fontsize=11)
        ax.set_title(f"{name.upper()} Calibration", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Quantile Calibration — Observed vs Nominal Coverage", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "quantile_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved quantile_calibration.png")


# ============================================================================
# 4. PIT histogram (probability integral transform)
# ============================================================================

def plot_pit_histogram(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 50,
) -> None:
    """Plot the PIT histogram for each parameter.

    The PIT value for each track is the linearly interpolated quantile level
    at which the predicted CDF equals the truth value.  A uniform PIT
    histogram indicates perfect calibration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name]               # (N,)
        q = quantiles[name]             # (N, Q)
        levels = quantile_levels[name]  # (Q,)

        # Vectorised PIT: for each track, count how many quantile
        # predictions are below the truth value to estimate the CDF position.
        # Compare truth against each quantile: (N, Q) bool
        below = (q <= t[:, None])  # True where quantile pred <= truth
        # Count quantiles below truth and convert to a fractional level
        n_below = below.sum(axis=1)  # (N,) int in [0, Q]
        # Linearly interpolate: clamp to [levels[0], levels[-1]]
        Q = len(levels)
        # Map count to a level: 0 → levels[0], Q → levels[-1]
        pit = np.interp(n_below, np.arange(Q + 1),
                        np.concatenate([[levels[0]], levels]))
        pit = pit.astype(np.float32)

        bins = np.linspace(float(levels[0]), float(levels[-1]), n_bins + 1)
        counts, _ = np.histogram(pit, bins=bins)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])

        # Expected uniform count
        expected = len(t) * (bins[1] - bins[0]) / (levels[-1] - levels[0])

        ax.step(bin_centres, counts, where="mid", color="steelblue", linewidth=1.5, label="PIT")
        ax.axhline(expected, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Uniform")

        ax.set_xlabel("PIT value", fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(f"{name.upper()} — PIT Histogram", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Probability Integral Transform — Uniformity Check", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "pit_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pit_histogram.png")


# ============================================================================
# 5. Quantile interval width vs η
# ============================================================================

def plot_quantile_width_vs_eta(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_eta_bins: int = 30,
) -> None:
    """Plot the mean quantile interval width as a function of η for symmetric
    quantile pairs (e.g. [5%,95%], [10%,90%], [25%,75%])."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute η from theta truth
    theta_truth = targets.get("theta")
    if theta_truth is None:
        return
    eta = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

    eta_edges = np.linspace(float(np.percentile(eta, 0.5)), float(np.percentile(eta, 99.5)), n_eta_bins + 1)
    eta_centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    params = [p for p in PARAMS if p in quantiles]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    pair_colors = ["#1b9e77", "#d95f02", "#7570b3"]

    for i, name in enumerate(params):
        ax = axes[i]
        q = quantiles[name]            # (N, Q)
        levels = quantile_levels[name]  # (Q,)
        scale = UNIT_SCALE.get(name, 1.0)

        # Find symmetric pairs around 0.5
        pairs = []
        for j_lo in range(len(levels) // 2):
            j_hi = len(levels) - 1 - j_lo
            if j_lo < j_hi:
                pairs.append((j_lo, j_hi))

        for pi, (j_lo, j_hi) in enumerate(pairs):
            tau_lo, tau_hi = levels[j_lo], levels[j_hi]
            widths = (q[:, j_hi] - q[:, j_lo]) * scale

            bin_means = []
            for b in range(n_eta_bins):
                mask = (eta >= eta_edges[b]) & (eta < eta_edges[b + 1])
                if mask.sum() > 0:
                    bin_means.append(float(np.mean(widths[mask])))
                else:
                    bin_means.append(np.nan)
            bin_means = np.array(bin_means)

            color = pair_colors[pi % len(pair_colors)]
            ax.step(eta_centres, bin_means, where="mid", color=color, linewidth=1.5,
                    label=f"[{tau_lo:.0%}, {tau_hi:.0%}]")

        unit = PARAM_VALUE_LABELS.get(name, name).split("[")[-1].rstrip("]") if "[" in PARAM_VALUE_LABELS.get(name, name) else ""
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(f"Interval width [{unit}]" if unit else "Interval width", fontsize=11)
        ax.set_title(f"{name.upper()} Quantile Width vs $\\eta$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Predicted Quantile Interval Width vs $\\eta$", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "quantile_width_vs_eta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved quantile_width_vs_eta.png")


# ============================================================================
# 6. Calibration vs η (per-bin calibration check)
# ============================================================================

def plot_calibration_vs_eta(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_eta_bins: int = 20,
) -> None:
    """Plot per-η-bin observed coverage for each quantile level.

    Reveals whether calibration degrades in specific η regions (e.g. forward).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_truth = targets.get("theta")
    if theta_truth is None:
        return
    eta = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))
    eta_edges = np.linspace(float(np.percentile(eta, 0.5)), float(np.percentile(eta, 99.5)), n_eta_bins + 1)
    eta_centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name]
        q = quantiles[name]
        levels = quantile_levels[name]

        for j, tau in enumerate(levels):
            obs_per_bin = []
            for b in range(n_eta_bins):
                mask = (eta >= eta_edges[b]) & (eta < eta_edges[b + 1])
                if mask.sum() > 50:
                    obs_per_bin.append(float(np.mean(t[mask] <= q[mask, j])))
                else:
                    obs_per_bin.append(np.nan)
            obs_per_bin = np.array(obs_per_bin)

            color = _quantile_color(tau)
            ls = _quantile_ls(tau)
            ax.plot(eta_centres, obs_per_bin, color=color, linestyle=ls, linewidth=1.3,
                    label=f"$\\tau={tau:.2f}$")
            # Dashed horizontal line at nominal
            ax.axhline(tau, color=color, linestyle=":", linewidth=0.6, alpha=0.4)

        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel("Observed coverage", fontsize=11)
        ax.set_title(f"{name.upper()} Calibration vs $\\eta$", fontsize=12)
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Per-$\\eta$-bin Quantile Calibration", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "calibration_vs_eta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved calibration_vs_eta.png")


# ============================================================================
# 7. Residual distribution with quantile bands
# ============================================================================

def plot_residual_with_quantile_bands(
    targets: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 120,
) -> None:
    """Residual distribution (pred - truth) with vertical lines showing
    the mean quantile residual (quantile_pred - truth) for each level.

    This directly shows how the predicted uncertainty envelope covers the
    actual residual distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in quantiles and p in preds]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        scale = UNIT_SCALE.get(name, 1.0)
        t = targets[name]
        p = preds[name]
        q = quantiles[name]
        levels = quantile_levels[name]

        residual = (p - t) * scale

        # Bin range from residuals
        lo = float(np.percentile(residual, 0.5))
        hi = float(np.percentile(residual, 99.5))
        bins = np.linspace(lo, hi, n_bins + 1)

        counts, _ = np.histogram(residual, bins=bins)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        ax.step(bin_centres, counts, where="mid", color="black", linewidth=1.5, label="Residual")
        ax.set_yscale("log")
        ymin = max(1, counts[counts > 0].min() * 0.5) if np.any(counts > 0) else 1
        ymax = counts.max() * 5
        ax.set_ylim(ymin, ymax)

        # Vertical lines at mean (quantile - truth) per level
        for j, tau in enumerate(levels):
            q_resid_mean = float(np.mean((q[:, j] - t) * scale))
            color = _quantile_color(tau)
            ls = _quantile_ls(tau)
            ax.axvline(q_resid_mean, color=color, linestyle=ls, linewidth=1.4,
                       label=f"$\\tau={tau:.2f}$")

        from hepattn.experiments.colliderml_regr.eval_utils import RESID_LABELS
        ax.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(f"{name.upper()} Residual + Quantile Residuals", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Residual Distribution with Quantile Positions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "residual_with_quantile_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved residual_with_quantile_bands.png")


# ============================================================================
# Summary table
# ============================================================================

def print_calibration_summary(
    targets: dict[str, np.ndarray],
    quantiles: dict[str, np.ndarray],
    quantile_levels: dict[str, np.ndarray],
) -> None:
    """Print a summary table of observed vs nominal coverage per parameter."""
    params = [p for p in PARAMS if p in quantiles and p in targets]
    if not params:
        print("No quantile predictions found — skipping calibration summary.")
        return

    print("\n" + "=" * 80)
    print("QUANTILE CALIBRATION SUMMARY")
    print("=" * 80)

    for name in params:
        t = targets[name]
        q = quantiles[name]
        levels = quantile_levels[name]

        print(f"\n  {name.upper()}")
        print(f"  {'Level':>8s}  {'Observed':>10s}  {'Nominal':>10s}  {'Delta':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

        deltas = []
        for j, tau in enumerate(levels):
            obs = float(np.mean(t <= q[:, j]))
            delta = obs - tau
            deltas.append(abs(delta))
            print(f"  {tau:8.2f}  {obs:10.4f}  {tau:10.4f}  {delta:+10.4f}")

        print(f"  Mean |Δ|: {np.mean(deltas):.4f}")

    print("\n" + "=" * 80)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantile alignment diagnostics for track regression predictions."
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to test_predictions.h5 (must contain a 'quantiles' group).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output plots.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Optional data directory (unused for now, kept for CLI consistency).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    data = load_predictions(args.predictions)

    targets = data["targets"]
    preds = data["preds"]
    quantiles_data = data.get("quantiles", {})
    quantile_levels = data.get("quantile_levels", {})

    if not quantiles_data:
        print("ERROR: No quantile predictions found in the HDF5 file.")
        print("Re-run inference with a quantile-loss model to generate quantile predictions.")
        return

    print(f"Found quantile predictions for: {list(quantiles_data.keys())}")
    for name in quantiles_data:
        q = quantiles_data[name]
        levels = quantile_levels[name]
        print(f"  {name}: shape={q.shape}, levels={levels}")

    # Print calibration summary table
    print_calibration_summary(targets, quantiles_data, quantile_levels)

    # Generate all plots
    print("\nGenerating plots...")

    plot_truth_distribution_with_quantiles(targets, quantiles_data, quantile_levels, output_dir)
    plot_quantile_spread_samples(targets, quantiles_data, quantile_levels, output_dir)
    plot_quantile_calibration(targets, quantiles_data, quantile_levels, output_dir)
    plot_pit_histogram(targets, quantiles_data, quantile_levels, output_dir)
    plot_quantile_width_vs_eta(targets, quantiles_data, quantile_levels, output_dir)
    plot_calibration_vs_eta(targets, quantiles_data, quantile_levels, output_dir)
    plot_residual_with_quantile_bands(targets, preds, quantiles_data, quantile_levels, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
