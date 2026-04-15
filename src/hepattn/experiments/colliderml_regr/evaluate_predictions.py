#!/usr/bin/env python3
"""Evaluate track regression predictions: precision and residual distribution plots.

Usage::

    python -m hepattn.experiments.colliderml_regr.evaluate_predictions \
        --predictions /path/to/test_predictions.h5 \
        --data-dir /scratch/colliderml/p0/p0_preprocessed \
        --output-dir /path/to/eval_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    FULL_RANGE_PARAMS,
    HEATMAP_RANGE,
    PARAM_LABELS,
    PARAM_VALUE_LABELS,
    PARAMS,
    RESID_LABELS,
    UNIT_SCALE,
    add_common_args,
    build_regime_data,
    compute_core_metrics_vs_eta,
    compute_precision_vs_eta,
    compute_precision_vs_eta_iterative,
    load_all_data,
    print_precision_summary,
    write_residual_statistics_report,
)


# ============================================================================
# Step-plot helpers
# ============================================================================

def _build_step_arrays(bin_edges: np.ndarray, values: np.ndarray):
    """Convert bin centers/values to step-plot compatible arrays."""
    x = np.repeat(bin_edges, 2)
    y = np.repeat(values, 2)
    y = np.concatenate([[y[0]], y, [y[-1]]])
    return x, y[: len(x)]


def _step_fill(ax, edges, values, err, color, alpha=0.25):
    """Draw a step fill-between band."""
    x_fill = np.repeat(edges, 2)[1:-1]
    lo = np.repeat(np.maximum(values - err, 0), 2)
    hi = np.repeat(values + err, 2)
    ax.fill_between(x_fill, lo, hi, alpha=alpha, color=color)


def _edges_for_centers(eta_bins, centers):
    """Map bin centers back to bin edges from the full eta_bins array."""
    bin_indices = np.searchsorted(eta_bins, centers, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(eta_bins) - 2)
    return np.append(eta_bins[bin_indices], eta_bins[bin_indices[-1] + 1])


# ============================================================================
# Precision vs η
# ============================================================================

def plot_precision_vs_eta(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_data: dict | None = None,
    acts_label: str = "ACTS CKF",
    eta_values: np.ndarray | None = None,
    regime_subtitle: str | None = None,
) -> None:
    """Plot precision (σ of residual) vs η, optionally with ACTS overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in precision_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        centers = data["eta_centers"]
        std = data["std"]
        std_err = data["std_err"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        edges = _edges_for_centers(eta_bins, centers)
        x, y = _build_step_arrays(edges, std * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                label=f"{ml_label} (σ={unbinned:.4f})")
        _step_fill(ax, edges, std * scale, std_err * scale, "steelblue")

        if acts_data is not None and name in acts_data:
            a = acts_data[name]
            a_scale = UNIT_SCALE.get(name, 1.0)
            a_unbinned = a["unbinned_std"] * a_scale
            a_edges = _edges_for_centers(eta_bins, a["eta_centers"])
            ax_x, ax_y = _build_step_arrays(a_edges, a["std"] * a_scale)
            ax.step(ax_x, ax_y, where="post", color="darkorange", linewidth=2.5,
                    linestyle="--", label=f"{acts_label} (σ={a_unbinned:.4f})")
            _step_fill(ax, a_edges, a["std"] * a_scale, a["std_err"] * a_scale,
                       "darkorange")

        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$", fontsize=12)
        ax.set_ylabel(PARAM_LABELS.get(name, f"σ({name})"), fontsize=12)
        ax.set_title(f"{name.upper()} Precision vs $\\eta$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / f"precision_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_summary_precision(precision_data, eta_bins, output_dir,
                            ml_label=ml_label, acts_data=acts_data, acts_label=acts_label,
                            eta_values=eta_values, regime_subtitle=regime_subtitle)


def _plot_summary_precision(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_data: dict | None = None,
    acts_label: str = "ACTS CKF",
    eta_values: np.ndarray | None = None,
    regime_subtitle: str | None = None,
    filename: str = "precision_summary.png",
    title_suffix: str = "",
    sigma_symbol: str = "σ",
    track_count: int | None = None,
    autoscale_y: bool = False,
    ml_cuts: dict[str, dict] | None = None,
    acts_cuts: dict[str, dict] | None = None,
) -> None:
    """Summary 2×3 panel with all precision vs η curves."""
    params = [p for p in PARAMS if p in precision_data]
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
        data = precision_data[name]
        centers = data["eta_centers"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        # Build legend labels using sigma_symbol (e.g. "σ", "RMS", "σ_G")
        if ml_cuts is not None and name in ml_cuts:
            ml_lbl = f"{ml_label} ({sigma_symbol}={unbinned:.4f}, n={ml_cuts[name]['n_kept']:,})"
        else:
            ml_lbl = f"{ml_label} ({sigma_symbol}={unbinned:.4f})"

        edges = _edges_for_centers(eta_bins, centers)
        x, y = _build_step_arrays(edges, data["std"] * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2, label=ml_lbl)

        if acts_data is not None and name in acts_data:
            a = acts_data[name]
            a_scale = UNIT_SCALE.get(name, 1.0)
            a_unbinned = a["unbinned_std"] * a_scale
            if acts_cuts is not None and name in acts_cuts:
                a_lbl = f"{acts_label} ({sigma_symbol}={a_unbinned:.4f}, n={acts_cuts[name]['n_kept']:,})"
            else:
                a_lbl = f"{acts_label} ({sigma_symbol}={a_unbinned:.4f})"
            a_edges = _edges_for_centers(eta_bins, a["eta_centers"])
            ax_x, ax_y = _build_step_arrays(a_edges, a["std"] * a_scale)
            ax.step(ax_x, ax_y, where="post", color="darkorange", linewidth=2,
                    linestyle="--", label=a_lbl)

        ax.legend(fontsize=7)
        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        # Build y-axis label using sigma_symbol instead of always "σ"
        param_sym = {"d0": "d_0", "z0": "z_0", "phi": r"\phi",
                     "theta": r"\theta", "qop": "q/p"}.get(name, name)
        unit = {"d0": "mm", "z0": "mm", "phi": "mrad",
                "theta": "mrad", "qop": "1/GeV"}.get(name, "")
        ax.set_ylabel(f"${sigma_symbol}({param_sym})$ [{unit}]")
        ax.set_title(f"{name} ({ml_label} {sigma_symbol}={unbinned:.4f})")
        ax.grid(True, alpha=0.3)
        if autoscale_y:
            all_vals = [data["std"] * scale]
            if acts_data is not None and name in acts_data:
                all_vals.append(acts_data[name]["std"] * UNIT_SCALE.get(name, 1.0))
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            if len(valid) > 0:
                vmin, vmax = float(np.min(valid)), float(np.max(valid))
                margin = (vmax - vmin) * 0.15 if vmax > vmin else vmax * 0.1
                ax.set_ylim(vmin - margin, vmax + margin)
        else:
            ax.set_ylim(bottom=0)

    # Fill remaining panel(s) with track count distribution over η
    next_panel = len(params)
    if eta_values is not None and next_panel < len(axes):
        ax = axes[next_panel]
        counts, _ = np.histogram(eta_values, bins=eta_bins)
        x, y = _build_step_arrays(eta_bins, counts.astype(float))
        count_label = track_count if track_count is not None else len(eta_values)
        ax.step(x, y, where="post", color="green", linewidth=2, linestyle="--",
                label=f"Tracks ({count_label:,} total)")
        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        ax.set_ylabel("Tracks per bin")
        ax.set_title("Track count vs $\\eta$")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        next_panel += 1

    for j in range(next_panel, len(axes)):
        axes[j].set_visible(False)

    title = f"Track Parameter Precision vs η{title_suffix}"
    if acts_data:
        title += f" — {ml_label} vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_core_metrics_summary(
    ml_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_data: dict | None = None,
    acts_label: str = "ACTS CKF",
    eta_values: np.ndarray | None = None,
    regime_subtitle: str | None = None,
    filename: str = "precision_summary_core95.png",
    core_fraction: float = 0.95,
) -> None:
    """Plot RMS, std, IQR vs η on inner `core_fraction` of residuals, for ML vs ACTS."""
    params = [p for p in PARAMS if p in ml_data]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    metrics = [("rms", "-", "RMS"), ("std", "--", "std"), ("iqr", ":", "IQR")]
    pct = int(round(core_fraction * 100))

    for i, name in enumerate(params):
        ax = axes[i]
        data = ml_data[name]
        scale = UNIT_SCALE.get(name, 1.0)
        edges = _edges_for_centers(eta_bins, data["eta_centers"])

        for key, ls, mlabel in metrics:
            x, y = _build_step_arrays(edges, data[key] * scale)
            ub = data[f"unbinned_{key}"] * scale
            ax.step(x, y, where="post", color="steelblue", linewidth=2,
                    linestyle=ls,
                    label=f"{ml_label} {mlabel} ({ub:.4f})")

        if acts_data is not None and name in acts_data:
            a = acts_data[name]
            a_scale = UNIT_SCALE.get(name, 1.0)
            a_edges = _edges_for_centers(eta_bins, a["eta_centers"])
            for key, ls, mlabel in metrics:
                x, y = _build_step_arrays(a_edges, a[key] * a_scale)
                ub = a[f"unbinned_{key}"] * a_scale
                ax.step(x, y, where="post", color="darkorange", linewidth=2,
                        linestyle=ls,
                        label=f"{acts_label} {mlabel} ({ub:.4f})")

        param_sym = {"d0": "d_0", "z0": "z_0", "phi": r"\phi",
                     "theta": r"\theta", "qop": "q/p"}.get(name, name)
        unit = {"d0": "mm", "z0": "mm", "phi": "mrad",
                "theta": "mrad", "qop": "1/GeV"}.get(name, "")
        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        ax.set_ylabel(f"metric$({param_sym})$ [{unit}]")
        ax.set_title(f"{name} (inner {pct}%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

    next_panel = len(params)
    if eta_values is not None and next_panel < len(axes):
        ax = axes[next_panel]
        counts, _ = np.histogram(eta_values, bins=eta_bins)
        x, y = _build_step_arrays(eta_bins, counts.astype(float) * core_fraction)
        ax.step(x, y, where="post", color="green", linewidth=2, linestyle="--",
                label=f"Core tracks (~{int(len(eta_values) * core_fraction):,})")
        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        ax.set_ylabel("Tracks per bin (core)")
        ax.set_title(f"Core track count vs $\\eta$ (inner {pct}%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        next_panel += 1

    for j in range(next_panel, len(axes)):
        axes[j].set_visible(False)

    title = f"Core-{pct}% Residual Metrics vs η — {ml_label}"
    if acts_data:
        title += f" vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Distribution summary (target vs prediction histograms)
# ============================================================================

def plot_distribution_summary(
    targets: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 80,
    pred_label: str = "SSM",
    target_label: str = "Truth",
    filename: str = "distribution_ssm.png",
    color: str = "steelblue",
    regime_subtitle: str | None = None,
) -> None:
    """Summary 2×3 panel comparing target and prediction distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in targets and p in preds]
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
        t = targets[name]
        p = preds[name]

        if name in FULL_RANGE_PARAMS:
            lo = min(float(np.min(t)), float(np.min(p)))
            hi = max(float(np.max(t)), float(np.max(p)))
        else:
            lo = min(float(np.percentile(t, 0.5)), float(np.percentile(p, 0.5)))
            hi = max(float(np.percentile(t, 99.5)), float(np.percentile(p, 99.5)))
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(t, bins=bins, histtype="step", linewidth=1.8,
                color="black", label=target_label)
        ax.hist(p, bins=bins, histtype="step", linewidth=1.8,
                color=color, linestyle="--", label=pred_label)

        ax.set_xlabel(PARAM_VALUE_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if name == "d0":
            ax.set_yscale("log")
            ymax = ax.get_ylim()[1]
            ax.set_ylim(bottom=1, top=ymax * 10)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = f"Track Parameter Distributions — {pred_label} vs {target_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Residual distributions (linear and log scale)
# ============================================================================

def plot_residual_distribution_summary(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 80,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    filename: str = "residual_summary.png",
    regime_subtitle: str | None = None,
) -> None:
    """Summary 2×3 panel showing residual distributions for ML (and optionally ACTS)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
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
        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name] * scale

        all_arrays = [ml_res]
        if acts_residuals is not None and name in acts_residuals:
            acts_res = acts_residuals[name] * scale
            all_arrays.append(acts_res)

        combined = np.concatenate(all_arrays)
        if name in FULL_RANGE_PARAMS:
            lo = float(np.min(combined))
            hi = float(np.max(combined))
        else:
            lo = float(np.percentile(combined, 0.5))
            hi = float(np.percentile(combined, 99.5))
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(ml_res, bins=bins, histtype="step", linewidth=1.8,
                color="steelblue", label=ml_label)

        if acts_residuals is not None and name in acts_residuals:
            ax.hist(acts_res, bins=bins, histtype="step", linewidth=1.8,
                    color="darkorange", linestyle="--", label=acts_label)

        ax.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

        if name == "d0":
            ax.set_yscale("log")
            ymax = ax.get_ylim()[1]
            ax.set_ylim(bottom=1, top=ymax * 10)
        else:
            # Add top padding so legend doesn't overlap peak
            ymax = ax.get_ylim()[1]
            ax.set_ylim(top=ymax * 1.4)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = f"Residual Distributions — {ml_label}"
    if acts_residuals is not None:
        title += f" vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_log_distributions(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    n_bins: int = 120,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    filename: str = "residual_summary_log.png",
    regime_subtitle: str | None = None,
    ml_rms_cuts: dict[str, dict] | None = None,
    ml_gauss_cuts: dict[str, dict] | None = None,
    acts_rms_cuts: dict[str, dict] | None = None,
    acts_gauss_cuts: dict[str, dict] | None = None,
) -> None:
    """Log-scale residual distributions with ratio panels and iterative cut boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return

    has_acts_global = acts_residuals is not None

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols

    if has_acts_global:
        fig, all_axes = plt.subplots(
            n_rows * 2, n_cols,
            figsize=(5 * n_cols, 6.5 * n_rows),
            gridspec_kw={"height_ratios": [3, 1] * n_rows},
        )
        all_axes = np.atleast_2d(all_axes)
    else:
        fig, axes_flat = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if not isinstance(axes_flat, np.ndarray):
            axes_flat = np.array([axes_flat])
        axes_flat = axes_flat.flatten()

    for i, name in enumerate(params):
        row = i // n_cols
        col = i % n_cols

        if has_acts_global:
            ax = all_axes[row * 2, col]
            ax_ratio = all_axes[row * 2 + 1, col]
        else:
            ax = axes_flat[i]
            ax_ratio = None

        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name] * scale

        all_arrays = [ml_res]
        has_acts = has_acts_global and name in acts_residuals
        if has_acts:
            acts_res = acts_residuals[name] * scale
            all_arrays.append(acts_res)

        combined = np.concatenate(all_arrays)
        # 3σ range (99.7%)
        lo = float(np.percentile(combined, 0.15))
        hi = float(np.percentile(combined, 99.85))
        bins = np.linspace(lo, hi, n_bins + 1)

        ml_counts, _ = np.histogram(ml_res, bins=bins)
        ax.hist(ml_res, bins=bins, histtype="step", linewidth=1.8,
                color="steelblue", label=ml_label)

        if has_acts:
            acts_counts, _ = np.histogram(acts_res, bins=bins)
            ax.hist(acts_res, bins=bins, histtype="step", linewidth=1.8,
                    color="darkorange", linestyle="--", label=acts_label)

        ax.set_yscale("log")
        ymax = ax.get_ylim()[1]
        ax.set_ylim(bottom=1, top=ymax * 10)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)

        # --- Iterative cut boundary lines ---
        cut_legend_entries = []
        if ml_rms_cuts is not None and name in ml_rms_cuts:
            c = ml_rms_cuts[name]
            ax.axvline(c["cut_lo"] * scale, color="steelblue", linewidth=0.9,
                       linestyle="--", alpha=0.7)
            ax.axvline(c["cut_hi"] * scale, color="steelblue", linewidth=0.9,
                       linestyle="--", alpha=0.7)
            cut_legend_entries.append(f"{ml_label} iter. RMS")
        if ml_gauss_cuts is not None and name in ml_gauss_cuts:
            c = ml_gauss_cuts[name]
            ax.axvline(c["cut_lo"] * scale, color="steelblue", linewidth=0.9,
                       linestyle=":", alpha=0.7)
            ax.axvline(c["cut_hi"] * scale, color="steelblue", linewidth=0.9,
                       linestyle=":", alpha=0.7)
            cut_legend_entries.append(f"{ml_label} Gauss fit")
        if acts_rms_cuts is not None and name in acts_rms_cuts:
            c = acts_rms_cuts[name]
            ax.axvline(c["cut_lo"] * scale, color="darkorange", linewidth=0.9,
                       linestyle="--", alpha=0.7)
            ax.axvline(c["cut_hi"] * scale, color="darkorange", linewidth=0.9,
                       linestyle="--", alpha=0.7)
            cut_legend_entries.append(f"{acts_label} iter. RMS")
        if acts_gauss_cuts is not None and name in acts_gauss_cuts:
            c = acts_gauss_cuts[name]
            ax.axvline(c["cut_lo"] * scale, color="darkorange", linewidth=0.9,
                       linestyle=":", alpha=0.7)
            ax.axvline(c["cut_hi"] * scale, color="darkorange", linewidth=0.9,
                       linestyle=":", alpha=0.7)
            cut_legend_entries.append(f"{acts_label} Gauss fit")

        # Build legend with cut line style indicators
        handles, labels = ax.get_legend_handles_labels()
        if cut_legend_entries:
            if ml_rms_cuts is not None and name in ml_rms_cuts:
                handles.append(Line2D([0], [0], color="gray", linewidth=0.9,
                                      linestyle="--", alpha=0.7))
                labels.append("Iter. RMS cuts")
            if ml_gauss_cuts is not None and name in ml_gauss_cuts:
                handles.append(Line2D([0], [0], color="gray", linewidth=0.9,
                                      linestyle=":", alpha=0.7))
                labels.append("Gauss fit cuts")
        ax.legend(handles, labels, fontsize=8, loc="upper left")

        # Ratio panel
        if has_acts and ax_ratio is not None:
            valid = (acts_counts > 0) & (ml_counts > 0)
            ratio = np.full_like(ml_counts, np.nan, dtype=float)
            ratio[valid] = ml_counts[valid] / acts_counts[valid]

            ax_ratio.step(bins[:-1], ratio, where="post",
                          color="black", linewidth=1.2)
            ax_ratio.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
            ax_ratio.set_ylabel(f"{ml_label}/{acts_label}", fontsize=9)
            ax_ratio.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)
            ax_ratio.set_ylim(0.0, 3.0)
            ax_ratio.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        elif ax_ratio is not None:
            ax_ratio.set_visible(False)
            ax.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)
        else:
            ax.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)

    # Hide unused panels
    if has_acts_global:
        for i in range(len(params), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            all_axes[row * 2, col].set_visible(False)
            all_axes[row * 2 + 1, col].set_visible(False)
    else:
        for j in range(len(params), len(axes_flat)):
            axes_flat[j].set_visible(False)

    title = f"Residual Distributions (log scale, 3σ range) — {ml_label}"
    if has_acts_global:
        title += f" vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Clipped residual distributions
# ============================================================================

def plot_clipped_residual_summary(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    ml_cuts: dict[str, dict],
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_cuts: dict[str, dict] | None = None,
    acts_label: str = "ACTS CKF",
    filename: str = "clipped_residual_summary.png",
    title_suffix: str = "",
    regime_subtitle: str | None = None,
    n_bins: int = 120,
    log_scale: bool = False,
) -> None:
    """Show residual distributions after iterative clipping."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals and p in ml_cuts]
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
        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name]
        c = ml_cuts[name]

        ml_mask = (ml_res >= c["cut_lo"]) & (ml_res <= c["cut_hi"])
        ml_clipped = ml_res[ml_mask] * scale
        lo, hi = c["cut_lo"] * scale, c["cut_hi"] * scale

        has_acts = (acts_residuals is not None and acts_cuts is not None
                    and name in acts_residuals and name in acts_cuts)
        if has_acts:
            ac = acts_cuts[name]
            acts_res = acts_residuals[name]
            acts_mask = (acts_res >= ac["cut_lo"]) & (acts_res <= ac["cut_hi"])
            acts_clipped = acts_res[acts_mask] * scale
            lo = min(lo, ac["cut_lo"] * scale)
            hi = max(hi, ac["cut_hi"] * scale)

        bins = np.linspace(lo, hi, n_bins + 1)
        ax.hist(ml_clipped, bins=bins, histtype="step", linewidth=1.8,
                color="steelblue", label=f"{ml_label} (n={len(ml_clipped):,})")
        if has_acts:
            ax.hist(acts_clipped, bins=bins, histtype="step", linewidth=1.8,
                    color="darkorange", linestyle="--",
                    label=f"{acts_label} (n={len(acts_clipped):,})")

        ax.set_xlabel(RESID_LABELS.get(name, name), fontsize=11)
        ax.set_ylabel("Tracks / bin", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")
            ymax = ax.get_ylim()[1]
            ax.set_ylim(bottom=1, top=ymax * 10)
        else:
            ymax = ax.get_ylim()[1]
            ax.set_ylim(top=ymax * 1.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    scale_tag = " (log scale)" if log_scale else ""
    title = f"Residual Distributions After Clipping{title_suffix}{scale_tag}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Mean residual vs η
# ============================================================================

def plot_mean_residual_vs_eta(
    ml_residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float],
    n_eta_bins: int,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    ml_rms_cuts: dict[str, dict] | None = None,
    ml_gauss_cuts: dict[str, dict] | None = None,
    acts_rms_cuts: dict[str, dict] | None = None,
    acts_gauss_cuts: dict[str, dict] | None = None,
    regime_subtitle: str | None = None,
) -> None:
    """Plot mean residual vs η as 6 separate 2×3 summaries (one per source × cut)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    bin_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    ml_eta = ml_residuals["eta"]
    acts_eta = acts_residuals["eta"] if acts_residuals is not None else None

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return

    def _binned_mean(eta, res, scale, cut=None):
        means, sems = [], []
        for j in range(len(eta_bins) - 1):
            mask = (eta >= eta_bins[j]) & (eta < eta_bins[j + 1])
            r = res[mask]
            if cut is not None:
                clip = (r >= cut["cut_lo"]) & (r <= cut["cut_hi"])
                r = r[clip]
            if len(r) > 1:
                means.append(np.mean(r) * scale)
                sems.append(np.std(r) / np.sqrt(len(r)) * scale)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        return np.array(means), np.array(sems)

    def _draw_summary(series_list, suptitle, filename):
        """Draw a 2×3 summary with one or more series per panel."""
        n_cols = min(3, len(params))
        n_rows = (len(params) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, name in enumerate(params):
            ax = axes[idx]
            scale = UNIT_SCALE.get(name, 1.0)
            for eta, res, cut, label, color, marker in series_list:
                if res is None or name not in res:
                    continue
                m, sem = _binned_mean(eta, res[name], scale,
                                      cut.get(name) if cut else None)
                ax.errorbar(bin_centers, m, yerr=sem, fmt=f"{marker}-",
                            color=color, ms=4, linewidth=1.5, capsize=3,
                            label=label)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(r"$\eta_{\mathrm{truth}}$", fontsize=11)
            ax.set_ylabel(f"Mean {RESID_LABELS.get(name, name)}", fontsize=11)
            ax.set_title(name.upper(), fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        for j in range(len(params), len(axes)):
            axes[j].set_visible(False)

        full_title = suptitle
        if regime_subtitle:
            full_title += f"\n{regime_subtitle}"
        plt.suptitle(full_title, fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- SSM summaries ---
    _draw_summary(
        [(ml_eta, ml_residuals, None, f"{ml_label} raw", "steelblue", "o")],
        rf"{ml_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (raw)",
        "mean_residual_vs_eta_ssm_raw.png",
    )
    if ml_rms_cuts is not None:
        _draw_summary(
            [(ml_eta, ml_residuals, ml_rms_cuts,
              f"{ml_label} RMS clip", "steelblue", "s")],
            rf"{ml_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (RMS clip)",
            "mean_residual_vs_eta_ssm_rms.png",
        )
    if ml_gauss_cuts is not None:
        _draw_summary(
            [(ml_eta, ml_residuals, ml_gauss_cuts,
              f"{ml_label} Gauss clip", "steelblue", "^")],
            rf"{ml_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (Gauss clip)",
            "mean_residual_vs_eta_ssm_gauss.png",
        )

    # --- ACTS summaries ---
    if acts_residuals is not None:
        _draw_summary(
            [(acts_eta, acts_residuals, None,
              f"{acts_label} raw", "darkorange", "o")],
            rf"{acts_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (raw)",
            "mean_residual_vs_eta_acts_raw.png",
        )
        if acts_rms_cuts is not None:
            _draw_summary(
                [(acts_eta, acts_residuals, acts_rms_cuts,
                  f"{acts_label} RMS clip", "darkorange", "s")],
                rf"{acts_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (RMS clip)",
                "mean_residual_vs_eta_acts_rms.png",
            )
        if acts_gauss_cuts is not None:
            _draw_summary(
                [(acts_eta, acts_residuals, acts_gauss_cuts,
                  f"{acts_label} Gauss clip", "darkorange", "^")],
                rf"{acts_label} — Mean Residual $\langle\Delta\rangle$ vs $\eta$ (Gauss clip)",
                "mean_residual_vs_eta_acts_gauss.png",
            )

        # --- Combined SSM + ACTS raw comparison ---
        _draw_summary(
            [(ml_eta, ml_residuals, None, f"{ml_label} raw", "steelblue", "o"),
             (acts_eta, acts_residuals, None, f"{acts_label} raw", "darkorange", "o")],
            rf"Mean Residual $\langle\Delta\rangle$ vs $\eta$ — {ml_label} vs {acts_label} (raw)",
            "mean_residual_vs_eta_comparison_raw.png",
        )


# ============================================================================
# Precision and mean residual vs pT
# ============================================================================

def _compute_pt(residuals: dict[str, np.ndarray], targets: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Compute truth pT from theta and qop targets embedded in residuals or targets dict."""
    if targets is not None and "theta" in targets and "qop" in targets:
        theta = targets["theta"]
        qop = targets["qop"]
    else:
        # Fallback: cannot compute pT without targets
        return None
    return np.abs(np.sin(theta) / (qop + 1e-12))


def plot_precision_vs_pt(
    ml_residuals: dict[str, np.ndarray],
    ml_pt: np.ndarray,
    pt_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_pt: np.ndarray | None = None,
    acts_label: str = "ACTS CKF",
    regime_subtitle: str | None = None,
    ml_rms_cuts: dict[str, dict] | None = None,
    acts_rms_cuts: dict[str, dict] | None = None,
) -> None:
    """Precision (std and iterative RMS) vs truth pT as 2×3 summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return
    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    min_tracks = 30

    def _binned_precision(pt, res, cut=None, use_rms=False):
        """Compute per-bin std or RMS (optionally after applying unbinned cut)."""
        vals, errs, counts = [], [], []
        for j in range(len(pt_bins) - 1):
            mask = (pt >= pt_bins[j]) & (pt < pt_bins[j + 1])
            r = res[mask]
            if cut is not None:
                clip = (r >= cut["cut_lo"]) & (r <= cut["cut_hi"])
                r = r[clip]
            n = len(r)
            if n >= min_tracks:
                if use_rms:
                    s = float(np.sqrt(np.mean(r**2)))
                else:
                    s = float(np.std(r))
                vals.append(s)
                errs.append(s / np.sqrt(2 * n))
                counts.append(n)
            else:
                vals.append(np.nan)
                errs.append(np.nan)
                counts.append(0)
        return np.array(vals), np.array(errs)

    def _draw_pt_summary(series_list, suptitle, filename, sigma_sym="σ",
                         use_rms=False):
        n_cols = min(3, len(params))
        n_rows = (len(params) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, name in enumerate(params):
            ax = axes[idx]
            scale = UNIT_SCALE.get(name, 1.0)
            param_sym = {"d0": "d_0", "z0": "z_0", "phi": r"\phi",
                         "theta": r"\theta", "qop": "q/p"}.get(name, name)
            unit = {"d0": "mm", "z0": "mm", "phi": "mrad",
                    "theta": "mrad", "qop": "1/GeV"}.get(name, "")

            for pt, res, cut, label, color, marker in series_list:
                if res is None or name not in res:
                    continue
                s, e = _binned_precision(pt, res[name],
                                         cut.get(name) if cut else None,
                                         use_rms=use_rms)
                edges = pt_bins
                x, y = _build_step_arrays(edges, s * scale)
                ax.step(x, y, where="post", color=color, linewidth=2,
                        linestyle="-" if "o" in marker else "--", label=label)

            ax.set_xlabel(r"$p_T^{\mathrm{truth}}$ [GeV]", fontsize=11)
            ax.set_ylabel(f"${sigma_sym}({param_sym})$ [{unit}]", fontsize=11)
            ax.set_title(name.upper(), fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        for j in range(len(params), len(axes)):
            axes[j].set_visible(False)

        full_title = suptitle
        if regime_subtitle:
            full_title += f"\n{regime_subtitle}"
        plt.suptitle(full_title, fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Standard std (no clipping)
    series_std = [(ml_pt, ml_residuals, None, ml_label, "steelblue", "o")]
    if acts_residuals is not None and acts_pt is not None:
        series_std.append((acts_pt, acts_residuals, None, acts_label, "darkorange", "o"))
    _draw_pt_summary(series_std,
                     r"Precision ($\sigma$) vs $p_T^{\mathrm{truth}}$",
                     "precision_vs_pt_std.png")

    # Plain RMS (no clipping)
    series_rms_plain = [(ml_pt, ml_residuals, None, ml_label, "steelblue", "o")]
    if acts_residuals is not None and acts_pt is not None:
        series_rms_plain.append((acts_pt, acts_residuals, None, acts_label, "darkorange", "o"))
    _draw_pt_summary(series_rms_plain,
                     r"Precision (RMS) vs $p_T^{\mathrm{truth}}$",
                     "precision_vs_pt_rms.png", sigma_sym="RMS",
                     use_rms=True)

    # Std after iterative 3σ clipping
    if ml_rms_cuts is not None:
        series_clipped = [(ml_pt, ml_residuals, ml_rms_cuts, ml_label, "steelblue", "o")]
        if acts_residuals is not None and acts_pt is not None and acts_rms_cuts is not None:
            series_clipped.append((acts_pt, acts_residuals, acts_rms_cuts,
                               acts_label, "darkorange", "o"))
        _draw_pt_summary(series_clipped,
                         r"Precision ($\sigma$, after iterative 3$\sigma$ clip) vs $p_T^{\mathrm{truth}}$",
                         "precision_vs_pt_clipped.png")


def plot_mean_residual_vs_pt(
    ml_residuals: dict[str, np.ndarray],
    ml_pt: np.ndarray,
    pt_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_pt: np.ndarray | None = None,
    acts_label: str = "ACTS CKF",
    regime_subtitle: str | None = None,
) -> None:
    """Mean residual vs truth pT — combined SSM + ACTS raw comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return
    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    def _binned_mean(pt, res, scale):
        means, sems = [], []
        for j in range(len(pt_bins) - 1):
            mask = (pt >= pt_bins[j]) & (pt < pt_bins[j + 1])
            r = res[mask]
            if len(r) > 1:
                means.append(np.mean(r) * scale)
                sems.append(np.std(r) / np.sqrt(len(r)) * scale)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        return np.array(means), np.array(sems)

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(params):
        ax = axes[idx]
        scale = UNIT_SCALE.get(name, 1.0)

        m, sem = _binned_mean(ml_pt, ml_residuals[name], scale)
        ax.errorbar(bin_centers, m, yerr=sem, fmt="o-",
                    color="steelblue", ms=4, linewidth=1.5, capsize=3,
                    label=f"{ml_label}")
        if acts_residuals is not None and acts_pt is not None and name in acts_residuals:
            am, asem = _binned_mean(acts_pt, acts_residuals[name], scale)
            ax.errorbar(bin_centers, am, yerr=asem, fmt="o-",
                        color="darkorange", ms=4, linewidth=1.5, capsize=3,
                        label=f"{acts_label}")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$p_T^{\mathrm{truth}}$ [GeV]", fontsize=11)
        ax.set_ylabel(f"Mean {RESID_LABELS.get(name, name)}", fontsize=11)
        ax.set_title(name.upper(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = rf"Mean Residual $\langle\Delta\rangle$ vs $p_T^{{\mathrm{{truth}}}}$ — {ml_label} vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "mean_residual_vs_pt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# SSM vs ACTS correlation heatmaps
# ============================================================================

def plot_ssm_vs_acts_heatmap(
    ml_preds: dict[str, np.ndarray],
    acts_reco: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_label: str = "ACTS CKF",
    filename: str = "ssm_vs_acts_heatmap.png",
    regime_subtitle: str | None = None,
    n_bins: int = 200,
) -> None:
    """2D correlation heatmap: SSM predictions vs ACTS reconstructed values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_preds and p in acts_reco]
    if not params:
        return

    # Verify alignment
    n_ml = len(ml_preds[params[0]])
    n_acts = len(acts_reco[params[0]])
    if n_ml != n_acts:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        x = acts_reco[name]
        y = ml_preds[name]

        if name in HEATMAP_RANGE:
            lo, hi = HEATMAP_RANGE[name]
        else:
            combined = np.concatenate([x, y])
            lo = float(np.percentile(combined, 0.5))
            hi = float(np.percentile(combined, 99.5))

        bins = np.linspace(lo, hi, n_bins + 1)
        h, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins])
        h = np.ma.masked_where(h == 0, h)

        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap="viridis",
                            norm=LogNorm(vmin=1))
        fig.colorbar(pcm, ax=ax, pad=0.02, aspect=30)

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, alpha=0.7, label="y = x")
        ax.set_xlabel(f"{acts_label} {PARAM_VALUE_LABELS.get(name, name)}", fontsize=10)
        ax.set_ylabel(f"{ml_label} {PARAM_VALUE_LABELS.get(name, name)}", fontsize=10)
        ax.set_title(f"{name.upper()} ({n_ml:,} tracks)", fontsize=12)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = f"{ml_label} Predictions vs {acts_label} Reco"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Evaluation suite
# ============================================================================

def run_evaluation_suite(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    eta_range: tuple[float, float],
    n_eta_bins: int,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    acts_precision_npz: dict | None = None,
    acts_precision_label: str = "ACTS (Selected, DM)",
    ml_preds: dict[str, np.ndarray] | None = None,
    ml_targets: dict[str, np.ndarray] | None = None,
    acts_reco_values: dict[str, np.ndarray] | None = None,
    acts_reco_targets: dict[str, np.ndarray] | None = None,
    regime_subtitle: str | None = None,
    **kwargs,
) -> dict[str, dict]:
    """Run precision analysis and save all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute ACTS precision data if residuals provided
    acts_prec: dict | None = None
    if acts_residuals is not None:
        print("  Computing ACTS precision statistics ...")
        _, acts_prec = compute_precision_vs_eta(acts_residuals, eta_range, n_eta_bins)

    if acts_prec is None and acts_precision_npz is not None:
        acts_prec = acts_precision_npz
        acts_label = acts_precision_label

    # --- Precision vs η (standard std) ---
    print("  Computing ML precision vs η (std) ...")
    eta_bins, ml_precision = compute_precision_vs_eta(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins,
    )
    print("  Plotting precision vs η (std) ...")
    eta_values = ml_residuals.get("eta")
    plot_precision_vs_eta(ml_precision, eta_bins, output_dir,
                          ml_label=ml_label, acts_data=acts_prec, acts_label=acts_label,
                          eta_values=eta_values, regime_subtitle=regime_subtitle)

    # --- Precision vs η (RMS, no iteration) ---
    print("  Computing ML precision vs η (RMS) ...")
    eta_bins_rms, ml_prec_rms = compute_precision_vs_eta(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, use_rms=True,
    )
    acts_prec_rms: dict | None = None
    if acts_residuals is not None:
        _, acts_prec_rms = compute_precision_vs_eta(
            acts_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, use_rms=True,
        )
    print("  Plotting precision vs η (RMS) ...")
    _plot_summary_precision(
        ml_prec_rms, eta_bins_rms, output_dir,
        ml_label=ml_label, acts_data=acts_prec_rms, acts_label=acts_label,
        eta_values=eta_values, regime_subtitle=regime_subtitle,
        filename="precision_summary_rms.png",
        title_suffix=" (RMS, no iteration)",
        sigma_symbol="RMS",
    )

    # --- Precision vs η (iterative RMS, 3σ clipping) ---
    print("  Computing ML precision vs η (iterative RMS) ...")
    eta_bins_irms, ml_prec_irms, ml_rms_cuts = compute_precision_vs_eta_iterative(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, method="iterative_rms",
    )
    acts_prec_irms = None
    acts_rms_cuts = None
    if acts_residuals is not None:
        _, acts_prec_irms, acts_rms_cuts = compute_precision_vs_eta_iterative(
            acts_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, method="iterative_rms",
        )
    ml_rms_count = int(np.mean([ml_rms_cuts[p]["n_kept"] for p in PARAMS]))
    print("  Plotting precision vs η (iterative RMS) ...")
    _plot_summary_precision(
        ml_prec_irms, eta_bins_irms, output_dir,
        ml_label=ml_label, acts_data=acts_prec_irms, acts_label=acts_label,
        eta_values=eta_values, regime_subtitle=regime_subtitle,
        filename="precision_summary_iterative_rms.png",
        title_suffix=" (iterative RMS, 3σ clip)",
        sigma_symbol="RMS",
        track_count=ml_rms_count,
        autoscale_y=True,
        ml_cuts=ml_rms_cuts,
        acts_cuts=acts_rms_cuts,
    )

    # --- Precision vs η (iterative Gaussian fit, 2σ clipping) ---
    print("  Computing ML precision vs η (Gaussian fit) ...")
    eta_bins_igf, ml_prec_igf, ml_gauss_cuts = compute_precision_vs_eta_iterative(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, method="iterative_gaussfit",
    )
    acts_prec_igf = None
    acts_gauss_cuts = None
    if acts_residuals is not None:
        _, acts_prec_igf, acts_gauss_cuts = compute_precision_vs_eta_iterative(
            acts_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, method="iterative_gaussfit",
        )
    ml_gf_count = int(np.mean([ml_gauss_cuts[p]["n_kept"] for p in PARAMS]))
    print("  Plotting precision vs η (Gaussian fit) ...")
    _plot_summary_precision(
        ml_prec_igf, eta_bins_igf, output_dir,
        ml_label=ml_label, acts_data=acts_prec_igf, acts_label=acts_label,
        eta_values=eta_values, regime_subtitle=regime_subtitle,
        filename="precision_summary_gaussfit.png",
        title_suffix=" (Gaussian core fit, 2σ clip)",
        sigma_symbol="σ",
        track_count=ml_gf_count,
        autoscale_y=True,
        ml_cuts=ml_gauss_cuts,
        acts_cuts=acts_gauss_cuts,
    )

    # --- Core-95% metrics (RMS, std, IQR) vs η, no iteration ---
    print("  Computing core-95% metrics vs η ...")
    eta_bins_c95, ml_core95 = compute_core_metrics_vs_eta(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, core_fraction=0.95,
    )
    acts_core95 = None
    if acts_residuals is not None:
        _, acts_core95 = compute_core_metrics_vs_eta(
            acts_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins, core_fraction=0.95,
        )
    print("  Plotting core-95% metrics summary ...")
    _plot_core_metrics_summary(
        ml_core95, eta_bins_c95, output_dir,
        ml_label=ml_label, acts_data=acts_core95, acts_label=acts_label,
        eta_values=eta_values, regime_subtitle=regime_subtitle,
        filename="precision_summary_core95.png",
        core_fraction=0.95,
    )

    # --- Distribution summary (target vs prediction) ---
    if ml_preds is not None and ml_targets is not None:
        print("  Plotting SSM distribution summary ...")
        plot_distribution_summary(ml_targets, ml_preds, output_dir,
                                  pred_label=ml_label, target_label="Truth",
                                  regime_subtitle=regime_subtitle)

    if acts_reco_values is not None and acts_reco_targets is not None:
        print("  Plotting ACTS reco distribution summary ...")
        plot_distribution_summary(acts_reco_targets, acts_reco_values, output_dir,
                                  pred_label=acts_label, target_label="Truth",
                                  filename="distribution_acts.png",
                                  color="darkorange",
                                  regime_subtitle=regime_subtitle)

    # --- Residual distributions ---
    print("  Plotting residual distribution summary ...")
    plot_residual_distribution_summary(
        ml_residuals, output_dir, ml_label=ml_label,
        acts_residuals=acts_residuals, acts_label=acts_label,
        regime_subtitle=regime_subtitle,
    )

    print("  Plotting log-scale residual distributions ...")
    plot_residual_log_distributions(
        ml_residuals, output_dir, ml_label=ml_label,
        acts_residuals=acts_residuals, acts_label=acts_label,
        regime_subtitle=regime_subtitle,
        ml_rms_cuts=ml_rms_cuts,
        ml_gauss_cuts=ml_gauss_cuts,
        acts_rms_cuts=acts_rms_cuts,
        acts_gauss_cuts=acts_gauss_cuts,
    )

    # --- Clipped residual distributions (linear + log) ---
    for log, suffix in [(False, ""), (True, "_log")]:
        print(f"  Plotting clipped residual distributions (iterative RMS{', log' if log else ''}) ...")
        plot_clipped_residual_summary(
            ml_residuals, output_dir, ml_cuts=ml_rms_cuts,
            ml_label=ml_label, acts_residuals=acts_residuals,
            acts_cuts=acts_rms_cuts, acts_label=acts_label,
            filename=f"clipped_residual_summary_rms{suffix}.png",
            title_suffix=" (iterative RMS, 3σ clip)",
            regime_subtitle=regime_subtitle, log_scale=log,
        )
        print(f"  Plotting clipped residual distributions (Gaussian fit{', log' if log else ''}) ...")
        plot_clipped_residual_summary(
            ml_residuals, output_dir, ml_cuts=ml_gauss_cuts,
            ml_label=ml_label, acts_residuals=acts_residuals,
            acts_cuts=acts_gauss_cuts, acts_label=acts_label,
            filename=f"clipped_residual_summary_gaussfit{suffix}.png",
            title_suffix=" (Gaussian core fit, 2σ clip)",
            regime_subtitle=regime_subtitle, log_scale=log,
        )

    # --- Mean residual vs η ---
    print("  Plotting mean residual vs η ...")
    plot_mean_residual_vs_eta(
        ml_residuals, eta_range, n_eta_bins, output_dir,
        ml_label=ml_label, acts_residuals=acts_residuals, acts_label=acts_label,
        ml_rms_cuts=ml_rms_cuts, ml_gauss_cuts=ml_gauss_cuts,
        acts_rms_cuts=acts_rms_cuts, acts_gauss_cuts=acts_gauss_cuts,
        regime_subtitle=regime_subtitle,
    )

    # --- Precision and mean residual vs pT ---
    if ml_targets is not None and "theta" in ml_targets and "qop" in ml_targets:
        ml_pt = np.abs(np.sin(ml_targets["theta"]) / (ml_targets["qop"] + 1e-12))
        pt_max = min(float(np.percentile(ml_pt, 99.5)), 20.0)
        pt_bins = np.linspace(0, pt_max, 31)

        acts_pt = None
        if (acts_reco_targets is not None
                and "theta" in acts_reco_targets and "qop" in acts_reco_targets):
            acts_pt = np.abs(np.sin(acts_reco_targets["theta"])
                             / (acts_reco_targets["qop"] + 1e-12))

        print("  Plotting precision vs pT ...")
        plot_precision_vs_pt(
            ml_residuals, ml_pt, pt_bins, output_dir,
            ml_label=ml_label, acts_residuals=acts_residuals, acts_pt=acts_pt,
            acts_label=acts_label, regime_subtitle=regime_subtitle,
            ml_rms_cuts=ml_rms_cuts, acts_rms_cuts=acts_rms_cuts,
        )
        print("  Plotting mean residual vs pT ...")
        plot_mean_residual_vs_pt(
            ml_residuals, ml_pt, pt_bins, output_dir,
            ml_label=ml_label, acts_residuals=acts_residuals, acts_pt=acts_pt,
            acts_label=acts_label, regime_subtitle=regime_subtitle,
        )

    # --- SSM vs ACTS correlation heatmap ---
    if acts_reco_values is not None and ml_preds is not None:
        # Ensure SSM preds and ACTS reco are aligned (same tracks)
        ml_for_heatmap = kwargs.get("ml_preds_matched", ml_preds)
        first_param = PARAMS[0]
        if (first_param in ml_for_heatmap and first_param in acts_reco_values
                and len(ml_for_heatmap[first_param]) == len(acts_reco_values[first_param])):
            print("  Plotting SSM vs ACTS correlation heatmap ...")
            plot_ssm_vs_acts_heatmap(
                ml_for_heatmap, acts_reco_values, output_dir,
                ml_label=ml_label, acts_label=acts_label,
                regime_subtitle=regime_subtitle,
            )

    # --- Residual statistics report ---
    print("  Writing residual statistics report ...")
    write_residual_statistics_report(
        ml_residuals, output_dir, ml_label=ml_label,
        acts_residuals=acts_residuals, acts_label=acts_label,
        regime_subtitle=regime_subtitle,
        eta_range=eta_range, n_eta_bins=n_eta_bins,
    )

    print(f"  Plots saved to {output_dir}")
    return {
        "ml_precision": ml_precision,
        "acts_precision": acts_prec,
        "ml_precision_irms": ml_prec_irms,
        "acts_precision_irms": acts_prec_irms,
        "ml_precision_igf": ml_prec_igf,
        "acts_precision_igf": acts_prec_igf,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate track regression predictions: precision and residual plots"
    )
    add_common_args(parser)
    args = parser.parse_args()

    ctx = load_all_data(args)
    regimes = build_regime_data(ctx)

    for regime_name, kwargs in regimes:
        print(f"\n{'=' * 70}")
        print(f"Regime: {regime_name}")
        print(f"{'=' * 70}")
        result = run_evaluation_suite(**kwargs)
        print_precision_summary(
            result["ml_precision"], regime_name,
            result["acts_precision"],
        )

    print(f"\nAll evaluation plots saved to {ctx['output_dir']}")


if __name__ == "__main__":
    main()
