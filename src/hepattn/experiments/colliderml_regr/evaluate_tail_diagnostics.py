#!/usr/bin/env python3
"""Tail diagnostics and bias checks for track regression predictions.

Generates:
- Prediction vs truth heatmaps (beamspot bias check)
- Mean residual vs truth d0 (beamspot bias check)
- Core vs tail width profiles as a function of η and pT
- Tail track kinematic distributions
- Tail overlap scatter (SSM vs ACTS residual correlation)
- Residual statistics report

Usage::

    python -m hepattn.experiments.colliderml_regr.evaluate_tail_diagnostics \
        --predictions /path/to/test_predictions.h5 \
        --data-dir /scratch/colliderml/p0/p0_preprocessed \
        --output-dir /path/to/eval_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    load_all_data,
    print_precision_summary,
    write_residual_statistics_report,
)


# ============================================================================
# Prediction vs truth heatmaps
# ============================================================================

def plot_pred_vs_truth_heatmap(
    targets: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    output_dir: Path,
    pred_label: str = "SSM",
    params: list[str] | None = None,
    n_bins: int = 150,
    filename: str | None = None,
    regime_subtitle: str | None = None,
) -> None:
    """2-D histogram (heatmap) of predicted vs truth values per parameter."""
    from matplotlib.colors import LogNorm

    output_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = [p for p in PARAMS if p in targets and p in preds]
    else:
        params = [p for p in params if p in targets and p in preds]
    if not params:
        return

    if filename is None:
        filename = "heatmap_ssm.png"

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        t = targets[name]
        p = preds[name]

        if name in HEATMAP_RANGE:
            lo, hi = HEATMAP_RANGE[name]
        elif name in FULL_RANGE_PARAMS:
            lo = min(float(np.min(t)), float(np.min(p)))
            hi = max(float(np.max(t)), float(np.max(p)))
        else:
            lo = min(float(np.percentile(t, 0.5)), float(np.percentile(p, 0.5)))
            hi = max(float(np.percentile(t, 99.5)), float(np.percentile(p, 99.5)))
        bins = np.linspace(lo, hi, n_bins + 1)

        h = ax.hist2d(t, p, bins=bins, norm=LogNorm(), cmin=1, cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="Counts")

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, alpha=0.8, label="y = x")

        label = PARAM_VALUE_LABELS.get(name, name)
        ax.set_xlabel(f"Truth {label}", fontsize=11)
        ax.set_ylabel(f"Predicted {label}", fontsize=11)
        ax.set_title(f"{name.upper()} {pred_label} Prediction vs Truth", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = f"{pred_label} — Predicted vs Truth"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Mean residual vs truth d0 (beamspot bias check)
# ============================================================================

def plot_mean_residual_vs_truth_d0(
    ml_targets: dict[str, np.ndarray],
    ml_preds: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_reco_values: dict[str, np.ndarray] | None = None,
    acts_reco_targets: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    n_bins: int = 40,
    filename: str = "bias_mean_vs_d0.png",
    regime_subtitle: str | None = None,
) -> None:
    """Plot mean residual of d0 and phi vs truth d0 to check beamspot bias."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "d0" not in ml_targets or "d0" not in ml_preds:
        return

    check_params = [p for p in ["d0", "phi"] if p in ml_targets and p in ml_preds]

    MEAN_LABELS = {
        "d0": r"$\mu(d_0)$ [mm]",
        "phi": r"$\mu(\phi)$ [mrad]",
    }

    n_cols = len(check_params)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    truth_d0 = ml_targets["d0"]
    lo = float(np.min(truth_d0))
    hi = float(np.max(truth_d0))
    d0_bins = np.linspace(lo, hi, n_bins + 1)
    bin_centers = 0.5 * (d0_bins[:-1] + d0_bins[1:])

    for col, name in enumerate(check_params):
        ax = axes[col]
        scale = UNIT_SCALE.get(name, 1.0)

        ml_res = (ml_preds[name] - ml_targets[name]) * scale
        ml_means, ml_errs = [], []
        for i in range(len(d0_bins) - 1):
            mask = (truth_d0 >= d0_bins[i]) & (truth_d0 < d0_bins[i + 1])
            n = int(np.sum(mask))
            if n > 2:
                ml_means.append(float(np.mean(ml_res[mask])))
                ml_errs.append(float(np.std(ml_res[mask]) / np.sqrt(n)))
            else:
                ml_means.append(np.nan)
                ml_errs.append(np.nan)

        ax.errorbar(bin_centers, np.array(ml_means), yerr=np.array(ml_errs), fmt="^",
                     color="steelblue", markersize=5, linewidth=1.2,
                     capsize=2, label=ml_label)

        if (acts_reco_values is not None and acts_reco_targets is not None
                and name in acts_reco_values and "d0" in acts_reco_targets):
            acts_truth_d0 = acts_reco_targets["d0"]
            acts_res = (acts_reco_values[name] - acts_reco_targets[name]) * scale

            acts_means, acts_errs = [], []
            for i in range(len(d0_bins) - 1):
                mask = (acts_truth_d0 >= d0_bins[i]) & (acts_truth_d0 < d0_bins[i + 1])
                n = int(np.sum(mask))
                if n > 2:
                    acts_means.append(float(np.mean(acts_res[mask])))
                    acts_errs.append(float(np.std(acts_res[mask]) / np.sqrt(n)))
                else:
                    acts_means.append(np.nan)
                    acts_errs.append(np.nan)

            ax.errorbar(bin_centers, np.array(acts_means), yerr=np.array(acts_errs), fmt="o",
                         color="darkorange", markersize=5, linewidth=1.2,
                         capsize=2, label=acts_label)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        if name == "phi":
            ax.set_ylim(-10, 10)
        ax.set_xlabel(PARAM_VALUE_LABELS.get("d0", r"$d_0$ [mm]"), fontsize=12)
        ax.set_ylabel(MEAN_LABELS.get(name, f"μ({name})"), fontsize=12)
        ax.set_title(f"Mean {name.upper()} Residual vs Truth $d_0$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    title = "Beamspot Bias Check"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Core vs tail width profiles
# ============================================================================

def _compute_binned_widths(
    residuals: np.ndarray,
    bin_var: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute core (inner 68%) and tail (outer 32%) widths per bin.

    Core: (P84.13 - P15.87) / 2  (equivalent to 1σ for Gaussian).
    Tail: std of residuals outside the inner 68%.
    """
    centers, cores, tails = [], [], []
    for i in range(len(bin_edges) - 1):
        mask = (bin_var >= bin_edges[i]) & (bin_var < bin_edges[i + 1])
        n = int(np.sum(mask))
        centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
        if n > 10:
            r = residuals[mask]
            p16, p84 = np.percentile(r, [15.87, 84.13])
            cores.append((p84 - p16) / 2.0)
            tail_mask = (r < p16) | (r > p84)
            tails.append(float(np.std(r[tail_mask])) if np.sum(tail_mask) > 2 else np.nan)
        else:
            cores.append(np.nan)
            tails.append(np.nan)
    return np.array(centers), np.array(cores), np.array(tails)


def plot_tail_profile_vs_kinematics(
    ml_residuals: dict[str, np.ndarray],
    ml_targets: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_targets: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    n_eta_bins: int = 20,
    n_pt_bins: int = 20,
    filename: str = "tail_profile_summary.png",
    regime_subtitle: str | None = None,
) -> None:
    """Core width (inner 68% half-width) and tail width (outer 32% std) vs eta and pT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return

    ml_eta = ml_residuals["eta"]
    ml_pt = np.abs(np.sin(ml_targets["theta"]) / (ml_targets["qop"] + 1e-12))

    eta_edges = np.linspace(-3.0, 3.0, n_eta_bins + 1)
    pt_edges = np.linspace(0.5, float(np.percentile(ml_pt, 99.5)), n_pt_bins + 1)

    has_acts = acts_residuals is not None and acts_targets is not None
    if has_acts:
        acts_eta = acts_residuals["eta"]
        acts_pt = np.abs(np.sin(acts_targets["theta"]) / (acts_targets["qop"] + 1e-12))

    n_cols = len(params)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(params):
        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name] * scale
        unit_label = PARAM_LABELS.get(name, f"Width ({name})")

        for row, (bin_var_ml, edges, xlabel) in enumerate([
            (ml_eta, eta_edges, r"$\eta_{\mathrm{truth}}$"),
            (ml_pt, pt_edges, r"$p_T^{\mathrm{truth}}$ [GeV]"),
        ]):
            ax = axes[row, col]
            centers, ml_core, ml_tail = _compute_binned_widths(ml_res, bin_var_ml, edges)

            ax.plot(centers, ml_core, "o-", color="steelblue", markersize=3,
                    linewidth=1.5, label=f"{ml_label} core")
            ax.plot(centers, ml_tail, "s--", color="steelblue", markersize=3,
                    linewidth=1.2, alpha=0.7, label=f"{ml_label} tail")

            if has_acts:
                acts_res = acts_residuals[name] * scale
                bin_var_acts = acts_eta if row == 0 else acts_pt
                a_centers, a_core, a_tail = _compute_binned_widths(
                    acts_res, bin_var_acts, edges)
                ax.plot(a_centers, a_core, "o-", color="darkorange", markersize=3,
                        linewidth=1.5, label=f"{acts_label} core")
                ax.plot(a_centers, a_tail, "s--", color="darkorange", markersize=3,
                        linewidth=1.2, alpha=0.7, label=f"{acts_label} tail")

            ax.set_xlabel(xlabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(unit_label, fontsize=10)
            if row == 0:
                ax.set_title(name.upper(), fontsize=12)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            if col == n_cols - 1:
                ax.legend(fontsize=7, loc="upper right")

    title = "Core (inner 68% half-width) vs Tail (outer 32% std) Width"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Tail track kinematics
# ============================================================================

def plot_tail_track_kinematics(
    ml_residuals: dict[str, np.ndarray],
    ml_targets: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_targets: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    nhits: np.ndarray | None = None,
    acts_nhits: np.ndarray | None = None,
    regime_subtitle: str | None = None,
) -> None:
    """Show eta/pT/nhits distributions of tail tracks (outside inner 68%) vs all tracks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals]
    if not params:
        return

    ml_eta = ml_residuals["eta"]
    ml_pt = np.abs(np.sin(ml_targets["theta"]) / (ml_targets["qop"] + 1e-12))

    has_acts = acts_residuals is not None and acts_targets is not None
    if has_acts:
        acts_eta = acts_residuals["eta"]
        acts_pt = np.abs(np.sin(acts_targets["theta"]) / (acts_targets["qop"] + 1e-12))

    for name in params:
        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name] * scale

        ml_p16, ml_p84 = np.percentile(ml_res, [15.87, 84.13])
        ml_tail_mask = (ml_res < ml_p16) | (ml_res > ml_p84)
        ml_tail_frac = np.sum(ml_tail_mask) / len(ml_res)

        kin_vars = [
            ("eta", ml_eta, r"$\eta_{\mathrm{truth}}$", np.linspace(-3, 3, 40)),
            ("pT", ml_pt, r"$p_T^{\mathrm{truth}}$ [GeV]",
             np.linspace(0.5, float(np.percentile(ml_pt, 99.5)), 40)),
        ]
        if nhits is not None:
            kin_vars.append(
                ("nhits", nhits, "Number of hits",
                 np.arange(int(np.min(nhits)) - 0.5, int(np.percentile(nhits, 99.5)) + 1.5, 1))
            )

        n_cols = len(kin_vars)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for col, (vname, ml_var, xlabel, bins) in enumerate(kin_vars):
            ax = axes[col]

            ax.hist(ml_var, bins=bins, histtype="stepfilled", density=True,
                    color="lightgray", edgecolor="gray", linewidth=0.8, label="All tracks")

            ax.hist(ml_var[ml_tail_mask], bins=bins, histtype="step", density=True,
                    color="steelblue", linewidth=1.8,
                    label=f"{ml_label} tail ({100 * ml_tail_frac:.1f}%)")

            if has_acts:
                acts_res = acts_residuals[name] * scale
                acts_p16, acts_p84 = np.percentile(acts_res, [15.87, 84.13])
                acts_tail_mask = (acts_res < acts_p16) | (acts_res > acts_p84)
                acts_tail_frac = np.sum(acts_tail_mask) / len(acts_res)

                acts_var = acts_eta if vname == "eta" else (
                    acts_pt if vname == "pT" else acts_nhits)
                if acts_var is not None:
                    ax.hist(acts_var[acts_tail_mask], bins=bins, histtype="step", density=True,
                            color="darkorange", linewidth=1.8, linestyle="--",
                            label=f"{acts_label} tail ({100 * acts_tail_frac:.1f}%)")

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        title = f"{name.upper()} — Kinematic Profile of Tail Tracks (outside inner 68%)"
        if regime_subtitle:
            title += f"\n{regime_subtitle}"
        plt.suptitle(title, fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / f"tail_kinematics_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================================
# Tail overlap scatter
# ============================================================================

def plot_tail_overlap_scatter(
    ml_residuals: dict[str, np.ndarray],
    acts_residuals: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_label: str = "ACTS CKF",
    n_bins: int = 120,
    filename: str = "tail_overlap_scatter.png",
    regime_subtitle: str | None = None,
) -> None:
    """2D histogram of SSM vs ACTS residuals to check if tail tracks overlap."""
    from matplotlib.colors import LogNorm

    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in PARAMS if p in ml_residuals and p in acts_residuals]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        scale = UNIT_SCALE.get(name, 1.0)
        ml_res = ml_residuals[name] * scale
        acts_res = acts_residuals[name] * scale

        combined = np.concatenate([ml_res, acts_res])
        lim = float(np.percentile(np.abs(combined), 99.5))
        bins = np.linspace(-lim, lim, n_bins + 1)

        h = ax.hist2d(ml_res, acts_res, bins=bins, norm=LogNorm(), cmin=1, cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="Counts")

        ml_q25, ml_q75 = np.percentile(ml_res, [25, 75])
        acts_q25, acts_q75 = np.percentile(acts_res, [25, 75])
        for q in [ml_q25, ml_q75]:
            ax.axvline(q, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6)
        for q in [acts_q25, acts_q75]:
            ax.axhline(q, color="darkorange", linewidth=0.8, linestyle=":", alpha=0.6)

        ml_iqr = ml_q75 - ml_q25
        acts_iqr = acts_q75 - acts_q25
        ml_core = (ml_res >= ml_q25 - 1.5 * ml_iqr) & (ml_res <= ml_q75 + 1.5 * ml_iqr)
        acts_core = (acts_res >= acts_q25 - 1.5 * acts_iqr) & (acts_res <= acts_q75 + 1.5 * acts_iqr)
        n_total = len(ml_res)
        n_both_core = np.sum(ml_core & acts_core)
        n_ml_tail = np.sum(~ml_core & acts_core)
        n_acts_tail = np.sum(ml_core & ~acts_core)
        n_both_tail = np.sum(~ml_core & ~acts_core)

        stats_text = (
            f"Both core: {100 * n_both_core / n_total:.1f}%\n"
            f"{ml_label} tail only: {100 * n_ml_tail / n_total:.1f}%\n"
            f"{acts_label} tail only: {100 * n_acts_tail / n_total:.1f}%\n"
            f"Both tail: {100 * n_both_tail / n_total:.1f}%"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.8))

        unit = {1.0: "", 1e3: " [mrad]"}.get(scale, "")
        ax.set_xlabel(f"{ml_label} $\\Delta${name}{unit}", fontsize=10)
        ax.set_ylabel(f"{acts_label} $\\Delta${name}{unit}", fontsize=10)
        ax.set_title(name.upper(), fontsize=12)
        ax.grid(True, alpha=0.3)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = f"Residual Correlation — {ml_label} vs {acts_label}"
    if regime_subtitle:
        title += f"\n{regime_subtitle}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Evaluation suite
# ============================================================================

def run_tail_diagnostics(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    ml_preds: dict[str, np.ndarray] | None = None,
    ml_targets: dict[str, np.ndarray] | None = None,
    acts_reco_values: dict[str, np.ndarray] | None = None,
    acts_reco_targets: dict[str, np.ndarray] | None = None,
    nhits: np.ndarray | None = None,
    acts_nhits: np.ndarray | None = None,
    regime_subtitle: str | None = None,
    **kwargs,
) -> None:
    """Run all tail diagnostic plots for one selection regime."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Residual statistics report ---
    print("  Writing residual statistics report ...")
    write_residual_statistics_report(
        ml_residuals, output_dir, ml_label=ml_label,
        acts_residuals=acts_residuals, acts_label=acts_label,
        regime_subtitle=regime_subtitle,
    )

    # --- Prediction vs truth heatmaps ---
    if ml_preds is not None and ml_targets is not None:
        print("  Plotting SSM prediction vs truth heatmaps ...")
        plot_pred_vs_truth_heatmap(ml_targets, ml_preds, output_dir,
                                   pred_label=ml_label, filename="heatmap_ssm.png",
                                   regime_subtitle=regime_subtitle)

    if acts_reco_values is not None and acts_reco_targets is not None:
        print("  Plotting ACTS prediction vs truth heatmaps ...")
        plot_pred_vs_truth_heatmap(acts_reco_targets, acts_reco_values, output_dir,
                                   pred_label=acts_label, filename="heatmap_acts.png",
                                   regime_subtitle=regime_subtitle)

    # --- Mean residual vs truth d0 (beamspot bias check) ---
    if ml_preds is not None and ml_targets is not None:
        print("  Plotting mean residual vs truth d0 (beamspot bias check) ...")
        plot_mean_residual_vs_truth_d0(
            ml_targets, ml_preds, output_dir, ml_label=ml_label,
            acts_reco_values=acts_reco_values,
            acts_reco_targets=acts_reco_targets,
            acts_label=acts_label,
            regime_subtitle=regime_subtitle,
        )

    # --- Core vs tail width profiles ---
    if ml_targets is not None:
        print("  Plotting core vs tail width profiles ...")
        plot_tail_profile_vs_kinematics(
            ml_residuals, ml_targets, output_dir, ml_label=ml_label,
            acts_residuals=acts_residuals,
            acts_targets=acts_reco_targets,
            acts_label=acts_label,
            regime_subtitle=regime_subtitle,
        )

        print("  Plotting tail track kinematics ...")
        plot_tail_track_kinematics(
            ml_residuals, ml_targets, output_dir, ml_label=ml_label,
            acts_residuals=acts_residuals,
            acts_targets=acts_reco_targets,
            acts_label=acts_label,
            nhits=nhits,
            acts_nhits=acts_nhits,
            regime_subtitle=regime_subtitle,
        )

    # --- Tail overlap scatter ---
    if acts_residuals is not None:
        ml_n = len(ml_residuals.get("eta", []))
        acts_n = len(acts_residuals.get("eta", []))
        if ml_n == acts_n:
            print("  Plotting tail overlap scatter ...")
            plot_tail_overlap_scatter(
                ml_residuals, acts_residuals, output_dir,
                ml_label=ml_label, acts_label=acts_label,
                regime_subtitle=regime_subtitle,
            )
        else:
            print(f"  Skipping tail overlap scatter (ML={ml_n:,} vs ACTS={acts_n:,} tracks)")

    print(f"  Plots saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tail diagnostics and bias checks for track regression predictions"
    )
    add_common_args(parser)
    args = parser.parse_args()

    ctx = load_all_data(args)
    regimes = build_regime_data(ctx)

    for regime_name, kwargs in regimes:
        print(f"\n{'=' * 70}")
        print(f"Regime: {regime_name}")
        print(f"{'=' * 70}")
        run_tail_diagnostics(**kwargs)

    print(f"\nAll tail diagnostic plots saved to {ctx['output_dir']}")


if __name__ == "__main__":
    main()
