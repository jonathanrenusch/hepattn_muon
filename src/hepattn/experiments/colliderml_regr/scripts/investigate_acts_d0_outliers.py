#!/usr/bin/env python3
"""Investigate ACTS CKF d0 outliers: tracks where truth d0 ≈ 0 but ACTS reco d0 is far off.

Produces diagnostic plots and a statistical report to characterise the
vertical stripe visible in the ACTS d0 prediction-vs-truth heatmap.

Usage::

    python -m hepattn.experiments.colliderml_regr.scripts.investigate_acts_d0_outliers \
        --data-dir /scratch/colliderml/p200_loose_finetune \
        --output-dir /shared/tracking/hepattn_muon/analysis/acts_d0_outlier_investigation \
        --n-shards 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm


# ============================================================================
# Constants
# ============================================================================

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
PARAM_IDX = {n: i for i, n in enumerate(PARAMS)}


# ============================================================================
# Data loading
# ============================================================================

def load_shard_data(shard_dir: Path) -> dict | None:
    """Load targets, ACTS reco, DM mask, metadata, and hit info from one shard."""
    sel_dir = shard_dir / "selected_tracks"
    targets_f = sel_dir / "track_targets.npy"
    reco_f = sel_dir / "acts_reco.npy"
    dm_f = sel_dir / "acts_dm_mask.npy"
    meta_f = sel_dir / "track_meta.npy"
    offsets_f = sel_dir / "track_hit_offsets.npy"
    pids_f = sel_dir / "track_particle_ids.npy"

    if not targets_f.exists() or not reco_f.exists():
        return None

    targets = np.load(targets_f)        # (N, 5)
    acts_reco = np.load(reco_f)          # (N, 5)
    acts_dm = np.load(dm_f)              # (N,)
    meta = np.load(meta_f)               # (N, 2): [pt, vertex_primary]
    offsets = np.load(offsets_f)          # (N+1,)
    nhits = np.diff(offsets).astype(np.int32)

    pids = np.load(pids_f) if pids_f.exists() else np.zeros(len(targets), dtype=np.int64)

    return {
        "targets": targets,
        "acts_reco": acts_reco,
        "acts_dm": acts_dm,
        "pt": meta[:, 0],
        "vertex_primary": meta[:, 1],
        "nhits": nhits,
        "particle_ids": pids,
    }


def load_dataset(data_dir: Path, split: str = "all", n_shards: int = -1) -> dict:
    """Load data from multiple shards.  split='all' loads every shard."""
    split_file = data_dir / "split.json"
    if split != "all" and split_file.exists():
        with open(split_file) as f:
            splits = json.load(f)
        shard_indices = sorted(splits[split])
    else:
        # Discover all shard directories
        shard_indices = sorted(
            int(d.name.split("_")[1])
            for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("shard_")
        )

    if n_shards > 0:
        shard_indices = shard_indices[:n_shards]

    accum = {k: [] for k in ["targets", "acts_reco", "acts_dm", "pt",
                               "vertex_primary", "nhits", "particle_ids",
                               "shard_idx"]}

    for idx in tqdm(shard_indices, desc="Loading shards", file=sys.stderr):
        shard_dir = data_dir / f"shard_{idx:04d}"
        d = load_shard_data(shard_dir)
        if d is None:
            continue
        n = len(d["targets"])
        for k in d:
            accum[k].append(d[k])
        accum["shard_idx"].append(np.full(n, idx, dtype=np.int32))

    return {k: np.concatenate(v, axis=0) for k, v in accum.items()}


# ============================================================================
# Analysis helpers
# ============================================================================

def compute_eta(theta: np.ndarray) -> np.ndarray:
    return -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))


def flag_d0_outliers(
    truth_d0: np.ndarray,
    acts_d0: np.ndarray,
    truth_d0_window: float = 0.5,
    acts_d0_threshold: float = 1.0,
) -> np.ndarray:
    """Flag tracks where |truth_d0| < window but |acts_d0| > threshold."""
    return (np.abs(truth_d0) < truth_d0_window) & (np.abs(acts_d0) > acts_d0_threshold)


# ============================================================================
# Plotting
# ============================================================================

def plot_heatmap_annotated(
    truth_d0: np.ndarray,
    acts_d0: np.ndarray,
    outlier_mask: np.ndarray,
    output_dir: Path,
    dm_label: str = "",
):
    """Reproduce the d0 heatmap with outlier region highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    bins = np.linspace(-5, 5, 201)

    # Left: full heatmap
    ax = axes[0]
    h, xe, ye = np.histogram2d(truth_d0, acts_d0, bins=[bins, bins])
    h = np.ma.masked_where(h == 0, h)
    pcm = ax.pcolormesh(xe, ye, h.T, cmap="viridis", norm=LogNorm(vmin=1))
    fig.colorbar(pcm, ax=ax, pad=0.02, aspect=30)
    ax.plot([-5, 5], [-5, 5], "r--", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"Truth $d_0$ [mm]")
    ax.set_ylabel(r"ACTS CKF $d_0$ [mm]")
    ax.set_title(f"ACTS d0 vs Truth{dm_label}\n({len(truth_d0):,} tracks)")
    ax.set_aspect("equal")

    # Highlight outlier box
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, -5), 1.0, 10, linewidth=2,
                     edgecolor="red", facecolor="none", linestyle="--")
    ax.add_patch(rect)
    n_out = int(np.sum(outlier_mask))
    ax.text(0.5, -4.5, f"{n_out:,} outliers\nin red box",
            color="red", fontsize=10, ha="left", va="bottom")

    # Right: zoomed into the outlier region
    ax = axes[1]
    zoom_bins_x = np.linspace(-1, 1, 101)
    zoom_bins_y = np.linspace(-5, 5, 201)
    h2, xe2, ye2 = np.histogram2d(truth_d0, acts_d0, bins=[zoom_bins_x, zoom_bins_y])
    h2 = np.ma.masked_where(h2 == 0, h2)
    pcm2 = ax.pcolormesh(xe2, ye2, h2.T, cmap="viridis", norm=LogNorm(vmin=1))
    fig.colorbar(pcm2, ax=ax, pad=0.02, aspect=30)
    ax.plot([-1, 1], [-1, 1], "r--", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"Truth $d_0$ [mm]")
    ax.set_ylabel(r"ACTS CKF $d_0$ [mm]")
    ax.set_title(f"Zoomed: |truth d0| < 1 mm{dm_label}")

    plt.tight_layout()
    fig.savefig(output_dir / "d0_heatmap_annotated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_kinematics(
    data: dict,
    outlier_mask: np.ndarray,
    good_mask: np.ndarray,
    output_dir: Path,
):
    """Compare kinematic distributions of outlier vs good tracks."""
    targets = data["targets"]
    acts_reco = data["acts_reco"]
    pt = data["pt"]
    nhits = data["nhits"]
    vp = data["vertex_primary"]

    eta_truth = compute_eta(targets[:, PARAM_IDX["theta"]])

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    def _hist_compare(ax, vals_good, vals_out, bins, xlabel, logy=False):
        ax.hist(vals_good, bins=bins, histtype="step", linewidth=1.5,
                color="steelblue", label=f"Good ({len(vals_good):,})", density=True)
        ax.hist(vals_out, bins=bins, histtype="step", linewidth=1.5,
                color="red", label=f"Outlier ({len(vals_out):,})", density=True)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Normalised density", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")

    # Row 0: pT, eta, nhits
    _hist_compare(axes[0, 0], pt[good_mask], pt[outlier_mask],
                  np.linspace(0, 5, 80), r"$p_T$ [GeV]", logy=True)
    _hist_compare(axes[0, 1], eta_truth[good_mask], eta_truth[outlier_mask],
                  np.linspace(-3, 3, 60), r"$\eta_{\mathrm{truth}}$")
    _hist_compare(axes[0, 2], nhits[good_mask], nhits[outlier_mask],
                  np.arange(2.5, 21.5, 1), "Number of hits")

    # Row 1: truth d0, z0, phi
    _hist_compare(axes[1, 0], targets[good_mask, 0], targets[outlier_mask, 0],
                  np.linspace(-1, 1, 80), r"Truth $d_0$ [mm]")
    _hist_compare(axes[1, 1], targets[good_mask, 1], targets[outlier_mask, 1],
                  np.linspace(-200, 200, 80), r"Truth $z_0$ [mm]")
    _hist_compare(axes[1, 2], targets[good_mask, 2], targets[outlier_mask, 2],
                  np.linspace(-np.pi, np.pi, 80), r"Truth $\phi$ [rad]")

    # Row 2: truth theta, qop, vertex_primary
    _hist_compare(axes[2, 0], targets[good_mask, 3], targets[outlier_mask, 3],
                  np.linspace(0, np.pi, 80), r"Truth $\theta$ [rad]")
    _hist_compare(axes[2, 1], targets[good_mask, 4], targets[outlier_mask, 4],
                  np.linspace(-2, 2, 80), r"Truth $q/p$ [1/GeV]")

    # Vertex primary as bar chart
    ax = axes[2, 2]
    cats = [0, 1]
    good_counts = [np.sum(vp[good_mask] == c) for c in cats]
    out_counts = [np.sum(vp[outlier_mask] == c) for c in cats]
    x = np.arange(len(cats))
    w = 0.35
    good_frac = np.array(good_counts) / max(sum(good_counts), 1)
    out_frac = np.array(out_counts) / max(sum(out_counts), 1)
    ax.bar(x - w / 2, good_frac, w, color="steelblue", label="Good")
    ax.bar(x + w / 2, out_frac, w, color="red", label="Outlier")
    ax.set_xticks(x)
    ax.set_xticklabels(["Non-primary vtx", "Primary vtx"])
    ax.set_ylabel("Fraction")
    ax.set_title("vertex_primary")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Kinematic Comparison: Outlier vs Good Tracks (DM, |truth d0| < 0.5 mm)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "outlier_kinematics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acts_residuals_all_params(
    data: dict,
    outlier_mask: np.ndarray,
    good_mask: np.ndarray,
    output_dir: Path,
):
    """Show ACTS residuals for all 5 params, comparing outlier vs good tracks."""
    targets = data["targets"]
    acts_reco = data["acts_reco"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    scales = {"d0": 1.0, "z0": 1.0, "phi": 1e3, "theta": 1e3, "qop": 1.0}
    units = {"d0": "mm", "z0": "mm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV"}

    for i, name in enumerate(PARAMS):
        ax = axes[i]
        idx = PARAM_IDX[name]
        scale = scales[name]

        res_good = (acts_reco[good_mask, idx] - targets[good_mask, idx]) * scale
        res_out = (acts_reco[outlier_mask, idx] - targets[outlier_mask, idx]) * scale

        # Wrap phi
        if name == "phi":
            res_good = ((res_good / scale + np.pi) % (2 * np.pi) - np.pi) * scale
            res_out = ((res_out / scale + np.pi) % (2 * np.pi) - np.pi) * scale

        combined = np.concatenate([res_good, res_out])
        lo = float(np.percentile(combined, 0.5))
        hi = float(np.percentile(combined, 99.5))
        bins = np.linspace(lo, hi, 100)

        ax.hist(res_good, bins=bins, histtype="step", linewidth=1.5,
                color="steelblue", label=f"Good (σ={np.std(res_good):.4f})", density=True)
        ax.hist(res_out, bins=bins, histtype="step", linewidth=1.5,
                color="red", label=f"Outlier (σ={np.std(res_out):.4f})", density=True)
        ax.set_xlabel(f"Δ{name} [{units[name]}]", fontsize=11)
        ax.set_ylabel("Normalised density")
        ax.set_title(name.upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    axes[5].set_visible(False)

    plt.suptitle("ACTS Residuals: Outlier vs Good Tracks", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "outlier_acts_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_fraction_vs_cuts(
    truth_d0: np.ndarray,
    acts_d0: np.ndarray,
    dm_mask: np.ndarray,
    output_dir: Path,
):
    """Sweep cut thresholds to quantify outlier population."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: outlier fraction vs truth_d0_window (acts threshold fixed at 1mm) ---
    ax = axes[0]
    dm_truth = truth_d0[dm_mask]
    dm_acts = acts_d0[dm_mask]
    windows = np.linspace(0.05, 2.0, 40)
    fracs = []
    counts = []
    for w in windows:
        in_window = np.abs(dm_truth) < w
        if np.sum(in_window) == 0:
            fracs.append(0)
            counts.append(0)
            continue
        outlier = np.abs(dm_acts[in_window]) > 1.0
        fracs.append(float(np.sum(outlier)) / float(np.sum(in_window)))
        counts.append(int(np.sum(outlier)))
    ax.plot(windows, fracs, "o-", color="steelblue", ms=4)
    ax.set_xlabel("|truth d0| window [mm]")
    ax.set_ylabel("Fraction with |ACTS d0| > 1 mm")
    ax.set_title("Outlier fraction vs truth d0 window")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: outlier count vs |acts_d0| threshold ---
    ax = axes[1]
    thresholds = np.linspace(0.2, 5.0, 50)
    near_zero = np.abs(dm_truth) < 0.5
    n_near = int(np.sum(near_zero))
    outlier_counts = [int(np.sum(np.abs(dm_acts[near_zero]) > t)) for t in thresholds]
    ax.plot(thresholds, outlier_counts, "o-", color="red", ms=3)
    ax.set_xlabel("|ACTS d0| threshold [mm]")
    ax.set_ylabel(f"Number of outliers (of {n_near:,} tracks)")
    ax.set_title("|truth d0| < 0.5 mm: outlier count vs threshold")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: ACTS d0 distribution for near-zero truth d0 ---
    ax = axes[2]
    near_acts = dm_acts[near_zero]
    bins = np.linspace(-5, 5, 200)
    ax.hist(near_acts, bins=bins, histtype="step", linewidth=1.5, color="steelblue")
    ax.set_xlabel("ACTS CKF d0 [mm]")
    ax.set_ylabel("Tracks")
    ax.set_title(f"ACTS d0 distribution (|truth d0| < 0.5 mm, {n_near:,} DM tracks)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # Mark ±1 mm
    ax.axvline(-1, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(1, color="red", ls="--", lw=1, alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_dir / "outlier_fraction_vs_cuts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_d0_residual_vs_nhits_pt(
    data: dict,
    dm_mask: np.ndarray,
    output_dir: Path,
):
    """2D scatter: |ACTS d0 residual| vs nhits and pT for DM tracks near d0=0."""
    targets = data["targets"]
    acts_reco = data["acts_reco"]
    pt = data["pt"]
    nhits = data["nhits"]

    truth_d0 = targets[:, 0]
    acts_d0 = acts_reco[:, 0]
    resid_d0 = np.abs(acts_d0 - truth_d0)

    # DM tracks with |truth d0| < 0.5 mm
    near_zero = dm_mask & (np.abs(truth_d0) < 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # |residual| vs nhits
    ax = axes[0]
    h, xe, ye = np.histogram2d(
        nhits[near_zero], resid_d0[near_zero],
        bins=[np.arange(2.5, 21.5, 1), np.linspace(0, 5, 100)],
    )
    h = np.ma.masked_where(h == 0, h)
    pcm = ax.pcolormesh(xe, ye, h.T, cmap="viridis", norm=LogNorm(vmin=1))
    fig.colorbar(pcm, ax=ax, pad=0.02)
    ax.set_xlabel("Number of hits")
    ax.set_ylabel("|ACTS d0 residual| [mm]")
    ax.set_title("DM tracks, |truth d0| < 0.5 mm")

    # |residual| vs pT
    ax = axes[1]
    h2, xe2, ye2 = np.histogram2d(
        pt[near_zero], resid_d0[near_zero],
        bins=[np.linspace(0, 5, 80), np.linspace(0, 5, 100)],
    )
    h2 = np.ma.masked_where(h2 == 0, h2)
    pcm2 = ax.pcolormesh(xe2, ye2, h2.T, cmap="viridis", norm=LogNorm(vmin=1))
    fig.colorbar(pcm2, ax=ax, pad=0.02)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("|ACTS d0 residual| [mm]")
    ax.set_title("DM tracks, |truth d0| < 0.5 mm")

    # pT distribution split by outlier status
    ax = axes[2]
    outlier_cut = resid_d0[near_zero] > 1.0
    pt_good_near = pt[near_zero][~outlier_cut]
    pt_out_near = pt[near_zero][outlier_cut]
    bins_pt = np.linspace(0, 5, 80)
    ax.hist(pt_good_near, bins=bins_pt, histtype="step", linewidth=1.5,
            color="steelblue", label=f"Good ({len(pt_good_near):,})", density=True)
    ax.hist(pt_out_near, bins=bins_pt, histtype="step", linewidth=1.5,
            color="red", label=f"Outlier ({len(pt_out_near):,})", density=True)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Normalised density")
    ax.set_title("pT of good vs outlier (|resid d0| > 1 mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "d0_residual_vs_nhits_pt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acts_d0_vs_truth_profile(
    truth_d0: np.ndarray,
    acts_d0: np.ndarray,
    output_dir: Path,
):
    """Profile plot: mean and std of ACTS d0 in truth d0 bins, plus outlier fraction."""
    bins = np.linspace(-5, 5, 101)
    centers = 0.5 * (bins[:-1] + bins[1:])

    means, stds, fracs, ns = [], [], [], []
    for i in range(len(bins) - 1):
        m = (truth_d0 >= bins[i]) & (truth_d0 < bins[i + 1])
        r = acts_d0[m]
        n = len(r)
        ns.append(n)
        if n > 5:
            means.append(np.mean(r))
            stds.append(np.std(r))
            fracs.append(float(np.sum(np.abs(r - np.mean(truth_d0[m])) > 1.0)) / n)
        else:
            means.append(np.nan)
            stds.append(np.nan)
            fracs.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(centers, means, "o-", color="steelblue", ms=3)
    ax.plot([-5, 5], [-5, 5], "r--", lw=0.8)
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("Mean ACTS d0 [mm]")
    ax.set_title("ACTS d0 profile")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(centers, stds, "o-", color="darkorange", ms=3)
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("Std of ACTS d0 [mm]")
    ax.set_title("ACTS d0 spread vs truth d0")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(centers, fracs, "o-", color="red", ms=3)
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("Fraction with |ACTS d0 - truth d0| > 1 mm")
    ax.set_title("Outlier fraction vs truth d0")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "acts_d0_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_d0_truth_distribution_narrow(
    truth_d0: np.ndarray,
    acts_d0: np.ndarray,
    output_dir: Path,
):
    """Zoom into truth d0 near zero — is there a spike at exactly 0?"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Very fine binning near 0
    ax = axes[0]
    bins = np.linspace(-0.1, 0.1, 201)
    ax.hist(truth_d0, bins=bins, histtype="step", linewidth=1.5, color="steelblue")
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("Tracks")
    ax.set_title("Truth d0 distribution (fine bins)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Check if there's a delta at exactly 0
    ax = axes[1]
    n_exactly_zero = int(np.sum(truth_d0 == 0.0))
    n_near_zero_1um = int(np.sum(np.abs(truth_d0) < 0.001))
    n_near_zero_10um = int(np.sum(np.abs(truth_d0) < 0.01))
    n_total = len(truth_d0)
    text = (
        f"Total tracks: {n_total:,}\n"
        f"Exactly d0 = 0.0: {n_exactly_zero:,} ({100*n_exactly_zero/n_total:.3f}%)\n"
        f"|d0| < 1 μm: {n_near_zero_1um:,} ({100*n_near_zero_1um/n_total:.3f}%)\n"
        f"|d0| < 10 μm: {n_near_zero_10um:,} ({100*n_near_zero_10um/n_total:.3f}%)\n"
    )

    # For those exactly-zero tracks, what does ACTS predict?
    if n_exactly_zero > 0:
        zero_acts = acts_d0[truth_d0 == 0.0]
        text += (
            f"\nFor d0 = 0 tracks:\n"
            f"  ACTS d0 mean: {np.nanmean(zero_acts):.4f} mm\n"
            f"  ACTS d0 std:  {np.nanstd(zero_acts):.4f} mm\n"
            f"  |ACTS d0| > 1mm: {int(np.sum(np.abs(zero_acts) > 1.0)):,}\n"
            f"  |ACTS d0| > 2mm: {int(np.sum(np.abs(zero_acts) > 2.0)):,}\n"
        )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_title("Near-zero truth d0 statistics")
    ax.axis("off")

    # ACTS d0 for exactly-zero truth d0
    ax = axes[2]
    if n_exactly_zero > 0:
        zero_acts = acts_d0[truth_d0 == 0.0]
        zero_acts_valid = zero_acts[np.isfinite(zero_acts)]
        if len(zero_acts_valid) > 0:
            bins = np.linspace(-5, 5, 200)
            ax.hist(zero_acts_valid, bins=bins, histtype="step",
                    linewidth=1.5, color="red")
            ax.set_xlabel("ACTS d0 [mm]")
            ax.set_ylabel("Tracks")
            ax.set_title(f"ACTS d0 for truth d0 = 0.0 ({len(zero_acts_valid):,} tracks)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No tracks with truth d0 = 0.0",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("ACTS d0 for truth d0 = 0.0")

    plt.tight_layout()
    fig.savefig(output_dir / "truth_d0_near_zero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_outlier_location(
    data: dict,
    outlier_mask: np.ndarray,
    good_mask: np.ndarray,
    output_dir: Path,
):
    """Show outliers in eta vs phi and eta vs pT space."""
    targets = data["targets"]
    pt = data["pt"]
    eta = compute_eta(targets[:, PARAM_IDX["theta"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # eta vs phi
    ax = axes[0]
    ax.scatter(eta[good_mask][::10], targets[good_mask, PARAM_IDX["phi"]][::10],
               s=0.5, alpha=0.1, color="steelblue", label="Good (1/10)")
    ax.scatter(eta[outlier_mask], targets[outlier_mask, PARAM_IDX["phi"]],
               s=3, alpha=0.5, color="red", label=f"Outlier ({int(np.sum(outlier_mask)):,})")
    ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
    ax.set_ylabel(r"$\phi_{\mathrm{truth}}$ [rad]")
    ax.set_title("Outlier location in η-φ space")
    ax.legend(fontsize=9, markerscale=5)
    ax.grid(True, alpha=0.3)

    # eta vs pT
    ax = axes[1]
    ax.scatter(eta[good_mask][::10], pt[good_mask][::10],
               s=0.5, alpha=0.1, color="steelblue", label="Good (1/10)")
    ax.scatter(eta[outlier_mask], pt[outlier_mask],
               s=3, alpha=0.5, color="red", label=f"Outlier ({int(np.sum(outlier_mask)):,})")
    ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
    ax.set_ylabel(r"$p_T$ [GeV]")
    ax.set_title(r"Outlier location in $\eta$-$p_T$ space")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=9, markerscale=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "outlier_location_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_d0_residual_conditional(
    data: dict,
    dm_mask: np.ndarray,
    output_dir: Path,
):
    """Show ACTS d0 residual conditioned on truth d0 in narrow slices."""
    targets = data["targets"]
    acts_reco = data["acts_reco"]

    truth_d0 = targets[:, 0]
    acts_d0 = acts_reco[:, 0]
    resid = acts_d0 - truth_d0

    slices = [
        ("|d0| < 0.01 mm", np.abs(truth_d0[dm_mask]) < 0.01),
        ("0.01 < |d0| < 0.1 mm", (np.abs(truth_d0[dm_mask]) >= 0.01) & (np.abs(truth_d0[dm_mask]) < 0.1)),
        ("0.1 < |d0| < 0.5 mm", (np.abs(truth_d0[dm_mask]) >= 0.1) & (np.abs(truth_d0[dm_mask]) < 0.5)),
        ("0.5 < |d0| < 1.0 mm", (np.abs(truth_d0[dm_mask]) >= 0.5) & (np.abs(truth_d0[dm_mask]) < 1.0)),
        ("1.0 < |d0| < 2.5 mm", (np.abs(truth_d0[dm_mask]) >= 1.0) & (np.abs(truth_d0[dm_mask]) < 2.5)),
        ("2.5 < |d0| < 5.0 mm", (np.abs(truth_d0[dm_mask]) >= 2.5) & (np.abs(truth_d0[dm_mask]) < 5.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    bins = np.linspace(-5, 5, 200)

    for i, (label, mask) in enumerate(slices):
        ax = axes[i]
        r = resid[dm_mask][mask]
        n = len(r)
        if n > 0:
            ax.hist(r, bins=bins, histtype="step", linewidth=1.5, color="steelblue")
            ax.set_yscale("log")
            n_big = int(np.sum(np.abs(r) > 1.0))
            ax.set_title(f"{label}\nn={n:,}, |resid|>1mm: {n_big:,} ({100*n_big/n:.2f}%)")
        else:
            ax.set_title(f"{label}\nn=0")
        ax.set_xlabel("ACTS d0 residual [mm]")
        ax.set_ylabel("Tracks")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="gray", ls="--", lw=0.8)

    plt.suptitle("ACTS d0 residual in truth d0 slices (DM tracks)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "d0_residual_conditional.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main report
# ============================================================================

def write_report(
    data: dict,
    dm_mask: np.ndarray,
    output_dir: Path,
):
    """Write a comprehensive text report."""
    targets = data["targets"]
    acts_reco = data["acts_reco"]
    pt = data["pt"]
    nhits = data["nhits"]
    vp = data["vertex_primary"]

    truth_d0 = targets[:, 0]
    acts_d0 = acts_reco[:, 0]
    has_acts = ~np.any(np.isnan(acts_reco), axis=1)

    lines = []
    lines.append("=" * 80)
    lines.append("ACTS CKF d0 OUTLIER INVESTIGATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Basic counts
    n_total = len(targets)
    n_has_acts = int(np.sum(has_acts))
    n_dm = int(np.sum(dm_mask))
    lines.append(f"Total tracks loaded:           {n_total:>12,}")
    lines.append(f"Tracks with ACTS match:        {n_has_acts:>12,}")
    lines.append(f"Double-matched (DM) tracks:    {n_dm:>12,}")
    lines.append("")

    # Focus on DM tracks
    dm_truth_d0 = truth_d0[dm_mask]
    dm_acts_d0 = acts_d0[dm_mask]
    dm_resid_d0 = dm_acts_d0 - dm_truth_d0

    lines.append("-" * 60)
    lines.append("DOUBLE-MATCHED TRACKS: d0 statistics")
    lines.append("-" * 60)
    lines.append(f"  Truth d0 mean:   {np.mean(dm_truth_d0):>12.6f} mm")
    lines.append(f"  Truth d0 std:    {np.std(dm_truth_d0):>12.6f} mm")
    lines.append(f"  ACTS d0 mean:    {np.mean(dm_acts_d0):>12.6f} mm")
    lines.append(f"  ACTS d0 std:     {np.std(dm_acts_d0):>12.6f} mm")
    lines.append(f"  Residual mean:   {np.mean(dm_resid_d0):>12.6f} mm")
    lines.append(f"  Residual std:    {np.std(dm_resid_d0):>12.6f} mm")
    lines.append("")

    # Outlier quantification at different thresholds
    lines.append("-" * 60)
    lines.append("OUTLIER QUANTIFICATION (DM tracks)")
    lines.append("-" * 60)
    lines.append("")
    lines.append("Definition: |truth d0| < window AND |ACTS d0 residual| > threshold")
    lines.append("")

    for window in [0.1, 0.25, 0.5, 1.0]:
        in_window = np.abs(dm_truth_d0) < window
        n_in = int(np.sum(in_window))
        lines.append(f"  |truth d0| < {window:.2f} mm  ({n_in:,} tracks):")
        for thresh in [0.5, 1.0, 2.0, 3.0]:
            outlier = np.abs(dm_resid_d0[in_window]) > thresh
            n_out = int(np.sum(outlier))
            pct = 100 * n_out / n_in if n_in > 0 else 0
            lines.append(f"    |residual| > {thresh:.1f} mm: {n_out:>8,}  ({pct:.3f}%)")
        lines.append("")

    # Tracks with exactly d0 = 0
    n_exactly_zero = int(np.sum(dm_truth_d0 == 0.0))
    lines.append("-" * 60)
    lines.append("TRACKS WITH EXACTLY truth d0 = 0.0 (DM)")
    lines.append("-" * 60)
    lines.append(f"  Count: {n_exactly_zero:,} of {n_dm:,} DM tracks ({100*n_exactly_zero/max(n_dm,1):.3f}%)")
    if n_exactly_zero > 0:
        zero_acts = dm_acts_d0[dm_truth_d0 == 0.0]
        lines.append(f"  ACTS d0 for these tracks:")
        lines.append(f"    mean:  {np.nanmean(zero_acts):.6f} mm")
        lines.append(f"    std:   {np.nanstd(zero_acts):.6f} mm")
        lines.append(f"    |d0| > 0.5mm: {int(np.sum(np.abs(zero_acts) > 0.5)):,}")
        lines.append(f"    |d0| > 1.0mm: {int(np.sum(np.abs(zero_acts) > 1.0)):,}")
        lines.append(f"    |d0| > 2.0mm: {int(np.sum(np.abs(zero_acts) > 2.0)):,}")
    lines.append("")

    # Cross-check with other parameters
    lines.append("-" * 60)
    lines.append("OUTLIER CHARACTERISATION (|truth d0| < 0.5, |ACTS resid| > 1 mm, DM)")
    lines.append("-" * 60)
    outlier = (np.abs(dm_truth_d0) < 0.5) & (np.abs(dm_resid_d0) > 1.0)
    good = (np.abs(dm_truth_d0) < 0.5) & (np.abs(dm_resid_d0) <= 1.0)
    n_out = int(np.sum(outlier))
    n_good = int(np.sum(good))
    dm_pt = pt[dm_mask]
    dm_nhits = nhits[dm_mask]
    dm_vp = vp[dm_mask]

    if n_out > 0:
        lines.append(f"  Outlier count: {n_out:,}")
        lines.append(f"  Good count:    {n_good:,}")
        lines.append(f"  Outlier fraction: {100*n_out/(n_out+n_good):.3f}%")
        lines.append("")
        lines.append("  Kinematic comparison (mean ± std):")
        for label, arr in [("pT [GeV]", dm_pt), ("nhits", dm_nhits)]:
            lines.append(f"    {label:>12s}: outlier={np.mean(arr[outlier]):.3f}±{np.std(arr[outlier]):.3f}"
                         f"  good={np.mean(arr[good]):.3f}±{np.std(arr[good]):.3f}")

        # Fraction primary vertex
        out_vp = float(np.mean(dm_vp[outlier] == 1))
        good_vp = float(np.mean(dm_vp[good] == 1))
        lines.append(f"    {'vp=1 frac':>12s}: outlier={out_vp:.3f}  good={good_vp:.3f}")

        # Other parameter residuals for outliers
        lines.append("")
        lines.append("  Other ACTS residuals for outlier tracks:")
        for name in PARAMS:
            idx = PARAM_IDX[name]
            scale = {"d0": 1.0, "z0": 1.0, "phi": 1e3, "theta": 1e3, "qop": 1.0}[name]
            unit = {"d0": "mm", "z0": "mm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV"}[name]
            res_out = (acts_reco[dm_mask][outlier, idx] - targets[dm_mask][outlier, idx]) * scale
            res_good = (acts_reco[dm_mask][good, idx] - targets[dm_mask][good, idx]) * scale
            if name == "phi":
                res_out = ((res_out / scale + np.pi) % (2 * np.pi) - np.pi) * scale
                res_good = ((res_good / scale + np.pi) % (2 * np.pi) - np.pi) * scale
            lines.append(f"    {name:>5s}: outlier σ={np.std(res_out):.4f} {unit}  "
                         f"good σ={np.std(res_good):.4f} {unit}")
    lines.append("")

    # Extrapolation to full dataset
    lines.append("-" * 60)
    lines.append("EXTRAPOLATION TO FULL DATASET")
    lines.append("-" * 60)
    n_shards_loaded = len(np.unique(data["shard_idx"]))
    total_shards = 1000
    scale_factor = total_shards / n_shards_loaded

    # Count outliers for ALL tracks (not just DM) — these affect training
    all_has_acts = ~np.any(np.isnan(acts_reco), axis=1)
    all_resid_d0 = acts_d0 - truth_d0
    all_near_zero = np.abs(truth_d0) < 0.5

    for context, mask_base in [("DM tracks", dm_mask), ("All tracks with ACTS", all_has_acts)]:
        m_truth = truth_d0[mask_base]
        m_resid = (acts_d0[mask_base] - truth_d0[mask_base])
        m_near = np.abs(m_truth) < 0.5
        n_loaded = int(np.sum(m_near))
        n_out_loaded = int(np.sum((m_near) & (np.abs(m_resid) > 1.0)))
        lines.append(f"  {context} (|truth d0|<0.5, |resid|>1mm):")
        lines.append(f"    In loaded shards ({n_shards_loaded}): {n_out_loaded:,}")
        lines.append(f"    Estimated in full dataset (~{total_shards} shards): ~{int(n_out_loaded*scale_factor):,}")
        lines.append("")

    # Suggested cuts
    lines.append("-" * 60)
    lines.append("SUGGESTED PREPROCESSING CUTS")
    lines.append("-" * 60)
    lines.append("")
    lines.append("Option 1 (conservative): Filter on ACTS d0 residual")
    lines.append("  When a track has an ACTS match (DM or not), discard if")
    lines.append("  |ACTS_d0 - truth_d0| > 2.0 mm")
    for thresh in [1.0, 2.0, 3.0]:
        removed_dm = int(np.sum(np.abs(dm_resid_d0) > thresh))
        lines.append(f"  At threshold {thresh:.1f} mm: removes {removed_dm:,} DM tracks "
                     f"({100*removed_dm/n_dm:.3f}%)")
    lines.append("")
    lines.append("Option 2 (targeted): Only remove if truth d0 ≈ 0")
    lines.append("  Discard if |truth_d0| < 0.1 mm AND |ACTS_d0 - truth_d0| > 1.0 mm")
    near_01 = np.abs(dm_truth_d0) < 0.1
    removed_targeted = int(np.sum(near_01 & (np.abs(dm_resid_d0) > 1.0)))
    lines.append(f"  Removes: {removed_targeted:,} DM tracks ({100*removed_targeted/n_dm:.3f}%)")
    lines.append("")
    lines.append("Option 3 (simulation-level): Flag truth d0 == 0.0 exactly")
    lines.append("  These may be simulation artifacts (particle not properly propagated)")
    lines.append(f"  Removes: {n_exactly_zero:,} DM tracks ({100*n_exactly_zero/n_dm:.3f}%)")

    report_text = "\n".join(lines)
    report_path = output_dir / "outlier_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    return report_text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Investigate ACTS CKF d0 outliers in preprocessed data"
    )
    parser.add_argument("--data-dir", type=str,
                        default="/scratch/colliderml/p200_loose_finetune")
    parser.add_argument("--output-dir", type=str,
                        default="/shared/tracking/hepattn_muon/analysis/acts_d0_outlier_investigation")
    parser.add_argument("--split", type=str, default="all",
                        help="Which split to load: train, val, test, or all")
    parser.add_argument("--n-shards", type=int, default=50,
                        help="Number of shards to load (-1 for all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ({args.n_shards} shards, split={args.split})...")
    data = load_dataset(data_dir, split=args.split, n_shards=args.n_shards)

    targets = data["targets"]
    acts_reco = data["acts_reco"]
    truth_d0 = targets[:, 0]
    acts_d0 = acts_reco[:, 0]

    # Masks
    has_acts = ~np.any(np.isnan(acts_reco), axis=1)
    dm_mask = data["acts_dm"]

    print(f"Loaded {len(targets):,} tracks, {int(np.sum(has_acts)):,} with ACTS, "
          f"{int(np.sum(dm_mask)):,} double-matched")

    # Define outlier mask: DM tracks with |truth d0| < 0.5 and |ACTS d0 resid| > 1mm
    outlier_mask = dm_mask & (np.abs(truth_d0) < 0.5) & (np.abs(acts_d0 - truth_d0) > 1.0)
    good_mask = dm_mask & (np.abs(truth_d0) < 0.5) & (np.abs(acts_d0 - truth_d0) <= 1.0)

    print(f"Outlier tracks (DM, |truth d0|<0.5, |resid|>1mm): {int(np.sum(outlier_mask)):,}")
    print(f"Good tracks    (DM, |truth d0|<0.5, |resid|≤1mm): {int(np.sum(good_mask)):,}")

    # Generate all plots
    print("\n--- Generating plots ---")

    print("  1/8  Annotated heatmap...")
    plot_heatmap_annotated(truth_d0[dm_mask], acts_d0[dm_mask], outlier_mask[dm_mask],
                           output_dir, dm_label=" (DM tracks)")

    print("  2/8  Outlier kinematics comparison...")
    plot_outlier_kinematics(data, outlier_mask, good_mask, output_dir)

    print("  3/8  ACTS residuals (all params) for outlier vs good...")
    plot_acts_residuals_all_params(data, outlier_mask, good_mask, output_dir)

    print("  4/8  Outlier fraction vs cut thresholds...")
    plot_outlier_fraction_vs_cuts(truth_d0, acts_d0, dm_mask, output_dir)

    print("  5/8  d0 residual vs nhits and pT...")
    plot_d0_residual_vs_nhits_pt(data, dm_mask, output_dir)

    print("  6/8  ACTS d0 profile vs truth d0...")
    plot_acts_d0_vs_truth_profile(truth_d0[dm_mask], acts_d0[dm_mask], output_dir)

    print("  7/8  Truth d0 near-zero analysis...")
    plot_d0_truth_distribution_narrow(truth_d0[dm_mask], acts_d0[dm_mask], output_dir)

    print("  8/8  Outlier location in η-φ and η-pT space...")
    plot_2d_outlier_location(data, outlier_mask, good_mask, output_dir)

    print("  Extra: Conditional residual distributions...")
    plot_d0_residual_conditional(data, dm_mask, output_dir)

    # Write comprehensive report
    print("\n--- Writing report ---")
    write_report(data, dm_mask, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
