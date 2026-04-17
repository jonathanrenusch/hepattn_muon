#!/usr/bin/env python3
"""Cross-analysis: compare SSM and CKF d0 outlier populations.

Loads SSM test predictions (HDF5) alongside ACTS reco from the same test shards,
then characterises overlap and distinct failure modes.

Usage::

    python -m hepattn.experiments.colliderml_regr.scripts.investigate_ssm_vs_ckf_d0_outliers \
        --ssm-predictions /shared/tracking/hepattn_muon/src/logs/comet_offline/00146ee2b502494a9b0637934d951e33/test_predictions.h5 \
        --data-dir /scratch/colliderml/p200_loose_finetune \
        --output-dir /shared/tracking/hepattn_muon/analysis/ssm_vs_ckf_d0_outliers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm


PARAMS = ["d0", "z0", "phi", "theta", "qop"]
PARAM_IDX = {n: i for i, n in enumerate(PARAMS)}


def compute_eta(theta: np.ndarray) -> np.ndarray:
    return -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))


def load_ssm_predictions(path: Path) -> dict:
    """Load SSM preds and targets from HDF5."""
    data = {"preds": {}, "targets": {}}
    with h5py.File(path, "r") as f:
        for group in ("preds", "targets"):
            for name in f[group]:
                data[group][name] = f[group][name][:]
    return data


def load_acts_test(data_dir: Path) -> dict:
    """Load ACTS reco, DM mask, metadata from test shards."""
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    test_shards = sorted(splits["test"])

    accum = {k: [] for k in ["acts_reco", "acts_dm", "pt", "nhits", "vertex_primary"]}
    for idx in tqdm(test_shards, desc="Loading ACTS test data", file=sys.stderr):
        sel_dir = data_dir / f"shard_{idx:04d}" / "selected_tracks"
        accum["acts_reco"].append(np.load(sel_dir / "acts_reco.npy"))
        accum["acts_dm"].append(np.load(sel_dir / "acts_dm_mask.npy"))
        meta = np.load(sel_dir / "track_meta.npy")
        accum["pt"].append(meta[:, 0])
        accum["vertex_primary"].append(meta[:, 1])
        offsets = np.load(sel_dir / "track_hit_offsets.npy")
        accum["nhits"].append(np.diff(offsets).astype(np.int32))

    return {k: np.concatenate(v, axis=0) for k, v in accum.items()}


def main():
    parser = argparse.ArgumentParser(description="SSM vs CKF d0 outlier cross-analysis")
    parser.add_argument("--ssm-predictions", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/scratch/colliderml/p200_loose_finetune")
    parser.add_argument("--output-dir", type=str,
                        default="/shared/tracking/hepattn_muon/analysis/ssm_vs_ckf_d0_outliers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SSM predictions...")
    ssm = load_ssm_predictions(Path(args.ssm_predictions))
    n_ssm = len(ssm["targets"]["d0"])

    print("Loading ACTS test data...")
    acts = load_acts_test(Path(args.data_dir))
    n_acts = len(acts["acts_reco"])

    assert n_ssm == n_acts, f"Track count mismatch: SSM={n_ssm}, ACTS={n_acts}"
    print(f"Aligned {n_ssm:,} test tracks")

    # Extract arrays
    truth_d0 = ssm["targets"]["d0"]
    ssm_d0 = ssm["preds"]["d0"]
    acts_d0 = acts["acts_reco"][:, 0]
    dm_mask = acts["acts_dm"]
    has_acts = ~np.isnan(acts_d0)
    pt = acts["pt"]
    nhits = acts["nhits"]
    eta = compute_eta(ssm["targets"]["theta"])

    ssm_resid_d0 = ssm_d0 - truth_d0
    acts_resid_d0 = acts_d0 - truth_d0  # NaN where no match

    # Define outlier masks (|truth d0| < 0.5 mm, |resid| > 1 mm)
    near_zero = np.abs(truth_d0) < 0.5
    ssm_outlier_all = near_zero & (np.abs(ssm_resid_d0) > 1.0)
    ckf_outlier_dm = near_zero & dm_mask & (np.abs(acts_resid_d0) > 1.0)
    ckf_outlier_any = near_zero & has_acts & (np.abs(acts_resid_d0) > 1.0)

    # Tracks where both fail (must have ACTS match)
    both_fail = near_zero & has_acts & (np.abs(ssm_resid_d0) > 1.0) & (np.abs(acts_resid_d0) > 1.0)
    ssm_only_fail = near_zero & has_acts & (np.abs(ssm_resid_d0) > 1.0) & (np.abs(acts_resid_d0) <= 1.0)
    ckf_only_fail = near_zero & has_acts & (np.abs(ssm_resid_d0) <= 1.0) & (np.abs(acts_resid_d0) > 1.0)

    # Also check DM subset
    both_fail_dm = near_zero & dm_mask & (np.abs(ssm_resid_d0) > 1.0) & (np.abs(acts_resid_d0) > 1.0)
    ssm_only_fail_dm = near_zero & dm_mask & (np.abs(ssm_resid_d0) > 1.0) & (np.abs(acts_resid_d0) <= 1.0)
    ckf_only_fail_dm = near_zero & dm_mask & (np.abs(ssm_resid_d0) <= 1.0) & (np.abs(acts_resid_d0) > 1.0)

    # ======================================================================
    # Report
    # ======================================================================
    lines = []
    lines.append("=" * 80)
    lines.append("SSM vs CKF d0 OUTLIER CROSS-ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"\nTotal test tracks: {n_ssm:,}")
    lines.append(f"Tracks with ACTS match: {int(np.sum(has_acts)):,}")
    lines.append(f"Double-matched tracks: {int(np.sum(dm_mask)):,}")
    lines.append(f"|truth d0| < 0.5 mm: {int(np.sum(near_zero)):,}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("OUTLIER COUNTS (|truth d0| < 0.5 mm, |residual| > 1 mm)")
    lines.append("-" * 60)
    lines.append(f"  SSM outliers (all tracks):       {int(np.sum(ssm_outlier_all)):>10,}")
    lines.append(f"  CKF outliers (DM tracks):        {int(np.sum(ckf_outlier_dm)):>10,}")
    lines.append(f"  CKF outliers (any ACTS match):   {int(np.sum(ckf_outlier_any)):>10,}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("OVERLAP ANALYSIS (tracks with ACTS match, |truth d0| < 0.5)")
    lines.append("-" * 60)
    n_has_acts_near = int(np.sum(near_zero & has_acts))
    lines.append(f"  Base population: {n_has_acts_near:,} tracks")
    lines.append(f"  Both SSM + CKF fail:    {int(np.sum(both_fail)):>10,}  ({100*np.sum(both_fail)/n_has_acts_near:.3f}%)")
    lines.append(f"  SSM only fails:         {int(np.sum(ssm_only_fail)):>10,}  ({100*np.sum(ssm_only_fail)/n_has_acts_near:.3f}%)")
    lines.append(f"  CKF only fails:         {int(np.sum(ckf_only_fail)):>10,}  ({100*np.sum(ckf_only_fail)/n_has_acts_near:.3f}%)")
    lines.append(f"  Neither fails:          {int(np.sum(near_zero & has_acts & (np.abs(ssm_resid_d0) <= 1.0) & (np.abs(acts_resid_d0) <= 1.0))):>10,}")
    lines.append("")

    n_dm_near = int(np.sum(near_zero & dm_mask))
    lines.append(f"  DM subset: {n_dm_near:,} tracks")
    lines.append(f"  Both fail (DM):         {int(np.sum(both_fail_dm)):>10,}  ({100*np.sum(both_fail_dm)/max(n_dm_near,1):.3f}%)")
    lines.append(f"  SSM only fails (DM):    {int(np.sum(ssm_only_fail_dm)):>10,}  ({100*np.sum(ssm_only_fail_dm)/max(n_dm_near,1):.3f}%)")
    lines.append(f"  CKF only fails (DM):    {int(np.sum(ckf_only_fail_dm)):>10,}  ({100*np.sum(ckf_only_fail_dm)/max(n_dm_near,1):.3f}%)")
    lines.append("")

    # For SSM outliers — which tracks does it fail on?
    lines.append("-" * 60)
    lines.append("SSM-ONLY OUTLIER CHARACTERISATION")
    lines.append("-" * 60)
    for label, mask in [("SSM outlier (all, near-zero d0)", ssm_outlier_all),
                        ("SSM outlier + has ACTS", ssm_outlier_all & has_acts),
                        ("SSM outlier + no ACTS match", ssm_outlier_all & ~has_acts),
                        ("SSM outlier + DM", ssm_outlier_all & dm_mask)]:
        n = int(np.sum(mask))
        if n == 0:
            lines.append(f"  {label}: 0 tracks")
            continue
        lines.append(f"  {label}: {n:,} tracks")
        lines.append(f"    pT mean: {np.mean(pt[mask]):.3f} GeV, median: {np.median(pt[mask]):.3f} GeV")
        lines.append(f"    nhits mean: {np.mean(nhits[mask]):.1f}")
        lines.append(f"    |eta| mean: {np.mean(np.abs(eta[mask])):.2f}")
        frac_low_pt = np.sum(pt[mask] < 0.5) / n
        frac_fwd = np.sum(np.abs(eta[mask]) > 2.0) / n
        lines.append(f"    frac pT < 0.5 GeV: {frac_low_pt:.3f}")
        lines.append(f"    frac |eta| > 2: {frac_fwd:.3f}")
        lines.append("")

    # Quantify the d0 → 0 collapse (horizontal stripe in SSM heatmap)
    lines.append("-" * 60)
    lines.append("SSM d0 → 0 COLLAPSE (horizontal stripe)")
    lines.append("-" * 60)
    for d0_thresh in [0.5, 1.0, 2.0, 3.0]:
        truth_far = np.abs(truth_d0) > d0_thresh
        ssm_near_zero_pred = np.abs(ssm_d0) < 0.1
        collapse = truth_far & ssm_near_zero_pred
        n_far = int(np.sum(truth_far))
        n_collapse = int(np.sum(collapse))
        lines.append(f"  |truth d0| > {d0_thresh:.1f} mm, |SSM d0| < 0.1 mm: "
                     f"{n_collapse:,} of {n_far:,} ({100*n_collapse/max(n_far,1):.2f}%)")
    lines.append("")

    # Impact on training: which tracks contribute most gradient noise?
    lines.append("-" * 60)
    lines.append("GRADIENT NOISE BUDGET — |SSM d0 residual| breakdown")
    lines.append("-" * 60)
    abs_resid = np.abs(ssm_resid_d0)
    total_abs_resid = float(np.sum(abs_resid))
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, np.inf)]:
        in_range = (abs_resid >= lo) & (abs_resid < hi)
        n_in = int(np.sum(in_range))
        contrib = float(np.sum(abs_resid[in_range]))
        lines.append(f"  |resid| in [{lo:.1f}, {hi:.1f}): {n_in:>10,} tracks ({100*n_in/n_ssm:.2f}%), "
                     f"abs resid sum: {contrib:.0f} ({100*contrib/total_abs_resid:.1f}% of total)")
    lines.append("")

    # Suggested cuts focusing on training data quality
    lines.append("-" * 60)
    lines.append("IMPACT OF PROPOSED CUTS ON SSM d0 PERFORMANCE")
    lines.append("-" * 60)
    # If we remove tracks that CKF also fails on
    for cut_label, cut_mask in [
        ("Remove DM tracks with |CKF d0 resid| > 2mm", ~(dm_mask & (np.abs(acts_resid_d0) > 2.0))),
        ("Remove tracks with |CKF d0 resid| > 2mm (any ACTS)", ~(has_acts & (np.abs(acts_resid_d0) > 2.0))),
        ("Remove low-pT pile-up (pT < 0.5 & non-hard-scatter)", ~((pt < 0.5) & (acts["vertex_primary"] != 1))),
        ("Require pT > 0.5 GeV", pt >= 0.5),
        ("Require nhits >= 6", nhits >= 6),
        ("Require pT > 0.5 AND nhits >= 6", (pt >= 0.5) & (nhits >= 6)),
    ]:
        n_kept = int(np.sum(cut_mask))
        ssm_std_before = float(np.std(ssm_resid_d0))
        ssm_std_after = float(np.std(ssm_resid_d0[cut_mask]))
        ssm_out_before = int(np.sum(np.abs(ssm_resid_d0) > 1.0))
        ssm_out_after = int(np.sum(np.abs(ssm_resid_d0[cut_mask]) > 1.0))
        lines.append(f"  {cut_label}:")
        lines.append(f"    Tracks: {n_ssm:,} -> {n_kept:,} ({100*(n_ssm-n_kept)/n_ssm:.2f}% removed)")
        lines.append(f"    SSM d0 σ(resid): {ssm_std_before:.4f} -> {ssm_std_after:.4f} mm")
        lines.append(f"    SSM d0 outliers (|r|>1mm): {ssm_out_before:,} -> {ssm_out_after:,}")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "ssm_vs_ckf_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)

    # ======================================================================
    # Plots
    # ======================================================================
    print("\n--- Generating plots ---")

    # 1. Venn-style overlap visualization
    print("  1/6  Overlap summary...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Venn as stacked bar for ACTS-matched tracks near d0=0
    ax = axes[0]
    categories = ["Both fail", "SSM only", "CKF only", "Neither"]
    counts = [int(np.sum(both_fail)), int(np.sum(ssm_only_fail)),
              int(np.sum(ckf_only_fail)),
              int(np.sum(near_zero & has_acts & (np.abs(ssm_resid_d0) <= 1.0) & (np.abs(acts_resid_d0) <= 1.0)))]
    colors = ["red", "steelblue", "darkorange", "lightgray"]
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors,
                                       autopct=lambda p: f"{int(p*sum(counts)/100):,}\n({p:.1f}%)",
                                       textprops={"fontsize": 9})
    ax.set_title(f"d0 outlier overlap (ACTS-matched, |truth d0| < 0.5)\n{sum(counts):,} tracks")

    # Outlier rate vs pT
    ax = axes[1]
    pt_bins = np.linspace(0.2, 5.0, 50)
    ssm_rate, ckf_rate = [], []
    for j in range(len(pt_bins) - 1):
        m = near_zero & has_acts & (pt >= pt_bins[j]) & (pt < pt_bins[j + 1])
        n_bin = int(np.sum(m))
        if n_bin > 50:
            ssm_rate.append(float(np.sum(np.abs(ssm_resid_d0[m]) > 1.0)) / n_bin)
            ckf_rate.append(float(np.sum(np.abs(acts_resid_d0[m]) > 1.0)) / n_bin)
        else:
            ssm_rate.append(np.nan)
            ckf_rate.append(np.nan)
    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    ax.plot(bin_centers, ssm_rate, "o-", color="steelblue", ms=3, label="SSM |resid|>1mm")
    ax.plot(bin_centers, ckf_rate, "s-", color="darkorange", ms=3, label="CKF |resid|>1mm")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Outlier fraction")
    ax.set_title("d0 outlier rate vs pT\n(ACTS-matched, |truth d0| < 0.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Outlier rate vs eta
    ax = axes[2]
    eta_bins = np.linspace(-3, 3, 31)
    ssm_rate_eta, ckf_rate_eta = [], []
    for j in range(len(eta_bins) - 1):
        m = near_zero & has_acts & (eta >= eta_bins[j]) & (eta < eta_bins[j + 1])
        n_bin = int(np.sum(m))
        if n_bin > 50:
            ssm_rate_eta.append(float(np.sum(np.abs(ssm_resid_d0[m]) > 1.0)) / n_bin)
            ckf_rate_eta.append(float(np.sum(np.abs(acts_resid_d0[m]) > 1.0)) / n_bin)
        else:
            ssm_rate_eta.append(np.nan)
            ckf_rate_eta.append(np.nan)
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    ax.plot(eta_centers, ssm_rate_eta, "o-", color="steelblue", ms=3, label="SSM |resid|>1mm")
    ax.plot(eta_centers, ckf_rate_eta, "s-", color="darkorange", ms=3, label="CKF |resid|>1mm")
    ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
    ax.set_ylabel("Outlier fraction")
    ax.set_title("d0 outlier rate vs η\n(ACTS-matched, |truth d0| < 0.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "overlap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Side-by-side heatmaps: SSM vs CKF
    print("  2/6  Side-by-side heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    bins = np.linspace(-5, 5, 201)

    for ax, pred_d0, label, cmap in [
        (axes[0], ssm_d0, "SSM", "viridis"),
        (axes[1], acts_d0[has_acts], "ACTS CKF", "viridis"),
    ]:
        td = truth_d0 if label == "SSM" else truth_d0[has_acts]
        h, xe, ye = np.histogram2d(td, pred_d0, bins=[bins, bins])
        h = np.ma.masked_where(h == 0, h)
        pcm = ax.pcolormesh(xe, ye, h.T, cmap=cmap, norm=LogNorm(vmin=1))
        fig.colorbar(pcm, ax=ax, pad=0.02, aspect=30)
        ax.plot([-5, 5], [-5, 5], "r--", lw=0.8)
        ax.set_xlabel("Truth d0 [mm]")
        ax.set_ylabel(f"{label} d0 [mm]")
        ax.set_title(f"{label} d0 vs Truth ({len(td):,} tracks)")
        ax.set_aspect("equal")

    # Panel 3: SSM residual vs CKF residual (scatter)
    ax = axes[2]
    dm_near = dm_mask & near_zero
    h, xe, ye = np.histogram2d(
        ssm_resid_d0[dm_near], acts_resid_d0[dm_near],
        bins=[np.linspace(-3, 3, 201), np.linspace(-3, 3, 201)],
    )
    h = np.ma.masked_where(h == 0, h)
    pcm = ax.pcolormesh(xe, ye, h.T, cmap="viridis", norm=LogNorm(vmin=1))
    fig.colorbar(pcm, ax=ax, pad=0.02, aspect=30)
    ax.plot([-3, 3], [-3, 3], "r--", lw=0.8)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.axvline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("SSM d0 residual [mm]")
    ax.set_ylabel("CKF d0 residual [mm]")
    ax.set_title(f"SSM vs CKF d0 residual (DM, |truth d0|<0.5)\n{int(np.sum(dm_near)):,} tracks")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_dir / "heatmaps_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. SSM quantile spread analysis (do outlier tracks have wide quantile spread?)
    print("  3/6  SSM d0 residual distribution by category...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins_resid = np.linspace(-5, 5, 200)

    ax = axes[0]
    for mask, label, color in [
        (both_fail, "Both fail", "red"),
        (ssm_only_fail, "SSM only", "steelblue"),
        (ckf_only_fail, "CKF only", "darkorange"),
    ]:
        if np.sum(mask) > 0:
            ax.hist(ssm_resid_d0[mask], bins=bins_resid, histtype="step",
                    linewidth=1.5, color=color, label=f"{label} ({int(np.sum(mask)):,})")
    ax.set_xlabel("SSM d0 residual [mm]")
    ax.set_ylabel("Tracks")
    ax.set_title("SSM residual by failure category")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mask, label, color in [
        (both_fail, "Both fail", "red"),
        (ssm_only_fail, "SSM only", "steelblue"),
        (ckf_only_fail, "CKF only", "darkorange"),
    ]:
        if np.sum(mask) > 0:
            ax.hist(acts_resid_d0[mask], bins=bins_resid, histtype="step",
                    linewidth=1.5, color=color, label=f"{label} ({int(np.sum(mask)):,})")
    ax.set_xlabel("CKF d0 residual [mm]")
    ax.set_ylabel("Tracks")
    ax.set_title("CKF residual by failure category")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: CDF of |SSM d0 residual| for all tracks
    ax = axes[2]
    sorted_abs = np.sort(np.abs(ssm_resid_d0))
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    # Subsample for plotting
    step = max(1, len(sorted_abs) // 10000)
    ax.plot(sorted_abs[::step], cdf[::step], color="steelblue", linewidth=1.5)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="|resid| = 1mm")
    ax.axvline(2.0, color="red", ls=":", lw=1, label="|resid| = 2mm")
    ax.set_xlabel("|SSM d0 residual| [mm]")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF of |SSM d0 residual| ({n_ssm:,} tracks)")
    ax.set_xlim(0, 5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "residual_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Kinematic distributions by failure category
    print("  4/6  Kinematic distributions by category...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    categories = [
        (both_fail, "Both fail", "red"),
        (ssm_only_fail, "SSM only fail", "steelblue"),
        (ckf_only_fail, "CKF only fail", "darkorange"),
    ]

    def _compare_hist(ax, vals_dict, bins, xlabel, logy=False):
        for mask, label, color in vals_dict:
            if np.sum(mask) > 0:
                ax.hist(vals_dict[0][0].__class__(vals_dict), bins=bins)

    for (mask, label, color) in categories:
        n = int(np.sum(mask))
        if n == 0:
            continue
        axes[0].hist(pt[mask], bins=np.linspace(0, 5, 60), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)
        axes[1].hist(eta[mask], bins=np.linspace(-3, 3, 60), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)
        axes[2].hist(nhits[mask], bins=np.arange(2.5, 21.5, 1), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)
        axes[3].hist(truth_d0[mask], bins=np.linspace(-0.5, 0.5, 60), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)
        axes[4].hist(ssm["targets"]["z0"][mask], bins=np.linspace(-200, 200, 60), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)
        axes[5].hist(ssm["targets"]["qop"][mask], bins=np.linspace(-2, 2, 60), histtype="step",
                     linewidth=1.5, color=color, label=f"{label} ({n:,})", density=True)

    xlabels = [r"$p_T$ [GeV]", r"$\eta$", "nhits", "truth d0 [mm]", "truth z0 [mm]", "truth q/p [1/GeV]"]
    for i, xlabel in enumerate(xlabels):
        axes[i].set_xlabel(xlabel, fontsize=11)
        axes[i].set_ylabel("Normalised density")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    plt.suptitle("Kinematic Distributions by Failure Category\n(ACTS-matched, |truth d0| < 0.5 mm)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "kinematics_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. SSM d0 performance improvement with pT/nhits cuts
    print("  5/6  Performance vs cuts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # σ(SSM d0 resid) vs pT cut
    ax = axes[0]
    pt_cuts = np.linspace(0.2, 2.0, 30)
    stds = [float(np.std(ssm_resid_d0[pt >= c])) for c in pt_cuts]
    n_kept = [int(np.sum(pt >= c)) for c in pt_cuts]
    ax.plot(pt_cuts, stds, "o-", color="steelblue", ms=4)
    ax.set_xlabel("Minimum pT cut [GeV]")
    ax.set_ylabel("σ(SSM d0 residual) [mm]")
    ax.set_title("SSM d0 precision vs pT cut")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(pt_cuts, [n / 1e6 for n in n_kept], "s--", color="gray", ms=3, alpha=0.5)
    ax2.set_ylabel("Tracks kept [M]", color="gray")

    # σ(SSM d0 resid) vs nhits cut
    ax = axes[1]
    nhit_cuts = np.arange(3, 15)
    stds_nh = [float(np.std(ssm_resid_d0[nhits >= c])) for c in nhit_cuts]
    n_kept_nh = [int(np.sum(nhits >= c)) for c in nhit_cuts]
    ax.plot(nhit_cuts, stds_nh, "o-", color="steelblue", ms=4)
    ax.set_xlabel("Minimum nhits cut")
    ax.set_ylabel("σ(SSM d0 residual) [mm]")
    ax.set_title("SSM d0 precision vs nhits cut")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(nhit_cuts, [n / 1e6 for n in n_kept_nh], "s--", color="gray", ms=3, alpha=0.5)
    ax2.set_ylabel("Tracks kept [M]", color="gray")

    # Combined: fraction of "gradient budget" from outlier tracks
    ax = axes[2]
    abs_r = np.abs(ssm_resid_d0)
    total = float(np.sum(abs_r ** 2))  # proportional to gradient
    thresholds = np.linspace(0, 5, 100)
    outlier_frac_gradient = [float(np.sum(abs_r[abs_r > t] ** 2)) / total for t in thresholds]
    outlier_frac_count = [float(np.sum(abs_r > t)) / n_ssm for t in thresholds]
    ax.plot(thresholds, outlier_frac_gradient, "-", color="red", linewidth=2, label="Fraction of Σ(resid²)")
    ax.plot(thresholds, outlier_frac_count, "--", color="steelblue", linewidth=2, label="Fraction of tracks")
    ax.set_xlabel("|SSM d0 residual| threshold [mm]")
    ax.set_ylabel("Fraction")
    ax.set_title("Gradient noise budget from outlier tracks")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)

    plt.tight_layout()
    fig.savefig(output_dir / "performance_vs_cuts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. SSM prediction at d0=0 (the horizontal stripe)
    print("  6/6  SSM collapse analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distribution of SSM d0 predictions for tracks with |truth d0| > 2mm
    ax = axes[0]
    far_mask = np.abs(truth_d0) > 2.0
    ssm_d0_far = ssm_d0[far_mask]
    bins_pred = np.linspace(-5, 5, 200)
    ax.hist(ssm_d0_far, bins=bins_pred, histtype="step", linewidth=1.5, color="steelblue")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("SSM predicted d0 [mm]")
    ax.set_ylabel("Tracks")
    ax.set_title(f"SSM d0 predictions for |truth d0| > 2 mm\n({int(np.sum(far_mask)):,} tracks)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # SSM mean prediction in truth d0 bins (profile)
    ax = axes[1]
    d0_bins = np.linspace(-5, 5, 101)
    d0_centers = 0.5 * (d0_bins[:-1] + d0_bins[1:])
    means = []
    for j in range(len(d0_bins) - 1):
        m = (truth_d0 >= d0_bins[j]) & (truth_d0 < d0_bins[j + 1])
        if np.sum(m) > 10:
            means.append(np.mean(ssm_d0[m]))
        else:
            means.append(np.nan)
    ax.plot(d0_centers, means, "o-", color="steelblue", ms=2, label="SSM mean pred")
    ax.plot([-5, 5], [-5, 5], "r--", lw=0.8, label="y=x")
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("Mean SSM predicted d0 [mm]")
    ax.set_title("SSM d0 prediction profile")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Regression slope
    ax = axes[2]
    stds_by_d0 = []
    for j in range(len(d0_bins) - 1):
        m = (truth_d0 >= d0_bins[j]) & (truth_d0 < d0_bins[j + 1])
        if np.sum(m) > 10:
            stds_by_d0.append(np.std(ssm_resid_d0[m]))
        else:
            stds_by_d0.append(np.nan)
    ax.plot(d0_centers, stds_by_d0, "o-", color="steelblue", ms=2)
    ax.set_xlabel("Truth d0 [mm]")
    ax.set_ylabel("σ(SSM d0 residual) [mm]")
    ax.set_title("SSM d0 precision vs truth d0")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "ssm_collapse_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
