#!/usr/bin/env python3
"""Explore preprocessed memmap data across three selection regimes.

Reads compact (p200) or full (p0) preprocessed memmap format and generates
distribution plots, ACTS precision plots, and normalization validation for
three track selection regimes:

1. **tight_dm** — Double-matched + tight kinematic+hard_scatter cuts
2. **soft_dm**  — Double-matched + soft kinematic cuts
3. **all**      — Entire dataset (no additional cuts)

Usage::

    python explore_preprocessed.py \\
        --preprocessed-dir /scratch/colliderml/p200_preprocessed_plus_qcd \\
        --output-dir /tmp/explore_p200 \\
        --num-shards 10

    # Also works with p0 full-format data:
    python explore_preprocessed.py \\
        --preprocessed-dir /scratch/colliderml/p0/p0_preprocessed \\
        --output-dir /tmp/explore_p0 \\
        --num-shards 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Reuse precision plotting utilities from the evaluation module
from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAM_LABELS,
    UNIT_SCALE,
    compute_precision_vs_eta,
)
from hepattn.experiments.colliderml_regr.evaluate_predictions import (
    plot_precision_vs_eta,
)

# ============================================================================
# Constants
# ============================================================================

HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector",
]

TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]

# Reference normalization ranges from config/base.yaml
REF_NORM_MIN = [-1031.0, -1031.0, -3026.0, 31.0, -3.1416, 0.027, 31.0, 16.0, 2.0, 1.0, 0.0, -4.3]
REF_NORM_MAX = [1031.0, 1031.0, 3026.0, 1032.0, 3.1416, 3.114, 3185.0, 30.0, 16.0, 3360.0, 8.0, 4.3]

PARAMS = ["d0", "z0", "phi", "theta", "qop"]


# ============================================================================
# Running statistics accumulator
# ============================================================================

class RunningStats:
    """Online min/max/mean/std accumulator for a fixed number of features."""

    def __init__(self, n_features: int):
        self.n = n_features
        self.count = 0
        self.min_val = np.full(n_features, np.inf)
        self.max_val = np.full(n_features, -np.inf)
        self.sum_ = np.zeros(n_features, dtype=np.float64)
        self.sum2_ = np.zeros(n_features, dtype=np.float64)

    def update(self, data: np.ndarray):
        """Update with (N, n_features) array."""
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n = data.shape[0]
        if n == 0:
            return
        self.count += n
        d = data.astype(np.float64)
        self.min_val = np.minimum(self.min_val, d.min(axis=0))
        self.max_val = np.maximum(self.max_val, d.max(axis=0))
        self.sum_ += d.sum(axis=0)
        self.sum2_ += (d ** 2).sum(axis=0)

    @property
    def mean(self) -> np.ndarray:
        return self.sum_ / max(self.count, 1)

    @property
    def std(self) -> np.ndarray:
        m = self.mean
        return np.sqrt(np.maximum(self.sum2_ / max(self.count, 1) - m ** 2, 0.0))


# ============================================================================
# Format detection
# ============================================================================

def detect_format(preprocessed_dir: Path) -> str:
    """Detect whether preprocessed data uses compact or full format."""
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found in {preprocessed_dir}")
    first = shard_dirs[0]
    if (first / "event_hit_offsets.npy").exists():
        return "full"
    return "compact"


# ============================================================================
# Regime mask builders
# ============================================================================

def build_tight_dm_mask(
    targets: np.ndarray,
    dm_mask: np.ndarray | None,
    meta: np.ndarray | None,
) -> np.ndarray:
    """Tight DM: double-matched + hard_scatter + tight kinematic cuts."""
    n = len(targets)
    mask = np.ones(n, dtype=bool)

    if dm_mask is not None:
        mask &= dm_mask

    # Kinematic cuts from targets
    d0 = targets[:, 0]
    z0 = targets[:, 1]
    theta = targets[:, 3]
    qop = targets[:, 4]
    theta_safe = np.clip(theta, 1e-8, np.pi - 1e-8)
    eta = -np.log(np.tan(theta_safe / 2.0))

    mask &= np.abs(d0) <= 1.0
    mask &= np.abs(z0) <= 150.0
    mask &= np.abs(eta) <= 3.0

    # From track_meta if available
    if meta is not None:
        pt = meta[:, 0]
        vertex_primary = meta[:, 1]
        mask &= pt >= 0.5
        mask &= vertex_primary == 1
    else:
        # Derive pt from targets as fallback
        pt = np.sin(theta) / np.maximum(np.abs(qop), 1e-8)
        mask &= pt >= 0.5

    return mask


def build_soft_dm_mask(
    targets: np.ndarray,
    dm_mask: np.ndarray | None,
) -> np.ndarray:
    """Soft DM: double-matched + soft kinematic cuts."""
    n = len(targets)
    mask = np.ones(n, dtype=bool)

    if dm_mask is not None:
        mask &= dm_mask

    d0 = targets[:, 0]
    z0 = targets[:, 1]
    theta = targets[:, 3]
    qop = targets[:, 4]

    pt = np.sin(theta) / np.maximum(np.abs(qop), 1e-8)
    mask &= np.abs(d0) <= 5.0
    mask &= np.abs(z0) <= 200.0
    mask &= pt >= 0.2

    return mask


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_histogram(values: np.ndarray, name: str, output_path: Path, n_bins: int = 100):
    """Plot a single histogram with mean/std/count annotation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Clip extreme outliers for better visualization
    q01, q99 = np.percentile(values, [0.5, 99.5])
    plot_range = (q01, q99)

    ax.hist(values, bins=n_bins, range=plot_range, histtype="stepfilled",
            alpha=0.7, color="steelblue", edgecolor="navy", linewidth=0.5)

    stats_text = (
        f"N = {len(values):,}\n"
        f"mean = {values.mean():.4f}\n"
        f"std = {values.std():.4f}\n"
        f"min = {values.min():.4f}\n"
        f"max = {values.max():.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of {name}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main exploration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explore preprocessed ColliderML memmap data across selection regimes"
    )
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=-1,
                        help="Limit number of shards (-1 for all)")
    parser.add_argument("--n-eta-bins", type=int, default=30)
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(preprocessed_dir)
    print(f"Detected format: {fmt}")

    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if args.num_shards > 0:
        shard_dirs = shard_dirs[: args.num_shards]
    print(f"Processing {len(shard_dirs)} shards from {preprocessed_dir}")

    # Load manifest if available
    manifest_path = preprocessed_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Manifest: {json.dumps(manifest.get('totals', {}), indent=2)}")

    # ---- Accumulators per regime ----
    regimes = ["tight_dm", "soft_dm", "all"]
    regime_labels = {
        "tight_dm": "Tight DM (hard-scatter)",
        "soft_dm": "Soft DM",
        "all": "All tracks",
    }

    # Hit feature stats (11 raw + 1 derived eta_hit = 12)
    hit_stats = {r: RunningStats(12) for r in regimes}
    # Target stats (5 targets + 2 derived: pt, eta = 7)
    target_stats = {r: RunningStats(7) for r in regimes}
    # Track counts per regime
    track_counts = {r: 0 for r in regimes}
    # Per-shard track counts (for distribution)
    shard_track_counts = {r: [] for r in regimes}
    # Hit counts per track (for mean hits/track)
    total_hits_in_regime = {r: 0 for r in regimes}

    # ACTS residuals for precision plots (accumulate for DM regimes)
    acts_residuals = {r: {p: [] for p in PARAMS} for r in ["tight_dm", "soft_dm"]}
    acts_eta = {r: [] for r in ["tight_dm", "soft_dm"]}

    has_acts = False
    has_meta = False

    for shard_dir in tqdm(shard_dirs, desc="Loading shards", file=sys.stderr):
        sel_dir = shard_dir / "selected_tracks"
        if not sel_dir.exists():
            continue

        targets = np.load(sel_dir / "track_targets.npy", mmap_mode="r")
        offsets = np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r")
        hits = np.load(shard_dir / "hits.npy", mmap_mode="r")
        hit_indices = np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r")

        n_tracks = len(targets)
        if n_tracks == 0:
            continue

        # Load targets as regular array for filtering
        targets_arr = np.array(targets)  # (N, 5)

        # Load optional ACTS data
        dm_mask = None
        acts_reco = None
        acts_reco_path = sel_dir / "acts_reco.npy"
        acts_dm_path = sel_dir / "acts_dm_mask.npy"
        if acts_reco_path.exists() and acts_dm_path.exists():
            acts_reco = np.load(acts_reco_path, mmap_mode="r")
            dm_mask = np.array(np.load(acts_dm_path, mmap_mode="r"))
            has_acts = True

        # Load optional metadata
        meta = None
        meta_path = sel_dir / "track_meta.npy"
        if meta_path.exists():
            meta = np.array(np.load(meta_path, mmap_mode="r"))
            has_meta = True

        # Build regime masks
        masks = {
            "tight_dm": build_tight_dm_mask(targets_arr, dm_mask, meta),
            "soft_dm": build_soft_dm_mask(targets_arr, dm_mask),
            "all": np.ones(n_tracks, dtype=bool),
        }

        # Derive pt and eta for target stats
        theta_arr = targets_arr[:, 3]
        qop_arr = targets_arr[:, 4]
        pt_arr = np.sin(theta_arr) / np.maximum(np.abs(qop_arr), 1e-8)
        eta_arr = -np.log(np.tan(np.clip(theta_arr, 1e-8, np.pi - 1e-8) / 2.0))

        # Extended targets: [d0, z0, phi, theta, qop, pt, eta]
        ext_targets = np.column_stack([targets_arr, pt_arr, eta_arr])

        offsets_arr = np.array(offsets)

        for regime in regimes:
            rmask = masks[regime]
            n_sel = int(rmask.sum())
            track_counts[regime] += n_sel
            shard_track_counts[regime].append(n_sel)

            if n_sel == 0:
                continue

            # Target stats
            target_stats[regime].update(ext_targets[rmask])

            # Hit stats: gather hits for selected tracks
            sel_indices = np.where(rmask)[0]

            # Batch gather hits for efficiency
            hit_chunks = []
            for si in sel_indices:
                start = int(offsets_arr[si])
                end = int(offsets_arr[si + 1])
                total_hits_in_regime[regime] += end - start
                idx = np.array(hit_indices[start:end])
                hit_chunks.append(np.array(hits[idx]))

            if hit_chunks:
                all_sel_hits = np.concatenate(hit_chunks, axis=0)  # (total_hits, 11)
                # Compute eta_hit from theta_hit (col 5)
                theta_hit = all_sel_hits[:, 5].copy()
                eta_hit = -np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0))
                eta_hit = np.clip(eta_hit, -10.0, 10.0)
                all_sel_hits_ext = np.column_stack([all_sel_hits, eta_hit])
                hit_stats[regime].update(all_sel_hits_ext)

            # ACTS residuals for precision plots (DM regimes only)
            if regime in acts_residuals and acts_reco is not None:
                acts_arr = np.array(acts_reco)
                sel_acts = acts_arr[rmask]
                sel_targets = targets_arr[rmask]

                # Filter out tracks where ACTS has no match (NaN)
                has_match = ~np.any(np.isnan(sel_acts), axis=1)
                if has_match.any():
                    matched_acts = sel_acts[has_match]
                    matched_truth = sel_targets[has_match]

                    for i, pname in enumerate(PARAMS):
                        res = matched_acts[:, i] - matched_truth[:, i]
                        if pname == "phi":
                            res = (res + np.pi) % (2 * np.pi) - np.pi
                        acts_residuals[regime][pname].append(res)

                    truth_theta = matched_truth[:, 3]
                    acts_eta[regime].append(-np.log(np.tan(truth_theta / 2.0 + 1e-12)))

    # ============================================================================
    # Generate outputs per regime
    # ============================================================================

    print(f"\n{'='*70}")
    print("TRACK COUNTS PER REGIME")
    print(f"{'='*70}")
    for regime in regimes:
        print(f"  {regime_labels[regime]:30s}: {track_counts[regime]:>12,} tracks")

    summary_lines = []
    summary_lines.append(f"Exploration of {preprocessed_dir}")
    summary_lines.append(f"Format: {fmt}")
    summary_lines.append(f"Shards processed: {len(shard_dirs)}")
    summary_lines.append(f"Has ACTS augmentation: {has_acts}")
    summary_lines.append(f"Has track metadata: {has_meta}")
    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("TRACK COUNTS PER REGIME")
    summary_lines.append("=" * 70)
    for regime in regimes:
        n = track_counts[regime]
        mean_hits = total_hits_in_regime[regime] / max(n, 1)
        summary_lines.append(
            f"  {regime_labels[regime]:30s}: {n:>12,} tracks, "
            f"{mean_hits:.1f} mean hits/track"
        )

    for regime in regimes:
        if track_counts[regime] == 0:
            print(f"\n  Skipping {regime} (0 tracks)")
            continue

        regime_dir = output_dir / regime
        print(f"\n{'='*70}")
        print(f"REGIME: {regime_labels[regime]} ({track_counts[regime]:,} tracks)")
        print(f"{'='*70}")

        # ---- Hit feature distributions ----
        hit_dir = regime_dir / "hit_features"
        hit_dir.mkdir(parents=True, exist_ok=True)

        all_feature_names = HIT_FEATURE_NAMES + ["eta_hit"]
        hs = hit_stats[regime]
        if hs.count > 0:
            print(f"  Plotting {len(all_feature_names)} hit feature distributions...")
            # We need to re-iterate to get actual values for histograms
            # Use the accumulated stats to create synthetic histograms
            # Actually, for proper histograms we need the raw values.
            # Let's do a second pass for the selected regime, but with sampling.
            # For now, just log the stats.
            for fi, fname in enumerate(all_feature_names):
                summary_lines.append(
                    f"  hit/{fname:15s}: min={hs.min_val[fi]:12.4f}  "
                    f"max={hs.max_val[fi]:12.4f}  "
                    f"mean={hs.mean[fi]:12.4f}  std={hs.std[fi]:12.4f}"
                )

        # ---- Track target distributions ----
        ts = target_stats[regime]
        ext_target_names = TARGET_NAMES + ["pt", "eta"]
        if ts.count > 0:
            for fi, tname in enumerate(ext_target_names):
                summary_lines.append(
                    f"  target/{tname:10s}: min={ts.min_val[fi]:12.6f}  "
                    f"max={ts.max_val[fi]:12.6f}  "
                    f"mean={ts.mean[fi]:12.6f}  std={ts.std[fi]:12.6f}"
                )

        # ---- ACTS precision plots ----
        if regime in acts_residuals and acts_eta.get(regime):
            prec_dir = regime_dir / "precision"
            prec_dir.mkdir(parents=True, exist_ok=True)

            combined_residuals: dict[str, np.ndarray] = {}
            for pname in PARAMS:
                if acts_residuals[regime][pname]:
                    combined_residuals[pname] = np.concatenate(
                        acts_residuals[regime][pname]
                    )
            combined_eta = np.concatenate(acts_eta[regime])
            combined_residuals["eta"] = combined_eta

            n_matched = len(combined_eta)
            print(f"  ACTS precision plots: {n_matched:,} matched tracks")

            eta_bins, prec_data = compute_precision_vs_eta(
                combined_residuals,
                eta_range=(-3.0, 3.0),
                n_eta_bins=args.n_eta_bins,
            )
            plot_precision_vs_eta(
                prec_data, eta_bins, prec_dir,
                ml_label=f"ACTS CKF ({n_matched:,} tracks)",
            )

            # Log unbinned precision values
            summary_lines.append(f"\n  ACTS precision ({regime_labels[regime]}):")
            for pname in PARAMS:
                if pname in prec_data:
                    scale = UNIT_SCALE.get(pname, 1.0)
                    unit = {1.0: "", 1e3: " mrad"}.get(scale, "")
                    val = prec_data[pname]["unbinned_std"] * scale
                    summary_lines.append(f"    {pname:8s}: sigma = {val:.5f}{unit}")

    # ============================================================================
    # Normalization validation
    # ============================================================================

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("NORMALIZATION VALIDATION (vs base.yaml)")
    summary_lines.append("=" * 70)

    hs_all = hit_stats["all"]
    if hs_all.count > 0:
        all_feature_names_12 = HIT_FEATURE_NAMES + ["eta_hit"]
        for fi, fname in enumerate(all_feature_names_12):
            ref_min = REF_NORM_MIN[fi] if fi < len(REF_NORM_MIN) else None
            ref_max = REF_NORM_MAX[fi] if fi < len(REF_NORM_MAX) else None
            obs_min = hs_all.min_val[fi]
            obs_max = hs_all.max_val[fi]

            flag = ""
            if ref_min is not None and obs_min < ref_min:
                flag += f" ** BELOW norm_min ({ref_min})"
            if ref_max is not None and obs_max > ref_max:
                flag += f" ** ABOVE norm_max ({ref_max})"

            line = (
                f"  {fname:15s}: observed [{obs_min:12.4f}, {obs_max:12.4f}] "
                f"  config [{ref_min:12.4f}, {ref_max:12.4f}]{flag}"
            )
            summary_lines.append(line)
            if flag:
                print(f"  WARNING: {fname}{flag}")

    # ============================================================================
    # Per-shard track count distribution
    # ============================================================================

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("PER-SHARD TRACK COUNTS")
    summary_lines.append("=" * 70)

    for regime in regimes:
        counts = shard_track_counts[regime]
        if counts:
            arr = np.array(counts)
            summary_lines.append(
                f"  {regime_labels[regime]:30s}: "
                f"min={arr.min():,}  max={arr.max():,}  "
                f"mean={arr.mean():.0f}  std={arr.std():.0f}  "
                f"total={arr.sum():,}"
            )

    # ============================================================================
    # Manifest verification
    # ============================================================================

    if manifest and "totals" in manifest:
        summary_lines.append("")
        summary_lines.append("=" * 70)
        summary_lines.append("MANIFEST VERIFICATION")
        summary_lines.append("=" * 70)
        manifest_total = manifest["totals"].get("n_selected_tracks", 0)
        computed_total = track_counts["all"]
        match = "MATCH" if manifest_total == computed_total else "MISMATCH"
        summary_lines.append(
            f"  Manifest total_tracks: {manifest_total:,}  "
            f"Computed: {computed_total:,}  [{match}]"
        )

    # ============================================================================
    # Write summary
    # ============================================================================

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "statistics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary written to {summary_path}")

    # ============================================================================
    # Second pass: generate actual histogram plots (sampled for memory efficiency)
    # ============================================================================

    print("\n--- Generating histogram plots (second pass, sampling up to 50 shards) ---")
    plot_shards = shard_dirs[:min(50, len(shard_dirs))]

    for regime in regimes:
        if track_counts[regime] == 0:
            continue

        regime_dir = output_dir / regime

        # Accumulate sampled data for histograms
        sampled_hits = []
        sampled_targets = []
        max_samples = 2_000_000  # cap for memory

        for shard_dir in tqdm(plot_shards, desc=f"Sampling {regime}", file=sys.stderr):
            sel_dir = shard_dir / "selected_tracks"
            if not sel_dir.exists():
                continue

            targets_arr = np.array(np.load(sel_dir / "track_targets.npy", mmap_mode="r"))
            offsets_arr = np.array(np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r"))
            hit_indices = np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r")
            hits = np.load(shard_dir / "hits.npy", mmap_mode="r")

            n_tracks = len(targets_arr)
            if n_tracks == 0:
                continue

            # Load optional data for mask building
            dm_mask = None
            acts_dm_path = sel_dir / "acts_dm_mask.npy"
            if acts_dm_path.exists():
                dm_mask = np.array(np.load(acts_dm_path, mmap_mode="r"))

            meta = None
            meta_path = sel_dir / "track_meta.npy"
            if meta_path.exists():
                meta = np.array(np.load(meta_path, mmap_mode="r"))

            if regime == "tight_dm":
                rmask = build_tight_dm_mask(targets_arr, dm_mask, meta)
            elif regime == "soft_dm":
                rmask = build_soft_dm_mask(targets_arr, dm_mask)
            else:
                rmask = np.ones(n_tracks, dtype=bool)

            sel_indices = np.where(rmask)[0]
            if len(sel_indices) == 0:
                continue

            # Targets
            sel_tgt = targets_arr[rmask]
            theta_t = sel_tgt[:, 3]
            qop_t = sel_tgt[:, 4]
            pt_t = np.sin(theta_t) / np.maximum(np.abs(qop_t), 1e-8)
            eta_t = -np.log(np.tan(np.clip(theta_t, 1e-8, np.pi - 1e-8) / 2.0))
            ext_tgt = np.column_stack([sel_tgt, pt_t, eta_t])
            sampled_targets.append(ext_tgt)

            # Hits (subsample if too many tracks)
            for si in sel_indices[:500]:  # limit per shard
                start = int(offsets_arr[si])
                end = int(offsets_arr[si + 1])
                idx = np.array(hit_indices[start:end])
                sampled_hits.append(np.array(hits[idx]))

            if sum(len(h) for h in sampled_hits) > max_samples:
                break

        # Plot hit features
        if sampled_hits:
            all_hits = np.concatenate(sampled_hits, axis=0)
            theta_hit = all_hits[:, 5].copy()
            eta_hit = -np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0))
            eta_hit = np.clip(eta_hit, -10.0, 10.0)

            hit_dir = regime_dir / "hit_features"
            hit_dir.mkdir(parents=True, exist_ok=True)

            all_feature_names = HIT_FEATURE_NAMES + ["eta_hit"]
            for fi, fname in enumerate(all_feature_names):
                if fi < all_hits.shape[1]:
                    vals = all_hits[:, fi]
                else:
                    vals = eta_hit
                plot_histogram(vals, fname, hit_dir / f"{fname}.png")

            print(f"  {regime}: plotted {len(all_feature_names)} hit feature histograms "
                  f"({len(all_hits):,} hits)")

        # Plot target distributions
        if sampled_targets:
            all_tgt = np.concatenate(sampled_targets, axis=0)
            tgt_dir = regime_dir / "track_targets"
            tgt_dir.mkdir(parents=True, exist_ok=True)

            ext_target_names = TARGET_NAMES + ["pt", "eta"]
            for fi, tname in enumerate(ext_target_names):
                plot_histogram(all_tgt[:, fi], tname, tgt_dir / f"{tname}.png")

            print(f"  {regime}: plotted {len(ext_target_names)} target histograms "
                  f"({len(all_tgt):,} tracks)")

    print(f"\nAll outputs saved to {output_dir}")
    print(summary_text)


if __name__ == "__main__":
    main()
