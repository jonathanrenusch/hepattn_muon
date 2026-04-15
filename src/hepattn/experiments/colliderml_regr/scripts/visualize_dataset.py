#!/usr/bin/env python3
# ruff: noqa: TID252
"""Quick visualization of a preprocessed ColliderML dataset.

Generates ground-truth distribution histograms and event displays from the
first shard(s).  Designed to run fast (seconds, not minutes) for a quick
sanity check of a newly preprocessed dataset.

Usage::

    python visualize_dataset.py \
        --preprocessed-dir /eos/project/e/end-to-end-colliderml/data/p200_loose_pretrain \
        --output-dir /tmp/viz_p200_loose_pretrain

    # More events / tracks:
    python visualize_dataset.py \
        --preprocessed-dir /path/to/dataset \
        --output-dir /tmp/viz \
        --num-shards 2 --num-events 10 --tracks-per-event 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Constants
# ============================================================================

HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector",
]
TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]


# ============================================================================
# Histogram plotting
# ============================================================================

def plot_histogram(values: np.ndarray, name: str, output_path: Path, n_bins: int = 100):
    """Single histogram with stats annotation."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    q01, q99 = np.percentile(values, [0.5, 99.5])
    ax.hist(values, bins=n_bins, range=(q01, q99), histtype="stepfilled",
            alpha=0.7, color="steelblue", edgecolor="navy", linewidth=0.5)
    stats = (
        f"N = {len(values):,}\n"
        f"mean = {values.mean():.4f}\n"
        f"std = {values.std():.4f}\n"
        f"min = {values.min():.4f}\n"
        f"max = {values.max():.4f}"
    )
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, va="top", ha="right",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.set_title(name)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Event display
# ============================================================================

def plot_event_display(
    track_hits_list: list[np.ndarray],
    event_idx: int,
    output_dir: Path,
):
    """Plot xy, xz, yz projections for one event with multiple tracks."""
    projections = [
        ("xy", 0, 1, "x [mm]", "y [mm]"),
        ("zx", 2, 0, "z [mm]", "x [mm]"),
        ("zy", 2, 1, "z [mm]", "y [mm]"),
    ]
    cmap = plt.cm.tab20
    n_tracks = len(track_hits_list)

    for proj_name, ci, cj, xlabel, ylabel in projections:
        fig, ax = plt.subplots(figsize=(8, 7))
        for ti, hits in enumerate(track_hits_list):
            color = cmap(ti % 20)
            ax.plot(hits[:, ci], hits[:, cj], "-", color=color, alpha=0.5, linewidth=0.8)
            ax.scatter(hits[:, ci], hits[:, cj], color=color, s=8, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Event {event_idx} — {proj_name} projection ({n_tracks} tracks)")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(output_dir / f"event_{event_idx:03d}_{proj_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quick visualization of preprocessed ColliderML dataset")
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards to load (default: 1)")
    parser.add_argument("--num-events", type=int, default=10,
                        help="Number of events to display (default: 10)")
    parser.add_argument("--tracks-per-event", type=int, default=20,
                        help="Tracks to sample per event display (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found in {preprocessed_dir}")
    shard_dirs = shard_dirs[: args.num_shards]
    print(f"Loading {len(shard_dirs)} shard(s) from {preprocessed_dir}")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load data from shard(s)
    # ------------------------------------------------------------------
    all_targets = []
    all_hit_feats = []
    all_hit_counts = []
    # For event displays: track hits grouped by event
    event_tracks: dict[int, list[np.ndarray]] = {}  # event_global_idx -> list of (L, 11) arrays
    global_event_offset = 0

    for shard_dir in shard_dirs:
        sel_dir = shard_dir / "selected_tracks"
        if not sel_dir.exists():
            continue

        hits = np.load(shard_dir / "hits.npy", mmap_mode="r")
        targets = np.load(sel_dir / "track_targets.npy", mmap_mode="r")
        offsets = np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r")
        hit_indices = np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r")
        event_idx = np.load(sel_dir / "track_event_idx.npy", mmap_mode="r")

        n_tracks = len(targets)
        if n_tracks == 0:
            continue

        targets_arr = np.array(targets)
        offsets_arr = np.array(offsets)
        event_idx_arr = np.array(event_idx)
        all_targets.append(targets_arr)

        # Gather per-track hit counts and hit features
        for ti in range(n_tracks):
            start = int(offsets_arr[ti])
            end = int(offsets_arr[ti + 1])
            n_hits = end - start
            all_hit_counts.append(n_hits)

            idx = np.array(hit_indices[start:end])
            track_feats = np.array(hits[idx])  # (L, 11)

            # Sample hit features (keep all for histograms — memory bounded by shard count)
            all_hit_feats.append(track_feats)

            # Group for event display
            ev_global = int(event_idx_arr[ti]) + global_event_offset
            if ev_global not in event_tracks:
                event_tracks[ev_global] = []
            event_tracks[ev_global].append(track_feats)

        global_event_offset += int(event_idx_arr.max()) + 1 if n_tracks > 0 else 0

    if not all_targets:
        print("No tracks found in selected shards.")
        return

    targets_all = np.concatenate(all_targets, axis=0)
    hit_counts = np.array(all_hit_counts)
    print(f"Total tracks: {len(targets_all):,}, total events with tracks: {len(event_tracks)}")

    # ------------------------------------------------------------------
    # 1. Ground truth distributions
    # ------------------------------------------------------------------
    dist_dir = output_dir / "distributions"
    dist_dir.mkdir(exist_ok=True)

    # Target parameters
    for i, name in enumerate(TARGET_NAMES):
        plot_histogram(targets_all[:, i], name, dist_dir / f"target_{name}.png")

    # Derived: pt, eta
    theta_arr = targets_all[:, 3]
    qop_arr = targets_all[:, 4]
    pt_arr = np.sin(theta_arr) / np.maximum(np.abs(qop_arr), 1e-8)
    eta_arr = -np.log(np.tan(np.clip(theta_arr, 1e-8, np.pi - 1e-8) / 2.0))
    plot_histogram(pt_arr, "pt [GeV]", dist_dir / "target_pt.png")
    plot_histogram(eta_arr, "eta", dist_dir / "target_eta.png")

    # Hits per track
    plot_histogram(hit_counts.astype(np.float64), "hits_per_track", dist_dir / "hits_per_track.png")

    # Hit features
    all_hits_concat = np.concatenate(all_hit_feats, axis=0)
    for i, name in enumerate(HIT_FEATURE_NAMES):
        plot_histogram(all_hits_concat[:, i], name, dist_dir / f"hit_{name}.png")

    # Derived eta_hit
    theta_hit = all_hits_concat[:, 5].copy()
    eta_hit = -np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0))
    eta_hit = np.clip(eta_hit, -10.0, 10.0)
    plot_histogram(eta_hit, "eta_hit", dist_dir / "hit_eta_hit.png")

    print(f"Distribution plots saved to {dist_dir} "
          f"({5 + 2 + 1 + len(HIT_FEATURE_NAMES) + 1} histograms)")

    # ------------------------------------------------------------------
    # 2. Event displays
    # ------------------------------------------------------------------
    evd_dir = output_dir / "event_displays"
    evd_dir.mkdir(exist_ok=True)

    sorted_events = sorted(event_tracks.keys())
    display_events = sorted_events[: args.num_events]

    for ev_idx in display_events:
        tracks = event_tracks[ev_idx]
        if len(tracks) > args.tracks_per_event:
            chosen = rng.choice(len(tracks), size=args.tracks_per_event, replace=False)
            tracks = [tracks[i] for i in sorted(chosen)]
        plot_event_display(tracks, ev_idx, evd_dir)

    print(f"Event displays saved to {evd_dir} "
          f"({len(display_events)} events x 3 projections)")
    print(f"\nAll outputs in {output_dir}")


if __name__ == "__main__":
    main()
