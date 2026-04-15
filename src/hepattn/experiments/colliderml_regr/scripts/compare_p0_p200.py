#!/usr/bin/env python3
"""Side-by-side comparison of p0 and p200 datasets.

Loads data through the PyTorch dataloader (ColliderMLTrackDataset + collate_tracks)
so we see exactly what the model sees. Produces:

1. Side-by-side target distribution histograms (d0, z0, phi, theta, qop, pt, eta)
2. Side-by-side input feature histograms (all 12 hit features)
3. 10 event displays per dataset (p200: 50 random tracks, p0: all tracks)

Usage (from /shared/tracking/hepattn_muon/src):
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.compare_p0_p200
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from torch.utils.data import DataLoader

from hepattn.experiments.colliderml_regr.data import (
    ColliderMLTrackDataset,
    collate_tracks,
)

# ============================================================================
# Config
# ============================================================================

P0_DIR = Path("/scratch/colliderml/p0/p0_preprocessed")
P200_DIR = Path("/scratch/colliderml/p200_preprocessed_plus_qcd")
OUTPUT_DIR = Path("/shared/tracking/logs/compare_p0_p200")

N_SHARDS = 100  # first 100 shards ≈ 100k events worth of tracks
N_EVENT_DISPLAYS = 10
P200_TRACKS_PER_EVENT = 50  # random sample for p200
BATCH_SIZE = 4096
NUM_WORKERS = 4
SEED = 42

TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]
TARGET_UNITS = {"d0": "mm", "z0": "mm", "phi": "rad", "theta": "rad", "qop": "1/GeV"}
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector", "eta_hit",
]

# ============================================================================
# Helpers
# ============================================================================


def load_shard_indices(preprocessed_dir: Path, n_shards: int) -> list[int]:
    """Return the first n_shards shard indices from the train split."""
    import json
    split_path = preprocessed_dir / "split.json"
    with open(split_path) as f:
        split = json.load(f)
    # Use train shards so we see what the model trains on
    return split["train"][:n_shards]


def collect_from_dataloader(ds, max_tracks: int = 100_000) -> dict:
    """Run through the dataloader and collect inputs + targets as numpy arrays."""
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_tracks,
    )

    all_hit_feats = []
    all_targets = []
    all_hit_s = []
    all_hit_valid = []
    all_lengths = []
    n_collected = 0

    for inputs, targets in loader:
        B = inputs["hit_features"].shape[0]
        valid = inputs["hit_valid"]  # (B, L)

        # Collect targets
        tgt = torch.stack([targets[k] for k in TARGET_NAMES], dim=-1).numpy()  # (B, 5)
        all_targets.append(tgt)

        # Collect hit features — only valid hits
        feats = inputs["hit_features"]  # (B, L, 12)
        s_vals = inputs["hit_s"]  # (B, L)
        for i in range(B):
            mask = valid[i].numpy()
            all_hit_feats.append(feats[i, mask].numpy())
            all_lengths.append(int(mask.sum()))

        n_collected += B
        if n_collected >= max_tracks:
            break

    return {
        "targets": np.concatenate(all_targets, axis=0),  # (N, 5)
        "hit_feats": np.concatenate(all_hit_feats, axis=0),  # (total_hits, 12)
        "lengths": all_lengths,
    }


def derive_pt_eta(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive pt and eta from theta and qop columns."""
    theta = targets[:, 3]
    qop = targets[:, 4]
    pt = np.sin(theta) / np.maximum(np.abs(qop), 1e-8)
    eta = -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))
    return pt, eta


# ============================================================================
# Plotting: side-by-side histograms
# ============================================================================


def plot_side_by_side_hist(
    vals_p0: np.ndarray,
    vals_p200: np.ndarray,
    name: str,
    unit: str,
    output_path: Path,
    n_bins: int = 120,
    log_scale: bool = False,
):
    """Plot overlapping histograms for p0 and p200."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Full range — no percentile clipping
    combined = np.concatenate([vals_p0, vals_p200])
    plot_range = (float(combined.min()), float(combined.max()))

    ax.hist(vals_p0, bins=n_bins, range=plot_range, histtype="stepfilled",
            alpha=0.5, color="steelblue", edgecolor="navy", linewidth=0.5,
            label=f"p0 (N={len(vals_p0):,})", density=True)
    ax.hist(vals_p200, bins=n_bins, range=plot_range, histtype="stepfilled",
            alpha=0.5, color="coral", edgecolor="darkred", linewidth=0.5,
            label=f"p200 (N={len(vals_p200):,})", density=True)

    xlabel = f"{name} [{unit}]" if unit else name
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{name} — p0 vs p200", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    # Stats annotation
    stats = (
        f"p0:   μ={vals_p0.mean():.4f}, σ={vals_p0.std():.4f}\n"
        f"p200: μ={vals_p200.mean():.4f}, σ={vals_p200.std():.4f}"
    )
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, va="top", ha="right",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Plotting: event displays
# ============================================================================


def load_raw_shard_tracks(preprocessed_dir: Path, shard_idx: int):
    """Load all tracks from one shard with event grouping for event displays."""
    shard_dir = preprocessed_dir / f"shard_{shard_idx:04d}"
    sel_dir = shard_dir / "selected_tracks"

    hits = np.load(shard_dir / "hits.npy", mmap_mode="r")
    targets = np.load(sel_dir / "track_targets.npy", mmap_mode="r")
    offsets = np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r")
    hit_indices = np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r")
    event_idx = np.load(sel_dir / "track_event_idx.npy", mmap_mode="r")

    return {
        "hits": hits,
        "targets": np.array(targets),
        "offsets": np.array(offsets),
        "hit_indices": hit_indices,
        "event_idx": np.array(event_idx),
    }


def get_track_hits(data: dict, track_idx: int) -> np.ndarray:
    """Extract hit positions for a given track index."""
    start = int(data["offsets"][track_idx])
    end = int(data["offsets"][track_idx + 1])
    idx = np.array(data["hit_indices"][start:end])
    return np.array(data["hits"][idx])  # (L, 11)


def plot_event_display(
    data: dict,
    event_id: int,
    output_path: Path,
    dataset_label: str,
    max_tracks: int | None = None,
    rng: np.random.RandomState | None = None,
):
    """Plot an r-z and x-y event display, matching explore_preprocessed style."""
    mask = data["event_idx"] == event_id
    track_indices = np.where(mask)[0]
    n_total = len(track_indices)

    if n_total == 0:
        return

    # Subsample if needed
    if max_tracks is not None and n_total > max_tracks:
        if rng is None:
            rng = np.random.RandomState(42)
        track_indices = rng.choice(track_indices, max_tracks, replace=False)
        subtitle = f"{max_tracks} / {n_total} tracks (random sample)"
    else:
        subtitle = f"{n_total} tracks"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by track pt
    colors = cm.viridis(np.linspace(0.15, 0.95, len(track_indices)))
    rng_color = np.random.RandomState(event_id)
    rng_color.shuffle(colors)

    for i, tidx in enumerate(track_indices):
        hit_feats = get_track_hits(data, tidx)
        x = hit_feats[:, 0]
        y = hit_feats[:, 1]
        z = hit_feats[:, 2]
        r = hit_feats[:, 3]

        # Sign r by y to get signed-r for r-z view
        r_signed = r * np.sign(y + 1e-12)

        c = colors[i]

        # r-z view
        axes[0].plot(z, r_signed, '-o', color=c, markersize=2.0, linewidth=0.6, alpha=0.7)

        # x-y view
        axes[1].plot(x, y, '-o', color=c, markersize=2.0, linewidth=0.6, alpha=0.7)

    axes[0].set_xlabel("z [mm]", fontsize=11)
    axes[0].set_ylabel("signed r [mm]", fontsize=11)
    axes[0].set_title("r-z view", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    axes[1].set_xlabel("x [mm]", fontsize=11)
    axes[1].set_ylabel("y [mm]", fontsize=11)
    axes[1].set_title("x-y view", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")

    fig.suptitle(f"{dataset_label} — Event {event_id} ({subtitle})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_dir = OUTPUT_DIR / "histograms"
    hist_dir.mkdir(exist_ok=True)
    evt_dir = OUTPUT_DIR / "event_displays"
    evt_dir.mkdir(exist_ok=True)

    print(f"Output: {OUTPUT_DIR}")

    # ---- Load data through PyTorch dataloaders ----
    print("\n--- Loading p0 data through dataloader ---")
    p0_shards = load_shard_indices(P0_DIR, N_SHARDS)
    p0_ds = ColliderMLTrackDataset(P0_DIR, p0_shards, load_acts=True)
    print(f"  p0: {len(p0_ds):,} tracks from shards {p0_shards[0]}–{p0_shards[-1]}")

    print("\n--- Loading p200 data through dataloader ---")
    p200_shards = load_shard_indices(P200_DIR, N_SHARDS)
    p200_ds = ColliderMLTrackDataset(P200_DIR, p200_shards, load_acts=True)
    print(f"  p200: {len(p200_ds):,} tracks from shards {p200_shards[0]}–{p200_shards[-1]}")

    print("\n--- Collecting p0 batches ---")
    p0_data = collect_from_dataloader(p0_ds, max_tracks=1_000_000)
    print(f"  Collected {len(p0_data['targets']):,} tracks, {len(p0_data['hit_feats']):,} hits")

    print("\n--- Collecting p200 batches ---")
    p200_data = collect_from_dataloader(p200_ds, max_tracks=1_000_000)
    print(f"  Collected {len(p200_data['targets']):,} tracks, {len(p200_data['hit_feats']):,} hits")

    # ---- Target distributions ----
    print("\n--- Plotting target distributions ---")
    tgt_dir = hist_dir / "targets"
    tgt_dir.mkdir(exist_ok=True)

    p0_pt, p0_eta = derive_pt_eta(p0_data["targets"])
    p200_pt, p200_eta = derive_pt_eta(p200_data["targets"])

    # ---- Summary statistics text file ----
    print("\n--- Writing summary statistics ---")
    all_var_names = TARGET_NAMES + ["pt", "eta"]
    all_var_units = {**TARGET_UNITS, "pt": "GeV", "eta": ""}

    p0_arrays = {name: p0_data["targets"][:, i] for i, name in enumerate(TARGET_NAMES)}
    p0_arrays["pt"] = p0_pt
    p0_arrays["eta"] = p0_eta

    p200_arrays = {name: p200_data["targets"][:, i] for i, name in enumerate(TARGET_NAMES)}
    p200_arrays["pt"] = p200_pt
    p200_arrays["eta"] = p200_eta

    lines = []
    lines.append("=" * 100)
    lines.append("  p0 vs p200 — Target Variable Statistics (side-by-side)")
    lines.append("=" * 100)
    lines.append(f"  p0   tracks: {len(p0_data['targets']):>12,}")
    lines.append(f"  p200 tracks: {len(p200_data['targets']):>12,}")
    lines.append(f"  p0   shards: {N_SHARDS} (train split)")
    lines.append(f"  p200 shards: {N_SHARDS} (train split)")
    lines.append("")

    header = (
        f"  {'Variable':<10s} {'Unit':<8s} │ "
        f"{'':^47s} │ "
        f"{'':^47s}"
    )
    header2 = (
        f"  {'':10s} {'':8s} │ "
        f"{'p0':^47s} │ "
        f"{'p200':^47s}"
    )
    header3 = (
        f"  {'':10s} {'':8s} │ "
        f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s} │ "
        f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s}"
    )
    sep = "  " + "─" * 10 + " " + "─" * 8 + "─┼─" + "─" * 47 + "─┼─" + "─" * 47

    lines.append(header2)
    lines.append(header3)
    lines.append(sep)

    for name in all_var_names:
        unit = all_var_units.get(name, "")
        v0 = p0_arrays[name]
        v2 = p200_arrays[name]
        row = (
            f"  {name:<10s} {unit:<8s} │ "
            f"{v0.min():>11.4f} {v0.max():>11.4f} {v0.mean():>11.4f} {v0.std():>11.4f} │ "
            f"{v2.min():>11.4f} {v2.max():>11.4f} {v2.mean():>11.4f} {v2.std():>11.4f}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append("")

    # Also add hit feature stats
    lines.append("=" * 100)
    lines.append("  p0 vs p200 — Input Feature Statistics (side-by-side)")
    lines.append("=" * 100)
    lines.append(f"  p0   hits:   {len(p0_data['hit_feats']):>12,}")
    lines.append(f"  p200 hits:   {len(p200_data['hit_feats']):>12,}")
    lines.append("")

    header2_h = (
        f"  {'':12s} │ "
        f"{'p0':^47s} │ "
        f"{'p200':^47s}"
    )
    header3_h = (
        f"  {'Feature':<12s} │ "
        f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s} │ "
        f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s}"
    )
    sep_h = "  " + "─" * 12 + "─┼─" + "─" * 47 + "─┼─" + "─" * 47

    lines.append(header2_h)
    lines.append(header3_h)
    lines.append(sep_h)

    for i, name in enumerate(HIT_FEATURE_NAMES):
        v0 = p0_data["hit_feats"][:, i]
        v2 = p200_data["hit_feats"][:, i]
        row = (
            f"  {name:<12s} │ "
            f"{v0.min():>11.4f} {v0.max():>11.4f} {v0.mean():>11.4f} {v0.std():>11.4f} │ "
            f"{v2.min():>11.4f} {v2.max():>11.4f} {v2.mean():>11.4f} {v2.std():>11.4f}"
        )
        lines.append(row)

    # nhits per track
    lines.append(sep_h)
    lines.append("")
    p0_len = np.array(p0_data["lengths"], dtype=np.float64)
    p200_len = np.array(p200_data["lengths"], dtype=np.float64)
    lines.append(
        f"  {'nhits/track':<12s} │ "
        f"{p0_len.min():>11.1f} {p0_len.max():>11.1f} {p0_len.mean():>11.2f} {p0_len.std():>11.2f} │ "
        f"{p200_len.min():>11.1f} {p200_len.max():>11.1f} {p200_len.mean():>11.2f} {p200_len.std():>11.2f}"
    )
    lines.append("")

    summary_text = "\n".join(lines)
    summary_path = OUTPUT_DIR / "statistics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(summary_text)
    print(f"\n  Written to {summary_path}")

    for i, name in enumerate(TARGET_NAMES):
        plot_side_by_side_hist(
            p0_data["targets"][:, i],
            p200_data["targets"][:, i],
            name, TARGET_UNITS.get(name, ""),
            tgt_dir / f"{name}.png",
            log_scale=(name in ("d0", "qop")),
        )
        print(f"  {name}")

    # pt and eta (derived)
    plot_side_by_side_hist(p0_pt, p200_pt, "pt", "GeV", tgt_dir / "pt.png", log_scale=True)
    print("  pt")
    plot_side_by_side_hist(p0_eta, p200_eta, "eta", "", tgt_dir / "eta.png")
    print("  eta")

    # ---- Input feature distributions ----
    print("\n--- Plotting input feature distributions ---")
    feat_dir = hist_dir / "input_features"
    feat_dir.mkdir(exist_ok=True)

    for i, name in enumerate(HIT_FEATURE_NAMES):
        is_discrete = name in ("volume_id", "layer_id", "surface_id", "detector")
        plot_side_by_side_hist(
            p0_data["hit_feats"][:, i],
            p200_data["hit_feats"][:, i],
            name, "",
            feat_dir / f"{name}.png",
            n_bins=50 if is_discrete else 120,
        )
        print(f"  {name}")

    # ---- nhits per track distribution ----
    print("\n--- Plotting nhits per track ---")
    p0_lengths = np.array(p0_data["lengths"])
    p200_lengths = np.array(p200_data["lengths"])
    plot_side_by_side_hist(
        p0_lengths.astype(np.float64),
        p200_lengths.astype(np.float64),
        "nhits_per_track", "",
        hist_dir / "nhits_per_track.png",
        n_bins=20,
    )
    print("  nhits_per_track")

    # ---- Event displays ----
    print("\n--- Generating event displays ---")
    rng = np.random.RandomState(SEED)

    # p0 event displays
    print("  Loading p0 shard 0 for event displays...")
    p0_shard_data = load_raw_shard_tracks(P0_DIR, p0_shards[0])
    p0_events = np.unique(p0_shard_data["event_idx"])
    p0_display_events = p0_events[:N_EVENT_DISPLAYS]

    for ev in p0_display_events:
        plot_event_display(
            p0_shard_data, ev,
            evt_dir / f"p0_event_{ev:04d}.png",
            dataset_label="p0 (pileup-0)",
            max_tracks=None,  # show all
        )
        print(f"    p0 event {ev}")

    # p200 event displays
    print("  Loading p200 shard 0 for event displays...")
    p200_shard_data = load_raw_shard_tracks(P200_DIR, p200_shards[0])
    p200_events = np.unique(p200_shard_data["event_idx"])
    p200_display_events = p200_events[:N_EVENT_DISPLAYS]

    for ev in p200_display_events:
        plot_event_display(
            p200_shard_data, ev,
            evt_dir / f"p200_event_{ev:04d}.png",
            dataset_label="p200 (pileup-200)",
            max_tracks=P200_TRACKS_PER_EVENT,
            rng=rng,
        )
        print(f"    p200 event {ev}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
