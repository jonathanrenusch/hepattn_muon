"""Downsample p200_core_finetune so the d0 distribution is ~uniform.

Strategy (two-pass):
  1. Build a global d0 histogram (N_BINS bins over [-2.5, +2.5] mm) from
     all shards' track_targets.npy column 0.
  2. Compute per-bin accept probability
        p[b] = min(1, TARGET_PER_BIN / hist[b])
     which keeps all tracks in under-populated (tail) bins and downsamples
     over-populated (core) bins to the target, yielding a roughly uniform
     final distribution.

For each shard we rewrite the per-track arrays to the kept subset and
rebuild the CSR hit index (hit_offsets + hit_indices).  The shared
``hits.npy`` is symlinked — kept hit indices still point into the
original hit pool unchanged, so there's no reindexing needed and the
new dataset is trivially small on disk.

Output directory layout matches the source (one shard per dir, same
selected_tracks/* files + shared split.json copied over).

Finally: writes ``d0_distribution.png`` into the new dir, overlaying the
natural and downsampled d0 distributions so you can see how uniform the
result is.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_SRC = Path("/scratch/colliderml/p200_core_finetune")
DEFAULT_DST = Path("/scratch/colliderml/p200_core_finetune_uniform_d0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--d0-min-mm", type=float, default=-2.5)
    ap.add_argument("--d0-max-mm", type=float, default=2.5)
    ap.add_argument("--target-per-bin", type=int, default=6000,
                    help="Target number of kept tracks per bin.  Set close to "
                         "the minimum-populated natural bin (~5.7k for the d0 "
                         "tails) to get strictly-uniform output.  6000 * 500 = "
                         "3M target total (~1.2%% of the 240M natural dataset).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink-hits", action="store_true", default=True,
                    help="Symlink shared hits.npy instead of copying (default).")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    with open(args.src / "split.json") as f:
        split = json.load(f)
    all_shards = sorted(set(split["train"] + split["val"] + split["test"]))
    print(f"Discovered {len(all_shards)} shards across train/val/test.")

    # ---- Pass 1: histogram ----
    edges = np.linspace(args.d0_min_mm, args.d0_max_mm, args.n_bins + 1)
    hist = np.zeros(args.n_bins, dtype=np.int64)
    n_total = 0
    for s in tqdm(all_shards, desc="Pass 1 (histogram)"):
        d0 = np.load(
            args.src / f"shard_{s:04d}" / "selected_tracks" / "track_targets.npy",
            mmap_mode="r",
        )[:, 0]
        h, _ = np.histogram(d0, bins=edges)
        hist += h
        n_total += int(len(d0))
    print(f"Total natural tracks: {n_total:,}")
    print(f"Natural hist min/median/max: {hist.min()}, {int(np.median(hist))}, {hist.max()}")

    accept_prob = np.minimum(
        1.0, float(args.target_per_bin) / np.maximum(hist, 1).astype(np.float64)
    )

    # ---- Pass 2: filter + write ----
    rng = np.random.default_rng(args.seed)
    new_hist = np.zeros_like(hist)
    n_kept = 0
    for s in tqdm(all_shards, desc="Pass 2 (filter + write)"):
        src_dir = args.src / f"shard_{s:04d}"
        dst_dir = args.dst / f"shard_{s:04d}"
        sel_src = src_dir / "selected_tracks"
        sel_dst = dst_dir / "selected_tracks"
        sel_dst.mkdir(parents=True, exist_ok=True)

        targets = np.load(sel_src / "track_targets.npy")
        d0 = targets[:, 0]
        bin_idx = np.clip(np.digitize(d0, edges) - 1, 0, args.n_bins - 1)
        probs = accept_prob[bin_idx]
        keep_mask = rng.random(len(d0)) < probs
        kept_idx = np.nonzero(keep_mask)[0].astype(np.int64)
        n_kept += int(len(kept_idx))

        if len(kept_idx) == 0:
            continue

        # Update new histogram.
        h, _ = np.histogram(d0[keep_mask], bins=edges)
        new_hist += h

        # Filter per-track arrays.
        np.save(sel_dst / "track_targets.npy", targets[kept_idx])
        for fname in (
            "track_meta",
            "track_event_idx",
            "track_particle_ids",
            "acts_reco",
            "acts_dm_mask",
        ):
            a = np.load(sel_src / f"{fname}.npy")
            np.save(sel_dst / f"{fname}.npy", a[kept_idx])

        # Rebuild CSR hit indices.
        hit_indices = np.load(sel_src / "track_hit_indices.npy")
        hit_offsets = np.load(sel_src / "track_hit_offsets.npy")  # (N+1,)
        segments = [
            hit_indices[hit_offsets[i] : hit_offsets[i + 1]] for i in kept_idx
        ]
        if segments:
            new_indices = np.concatenate(segments)
        else:
            new_indices = np.array([], dtype=hit_indices.dtype)
        seg_lens = np.fromiter((len(x) for x in segments), dtype=hit_offsets.dtype, count=len(segments))
        new_offsets = np.zeros(len(kept_idx) + 1, dtype=hit_offsets.dtype)
        new_offsets[1:] = np.cumsum(seg_lens)
        np.save(sel_dst / "track_hit_indices.npy", new_indices)
        np.save(sel_dst / "track_hit_offsets.npy", new_offsets)

        # Symlink hits.npy (indices still valid against the original pool).
        dst_hits = dst_dir / "hits.npy"
        if not dst_hits.exists():
            src_hits = (src_dir / "hits.npy").resolve()
            os.symlink(src_hits, dst_hits)

    # Copy split file (track indices are shard-level; split structure is preserved).
    (args.dst / "split.json").write_text(json.dumps(split))
    print(f"\nKept {n_kept:,} / {n_total:,} tracks ({n_kept / max(n_total,1) * 100:.2f}%)")
    print(f"New hist min/median/max: {new_hist.min()}, {int(np.median(new_hist))}, {new_hist.max()}")

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for ax, log_scale in ((axes[0], False), (axes[1], True)):
        ax.plot(centers, hist, "-", color="C0", linewidth=1.5,
                label=f"natural (N={n_total:,})")
        ax.plot(centers, new_hist, "-", color="C1", linewidth=1.5,
                label=f"uniform-downsampled (N={n_kept:,})")
        ax.axhline(args.target_per_bin, color="gray", linestyle="--", alpha=0.6,
                   label=f"target per bin = {args.target_per_bin}")
        ax.set_xlabel("truth d0 [mm]")
        ax.set_ylabel("tracks / bin")
        ax.set_title("linear-y" if not log_scale else "log-y")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        f"d0 distribution: natural vs uniform-downsampled "
        f"({args.n_bins} bins over [{args.d0_min_mm:.1f}, {args.d0_max_mm:.1f}] mm)",
        fontsize=13,
    )
    plot_path = args.dst / "d0_distribution.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
