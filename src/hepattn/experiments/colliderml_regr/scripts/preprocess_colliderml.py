#!/usr/bin/env python3
# ruff: noqa: TID252, PLR0915, C901
"""Preprocess ColliderML Release-1 parquet shards into fast memmap format.

This script converts the raw parquet-based ColliderML dataset into a flat,
memory-mapped numpy format that is optimised for both:

1. **Track parameter regression** — per-selected-track random access with
   pre-sorted hits and pre-computed targets.
2. **Future hit-to-track assignment** — full event-level access to all hits
   and particles with preserved structure.

Output structure (per shard)::

    <output_dir>/shard_XXXX/
        hits.npy               — (total_hits, N_HIT_FEATURES) float32 memmap
        particles.npy          — (total_particles, N_PARTICLE_FEATURES) float32 memmap
        event_hit_offsets.npy  — (n_events + 1,) int64
        event_particle_offsets.npy — (n_events + 1,) int64
        selected_tracks/
            track_targets.npy       — (N_selected, 5) float32  [d0, z0, phi, theta, qop]
            track_hit_indices.npy   — (total_selected_hits,) int32  CSR values
            track_hit_offsets.npy   — (N_selected + 1,) int32  CSR offsets
            track_event_idx.npy     — (N_selected,) int32  event index within shard

Usage::

    python preprocess_colliderml.py --data-dir /scratch/colliderml/p0 \\
        --output-dir /scratch/colliderml/p0_preprocessed \\
        --num-shards -1 --num-workers 8

Quick test (2 shards)::

    python preprocess_colliderml.py --data-dir /scratch/colliderml/p0 \\
        --output-dir /scratch/colliderml/p0_preprocessed_test \\
        --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from hepattn.experiments.colliderml_regr.utils.selection_utils import load_selection_defaults


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hit features: x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector
N_HIT_FEATURES = 11
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector",
]

# Particle features: particle_id, pdg_id, charge, px, py, pz, perigee_d0, perigee_z0,
#                     phi, theta, qop, primary
N_PARTICLE_FEATURES = 12
PARTICLE_FEATURE_NAMES = [
    "particle_id", "pdg_id", "charge", "px", "py", "pz",
    "perigee_d0", "perigee_z0", "phi", "theta", "qop", "primary",
]

# Track target parameters
TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]

# Track selection defaults (loaded from shared selection_defaults.yaml)
DEFAULT_SELECTION = load_selection_defaults()


# ---------------------------------------------------------------------------
# Per-shard processing
# ---------------------------------------------------------------------------


def process_shard(
    particle_file: Path,
    hits_file: Path,
    output_dir: Path,
    selection: dict,
) -> dict:
    """Process a single shard pair → write memmap files.

    Returns a summary dict with counts.
    """
    sel = selection

    # Read with column projection
    ptable = pq.read_table(
        particle_file,
        columns=[
            "event_id", "particle_id", "pdg_id", "charge",
            "px", "py", "pz", "perigee_d0", "perigee_z0",
            "primary", "vertex_primary",
        ],
    )
    htable = pq.read_table(
        hits_file,
        columns=[
            "event_id", "particle_id",
            "x", "y", "z",
            "volume_id", "layer_id", "surface_id", "detector",
        ],
    )

    n_events = ptable.num_rows

    # Accumulators
    all_hit_feats = []       # list of (nhits_event, N_HIT_FEATURES)
    all_part_feats = []      # list of (nparts_event, N_PARTICLE_FEATURES)
    event_hit_counts = []
    event_part_counts = []

    all_track_targets = []   # list of (n_sel, 5)
    all_track_hit_idx = []   # list of 1-D arrays of global hit indices
    all_track_lengths = []   # list of ints
    all_track_event_idx = [] # list of ints

    global_hit_offset = 0

    for ev in range(n_events):
        # ---- Particles ------------------------------------------------
        pid = np.array(ptable.column("particle_id")[ev].as_py(), dtype=np.int64)
        pdg = np.array(ptable.column("pdg_id")[ev].as_py(), dtype=np.int32)
        charge = np.array(ptable.column("charge")[ev].as_py(), dtype=np.float32)
        px = np.array(ptable.column("px")[ev].as_py(), dtype=np.float64)
        py = np.array(ptable.column("py")[ev].as_py(), dtype=np.float64)
        pz = np.array(ptable.column("pz")[ev].as_py(), dtype=np.float64)
        d0 = np.array(ptable.column("perigee_d0")[ev].as_py(), dtype=np.float64)
        z0 = np.array(ptable.column("perigee_z0")[ev].as_py(), dtype=np.float64)
        is_primary = np.array(ptable.column("primary")[ev].as_py(), dtype=bool)
        vertex_primary = np.array(ptable.column("vertex_primary")[ev].as_py(), dtype=np.int32)

        # Derived kinematics
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        theta = np.arccos(np.clip(pz / (p + 1e-12), -1.0, 1.0))
        phi = np.arctan2(py, px)
        eta = -np.log(np.tan(theta / 2.0 + 1e-12))
        qop = np.where(p > 0, charge / p, 0.0)

        nparts = len(pid)

        # Particle feature matrix
        part_feats = np.zeros((nparts, N_PARTICLE_FEATURES), dtype=np.float32)
        part_feats[:, 0] = pid.astype(np.float32)
        part_feats[:, 1] = pdg.astype(np.float32)
        part_feats[:, 2] = charge
        part_feats[:, 3] = px.astype(np.float32)
        part_feats[:, 4] = py.astype(np.float32)
        part_feats[:, 5] = pz.astype(np.float32)
        part_feats[:, 6] = d0.astype(np.float32)
        part_feats[:, 7] = z0.astype(np.float32)
        part_feats[:, 8] = phi.astype(np.float32)
        part_feats[:, 9] = theta.astype(np.float32)
        part_feats[:, 10] = qop.astype(np.float32)
        part_feats[:, 11] = is_primary.astype(np.float32)
        all_part_feats.append(part_feats)
        event_part_counts.append(nparts)

        # ---- Hits -----------------------------------------------------
        hx = np.array(htable.column("x")[ev].as_py(), dtype=np.float64)
        hy = np.array(htable.column("y")[ev].as_py(), dtype=np.float64)
        hz = np.array(htable.column("z")[ev].as_py(), dtype=np.float64)
        h_pid = np.array(htable.column("particle_id")[ev].as_py(), dtype=np.int64)
        h_vol = np.array(htable.column("volume_id")[ev].as_py(), dtype=np.int32)
        h_lay = np.array(htable.column("layer_id")[ev].as_py(), dtype=np.int32)
        h_surf = np.array(htable.column("surface_id")[ev].as_py(), dtype=np.int32)
        h_det = np.array(htable.column("detector")[ev].as_py(), dtype=np.int32)

        nhits = len(hx)

        # Derived hit features
        r = np.sqrt(hx**2 + hy**2)
        phi_hit = np.arctan2(hy, hx)
        theta_hit = np.arccos(np.clip(hz / (np.sqrt(hx**2 + hy**2 + hz**2) + 1e-12), -1.0, 1.0))
        s = np.sqrt(hx**2 + hy**2 + hz**2)  # distance from IP

        hit_feats = np.zeros((nhits, N_HIT_FEATURES), dtype=np.float32)
        hit_feats[:, 0] = hx.astype(np.float32)
        hit_feats[:, 1] = hy.astype(np.float32)
        hit_feats[:, 2] = hz.astype(np.float32)
        hit_feats[:, 3] = r.astype(np.float32)
        hit_feats[:, 4] = phi_hit.astype(np.float32)
        hit_feats[:, 5] = theta_hit.astype(np.float32)
        hit_feats[:, 6] = s.astype(np.float32)
        hit_feats[:, 7] = h_vol.astype(np.float32)
        hit_feats[:, 8] = h_lay.astype(np.float32)
        hit_feats[:, 9] = h_surf.astype(np.float32)
        hit_feats[:, 10] = h_det.astype(np.float32)
        all_hit_feats.append(hit_feats)
        event_hit_counts.append(nhits)

        # ---- Track selection ------------------------------------------
        # Count hits per particle
        if nhits > 0:
            unique_pids, counts = np.unique(h_pid, return_counts=True)
            pid_to_nhits = dict(zip(unique_pids.tolist(), counts.tolist()))
        else:
            pid_to_nhits = {}

        nhits_per_particle = np.array(
            [pid_to_nhits.get(int(p), 0) for p in pid],
            dtype=np.int32,
        )

        mask = np.ones(nparts, dtype=bool)
        mask &= nhits_per_particle >= sel["min_hits"]
        if "max_hits" in sel:
            mask &= nhits_per_particle <= sel["max_hits"]
        if sel["primary"]:
            mask &= is_primary
        if sel["hard_scatter"]:
            mask &= vertex_primary == 1
        mask &= pt >= sel["pt_min"]
        mask &= (eta >= sel["eta_min"]) & (eta <= sel["eta_max"])
        mask &= charge != 0  # charged particles only
        # Filter out particles with NaN perigee parameters
        mask &= np.isfinite(d0) & np.isfinite(z0)
        # Perigee range cuts (remove extreme outliers)
        mask &= (d0 >= sel["d0_min"]) & (d0 <= sel["d0_max"])
        mask &= (z0 >= sel["z0_min"]) & (z0 <= sel["z0_max"])

        sel_indices = np.where(mask)[0]

        for si in sel_indices:
            sel_pid = int(pid[si])
            # Find hits belonging to this particle
            hit_mask = h_pid == sel_pid
            track_hit_local = np.where(hit_mask)[0]

            if len(track_hit_local) < sel["min_hits"]:
                continue
            if "max_hits" in sel and len(track_hit_local) > sel["max_hits"]:
                continue

            # Sort track hits by s (distance from IP)
            track_s = s[track_hit_local]
            sort_order = np.argsort(track_s)
            track_hit_local = track_hit_local[sort_order]

            # Global indices
            track_hit_global = track_hit_local + global_hit_offset

            # Targets
            targets = np.array([
                d0[si], z0[si], phi[si], theta[si], qop[si],
            ], dtype=np.float32)

            all_track_targets.append(targets)
            all_track_hit_idx.append(track_hit_global.astype(np.int32))
            all_track_lengths.append(len(track_hit_global))
            all_track_event_idx.append(ev)

        global_hit_offset += nhits

    # ---- Write output -------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = output_dir / "selected_tracks"
    sel_dir.mkdir(exist_ok=True)

    # Concatenate
    if all_hit_feats:
        hits_arr = np.concatenate(all_hit_feats, axis=0)
    else:
        hits_arr = np.zeros((0, N_HIT_FEATURES), dtype=np.float32)

    if all_part_feats:
        parts_arr = np.concatenate(all_part_feats, axis=0)
    else:
        parts_arr = np.zeros((0, N_PARTICLE_FEATURES), dtype=np.float32)

    # Event offsets (CSR-style cumsum)
    event_hit_offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(event_hit_counts, out=event_hit_offsets[1:])
    event_part_offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(event_part_counts, out=event_part_offsets[1:])

    # Selected tracks
    n_selected = len(all_track_targets)
    if n_selected > 0:
        track_targets = np.stack(all_track_targets, axis=0)  # (N, 5)
        track_hit_indices = np.concatenate(all_track_hit_idx)  # flat CSR values
        track_hit_offsets = np.zeros(n_selected + 1, dtype=np.int32)
        np.cumsum(all_track_lengths, out=track_hit_offsets[1:])
        track_event_idx = np.array(all_track_event_idx, dtype=np.int32)
    else:
        track_targets = np.zeros((0, 5), dtype=np.float32)
        track_hit_indices = np.zeros(0, dtype=np.int32)
        track_hit_offsets = np.zeros(1, dtype=np.int32)
        track_event_idx = np.zeros(0, dtype=np.int32)

    # Save
    np.save(output_dir / "hits.npy", hits_arr)
    np.save(output_dir / "particles.npy", parts_arr)
    np.save(output_dir / "event_hit_offsets.npy", event_hit_offsets)
    np.save(output_dir / "event_particle_offsets.npy", event_part_offsets)
    np.save(sel_dir / "track_targets.npy", track_targets)
    np.save(sel_dir / "track_hit_indices.npy", track_hit_indices)
    np.save(sel_dir / "track_hit_offsets.npy", track_hit_offsets)
    np.save(sel_dir / "track_event_idx.npy", track_event_idx)

    return {
        "n_events": n_events,
        "n_hits": len(hits_arr),
        "n_particles": len(parts_arr),
        "n_selected_tracks": n_selected,
        "n_selected_hits": len(track_hit_indices),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Preprocess ColliderML to memmap format")
    parser.add_argument("--data-dir", type=str, default="/scratch/colliderml/p0",
                        help="Root directory containing parquet subdirectories")
    parser.add_argument("--output-dir", type=str, default="/scratch/colliderml/p0_preprocessed",
                        help="Output directory for preprocessed shards")
    parser.add_argument("--num-shards", type=int, default=-1,
                        help="Number of shards to process (-1 for all)")
    parser.add_argument("--selection", type=str, default=None,
                        help="JSON string of selection overrides")
    parser.add_argument("--particles-subdir", type=str,
                        default="ttbar_pu0_particles_recorded_only")
    parser.add_argument("--hits-subdir", type=str,
                        default="ttbar_pu0_tracker_hits")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = dict(DEFAULT_SELECTION)
    if args.selection:
        selection.update(json.loads(args.selection))

    particles_dir = data_dir / args.particles_subdir
    hits_dir = data_dir / args.hits_subdir

    # Support both flat (*.parquet) and nested HuggingFace dataset layouts
    # (e.g. data/<name>/*.parquet)
    particle_files = sorted(particles_dir.glob("*.parquet"))
    if not particle_files:
        particle_files = sorted(particles_dir.rglob("*.parquet"))
    hits_files = sorted(hits_dir.glob("*.parquet"))
    if not hits_files:
        hits_files = sorted(hits_dir.rglob("*.parquet"))

    # Match by name
    pf_by_name = {f.name: f for f in particle_files}
    hf_by_name = {f.name: f for f in hits_files}
    common = sorted(set(pf_by_name) & set(hf_by_name))
    if not common:
        raise FileNotFoundError(
            f"No matching shards in {particles_dir} and {hits_dir}.\n"
            f"  Found {len(particle_files)} particle files, {len(hits_files)} hit files."
        )

    if args.num_shards > 0:
        common = common[: args.num_shards]

    print(f"Processing {len(common)} shards")
    print(f"Selection: {selection}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")

    totals = {
        "n_events": 0,
        "n_hits": 0,
        "n_particles": 0,
        "n_selected_tracks": 0,
        "n_selected_hits": 0,
    }

    # Build job list, skipping already-completed shards
    jobs = []
    n_skipped = 0
    for shard_name in common:
        shard_idx = int(shard_name.split("-")[1])
        shard_out = output_dir / f"shard_{shard_idx:04d}"
        # A shard is complete if its selected_tracks/track_targets.npy exists
        if (shard_out / "selected_tracks" / "track_targets.npy").exists():
            n_skipped += 1
            continue
        jobs.append((pf_by_name[shard_name], hf_by_name[shard_name], shard_out, selection))

    if n_skipped:
        print(f"Skipping {n_skipped} already-processed shards, {len(jobs)} remaining")

    t0 = time.time()

    if args.num_workers <= 1:
        # Sequential (original behaviour)
        for i, (pf, hf, shard_out, sel_cfg) in enumerate(tqdm(jobs, desc="Processing shards")):
            stats = process_shard(pf, hf, shard_out, sel_cfg)
            for k in totals:
                totals[k] += stats[k]
    else:
        # Parallel via ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(process_shard, pf, hf, shard_out, sel_cfg): i
                for i, (pf, hf, shard_out, sel_cfg) in enumerate(jobs)
            }
            with tqdm(total=len(futures), desc="Processing shards") as pbar:
                for future in as_completed(futures):
                    stats = future.result()
                    for k in totals:
                        totals[k] += stats[k]
                    pbar.update(1)

    elapsed = time.time() - t0

    # Write manifest
    manifest = {
        "num_shards": len(common),
        "selection": selection,
        "hit_feature_names": HIT_FEATURE_NAMES,
        "particle_feature_names": PARTICLE_FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "totals": totals,
        "processing_time_s": elapsed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Events:          {totals['n_events']:>12,}")
    print(f"  Hits:            {totals['n_hits']:>12,}")
    print(f"  Particles:       {totals['n_particles']:>12,}")
    print(f"  Selected tracks: {totals['n_selected_tracks']:>12,}")
    print(f"  Selected hits:   {totals['n_selected_hits']:>12,}")
    print(f"  Output:          {output_dir}")
    print(f"  Manifest:        {output_dir / 'manifest.json'}")

    # Auto-create split.json (90/5/5) if it doesn't already exist
    split_path = output_dir / "split.json"
    if not split_path.exists():
        from hepattn.experiments.colliderml_regr.scripts.create_split import create_split

        print(f"\nCreating default train/val/test split (90/5/5)...")
        create_split(preprocessed_dir=output_dir)
    else:
        print(f"\nSplit file already exists: {split_path}")


if __name__ == "__main__":
    main()
