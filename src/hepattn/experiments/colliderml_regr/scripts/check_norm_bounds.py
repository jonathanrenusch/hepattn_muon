#!/usr/bin/env python3
"""
Check whether the input-feature and target normalisation bounds hard-coded
in the v2 base configs are still consistent with the actual data on disk.

Scans a configurable number of shards from each dataset, computes per-column
min/max for both hit features and track targets, and compares against the
bounds declared in:
  - base.yaml  norm_min / norm_max  (input features, maps to [0,1])
  - loss configs  norm_min / norm_max  (targets, maps to [-1,1])

Usage:
    python check_norm_bounds.py                    # defaults: 50 shards per dataset
    python check_norm_bounds.py --max-shards 200   # scan more shards
    python check_norm_bounds.py --all               # scan ALL shards (slow)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── datasets to check ────────────────────────────────────────────────────────
DATASETS: dict[str, str] = {
    "p0_loose_pretrain":  "/scratch/colliderml/p0_loose_pretrain",
    "p200_loose_finetune": "/scratch/colliderml/p200_loose_finetune",
    # Core datasets (may not exist yet — the script will skip them)
    "p0_core_pretrain":   "/scratch/colliderml/p0_core_pretrain",
    "p200_core_finetune": "/scratch/colliderml/p200_core_finetune",
    "p200_core_kf_matched_finetune": "/scratch/colliderml/p200_core_kf_matched_finetune",
    "p200_core_kf_hits_finetune":    "/scratch/colliderml/p200_core_kf_hits_finetune",
}

# ── config bounds (from v2/base.yaml) ────────────────────────────────────────
# 12 features: x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id,
#              surface_id, detector, eta_hit
# First 11 are stored in hits.npy; eta_hit is derived from theta_hit at load time.
INPUT_FIELDS = [
    "x", "y", "z", "r", "phi_hit", "theta_hit",
    "s", "volume_id", "layer_id", "surface_id", "detector", "eta_hit",
]
CONFIG_NORM_MIN = np.array(
    [-1031.0, -1031.0, -3026.0, 31.0, -3.1416, 0.027,
     31.0, 16.0, 2.0, 1.0, 0.0, -4.3],
    dtype=np.float64,
)
CONFIG_NORM_MAX = np.array(
    [1031.0, 1031.0, 3026.0, 1032.0, 3.1416, 3.114,
     3185.0, 30.0, 16.0, 3360.0, 8.0, 4.3],
    dtype=np.float64,
)

TARGET_FIELDS = ["d0", "z0", "phi", "theta", "qop"]

# Target bounds per dataset variant (from the loss configs).
# loose configs use wider d0/qop; core configs use tighter cuts.
TARGET_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "loose": {
        "d0":    (-5.0,   5.0),
        "z0":    (-200.0, 200.0),
        "phi":   (None,   None),     # circular loss, no linear norm
        "theta": (-3.0,   3.0),      # normalised in eta-space
        "qop":   (-5.0,   5.0),
    },
    "core": {
        "d0":    (-2.5,   2.5),
        "z0":    (-200.0, 200.0),
        "phi":   (None,   None),
        "theta": (-3.0,   3.0),
        "qop":   (-2.0,   2.0),
    },
}


def theta_to_eta(theta: np.ndarray) -> np.ndarray:
    """Convert polar angle theta to pseudorapidity eta."""
    theta_safe = np.clip(theta, 1e-8, np.pi - 1e-8)
    return -np.log(np.tan(theta_safe / 2.0))


def scan_dataset(
    dataset_path: str,
    max_shards: int | None,
) -> dict | None:
    """Scan shards and return per-column min/max for hits and targets."""
    dpath = Path(dataset_path)
    if not dpath.exists():
        return None

    shard_dirs = sorted(dpath.glob("shard_*"))
    if not shard_dirs:
        return None

    if max_shards is not None:
        # Sample evenly across the dataset
        step = max(1, len(shard_dirs) // max_shards)
        shard_dirs = shard_dirs[::step][:max_shards]

    n_stored_feats = 11  # features stored in hits.npy (without eta_hit)

    hit_global_min = np.full(n_stored_feats, np.inf, dtype=np.float64)
    hit_global_max = np.full(n_stored_feats, -np.inf, dtype=np.float64)
    tgt_global_min = np.full(5, np.inf, dtype=np.float64)
    tgt_global_max = np.full(5, -np.inf, dtype=np.float64)
    total_hits = 0
    total_tracks = 0

    for shard in shard_dirs:
        hits_file = shard / "hits.npy"
        targets_file = shard / "selected_tracks" / "track_targets.npy"

        if not hits_file.exists() or not targets_file.exists():
            continue

        hits = np.load(hits_file).astype(np.float64)
        targets = np.load(targets_file).astype(np.float64)

        if hits.size == 0 or targets.size == 0:
            continue

        hit_global_min = np.minimum(hit_global_min, hits.min(axis=0))
        hit_global_max = np.maximum(hit_global_max, hits.max(axis=0))
        tgt_global_min = np.minimum(tgt_global_min, targets.min(axis=0))
        tgt_global_max = np.maximum(tgt_global_max, targets.max(axis=0))

        total_hits += hits.shape[0]
        total_tracks += targets.shape[0]

    if total_hits == 0:
        return None

    # Derive eta_hit bounds from theta_hit bounds (col 5)
    theta_min_val = hit_global_min[5]
    theta_max_val = hit_global_max[5]
    # eta is monotonically decreasing in theta, so:
    eta_from_theta_max = theta_to_eta(theta_min_val)  # small theta -> large eta
    eta_from_theta_min = theta_to_eta(theta_max_val)  # large theta -> small eta
    # Clip as the data loader does
    eta_min = np.clip(eta_from_theta_min, -10.0, 10.0)
    eta_max = np.clip(eta_from_theta_max, -10.0, 10.0)

    # Also convert track-level theta target to eta for comparison
    tgt_eta_min = theta_to_eta(tgt_global_max[3])  # theta max -> eta min
    tgt_eta_max = theta_to_eta(tgt_global_min[3])  # theta min -> eta max

    return {
        "hit_min": hit_global_min,
        "hit_max": hit_global_max,
        "eta_hit_min": eta_min,
        "eta_hit_max": eta_max,
        "tgt_min": tgt_global_min,
        "tgt_max": tgt_global_max,
        "tgt_eta_min": tgt_eta_min,
        "tgt_eta_max": tgt_eta_max,
        "total_hits": total_hits,
        "total_tracks": total_tracks,
        "n_shards_scanned": len(shard_dirs),
    }


def print_header(title: str) -> None:
    w = 80
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def check_bounds(
    field: str,
    data_min: float,
    data_max: float,
    cfg_min: float,
    cfg_max: float,
) -> list[str]:
    """Return list of warning strings (empty if OK)."""
    warnings = []
    margin_lo = (data_min - cfg_min) / max(abs(cfg_max - cfg_min), 1e-12)
    margin_hi = (cfg_max - data_max) / max(abs(cfg_max - cfg_min), 1e-12)

    if data_min < cfg_min:
        warnings.append(
            f"  !! {field:>12s}: data min {data_min:+12.4f} < config min {cfg_min:+12.4f}  "
            f"(CLIPPED — {abs(margin_lo)*100:.1f}% below range)"
        )
    if data_max > cfg_max:
        warnings.append(
            f"  !! {field:>12s}: data max {data_max:+12.4f} > config max {cfg_max:+12.4f}  "
            f"(CLIPPED — {abs(margin_hi)*100:.1f}% above range)"
        )
    # Warn if margin is very tight (< 1%)
    if data_min >= cfg_min and margin_lo < 0.01 and margin_lo >= 0:
        warnings.append(
            f"  ~~ {field:>12s}: data min {data_min:+12.4f} very close to config min {cfg_min:+12.4f}  "
            f"(margin {margin_lo*100:.2f}%)"
        )
    if data_max <= cfg_max and margin_hi < 0.01 and margin_hi >= 0:
        warnings.append(
            f"  ~~ {field:>12s}: data max {data_max:+12.4f} very close to config max {cfg_max:+12.4f}  "
            f"(margin {margin_hi*100:.2f}%)"
        )
    # Warn if bounds are very loose (> 50% wasted range)
    if margin_lo > 0.5:
        warnings.append(
            f"  ?? {field:>12s}: data min {data_min:+12.4f} far from config min {cfg_min:+12.4f}  "
            f"(margin {margin_lo*100:.1f}% — wasting resolution)"
        )
    if margin_hi > 0.5:
        warnings.append(
            f"  ?? {field:>12s}: data max {data_max:+12.4f} far from config max {cfg_max:+12.4f}  "
            f"(margin {margin_hi*100:.1f}% — wasting resolution)"
        )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shards", type=int, default=50,
                        help="Max shards to scan per dataset (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Scan all shards (overrides --max-shards)")
    args = parser.parse_args()
    max_shards = None if args.all else args.max_shards

    any_issues = False

    for ds_name, ds_path in DATASETS.items():
        print_header(f"Dataset: {ds_name}  ({ds_path})")

        result = scan_dataset(ds_path, max_shards)
        if result is None:
            print(f"  SKIPPED — directory missing or empty")
            continue

        print(f"  Scanned {result['n_shards_scanned']} shards  "
              f"| {result['total_hits']:,} hits  "
              f"| {result['total_tracks']:,} tracks")

        # ── Input feature bounds ─────────────────────────────────────────
        print()
        print(f"  {'FEATURE':>12s}  {'data min':>14s}  {'cfg min':>14s}  "
              f"{'data max':>14s}  {'cfg max':>14s}  STATUS")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}")

        all_warnings: list[str] = []

        for i, field in enumerate(INPUT_FIELDS):
            if field == "eta_hit":
                d_min = result["eta_hit_min"]
                d_max = result["eta_hit_max"]
            else:
                d_min = result["hit_min"][i]
                d_max = result["hit_max"][i]
            c_min = CONFIG_NORM_MIN[i]
            c_max = CONFIG_NORM_MAX[i]

            ok = d_min >= c_min and d_max <= c_max
            status = "OK" if ok else "!! OUT OF RANGE"
            print(f"  {field:>12s}  {d_min:+14.4f}  {c_min:+14.4f}  "
                  f"{d_max:+14.4f}  {c_max:+14.4f}  {status}")

            ws = check_bounds(field, d_min, d_max, c_min, c_max)
            all_warnings.extend(ws)

        # ── Target bounds ────────────────────────────────────────────────
        variant = "core" if "core" in ds_name else "loose"
        tgt_bounds = TARGET_BOUNDS[variant]

        print()
        print(f"  Target bounds (variant: {variant}):")
        print(f"  {'TARGET':>12s}  {'data min':>14s}  {'cfg min':>14s}  "
              f"{'data max':>14s}  {'cfg max':>14s}  STATUS")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}")

        for i, field in enumerate(TARGET_FIELDS):
            c_min_t, c_max_t = tgt_bounds[field]

            if field == "phi":
                # Circular loss — no linear normalisation, just report range
                d_min_t = result["tgt_min"][i]
                d_max_t = result["tgt_max"][i]
                print(f"  {field:>12s}  {d_min_t:+14.4f}  {'(circular)':>14s}  "
                      f"{d_max_t:+14.4f}  {'(circular)':>14s}  N/A")
                continue

            if field == "theta":
                # Loss normalises in eta-space
                d_min_t = result["tgt_eta_min"]
                d_max_t = result["tgt_eta_max"]
                print(f"  {'theta(η)':>12s}  {d_min_t:+14.4f}  {c_min_t:+14.4f}  "
                      f"{d_max_t:+14.4f}  {c_max_t:+14.4f}  ", end="")
            else:
                d_min_t = result["tgt_min"][i]
                d_max_t = result["tgt_max"][i]
                print(f"  {field:>12s}  {d_min_t:+14.4f}  {c_min_t:+14.4f}  "
                      f"{d_max_t:+14.4f}  {c_max_t:+14.4f}  ", end="")

            ok = d_min_t >= c_min_t and d_max_t <= c_max_t
            status = "OK" if ok else "!! OUT OF RANGE"
            print(status)

            ws = check_bounds(field if field != "theta" else "theta(η)",
                              d_min_t, d_max_t, c_min_t, c_max_t)
            all_warnings.extend(ws)

            # For targets that are out-of-range, show the normalised value
            if not ok:
                norm_min_val = 2.0 * (d_min_t - c_min_t) / (c_max_t - c_min_t) - 1.0
                norm_max_val = 2.0 * (d_max_t - c_min_t) / (c_max_t - c_min_t) - 1.0
                all_warnings.append(
                    f"       {field:>12s}: normalised range [{norm_min_val:+.4f}, {norm_max_val:+.4f}] "
                    f"(should be [-1, +1])"
                )

        if all_warnings:
            any_issues = True
            print()
            print("  WARNINGS:")
            for w in all_warnings:
                print(w)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_header("SUMMARY")
    if any_issues:
        print("  Some bounds are tight, wasteful, or out-of-range — see warnings above.")
    else:
        print("  All normalisation bounds are consistent with the data on disk.")
    print()


if __name__ == "__main__":
    main()
