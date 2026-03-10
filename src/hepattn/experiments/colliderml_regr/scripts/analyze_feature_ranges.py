#!/usr/bin/env python3
"""Analyze min/max ranges of all hit features for normalization.

Reads the preprocessed memmap data to compute per-feature statistics
needed for min-max normalization of input features.

Usage::

    python analyze_feature_ranges.py --preprocessed-dir /scratch/colliderml/p0_preprocessed_test --num-shards 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


# Feature names matching the preprocessing output (N_HIT_FEATURES=11)
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector",
]

# We also compute derived features
DERIVED_FEATURE_NAMES = ["eta_hit"]

TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]


def main():
    parser = argparse.ArgumentParser(description="Analyze feature ranges")
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None,
                        help="Output YAML file (default: config/normalisation_ranges.yaml)")
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if args.num_shards > 0:
        shard_dirs = shard_dirs[: args.num_shards]

    print(f"Analyzing {len(shard_dirs)} shards from {preprocessed_dir}")

    # Track running min/max/sum/sum2/count for each feature
    n_raw = len(HIT_FEATURE_NAMES)
    feature_min = np.full(n_raw, np.inf)
    feature_max = np.full(n_raw, -np.inf)
    feature_sum = np.zeros(n_raw, dtype=np.float64)
    feature_sum2 = np.zeros(n_raw, dtype=np.float64)
    total_hits = 0

    # Also for derived eta
    eta_min = np.inf
    eta_max = -np.inf
    eta_sum = 0.0
    eta_sum2 = 0.0

    # Also track per-feature unique count for categorical features
    vol_ids = set()
    lay_ids = set()
    surf_ids = set()
    det_ids = set()

    for shard_dir in shard_dirs:
        hits_path = shard_dir / "hits.npy"
        if not hits_path.exists():
            continue

        hits = np.load(hits_path, mmap_mode="r")  # (N, n_features)
        n = hits.shape[0]
        n_feat = min(hits.shape[1], n_raw)

        total_hits += n

        for f in range(n_feat):
            col = np.array(hits[:, f], dtype=np.float64)
            feature_min[f] = min(feature_min[f], col.min())
            feature_max[f] = max(feature_max[f], col.max())
            feature_sum[f] += col.sum()
            feature_sum2[f] += (col ** 2).sum()

        # Compute eta from theta_hit (col 5)
        theta_hit = np.array(hits[:, 5], dtype=np.float64)
        eta_hit = -np.log(np.tan(theta_hit / 2.0 + 1e-12))
        eta_hit = np.clip(eta_hit, -10, 10)  # clip extreme values
        eta_min = min(eta_min, eta_hit.min())
        eta_max = max(eta_max, eta_hit.max())
        eta_sum += eta_hit.sum()
        eta_sum2 += (eta_hit ** 2).sum()

        # Categorical unique values (only if we have >= 11 features)
        if n_feat >= 8:
            vol_ids.update(np.unique(hits[:, 7]).astype(int).tolist())
        if n_feat >= 9:
            lay_ids.update(np.unique(hits[:, 8]).astype(int).tolist())
        if n_feat >= 10:
            surf_ids.update(np.unique(hits[:, 9]).astype(int).tolist())
        if n_feat >= 11:
            det_ids.update(np.unique(hits[:, 10]).astype(int).tolist())

    # Compute means and stds
    feature_mean = feature_sum / total_hits
    feature_std = np.sqrt(feature_sum2 / total_hits - feature_mean ** 2)

    eta_mean = eta_sum / total_hits
    eta_std = np.sqrt(eta_sum2 / total_hits - eta_mean ** 2)

    # Print results
    print(f"\nTotal hits analyzed: {total_hits:,}")
    print(f"\n{'Feature':<15} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 67)
    for i, name in enumerate(HIT_FEATURE_NAMES[:min(n_raw, n_feat)]):
        print(f"{name:<15} {feature_min[i]:>12.4f} {feature_max[i]:>12.4f} "
              f"{feature_mean[i]:>12.4f} {feature_std[i]:>12.4f}")
    print(f"{'eta_hit':<15} {eta_min:>12.4f} {eta_max:>12.4f} "
          f"{eta_mean:>12.4f} {eta_std:>12.4f}")

    print(f"\nCategorical features:")
    print(f"  volume_id  unique values ({len(vol_ids)}): {sorted(vol_ids)}")
    print(f"  layer_id   unique values ({len(lay_ids)}): {sorted(lay_ids)}")
    print(f"  surface_id unique values ({len(surf_ids)}): {sorted(surf_ids)[:20]}{'...' if len(surf_ids) > 20 else ''}")
    print(f"  detector   unique values ({len(det_ids)}): {sorted(det_ids)}")

    # Build normalisation config
    # For normalization, we use the physical ranges for continuous features
    # and identity (no norm) for categorical/angular features
    norm_config = {}
    for i, name in enumerate(HIT_FEATURE_NAMES[:min(n_raw, n_feat)]):
        norm_config[name] = {
            "min": float(feature_min[i]),
            "max": float(feature_max[i]),
            "mean": float(feature_mean[i]),
            "std": float(feature_std[i]),
        }
    norm_config["eta_hit"] = {
        "min": float(eta_min),
        "max": float(eta_max),
        "mean": float(eta_mean),
        "std": float(eta_std),
    }

    # Also compute target ranges (from selected tracks)
    print("\n--- Target parameter ranges ---")
    target_all = {t: [] for t in TARGET_NAMES}
    for shard_dir in shard_dirs:
        targets_path = shard_dir / "selected_tracks" / "track_targets.npy"
        if not targets_path.exists():
            continue
        targets = np.load(targets_path, mmap_mode="r")
        for i, name in enumerate(TARGET_NAMES):
            target_all[name].append(np.array(targets[:, i], dtype=np.float64))

    for name in TARGET_NAMES:
        vals = np.concatenate(target_all[name])
        norm_config[f"target_{name}"] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
        print(f"  {name:<8} min={vals.min():.6f}  max={vals.max():.6f}  "
              f"mean={vals.mean():.6f}  std={vals.std():.6f}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).resolve().parent.parent / "config" / "normalisation_ranges.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        yaml.dump(norm_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved normalisation ranges to {out_path}")


if __name__ == "__main__":
    main()
