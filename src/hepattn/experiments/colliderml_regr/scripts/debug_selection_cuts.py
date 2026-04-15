"""Standalone: report selection-cut drop rates on the full testing set.

Applies each cut from selection_defaults.yaml (with hard_scatter=false) and
reports (a) per-cut independent drop %, (b) sequential (cumulative) drop %,
independent of any DM filter. Writes a .txt summary.

Usage:
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.debug_selection_cuts \\
        --predictions <path/to/test_predictions.h5> \\
        --data-dir <preprocessed dataset root> \\
        --output <path/to/summary.txt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_targets(pred_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(pred_path, "r") as f:
        return {name: f["targets"][name][:] for name in f["targets"]}


def load_nhits(data_dir: Path, split: str = "test") -> np.ndarray | None:
    split_file = data_dir / "split.json"
    if not split_file.exists():
        return None
    with open(split_file) as f:
        shard_indices = sorted(json.load(f).get(split, []))
    if not shard_indices:
        return None
    chunks = []
    for idx in tqdm(shard_indices, desc="Loading nhits", file=sys.stderr):
        offsets = np.load(data_dir / f"shard_{idx:04d}" / "selected_tracks" / "track_hit_offsets.npy")
        chunks.append(np.diff(offsets).astype(np.int32))
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    targets = load_targets(Path(args.predictions))
    n_total = len(targets["d0"])
    print(f"Loaded {n_total:,} testing tracks from predictions file")

    nhits = load_nhits(Path(args.data_dir), args.split)
    if nhits is None:
        sys.exit("Could not load nhits from data-dir")
    if len(nhits) != n_total:
        n_min = min(len(nhits), n_total)
        print(f"WARNING: nhits has {len(nhits):,}, targets has {n_total:,} — truncating to {n_min:,}")
        nhits = nhits[:n_min]
        for k in targets:
            targets[k] = targets[k][:n_min]
        n_total = n_min

    theta = targets["theta"]
    qop = targets["qop"]
    eta = -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))
    pt = np.abs(np.sin(theta) / (qop + 1e-12))

    # selection_defaults.yaml with hard_scatter=false
    cuts = [
        ("pt_min >= 0.5 GeV",     pt >= 0.5),
        ("eta_min >= -3.0",       eta >= -3.0),
        ("eta_max <=  3.0",       eta <=  3.0),
        ("min_hits >= 6",         nhits >= 6),
        ("max_hits <= 20",        nhits <= 20),
        ("d0_min >= -1 mm",       targets["d0"] >= -1.0),
        ("d0_max <=  1 mm",       targets["d0"] <=  1.0),
        ("z0_min >= -150 mm",     targets["z0"] >= -150.0),
        ("z0_max <=  150 mm",     targets["z0"] <=  150.0),
    ]

    lines: list[str] = []
    lines.append("Selection-cut debug — applied independently of DM filter")
    lines.append(f"Predictions:  {args.predictions}")
    lines.append(f"Data dir:     {args.data_dir}  (split={args.split})")
    lines.append(f"Total tracks: {n_total:,}")
    lines.append("")
    lines.append("Upstream (already applied at preprocessing):")
    lines.append("  primary=True, charged=True, hard_scatter=False (no cut),")
    lines.append("  min_hits=3, max_hits=20, pt_min=0.2, eta∈[-3,3], |d0|<=5, |z0|<=200")
    lines.append("")

    lines.append("Independent drop rates (each cut applied in isolation):")
    lines.append(f"  {'Cut':<22}  {'dropped':>12}  {'% of total':>12}")
    for label, mask in cuts:
        dropped = int(n_total - np.sum(mask))
        pct = 100.0 * dropped / n_total
        lines.append(f"  {label:<22}  {dropped:>12,}  {pct:>11.3f}%")
    lines.append("")

    lines.append("Sequential drop rates (cuts applied in order):")
    lines.append(f"  {'Cut':<22}  {'kept':>12}  {'dropped':>12}  {'% of input':>11}  {'% of total':>11}")
    running = np.ones(n_total, dtype=bool)
    for label, mask in cuts:
        before = int(np.sum(running))
        running &= mask
        after = int(np.sum(running))
        dropped = before - after
        pct_input = 100.0 * dropped / max(before, 1)
        pct_total = 100.0 * dropped / n_total
        lines.append(f"  {label:<22}  {after:>12,}  {dropped:>12,}  {pct_input:>10.3f}%  {pct_total:>10.3f}%")

    n_final = int(np.sum(running))
    final_pct = 100.0 * n_final / n_total
    lines.append("")
    lines.append(f"FINAL after all selection cuts: {n_final:,} / {n_total:,}  ({final_pct:.3f}% retained)")
    lines.append(f"TOTAL DROPPED: {n_total - n_final:,}  ({100.0 - final_pct:.3f}% of all tracks)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
