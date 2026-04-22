#!/usr/bin/env python3
"""Pre-compute per-bin truth histograms for class-rebalanced DFL training.

For each parameter, bin the empirical truth distribution from the
preprocessed pretrain data using the **same** u-index scheme the
:class:`BinnedDFLQuantileOffsetLoss` uses at training time.  The result
is a ``(n_bins,)`` torch tensor of integer counts that the loss loads at
init and converts to per-bin CE weights via
``w_bin ∝ 1 / hist[bin]^class_balance_power`` (default sqrt).

Scope
-----
Only ``d0`` is implemented in this first cut — it is the only parameter
in the YOLO line that uses **linear** binning, and therefore the only
one where the 95%/5% core/tail imbalance creates the Bayesian soft-
collapse attractor that class rebalancing is designed to fix.

The other four YOLO parameters (z0, φ, θ, q/p) use **CDF-warped**
binning via :class:`MonotonicSplineTransform`, which by construction
places each bin at a (near-)uniform 1/K probability mass.  For those,
``1/√hist`` is near-constant and class rebalancing would be a
near-no-op.  If/when there's evidence it helps, extending this script
is straightforward — add the binning spec + reproduce ``_target_to_u``
for that parameter's mode.  φ specifically also needs the
``innermost_phi`` anchor extracted per-track from the hit sequence
(computed at collate time in ``data.py``); skip for now.

Usage
-----
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.compute_bin_histograms \
        --param d0 [--num-shards 100]

Output
------
    config/NeurIPS_retraining/v2/core_configs/splines/bin_hist_d0.pt
        torch.LongTensor of shape ``(n_bins,)``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Column layout of the preprocessed ``track_targets.npy``.
TARGET_COLS = {"d0": 0, "z0": 1, "phi": 2, "theta": 3, "qop": 4}

# Per-parameter bin spec — must match the loss config in the YOLO YAMLs.
# Only the linear-binned parameters live here; CDF-binned ones are
# approximately uniform after warping and don't need a rebalance buffer.
BIN_SPEC: dict[str, dict] = {
    "d0": {
        # Full-range uniform linear binning, no overflow.  Covers the
        # entire physics-selected range |d0| <= 2.5 mm with 15.625 μm
        # bin width (same as the previous 66-bin core — CKF-competitive
        # resolution in the center, plus fine-grained bins for the 4%
        # heavy tail beyond ±0.5 mm).
        "binning": "linear",
        "n_bins": 320,
        "range_min": -2.5,
        "range_max": 2.5,
        "n_overflow": 0,
    },
}


def _target_to_u_linear(target: np.ndarray, spec: dict) -> np.ndarray:
    """Reimplementation of ``BinnedDFLQuantileOffsetLoss._target_to_u``
    for the linear case (with or without overflow).  Must stay
    bit-identical to the loss so the buffer matches what the loss will
    see at train time.
    """
    n_bins = spec["n_bins"]
    n_overflow = spec.get("n_overflow", 0)
    lo, hi = spec["range_min"], spec["range_max"]
    if n_overflow == 0:
        # Pure uniform; clamp to [0, 1] like the loss does.
        u = (target - lo) / (hi - lo)
        return np.clip(u, 0.0, 1.0)
    if n_overflow == 2:
        k_core = n_bins - 2
        core_u = (target - lo) / (hi - lo)
        u_mapped = (core_u * k_core + 1.0) / n_bins
        u_low = 0.5 / n_bins
        u_high = (n_bins - 0.5) / n_bins
        u_mapped = np.where(target < lo, u_low, u_mapped)
        u_mapped = np.where(target > hi, u_high, u_mapped)
        return u_mapped
    raise ValueError(f"n_overflow must be 0 or 2, got {n_overflow}")


def _u_to_hard_bin(u: np.ndarray, n_bins: int) -> np.ndarray:
    """Hard-assign each sample to its nearest bin via ``round(u*K - 0.5)``.

    Note this is a *hard* assignment (argmax of the soft target), not
    the 2-bin triangular split DFL uses at training time.  For class
    rebalancing the hard count is the standard DIR/LDS choice and is
    numerically indistinguishable from the soft count at the precision
    we need (both are dominated by the central 2–3 bins).
    """
    continuous = u * n_bins - 0.5
    bin_idx = np.clip(np.round(continuous).astype(np.int64), 0, n_bins - 1)
    return bin_idx


def load_targets(preprocessed_dir: Path, col: int, num_shards: int) -> np.ndarray:
    """Concatenate one parameter's targets across shards."""
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]
    chunks: list[np.ndarray] = []
    for sd in tqdm(shard_dirs, desc="loading targets"):
        tgt_path = sd / "selected_tracks" / "track_targets.npy"
        if not tgt_path.exists():
            continue
        arr = np.load(tgt_path)
        if arr.size == 0:
            continue
        chunks.append(arr[:, col])
    if not chunks:
        raise ValueError(f"No track_targets.npy under {preprocessed_dir}")
    return np.concatenate(chunks).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", type=Path,
                    default=Path("/scratch/colliderml/p0_core_pretrain"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "config/NeurIPS_retraining/v2/core_configs/splines")
    ap.add_argument("--param", choices=list(BIN_SPEC.keys()), default="d0")
    ap.add_argument("--num-shards", type=int, default=-1,
                    help="Number of shards to read; -1 = all.")
    args = ap.parse_args()

    spec = BIN_SPEC[args.param]
    col = TARGET_COLS[args.param]

    targets = load_targets(args.preprocessed_dir, col, args.num_shards)
    print(f"loaded {len(targets):,} {args.param} targets "
          f"(min={targets.min():.4f}, max={targets.max():.4f}, "
          f"std={targets.std():.4f})")

    if spec["binning"] != "linear":
        raise NotImplementedError(
            f"Only linear binning is implemented; got {spec['binning']}"
        )

    u = _target_to_u_linear(targets, spec)
    bin_idx = _u_to_hard_bin(u, spec["n_bins"])
    hist = np.bincount(bin_idx, minlength=spec["n_bins"]).astype(np.int64)

    # Stats for a sanity check before writing out.
    total = hist.sum()
    non_empty = int((hist > 0).sum())
    peak_bin = int(hist.argmax())
    peak_frac = float(hist[peak_bin] / max(total, 1))
    print(f"hist:   min={hist.min()}, max={hist.max()}, mean={hist.mean():.1f}, "
          f"non_empty={non_empty}/{spec['n_bins']}")
    print(f"peak:   bin {peak_bin} holds {hist[peak_bin]:,} tracks ({peak_frac:.1%})")

    # Expected class-balance weight dynamic range (pre-clip):
    with np.errstate(divide="ignore"):
        w_raw = 1.0 / np.maximum(hist.astype(np.float64), 1.0) ** 0.5
    w_raw = w_raw / w_raw.mean()
    print(f"w_bin raw (sqrt-rebalance, pre-clip): "
          f"min={w_raw.min():.4f}, max={w_raw.max():.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"bin_hist_{args.param}.pt"
    torch.save(torch.from_numpy(hist).long(), out_path)
    print(f"wrote {out_path} (shape={tuple(hist.shape)})")


if __name__ == "__main__":
    main()
