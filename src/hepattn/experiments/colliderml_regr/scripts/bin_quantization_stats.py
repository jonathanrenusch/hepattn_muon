#!/usr/bin/env python3
"""Quantization-only residual stats for the YOLO binning scheme.

Answers: "if the classification head perfectly predicts the correct bin for
every sample and we read off the bin *center* as the point prediction
(no within-bin offset correction), what residual stats do we get?"

This is the floor of what the cls head alone can achieve on each parameter —
the offset head's job is to close the gap between this and the physics floor
(beamspot / CKF).  Useful for sanity-checking bin counts and comparing to the
ACTS CKF baseline reported in CLAUDE.md.

Usage:
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.bin_quantization_stats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hepattn.experiments.colliderml_regr.losses import (
    BinnedDFLQuantileOffsetLoss,
    _eta_to_theta,
)


TARGET_COLS = {"d0": 0, "z0": 1, "phi": 2, "theta": 3, "qop": 4}
SPLINES = Path(__file__).resolve().parent.parent / (
    "config/NeurIPS_retraining/v2/core_configs/splines"
)

# Per-parameter loss config — matches yolo_all5_scaled_pretrain.yaml exactly
# (minus the tail_weight / class_balance args, which don't affect binning).
LOSS_CONFIGS = {
    "d0":   dict(n_bins=66, binning="linear",
                 range_min=-0.5, range_max=0.5, n_overflow=2),
    "z0":   dict(n_bins=64, binning="cdf",
                 spline_config=str(SPLINES / "spline_z0.yaml")),
    "phi":  dict(n_bins=64, binning="cdf",
                 spline_config=str(SPLINES / "spline_delta_phi.yaml")),
    "theta": dict(n_bins=64, binning="cdf_eta",
                  spline_config=str(SPLINES / "spline_theta_eta.yaml")),
    "qop":  dict(n_bins=64, binning="cdf",
                 spline_config=str(SPLINES / "spline_qop.yaml")),
}

# Convenience: unit + display scale used in the final report (CLAUDE.md §3 layout).
DISPLAY = {
    "d0":    ("mm",     1.0),
    "z0":    ("mm",     1.0),
    "phi":   ("mrad",   1000.0),
    "theta": ("mrad",   1000.0),
    "qop":   ("1/GeV",  1.0),
}

# CKF baseline on DM subset from CLAUDE.md §"Current results" (IQR/1.349 column)
# Values already in DISPLAY units (mm, mrad, 1/GeV) — do NOT rescale below.
CKF_BASELINE_IQR_DISPLAY = {
    "d0":    0.0599,  # mm
    "z0":    0.113,   # mm
    "phi":   1.877,   # mrad   (already in mrad)
    "theta": 0.700,   # mrad   (already in mrad)
    "qop":   0.00290, # 1/GeV
}


def load_targets(preprocessed_dir: Path, num_shards: int) -> np.ndarray:
    shards = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shards = shards[:num_shards]
    chunks: list[np.ndarray] = []
    for sd in tqdm(shards, desc="load"):
        tgt = sd / "selected_tracks" / "track_targets.npy"
        if tgt.exists():
            arr = np.load(tgt)
            if arr.size > 0:
                chunks.append(arr)
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _physical_bin_center(loss_module: BinnedDFLQuantileOffsetLoss,
                         bin_idx: torch.Tensor) -> torch.Tensor:
    """Physical bin center for a (N,) long tensor of bin indices."""
    if loss_module.binning == "linear":
        return loss_module.phys_centers[bin_idx]
    u_center = (bin_idx.to(torch.float64) + 0.5) / loss_module.n_bins
    if loss_module.binning == "cdf":
        return loss_module.spline.inverse(u_center)
    # cdf_eta: u → eta → theta
    return _eta_to_theta(loss_module.spline.inverse(u_center))


def quantization_residuals(loss_module: BinnedDFLQuantileOffsetLoss,
                           target_phys: np.ndarray) -> np.ndarray:
    """Hard-bin each truth, compute pred = bin center, return residuals (np)."""
    t = torch.as_tensor(target_phys, dtype=torch.float64)
    u = loss_module._target_to_u(t)
    # Match _soft_target's hard-bin convention: round to nearest bin center
    # in index space (center of bin k lives at continuous = k).
    continuous = u * loss_module.n_bins - 0.5
    bin_idx = continuous.round().clamp(0, loss_module.n_bins - 1).long()
    pred = _physical_bin_center(loss_module, bin_idx).to(torch.float64)
    return (pred - t).numpy()


def summarise(residual: np.ndarray, scale: float) -> dict[str, float]:
    r = residual * scale
    q5, q25, q75, q95 = np.percentile(r, [5, 25, 75, 95])
    return {
        "std": float(r.std()),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "iqr_over_1.349": float((q75 - q25) / 1.349),
        "p5_p95_width": float(q95 - q5),
    }


def phi_proxy_stats(loss_module: BinnedDFLQuantileOffsetLoss,
                    n_samples: int,
                    rng: np.random.Generator) -> dict[str, float]:
    """φ needs delta_phi = phi_truth − innermost_phi.  innermost_phi is a hit
    feature, not in track_targets.npy, so we can't compute it here without
    reading the whole hit stream.  Instead we use the fact that the spline
    was fit so that ``u_delta = spline.forward(delta_phi)`` is approximately
    uniform — and read the quantization statistics off a uniform ``u`` sample.
    """
    u = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float64)
    delta_phi_truth = loss_module.spline.inverse(u).to(torch.float64)
    # Hard-bin in u-space
    bin_idx = (u * loss_module.n_bins - 0.5).round().clamp(
        0, loss_module.n_bins - 1).long()
    pred = _physical_bin_center(loss_module, bin_idx).to(torch.float64)
    residual = (pred - delta_phi_truth).numpy()
    return summarise(residual, DISPLAY["phi"][1])


def main():
    preprocessed_dir = Path("/scratch/colliderml/p0_core_pretrain")
    num_shards = 50  # ~3.6 M tracks, plenty for stats
    rng = np.random.default_rng(0)

    targets = load_targets(preprocessed_dir, num_shards)
    print(f"loaded {len(targets):,} tracks from {num_shards} shards")

    results: dict[str, dict[str, float]] = {}
    for name, cfg in LOSS_CONFIGS.items():
        loss_module = BinnedDFLQuantileOffsetLoss(**cfg).double()
        _, scale = DISPLAY[name]

        if name == "phi":
            results[name] = phi_proxy_stats(loss_module, len(targets), rng)
            continue

        col = TARGET_COLS[name]
        vals = targets[:, col]
        residual = quantization_residuals(loss_module, vals)
        results[name] = summarise(residual, scale)

    # Also compute the idealised per-bin-uniform RMS (for CDF this is
    # bin_width/(2√3) in u-space, projected through the local spline slope
    # to physical).  We report the empirical above; this is the "if truth
    # were uniformly distributed within each bin" floor which is tighter
    # than the real-distribution number only for the linear case.

    print()
    unit_col = "unit"
    hdr = f"{'param':<7} {unit_col:<7} {'std':>12} {'rms':>12} {'IQR/1.349':>12} {'p5-p95':>12}  {'CKF IQR':>10} {'ratio':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        unit, _ = DISPLAY[name]
        ckf = CKF_BASELINE_IQR_DISPLAY[name]
        ratio = r["iqr_over_1.349"] / ckf if ckf > 0 else float("nan")
        print(f"{name:<7} {unit:<7} "
              f"{r['std']:>12.4g} {r['rms']:>12.4g} {r['iqr_over_1.349']:>12.4g} "
              f"{r['p5_p95_width']:>12.4g}  {ckf:>10.4g} {ratio:>6.2f}x")


if __name__ == "__main__":
    main()
