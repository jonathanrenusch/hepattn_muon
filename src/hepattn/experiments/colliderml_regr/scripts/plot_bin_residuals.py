#!/usr/bin/env python3
"""Per-bin residual distributions assuming perfect classification.

For each of the 5 track parameters, bin every truth to its nearest bin
center using the **proposed scaled** bin counts (d0=66 linear, z0=512,
phi=32, theta=1024, qop=256 — all CDF-warped except d0), then plot the
residual *in bin-width units* that the offset head would have to regress
if classification were perfect.  Useful for judging whether 7-quantile
pinball is a good loss for this target (uniform ±0.5 → flat quantile
spacing is ideal; skewed → median moves off zero; heavy tails → pinball
robustness matters).

Outputs (under ``logs/yolo_bin_residuals/``):
    - ``bin_residuals_overview.png``: 5-panel aggregate residual histograms
      (all tracks pooled, one panel per parameter).
    - ``bin_residuals_per_bin_<param>.png``: 3x3 grid showing residuals for
      9 representative bins (lowest / 1st quartile / mode / ... / highest).
      Reveals whether the shape is consistent across bins or bin-dependent.
    - ``bin_residuals_physical_<param>.png``: same as aggregate but residual
      converted back to physical units for intuition on actual precision.

Assumptions:
    - Perfect classification: truth assigned to its nearest bin's index.
      Residual = (u_truth * K - 0.5) - round(u_truth * K - 0.5), always
      in [-0.5, +0.5] by construction.
    - phi uses the u~U[0,1] proxy for delta_phi (same as
      bin_quantization_stats.py).  Not exact for phi because the hits
      aren't read, but quantitatively close.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hepattn.experiments.colliderml_regr.losses import (
    BinnedDFLQuantileOffsetLoss,
    _eta_to_theta,
)


TARGET_COLS = {"d0": 0, "z0": 1, "phi": 2, "theta": 3, "qop": 4}
SPLINES = Path(__file__).resolve().parent.parent / (
    "config/NeurIPS_retraining/v2/core_configs/splines"
)

# Proposed scaled-config bin counts (matches the bin-count revision we
# discussed: anchor cls-only floor at the SSM IQR, slightly below for
# z0 / phi / qop, slightly above for theta with offset-head closing).
LOSS_CONFIGS = {
    "d0":   dict(n_bins=320,  binning="linear",
                 range_min=-2.5, range_max=2.5, n_overflow=0),
    "z0":   dict(n_bins=512,  binning="cdf",
                 spline_config=str(SPLINES / "spline_z0.yaml")),
    "phi":  dict(n_bins=32,   binning="cdf",
                 spline_config=str(SPLINES / "spline_delta_phi.yaml")),
    "theta": dict(n_bins=1024, binning="cdf_eta",
                  spline_config=str(SPLINES / "spline_theta_eta.yaml")),
    "qop":  dict(n_bins=256,  binning="cdf",
                 spline_config=str(SPLINES / "spline_qop.yaml")),
}

DISPLAY = {
    "d0":    ("mm",    1.0),
    "z0":    ("mm",    1.0),
    "phi":   ("mrad",  1000.0),
    "theta": ("mrad",  1000.0),
    "qop":   ("1/GeV", 1.0),
}

NUM_SHARDS = 50  # ~3.5 M tracks


def load_targets(preprocessed_dir: Path, num_shards: int) -> np.ndarray:
    shards = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shards = shards[:num_shards]
    chunks = []
    for sd in tqdm(shards, desc="load"):
        tgt = sd / "selected_tracks" / "track_targets.npy"
        if tgt.exists():
            a = np.load(tgt)
            if a.size > 0:
                chunks.append(a)
    return np.concatenate(chunks, axis=0).astype(np.float64)


def compute_bin_and_residual(loss: BinnedDFLQuantileOffsetLoss,
                             target_phys: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_idx, residual_bin_widths, residual_physical)."""
    t = torch.as_tensor(target_phys, dtype=torch.float64)
    u = loss._target_to_u(t)
    continuous = u * loss.n_bins - 0.5
    bin_idx = continuous.round().clamp(0, loss.n_bins - 1).long()
    residual_bin_widths = (continuous - bin_idx.to(continuous.dtype)).numpy()

    # Physical-space residual:  pred_physical - truth
    if loss.binning == "linear":
        pred_phys = loss.phys_centers[bin_idx].to(torch.float64)
    else:
        u_center = (bin_idx.to(torch.float64) + 0.5) / loss.n_bins
        if loss.binning == "cdf":
            pred_phys = loss.spline.inverse(u_center)
        else:
            pred_phys = _eta_to_theta(loss.spline.inverse(u_center))
    residual_phys = (pred_phys - t).numpy()

    return bin_idx.numpy(), residual_bin_widths, residual_phys


def phi_proxy(loss: BinnedDFLQuantileOffsetLoss,
              n_samples: int,
              rng: np.random.Generator
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use u ~ U[0,1], invert spline → synthetic delta_phi."""
    u = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float64)
    delta_phi = loss.spline.inverse(u)
    return compute_bin_and_residual(loss, delta_phi.numpy())


def plot_aggregate(residuals: dict[str, np.ndarray],
                   out_dir: Path) -> None:
    """5-panel overview: residuals pooled across all bins, per parameter."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, res) in zip(axes, residuals.items()):
        ax.hist(res, bins=100, density=True, color="steelblue",
                edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="r", ls="--", lw=1, label="bin center")
        # Overlay the theoretical uniform density 1 on [-0.5, 0.5] (DFL's implicit
        # prior for the offset target under perfect CDF warping).
        ax.axhline(1.0, color="orange", ls=":", lw=1, label="Uniform(-½,½)")
        ax.set_xlabel(f"{name}: residual [bin widths]")
        ax.set_ylabel("density")
        ax.set_xlim(-0.55, 0.55)
        q25, med, q75 = np.percentile(res, [25, 50, 75])
        ax.set_title(f"{name}  K={LOSS_CONFIGS[name]['n_bins']}\n"
                     f"med={med:+.3f}  IQR=[{q25:+.3f},{q75:+.3f}]",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Offset-head target distribution (residual | perfect "
                 "classification) — pooled over all bins", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "bin_residuals_overview.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_per_bin_slice(name: str,
                       bin_idx: np.ndarray,
                       residual: np.ndarray,
                       K: int,
                       out_dir: Path) -> None:
    """3x3 grid: residuals for 9 bins spanning low-to-high bin index."""
    # Pick 9 bins including the two modes (central in linear, central u-bin
    # in CDF) and a spread of bin indices to see shape variation.
    bins_to_show = np.unique(np.round(
        np.linspace(0, K - 1, 9)
    ).astype(int))
    # For d0 specifically the two central bins 32, 33 are the "prompt"
    # bins — force them into the selection.
    if name == "d0":
        bins_to_show = np.unique(np.concatenate(
            [bins_to_show[:3], [32, 33], bins_to_show[-4:]]
        ))[:9]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle(f"{name}: per-bin offset target (residual | perfect cls)",
                 fontsize=13, fontweight="bold")

    for ax, k in zip(axes.flat, bins_to_show):
        sel = bin_idx == k
        n_in_bin = sel.sum()
        if n_in_bin < 20:
            ax.text(0.5, 0.5, f"bin {k}\nN={n_in_bin}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="grey")
            ax.set_xlim(-0.55, 0.55)
            continue
        r = residual[sel]
        ax.hist(r, bins=50, density=True, color="steelblue",
                edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.axhline(1.0, color="orange", ls=":", lw=1)
        q25, med, q75 = np.percentile(r, [25, 50, 75])
        ax.set_title(f"bin {k} (N={n_in_bin:,})\n"
                     f"med={med:+.3f}  IQR=[{q25:+.3f},{q75:+.3f}]",
                     fontsize=9)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xlabel("residual [bin widths]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"bin_residuals_per_bin_{name}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_physical(residuals_phys: dict[str, np.ndarray],
                  out_dir: Path) -> None:
    """5-panel residuals in *physical* units on linear AND log y scale."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for col, (name, res) in enumerate(residuals_phys.items()):
        unit, scale = DISPLAY[name]
        r = res * scale
        q1, q99 = np.percentile(r, [1, 99])
        bins = np.linspace(q1, q99, 120)
        for row, ylog in enumerate([False, True]):
            ax = axes[row, col]
            ax.hist(r, bins=bins, color="steelblue",
                    edgecolor="black", linewidth=0.3, alpha=0.85)
            ax.axvline(0, color="r", ls="--", lw=1)
            ax.set_xlabel(f"{name} residual [{unit}]")
            if row == 0:
                ax.set_ylabel("count")
                ax.set_title(
                    f"{name}  K={LOSS_CONFIGS[name]['n_bins']}"
                    f"  std={r.std():.3g}  "
                    f"IQR/1.349={np.subtract(*np.percentile(r, [75, 25]))/1.349:.3g}",
                    fontsize=10,
                )
            else:
                ax.set_yscale("log")
                ax.set_ylabel("count (log)")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Physical-unit residual  (bin_center − truth)  under perfect "
                 "classification — linear (top) and log y (bottom)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "bin_residuals_physical.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def main():
    preprocessed_dir = Path("/scratch/colliderml/p0_core_pretrain")
    out_dir = Path("/shared/tracking/logs/yolo_bin_residuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"Loading targets from {preprocessed_dir} …")
    targets = load_targets(preprocessed_dir, NUM_SHARDS)
    print(f"Loaded {len(targets):,} tracks")

    residuals: dict[str, np.ndarray] = {}
    residuals_phys: dict[str, np.ndarray] = {}
    per_bin: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, cfg in LOSS_CONFIGS.items():
        print(f"\n[{name}] K={cfg['n_bins']}  binning={cfg['binning']}")
        loss = BinnedDFLQuantileOffsetLoss(**cfg).double()
        if name == "phi":
            bin_idx, res_bw, res_phys = phi_proxy(loss, len(targets), rng)
        else:
            bin_idx, res_bw, res_phys = compute_bin_and_residual(
                loss, targets[:, TARGET_COLS[name]]
            )
        residuals[name] = res_bw
        residuals_phys[name] = res_phys
        per_bin[name] = (bin_idx, res_bw)

        q5, q25, q50, q75, q95 = np.percentile(res_bw, [5, 25, 50, 75, 95])
        print(f"   residual (bin widths):  med={q50:+.4f}  "
              f"IQR=[{q25:+.3f},{q75:+.3f}]  p5-p95=[{q5:+.3f},{q95:+.3f}]")

    print("\nPlotting …")
    plot_aggregate(residuals, out_dir)
    for name in LOSS_CONFIGS:
        bin_idx, res_bw = per_bin[name]
        plot_per_bin_slice(name, bin_idx, res_bw,
                           LOSS_CONFIGS[name]["n_bins"], out_dir)
    plot_physical(residuals_phys, out_dir)

    print(f"\nAll plots in: {out_dir}")


if __name__ == "__main__":
    main()
