#!/usr/bin/env python3
"""Plot Δη = η_true − η_innermost_hit and the inner core of d0 on log scale.

Mirrors the data path used by ``plot_delta_phi_distribution.py``.  Innermost
hit is the one with smallest arc length ``s``; hit column 5 holds θ_hit,
which we convert to η = -log(tan(θ/2)).

Outputs under ``.../v2/core_configs/splines/``:
    delta_eta_hist_log.png       — full range + zoomed core, log-y
    delta_eta_summary.txt        — quantile summary
    d0_inner_distribution_log.png — inner 95% + inner 68% views, log-y
    d0_inner_summary.txt         — quantile summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PREPROCESSED_DIR = Path("/scratch/colliderml/p200_core_finetune")

D0_COL = 0
THETA_COL = 3
THETA_HIT_COL = 5
S_COL = 6


def theta_to_eta(theta: np.ndarray) -> np.ndarray:
    half = np.clip(theta * 0.5, 1e-7, np.pi * 0.5 - 1e-7)
    return -np.log(np.tan(half))


def load_data(preprocessed_dir: Path, num_shards: int) -> tuple[np.ndarray, np.ndarray]:
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    d0_all: list[np.ndarray] = []
    delta_eta_all: list[np.ndarray] = []

    for sd in tqdm(shard_dirs, desc="loading shards"):
        sel = sd / "selected_tracks"
        tgt_path = sel / "track_targets.npy"
        off_path = sel / "track_hit_offsets.npy"
        idx_path = sel / "track_hit_indices.npy"
        hits_path = sd / "hits.npy"
        if not all(p.exists() for p in (tgt_path, off_path, idx_path, hits_path)):
            continue

        targets = np.load(tgt_path, mmap_mode="r")
        offsets = np.load(off_path, mmap_mode="r")
        hit_indices = np.load(idx_path, mmap_mode="r")
        hits = np.load(hits_path, mmap_mode="r")

        n = targets.shape[0]
        if n == 0:
            continue

        d0_true = np.asarray(targets[:, D0_COL], dtype=np.float64)
        theta_true = np.asarray(targets[:, THETA_COL], dtype=np.float64)
        eta_true = theta_to_eta(theta_true)

        innermost_eta = np.empty(n, dtype=np.float64)
        off = np.asarray(offsets)
        hidx = np.asarray(hit_indices)
        harr = np.asarray(hits)
        for t in range(n):
            start, end = int(off[t]), int(off[t + 1])
            if end <= start:
                innermost_eta[t] = np.nan
                continue
            idx_slice = hidx[start:end]
            s_vals = harr[idx_slice, S_COL]
            inner = int(idx_slice[int(np.argmin(s_vals))])
            innermost_eta[t] = theta_to_eta(np.asarray(harr[inner, THETA_HIT_COL], dtype=np.float64))

        good = np.isfinite(innermost_eta) & np.isfinite(eta_true)
        d0_all.append(d0_true[good])
        delta_eta_all.append((eta_true[good] - innermost_eta[good]).astype(np.float64))

    if not delta_eta_all:
        raise RuntimeError(f"No valid shards under {preprocessed_dir}")

    d0 = np.concatenate(d0_all)
    delta_eta = np.concatenate(delta_eta_all)
    print(f"\nLoaded {len(d0):,} tracks from {len(shard_dirs)} shards")
    return d0, delta_eta


def plot_delta_eta(delta: np.ndarray, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Δη = η_true − η_innermost_hit   ({len(delta):,} tracks)",
        fontsize=13, fontweight="bold",
    )

    # Full range, log-y
    ax = axes[0]
    xmax_full = float(np.quantile(np.abs(delta), 0.9999))
    bins = np.linspace(-xmax_full, xmax_full, 501)
    trimmed_full = delta[(delta >= -xmax_full) & (delta <= xmax_full)]
    ax.hist(trimmed_full, bins=bins, color="darkgreen", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Δη")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title(f"Full range  (±{xmax_full:.3f}, p99.99 of |Δη|)")
    ax.grid(True, which="both", alpha=0.3)

    # Zoom: inner 99.8% (p0.1 .. p99.9), log-y
    ax = axes[1]
    p01, p999 = np.quantile(delta, [0.001, 0.999])
    hw = max(abs(p01), abs(p999))
    zoom = delta[(delta >= -hw) & (delta <= hw)]
    bins_z = np.linspace(-hw, hw, 401)
    ax.hist(zoom, bins=bins_z, color="darkgreen", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Δη")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title(f"Core (p0.1 ↔ p99.9):  ±{hw:.4f}  ({len(zoom) / len(delta) * 100:.2f}% of tracks)")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = outdir / "delta_eta_hist_log.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_d0_inner(d0: np.ndarray, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"d0 truth distribution   ({len(d0):,} tracks)",
        fontsize=13, fontweight="bold",
    )

    # Inner 95% — user requested
    ax = axes[0]
    p025, p975 = np.quantile(d0, [0.025, 0.975])
    hw95 = max(abs(p025), abs(p975))
    zoom95 = d0[(d0 >= -hw95) & (d0 <= hw95)]
    bins95 = np.linspace(-hw95, hw95, 501)
    ax.hist(zoom95, bins=bins95, color="purple", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("d0  [mm]")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title(f"Inner 95% (p2.5 ↔ p97.5):  |d0| ≤ {hw95:.4f} mm")
    ax.grid(True, which="both", alpha=0.3)

    # Inner 68% — zoom into the peak
    ax = axes[1]
    p16, p84 = np.quantile(d0, [0.16, 0.84])
    hw68 = max(abs(p16), abs(p84))
    zoom68 = d0[(d0 >= -hw68) & (d0 <= hw68)]
    bins68 = np.linspace(-hw68, hw68, 401)
    ax.hist(zoom68, bins=bins68, color="purple", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("d0  [mm]")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title(f"Inner 68% (peak):  |d0| ≤ {hw68:.5f} mm")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = outdir / "d0_inner_distribution_log.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def summary_text(name: str, x: np.ndarray, unit: str) -> str:
    qs = [1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
          0.99, 0.999, 0.9999]
    vals = np.quantile(x, qs)
    lines = [
        f"{name}  [{unit}]",
        f"n_tracks : {len(x):,}",
        f"range    : [{x.min():+.6f}, {x.max():+.6f}] {unit}",
        f"mean     : {x.mean():+.6e}",
        f"std      : {x.std():.6e}",
        f"median   : {np.median(x):+.6e}",
        "",
        "quantiles:",
    ]
    for q, v in zip(qs, vals):
        lines.append(f"  p{q:<10g} : {v:+.6e}")
    lines.append("")
    p05, p95 = np.quantile(x, [0.05, 0.95])
    half_width_90 = 0.5 * (p95 - p05)
    lines.append(f"90%-CDF half-width (|p95-p05|/2): {half_width_90:.6e} {unit}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Δη and inner-core d0 on log scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--preprocessed-dir", type=str, default=str(PREPROCESSED_DIR))
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    if args.output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "config" / "NeurIPS_retraining" / "v2" / "core_configs" / "splines"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Δη + d0 inner-core distribution analysis")
    print("=" * 70)
    print(f"  data   : {preprocessed_dir}")
    print(f"  shards : {args.num_shards}")
    print(f"  out    : {output_dir}")

    d0, delta_eta = load_data(preprocessed_dir, args.num_shards)

    out_eta = plot_delta_eta(delta_eta, output_dir)
    print(f"  saved: {out_eta.name}")
    out_d0 = plot_d0_inner(d0, output_dir)
    print(f"  saved: {out_d0.name}")

    eta_summary = summary_text("Δη", delta_eta, "dimensionless")
    (output_dir / "delta_eta_summary.txt").write_text(eta_summary + "\n")
    print("\n" + eta_summary)

    d0_summary = summary_text("d0", d0, "mm")
    (output_dir / "d0_inner_summary.txt").write_text(d0_summary + "\n")
    print("\n" + d0_summary)


if __name__ == "__main__":
    main()
