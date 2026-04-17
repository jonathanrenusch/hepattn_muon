#!/usr/bin/env python3
"""Plot the Δφ = wrap(φ_true − φ_innermost_hit) distribution on log scale.

Mirrors the data-loading path used at training time (see data.py: innermost_idx
picks the hit with smallest arc length ``s``; feature column 4 is ``phi_hit``)
so the distribution matches exactly what the Gaussian/Quantile phi loss sees
when ``delta_anchor: innermost_phi`` is configured.

Also writes per-quantile summary + a spline knot proposal so the user can
eyeball whether a CDF-warped spline (like the d0 one) would help resolve
the core.

Usage::

    python plot_delta_phi_distribution.py                 # default: 20 shards
    python plot_delta_phi_distribution.py --num-shards 100
    python plot_delta_phi_distribution.py --all           # use every shard

Outputs (under ``colliderml_regr/config/NeurIPS_retraining/v2/core/splines/``):
    delta_phi_hist_log.png      — log-y histogram, full range + zoomed core
    delta_phi_cdf.png           — empirical CDF
    delta_phi_summary.txt       — quantile summary + spline-knot suggestion
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
PHI_COL = 2     # track_targets columns: d0, z0, phi, theta, qop
PHI_HIT_COL = 4  # hits columns: x, y, z, r, phi_hit, theta_hit, s, ...
S_COL = 6


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles into [-π, π]."""
    return np.remainder(x + np.pi, 2.0 * np.pi) - np.pi


def load_delta_phi(preprocessed_dir: Path, num_shards: int) -> np.ndarray:
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    deltas: list[np.ndarray] = []
    for sd in tqdm(shard_dirs, desc="Computing Δφ"):
        sel = sd / "selected_tracks"
        tgt_path = sel / "track_targets.npy"
        off_path = sel / "track_hit_offsets.npy"
        idx_path = sel / "track_hit_indices.npy"
        hits_path = sd / "hits.npy"
        if not (tgt_path.exists() and off_path.exists() and idx_path.exists() and hits_path.exists()):
            continue

        targets = np.load(tgt_path, mmap_mode="r")
        offsets = np.load(off_path, mmap_mode="r")
        hit_indices = np.load(idx_path, mmap_mode="r")
        hits = np.load(hits_path, mmap_mode="r")

        n_tracks = targets.shape[0]
        if n_tracks == 0:
            continue

        phi_true = np.asarray(targets[:, PHI_COL], dtype=np.float64)

        # Innermost-hit lookup: per track, gather s values for its hits and
        # take the argmin.  Done with a CSR-style loop over offsets — the
        # inner segments are tiny (≤20 hits) so this is fast and keeps memory
        # flat regardless of shard size.
        innermost_phi = np.empty(n_tracks, dtype=np.float64)
        off = np.asarray(offsets)
        hidx = np.asarray(hit_indices)
        hits_arr = np.asarray(hits)
        for t in range(n_tracks):
            start = int(off[t])
            end = int(off[t + 1])
            if end <= start:
                innermost_phi[t] = np.nan
                continue
            idx_slice = hidx[start:end]
            s_vals = hits_arr[idx_slice, S_COL]
            inner = int(idx_slice[int(np.argmin(s_vals))])
            innermost_phi[t] = hits_arr[inner, PHI_HIT_COL]

        good = np.isfinite(innermost_phi)
        delta = _wrap_pi(phi_true[good] - innermost_phi[good])
        deltas.append(delta.astype(np.float64))

    if not deltas:
        raise RuntimeError(f"No valid shards found under {preprocessed_dir}")

    out = np.concatenate(deltas)
    print(f"\nLoaded Δφ for {len(out):,} tracks from {len(shard_dirs)} shards")
    return out


def plot_log_hist(delta: np.ndarray, out_path: Path, window: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Δφ = wrap(φ_true − φ_innermost_hit) — {len(delta):,} tracks",
        fontsize=13, fontweight="bold",
    )

    # Full range, log y
    ax = axes[0]
    bins = np.linspace(-np.pi, np.pi, 501)
    ax.hist(delta, bins=bins, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Δφ  [rad]")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title("Full range [-π, π]")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(-window, color="crimson", ls="--", lw=1, label=f"±{window:g} rad")
    ax.axvline(+window, color="crimson", ls="--", lw=1)
    ax.legend(fontsize=9, loc="upper right")

    # Zoomed to ±window, log y
    ax = axes[1]
    zoom = delta[np.abs(delta) <= window]
    bins_z = np.linspace(-window, window, 401)
    ax.hist(zoom, bins=bins_z, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Δφ  [rad]")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title(f"Zoom: |Δφ| ≤ {window:g} rad  ({len(zoom)/len(delta)*100:.2f}% of tracks)")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cdf(delta: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sorted_d = np.sort(delta)
    ecdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

    ax = axes[0]
    step = max(1, len(sorted_d) // 20000)
    ax.plot(sorted_d[::step], ecdf[::step], lw=1.2, color="tab:blue")
    ax.set_xlabel("Δφ  [rad]")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF (full range)")
    ax.grid(True, alpha=0.3)

    # Mass-vs-window plot: "fraction inside |Δφ| ≤ x" over a log x axis.
    # This is what tells us whether a ±0.2 rad linear norm wastes resolution.
    ax = axes[1]
    windows = np.logspace(-5, np.log10(np.pi), 200)
    frac_inside = np.array([np.mean(np.abs(delta) <= w) for w in windows])
    ax.plot(windows, frac_inside, lw=1.5, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("window  w  [rad]")
    ax.set_ylabel("fraction with |Δφ| ≤ w")
    ax.set_title("Mass vs. symmetric window (log x)")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(0.2, color="crimson", ls="--", lw=1, label="current norm bound  0.2")
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def summary_text(delta: np.ndarray) -> str:
    qs = [1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
          0.99, 0.999, 0.9999]
    vals = np.quantile(delta, qs)
    lines = [
        f"n_tracks          : {len(delta):,}",
        f"range             : [{delta.min():+.6f}, {delta.max():+.6f}] rad",
        f"mean              : {delta.mean():+.6e}",
        f"std               : {delta.std():.6e}",
        f"median            : {np.median(delta):+.6e}",
        "",
        "quantiles (symmetric in τ):",
    ]
    for q, v in zip(qs, vals):
        lines.append(f"  p{q:<10g}: {v:+.6e}")
    lines.append("")
    for w in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        frac = float(np.mean(np.abs(delta) <= w))
        lines.append(f"  fraction |Δφ| ≤ {w:>5g} : {frac*100:.4f}%")
    lines.append("")
    # Resolution "efficiency" of the current linear norm bound:
    current_bound = 0.2
    inside = np.mean(np.abs(delta) <= current_bound)
    # 90% CDF half-width — what a spline would effectively achieve as the
    # analogue of d0's "48× better resolution in the core".
    p05, p95 = np.quantile(delta, [0.05, 0.95])
    half_width_90 = 0.5 * (p95 - p05)
    ratio = current_bound / max(half_width_90, 1e-12)
    lines += [
        f"current norm bound               : ±{current_bound:g} rad",
        f"  fraction inside                : {inside*100:.4f}%",
        f"  → fraction clipped to |t_norm|>1: {(1 - inside)*100:.4f}%",
        f"90%-CDF half-width (|p95-p05|/2) : {half_width_90:.6e} rad",
        f"current bound / half-width-90    : {ratio:.2f}×",
        "",
        "If the 90%-CDF half-width is « 0.2 rad, a CDF-warped spline gives",
        "roughly  (0.2 / half-width-90)  more output resolution in the core",
        "at the cost of slightly coarser resolution in the tails — same",
        "trade-off that made the d0 spline a ~48× win.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Δφ = wrap(φ_true − φ_innermost_hit) on log scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--preprocessed-dir", type=str, default=str(PREPROCESSED_DIR))
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="Use every shard")
    ap.add_argument(
        "--zoom", type=float, default=0.2,
        help="half-width (rad) of the zoomed histogram panel",
    )
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    if args.output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "config" / "NeurIPS_retraining" / "v2" / "core" / "splines"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_shards = -1 if args.all else args.num_shards
    print("=" * 70)
    print("Δφ distribution analysis")
    print("=" * 70)
    print(f"  data   : {preprocessed_dir}")
    print(f"  shards : {'all' if num_shards == -1 else num_shards}")
    print(f"  output : {output_dir}")

    delta = load_delta_phi(preprocessed_dir, num_shards=num_shards)

    plot_log_hist(delta, output_dir / "delta_phi_hist_log.png", window=args.zoom)
    plot_cdf(delta, output_dir / "delta_phi_cdf.png")

    summary = summary_text(delta)
    summary_path = output_dir / "delta_phi_summary.txt"
    summary_path.write_text(summary + "\n")
    print(f"  Saved: {summary_path.name}\n")
    print(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
