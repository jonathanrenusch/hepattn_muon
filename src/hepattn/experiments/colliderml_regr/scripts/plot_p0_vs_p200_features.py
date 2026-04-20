#!/usr/bin/env python3
"""Overlay hit-feature distributions of the p0_core_pretrain and p200_core_finetune
datasets to visualise the pretrain vs finetune domain shift.

Hit columns per manifest.json::

    0 x   1 y   2 z   3 r   4 phi_hit   5 theta_hit   6 s
    7 volume_id   8 layer_id   9 surface_id   10 detector

eta_hit is computed at training time from theta_hit (η = -log(tan(θ/2)));
we reproduce it here.

Outputs under ``.../v2/core_configs/splines/``:
    p0_vs_p200_hit_features.png      — 3×4 grid of per-feature overlays
    p0_vs_p200_track_structure.png   — hits-per-track + tracks-per-shard
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


P0 = Path("/scratch/colliderml/p0_core_pretrain")
P200 = Path("/scratch/colliderml/p200_core_finetune")

HIT_COLS = ["x", "y", "z", "r", "phi_hit", "theta_hit", "s",
            "volume_id", "layer_id", "surface_id", "detector"]
CATEGORICAL = {"volume_id", "layer_id", "detector"}  # surface_id is huge-range, treat continuous

COLOR_P0 = "#1f77b4"   # blue
COLOR_P200 = "#d62728" # red
ALPHA = 0.55


def theta_to_eta(theta: np.ndarray) -> np.ndarray:
    half = np.clip(theta * 0.5, 1e-7, np.pi * 0.5 - 1e-7)
    return -np.log(np.tan(half))


def load_dataset(root: Path, num_shards: int) -> dict[str, np.ndarray]:
    shard_dirs = sorted(root.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    hit_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    hits_per_track: list[np.ndarray] = []
    tracks_per_shard: list[int] = []

    for sd in tqdm(shard_dirs, desc=f"loading {root.name}"):
        hits_path = sd / "hits.npy"
        off_path = sd / "selected_tracks" / "track_hit_offsets.npy"
        tgt_path = sd / "selected_tracks" / "track_targets.npy"
        if not (hits_path.exists() and off_path.exists() and tgt_path.exists()):
            continue
        hits = np.asarray(np.load(hits_path, mmap_mode="r"), dtype=np.float32)
        offsets = np.asarray(np.load(off_path, mmap_mode="r"), dtype=np.int64)
        targets = np.asarray(np.load(tgt_path, mmap_mode="r"), dtype=np.float32)
        # sub-sample hits to keep memory bounded (~200k per shard)
        if hits.shape[0] > 200_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(hits.shape[0], size=200_000, replace=False)
            hits = hits[idx]
        hit_chunks.append(hits)
        target_chunks.append(targets)
        hits_per_track.append(np.diff(offsets))
        tracks_per_shard.append(int(offsets.shape[0] - 1))

    hits_all = np.concatenate(hit_chunks, axis=0)
    tgt_all = np.concatenate(target_chunks, axis=0)
    hpt_all = np.concatenate(hits_per_track, axis=0)
    return {
        "hits": hits_all,
        "targets": tgt_all,  # columns: d0, z0, phi, theta, qop
        "hits_per_track": hpt_all,
        "tracks_per_shard": np.array(tracks_per_shard, dtype=np.int64),
    }


def plot_target_overlay(ds0: dict, ds200: dict, outdir: Path) -> Path:
    t0 = ds0["targets"].astype(np.float64)
    t200 = ds200["targets"].astype(np.float64)

    # Derived: eta = -log(tan(theta/2)),  pT = sin(theta) / |qop|
    eta0 = theta_to_eta(t0[:, 3]); eta200 = theta_to_eta(t200[:, 3])
    qop0, qop200 = t0[:, 4], t200[:, 4]
    pt0 = np.sin(t0[:, 3]) / np.abs(qop0).clip(min=1e-9)
    pt200 = np.sin(t200[:, 3]) / np.abs(qop200).clip(min=1e-9)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(
        f"Track-target distributions — p0 ({t0.shape[0]:,} tracks) vs p200 ({t200.shape[0]:,} tracks)",
        fontsize=14, fontweight="bold",
    )

    panels: list[tuple[str, np.ndarray, np.ndarray, str, bool]] = [
        # name,            x0,          x200,         unit/label,           clip_to_inner_999
        ("d0",             t0[:, 0],    t200[:, 0],   "d0  [mm]",           True),
        ("z0",             t0[:, 1],    t200[:, 1],   "z0  [mm]",           True),
        ("phi",            t0[:, 2],    t200[:, 2],   "phi  [rad]",         False),
        ("theta",          t0[:, 3],    t200[:, 3],   "theta  [rad]",       False),
        ("eta (derived)",  eta0,        eta200,       "eta",                False),
        ("qop",            qop0,        qop200,       "qop  [1/GeV]",       True),
        ("pT (derived)",   pt0,         pt200,        "pT  [GeV]  (log-x)", True),
        ("log10(pT)",      np.log10(np.clip(pt0, 1e-3, None)),
                           np.log10(np.clip(pt200, 1e-3, None)),
                           "log10(pT / GeV)",        False),
    ]

    for i, (name, x0, x200, xlabel, clip) in enumerate(panels):
        ax = axes[i]
        if clip:
            lo = min(np.quantile(x0, 0.001), np.quantile(x200, 0.001))
            hi = max(np.quantile(x0, 0.999), np.quantile(x200, 0.999))
        else:
            lo = min(x0.min(), x200.min())
            hi = max(x0.max(), x200.max())
        if name == "pT (derived)":
            # log-spaced bins on pT
            bins = np.logspace(np.log10(max(lo, 0.1)), np.log10(min(hi, 500.0)), 101)
            ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 101)
        ax.hist(x0,   bins=bins, **_hist_kwargs(COLOR_P0,   "p0"))
        ax.hist(x200, bins=bins, **_hist_kwargs(COLOR_P200, "p200"))
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density (log)")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = outdir / "p0_vs_p200_track_targets.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _hist_kwargs(color: str, label: str) -> dict:
    return dict(color=color, alpha=ALPHA, label=label, edgecolor="none", density=True)


def plot_feature_overlay(ds0: dict, ds200: dict, outdir: Path) -> Path:
    hits0 = ds0["hits"]
    hits200 = ds200["hits"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()
    fig.suptitle(
        f"Hit-feature distributions — p0 ({hits0.shape[0]:,} hits) vs p200 ({hits200.shape[0]:,} hits)",
        fontsize=14, fontweight="bold",
    )

    # 11 stored features + eta_hit derived = 12 panels
    for i, name in enumerate(HIT_COLS):
        ax = axes[i]
        x0 = hits0[:, i]
        x200 = hits200[:, i]
        if name in CATEGORICAL:
            vmin = int(min(x0.min(), x200.min()))
            vmax = int(max(x0.max(), x200.max()))
            bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
        else:
            lo = min(np.quantile(x0, 0.001), np.quantile(x200, 0.001))
            hi = max(np.quantile(x0, 0.999), np.quantile(x200, 0.999))
            bins = np.linspace(lo, hi, 101)
        ax.hist(x0, bins=bins, **_hist_kwargs(COLOR_P0, "p0"))
        ax.hist(x200, bins=bins, **_hist_kwargs(COLOR_P200, "p200"))
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("density (log)")
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    # Panel 12 — eta_hit derived
    ax = axes[11]
    e0 = theta_to_eta(hits0[:, 5].astype(np.float64))
    e200 = theta_to_eta(hits200[:, 5].astype(np.float64))
    lo = min(np.quantile(e0, 0.001), np.quantile(e200, 0.001))
    hi = max(np.quantile(e0, 0.999), np.quantile(e200, 0.999))
    bins = np.linspace(lo, hi, 101)
    ax.hist(e0, bins=bins, **_hist_kwargs(COLOR_P0, "p0"))
    ax.hist(e200, bins=bins, **_hist_kwargs(COLOR_P200, "p200"))
    ax.set_yscale("log")
    ax.set_xlabel("eta_hit  (derived)")
    ax.set_ylabel("density (log)")
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    out = outdir / "p0_vs_p200_hit_features.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_track_structure(ds0: dict, ds200: dict, outdir: Path) -> Path:
    hpt0 = ds0["hits_per_track"]
    hpt200 = ds200["hits_per_track"]
    tps0 = ds0["tracks_per_shard"]
    tps200 = ds200["tracks_per_shard"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Track structure — p0 vs p200",
        fontsize=14, fontweight="bold",
    )

    # Hits per track
    ax = axes[0]
    bmin = int(min(hpt0.min(), hpt200.min()))
    bmax = int(max(hpt0.max(), hpt200.max()))
    bins = np.arange(bmin - 0.5, bmax + 1.5, 1.0)
    ax.hist(hpt0, bins=bins, **_hist_kwargs(COLOR_P0, f"p0  ({len(hpt0):,} tracks)"))
    ax.hist(hpt200, bins=bins, **_hist_kwargs(COLOR_P200, f"p200 ({len(hpt200):,} tracks)"))
    ax.set_yscale("log")
    ax.set_xlabel("hits per track")
    ax.set_ylabel("density (log)")
    ax.set_title(f"p0 mean={hpt0.mean():.2f}   p200 mean={hpt200.mean():.2f}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")

    # Tracks per shard (proxy for tracks per event after preprocessing)
    ax = axes[1]
    lo = int(min(tps0.min(), tps200.min()))
    hi = int(max(tps0.max(), tps200.max()))
    bins = np.linspace(lo * 0.99, hi * 1.01, 40)
    ax.hist(tps0, bins=bins, **_hist_kwargs(COLOR_P0, f"p0  (mean {tps0.mean():,.0f})"))
    ax.hist(tps200, bins=bins, **_hist_kwargs(COLOR_P200, f"p200 (mean {tps200.mean():,.0f})"))
    ax.set_xlabel("tracks per shard")
    ax.set_ylabel("density")
    ax.set_title("Shard-level track multiplicity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out = outdir / "p0_vs_p200_track_structure.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay hit-feature distributions: p0_core_pretrain vs p200_core_finetune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--p0", type=str, default=str(P0))
    ap.add_argument("--p200", type=str, default=str(P200))
    ap.add_argument("--num-shards", type=int, default=5,
                    help="shards per dataset (hits sub-sampled to 200k per shard).")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    p0 = Path(args.p0)
    p200 = Path(args.p200)
    if args.output_dir is None:
        outdir = (
            Path(__file__).resolve().parent.parent
            / "config" / "NeurIPS_retraining" / "v2" / "core_configs" / "splines"
        )
    else:
        outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("p0 vs p200 — feature overlay")
    print("=" * 70)
    print(f"  p0     : {p0}")
    print(f"  p200   : {p200}")
    print(f"  shards : {args.num_shards}  per dataset")
    print(f"  out    : {outdir}")

    ds0 = load_dataset(p0, args.num_shards)
    ds200 = load_dataset(p200, args.num_shards)

    out1 = plot_feature_overlay(ds0, ds200, outdir)
    print(f"  saved: {out1.name}")
    out2 = plot_track_structure(ds0, ds200, outdir)
    print(f"  saved: {out2.name}")
    out3 = plot_target_overlay(ds0, ds200, outdir)
    print(f"  saved: {out3.name}")


if __name__ == "__main__":
    main()
