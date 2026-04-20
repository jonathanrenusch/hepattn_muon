#!/usr/bin/env python3
"""Diagnostic plots for the ACTS double-matched regime.

Loads a couple of shards from the p200 dataset, computes per-track hit purity
and hit efficiency, and produces a set of plots focused on the double-matched
(purity > 0.75 AND hit-efficiency > 0.75) sample.

Run:
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.plot_double_matched_diagnostics
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm

DATA_DIR = Path("/eos/project/n/ngt2-4/data/ColliderML-Release-1.old/data")
PREFIX = "ttbar_pu200"
N_SHARDS = 2
OUT_DIR = Path("/shared/tracking/logs/acts_tracking_evaluation_p200/double_matched_diagnostics")

PURITY_THR = 0.75
EFF_THR = 0.75


def load_shards(n_shards: int = N_SHARDS):
    dfs = {}
    for key in ("particles", "tracker_hits", "tracks"):
        paths = sorted((DATA_DIR / f"{PREFIX}_{key}").glob("train-*.parquet"))[:n_shards]
        dfs[key] = pl.concat([pl.read_parquet(p) for p in paths])
        print(f"loaded {key}: {len(dfs[key])} events from {len(paths)} shard(s)")
    return dfs["particles"], dfs["tracker_hits"], dfs["tracks"]


def compute_matching(particles_df: pl.DataFrame, hits_df: pl.DataFrame,
                     tracks_df: pl.DataFrame) -> pl.DataFrame:
    """Vectorised per-track purity + hit-efficiency using polars joins."""

    # Hits indexed by their position within the event.
    hits_long = (
        hits_df.select("event_id", "particle_id")
        .explode("particle_id")
        .with_columns(
            pl.int_range(0, pl.len()).over("event_id").alias("hit_idx").cast(pl.UInt32)
        )
    )

    # Total hits per (event, particle) — denominator for hit efficiency.
    hits_per_particle = (
        hits_long.group_by("event_id", "particle_id")
        .len()
        .rename({"len": "n_particle_hits", "particle_id": "majority_particle_id"})
    )

    # Tracks: one row per track.
    tracks_ex = tracks_df.explode(
        ["d0", "z0", "phi", "theta", "qop", "majority_particle_id", "hit_ids", "track_id"]
    ).with_columns(pl.col("hit_ids").list.len().alias("n_track_hits"))

    # One row per track-hit link; join particle_id of each hit.
    track_hits = (
        tracks_ex.select("event_id", "track_id", "majority_particle_id", "hit_ids")
        .explode("hit_ids")
        .join(hits_long, left_on=["event_id", "hit_ids"],
              right_on=["event_id", "hit_idx"], how="left")
        .with_columns(
            (pl.col("particle_id") == pl.col("majority_particle_id")).cast(pl.UInt32).alias("is_majority")
        )
        .group_by("event_id", "track_id")
        .agg(pl.col("is_majority").sum().alias("n_majority"))
    )

    tracks_ex = tracks_ex.join(track_hits, on=["event_id", "track_id"], how="left") \
                         .join(hits_per_particle, on=["event_id", "majority_particle_id"], how="left")

    tracks_ex = tracks_ex.with_columns([
        (pl.col("n_majority") / pl.col("n_track_hits")).alias("purity"),
        (pl.col("n_majority") / pl.col("n_particle_hits")).alias("hit_eff"),
    ]).with_columns(
        ((pl.col("purity") > PURITY_THR) & (pl.col("hit_eff") > EFF_THR)).alias("is_dm")
    )

    # Attach truth kinematics (pt_truth, eta_truth) from particles.
    particles_long = (
        particles_df.select("event_id", "particle_id", "px", "py", "pz", "primary",
                            "vertex_primary", "charge")
        .explode(["particle_id", "px", "py", "pz", "primary", "vertex_primary", "charge"])
        .with_columns([
            (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt().alias("pt_truth"),
            (pl.col("pz") / (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt())
                .arcsinh().alias("eta_truth"),
        ])
        .rename({"particle_id": "majority_particle_id"})
    )

    tracks_ex = tracks_ex.join(
        particles_long.select("event_id", "majority_particle_id", "pt_truth",
                              "eta_truth", "primary", "vertex_primary", "charge"),
        on=["event_id", "majority_particle_id"], how="left",
    )

    return tracks_ex


def binned(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    """Return (centers, mean, stderr, count) for y in x-bins."""
    idx = np.digitize(x, bins) - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)
    idx = idx[valid]; y = y[valid]
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(centers), np.nan)
    err = np.full(len(centers), np.nan)
    count = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = idx == i
        n = int(sel.sum())
        count[i] = n
        if n > 0:
            mean[i] = y[sel].mean()
            err[i] = y[sel].std() / np.sqrt(max(n, 1))
    return centers, mean, err, count


def step_fill(ax, bins, lo, hi, **kw):
    x = np.repeat(bins, 2)[1:-1]
    ylo = np.repeat(lo, 2); yhi = np.repeat(hi, 2)
    ax.fill_between(x, ylo, yhi, step=None, **kw)


def frac_err(k, n):
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    return np.sqrt(np.where(n > 0, p * (1 - p) / np.maximum(n, 1), 0.0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    particles_df, hits_df, tracks_df = load_shards()
    tracks = compute_matching(particles_df, hits_df, tracks_df)

    # Restrict to matched (non-null majority + primary + charged) for the quality study.
    matched = tracks.filter(
        pl.col("n_particle_hits").is_not_null() & pl.col("n_track_hits") > 0
    )
    primary_matched = matched.filter(
        (pl.col("primary") == True) & (pl.col("charge") != 0)  # noqa: E712
    )
    print(f"all tracks: {len(tracks)}; matched: {len(matched)}; "
          f"primary+charged matched: {len(primary_matched)}")

    df = primary_matched.to_pandas()
    dm = df[df["is_dm"]]
    print(f"double-matched fraction (primary+charged): {len(dm) / max(len(df), 1):.3f}")

    pt = df["pt_truth"].to_numpy()
    eta = df["eta_truth"].to_numpy()
    purity = df["purity"].to_numpy()
    hit_eff = df["hit_eff"].to_numpy()
    is_dm = df["is_dm"].to_numpy().astype(bool)

    eta_bins = np.linspace(-3.0, 3.0, 31)
    pt_bins = np.concatenate([np.linspace(0.5, 5.0, 19), np.logspace(np.log10(5.0), np.log10(50.0), 11)[1:]])

    # ---------- 1) Joint distribution of purity vs hit efficiency ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(purity, hit_eff, bins=[np.linspace(0, 1, 51)] * 2,
                  norm=LogNorm(), cmap="viridis")
    ax.axvline(PURITY_THR, color="red", ls="--", lw=1.2)
    ax.axhline(EFF_THR, color="red", ls="--", lw=1.2)
    ax.fill_betweenx([EFF_THR, 1.0], PURITY_THR, 1.0, color="red", alpha=0.08)
    frac_dm = is_dm.mean()
    ax.text(0.78, 0.95, f"DM region\nfrac = {frac_dm:.3f}",
            transform=ax.transAxes, ha="center", va="top", color="red", fontsize=10)
    ax.set_xlabel("Hit purity"); ax.set_ylabel("Hit efficiency")
    ax.set_title(f"Purity vs hit-efficiency ({len(df):,} primary+charged matched tracks)")
    fig.colorbar(h[3], ax=ax, label="Tracks")
    fig.tight_layout(); fig.savefig(OUT_DIR / "joint_purity_efficiency.png", dpi=150)
    plt.close(fig)

    # ---------- 2) 1D distributions (all vs within-DM) ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, arr, name in zip(axes, [purity, hit_eff], ["Hit purity", "Hit efficiency"]):
        bins = np.linspace(0, 1, 51)
        ax.hist(arr, bins=bins, histtype="step", lw=2, color="grey",
                label=f"All ({len(arr):,}, μ={arr.mean():.3f})")
        ax.hist(arr[is_dm], bins=bins, histtype="step", lw=2, color="steelblue",
                label=f"Within DM ({int(is_dm.sum()):,}, μ={arr[is_dm].mean():.3f})")
        ax.axvline(0.75, color="red", ls="--", lw=1.2, label="0.75 threshold")
        ax.set_xlabel(name); ax.set_ylabel("Tracks"); ax.set_yscale("log")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "distributions_1d.png", dpi=150)
    plt.close(fig)

    # ---------- 3) Mean purity/eff vs eta, all vs within-DM ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arr, name, col in zip(axes, [purity, hit_eff],
                                   ["Mean hit purity", "Mean hit efficiency"],
                                   ["steelblue", "darkorange"]):
        c, m, e, n = binned(eta, arr, eta_bins)
        ax.errorbar(c, m, yerr=e, fmt="o-", color="grey", alpha=0.7,
                    label=f"All matched (avg: {arr.mean():.3f})")
        c2, m2, e2, n2 = binned(eta[is_dm], arr[is_dm], eta_bins)
        ax.errorbar(c2, m2, yerr=e2, fmt="o-", color=col,
                    label=f"Within DM (avg: {arr[is_dm].mean():.3f})")
        ax.axhline(0.75, color="red", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel(r"$\eta$"); ax.set_ylabel(name); ax.set_ylim(0.5, 1.02)
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Matching quality vs eta (matched tracks vs double-matched subset)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "quality_vs_eta.png", dpi=150)
    plt.close(fig)

    # ---------- 4) Mean purity/eff vs pt, all vs within-DM ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arr, name, col in zip(axes, [purity, hit_eff],
                                   ["Mean hit purity", "Mean hit efficiency"],
                                   ["steelblue", "darkorange"]):
        c, m, e, n = binned(pt, arr, pt_bins)
        ax.errorbar(c, m, yerr=e, fmt="o-", color="grey", alpha=0.7,
                    label=f"All matched (avg: {arr.mean():.3f})")
        c2, m2, e2, n2 = binned(pt[is_dm], arr[is_dm], pt_bins)
        ax.errorbar(c2, m2, yerr=e2, fmt="o-", color=col,
                    label=f"Within DM (avg: {arr[is_dm].mean():.3f})")
        ax.axhline(0.75, color="red", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel(r"$p_T$ [GeV]"); ax.set_ylabel(name); ax.set_xscale("log")
        ax.set_ylim(0.5, 1.02); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Matching quality vs $p_T$ (matched tracks vs double-matched subset)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "quality_vs_pt.png", dpi=150)
    plt.close(fig)

    # ---------- 5) DM fraction vs eta and pt with binomial errors ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def dm_frac(x, bins):
        centers = 0.5 * (bins[:-1] + bins[1:])
        idx = np.digitize(x, bins) - 1
        k = np.zeros(len(centers)); n = np.zeros(len(centers))
        for i in range(len(centers)):
            sel = idx == i
            n[i] = sel.sum(); k[i] = is_dm[sel].sum() if sel.any() else 0
        f = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        return centers, f, frac_err(k, n)

    c, f, ferr = dm_frac(eta, eta_bins)
    axes[0].errorbar(c, f, yerr=ferr, fmt="o-", color="green")
    axes[0].set_xlabel(r"$\eta$"); axes[0].set_ylabel("DM fraction")
    axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"DM fraction vs η (overall {frac_dm:.3f})")

    c, f, ferr = dm_frac(pt, pt_bins)
    axes[1].errorbar(c, f, yerr=ferr, fmt="o-", color="green")
    axes[1].set_xlabel(r"$p_T$ [GeV]"); axes[1].set_ylabel("DM fraction")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("DM fraction vs $p_T$")

    fig.tight_layout(); fig.savefig(OUT_DIR / "dm_fraction.png", dpi=150)
    plt.close(fig)

    # ---------- 6) Within-DM distributions zoomed in [0.75, 1] ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins_zoom = np.linspace(0.75, 1.0, 51)
    for ax, arr, name, col in zip(axes, [purity[is_dm], hit_eff[is_dm]],
                                   ["Hit purity (within DM)", "Hit efficiency (within DM)"],
                                   ["steelblue", "darkorange"]):
        ax.hist(arr, bins=bins_zoom, color=col, alpha=0.7, edgecolor="black")
        ax.axvline(arr.mean(), color="red", ls="--",
                   label=f"mean = {arr.mean():.3f}")
        ax.set_xlabel(name); ax.set_ylabel("Tracks")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(f"Within the DM regime ({int(is_dm.sum()):,} tracks)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "within_dm_zoom.png", dpi=150)
    plt.close(fig)

    # ---------- 7) Fraction of tracks that are NOT perfectly matched ---------
    # Three imperfection categories:
    #   (a) purity < 1.0
    #   (b) hit efficiency < 1.0
    #   (c) 100%-DM fail  =  NOT (purity == 1 AND hit_eff == 1)
    # "100%-DM" is the strict version of the 0.75/0.75 DM criterion.
    imperfect_pur = purity < 1.0
    imperfect_eff = hit_eff < 1.0
    imperfect_dm100 = imperfect_pur | imperfect_eff  # fail 100%-DM

    global_pur = imperfect_pur.mean()
    global_eff = imperfect_eff.mean()
    global_dm = imperfect_dm100.mean()

    # Same restricted to the 0.75 DM subset (i.e. "how imperfect even inside DM?").
    dm_pur = imperfect_pur[is_dm].mean() if is_dm.any() else np.nan
    dm_eff = imperfect_eff[is_dm].mean() if is_dm.any() else np.nan
    dm_dm = imperfect_dm100[is_dm].mean() if is_dm.any() else np.nan

    def frac_vs(x, bins, flag):
        centers = 0.5 * (bins[:-1] + bins[1:])
        idx = np.digitize(x, bins) - 1
        k = np.zeros(len(centers)); n = np.zeros(len(centers))
        for i in range(len(centers)):
            sel = idx == i
            n[i] = sel.sum(); k[i] = flag[sel].sum() if sel.any() else 0
        f = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        return centers, f, frac_err(k, n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    cats = [
        ("purity < 1.0",       imperfect_pur,    "steelblue", global_pur),
        ("hit eff. < 1.0",     imperfect_eff,    "darkorange", global_eff),
        ("fails 100%-DM",      imperfect_dm100,  "crimson",    global_dm),
    ]
    for label, flag, col, avg in cats:
        c, f, fe = frac_vs(eta, eta_bins, flag)
        axes[0].errorbar(c, f, yerr=fe, fmt="o-", color=col,
                         label=f"{label} (global: {avg:.3f})")
        c, f, fe = frac_vs(pt, pt_bins, flag)
        axes[1].errorbar(c, f, yerr=fe, fmt="o-", color=col,
                         label=f"{label} (global: {avg:.3f})")

    axes[0].set_xlabel(r"$\eta$"); axes[0].set_ylabel("Fraction of tracks imperfect")
    axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_title("Imperfect-matching fraction vs $\\eta$")

    axes[1].set_xlabel(r"$p_T$ [GeV]"); axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[1].set_title("Imperfect-matching fraction vs $p_T$")

    fig.suptitle(
        f"Primary+charged matched tracks ({len(df):,}). "
        f"Within 0.75-DM subset: p<1={dm_pur:.3f}, eff<1={dm_eff:.3f}, "
        f"fails100%-DM={dm_dm:.3f}.",
        fontsize=11,
    )
    fig.tight_layout(); fig.savefig(OUT_DIR / "imperfect_fraction.png", dpi=150)
    plt.close(fig)

    # Summary
    summary = {
        "n_events": int(particles_df["event_id"].n_unique()),
        "n_tracks": int(len(tracks)),
        "n_matched_primary_charged": int(len(df)),
        "n_double_matched": int(is_dm.sum()),
        "dm_fraction": float(frac_dm),
        "mean_purity_all": float(purity.mean()),
        "mean_purity_dm": float(purity[is_dm].mean()),
        "mean_hit_eff_all": float(hit_eff.mean()),
        "mean_hit_eff_dm": float(hit_eff[is_dm].mean()),
        "imperfect_purity_global": float(global_pur),
        "imperfect_hiteff_global": float(global_eff),
        "fails_100pct_dm_global": float(global_dm),
        "imperfect_purity_within_dm75": float(dm_pur),
        "imperfect_hiteff_within_dm75": float(dm_eff),
        "fails_100pct_dm_within_dm75": float(dm_dm),
    }
    (OUT_DIR / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n"
    )
    print("wrote plots to", OUT_DIR)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
