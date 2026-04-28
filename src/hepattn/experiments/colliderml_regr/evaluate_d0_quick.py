"""Fast analysis for d0-only classifier predictions.

Four plots, double-matched regime only:

1. ``heatmap_d0.png`` — 2x2 grid of truth-vs-pred 2D heatmaps:
     top row   : SSM (left)  vs ACTS (right) on ALL DM tracks
     bottom row: same pair restricted to iter-3sigma-clipped residuals

2. ``d0_residuals.png`` — 2x2 grid of residual histograms:
     top row   : ALL DM, linear-y      | ALL DM, log-y
     bottom row: iter-3sigma-clipped, linear-y | iter-3sigma-clipped, log-y
     SSM and ACTS overlaid in each panel.

3. ``d0_rms_vs_eta.png`` — 1x2 grid:
     left : RMS(d0) vs eta before clipping     (SSM + ACTS)
     right: RMS(d0) vs eta after iter-3sigma    (SSM + ACTS)

4. ``d0_fullrange.png`` — big-picture view over the full -2.5 to 2.5 mm:
     top row   : SSM (left) vs ACTS (right) truth-vs-pred heatmaps, full range
     bottom row: 1D d0 distribution (truth + SSM pred + ACTS pred),
                 linear-y and log-y overlaid.
     Shows whether the predictor covers the full tail spectrum (e.g. does
     the range-split classifier's outer head actually populate |d0| > 30um
     tracks, or does the router never fire?).

Only consumes ``preds/d0`` + ``targets/d0`` + ``targets/theta`` from the h5,
so it works equally well on the tiny d0-only h5 and on the full 5-head h5
(the other parameters are ignored).

Usage::

    pixi run python -m hepattn.experiments.colliderml_regr.evaluate_d0_quick \\
        --predictions <h5> --data-dir <p200_core_finetune> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    iterative_rms_convergence,
    load_acts_augmentation,
)

D0_RANGE_MM = (-0.2, 0.2)
D0_RANGE_CLIP_MM = (-0.05, 0.05)
D0_RANGE_FULL_MM = (-2.5, 2.5)
ETA_RANGE = (-3.0, 3.0)
N_ETA_BINS = 30
HEATMAP_BINS = 200


# ----------------------------------------------------------------------------
# Data loading — minimal, d0-only.
# ----------------------------------------------------------------------------

def _load_d0_predictions(
    path: Path,
    data_dir: Path | None = None,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred_d0, truth_d0, truth_theta).

    If truth theta is missing from the h5 (older writer versions only wrote
    targets for params that the model also predicted), falls back to loading
    ``track_targets.npy`` from the preprocessed shards — column 3 is theta.
    """
    with h5py.File(path, "r") as f:
        pred = f["preds"]["d0"][:]
        truth_d0 = f["targets"]["d0"][:]
        if "theta" in f["targets"]:
            truth_theta = f["targets"]["theta"][:]
        else:
            truth_theta = None

    if truth_theta is None:
        if data_dir is None:
            raise RuntimeError(
                "h5 does not contain targets/theta and no --data-dir "
                "fallback was provided."
            )
        truth_theta = _load_truth_theta_from_shards(data_dir, split)
        if len(truth_theta) != len(truth_d0):
            raise RuntimeError(
                f"Shard-loaded theta length {len(truth_theta):,} does not "
                f"match h5 d0 length {len(truth_d0):,} — split mismatch?"
            )
    return pred, truth_d0, truth_theta


def _load_truth_theta_from_shards(data_dir: Path, split: str) -> np.ndarray:
    import json
    split_file = data_dir / "split.json"
    with open(split_file) as f:
        splits = json.load(f)
    shard_indices = sorted(splits.get(split, []))
    pieces = []
    for idx in shard_indices:
        t = np.load(data_dir / f"shard_{idx:04d}" / "selected_tracks" / "track_targets.npy")
        pieces.append(t[:, 3])  # column 3 = theta
    return np.concatenate(pieces, axis=0)


def _eta_from_theta(theta: np.ndarray) -> np.ndarray:
    return -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))


# ----------------------------------------------------------------------------
# Plot 1 — heatmaps (truth vs pred, SSM and ACTS side by side).
# ----------------------------------------------------------------------------

def plot_heatmaps(
    ssm_truth: np.ndarray,
    ssm_pred: np.ndarray,
    acts_truth: np.ndarray,
    acts_pred: np.ndarray,
    ssm_clip_lo: float,
    ssm_clip_hi: float,
    acts_clip_lo: float,
    acts_clip_hi: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)

    # Top row — ALL DM tracks in the full |d0|<0.2 mm window.
    def _hm(ax, truth, pred, title, rng):
        h, xe, ye = np.histogram2d(
            truth, pred, bins=HEATMAP_BINS, range=[rng, rng]
        )
        im = ax.pcolormesh(
            xe, ye, h.T, norm="log",
            shading="auto", cmap="viridis",
        )
        ax.plot(rng, rng, "r--", alpha=0.6, linewidth=1)
        ax.set_xlim(rng); ax.set_ylim(rng)
        ax.set_xlabel("truth d0 [mm]")
        ax.set_ylabel("pred d0 [mm]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="tracks / bin")

    _hm(axes[0, 0], ssm_truth, ssm_pred, f"SSM — all DM ({len(ssm_truth):,})", D0_RANGE_MM)
    _hm(axes[0, 1], acts_truth, acts_pred, f"ACTS — all DM ({len(acts_truth):,})", D0_RANGE_MM)

    # Bottom row — restrict to iter-3sigma-clipped residuals.
    ssm_res = ssm_pred - ssm_truth
    acts_res = acts_pred - acts_truth
    m_ssm = (ssm_res >= ssm_clip_lo) & (ssm_res <= ssm_clip_hi)
    m_acts = (acts_res >= acts_clip_lo) & (acts_res <= acts_clip_hi)

    _hm(
        axes[1, 0], ssm_truth[m_ssm], ssm_pred[m_ssm],
        f"SSM — iter-3σ clipped ({m_ssm.sum():,} / {len(ssm_truth):,})",
        D0_RANGE_CLIP_MM,
    )
    _hm(
        axes[1, 1], acts_truth[m_acts], acts_pred[m_acts],
        f"ACTS — iter-3σ clipped ({m_acts.sum():,} / {len(acts_truth):,})",
        D0_RANGE_CLIP_MM,
    )

    fig.suptitle("d0: truth vs prediction — double-matched", fontsize=14)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 2 — residual histograms, pre- and post-clip, linear and log y.
# ----------------------------------------------------------------------------

def plot_residuals(
    ssm_res: np.ndarray,
    acts_res: np.ndarray,
    ssm_clip_lo: float,
    ssm_clip_hi: float,
    acts_clip_lo: float,
    acts_clip_hi: float,
    ssm_rms_all: float,
    acts_rms_all: float,
    ssm_rms_clip: float,
    acts_rms_clip: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    def _hist(ax, lo, hi, log, ssm_r, acts_r, title):
        bins = np.linspace(lo, hi, 200)
        ax.hist(
            ssm_r, bins=bins, histtype="step", linewidth=1.6,
            color="C0", label=f"SSM  (RMS={np.std(ssm_r)*1e3:.1f} μm, N={len(ssm_r):,})",
        )
        ax.hist(
            acts_r, bins=bins, histtype="step", linewidth=1.6,
            color="C1", label=f"ACTS (RMS={np.std(acts_r)*1e3:.1f} μm, N={len(acts_r):,})",
        )
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("d0 residual (pred − truth) [mm]")
        ax.set_ylabel("tracks / bin")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    # Top row: pre-clip (wider window).
    _hist(axes[0, 0], D0_RANGE_MM[0], D0_RANGE_MM[1], False,
          ssm_res, acts_res, "ALL DM — linear y")
    _hist(axes[0, 1], D0_RANGE_MM[0], D0_RANGE_MM[1], True,
          ssm_res, acts_res, "ALL DM — log y")

    # Bottom row: post-clip.
    ssm_mask = (ssm_res >= ssm_clip_lo) & (ssm_res <= ssm_clip_hi)
    acts_mask = (acts_res >= acts_clip_lo) & (acts_res <= acts_clip_hi)

    clip_lo = min(ssm_clip_lo, acts_clip_lo)
    clip_hi = max(ssm_clip_hi, acts_clip_hi)
    _hist(axes[1, 0], clip_lo, clip_hi, False,
          ssm_res[ssm_mask], acts_res[acts_mask],
          f"iter-3σ clipped — linear y  (SSM σ={ssm_rms_clip*1e3:.1f} μm, ACTS σ={acts_rms_clip*1e3:.1f} μm)")
    _hist(axes[1, 1], clip_lo, clip_hi, True,
          ssm_res[ssm_mask], acts_res[acts_mask],
          "iter-3σ clipped — log y")

    fig.suptitle("d0 residuals — double-matched", fontsize=14)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 3 — RMS(d0) vs eta, before + after iterative clip.
# ----------------------------------------------------------------------------

def _rms_vs_eta(
    eta: np.ndarray,
    res: np.ndarray,
    use_iterative: bool,
    min_tracks: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (centers, rms_per_bin, n_kept_per_bin, total_kept)."""
    bins = np.linspace(ETA_RANGE[0], ETA_RANGE[1], N_ETA_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    out = np.full(N_ETA_BINS, np.nan)
    n_kept = np.zeros(N_ETA_BINS, dtype=int)
    total_kept = 0
    for i in range(N_ETA_BINS):
        m = (eta >= bins[i]) & (eta < bins[i + 1])
        if m.sum() < min_tracks:
            continue
        r = res[m]
        if use_iterative:
            conv = iterative_rms_convergence(r)
            out[i] = conv["rms"]
            kept = int(round(conv["frac_kept"] * m.sum()))
        else:
            out[i] = float(np.std(r))
            kept = int(m.sum())
        n_kept[i] = kept
        total_kept += kept
    return centers, out, n_kept, total_kept


def plot_rms_vs_eta(
    ssm_res: np.ndarray,
    ssm_eta: np.ndarray,
    acts_res: np.ndarray,
    acts_eta: np.ndarray,
    output_path: Path,
    ssm_pt: np.ndarray | None = None,
    acts_pt: np.ndarray | None = None,
) -> None:
    """2×3 panel of RMS(d0) vs η.

    Top row: all DM tracks — before clipping, after iter-3σ clip, and the
    track-count-per-η-bin (same for SSM/CKF since DM is shared).
    Bottom row: iter-3σ-clipped RMS for three pT slices [0.5, 5] GeV,
    [5, 10] GeV, and ≥5 GeV. Each legend entry reports how many tracks
    contributed after clipping.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    def _draw(ax, eta_s, r_s, eta_a, r_a, iterative, title):
        c_s, y_s, _, n_s = _rms_vs_eta(eta_s, r_s, iterative)
        c_a, y_a, _, n_a = _rms_vs_eta(eta_a, r_a, iterative)
        ax.plot(c_s, y_s * 1e3, "o-", color="C0",
                label=f"SSM  (N={n_s:,})")
        ax.plot(c_a, y_a * 1e3, "s-", color="C1",
                label=f"ACTS (N={n_a:,})")
        ax.set_xlabel("truth η")
        ax.set_ylabel("RMS(d0) [μm]")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", fontsize=9)
        if iterative:
            ax.set_ylim(bottom=0)

    # Top row: all tracks (before, after) + count-per-bin.
    _draw(axes[0, 0], ssm_eta, ssm_res, acts_eta, acts_res, False,
          "all DM — before clipping")
    _draw(axes[0, 1], ssm_eta, ssm_res, acts_eta, acts_res, True,
          "all DM — after iter-3σ clip")
    # 3rd panel top: track count per η bin.
    eta_bins = np.linspace(ETA_RANGE[0], ETA_RANGE[1], N_ETA_BINS + 1)
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    counts, _ = np.histogram(ssm_eta, bins=eta_bins)
    axes[0, 2].step(eta_centers, counts, where="mid", color="green")
    axes[0, 2].set_xlabel("truth η"); axes[0, 2].set_ylabel("DM tracks / bin")
    axes[0, 2].set_title(f"Track count vs η  (total DM = {len(ssm_eta):,})")
    axes[0, 2].grid(alpha=0.25)

    # Bottom row: pT slices (after iter-3σ clip).
    if ssm_pt is None or acts_pt is None:
        for ax in axes[1]:
            ax.set_visible(False)
        fig.suptitle("d0 resolution vs pseudorapidity — double-matched",
                     fontsize=14)
    else:
        pt_slices = [
            ("0.5–5 GeV", (0.5, 5.0)),
            ("5–10 GeV",  (5.0, 10.0)),
            (">5 GeV",    (5.0, np.inf)),
        ]
        for ax, (label, (lo, hi)) in zip(axes[1], pt_slices):
            m_s = (ssm_pt >= lo) & (ssm_pt < hi)
            m_a = (acts_pt >= lo) & (acts_pt < hi)
            _draw(ax,
                  ssm_eta[m_s], ssm_res[m_s],
                  acts_eta[m_a], acts_res[m_a],
                  True,
                  f"{label} — after iter-3σ clip")
        fig.suptitle(
            "d0 resolution vs pseudorapidity — double-matched, with pT splits",
            fontsize=14,
        )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_rms_vs_eta_ssm_only(
    ssm_res: np.ndarray,
    ssm_eta: np.ndarray,
    output_path: Path,
) -> None:
    """SSM-only version of the RMS-vs-η plot.

    The joint SSM+ACTS plot uses a single auto-scaled y-axis; when ACTS'
    pre/post-clip RMS is ~5-10x the SSM's (typical in this study), that
    scaling flattens the SSM curve and hides real eta-dependent structure.
    This SSM-only variant lets matplotlib scale to the SSM's actual range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for ax, iterative, title in [
        (axes[0], False, "RMS(d0) vs η — before clipping"),
        (axes[1], True, "RMS(d0) vs η — after iter-3σ clip"),
    ]:
        c, r, _, n_tot = _rms_vs_eta(ssm_eta, ssm_res, iterative)
        r_um = r * 1e3
        ax.plot(c, r_um, "o-", color="C0", linewidth=1.6, markersize=5,
                label=f"SSM (N={n_tot:,})")
        ax.set_xlabel("truth η")
        ax.set_ylabel("RMS(d0) [μm]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", fontsize=10)
        finite = np.isfinite(r_um)
        if finite.any():
            lo, hi = float(np.nanmin(r_um[finite])), float(np.nanmax(r_um[finite]))
            margin = 0.10 * (hi - lo) if hi > lo else 0.1 * hi
            ax.set_ylim(max(lo - margin, 0), hi + margin)

    fig.suptitle("d0 resolution vs pseudorapidity — SSM only (native scale)",
                 fontsize=14)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 5 — SSM-only residuals (no ACTS) so the x-scale fits SSM's actual
# distribution width.  Overlaying ACTS inflates the x-range ~3x and hides
# the structure of the SSM histogram to the eye.
# ----------------------------------------------------------------------------

def plot_residuals_ssm_only(
    ssm_res: np.ndarray,
    ssm_clip_lo: float,
    ssm_clip_hi: float,
    ssm_std_all: float,
    ssm_rms_clip: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # Pre-clip window: a few SSM stds wide so tails are visible but the bulk
    # still dominates (vs the D0_RANGE_MM = ±200 um which is pulled by ACTS).
    pre_half = max(6.0 * ssm_std_all, 5.0 * ssm_rms_clip)
    pre_lo, pre_hi = -pre_half, pre_half

    # Post-clip window: the iter-3σ clip bounds (symmetrised for the plot).
    post_half = max(abs(ssm_clip_lo), abs(ssm_clip_hi))
    post_lo, post_hi = -post_half, post_half

    ssm_clipped = ssm_res[(ssm_res >= ssm_clip_lo) & (ssm_res <= ssm_clip_hi)]

    def _hist(ax, data, lo, hi, log, title, label):
        bins = np.linspace(lo, hi, 240)
        ax.hist(data, bins=bins, histtype="step", linewidth=1.6, color="C0", label=label)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("d0 residual (pred − truth) [mm]")
        ax.set_ylabel("tracks / bin")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    _hist(axes[0, 0], ssm_res, pre_lo, pre_hi, False,
          f"ALL — linear y  (x ∈ ±{pre_half*1e3:.0f} μm, {len(ssm_res):,} tracks)",
          f"SSM (std = {ssm_std_all*1e3:.1f} μm)")
    _hist(axes[0, 1], ssm_res, pre_lo, pre_hi, True,
          "ALL — log y", f"SSM (std = {ssm_std_all*1e3:.1f} μm)")
    _hist(axes[1, 0], ssm_clipped, post_lo, post_hi, False,
          f"iter-3σ clipped — linear y  (x ∈ ±{post_half*1e3:.0f} μm, "
          f"{len(ssm_clipped):,} tracks)",
          f"SSM (σ = {ssm_rms_clip*1e3:.1f} μm)")
    _hist(axes[1, 1], ssm_clipped, post_lo, post_hi, True,
          "iter-3σ clipped — log y",
          f"SSM (σ = {ssm_rms_clip*1e3:.1f} μm)")

    fig.suptitle(
        "d0 residuals (SSM only — x-scale set by SSM, no ACTS compression)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot 4 — full-range view (-2.5 to 2.5 mm) for the big picture.
# ----------------------------------------------------------------------------

def analyze_collapse_modes(
    truth: np.ndarray,
    ssm_pred: np.ndarray,
    acts_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Quantify two d0 collapse modes for SSM vs CKF.

    Two visual artefacts appear on the truth-vs-pred heatmap:

    * **Horizontal band at pred ≈ 0**: model predicts ~zero for tracks
      whose truth |d0| is clearly non-zero ("primary prior"). In the
      CLAUDE.md this is the d0-collapse issue — the ≤20-hit sequence does
      not override the training-distribution mode (95 % of mass within
      |d0| ≤ 31 µm) and pred collapses toward the beamspot.
    * **Vertical band at truth ≈ 0**: model predicts clearly non-zero d0
      for tracks whose truth d0 is essentially at the primary vertex.
      The CKF exhibits this too (per-track propagation noise on primaries
      with few inner hits), and the physics claim is that the SSM
      exhibits it *less* often than the CKF.

    We measure both modes with per-band conditional probabilities and emit
    both a machine-readable text summary and a bar chart that puts SSM
    and CKF side by side. Also includes the truth-d0 mass summary so the
    band thresholds are data-driven (not the ±31 µm CLAUDE.md shorthand).
    """
    # -- data-driven truth mass summary ---------------------------------
    abs_truth = np.abs(truth)
    q_levels = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
    q_vals = np.quantile(abs_truth, q_levels)

    # -- band definitions -----------------------------------------------
    # Near-zero truth band (primary): tracks with |truth d0| below the 68%
    # inner core. Non-zero truth band (secondary-ish): tracks beyond the
    # 95% core, where the network has to actually estimate d0, not default
    # to the mode. Pred-zero threshold: 5 µm (same as the CLAUDE.md
    # diagnostic on run ac72e5c9).
    tau_primary = float(q_vals[list(q_levels).index(0.68)])
    tau_secondary = float(q_vals[list(q_levels).index(0.95)])
    delta_pred_zero = 5e-3  # mm (5 µm)

    # Horizontal collapse: tracks with |truth| >= tau_secondary, fraction
    # with |pred| < delta_pred_zero. Evaluated at multiple truth bands.
    def horizontal_rates(pred):
        rates = {}
        for lo, hi in [
            (tau_secondary, 0.10),
            (0.10, 0.30),
            (0.30, 1.00),
            (1.00, 2.50),
        ]:
            m = (abs_truth >= lo) & (abs_truth < hi)
            if m.sum() == 0:
                rates[(lo, hi)] = (np.nan, 0)
            else:
                hit = (np.abs(pred[m]) < delta_pred_zero).sum()
                rates[(lo, hi)] = (hit / m.sum(), int(m.sum()))
        # overall
        m_tail = abs_truth >= tau_secondary
        rates["overall_tail"] = (
            float(np.mean(np.abs(pred[m_tail]) < delta_pred_zero)),
            int(m_tail.sum()),
        )
        return rates

    # Vertical collapse: tracks with |truth| < tau_primary, fraction with
    # |pred| >= delta_spread (scanned at multiple δ).
    def vertical_rates(pred):
        m = abs_truth < tau_primary
        rates = {}
        for delta in [0.01, 0.03, 0.10, 0.30]:
            if m.sum() == 0:
                rates[delta] = (np.nan, 0)
            else:
                hit = (np.abs(pred[m]) >= delta).sum()
                rates[delta] = (hit / m.sum(), int(m.sum()))
        return rates

    ssm_h = horizontal_rates(ssm_pred)
    acts_h = horizontal_rates(acts_pred)
    ssm_v = vertical_rates(ssm_pred)
    acts_v = vertical_rates(acts_pred)

    # -- text summary ---------------------------------------------------
    txt = output_dir / "collapse_modes.txt"
    with open(txt, "w") as f:
        f.write("d0 collapse-mode quantification — SSM vs ACTS CKF\n")
        f.write("=" * 60 + "\n\n")

        f.write("Truth |d0| mass summary (data-driven thresholds):\n")
        for q, v in zip(q_levels, q_vals):
            f.write(f"  {int(q*1000)/10:>5.1f}%  <= {v*1e3:>10.3f} um\n")
        f.write(f"\n  tau_primary   (68% inner core) = {tau_primary*1e3:.3f} um\n")
        f.write(f"  tau_secondary (95% outer edge) = {tau_secondary*1e3:.3f} um\n")
        f.write(f"  delta_pred_zero (horizontal band half-width) = {delta_pred_zero*1e3:.1f} um\n\n")

        f.write(
            "Mode 1 — horizontal collapse: P(|pred| < 5 um | |truth| in band)\n"
            "  (how often the predictor collapses to ~0 on genuinely non-zero d0)\n\n"
        )
        f.write(f"{'truth |d0| band [mm]':<28} {'SSM':>14} {'CKF':>14}\n")
        for band in list(ssm_h.keys()):
            if band == "overall_tail":
                label = f"  overall |truth|≥{tau_secondary*1e3:.1f}um"
            else:
                lo, hi = band
                label = f"  [{lo:.3f}, {hi:.2f})"
            s_r, s_n = ssm_h[band]
            a_r, a_n = acts_h[band]
            f.write(
                f"{label:<28} "
                f"{s_r*100:>8.3f}% (N={s_n:>8,}) "
                f"{a_r*100:>8.3f}% (N={a_n:>8,})\n".replace(" nan%", "  nan%")
            )
        f.write("\n")

        f.write(
            "Mode 2 — vertical collapse: P(|pred| >= delta | |truth| < tau_primary)\n"
            "  (how often the predictor scatters far from 0 on truly-primary tracks.\n"
            "   CKF is expected to do this because it propagates per-track noise;\n"
            "   SSM should do it *less* if it is not just mimicking the CKF.)\n\n"
        )
        f.write(f"{'delta [mm]':<18} {'SSM':>14} {'CKF':>14}    ratio SSM/CKF\n")
        for d in sorted(ssm_v.keys()):
            s_r, s_n = ssm_v[d]
            a_r, a_n = acts_v[d]
            ratio = s_r / a_r if a_r > 0 else np.nan
            f.write(
                f"  |pred|>={d:<9.3f}  "
                f"{s_r*100:>8.3f}% (N={s_n:>8,}) "
                f"{a_r*100:>8.3f}% (N={a_n:>8,})   {ratio:>6.3f}\n"
            )
        f.write("\n")

        # Null baseline: if pred were drawn from the marginal truth dist,
        # horizontal-collapse rate on the secondary tail would equal
        # P(|truth| < delta_pred_zero) ~ mass of truth within 5 µm.
        prior_zero = float(np.mean(abs_truth < delta_pred_zero))
        f.write(
            f"Null baseline (pred ~ marginal truth): P(|truth| < 5 um) = "
            f"{prior_zero*100:.3f}%.\n"
            "A predictor that ignores the hits and samples from the truth\n"
            "distribution would hit the horizontal band at this rate.\n"
        )

    # -- bar chart ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Horizontal collapse by truth-|d0| band.
    band_keys = [k for k in ssm_h if k != "overall_tail"]
    band_labels = [f"[{lo:.2f}, {hi:.2f})" for (lo, hi) in band_keys]
    ssm_vals = [ssm_h[k][0] * 100 for k in band_keys]
    acts_vals = [acts_h[k][0] * 100 for k in band_keys]
    x = np.arange(len(band_keys))
    w = 0.38
    axes[0].bar(x - w/2, ssm_vals, w, color="C0", label="SSM")
    axes[0].bar(x + w/2, acts_vals, w, color="C1", label="CKF")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_labels, rotation=15, fontsize=9)
    axes[0].set_ylabel(f"P(|pred| < {delta_pred_zero*1e3:.0f} µm)  [%]")
    axes[0].set_xlabel("truth |d0| band [mm]")
    axes[0].set_title("Horizontal collapse (pred→0 on non-zero truth)")
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")

    # Vertical collapse by δ threshold.
    deltas = sorted(ssm_v.keys())
    ssm_vals_v = [ssm_v[d][0] * 100 for d in deltas]
    acts_vals_v = [acts_v[d][0] * 100 for d in deltas]
    x = np.arange(len(deltas))
    axes[1].bar(x - w/2, ssm_vals_v, w, color="C0", label="SSM")
    axes[1].bar(x + w/2, acts_vals_v, w, color="C1", label="CKF")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"≥{d*1e3:.0f} µm" for d in deltas])
    axes[1].set_ylabel(f"P(|pred| ≥ δ  |  |truth| < {tau_primary*1e3:.1f} µm)  [%]")
    axes[1].set_xlabel("pred threshold δ")
    axes[1].set_title("Vertical collapse (pred scattered on truly-primary)")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].set_yscale("log")

    fig.suptitle("d0 collapse modes — SSM vs CKF, double-matched", fontsize=14)
    fig.savefig(output_dir / "collapse_modes.png", dpi=130)
    plt.close(fig)


def plot_ssm_vs_acts(
    ssm_pred: np.ndarray,
    acts_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Heatmap of CKF prediction vs SSM prediction on the DM subset.

    Two panels: full tracker range (-2.5 to 2.5 mm) and zoomed core
    (-0.2 to 0.2 mm). Quantifies how correlated the two predictors are
    when they both fire — a diagonal indicates the SSM is shadowing the
    CKF; scatter off-diagonal indicates the SSM is making genuinely
    different decisions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    pearson = float(np.corrcoef(ssm_pred, acts_pred)[0, 1])

    for ax, rng, title in [
        (axes[0], D0_RANGE_FULL_MM, "full range (-2.5 to 2.5 mm)"),
        (axes[1], D0_RANGE_MM, "core (±0.2 mm)"),
    ]:
        h, xe, ye = np.histogram2d(
            ssm_pred, acts_pred, bins=HEATMAP_BINS, range=[rng, rng]
        )
        im = ax.pcolormesh(xe, ye, h.T, norm="log", shading="auto", cmap="viridis")
        ax.plot(rng, rng, "r--", alpha=0.6, linewidth=1)
        ax.set_xlim(rng); ax.set_ylim(rng)
        ax.set_xlabel("SSM pred d0 [mm]")
        ax.set_ylabel("CKF pred d0 [mm]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="tracks / bin")

    fig.suptitle(
        f"CKF vs SSM d0 prediction — double-matched (Pearson r = {pearson:.4f}, "
        f"N = {len(ssm_pred):,})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_fullrange_clipped(
    ssm_truth: np.ndarray,
    ssm_pred: np.ndarray,
    acts_truth: np.ndarray,
    acts_pred: np.ndarray,
    ssm_clip_lo: float,
    ssm_clip_hi: float,
    acts_clip_lo: float,
    acts_clip_hi: float,
    output_path: Path,
) -> None:
    """Full-range (-2.5 to 2.5 mm) heatmap of the iter-3σ-clipped subset.

    Same full tracker-range axes as ``plot_fullrange``, but restricted to the
    residual-clipped subset so the bulk structure of SSM and CKF is shown on
    the physics-wide scale without tail pixels dominating the colour norm.
    SSM (left) and CKF (right) side by side with a shared log colour scale.
    """
    ssm_res = ssm_pred - ssm_truth
    acts_res = acts_pred - acts_truth
    m_ssm = (ssm_res >= ssm_clip_lo) & (ssm_res <= ssm_clip_hi)
    m_acts = (acts_res >= acts_clip_lo) & (acts_res <= acts_clip_hi)

    h_ssm, xe, ye = np.histogram2d(
        ssm_truth[m_ssm], ssm_pred[m_ssm],
        bins=HEATMAP_BINS, range=[D0_RANGE_FULL_MM, D0_RANGE_FULL_MM],
    )
    h_acts, _, _ = np.histogram2d(
        acts_truth[m_acts], acts_pred[m_acts],
        bins=HEATMAP_BINS, range=[D0_RANGE_FULL_MM, D0_RANGE_FULL_MM],
    )

    vmax = max(h_ssm.max(), h_acts.max())
    vmin = 1
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    def _draw(ax, h, title):
        im = ax.pcolormesh(xe, ye, h.T, norm=norm, shading="auto", cmap="viridis")
        ax.plot(D0_RANGE_FULL_MM, D0_RANGE_FULL_MM, "r--", alpha=0.6, linewidth=1)
        ax.set_xlim(D0_RANGE_FULL_MM); ax.set_ylim(D0_RANGE_FULL_MM)
        ax.set_xlabel("truth d0 [mm]")
        ax.set_ylabel("pred d0 [mm]")
        ax.set_title(title)
        return im

    im0 = _draw(axes[0], h_ssm,
                f"SSM — iter-3σ clipped, full range ({m_ssm.sum():,} / {len(ssm_truth):,})")
    im1 = _draw(axes[1], h_acts,
                f"CKF — iter-3σ clipped, full range ({m_acts.sum():,} / {len(acts_truth):,})")
    fig.colorbar(im1, ax=axes, label="tracks / bin", shrink=0.9)

    fig.suptitle(
        "d0 truth vs prediction — iter-3σ clipped, full tracker range "
        f"({D0_RANGE_FULL_MM[0]:.1f} to {D0_RANGE_FULL_MM[1]:.1f} mm)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_fullrange(
    truth: np.ndarray,
    ssm_pred: np.ndarray,
    acts_pred: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    def _hm(ax, t, p, title):
        h, xe, ye = np.histogram2d(
            t, p, bins=HEATMAP_BINS, range=[D0_RANGE_FULL_MM, D0_RANGE_FULL_MM]
        )
        im = ax.pcolormesh(xe, ye, h.T, norm="log", shading="auto", cmap="viridis")
        ax.plot(D0_RANGE_FULL_MM, D0_RANGE_FULL_MM, "r--", alpha=0.6, linewidth=1)
        ax.set_xlim(D0_RANGE_FULL_MM); ax.set_ylim(D0_RANGE_FULL_MM)
        ax.set_xlabel("truth d0 [mm]")
        ax.set_ylabel("pred d0 [mm]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="tracks / bin")

    _hm(axes[0, 0], truth, ssm_pred, f"SSM — full range ({len(truth):,} tracks)")
    _hm(axes[0, 1], truth, acts_pred, f"ACTS — full range ({len(truth):,} tracks)")

    # 1D distribution — truth + SSM + ACTS.  Use the same binning so visual
    # discrepancies between populations are honest.
    bins = np.linspace(*D0_RANGE_FULL_MM, 250)
    for ax, use_log, title in [
        (axes[1, 0], False, "d0 distribution — linear y"),
        (axes[1, 1], True, "d0 distribution — log y"),
    ]:
        ax.hist(truth, bins=bins, histtype="step", linewidth=1.8, color="k",
                label=f"truth (N={len(truth):,})")
        ax.hist(ssm_pred, bins=bins, histtype="step", linewidth=1.6, color="C0",
                label="SSM pred")
        ax.hist(acts_pred, bins=bins, histtype="step", linewidth=1.6, color="C1",
                label="ACTS pred")
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("d0 [mm]")
        ax.set_ylabel("tracks / bin")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "d0 full-range view — double-matched subset over full tracker range "
        f"({D0_RANGE_FULL_MM[0]:.1f} to {D0_RANGE_FULL_MM[1]:.1f} mm)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestration.
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Path to test_predictions.h5")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Preprocessed data dir (for acts_reco.npy + acts_dm_mask.npy)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.predictions}")
    pred_d0, truth_d0, truth_theta = _load_d0_predictions(
        args.predictions, data_dir=args.data_dir, split=args.split,
    )
    print(f"  loaded {len(pred_d0):,} tracks")

    print(f"Loading ACTS augmentation from {args.data_dir}")
    acts_info = load_acts_augmentation(args.data_dir, split=args.split)
    if acts_info is None:
        raise RuntimeError(f"ACTS augmentation not found under {args.data_dir}")
    acts_reco, acts_dm_mask, _ = acts_info
    if len(acts_reco) != len(pred_d0):
        raise RuntimeError(
            f"Size mismatch: predictions have {len(pred_d0):,} tracks but "
            f"ACTS augmentation has {len(acts_reco):,}"
        )

    # Restrict to DM + ACTS-matched (non-NaN acts_reco d0) subset.
    acts_has_match = ~np.isnan(acts_reco[:, 0])
    dm_mask = acts_dm_mask & acts_has_match
    print(f"  double-matched: {dm_mask.sum():,} / {len(dm_mask):,} tracks")

    ssm_truth = truth_d0[dm_mask]
    ssm_pred = pred_d0[dm_mask]
    ssm_res = ssm_pred - ssm_truth
    ssm_eta = _eta_from_theta(truth_theta[dm_mask])

    acts_truth = truth_d0[dm_mask]
    acts_pred = acts_reco[dm_mask, 0]
    acts_res = acts_pred - acts_truth
    acts_eta = _eta_from_theta(truth_theta[dm_mask])

    # Truth qop (from h5 if present, else shard fallback) for pT splits.
    truth_qop = None
    try:
        with h5py.File(args.predictions, "r") as f:
            if "qop" in f["targets"]:
                truth_qop = f["targets"]["qop"][:]
    except Exception:
        pass
    if truth_qop is None:
        pieces = []
        import json as _json
        with open(args.data_dir / "split.json") as f:
            shards = sorted(_json.load(f).get(args.split, []))
        for idx in shards:
            t = np.load(args.data_dir / f"shard_{idx:04d}" / "selected_tracks"
                        / "track_targets.npy")
            pieces.append(t[:, 4])
        truth_qop = np.concatenate(pieces, axis=0)
    pt_dm = (np.sin(np.clip(truth_theta[dm_mask], 1e-8, np.pi - 1e-8))
             / np.clip(np.abs(truth_qop[dm_mask]), 1e-8, None))

    # Iter-3sigma convergence — one pass each.
    ssm_conv = iterative_rms_convergence(ssm_res)
    acts_conv = iterative_rms_convergence(acts_res)
    print(
        f"  SSM  d0 iter-3σ RMS = {ssm_conv['rms']*1e3:.2f} μm "
        f"(kept {ssm_conv['frac_kept']*100:.1f}%, cut=[{ssm_conv['cut_lo']:.4f},{ssm_conv['cut_hi']:.4f}])"
    )
    print(
        f"  ACTS d0 iter-3σ RMS = {acts_conv['rms']*1e3:.2f} μm "
        f"(kept {acts_conv['frac_kept']*100:.1f}%, cut=[{acts_conv['cut_lo']:.4f},{acts_conv['cut_hi']:.4f}])"
    )

    # Plot 1 — heatmaps.
    out1 = args.output_dir / "heatmap_d0.png"
    plot_heatmaps(
        ssm_truth, ssm_pred, acts_truth, acts_pred,
        ssm_conv["cut_lo"], ssm_conv["cut_hi"],
        acts_conv["cut_lo"], acts_conv["cut_hi"],
        out1,
    )
    print(f"  wrote {out1}")

    # Plot 2 — residual histograms.
    out2 = args.output_dir / "d0_residuals.png"
    plot_residuals(
        ssm_res, acts_res,
        ssm_conv["cut_lo"], ssm_conv["cut_hi"],
        acts_conv["cut_lo"], acts_conv["cut_hi"],
        float(np.std(ssm_res)), float(np.std(acts_res)),
        ssm_conv["rms"], acts_conv["rms"],
        out2,
    )
    print(f"  wrote {out2}")

    # Plot 3 — RMS vs eta.
    out3 = args.output_dir / "d0_rms_vs_eta.png"
    plot_rms_vs_eta(ssm_res, ssm_eta, acts_res, acts_eta, out3,
                    ssm_pt=pt_dm, acts_pt=pt_dm)
    print(f"  wrote {out3}")

    # Plot 4 — full-range big picture (-2.5 to 2.5 mm).
    out4 = args.output_dir / "d0_fullrange.png"
    plot_fullrange(ssm_truth, ssm_pred, acts_pred, out4)
    print(f"  wrote {out4}")

    # SSM pred vs CKF pred correlation heatmap.
    out_corr = args.output_dir / "heatmap_ssm_vs_acts.png"
    plot_ssm_vs_acts(ssm_pred, acts_pred, out_corr)
    print(f"  wrote {out_corr}")

    # Collapse-mode quantification (SSM vs CKF).
    analyze_collapse_modes(ssm_truth, ssm_pred, acts_pred, args.output_dir)
    print(f"  wrote {args.output_dir / 'collapse_modes.txt'} + collapse_modes.png")

    # Plot 4b — full-range iter-3σ clipped heatmap (SSM vs CKF side by side).
    out4b = args.output_dir / "d0_fullrange_clipped.png"
    plot_fullrange_clipped(
        ssm_truth, ssm_pred, acts_truth, acts_pred,
        ssm_conv["cut_lo"], ssm_conv["cut_hi"],
        acts_conv["cut_lo"], acts_conv["cut_hi"],
        out4b,
    )
    print(f"  wrote {out4b}")

    # Plot 5 — SSM-only residuals (no ACTS → scale fits SSM's actual width).
    out5 = args.output_dir / "d0_residuals_ssm_only.png"
    plot_residuals_ssm_only(
        ssm_res,
        ssm_conv["cut_lo"], ssm_conv["cut_hi"],
        float(np.std(ssm_res)), ssm_conv["rms"],
        out5,
    )
    print(f"  wrote {out5}")

    # Plot 6 — SSM-only RMS vs η (so the SSM curve isn't flattened by ACTS
    # auto-scale dominating the shared y-axis in plot_rms_vs_eta).
    out6 = args.output_dir / "d0_rms_vs_eta_ssm_only.png"
    plot_rms_vs_eta_ssm_only(ssm_res, ssm_eta, out6)
    print(f"  wrote {out6}")


if __name__ == "__main__":
    main()
