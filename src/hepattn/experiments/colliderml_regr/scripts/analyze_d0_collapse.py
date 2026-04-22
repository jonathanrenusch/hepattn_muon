#!/usr/bin/env python3
"""Five-panel diagnostic for the d0 core-collapse artifact.

Panels:
  A  Mean d0 residual vs truth d0          (SSM vs CKF)
  B  RMS  d0 residual vs truth d0          (SSM vs CKF)
  C  Log-y distribution of d0 values       (truth, SSM pred, CKF pred)
  D  Pred vs truth heatmap for the SSM — the "cross" pattern that gives
     the collapse its visual signature
  E  Log-y histogram of d0 residuals       (SSM vs CKF precision)

Also prints a significance test for the pred-≈-0 collapse band, using the
SSM's own core σ as a per-track Gaussian null kernel.

Usage::

    python -m hepattn.experiments.colliderml_regr.scripts.analyze_d0_collapse \
        --predictions /path/to/test_predictions.h5 \
        --data-dir    /scratch/colliderml/p200_core_finetune \
        --output      /shared/tracking/logs/d0_collapse_analysis.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    load_acts_augmentation,
    load_predictions,
)


SSM_COLOR = "steelblue"
ACTS_COLOR = "darkorange"
TRUTH_COLOR = "black"
EPS_COLLAPSE = 0.005  # mm — pred band used for the collapse significance test


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------

def _binned_stats(truth_d0, residual, edges, n_min=2):
    means, mean_errs, rms, rms_errs = [], [], [], []
    for i in range(len(edges) - 1):
        m = (truth_d0 >= edges[i]) & (truth_d0 < edges[i + 1])
        n = int(m.sum())
        if n > n_min:
            r = residual[m]
            mu = float(np.mean(r))
            sd = float(np.std(r))
            rm = float(np.sqrt(np.mean(r * r)))
            means.append(mu); mean_errs.append(sd / np.sqrt(n))
            rms.append(rm); rms_errs.append(rm / np.sqrt(2 * n))
        else:
            means.append(np.nan); mean_errs.append(np.nan)
            rms.append(np.nan); rms_errs.append(np.nan)
    return (np.array(means), np.array(mean_errs), np.array(rms), np.array(rms_errs))


def panel_mean_residual(ax, truth_d0, ssm_res, acts_truth_d0, acts_res, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    s_mu, s_mu_err, _, _ = _binned_stats(truth_d0, ssm_res, edges)
    ax.errorbar(centers, s_mu, yerr=s_mu_err, fmt="^", color=SSM_COLOR,
                markersize=4.5, linewidth=1.2, capsize=2, label="SSM")
    if acts_truth_d0 is not None:
        a_mu, a_mu_err, _, _ = _binned_stats(acts_truth_d0, acts_res, edges)
        ax.errorbar(centers, a_mu, yerr=a_mu_err, fmt="o", color=ACTS_COLOR,
                    markersize=4.5, linewidth=1.2, capsize=2, label="ACTS CKF")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"truth $d_0$ [mm]")
    ax.set_ylabel(r"$\mu(\Delta d_0)$ [mm]")
    ax.set_title(r"Mean $d_0$ residual vs truth $d_0$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def panel_rms_residual(ax, truth_d0, ssm_res, acts_truth_d0, acts_res, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    _, _, s_rms, s_rms_err = _binned_stats(truth_d0, ssm_res, edges)
    ax.errorbar(centers, s_rms, yerr=s_rms_err, fmt="^", color=SSM_COLOR,
                markersize=4.5, linewidth=1.2, capsize=2, label="SSM")
    if acts_truth_d0 is not None:
        _, _, a_rms, a_rms_err = _binned_stats(acts_truth_d0, acts_res, edges)
        ax.errorbar(centers, a_rms, yerr=a_rms_err, fmt="o", color=ACTS_COLOR,
                    markersize=4.5, linewidth=1.2, capsize=2, label="ACTS CKF")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"truth $d_0$ [mm]")
    ax.set_ylabel(r"RMS$(\Delta d_0)$ [mm]")
    ax.set_title(r"RMS $d_0$ residual vs truth $d_0$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def panel_d0_distribution(ax, truth_d0, ssm_pred, acts_pred):
    lo, hi = -3.0, 3.0
    bins = np.linspace(lo, hi, 201)
    ax.hist(np.clip(truth_d0, lo, hi), bins=bins, histtype="step", linewidth=1.8,
            color=TRUTH_COLOR, label="truth")
    ax.hist(np.clip(ssm_pred, lo, hi), bins=bins, histtype="step", linewidth=1.6,
            color=SSM_COLOR, label="SSM prediction")
    if acts_pred is not None:
        ax.hist(np.clip(acts_pred, lo, hi), bins=bins, histtype="step",
                linewidth=1.6, color=ACTS_COLOR, label="ACTS CKF prediction")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_0$ [mm]")
    ax.set_ylabel("tracks per bin (log)")
    ax.set_title(r"$d_0$ value distributions (truth vs predictions)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)


def panel_pred_vs_truth_heatmap(ax, truth_d0, ssm_pred, lim=2.5, n_bins=150):
    """2-D histogram of SSM pred vs truth d0 — visualises the 'cross'."""
    from matplotlib.colors import LogNorm
    t = np.clip(truth_d0, -lim, lim)
    p = np.clip(ssm_pred, -lim, lim)
    bins = np.linspace(-lim, lim, n_bins + 1)
    h = ax.hist2d(t, p, bins=bins, norm=LogNorm(), cmin=1, cmap="viridis")
    plt.colorbar(h[3], ax=ax, label="tracks per cell (log)")
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.1, alpha=0.85,
            label="y = x")
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.55, linestyle="--")
    ax.axvline(0, color="white", linewidth=0.6, alpha=0.55, linestyle="--")
    ax.set_xlabel(r"truth $d_0$ [mm]")
    ax.set_ylabel(r"SSM predicted $d_0$ [mm]")
    ax.set_title(r"SSM $d_0$: prediction vs truth — the collapse cross")
    ax.legend(fontsize=9, loc="upper left")


def panel_residual_precision(ax, ssm_res, acts_res):
    """Log-y histogram of d0 residuals (precision comparison)."""
    combined = ssm_res if acts_res is None else np.concatenate([ssm_res, acts_res])
    lim = float(np.percentile(np.abs(combined), 99.9))
    lim = max(lim, 0.3)
    bins = np.linspace(-lim, lim, 301)
    ax.hist(np.clip(ssm_res, -lim, lim), bins=bins, histtype="step",
            linewidth=1.8, color=SSM_COLOR, label=f"SSM  (RMS={np.sqrt(np.mean(ssm_res ** 2)):.3f} mm)")
    if acts_res is not None:
        ax.hist(np.clip(acts_res, -lim, lim), bins=bins, histtype="step",
                linewidth=1.6, color=ACTS_COLOR,
                label=f"ACTS CKF  (RMS={np.sqrt(np.mean(acts_res ** 2)):.3f} mm)")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta d_0 = d_0^{\mathrm{pred}} - d_0^{\mathrm{truth}}$ [mm]")
    ax.set_ylabel("tracks per bin (log)")
    ax.set_title(r"$d_0$ residual precision — SSM vs CKF")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)


# ----------------------------------------------------------------------------
# Significance test
# ----------------------------------------------------------------------------

def compute_collapse_stats(truth_d0, ssm_pred, acts_pred, acts_mask, eps=EPS_COLLAPSE):
    """Return (sigma, per_band_rows, combined_row) for the pred-≈-0 collapse test."""
    calib_mask = np.abs(truth_d0) < 0.003
    sigma = float(np.std(ssm_pred[calib_mask]))

    bands = [(0.05, 0.10), (0.10, 0.30), (0.30, 1.00), (1.00, 2.50)]
    per_band = []
    for lo, hi in bands:
        m = (np.abs(truth_d0) >= lo) & (np.abs(truth_d0) < hi)
        n_b = int(m.sum())
        if n_b == 0:
            continue
        obs = int(np.sum(np.abs(ssm_pred[m]) < eps))
        p = norm.cdf(eps, loc=truth_d0[m], scale=sigma) - \
            norm.cdf(-eps, loc=truth_d0[m], scale=sigma)
        exp = float(p.sum())
        var = float((p * (1 - p)).sum())
        z = (obs - exp) / np.sqrt(var) if var > 0 else np.inf
        if acts_pred is not None:
            am = m & acts_mask
            acts_frac = (float(np.mean(np.abs(acts_pred[am]) < eps))
                         if am.sum() > 0 else float("nan"))
        else:
            acts_frac = float("nan")
        per_band.append({"lo": lo, "hi": hi, "n": n_b, "obs": obs,
                         "frac": obs / n_b, "exp": exp, "z": z,
                         "acts_frac": acts_frac})

    m_all = np.abs(truth_d0) >= 0.05
    n_all = int(m_all.sum())
    obs_all = int(np.sum(np.abs(ssm_pred[m_all]) < eps))
    p_all = norm.cdf(eps, loc=truth_d0[m_all], scale=sigma) - \
            norm.cdf(-eps, loc=truth_d0[m_all], scale=sigma)
    exp_all = float(p_all.sum())
    var_all = float((p_all * (1 - p_all)).sum())
    z_all = (obs_all - exp_all) / np.sqrt(var_all) if var_all > 0 else np.inf
    if acts_pred is not None and acts_mask is not None:
        am_all = m_all & acts_mask
        acts_frac_all = (float(np.mean(np.abs(acts_pred[am_all]) < eps))
                         if am_all.sum() > 0 else float("nan"))
    else:
        acts_frac_all = float("nan")
    combined = {"n": n_all, "obs": obs_all, "frac": obs_all / n_all,
                "exp": exp_all, "z": z_all, "acts_frac": acts_frac_all}
    return sigma, per_band, combined


def print_collapse_stats(sigma, per_band, combined, eps=EPS_COLLAPSE):
    print(f"\n[significance] SSM core σ at |truth|<0.003 mm: {sigma:.4f} mm")
    print(f"\nCollapse band |pred|<{eps} mm:")
    print(f"{'|truth| band':20s} {'N':>10s} {'coll':>10s} {'frac':>8s} "
          f"{'E(H0)':>10s} {'Z':>10s} {'ACTS_frac':>10s}")
    for r in per_band:
        print(f"{r['lo']:>4.2f}-{r['hi']:<4.2f}{'':10s} {r['n']:>10,d} "
              f"{r['obs']:>10,d} {r['frac']:>8.4f} {r['exp']:>10.2f} "
              f"{r['z']:>10.1f} {r['acts_frac']:>10.4f}")
    print(f"\nCombined (|truth d0|>=0.05 mm): {combined['obs']:,}/{combined['n']:,} "
          f"({combined['frac'] * 100:.2f}%), H0 expects {combined['exp']:.1f}, "
          f"Z={combined['z']:,.1f}")


def annotate_collapse_stats(ax, per_band, combined, eps=EPS_COLLAPSE):
    """Paste a compact collapse-severity box onto a panel."""
    lines = [
        r"$\bf{d_0\ collapse\ severity}$  (fraction predicted |$d_0$|$<$"
        + f"{eps*1e3:.0f} μm)",
        f"{'|truth d0| band':20s}  {'SSM':>7s}  {'ACTS':>7s}  {'× excess':>9s}",
    ]
    for r in per_band:
        excess = (r["frac"] / r["acts_frac"]) if r["acts_frac"] > 0 else float("inf")
        lines.append(
            f"{r['lo']:4.2f}–{r['hi']:<4.2f} mm         "
            f"{r['frac']*100:6.1f}%  {r['acts_frac']*100:6.2f}%  ×{excess:7.1f}"
        )
    excess_comb = (combined["frac"] / combined["acts_frac"]
                   if combined["acts_frac"] > 0 else float("inf"))
    lines.append(
        f"{'combined ≥ 0.05 mm':20s}  "
        f"{combined['frac']*100:6.2f}%  {combined['acts_frac']*100:6.2f}%  "
        f"×{excess_comb:7.0f}"
    )
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.5", alpha=0.92))


def annotate_problem_callout(ax, combined):
    """Big red text on the heatmap panel explaining the exact problem."""
    frac = combined["frac"] * 100
    acts = combined["acts_frac"] * 100
    msg = (
        "PROBLEM: for tracks whose true $d_0$ is clearly non-zero\n"
        "(|truth $d_0$| ≥ 0.05 mm), the SSM still predicts ≈ 0\n"
        f"for {frac:.1f}% of them — vs {acts:.2f}% for ACTS CKF.\n"
        "This is the horizontal band at pred = 0."
    )
    ax.text(0.98, 0.02, msg, transform=ax.transAxes,
            fontsize=9, color="darkred", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="darkred", alpha=0.95))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to SSM test_predictions.h5")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Preprocessed data dir for ACTS augmentation")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: alongside predictions)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dm-only", action="store_true",
                        help="Restrict to the ACTS double-matched subset "
                             "(requires --data-dir).")
    args = parser.parse_args()

    data = load_predictions(args.predictions)
    truth_d0 = np.asarray(data["targets"]["d0"], dtype=np.float64)
    ssm_pred = np.asarray(data["preds"]["d0"], dtype=np.float64)
    ssm_res = ssm_pred - truth_d0
    n = len(truth_d0)
    print(f"Loaded {n:,} SSM predictions from {args.predictions}")

    acts_pred = acts_res = acts_truth = acts_match = None
    acts_pred_masked = None
    regime_tag = "all_selected"
    if args.data_dir:
        loaded = load_acts_augmentation(args.data_dir, split=args.split)
        if loaded is not None:
            acts_reco, acts_dm_mask, _ = loaded
            if len(acts_reco) != n:
                k = min(len(acts_reco), n)
                acts_reco = acts_reco[:k]; acts_dm_mask = acts_dm_mask[:k]
                truth_d0 = truth_d0[:k]; ssm_pred = ssm_pred[:k]; ssm_res = ssm_res[:k]
            match = ~np.any(np.isnan(acts_reco), axis=1)

            if args.dm_only:
                dm = acts_dm_mask & match
                n_dm = int(dm.sum())
                print(f"  Filtering to double-matched subset: {n_dm:,} tracks "
                      f"(from {int(match.sum()):,} ACTS-matched, "
                      f"{len(acts_reco):,} selected)")
                truth_d0 = truth_d0[dm]
                ssm_pred = ssm_pred[dm]
                ssm_res = ssm_pred - truth_d0
                acts_pred = acts_reco[dm, PARAMS.index("d0")].astype(np.float64)
                acts_res = acts_pred - truth_d0
                acts_pred_masked = acts_pred
                acts_match = np.ones_like(truth_d0, dtype=bool)
                regime_tag = "double_matched"
                n = n_dm
            else:
                acts_match = match
                acts_pred = acts_reco[:, PARAMS.index("d0")].astype(np.float64)
                acts_truth = truth_d0[acts_match]
                acts_res = acts_pred[acts_match] - acts_truth
                acts_pred_masked = acts_pred[acts_match]
                print(f"  ACTS matched: {int(acts_match.sum()):,}")
        else:
            print(f"  ACTS augmentation not found in {args.data_dir}")

    # Bin edges for panels A/B — pinned to typical truth d0 range used in other plots
    edges = np.linspace(-2.5, 2.5, 41)

    fig, axd = plt.subplot_mosaic(
        [["A", "B", "C"],
         ["D", "E", "E"]],
        figsize=(20, 11),
        gridspec_kw={"hspace": 0.32, "wspace": 0.28},
    )

    # In DM-only mode SSM and ACTS share the same filtered tracks, so reuse truth_d0
    acts_bin_var = acts_truth if acts_truth is not None else (truth_d0 if args.dm_only else None)
    panel_mean_residual(axd["A"], truth_d0, ssm_res, acts_bin_var, acts_res, edges)
    panel_rms_residual(axd["B"], truth_d0, ssm_res, acts_bin_var, acts_res, edges)
    panel_d0_distribution(axd["C"], truth_d0, ssm_pred,
                          acts_pred_masked if acts_pred is not None else None)
    panel_pred_vs_truth_heatmap(axd["D"], truth_d0, ssm_pred)
    panel_residual_precision(axd["E"], ssm_res, acts_res)

    if acts_pred is not None:
        acts_for_stats, mask_for_stats = acts_pred, acts_match
    else:
        acts_for_stats, mask_for_stats = None, None
    sigma, per_band, combined = compute_collapse_stats(
        truth_d0, ssm_pred, acts_for_stats, mask_for_stats)
    annotate_collapse_stats(axd["E"], per_band, combined)
    annotate_problem_callout(axd["D"], combined)

    title_bits = [f"d0 collapse diagnostic — {regime_tag}", f"{n:,} tracks"]
    fig.suptitle("    ·    ".join(title_bits), fontsize=14, y=0.99)

    out = args.output
    if out is None:
        out = args.predictions.parent / "d0_collapse_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out}")

    abs_d0 = np.abs(truth_d0)
    print("\nTruth |d0| concentration (reference):")
    for q in (0.50, 0.68, 0.90, 0.95, 0.99):
        v = float(np.quantile(abs_d0, q))
        unit = f"{v*1e3:.2f} μm" if v < 1e-2 else f"{v:.3f} mm"
        print(f"  {int(q*100):2d}% of tracks have |truth d0| < {unit}")

    print_collapse_stats(sigma, per_band, combined)


if __name__ == "__main__":
    main()
