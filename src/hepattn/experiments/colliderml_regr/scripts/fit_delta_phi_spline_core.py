#!/usr/bin/env python3
"""Fit a monotonic PCHIP spline to the Δφ = wrap(φ − φ_innermost_hit) distribution.

Used together with :class:`SplineQuantileLoss` for the phi parameter in the
``ultimate_quantile`` configs.  Unlike d0 (peaked-at-zero, heavy-tailed), the
Δφ marginal is **bimodal** — two symmetric peaks at ≈ ±0.02 rad from the
magnetic-field bend between IP and the innermost measured hit, with a ~10×
dip at 0 and a fast-decaying shoulder to ±0.1.

Strategy:
    - CDF spline naturally allocates output resolution where tracks actually
      live (the peaks) via steep-CDF regions; the dip at 0 gets a flat CDF
      stretch.  Quantile-based knot placement handles this automatically.
    - Dense tail quantiles keep the rare outliers (47/4.8M beyond ±0.2)
      from breaking the monotone interpolant.

Critical validation: the physical-space round-trip RMS must be ≪ 0.5 mrad
(5e-4 rad), because that is the target phi resolution.  Script exits with
non-zero status if the max round-trip error approaches that floor.

Usage::

    python fit_delta_phi_spline_core.py                 # 50 shards default
    python fit_delta_phi_spline_core.py --num-shards 200
    python fit_delta_phi_spline_core.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.colliderml_regr.spline import (
    evaluate_pchip_np as evaluate_pchip,
    fritsch_carlson_slopes_np as fritsch_carlson_slopes,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("/scratch/colliderml/p200_core_finetune")
PHI_COL = 2        # track_targets columns: d0, z0, phi, theta, qop
PHI_HIT_COL = 4    # hit features: x, y, z, r, phi_hit, theta_hit, s, ...
S_COL = 6

# Target physical resolution.  Spline round-trip error must sit well below
# this so the spline itself is never the limiting factor.
# Spline physical support — matches the current linear norm bound so that
# the ~47-in-4.8M outliers beyond this get clamped to the endpoints by
# MonotonicSplineTransform.forward (which applies x.clamp(kx[0], kx[-1])),
# the same as what happens with norm_min/norm_max ± 0.2 today.
SPLINE_ABS_BOUND = 0.2

TARGET_RMS_RAD = 5.0e-4       # 0.5 mrad — desired model-level phi RMS.
# The spline's round-trip error adds in quadrature on top of the model's
# regression residual.  For the spline contribution to stay under ~1% of
# the final RMS we need σ_spline ≤ TARGET_RMS / 10 — i.e. RMS well under
# 50 µrad.  (At 50 µrad: √(500² + 50²) = 502.5 µrad → 0.5% penalty.)
FAIL_IF_RMS_ABOVE = 5.0e-5    # 50 µrad — ≤ 10% of target RMS in quadrature
# The bulk-max is a single-track worst case, not an RMS contributor, so we
# only require it to stay below the target itself — crossing 500 µrad in
# the worst-case track means that track is spline-floor-limited.
FAIL_IF_MAX_ABOVE = 5.0e-4    # 0.5 mrad

# Physical-space shoulder anchors: ensure the PCHIP has enough knots in the
# low-density shoulder region (|Δφ| ≈ 0.08–0.20 rad) where quantile-only
# placement puts very sparse knots.
SHOULDER_ANCHORS_RAD: list[float] = [
    -0.20, -0.18, -0.16, -0.15, -0.14, -0.13, -0.12, -0.11,
    -0.10, -0.09, -0.08, -0.07, -0.06, -0.055, -0.05, -0.045,
    +0.045, +0.05, +0.055, +0.06, +0.07, +0.08, +0.09, +0.10,
    +0.11, +0.12, +0.13, +0.14, +0.15, +0.16, +0.18, +0.20,
]

# Knot strategy for Δφ.
#
# The distribution is bimodal with peaks at ≈ ±0.02 rad and a dip at 0.
# Quantile-based knot placement naturally clusters knots where density is
# high (the peaks) and spreads them where density is low (the dip + tails).
# We supplement the uniform-quantile backbone with:
#   - dense transition quantiles around the shoulders (CDF ~0.02–0.1 and
#     ~0.9–0.98) to capture the decay from peaks into tails;
#   - dense mid-CDF points (~0.45–0.55) to resolve the dip between peaks;
#   - extreme-tail quantiles (down to 1e-5, up to 1-1e-5) so the rare
#     outliers don't pin the endpoints at a single rare sample.
NUM_CORE_KNOTS = 80

TAIL_QUANTILES = [
    # Lower extreme tail
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
    # Lower tail / shoulder-to-tail
    1e-3, 2e-3, 5e-3,
    # Lower shoulder (peak → valley between peaks)
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10,
    # Mid-CDF — the valley between the two peaks
    0.40, 0.42, 0.44, 0.46, 0.48, 0.49,
    0.50, 0.51, 0.52, 0.54, 0.56, 0.58, 0.60,
    # Upper shoulder (mirror of lower)
    0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99,
    0.995, 0.998, 0.999,
    # Upper extreme tail
    0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999,
]


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return np.remainder(x + np.pi, 2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Data loading — replicates data.py's innermost-hit selection exactly.
# ---------------------------------------------------------------------------


def load_delta_phi(preprocessed_dir: Path, num_shards: int) -> np.ndarray:
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    chunks: list[np.ndarray] = []
    for sd in tqdm(shard_dirs, desc="Computing Δφ"):
        sel = sd / "selected_tracks"
        targets = np.load(sel / "track_targets.npy", mmap_mode="r")
        offsets = np.load(sel / "track_hit_offsets.npy", mmap_mode="r")
        hit_indices = np.load(sel / "track_hit_indices.npy", mmap_mode="r")
        hits = np.load(sd / "hits.npy", mmap_mode="r")
        n = targets.shape[0]
        if n == 0:
            continue

        phi_true = np.asarray(targets[:, PHI_COL], dtype=np.float64)
        off = np.asarray(offsets)
        hidx = np.asarray(hit_indices)
        hits_arr = np.asarray(hits)
        inner_phi = np.empty(n, dtype=np.float64)
        for t in range(n):
            s, e = int(off[t]), int(off[t + 1])
            if e <= s:
                inner_phi[t] = np.nan
                continue
            idx = hidx[s:e]
            svals = hits_arr[idx, S_COL]
            inner_phi[t] = hits_arr[int(idx[int(np.argmin(svals))]), PHI_HIT_COL]

        good = np.isfinite(inner_phi)
        chunks.append(_wrap_pi(phi_true[good] - inner_phi[good]).astype(np.float64))

    if not chunks:
        raise RuntimeError(f"No valid shards in {preprocessed_dir}")
    delta = np.concatenate(chunks)
    print(f"\nLoaded Δφ for {len(delta):,} tracks from {len(shard_dirs)} shards")
    return delta


# ---------------------------------------------------------------------------
# Knot placement
# ---------------------------------------------------------------------------


def compute_knots(
    values: np.ndarray,
    num_core: int,
    tail_quantiles: list[float],
    abs_bound: float,
    shoulder_anchors_rad: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit knots for a CDF spline restricted to ``|x| ≤ abs_bound``.

    Tracks outside the bound are discarded from the fit — they will be
    clamped to τ ∈ {0, 1} at evaluation time by
    :meth:`MonotonicSplineTransform.forward`, which mirrors the behaviour
    of the linear ``norm_min/norm_max`` bound used in the non-spline
    variant of the config.
    """
    mask = np.abs(values) <= abs_bound
    bulk = values[mask]
    n_dropped = int(values.size - bulk.size)
    print(f"    bulk-support fit: {bulk.size:,} tracks in [{-abs_bound:+.2f}, {+abs_bound:+.2f}] rad"
          f"  ({n_dropped:,} outliers discarded)")

    # Quantile-based knots on the bulk.  Exclude q=0 and q=1 (they would
    # land on bulk.min()/bulk.max(), which can be arbitrarily close to the
    # fixed ±abs_bound endpoints and create duplicate-y knots that make the
    # inverse spline singular).  Interior knots only; endpoints are pinned
    # separately below.
    core_q = np.linspace(0.0, 1.0, num_core + 2)[1:-1]
    tail_q = [q for q in tail_quantiles if 0.0 < q < 1.0]
    all_q = np.unique(np.concatenate([core_q, tail_q]))
    all_q = np.sort(np.clip(all_q, 1e-9, 1.0 - 1e-9))
    qk_x = np.quantile(bulk, all_q)
    qk_y = all_q.copy()

    # Physical-anchor knots — fill in the low-density shoulder where the
    # CDF is nearly flat and quantile-based placement leaves big gaps.
    # CDF at each anchor is the empirical fraction within the BULK.
    if shoulder_anchors_rad:
        ax = np.asarray(sorted([a for a in shoulder_anchors_rad
                                if abs(a) < abs_bound]), dtype=np.float64)
        ay = np.array([float(np.mean(bulk <= a)) for a in ax])
        knot_x = np.concatenate([qk_x, ax])
        knot_y = np.concatenate([qk_y, ay])
    else:
        knot_x = qk_x
        knot_y = qk_y

    # Sort, dedup, and enforce y-monotonicity
    order = np.argsort(knot_x)
    knot_x = knot_x[order]
    knot_y = knot_y[order]
    _, unique_idx = np.unique(knot_x, return_index=True)
    unique_idx = np.sort(unique_idx)
    knot_x = knot_x[unique_idx]
    knot_y = knot_y[unique_idx]
    knot_y = np.maximum.accumulate(knot_y)

    # Pin endpoints to exactly ±abs_bound with CDF 0 and 1.  With q=0/q=1
    # excluded above and shoulder anchors only at |x| < abs_bound, the
    # pre-endpoint knots are guaranteed to be strictly inside (-abs_bound,
    # +abs_bound), so we just prepend/append.
    knot_x = np.concatenate([[-abs_bound], knot_x, [+abs_bound]])
    knot_y = np.concatenate([[0.0], knot_y, [1.0]])
    return knot_x, knot_y


# ---------------------------------------------------------------------------
# Diagnostic plots (mirrors fit_d0_spline_core.py for direct comparability)
# ---------------------------------------------------------------------------


def plot_spline_fit(delta, knot_x, knot_y, slopes, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Δφ Spline Fit — Core Finetune ({len(delta):,} tracks, {len(knot_x)} knots)",
        fontsize=14, fontweight="bold",
    )

    ax = axes[0]
    sorted_d = np.sort(delta)
    ecdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    fine = np.linspace(knot_x[0], knot_x[-1], 5000)
    ax.plot(sorted_d[::100], ecdf[::100], "b-", alpha=0.4, lw=0.5, label="Empirical CDF")
    ax.plot(fine, evaluate_pchip(fine, knot_x, knot_y, slopes), "r-", lw=2, label="PCHIP spline")
    ax.plot(knot_x, knot_y, "ko", ms=3, label=f"Knots (n={len(knot_x)})")
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF vs Spline")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    transformed = evaluate_pchip(delta, knot_x, knot_y, slopes)
    ax.hist(transformed, bins=100, density=True, alpha=0.7,
            color="steelblue", edgecolor="black", lw=0.3)
    ax.axhline(1.0, color="r", ls="--", lw=1.5, label="Ideal uniform")
    ax.set_xlabel("Transformed Δφ (CDF space)")
    ax.set_ylabel("Density")
    ax.set_title("Transformed Histogram (should be flat)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    ax = axes[2]
    spline_at_data = evaluate_pchip(sorted_d, knot_x, knot_y, slopes)
    residuals = ecdf - spline_at_data
    step = max(1, len(residuals) // 10000)
    ax.plot(sorted_d[::step], residuals[::step], "b-", lw=0.5, alpha=0.6)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("Residual (ECDF − Spline)")
    ax.set_title("CDF Fit Residuals")
    ax.grid(True, alpha=0.3)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_err = float(np.max(np.abs(residuals)))
    stats = f"RMSE: {rmse:.6f}\nMax |err|: {max_err:.6f}\nKS stat: {max_err:.6f}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})

    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_spline_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delta_phi_spline_fit.png  (CDF-space RMSE={rmse:.6f}, KS={max_err:.6f})")


def plot_qq_uniform(delta, knot_x, knot_y, slopes, out_dir):
    transformed = np.sort(evaluate_pchip(delta, knot_x, knot_y, slopes))
    n = len(transformed)
    theoretical = np.linspace(0, 1, n)
    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, n // 5000)
    ax.scatter(theoretical[::step], transformed[::step], s=1, alpha=0.3, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Theoretical uniform quantiles")
    ax.set_ylabel("Empirical (transformed) quantiles")
    ax.set_title("Q-Q Plot: Spline-Transformed Δφ vs Uniform(0,1)")
    ks = float(np.max(np.abs(transformed - theoretical)))
    ax.text(0.05, 0.92, f"KS = {ks:.5f}", transform=ax.transAxes, fontsize=11,
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_qq_uniform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delta_phi_qq_uniform.png  (KS={ks:.5f})")


def plot_calibration(delta, knot_x, knot_y, slopes, out_dir):
    transformed = evaluate_pchip(delta, knot_x, knot_y, slopes)
    q_levels = np.linspace(0.01, 0.99, 200)
    observed = np.array([np.mean(transformed <= q) for q in q_levels])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(q_levels, observed, "-", lw=2, color="tab:blue", label="Δφ spline")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("Expected quantile level")
    ax.set_ylabel("Observed fraction below level")
    ax.set_title("Δφ Spline Calibration")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    max_cal_err = float(np.max(np.abs(observed - q_levels)))
    ax.text(0.05, 0.92, f"Max cal. error: {max_cal_err:.5f}", transform=ax.transAxes,
            fontsize=11, bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delta_phi_calibration.png  (max cal err={max_cal_err:.5f})")


def plot_pdf_overlay(delta, knot_x, knot_y, slopes, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Δφ Spline PDF vs Histogram", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(delta, bins=300, density=True, alpha=0.4, color="steelblue",
            edgecolor="none", label="Histogram")
    fine = np.linspace(knot_x[0], knot_x[-1], 20000)
    fine_y = evaluate_pchip(fine, knot_x, knot_y, slopes)
    dx = fine[1] - fine[0]
    deriv = np.gradient(fine_y, dx)
    ax.plot(fine, deriv, "r-", lw=2, label="Spline dCDF/dx")
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("Density")
    ax.set_title("Full range")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    mask = np.abs(delta) < 0.1
    ax.hist(delta[mask], bins=300, density=True, alpha=0.4, color="steelblue",
            edgecolor="none", label="Histogram (|Δφ|<0.1)")
    inner = (fine >= -0.1) & (fine <= 0.1)
    ax.plot(fine[inner], deriv[inner], "r-", lw=2, label="Spline dCDF/dx")
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("Density")
    ax.set_title("Zoomed: |Δφ| < 0.1 rad (bimodal core)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_pdf_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: delta_phi_pdf_overlay.png")


def plot_transformed_hist(delta, knot_x, knot_y, slopes, out_dir):
    transformed = evaluate_pchip(delta, knot_x, knot_y, slopes)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, edges, _ = ax.hist(transformed, bins=100, density=True, alpha=0.7,
                                color="steelblue", edgecolor="black", lw=0.3)
    ax.axhline(1.0, color="r", ls="--", lw=1.5, label="Ideal uniform density")
    ax.set_xlabel("Transformed Δφ (CDF space)")
    ax.set_ylabel("Density")
    ax.set_title(f"Transformed Δφ Histogram — {len(delta):,} tracks, 100 bins")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    bin_counts = np.histogram(transformed, bins=100)[0]
    expected = len(delta) / 100
    chi2 = float(np.sum((bin_counts - expected) ** 2 / expected))
    max_dev = float(np.max(np.abs(counts - 1.0)))
    stats = f"Chi² (100 bins): {chi2:.1f}\nMax |density − 1|: {max_dev:.4f}"
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_transformed_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delta_phi_transformed_hist.png  (chi²={chi2:.1f}, max_dev={max_dev:.4f})")


# ---------------------------------------------------------------------------
# Physical-space round-trip validation (the KEY check for phi)
# ---------------------------------------------------------------------------


def _invert_spline(u: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    """Invert the PCHIP CDF spline — evaluates x(τ) via a PCHIP built on
    (knot_y → knot_x).  Matches MonotonicSplineTransform.inverse's construction
    (swap axes + Fritsch-Carlson slopes on the swapped ordinates).
    """
    inv_slopes = fritsch_carlson_slopes(knot_y, knot_x)
    return evaluate_pchip(u, knot_y, knot_x, inv_slopes)


BULK_ABS_BOUND = 0.2   # rad — where ≥99.999% of tracks live (≤1-in-100k outside)


def validate_roundtrip(
    delta: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    slopes: np.ndarray,
    out_dir: Path,
) -> tuple[float, float, float, float]:
    """Compute max/RMS round-trip error: x → CDF(x) → inverse(CDF(x)) vs x.

    This is what actually bounds the spline's contribution to phi resolution:
    at inference time the model produces τ ∈ [0, 1] and we invert via the
    spline.  Any interpolation error here sits directly on top of the
    regression residual in physical space.

    Returns ``(rms_bulk, max_bulk, rms_all, max_all)`` — we split on
    |Δφ| ≤ ``BULK_ABS_BOUND`` because the rare (≤1-in-100k) outliers beyond
    that fall in a single PCHIP segment that spans a physical range of
    several radians, so their round-trip error tells us nothing about the
    spline's usefulness for the regression we actually care about.
    """
    u = evaluate_pchip(delta, knot_x, knot_y, slopes)
    u = np.clip(u, 0.0, 1.0)
    roundtrip = _invert_spline(u, knot_x, knot_y)
    err = roundtrip - delta
    rms_all = float(np.sqrt(np.mean(err ** 2)))
    max_all = float(np.max(np.abs(err)))

    bulk_mask = np.abs(delta) <= BULK_ABS_BOUND
    err_bulk = err[bulk_mask]
    rms = float(np.sqrt(np.mean(err_bulk ** 2)))
    max_abs = float(np.max(np.abs(err_bulk)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Δφ Spline Round-Trip Error  (x → CDF → inverse)", fontsize=13, fontweight="bold")

    ax = axes[0]
    # Show error vs Δφ position.  Subsample for plotting.
    step = max(1, len(delta) // 20000)
    order = np.argsort(delta[::step])
    ax.plot(delta[::step][order], err[::step][order] * 1e6, "b-", lw=0.6, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(+TARGET_RMS_RAD * 1e6, color="crimson", ls="--", lw=1,
               label=f"target RMS floor = {TARGET_RMS_RAD*1e6:.0f} µrad")
    ax.axhline(-TARGET_RMS_RAD * 1e6, color="crimson", ls="--", lw=1)
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("round-trip error  [µrad]")
    ax.set_title("Error vs Δφ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(err * 1e6, bins=400, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("round-trip error  [µrad]")
    ax.set_ylabel("tracks per bin  (log)")
    ax.set_title("Error distribution")
    stats = (f"Bulk (|Δφ|≤{BULK_ABS_BOUND}):\n"
             f"  RMS:     {rms*1e6:.3f} µrad\n"
             f"  Max |e|: {max_abs*1e6:.3f} µrad\n"
             f"All tracks:\n"
             f"  RMS:     {rms_all*1e6:.3f} µrad\n"
             f"  Max |e|: {max_all*1e6:.3f} µrad\n"
             f"target:   {TARGET_RMS_RAD*1e6:.0f} µrad")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round", "alpha": 0.8, "facecolor": "wheat"})
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "delta_phi_roundtrip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delta_phi_roundtrip.png  "
          f"(bulk RMS={rms*1e6:.3f} µrad, bulk max={max_abs*1e6:.3f} µrad)")
    return rms, max_abs, rms_all, max_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preprocessed-dir", type=str, default=str(PREPROCESSED_DIR))
    ap.add_argument("--num-shards", type=int, default=50)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    if args.output_dir is None:
        out_dir = (
            Path(__file__).resolve().parent.parent
            / "config" / "NeurIPS_retraining" / "v2" / "core" / "splines"
        )
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_shards = -1 if args.all else args.num_shards

    print("=" * 70)
    print("Δφ Spline Fitting — Core Finetune Dataset")
    print("=" * 70)
    print(f"  Data     : {preprocessed_dir}")
    print(f"  Output   : {out_dir}")
    print(f"  Shards   : {'all' if num_shards == -1 else num_shards}")
    print(f"  Core kts : {NUM_CORE_KNOTS}")
    print(f"  Tail Qs  : {len(TAIL_QUANTILES)}")
    print()

    delta = load_delta_phi(preprocessed_dir, num_shards=num_shards)

    print("\n  Δφ distribution summary:")
    print(f"    range : [{delta.min():+.6f}, {delta.max():+.6f}] rad")
    print(f"    mean  : {delta.mean():+.6e}")
    print(f"    std   : {delta.std():.6e}")
    for p in [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]:
        print(f"    p{p:<6g}: {np.quantile(delta, p):+.6e}")
    print()

    print("  Fitting spline knots…")
    knot_x, knot_y = compute_knots(
        delta, NUM_CORE_KNOTS, TAIL_QUANTILES,
        abs_bound=SPLINE_ABS_BOUND,
        shoulder_anchors_rad=SHOULDER_ANCHORS_RAD,
    )
    slopes = fritsch_carlson_slopes(knot_x, knot_y)
    print(f"    total knots          : {len(knot_x)}")
    print(f"    x range              : [{knot_x[0]:+.6e}, {knot_x[-1]:+.6e}]")
    print(f"    knots with |x| < 0.01: {int(np.sum(np.abs(knot_x) < 0.01))}")
    print(f"    knots with |x| > 0.05: {int(np.sum(np.abs(knot_x) > 0.05))}")

    spline_at_knots = evaluate_pchip(knot_x, knot_x, knot_y, slopes)
    is_monotone = bool(np.all(np.diff(spline_at_knots) >= 0))
    print(f"    monotonicity         : {'PASS' if is_monotone else 'FAIL'}")
    if not is_monotone:
        print("  !! Non-monotone spline — aborting, check knot placement.")
        return 2

    # Write the spline config
    cfg = {
        "name": "delta_phi",
        "units": "rad",
        "knot_x": [float(v) for v in knot_x],
        "knot_y": [float(v) for v in knot_y],
        "num_tracks": int(len(delta)),
    }
    cfg_path = out_dir / "spline_delta_phi.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"    config saved         : {cfg_path}")
    print()

    print("  Generating diagnostic plots…")
    plot_spline_fit(delta, knot_x, knot_y, slopes, out_dir)
    plot_qq_uniform(delta, knot_x, knot_y, slopes, out_dir)
    plot_calibration(delta, knot_x, knot_y, slopes, out_dir)
    plot_pdf_overlay(delta, knot_x, knot_y, slopes, out_dir)
    plot_transformed_hist(delta, knot_x, knot_y, slopes, out_dir)

    print("\n  Validating physical-space round-trip error…")
    rms, max_abs, rms_all, max_all = validate_roundtrip(delta, knot_x, knot_y, slopes, out_dir)

    n_total = len(delta)
    n_bulk = int(np.sum(np.abs(delta) <= BULK_ABS_BOUND))
    print()
    print("=" * 70)
    print(f"  Target phi RMS                   : {TARGET_RMS_RAD*1e6:.0f} µrad")
    print(f"  Bulk (|Δφ| ≤ {BULK_ABS_BOUND}, {n_bulk/n_total*100:.4f}% of tracks):")
    print(f"    round-trip RMS                 : {rms*1e6:.3f} µrad")
    print(f"    round-trip max |e|             : {max_abs*1e6:.3f} µrad")
    print(f"  All tracks ({n_total:,}):")
    print(f"    round-trip RMS                 : {rms_all*1e6:.3f} µrad")
    print(f"    round-trip max |e| (tail-only) : {max_all*1e6:.3f} µrad  ← driven by "
          f"≤1-in-100k outliers")
    print(f"  Safety threshold on BULK RMS     : {FAIL_IF_RMS_ABOVE*1e6:.1f} µrad"
          f"  (= target / 10 — quadrature-adds ≤ 0.5%)")
    print(f"  Safety threshold on BULK max     : {FAIL_IF_MAX_ABOVE*1e6:.1f} µrad"
          f"  (= target — per-track worst case)")
    print("=" * 70)

    fail_rms = rms > FAIL_IF_RMS_ABOVE
    fail_max = max_abs > FAIL_IF_MAX_ABOVE
    if fail_rms or fail_max:
        if fail_rms:
            print(f"\n  FAIL: bulk RMS {rms*1e6:.3f} µrad exceeds safety threshold "
                  f"{FAIL_IF_RMS_ABOVE*1e6:.1f} µrad — spline would set the resolution floor.")
        if fail_max:
            print(f"\n  FAIL: bulk max {max_abs*1e6:.3f} µrad exceeds safety threshold "
                  f"{FAIL_IF_MAX_ABOVE*1e6:.1f} µrad — some bulk tracks would see spline-limited precision.")
        return 1

    print("\n  PASS: bulk round-trip RMS and max are both below their safety thresholds.")
    print(f"        Bulk RMS = {rms*1e6:.1f} µrad ≪ target = {TARGET_RMS_RAD*1e6:.0f} µrad.")
    print("        (Extreme-tail max is large but those tracks are un-resolvable by any\n"
          "         method and are <1-in-100k of the sample.)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
