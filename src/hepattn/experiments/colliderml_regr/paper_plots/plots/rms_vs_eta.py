"""RMS vs η, SSM vs CKF, pre-clip + iter-3σ, with bootstrap 2σ band.

- `individuals/rms_vs_eta_<p>.{pdf,png}` — per-param
- `rms_vs_eta_summary.{pdf,png}`        — 2×3 (5 params + η step hist)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    iterative_rms_convergence,
)

from .. import save_fig
from ..stats import DISPLAY_SCALE, DISPLAY_UNIT
from ._panels import fill_eta_stephist, make_grid


def _binwise(eta, res, edges, fn, n_boot, seed, min_n=30):
    rng = np.random.default_rng(seed)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.full(len(centers), np.nan)
    sg = np.full(len(centers), np.nan)
    bin_idx = np.clip(np.digitize(eta, edges) - 1, 0, len(centers) - 1)
    for b in range(len(centers)):
        sel = res[bin_idx == b]
        if len(sel) < min_n:
            continue
        mu[b] = fn(sel)
        if n_boot > 0:
            samps = np.empty(n_boot)
            for k in range(n_boot):
                idx = rng.integers(0, len(sel), len(sel))
                samps[k] = fn(sel[idx])
            sg[b] = np.std(samps, ddof=1)
        else:
            sg[b] = 0.0
    return mu, sg


def _draw_one(ax, eta, ssm, ckf, p, *, n_boot, scale):
    edges = np.linspace(-3.0, 3.0, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    iter_rms = lambda x: iterative_rms_convergence(x)["rms"]

    ssm_post, ssm_post_s = _binwise(eta, ssm, edges, iter_rms, n_boot, seed=12)
    ssm_pre,  ssm_pre_s  = _binwise(eta, ssm, edges, raw_rms,  n_boot, seed=11)
    ckf_post, ckf_post_s = _binwise(eta, ckf, edges, iter_rms, n_boot, seed=14)
    ckf_pre,  ckf_pre_s  = _binwise(eta, ckf, edges, raw_rms,  n_boot, seed=13)

    # 2σ bands (95% CI under bootstrap normal approx)
    ax.plot(centers, ssm_post * scale, "-", color="C0", lw=1.8, label="SSM (iter-3σ)")
    ax.fill_between(centers, (ssm_post - 2 * ssm_post_s) * scale,
                    (ssm_post + 2 * ssm_post_s) * scale, color="C0", alpha=0.20)
    ax.plot(centers, ssm_pre * scale, "--", color="C0", lw=1.0, alpha=0.7,
            label="SSM (pre-clip RMS)")
    ax.plot(centers, ckf_post * scale, "-", color="C3", lw=1.8, label="CKF (iter-3σ)")
    ax.fill_between(centers, (ckf_post - 2 * ckf_post_s) * scale,
                    (ckf_post + 2 * ckf_post_s) * scale, color="C3", alpha=0.20)
    ax.plot(centers, ckf_pre * scale, "--", color="C3", lw=1.0, alpha=0.7,
            label="CKF (pre-clip RMS)")
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel(f"RMS [{DISPLAY_UNIT[p]}]")
    ax.set_yscale("log")
    ax.set_xlim(-3, 3)
    ax.set_title(p)


def make(res: dict, plots_dir: Path, *, n_boot: int = 50) -> None:
    individuals = plots_dir / "individuals"
    eta = res["eta"]

    # Per-param singles
    for p in PARAMS:
        scale = DISPLAY_SCALE[p]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        _draw_one(ax, eta, res[f"ssm_{p}"], res[f"ckf_{p}"], p, n_boot=n_boot, scale=scale)
        ax.legend(ncol=2, fontsize=8.5)
        save_fig(fig, individuals, f"rms_vs_eta_{p}")

    # Summary 2×3
    fig, axes = make_grid()
    for i, p in enumerate(PARAMS):
        scale = DISPLAY_SCALE[p]
        _draw_one(axes[i], eta, res[f"ssm_{p}"], res[f"ckf_{p}"], p,
                  n_boot=n_boot, scale=scale)
        if i == 0:
            axes[i].legend(ncol=2, fontsize=8.0)
    fill_eta_stephist(axes[5], eta)
    fig.suptitle(f"RMS vs η — bands = bootstrap ±2σ — DM, N={res['count']:,}",
                 y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, "rms_vs_eta_summary")
