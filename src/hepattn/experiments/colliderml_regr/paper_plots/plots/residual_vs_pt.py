"""Residual vs pT — 2×3 summary panel (5 params + η step hist), per-param singles
in individuals/.  Always after iter-3σ clip on the DM regime.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    iterative_rms_convergence,
)

from .. import save_fig
from ..stats import DISPLAY_SCALE, DISPLAY_UNIT
from ._panels import fill_eta_stephist, make_grid

PT_EDGES = np.logspace(np.log10(0.5), np.log10(30.0), 81)


def _resid_label(p: str) -> str:
    return rf"$\Delta {p}$ [{DISPLAY_UNIT[p]}]"


def _draw_one(ax, pt, r, p, *, fig=None):
    scale = DISPLAY_SCALE[p]
    cut = iterative_rms_convergence(r)
    keep = (r >= cut["cut_lo"]) & (r <= cut["cut_hi"])
    r_kept = r[keep] * scale
    pt_kept = pt[keep]
    win = max(3.0 * np.std(r_kept), 1e-9)
    res_edges = np.linspace(-win, win, 121)
    H, xe, ye = np.histogram2d(pt_kept, r_kept, bins=[PT_EDGES, res_edges])
    pc = ax.pcolormesh(xe, ye, H.T,
                       norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
                       cmap="viridis", shading="auto")
    ax.axhline(0.0, color="r", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 30.0)
    ax.set_ylim(-win, win)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(_resid_label(p))
    ax.set_title(p)
    if fig is not None:
        fig.colorbar(pc, ax=ax, shrink=0.8)


def _summary(res, model_key, label, plots_dir, stem):
    fig, axes = make_grid(figsize=(14.5, 8.5))
    pt = res["pt"]
    for i, p in enumerate(PARAMS):
        _draw_one(axes[i], pt, res[f"{model_key}_{p}"], p, fig=fig)
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(f"{label} residual vs $p_T$ — after iter-3σ clip, DM, N={res['count']:,}",
                 y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"
    pt = res["pt"]

    # Per-param singles for SSM and CKF
    for p in PARAMS:
        for who in ("ssm", "ckf"):
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            _draw_one(ax, pt, res[f"{who}_{p}"], p, fig=fig)
            ax.set_title(f"{who.upper()}: {p}")
            save_fig(fig, individuals, f"residual_vs_pt_{p}_{who}")

    # Summary 2×3 panels (one per model)
    _summary(res, "ssm", "SSM", plots_dir, "residual_vs_pt_summary_ssm")
    _summary(res, "ckf", "CKF", plots_dir, "residual_vs_pt_summary_ckf")
