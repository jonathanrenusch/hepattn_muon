"""Heatmap pred vs truth, per-param + 2×3 summary (one figure each for SSM, CKF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    PARAM_VALUE_LABELS,
)

from .. import save_fig
from ._panels import fill_eta_stephist, make_grid

HEATMAP_RANGE = {
    "d0": (-1.0, 1.0),
    "z0": (-100.0, 100.0),
    "phi": (-np.pi, np.pi),
    "theta": (0.2, np.pi - 0.2),
    "qop": (-0.5, 0.5),
}


def _draw_one(ax, truth, pred, p, *, with_colorbar=True, fig=None):
    lo, hi = HEATMAP_RANGE[p]
    bins = np.linspace(lo, hi, 121)
    H, xe, ye = np.histogram2d(truth, pred, bins=[bins, bins])
    pc = ax.pcolormesh(xe, ye, H.T,
                       norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
                       cmap="viridis", shading="auto")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel(f"truth {PARAM_VALUE_LABELS[p]}")
    ax.set_ylabel(f"pred {PARAM_VALUE_LABELS[p]}")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(p)
    if with_colorbar and fig is not None:
        fig.colorbar(pc, ax=ax, shrink=0.85)


def _summary(res: dict, model_key: str, label: str, plots_dir: Path,
             stem: str) -> None:
    fig, axes = make_grid(figsize=(14.5, 9.0))
    for i, p in enumerate(PARAMS):
        _draw_one(axes[i], res[f"truth_{p}"], res[f"pred_{model_key}_{p}"], p,
                  with_colorbar=True, fig=fig)
        # heatmap aspect=equal can clip subplot — relax for grid:
        axes[i].set_aspect("auto")
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(f"{label}: prediction vs truth — DM, N={res['count']:,}", y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"

    # Per-param singles
    for p in PARAMS:
        for who, key in (("ssm", "ssm"), ("ckf", "ckf")):
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            _draw_one(ax, res[f"truth_{p}"], res[f"pred_{key}_{p}"], p,
                      with_colorbar=True, fig=fig)
            ax.set_title(f"{who.upper()}: {p}")
            save_fig(fig, individuals, f"heatmap_pred_vs_truth_{p}_{who}")

    # Summary 2×3 panels
    _summary(res, "ssm", "SSM", plots_dir, "heatmap_pred_vs_truth_summary_ssm")
    _summary(res, "ckf", "CKF", plots_dir, "heatmap_pred_vs_truth_summary_ckf")
