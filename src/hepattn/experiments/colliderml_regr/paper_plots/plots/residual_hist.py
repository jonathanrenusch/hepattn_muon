"""Residual histograms — 2×3 summary panel (5 params + η step hist).

Four summary figures per run, in plots/:
  residual_hist_summary_{linear,logy}_{preclip,postclip}.{pdf,png}
Per-param singles go into individuals/.
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


def _resid_label(p: str) -> str:
    return rf"$\Delta {p}$ [{DISPLAY_UNIT[p]}]"


def _prep(ssm, ckf, p, post_clip):
    scale = DISPLAY_SCALE[p]
    if post_clip:
        cs = iterative_rms_convergence(ssm)
        cc = iterative_rms_convergence(ckf)
        ssm = ssm[(ssm >= cs["cut_lo"]) & (ssm <= cs["cut_hi"])]
        ckf = ckf[(ckf >= cc["cut_lo"]) & (ckf <= cc["cut_hi"])]
        win = max(np.std(ssm), np.std(ckf)) * 4.0 * scale
    else:
        win = max(np.std(ssm), np.std(ckf)) * 6.0 * scale
    return ssm * scale, ckf * scale, np.linspace(-win, win, 121)


def _draw_one(ax, ssm, ckf, bins, p, *, log_y, with_legend=False):
    ax.hist(ckf, bins=bins, histtype="step", color="C3", lw=1.4,
            label="CKF (ACTS)", density=True)
    ax.hist(ssm, bins=bins, histtype="step", color="C0", lw=1.4,
            label="SSM", density=True)
    ax.axvline(0, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel(_resid_label(p))
    ax.set_ylabel("density")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(p)
    if with_legend:
        ax.legend(loc="upper right", fontsize=8.5)


def _summary(res, *, post_clip, log_y, plots_dir, stem, title):
    fig, axes = make_grid()
    for i, p in enumerate(PARAMS):
        ssm, ckf, bins = _prep(res[f"ssm_{p}"], res[f"ckf_{p}"], p, post_clip)
        _draw_one(axes[i], ssm, ckf, bins, p, log_y=log_y, with_legend=(i == 0))
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def _individual(res, *, post_clip, log_y, plots_dir, stem_fmt):
    for p in PARAMS:
        ssm, ckf, bins = _prep(res[f"ssm_{p}"], res[f"ckf_{p}"], p, post_clip)
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        _draw_one(ax, ssm, ckf, bins, p, log_y=log_y, with_legend=True)
        save_fig(fig, plots_dir, stem_fmt.format(p=p))


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"
    n = res["count"]
    base = f"DM, N={n:,}"

    variants = [
        # (post_clip, log_y, stem, title)
        (False, False, "residual_hist_summary_linear_preclip",
         f"Residuals (linear y, pre-clip) — {base}"),
        (True, False, "residual_hist_summary_linear_postclip",
         f"Residuals (linear y, post iter-3σ) — {base}"),
        (False, True, "residual_hist_summary_logy_preclip",
         f"Residuals (log y, pre-clip — tail comparison) — {base}"),
        (True, True, "residual_hist_summary_logy_postclip",
         f"Residuals (log y, post iter-3σ) — {base}"),
    ]
    for post, logy, stem, title in variants:
        _summary(res, post_clip=post, log_y=logy, plots_dir=plots_dir,
                 stem=stem, title=title)
        # And per-param singles
        single_stem = stem.replace("residual_hist_summary",
                                   "residual_hist") + "_{p}"
        _individual(res, post_clip=post, log_y=logy, plots_dir=individuals,
                    stem_fmt=single_stem)
