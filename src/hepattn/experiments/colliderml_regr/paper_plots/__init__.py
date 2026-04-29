"""Unified paper-plot pipeline for the NeurIPS submission.

One CLI per Comet run id, producing a self-contained reproducibility bundle
under /shared/tracking/logs_Neurips/paper_plots/<nicename>/ with config copy,
checkpoint symlink, predictions h5 symlink, plots (PDF + PNG), and stats.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


PAPER_PLOTS_ROOT = Path("/shared/tracking/logs_Neurips/paper_plots")
DATA_DIR = Path("/scratch/colliderml/p200_core_finetune")
COMET_OFFLINE_ROOT = Path("/shared/tracking/hepattn_muon/src/logs/comet_offline")


def apply_paper_style() -> None:
    """Set rcParams for paper-grade figures."""
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # editable text in vector PDF
        "ps.fonttype": 42,
    })


def save_fig(fig, output_dir: Path | str, stem: str) -> None:
    """Save figure as both PDF (vector) and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
