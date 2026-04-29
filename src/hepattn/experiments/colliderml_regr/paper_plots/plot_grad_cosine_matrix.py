"""Single-panel cosine matrix figure (PDF + PNG) consistent with the paper style.

Loads ``grad_cosines.npz`` written by the upstream
``scripts/gradient_cosine_analysis.py`` and produces ONE 5×5 heatmap with
``mean ± 2σ`` annotated in each cell.  No second std panel.

Usage:
    python -m hepattn.experiments.colliderml_regr.paper_plots.plot_grad_cosine_matrix \
        --nicename <bundle-dir-name>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import PAPER_PLOTS_ROOT, apply_paper_style, save_fig


def _read_batch_size(summary_path: Path) -> int | None:
    """Extract ``BS=NNNN`` from the upstream summary.txt if present."""
    if not summary_path.exists():
        return None
    m = re.search(r"BS\s*=\s*(\d+)", summary_path.read_text())
    return int(m.group(1)) if m else None


def make(npz_path: Path, out_dir: Path, *, stem: str = "cos_heatmap",
         batch_size: int | None = None) -> None:
    apply_paper_style()
    data = np.load(npz_path, allow_pickle=True)
    cos = np.asarray(data["cos_per_batch"], dtype=np.float64)  # (n_batches, P, P)
    params = [str(s) for s in data["params"]]
    P = len(params)
    if batch_size is None:
        batch_size = _read_batch_size(out_dir / "summary.txt")

    mean = cos.mean(axis=0)
    n = cos.shape[0]
    # 2σ on the **mean estimate**, not on individual-batch values.
    # std_across_batches characterises batch-to-batch variability and does
    # not shrink with N; the standard error on the mean is std / √N.
    sem = (cos.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # Diverging colormap consistent with paper convention: positive cosine → blue
    # (the SSM-side colour in line plots), negative → red.
    # Mask the diagonal (always +1 by construction): keeping it on the colour
    # scale would force vmin/vmax to ±1 and squash the small off-diagonal
    # signal flat.  Tight vmin/vmax = ±0.2 so off-diagonal contrast is legible.
    off = mean.copy()
    np.fill_diagonal(off, np.nan)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.92")  # light grey — diagonal cells, neutral on the eye
    im = ax.imshow(off, cmap=cmap, vmin=-0.2, vmax=0.2,
                   interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("mean cosine")

    ax.set_xticks(range(P))
    ax.set_xticklabels(params)
    ax.set_yticks(range(P))
    ax.set_yticklabels(params)
    ax.set_title("Per-parameter loss-gradient cosine on shared trunk")
    # Cell separators only (no coordinate grid).  Minor ticks at half-integers
    # gives a clean light separator between boxes.
    ax.grid(False, which="major")
    ax.set_xticks(np.arange(P + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(P + 1) - 0.5, minor=True)
    ax.grid(True, which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    # Annotate every cell with black text — colour-switch was uneven on the
    # ±0.2 RdBu scale (some cells flipped to white when |μ| > 0.13).
    for i in range(P):
        for j in range(P):
            m = mean[i, j]
            s2 = 2.0 * sem[i, j]
            txt = f"{m:+.2f}" if i == j else f"{m:+.2f} ± {s2:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black",
                    fontsize=6.5)
    n_batches = n

    if batch_size is not None:
        n_grad = n_batches * batch_size
        n_str = f"N = {n_grad:,} gradient samples"
    else:
        n_str = f"{n_batches} minibatches"
    fig.text(0.5, 0.005,
             f"Annotations: mean ± 2σ over {n_str}  ·  "
             "diagonal = 1 by construction",
             ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, out_dir, stem)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nicename", required=True)
    p.add_argument("--output-root", default=str(PAPER_PLOTS_ROOT))
    args = p.parse_args(argv)

    bundle = Path(args.output_root) / args.nicename
    npz = bundle / "grad_cos" / "grad_cosines.npz"
    if not npz.exists():
        raise SystemExit(f"npz not found: {npz} — run grad_cosine.py first")
    make(npz, bundle / "grad_cos")
    print(f"[plot_grad_cosine_matrix] wrote {bundle / 'grad_cos' / 'cos_heatmap'}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
