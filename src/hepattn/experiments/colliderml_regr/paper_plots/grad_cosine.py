"""Optional gradient-cosine-similarity plot — wraps scripts/gradient_cosine_analysis.py.

Resolves <output_root>/<nicename>/best.ckpt + config.yaml from the bundle and
writes ``plots/grad_cosine_scores.{pdf,png}`` plus ``grad_cosine_summary.txt``
into the SAME nicename dir, so reproducibility lives in one place.

Run separately from cli.py:
    python -m hepattn.experiments.colliderml_regr.paper_plots.grad_cosine \
        --nicename ssmcls_q7_p0pretrain_zeroshot_7972d00d_ep49 \
        [--n-batches 20] [--batch-size 2048]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import DATA_DIR, PAPER_PLOTS_ROOT


SCRIPT = Path(
    "/shared/tracking/hepattn_muon/src/hepattn/experiments/colliderml_regr/"
    "scripts/gradient_cosine_analysis.py"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nicename", required=True)
    p.add_argument("--output-root", default=str(PAPER_PLOTS_ROOT))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args(argv)

    bundle = Path(args.output_root) / args.nicename
    if not bundle.is_dir():
        raise SystemExit(f"bundle not found: {bundle} — run cli.py first")

    cfg = bundle / "config.yaml"
    ckpt = bundle / "best.ckpt"
    summary_dir = bundle  # grad_cosine_summary.txt at top level of bundle

    if not SCRIPT.exists():
        raise SystemExit(f"reference script not found: {SCRIPT}")

    grad_dir = bundle / "grad_cos"
    grad_dir.mkdir(exist_ok=True)
    cmd = [
        "pixi", "run", "python", str(SCRIPT),
        "--config", str(cfg),
        "--ckpt", str(ckpt),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(grad_dir),
        "--n-batches", str(args.n_batches),
        "--batch-size", str(args.batch_size),
    ]
    print("[grad_cosine]", " ".join(cmd))
    rc = subprocess.call(cmd, cwd="/shared/tracking/hepattn_muon",
                         env={"CUDA_VISIBLE_DEVICES": str(args.gpu), **__import__("os").environ})
    if rc != 0:
        raise SystemExit(f"gradient_cosine_analysis.py failed rc={rc}")

    # Mirror the summary.txt to bundle root for visibility
    src_summary = grad_dir / "summary.txt"
    if src_summary.exists():
        (summary_dir / "grad_cosine_summary.txt").write_text(src_summary.read_text())
    print(f"[grad_cosine] wrote into {grad_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
