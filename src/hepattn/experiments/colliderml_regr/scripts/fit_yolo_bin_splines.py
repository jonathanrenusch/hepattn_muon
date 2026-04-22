#!/usr/bin/env python3
"""Fit monotonic PCHIP CDF splines for the YOLO binned-DFL parameters.

Produces three YAML spline configs under
``config/NeurIPS_retraining/v2/core_configs/splines/``:

    spline_z0.yaml         — z0 physical → [0, 1] empirical CDF
    spline_theta_eta.yaml  — η (derived from θ) → [0, 1] empirical CDF
    spline_qop.yaml        — q/p physical → [0, 1] empirical CDF

d0 is NOT fit here — the YOLO plan uses linear bins + overflow for d0
(CDF-warping would amplify the collapse attractor).  φ reuses the
existing ``spline_delta_phi.yaml``.

Usage::

    python fit_yolo_bin_splines.py                  # default: p0_core_pretrain, 40 shards
    python fit_yolo_bin_splines.py --all            # all 1000 shards
    python fit_yolo_bin_splines.py --num-shards 10  # fast smoke

The output schema matches ``MonotonicSplineTransform.from_config``:

    name: <param>
    units: <units>
    knot_x: [...]
    knot_y: [...]
"""

from __future__ import annotations

import argparse
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


# Target column layout in ``selected_tracks/track_targets.npy``
TARGET_COLS = {"d0": 0, "z0": 1, "phi": 2, "theta": 3, "qop": 4}

# Per-parameter knot strategy.  num_core uniformly-spaced quantiles +
# dense tail quantiles for tail resolution.
_STD_TAIL_Q = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
               0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]

PARAM_SPEC: dict[str, dict] = {
    "z0":        {"num_core": 50, "tail_q": _STD_TAIL_Q, "extra_q": [], "units": "mm"},
    "theta_eta": {"num_core": 50, "tail_q": _STD_TAIL_Q, "extra_q": [], "units": "eta (unitless)"},
    # q/p is bimodal (charge sign).  Densify near q/p = 0 transition band.
    "qop":       {
        "num_core": 50,
        "tail_q": _STD_TAIL_Q,
        "extra_q": [0.35, 0.4, 0.45, 0.48, 0.52, 0.55, 0.6, 0.65],
        "units": "1/GeV",
    },
}


def load_targets(preprocessed_dir: Path, num_shards: int) -> dict[str, np.ndarray]:
    """Load track_targets for z0, theta, qop and derive eta."""
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    if num_shards > 0:
        shard_dirs = shard_dirs[:num_shards]

    z0s: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    qops: list[np.ndarray] = []
    for sd in tqdm(shard_dirs, desc="Loading targets"):
        tgt_path = sd / "selected_tracks" / "track_targets.npy"
        if not tgt_path.exists():
            continue
        arr = np.load(tgt_path)
        if arr.size == 0:
            continue
        z0s.append(arr[:, TARGET_COLS["z0"]])
        thetas.append(arr[:, TARGET_COLS["theta"]])
        qops.append(arr[:, TARGET_COLS["qop"]])

    if not z0s:
        raise ValueError(f"No track_targets.npy found under {preprocessed_dir}")

    z0 = np.concatenate(z0s).astype(np.float64)
    theta = np.concatenate(thetas).astype(np.float64)
    qop = np.concatenate(qops).astype(np.float64)

    # η = -ln(tan(θ/2))  (matches the training-time η-space convention)
    # Guard against tan(0) ≈ 0 and tan(π/2) blow-up at the η=±3 edges.
    theta_safe = np.clip(theta, 1e-6, np.pi - 1e-6)
    eta = -np.log(np.tan(theta_safe / 2.0))

    print(f"Loaded {len(z0):,} tracks from {len(shard_dirs)} shards")
    print(f"  z0:   [{z0.min():.3f}, {z0.max():.3f}] mm,   std={z0.std():.3f}")
    print(f"  eta:  [{eta.min():.3f}, {eta.max():.3f}],     std={eta.std():.3f}")
    print(f"  qop:  [{qop.min():.4f}, {qop.max():.4f}] 1/GeV, std={qop.std():.4f}")

    return {"z0": z0, "theta_eta": eta, "qop": qop}


def compute_knots(values: np.ndarray, num_core: int,
                  tail_q: list[float], extra_q: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (knot_x, knot_y) with knot_y monotone in [0, 1]."""
    core_q = np.linspace(0.0, 1.0, num_core)
    all_q = np.unique(np.concatenate([core_q, tail_q, extra_q]))
    all_q = np.clip(all_q, 0.0, 1.0)
    all_q = np.sort(all_q)

    knot_x = np.quantile(values, all_q)

    # Remove duplicates (collisions in steep-CDF regions)
    _, unique_idx = np.unique(knot_x, return_index=True)
    unique_idx = np.sort(unique_idx)
    knot_x = knot_x[unique_idx]
    knot_y = all_q[unique_idx]

    # Exact endpoint anchoring (avoid float wobble at 0.0 / 1.0)
    knot_y[0] = 0.0
    knot_y[-1] = 1.0
    return knot_x, knot_y


def plot_diagnostics(values: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray,
                     slopes: np.ndarray, name: str, out_dir: Path) -> None:
    """Single-figure CDF + transformed-histogram diagnostic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} spline fit  (n={len(values):,})", fontsize=12, fontweight="bold")

    # Panel 1: empirical CDF vs fitted spline
    ax = axes[0]
    order = np.argsort(values)
    sorted_v = values[order]
    ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    fine_x = np.linspace(knot_x[0], knot_x[-1], 2000)
    spline_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)
    ax.plot(sorted_v, ecdf, "b-", alpha=0.3, linewidth=0.6, label="Empirical CDF")
    ax.plot(fine_x, spline_y, "r-", linewidth=1.5, label="PCHIP spline")
    ax.plot(knot_x, knot_y, "ko", markersize=3, label=f"Knots (n={len(knot_x)})", zorder=5)
    ax.set_xlabel(name)
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: histogram of transformed values, should be ~uniform
    ax = axes[1]
    transformed = evaluate_pchip(values, knot_x, knot_y, slopes)
    ax.hist(transformed, bins=60, density=True, alpha=0.7, color="steelblue",
            edgecolor="black", linewidth=0.3)
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.5, label="Ideal uniform")
    ax.set_xlabel(f"Transformed {name}")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"yolo_spline_{name}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: {out_path}")


def fit_and_save(name: str, values: np.ndarray, spec: dict, out_dir: Path) -> None:
    """Fit one param and save YAML + diagnostic plot."""
    print(f"\n[{name}] fitting with {spec['num_core']} core knots + "
          f"{len(spec['tail_q'])} tail + {len(spec['extra_q'])} extra")
    knot_x, knot_y = compute_knots(values, spec["num_core"], spec["tail_q"], spec["extra_q"])
    slopes = fritsch_carlson_slopes(knot_x, knot_y)

    # Spot-check monotonicity on a dense grid
    fine_x = np.linspace(knot_x[0], knot_x[-1], 5000)
    fine_y = evaluate_pchip(fine_x, knot_x, knot_y, slopes)
    violations = np.sum(np.diff(fine_y) < -1e-12)
    if violations > 0:
        print(f"  WARN: {violations} monotonicity violations on dense grid")
    else:
        print(f"  monotone OK ({len(knot_x)} knots)")

    # Write YAML
    yaml_path = out_dir / f"spline_{name}.yaml"
    payload = {
        "name": name,
        "units": spec["units"],
        "knot_x": [float(x) for x in knot_x],
        "knot_y": [float(y) for y in knot_y],
        "num_tracks": int(len(values)),
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"  wrote {yaml_path}  ({len(knot_x)} knots)")

    plot_diagnostics(values, knot_x, knot_y, slopes, name, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", type=Path,
                    default=Path("/scratch/colliderml/p0_core_pretrain"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "config/NeurIPS_retraining/v2/core_configs/splines")
    ap.add_argument("--num-shards", type=int, default=40,
                    help="Number of shards to sample from (default 40 ≈ 4%% of data)")
    ap.add_argument("--all", action="store_true",
                    help="Use all shards (overrides --num-shards)")
    args = ap.parse_args()

    num_shards = -1 if args.all else args.num_shards
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_targets(args.preprocessed_dir, num_shards)
    for name, values in data.items():
        fit_and_save(name, values, PARAM_SPEC[name], args.output_dir)


if __name__ == "__main__":
    main()
