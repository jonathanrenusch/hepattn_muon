#!/usr/bin/env python3
"""Evaluate track regression predictions: precision plots.

Reads the HDF5 file produced by ``RegressionPredictionWriter`` and generates
precision (σ of residual vs η) plots in two selection regimes (when ACTS
augmentation data is available):

1. **all_selected/** — All tracks reconstructed by the SSM vs all tracks
   reconstructed by ACTS (regardless of double-matching).

2. **double_matched/** — Only tracks that were double-matched by ACTS (>75%
   purity and >75% efficiency), comparing SSM and ACTS performance.

When ACTS augmentation data is not available, plots are saved directly to
the output directory without subdirectories (backwards-compatible).

Usage::

    python -m hepattn.experiments.colliderml_regr.evaluate_predictions \\
        --predictions /path/to/test_predictions.h5 \\
        --data-dir /scratch/colliderml/p0/p0_preprocessed \\
        --output-dir /path/to/eval_output

Without ACTS comparison (legacy mode)::

    python -m hepattn.experiments.colliderml_regr.evaluate_predictions \\
        --predictions /path/to/test_predictions.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================================
# Constants
# ============================================================================

PARAMS = ["d0", "z0", "phi", "theta", "qop"]

UNIT_SCALE = {
    "d0": 1.0,       # mm
    "z0": 1.0,       # mm
    "phi": 1e3,      # rad → mrad
    "theta": 1e3,    # rad → mrad
    "qop": 1.0,      # 1/GeV
}

PARAM_LABELS = {
    "d0": r"$\sigma(d_0)$ [mm]",
    "z0": r"$\sigma(z_0)$ [mm]",
    "phi": r"$\sigma(\phi)$ [mrad]",
    "theta": r"$\sigma(\theta)$ [mrad]",
    "qop": r"$\sigma(q/p)$ [1/GeV]",
}


# ============================================================================
# Data loading
# ============================================================================

def load_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load predictions and targets from HDF5 file.

    Returns ``{"preds": {name: array}, "targets": {name: array}}``.
    """
    data: dict[str, dict[str, np.ndarray]] = {"preds": {}, "targets": {}}
    with h5py.File(path, "r") as f:
        for group in ("preds", "targets"):
            for name in f[group]:
                data[group][name] = f[group][name][:]
    return data


def load_acts_augmentation(
    data_dir: Path,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load ACTS reco params and DM mask from preprocessed shards.

    Returns ``(acts_reco, acts_dm_mask)`` or ``None`` if not available.
    ``acts_reco`` has shape (N, 5) with columns [d0, z0, phi, theta, qop],
    NaN where ACTS had no matching track.
    ``acts_dm_mask`` has shape (N,) bool.
    """
    split_file = data_dir / "split.json"
    if not split_file.exists():
        return None

    with open(split_file) as f:
        splits = json.load(f)

    shard_indices = sorted(splits.get(split, []))
    if not shard_indices:
        return None

    all_reco, all_dm = [], []
    for idx in tqdm(shard_indices, desc="Loading ACTS augmentation", file=sys.stderr):
        sel_dir = data_dir / f"shard_{idx:04d}" / "selected_tracks"
        reco_file = sel_dir / "acts_reco.npy"
        dm_file = sel_dir / "acts_dm_mask.npy"
        if not reco_file.exists():
            return None  # Augmentation not available for this shard
        all_reco.append(np.load(reco_file))
        all_dm.append(np.load(dm_file))

    return np.concatenate(all_reco, axis=0), np.concatenate(all_dm, axis=0)


# ============================================================================
# Residuals
# ============================================================================

def compute_residuals(data: dict) -> dict[str, np.ndarray]:
    """Compute pred − truth residuals and truth eta for binning."""
    residuals: dict[str, np.ndarray] = {}
    for name in PARAMS:
        residuals[name] = data["preds"][name] - data["targets"][name]

    # Wrap phi residual to [-π, π]
    residuals["phi"] = (residuals["phi"] + np.pi) % (2 * np.pi) - np.pi

    # Truth eta from truth theta
    theta_truth = data["targets"]["theta"]
    residuals["eta"] = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

    return residuals


def compute_acts_residuals(
    acts_reco: np.ndarray,
    targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute ACTS residuals from (N, 5) reco array and target dict.

    Only includes tracks where ACTS had a match (non-NaN).
    Returns residuals dict with 'eta' and only the matched entries.
    """
    has_match = ~np.any(np.isnan(acts_reco), axis=1)
    reco = acts_reco[has_match]
    param_order = PARAMS  # [d0, z0, phi, theta, qop]

    residuals: dict[str, np.ndarray] = {}
    for i, name in enumerate(param_order):
        residuals[name] = reco[:, i] - targets[name][has_match]

    residuals["phi"] = (residuals["phi"] + np.pi) % (2 * np.pi) - np.pi

    theta_truth = targets["theta"][has_match]
    residuals["eta"] = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

    return residuals


def filter_residuals(
    residuals: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Filter residuals dict to a boolean or index mask."""
    return {k: v[mask] for k, v in residuals.items()}


# ============================================================================
# Step-plot helpers
# ============================================================================

def _build_step_arrays(bin_edges: np.ndarray, values: np.ndarray):
    """Convert bin centers/values to step-plot compatible arrays."""
    x = np.repeat(bin_edges, 2)
    y = np.repeat(values, 2)
    y = np.concatenate([[y[0]], y, [y[-1]]])
    return x, y[: len(x)]


def _step_fill(ax, edges, values, err, color, alpha=0.25):
    """Draw a step fill-between band."""
    x_fill = np.repeat(edges, 2)[1:-1]
    lo = np.repeat(np.maximum(values - err, 0), 2)
    hi = np.repeat(values + err, 2)
    ax.fill_between(x_fill, lo, hi, alpha=alpha, color=color)


def _edges_for_centers(eta_bins, centers):
    """Map bin centers back to bin edges from the full eta_bins array."""
    bin_indices = np.searchsorted(eta_bins, centers, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(eta_bins) - 2)
    return np.append(eta_bins[bin_indices], eta_bins[bin_indices[-1] + 1])


# ============================================================================
# Precision vs η
# ============================================================================

def compute_precision_vs_eta(
    residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Compute binned precision (std of residual) vs η.

    Returns ``(eta_bins, {param: {eta_centers, std, std_err, count, unbinned_std}})``.
    """
    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta = residuals["eta"]
    result: dict[str, dict[str, np.ndarray]] = {}

    for name in PARAMS:
        res = residuals[name]
        min_len = min(len(eta), len(res))
        e, r = eta[:min_len], res[:min_len]

        unbinned_std = float(np.std(r))

        centers, stds, counts = [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (e >= eta_bins[i]) & (e < eta_bins[i + 1])
            n = int(np.sum(mask))
            if n > 2:
                centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
                stds.append(np.std(r[mask]))
                counts.append(n)

        stds_arr = np.array(stds)
        counts_arr = np.array(counts, dtype=float)

        result[name] = {
            "eta_centers": np.array(centers),
            "std": stds_arr,
            "std_err": stds_arr / np.sqrt(2 * counts_arr),
            "count": counts_arr,
            "unbinned_std": unbinned_std,
        }

    return eta_bins, result


def plot_precision_vs_eta(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_data: dict | None = None,
    acts_label: str = "ACTS CKF",
) -> None:
    """Plot precision (σ of residual) vs η, optionally with ACTS overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in precision_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        centers = data["eta_centers"]
        std = data["std"]
        std_err = data["std_err"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        edges = _edges_for_centers(eta_bins, centers)
        x, y = _build_step_arrays(edges, std * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                label=f"{ml_label} (σ={unbinned:.4f})")
        _step_fill(ax, edges, std * scale, std_err * scale, "steelblue")

        if acts_data is not None and name in acts_data:
            a = acts_data[name]
            a_scale = UNIT_SCALE.get(name, 1.0)
            a_unbinned = a["unbinned_std"] * a_scale
            a_edges = _edges_for_centers(eta_bins, a["eta_centers"])
            ax_x, ax_y = _build_step_arrays(a_edges, a["std"] * a_scale)
            ax.step(ax_x, ax_y, where="post", color="darkorange", linewidth=2.5,
                    linestyle="--", label=f"{acts_label} (σ={a_unbinned:.4f})")
            _step_fill(ax, a_edges, a["std"] * a_scale, a["std_err"] * a_scale,
                       "darkorange")

        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$", fontsize=12)
        ax.set_ylabel(PARAM_LABELS.get(name, f"σ({name})"), fontsize=12)
        ax.set_title(f"{name.upper()} Precision vs $\\eta$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_precision_vs_eta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_summary_precision(precision_data, eta_bins, output_dir,
                            ml_label=ml_label, acts_data=acts_data, acts_label=acts_label)


def _plot_summary_precision(
    precision_data: dict,
    eta_bins: np.ndarray,
    output_dir: Path,
    ml_label: str = "SSM",
    acts_data: dict | None = None,
    acts_label: str = "ACTS CKF",
) -> None:
    """Summary 2×3 panel with all precision vs η curves."""
    params = [p for p in PARAMS if p in precision_data]
    if not params:
        return

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        data = precision_data[name]
        centers = data["eta_centers"]
        scale = UNIT_SCALE.get(name, 1.0)
        unbinned = data["unbinned_std"] * scale

        edges = _edges_for_centers(eta_bins, centers)
        x, y = _build_step_arrays(edges, data["std"] * scale)
        ax.step(x, y, where="post", color="steelblue", linewidth=2,
                label=f"{ml_label} (σ={unbinned:.4f})")

        if acts_data is not None and name in acts_data:
            a = acts_data[name]
            a_scale = UNIT_SCALE.get(name, 1.0)
            a_unbinned = a["unbinned_std"] * a_scale
            a_edges = _edges_for_centers(eta_bins, a["eta_centers"])
            ax_x, ax_y = _build_step_arrays(a_edges, a["std"] * a_scale)
            ax.step(ax_x, ax_y, where="post", color="darkorange", linewidth=2,
                    linestyle="--", label=f"{acts_label} (σ={a_unbinned:.4f})")
            ax.legend(fontsize=8)

        ax.set_xlabel(r"$\eta_{\mathrm{truth}}$")
        ax.set_ylabel(PARAM_LABELS.get(name, f"σ({name})"))
        ax.set_title(f"{name} ({ml_label} σ={unbinned:.4f})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)

    title = "Track Parameter Precision vs η"
    if acts_data:
        title += f"\n{ml_label} vs {acts_label}"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_precision_vs_eta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




# ============================================================================
# ACTS NPZ comparison (legacy, from evaluate_acts_tracking.py export)
# ============================================================================

def load_acts_precision_npz(
    path: str | Path,
) -> tuple[dict[str, dict[str, np.ndarray]], str] | None:
    """Load ACTS precision-vs-η from the NPZ written by evaluate_acts_tracking.py."""
    path = Path(path)
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=False)
    label_file = path.parent / (path.stem + "_label.txt")
    label = label_file.read_text().strip() if label_file.exists() else "ACTS (Selected, DM)"

    result: dict[str, dict[str, np.ndarray]] = {}
    for name in PARAMS:
        if f"{name}_std" not in npz:
            continue
        result[name] = {
            "eta_centers": npz[f"{name}_eta_centers"],
            "std": npz[f"{name}_std"],
            "std_err": npz.get(f"{name}_std_err", npz[f"{name}_std"] * 0.0),
            "count": npz.get(f"{name}_count", np.ones_like(npz[f"{name}_std"])),
            "unbinned_std": float(npz[f"{name}_unbinned_std"]),
        }
    return result, label


# ============================================================================
# Evaluation suite — runs all plots for one selection regime
# ============================================================================

def run_evaluation_suite(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    eta_range: tuple[float, float],
    n_eta_bins: int,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    acts_precision_npz: dict | None = None,
    acts_precision_label: str = "ACTS (Selected, DM)",
) -> dict[str, dict]:
    """Run precision analysis and save all plots.

    Returns precision data dict for printing summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute ACTS precision data if residuals provided
    acts_prec: dict | None = None
    if acts_residuals is not None:
        print("  Computing ACTS precision statistics ...")
        _, acts_prec = compute_precision_vs_eta(acts_residuals, eta_range, n_eta_bins)

    # If NPZ comparison is available but no shard-based ACTS, use NPZ for precision
    if acts_prec is None and acts_precision_npz is not None:
        acts_prec = acts_precision_npz
        acts_label = acts_precision_label

    # --- Precision vs η ---
    print("  Computing ML precision vs η ...")
    eta_bins, ml_precision = compute_precision_vs_eta(
        ml_residuals, eta_range=eta_range, n_eta_bins=n_eta_bins,
    )
    print("  Plotting precision vs η ...")
    plot_precision_vs_eta(ml_precision, eta_bins, output_dir,
                          ml_label=ml_label, acts_data=acts_prec, acts_label=acts_label)

    print(f"  Plots saved to {output_dir}")
    return {"ml_precision": ml_precision, "acts_precision": acts_prec}


def print_precision_summary(
    precision: dict[str, dict],
    label: str,
    acts_precision: dict | None = None,
    acts_label: str = "ACTS CKF",
) -> None:
    """Print table of unbinned σ values."""
    print(f"\n  {label} — parameter precisions (unbinned σ):")
    for name in PARAMS:
        if name not in precision:
            continue
        scale = UNIT_SCALE.get(name, 1.0)
        unit = {1.0: "", 1e3: " mrad"}.get(scale, "")
        ml_str = f"{precision[name]['unbinned_std'] * scale:.5f}{unit}"
        acts_str = ""
        if acts_precision and name in acts_precision:
            acts_str = f"  |  {acts_label}: {acts_precision[name]['unbinned_std'] * scale:.5f}{unit}"
        print(f"    {name:6s}: SSM σ = {ml_str}{acts_str}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate track regression predictions: precision and pull plots"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to test_predictions.h5 written by RegressionPredictionWriter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots (default: same dir as predictions file)",
    )
    parser.add_argument("--eta-min", type=float, default=-3.0)
    parser.add_argument("--eta-max", type=float, default=3.0)
    parser.add_argument("--n-eta-bins", type=int, default=30)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to preprocessed data root (e.g. /scratch/colliderml/p0/p0_preprocessed). "
             "If ACTS augmentation (acts_reco.npy) exists in the shards, enables "
             "comparison plots in all_selected/ and double_matched/ subdirectories.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to load ACTS data from (default: test)",
    )
    parser.add_argument(
        "--compare-acts-npz",
        type=str,
        default=None,
        help="Path to acts_precision_vs_eta.npz from evaluate_acts_tracking.py (legacy overlay)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "eval_plots"
    eta_range = (args.eta_min, args.eta_max)

    print(f"Loading predictions from {pred_path}")
    data = load_predictions(pred_path)
    n_tracks = len(data["preds"]["d0"])
    print(f"  Loaded {n_tracks:,} tracks")

    # Try to load ACTS augmentation from preprocessed shards
    acts_reco, acts_dm_mask = None, None
    if args.data_dir:
        result = load_acts_augmentation(Path(args.data_dir), split=args.split)
        if result is not None:
            acts_reco, acts_dm_mask = result
            n_matched = int(np.sum(~np.any(np.isnan(acts_reco), axis=1)))
            n_dm = int(np.sum(acts_dm_mask))
            print(f"  Loaded ACTS augmentation: {n_matched:,} ACTS-matched, "
                  f"{n_dm:,} double-matched out of {len(acts_reco):,} selected tracks")
            if len(acts_reco) != n_tracks:
                print(f"  WARNING: ACTS data has {len(acts_reco):,} tracks but HDF5 has "
                      f"{n_tracks:,}. Truncating to shorter.")
                min_n = min(len(acts_reco), n_tracks)
                acts_reco = acts_reco[:min_n]
                acts_dm_mask = acts_dm_mask[:min_n]
                for group in ("preds", "targets"):
                    for name in data[group]:
                        data[group][name] = data[group][name][:min_n]
        else:
            print(f"  ACTS augmentation not found in {args.data_dir} "
                  f"(missing acts_reco.npy in shards)")

    # Optional NPZ comparison (legacy path)
    acts_npz_data, acts_npz_label = None, "ACTS (Selected, DM)"
    if args.compare_acts_npz:
        result = load_acts_precision_npz(args.compare_acts_npz)
        if result is not None:
            acts_npz_data, acts_npz_label = result
            print(f"  Loaded ACTS NPZ comparison ({acts_npz_label})")

    # Compute ML residuals (all tracks)
    ml_residuals = compute_residuals(data)

    if acts_reco is not None:
        # ── Regime 1: all_selected ──
        # SSM: all tracks; ACTS: all tracks with a match (non-NaN)
        print("\n" + "=" * 70)
        print("Regime 1: ALL SELECTED TRACKS")
        print("=" * 70)

        all_acts_residuals = compute_acts_residuals(acts_reco, data["targets"])
        n_acts = len(all_acts_residuals["eta"])
        print(f"  SSM: {n_tracks:,} tracks | ACTS: {n_acts:,} matched tracks")

        r1 = run_evaluation_suite(
            ml_residuals, output_dir / "all_selected", eta_range, args.n_eta_bins,
            ml_label=f"SSM ({n_tracks:,} tracks)",
            acts_residuals=all_acts_residuals,
            acts_label=f"ACTS CKF ({n_acts:,} tracks)",
        )
        print_precision_summary(r1["ml_precision"], "All selected",
                                r1["acts_precision"], "ACTS CKF")

        # ── Regime 2: double_matched ──
        # Both SSM and ACTS filtered to double-matched tracks only
        print("\n" + "=" * 70)
        print("Regime 2: DOUBLE-MATCHED TRACKS ONLY")
        print("=" * 70)

        dm_mask = acts_dm_mask & ~np.any(np.isnan(acts_reco), axis=1)
        n_dm = int(np.sum(dm_mask))
        print(f"  {n_dm:,} double-matched tracks ({100 * n_dm / n_tracks:.1f}% of selected)")

        if n_dm > 0:
            dm_ml_label = f"SSM ({n_dm:,}/{n_tracks:,} DM tracks)"
            dm_acts_label = f"ACTS CKF ({n_dm:,}/{n_tracks:,} DM tracks)"
            dm_ml_residuals = filter_residuals(ml_residuals, dm_mask)

            # ACTS residuals for DM subset
            dm_acts_reco = acts_reco[dm_mask]
            dm_targets = {name: data["targets"][name][dm_mask] for name in PARAMS}
            dm_acts_residuals: dict[str, np.ndarray] = {}
            for i, name in enumerate(PARAMS):
                dm_acts_residuals[name] = dm_acts_reco[:, i] - dm_targets[name]
            dm_acts_residuals["phi"] = (
                (dm_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
            )
            dm_acts_residuals["eta"] = -np.log(
                np.tan(dm_targets["theta"] / 2.0 + 1e-12)
            )

            r2 = run_evaluation_suite(
                dm_ml_residuals, output_dir / "double_matched", eta_range, args.n_eta_bins,
                ml_label=dm_ml_label,
                acts_residuals=dm_acts_residuals,
                acts_label=dm_acts_label,
            )
            print_precision_summary(r2["ml_precision"], "Double-matched",
                                    r2["acts_precision"], "ACTS CKF")
        else:
            print("  No double-matched tracks found, skipping regime 2.")

    else:
        # No ACTS augmentation — legacy mode (single output dir)
        print("\nRunning in legacy mode (no ACTS comparison)")
        r = run_evaluation_suite(
            ml_residuals, output_dir, eta_range, args.n_eta_bins,
            acts_precision_npz=acts_npz_data,
            acts_precision_label=acts_npz_label,
        )
        print_precision_summary(r["ml_precision"], "All selected", r["acts_precision"])

    print(f"\nAll evaluation plots saved to {output_dir}")


if __name__ == "__main__":
    main()
