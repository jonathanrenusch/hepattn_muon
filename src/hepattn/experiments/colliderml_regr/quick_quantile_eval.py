#!/usr/bin/env python3
"""Quick quantile evaluation on a small subset of test tracks.

Loads the model from a checkpoint, runs inference on ~10K tracks,
saves predictions+quantiles to a temporary HDF5, then runs the
quantile alignment analysis.

Usage::

    python -m hepattn.experiments.colliderml_regr.quick_quantile_eval \
        --ckpt /path/to/checkpoint.ckpt \
        --config /path/to/config.yaml \
        --output-dir /path/to/output \
        [--n-tracks 10000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from hepattn.experiments.colliderml_regr.data import (
    ColliderMLTrackDataset,
    collate_tracks,
)
from hepattn.experiments.colliderml_regr.evaluate_quantile_alignment import main as run_analysis_main
from hepattn.experiments.colliderml_regr.losses import (
    EtaQuantileLoss,
    QuantileLoss,
    SplineQuantileLoss,
)
from hepattn.experiments.colliderml_regr.model import TrackRegressionWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick quantile alignment evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml from the experiment")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--n-tracks", type=int, default=10000, help="Number of tracks to evaluate")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "quick_test_predictions.h5"

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine test shards from split.json
    data_dir = Path(config["data"]["preprocessed_dir"])
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    test_shards = sorted(splits["test"])

    print(f"Loading test data from {data_dir} (shards: {test_shards[:5]}...)")
    dataset = ColliderMLTrackDataset(data_dir, test_shards, load_acts=False)
    n_total = len(dataset)
    n_use = min(args.n_tracks, n_total)
    print(f"Total test tracks: {n_total}, using first {n_use}")

    # Take a contiguous subset
    subset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tracks,
        num_workers=0,
    )

    # Load model from checkpoint
    print(f"Loading model from {args.ckpt}")
    wrapper = TrackRegressionWrapper.load_from_checkpoint(args.ckpt, map_location="cpu")
    wrapper.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)

    # Discover quantile-enabled parameters
    loss_module = wrapper.model.loss_module
    quantile_levels: dict[str, np.ndarray] = {}
    for name in loss_module.parameter_order:
        loss_fn = loss_module.losses[name]
        if isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss, SplineQuantileLoss)):
            quantile_levels[name] = loss_fn.quantiles.cpu().numpy()

    print(f"Quantile parameters: {list(quantile_levels.keys())}")

    # Run inference
    all_preds: dict[str, list[np.ndarray]] = {n: [] for n in ["d0", "z0", "phi", "theta", "qop"]}
    all_targets: dict[str, list[np.ndarray]] = {n: [] for n in ["d0", "z0", "phi", "theta", "qop"]}
    all_quantiles: dict[str, list[np.ndarray]] = {n: [] for n in quantile_levels}

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = wrapper.model(inputs)
            preds = wrapper.model.predict(outputs)
            q_preds = loss_module.predict_quantiles(outputs["pred"].to(device))

            for name in ["d0", "z0", "phi", "theta", "qop"]:
                all_preds[name].append(preds[name].cpu().float().numpy().ravel())
                all_targets[name].append(targets[name].cpu().float().numpy().ravel())
                if name in quantile_levels:
                    all_quantiles[name].append(q_preds[name].cpu().float().numpy())

            n_done = sum(len(v) for v in all_preds["d0"])
            print(f"  Batch {batch_idx + 1}: {n_done} tracks processed")

    # Concatenate and save
    print(f"Saving predictions to {pred_path}")
    with h5py.File(pred_path, "w") as f:
        grp_p = f.create_group("preds")
        grp_t = f.create_group("targets")
        grp_q = f.create_group("quantiles")

        for name in ["d0", "z0", "phi", "theta", "qop"]:
            grp_p.create_dataset(name, data=np.concatenate(all_preds[name]),
                                 compression="gzip", compression_opts=1)
            grp_t.create_dataset(name, data=np.concatenate(all_targets[name]),
                                 compression="gzip", compression_opts=1)

        for name in quantile_levels:
            ds = grp_q.create_dataset(name, data=np.concatenate(all_quantiles[name], axis=0),
                                      compression="gzip", compression_opts=1)
            ds.attrs["levels"] = quantile_levels[name]

    print(f"Predictions saved. Running analysis...")

    # Run analysis using sys.argv override
    import sys
    sys.argv = [
        "evaluate_quantile_alignment",
        "--predictions", str(pred_path),
        "--output-dir", str(output_dir),
    ]
    run_analysis_main()


if __name__ == "__main__":
    main()
