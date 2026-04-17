#!/usr/bin/env python3
"""Quick profiling script: measure data loading vs GPU step time.

Runs a few training steps with the real model and data pipeline, timing:
  - DataLoader iteration (data→CPU)
  - Host→Device transfer
  - Forward + backward pass (GPU compute)
  - GPU utilization via nvidia-smi polling

Usage:
    python profile_dataload.py --config <config.yaml> \
        --device 0 --steps 60 --warmup 10 --num-train-shards 5
"""

import argparse
import os
import subprocess
import threading
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import numpy as np
import torch
import yaml

torch.set_float32_matmul_precision("high")


def poll_gpu_util(device_id: int, interval: float, stop_event: threading.Event, samples: list):
    """Background thread: poll nvidia-smi for GPU utilization."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={device_id}", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            )
            samples.append((time.perf_counter(), int(out.strip())))
        except Exception:
            pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--steps", type=int, default=60, help="Total steps to run")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps (excluded from stats)")
    parser.add_argument("--num-train-shards", type=int, default=5, help="Limit shards for speed")
    args = parser.parse_args()

    # ---- Load config chain (base.yaml + study config) ----
    from pathlib import Path

    config_path = Path(args.config)
    base_path = config_path.parent / "base.yaml"

    merged = {}
    if base_path.exists():
        with open(base_path) as f:
            merged = yaml.safe_load(f) or {}
    with open(config_path) as f:
        study = yaml.safe_load(f) or {}

    # Deep merge study into base
    def deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(merged, study)

    # ---- Build datamodule ----
    from hepattn.experiments.colliderml_regr.data import ColliderMLRegrDataModule

    data_cfg = merged.get("data", {})
    dm = ColliderMLRegrDataModule(
        preprocessed_dir=data_cfg.get("preprocessed_dir"),
        batch_size=data_cfg.get("batch_size", 2048),
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
        num_train_shards=args.num_train_shards,
        load_acts=data_cfg.get("load_acts", False),
        streaming=data_cfg.get("streaming", False),
        shard_buffer_size=data_cfg.get("shard_buffer_size", 8),
    )

    # Fake trainer attributes for setup/dataloader
    class FakeTrainer:
        current_epoch = 0
        max_epochs = 50
        limit_train_batches = 1.0

    dm.trainer = FakeTrainer()
    dm.setup("fit")
    train_dl = dm.train_dataloader()

    device = torch.device(f"cuda:{args.device}")

    # ---- Build model ----
    from hepattn.experiments.colliderml_regr.model import TrackParameterRegressor

    model_cfg = merged.get("model", {}).get("model", {}).get("init_args", {})

    # Build encoder from class_path
    enc_cfg = model_cfg.get("encoder", {})
    enc_class_path = enc_cfg.get("class_path", "")
    enc_init = enc_cfg.get("init_args", {})

    import importlib
    mod_name, cls_name = enc_class_path.rsplit(".", 1)
    enc_cls = getattr(importlib.import_module(mod_name), cls_name)
    encoder = enc_cls(**enc_init)

    # Build loss module
    loss_cfg = model_cfg.get("loss_module", {})
    loss_class_path = loss_cfg.get("class_path", "")
    loss_init = loss_cfg.get("init_args", {})
    mod_name, cls_name = loss_class_path.rsplit(".", 1)
    loss_cls = getattr(importlib.import_module(mod_name), cls_name)
    loss_module = loss_cls(**loss_init)

    # Filter model_cfg to only TrackParameterRegressor init args
    regressor_kwargs = {}
    import inspect
    sig = inspect.signature(TrackParameterRegressor.__init__)
    for k, v in model_cfg.items():
        if k in sig.parameters and k not in ("encoder", "loss_module"):
            regressor_kwargs[k] = v

    model = TrackParameterRegressor(encoder=encoder, loss_module=loss_module, **regressor_kwargs)
    model = model.to(device)
    model.train()

    # Dummy optimizer for backward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {data_cfg.get('batch_size', 2048)}")
    print(f"Num workers: {data_cfg.get('num_workers', 8)}")
    print(f"Streaming: {data_cfg.get('streaming', False)}")
    print(f"Num train shards: {args.num_train_shards}")
    print(f"Device: {device}")
    print(f"Steps: {args.steps} (warmup: {args.warmup})")
    print()

    # ---- Start GPU utilization polling ----
    gpu_samples = []
    stop_poll = threading.Event()
    poll_thread = threading.Thread(
        target=poll_gpu_util, args=(args.device, 0.1, stop_poll, gpu_samples), daemon=True
    )
    poll_thread.start()

    # ---- Run profiling loop ----
    data_times = []
    transfer_times = []
    compute_times = []
    step_times = []

    dl_iter = iter(train_dl)
    total_steps = args.steps

    t_step_start = time.perf_counter()

    for step in range(total_steps):
        # 1) Data loading (CPU)
        t0 = time.perf_counter()
        try:
            batch = next(dl_iter)
        except StopIteration:
            print(f"DataLoader exhausted at step {step}, restarting")
            dl_iter = iter(train_dl)
            batch = next(dl_iter)
        t_data = time.perf_counter() - t0

        inputs, targets = batch

        # 2) Host → Device transfer
        t0 = time.perf_counter()
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        torch.cuda.synchronize(device)
        t_transfer = time.perf_counter() - t0

        # 3) Forward + backward
        t0 = time.perf_counter()
        outputs = model(inputs)
        valid_mask = targets.get("track_valid")
        losses = model.compute_loss(outputs, targets, valid_mask=valid_mask)
        loss = losses["total"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        t_compute = time.perf_counter() - t0

        t_step = time.perf_counter() - t_step_start
        t_step_start = time.perf_counter()

        if step >= args.warmup:
            data_times.append(t_data)
            transfer_times.append(t_transfer)
            compute_times.append(t_compute)
            step_times.append(t_step)

        if step % 10 == 0 or step == total_steps - 1:
            print(
                f"Step {step:3d} | data: {t_data*1000:7.1f} ms | "
                f"xfer: {t_transfer*1000:5.1f} ms | "
                f"compute: {t_compute*1000:7.1f} ms | "
                f"total: {t_step*1000:7.1f} ms | "
                f"loss: {loss.item():.4f}"
            )

    # ---- Stop polling ----
    stop_poll.set()
    poll_thread.join(timeout=2)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("PROFILING RESULTS (excluding warmup)")
    print("=" * 70)

    def stats(arr, name, unit="ms"):
        a = np.array(arr) * 1000  # to ms
        print(f"  {name:20s}: mean={a.mean():7.1f} {unit}  "
              f"std={a.std():6.1f}  min={a.min():6.1f}  max={a.max():7.1f}  "
              f"p50={np.median(a):6.1f}  p95={np.percentile(a, 95):7.1f}")

    stats(data_times, "Data loading")
    stats(transfer_times, "H2D transfer")
    stats(compute_times, "GPU compute")
    stats(step_times, "Total step")

    total_data = np.sum(data_times) * 1000
    total_compute = np.sum(compute_times) * 1000
    total_step = np.sum(step_times) * 1000
    idle_frac = total_data / total_step * 100

    print(f"\n  Total data loading:  {total_data:8.0f} ms")
    print(f"  Total GPU compute:   {total_compute:8.0f} ms")
    print(f"  Total step time:     {total_step:8.0f} ms")
    print(f"  GPU idle fraction:   {idle_frac:8.1f} % (data / total)")
    print(f"  Data/compute ratio:  {total_data/total_compute:8.2f}x")

    # GPU utilization from nvidia-smi
    if gpu_samples:
        utils = [u for _, u in gpu_samples]
        print(f"\n  nvidia-smi GPU util: mean={np.mean(utils):.0f}%  "
              f"min={np.min(utils)}%  max={np.max(utils)}%  "
              f"samples={len(utils)}  zeros={sum(1 for u in utils if u == 0)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
