"""Custom callbacks for the colliderml_regr experiment."""

from __future__ import annotations

from pathlib import Path
import subprocess

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
import torch
from torch import Tensor


class RegressionPredictionWriter(Callback):
    """Write track regression predictions and targets to an HDF5 file.

    Activates only during the ``test`` stage. Each batch appends into
    growable 1-D datasets so the file can be read as flat arrays by a
    downstream evaluation script.

    HDF5 layout::

        preds/{d0, z0, phi, theta, qop}   — (N,) float32
        targets/{d0, z0, phi, theta, qop}  — (N,) float32
    """

    def __init__(self) -> None:
        super().__init__()
        self.file: h5py.File | None = None
        self._output_path: Path | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        log_dir = Path(trainer.log_dir)
        self._output_path = log_dir / "test_predictions.h5"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self._output_path, "w")

        # Create resizable datasets (unknown total size)
        self.file.create_group("preds")
        self.file.create_group("targets")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.file is None:
            return

        preds: dict[str, Tensor] = outputs["preds"]
        targets: dict[str, Tensor] = outputs["targets"]

        for name in ["d0", "z0", "phi", "theta", "qop"]:
            p = preds[name].detach().float().cpu().numpy().ravel()
            t = targets[name].detach().float().cpu().numpy().ravel()

            # Append to datasets (create on first batch, resize on subsequent)
            for group_name, arr in [("preds", p), ("targets", t)]:
                grp = self.file[group_name]
                if name not in grp:
                    grp.create_dataset(
                        name,
                        data=arr,
                        maxshape=(None,),
                        chunks=True,
                        compression="gzip",
                        compression_opts=1,
                    )
                else:
                    ds = grp[name]
                    old_len = ds.shape[0]
                    ds.resize(old_len + len(arr), axis=0)
                    ds[old_len:] = arr

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return
        if self.file is not None:
            self.file.close()
            self.file = None
        if self._output_path is not None:
            print("-" * 80)
            print(f"Predictions written to {self._output_path}")
            print("-" * 80)


class MinimalGpuMonitor(Callback):
    """Log only coarse GPU utilization and memory utilization metrics.

    Metrics are designed to mirror a compact subset of ``nvidia-smi``:
    - ``gpu/utilization_pct``
    - ``gpu/memory_utilization_pct``
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._sync_dist = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    @staticmethod
    def _query_nvidia_smi(device_idx: int) -> tuple[float, float] | None:
        """Return ``(gpu_util_pct, mem_util_pct)`` from nvidia-smi if available."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device_idx),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            line = result.stdout.strip().splitlines()[0]
            gpu_util_str, mem_used_str, mem_total_str = [x.strip() for x in line.split(",")]
            gpu_util = float(gpu_util_str)
            mem_used = float(mem_used_str)
            mem_total = float(mem_total_str)
            mem_util = 100.0 * mem_used / max(mem_total, 1.0)
            return gpu_util, mem_util
        except (IndexError, ValueError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _torch_memory_util(device_idx: int) -> float:
        free_b, total_b = torch.cuda.mem_get_info(device_idx)
        used_b = total_b - free_b
        return 100.0 * float(used_b) / max(float(total_b), 1.0)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
        if not torch.cuda.is_available() or self.log_every_n_steps <= 0:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        device_idx = torch.cuda.current_device()
        stats = self._query_nvidia_smi(device_idx)

        if stats is not None:
            gpu_util, mem_util = stats
            pl_module.log(
                "gpu/utilization_pct",
                gpu_util,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=self._sync_dist,
            )
        else:
            mem_util = self._torch_memory_util(device_idx)

        pl_module.log(
            "gpu/memory_utilization_pct",
            mem_util,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )
