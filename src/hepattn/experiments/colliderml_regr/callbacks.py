"""Custom callbacks for the colliderml_regr experiment."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
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
