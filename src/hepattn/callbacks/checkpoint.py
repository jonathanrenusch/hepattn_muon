from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self, monitor: str = "val/loss", save_top_k: int = 1, **kwargs) -> None:
        filename = "epoch={epoch:03d}-" + monitor.replace("/", "_") + "={" + monitor + ":.5f}"
        super().__init__(save_top_k=save_top_k, monitor=monitor, filename=filename, auto_insert_metric_name=False, **kwargs)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
        if stage == "fit":
            if trainer.fast_dev_run:
                return
            log_dir = Path(trainer.log_dir)
            self.dirpath = str(log_dir / "ckpts")
