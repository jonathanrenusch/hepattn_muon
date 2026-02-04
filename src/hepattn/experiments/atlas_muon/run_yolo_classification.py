#!/usr/bin/env python3
"""Training script for YOLO-style classification pretraining.

Stage 1 of the 3-stage training pipeline:
- Classify track parameters into discrete bins
- Focal loss for pT to handle class imbalance
- Tracks unbinned physics metrics (MAE, residual std)

Usage:
    cd hepattn_muon/src
    pixi run python -m hepattn.experiments.atlas_muon.run_yolo_classification fit \
        --config hepattn/experiments/atlas_muon/configs/YOLO/stage1_cls_gpu0.yaml
"""

from hepattn.utils.cli import CLI
from hepattn.experiments.atlas_muon.yolo_classification_training import ClassificationTrainingModule
from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule


def main():
    """Run YOLO classification training via Lightning CLI."""
    cli = CLI(
        ClassificationTrainingModule,
        PerTrackDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
