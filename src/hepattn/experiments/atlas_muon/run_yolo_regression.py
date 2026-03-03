#!/usr/bin/env python3
"""Training script for YOLO-style regression training.

Stage 2 of the 3-stage training pipeline:
- Loads frozen encoder and classification heads from Stage 1
- Trains regression heads to predict per-bin offsets
- Multi-bin loss with softmax weighting from classifier
- Proper phi wrapping throughout

Usage:
    cd hepattn_muon/src
    pixi run python -m hepattn.experiments.atlas_muon.run_yolo_regression fit \
        --config hepattn/experiments/atlas_muon/configs/YOLO/stage2/stage2_reg_gpu0.yaml
"""

from hepattn.utils.cli import CLI
from hepattn.experiments.atlas_muon.yolo_regression_training import YOLORegressionTraining
from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule


def main():
    """Run YOLO regression training via Lightning CLI."""
    cli = CLI(
        YOLORegressionTraining,
        PerTrackDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
