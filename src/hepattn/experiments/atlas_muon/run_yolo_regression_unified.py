#!/usr/bin/env python3
"""Training script for YOLO-style unified regression training.

Stage 2 of the 3-stage training pipeline (unified version):
- Loads frozen encoder and classification heads from Stage 1
- Trains lightweight external regression heads (single scalar offset per variable)
- Offsets: eta=linear, phi=atan2-wrapped, pt=log-space
- Loss: Smooth L1

Usage:
    cd hepattn_muon/src
    pixi run python -m hepattn.experiments.atlas_muon.run_yolo_regression_unified fit \
        --config hepattn/experiments/atlas_muon/configs/YOLO/stage2/stage2_reg_unified_gpu0.yaml
"""

from hepattn.utils.cli import CLI
from hepattn.experiments.atlas_muon.yolo_regression_unified import YOLORegressionUnified
from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule


def main():
    """Run YOLO unified regression training via Lightning CLI."""
    cli = CLI(
        YOLORegressionUnified,
        PerTrackDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
