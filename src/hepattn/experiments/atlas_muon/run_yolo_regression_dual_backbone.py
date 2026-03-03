#!/usr/bin/env python3
"""Training script for YOLO dual-backbone regression.

Dual-backbone architecture:
- Frozen classification backbone (from Stage 1) → bin predictions
- Trainable regression backbone (same architecture, from scratch) → scalar offsets
- Each backbone processes the same hit sequence independently

The regression backbone learns its own representation optimized for
high-resolution within-bin offset prediction.

Usage:
    cd hepattn_muon/src
    pixi run python -m hepattn.experiments.atlas_muon.run_yolo_regression_dual_backbone fit \
        --config hepattn/experiments/atlas_muon/configs/YOLO/dual_backbone/dual_backbone_reg_gpu0.yaml
"""

from hepattn.utils.cli import CLI
from hepattn.experiments.atlas_muon.yolo_regression_dual_backbone import YOLORegressionDualBackbone
from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule


def main():
    """Run YOLO dual-backbone regression training via Lightning CLI."""
    cli = CLI(
        YOLORegressionDualBackbone,
        PerTrackDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
