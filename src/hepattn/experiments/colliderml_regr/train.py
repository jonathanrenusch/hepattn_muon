#!/usr/bin/env python3
"""Train track parameter regression model.

Usage::

    python train.py fit --config config/study1_baseline_l1.yaml

    # Override data dir for testing:
    python train.py fit --config config/study1_baseline_l1.yaml \\
        --data.preprocessed_dir /scratch/colliderml/p0_preprocessed_test \\
        --data.num_shards 2 --trainer.max_epochs 2
"""

import os

# Redirect Triton cache to /tmp to avoid AFS quota issues.
# Must be set before any torch/triton import.
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

from lightning.pytorch.cli import LightningCLI

from hepattn.experiments.colliderml_regr.data import ColliderMLRegrDataModule
from hepattn.experiments.colliderml_regr.model import TrackRegressionWrapper


def main():
    LightningCLI(
        model_class=TrackRegressionWrapper,
        datamodule_class=ColliderMLRegrDataModule,
        seed_everything_default=42,
    )


if __name__ == "__main__":
    main()
