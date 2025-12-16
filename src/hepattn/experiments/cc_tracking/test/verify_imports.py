#!/usr/bin/env python
"""Verify training imports work."""

import sys
sys.path.insert(0, 'src')

from hepattn.experiments.outward_tracking.run_outward_tracking import OutwardTracker
from hepattn.experiments.outward_tracking.data import OutwardTrackingDataModule

print('Successfully imported OutwardTracker and OutwardTrackingDataModule')
print('Run training with:')
print('  pixi run python -m hepattn.experiments.outward_tracking.run_outward_tracking fit --config src/hepattn/experiments/outward_tracking/configs/outward_tracking.yaml')
