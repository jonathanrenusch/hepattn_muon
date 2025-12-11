"""
Outward graph-based particle tracking experiment.

This approach builds a directed graph where edges point outward from the 
interaction point along tracks. Track extraction uses connected components,
eliminating the need for Hungarian matching.
"""

from hepattn.experiments.outward_tracking.data import (
    OutwardTrackingDataset,
    OutwardTrackingDataModule,
    OutwardTrackingCollator,
)

__all__ = [
    "OutwardTrackingDataset",
    "OutwardTrackingDataModule", 
    "OutwardTrackingCollator",
]
