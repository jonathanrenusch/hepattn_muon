"""
Connected component based particle tracking experiment.

This approach predicts adjacency matrices where edges indicate which hits
belong to the same particle track. Track extraction uses connected components,
eliminating the need for Hungarian matching.

Supports three adjacency matrix types:
- outward: Directed edges from inner to outer hits (sorted by r)
- bidirectional: Symmetric chain edges (outward | outward.T)
- full: All pairs of hits on the same track

Each type can optionally include self-connections (diagonal).
"""

from hepattn.experiments.cc_tracking.data import (
    CCTrackingDataset,
    CCTrackingDataModule,
    CCTrackingCollator,
)

__all__ = [
    "CCTrackingDataset",
    "CCTrackingDataModule", 
    "CCTrackingCollator",
]
