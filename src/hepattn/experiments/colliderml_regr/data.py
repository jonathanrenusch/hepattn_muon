"""ColliderML track regression DataModule.

Reads the preprocessed memmap format produced by
:mod:`scripts.preprocess_colliderml` and provides PyTorch DataLoaders with:

- Track-level random access via CSR-indexed hit arrays
- Dynamic padding to batch-max length (with length-bucketed sampling option)
- Configurable train / val / test splits (by shard index)

The dataset stores all hits and particles per-event for future use, but the
DataLoader only loads the selected-track subsequences needed for regression.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


# ============================================================================
# Dataset
# ============================================================================


class ColliderMLTrackDataset(Dataset):
    """Memory-mapped dataset for track parameter regression.

    Each sample is a single selected track, yielding:
    - ``hit_features``: ``(L, 10)`` float32 — per-hit feature vectors
    - ``hit_s``: ``(L,)`` float32 — distance from IP (for sorting)
    - ``targets``: ``(5,)`` float32 — [d0, z0, phi, theta, qop]

    Parameters
    ----------
    preprocessed_dir : str | Path
        Root directory with ``shard_XXXX/`` subdirectories.
    shard_indices : list[int]
        Which shards to include in this dataset split.
    max_hits : int
        Maximum number of hits per track. Tracks longer than this are
        truncated (keeps the closest hits to IP by s-sorted order).
    """

    # Global filter cuts applied at dataset load time (hardcoded)
    D0_MIN = -1.0
    D0_MAX = 1.0
    Z0_MIN = -150.0
    Z0_MAX = 150.0

    def __init__(
        self,
        preprocessed_dir: str | Path,
        shard_indices: list[int],
        max_hits: int = 50,
    ):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_hits = max_hits

        # Build global index: list of (shard_idx, local_track_idx)
        # and cache memmap references per shard
        self._shard_data: dict[int, dict[str, np.ndarray]] = {}
        self._global_index: list[tuple[int, int]] = []
        self._track_lengths: list[int] = []

        for si in sorted(shard_indices):
            shard_dir = self.preprocessed_dir / f"shard_{si:04d}"
            if not shard_dir.exists():
                continue

            sel_dir = shard_dir / "selected_tracks"
            targets = np.load(sel_dir / "track_targets.npy", mmap_mode="r")
            offsets = np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r")
            hit_indices = np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r")
            hits = np.load(shard_dir / "hits.npy", mmap_mode="r")

            n_tracks = len(targets)
            self._shard_data[si] = {
                "hits": hits,
                "targets": targets,
                "offsets": offsets,
                "hit_indices": hit_indices,
            }

            for t in range(n_tracks):
                # Apply global d0/z0 range filter
                tgt = targets[t]  # [d0, z0, phi, theta, qop]
                d0_val, z0_val = float(tgt[0]), float(tgt[1])
                if not (self.D0_MIN <= d0_val <= self.D0_MAX):
                    continue
                if not (self.Z0_MIN <= z0_val <= self.Z0_MAX):
                    continue
                length = int(offsets[t + 1] - offsets[t])
                self._global_index.append((si, t))
                self._track_lengths.append(min(length, max_hits))

        self._track_lengths_arr = np.array(self._track_lengths, dtype=np.int32)

    def __len__(self) -> int:
        return len(self._global_index)

    @property
    def track_lengths(self) -> np.ndarray:
        """Per-sample track lengths for bucketed sampling."""
        return self._track_lengths_arr

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        shard_idx, local_idx = self._global_index[idx]
        data = self._shard_data[shard_idx]

        offsets = data["offsets"]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])
        hit_idx = np.array(data["hit_indices"][start:end])

        # Truncate if needed
        if len(hit_idx) > self.max_hits:
            hit_idx = hit_idx[: self.max_hits]

        # Gather hit features
        # Preprocessed format: [x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector]
        hit_feats = np.array(data["hits"][hit_idx])  # (L, 11)

        # Compute derived eta from theta_hit (col 5)
        theta_hit = hit_feats[:, 5].copy()
        eta_hit = -np.log(np.tan(theta_hit / 2.0 + 1e-12))
        eta_hit = np.clip(eta_hit, -10.0, 10.0)

        # Append eta as an extra feature -> (L, 12)
        hit_feats = np.concatenate([hit_feats, eta_hit[:, None]], axis=1).astype(np.float32)

        hit_s = hit_feats[:, 6].copy()  # s column
        targets = np.array(data["targets"][local_idx])  # (5,)

        return {
            "hit_features": hit_feats,
            "hit_s": hit_s,
            "targets": targets,
            "length": len(hit_idx),
        }


# ============================================================================
# Collate function (dynamic padding)
# ============================================================================


def collate_tracks(batch: list[dict[str, np.ndarray]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Collate variable-length tracks into padded tensors.

    Returns
    -------
    inputs : dict
        - ``hit_features``: ``(B, max_L, D)`` float32 (D=12: 11 raw + eta)
        - ``hit_s``: ``(B, max_L)`` float32
        - ``hit_valid``: ``(B, max_L)`` bool
    targets : dict
        - ``d0``, ``z0``, ``phi``, ``theta``, ``qop``: each ``(B,)`` float32
        - ``track_valid``: ``(B,)`` bool (all True for selected tracks)
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    feat_dim = batch[0]["hit_features"].shape[-1]

    hit_features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    hit_s = torch.zeros(batch_size, max_len, dtype=torch.float32)
    hit_valid = torch.zeros(batch_size, max_len, dtype=torch.bool)

    d0 = torch.empty(batch_size, dtype=torch.float32)
    z0 = torch.empty(batch_size, dtype=torch.float32)
    phi = torch.empty(batch_size, dtype=torch.float32)
    theta = torch.empty(batch_size, dtype=torch.float32)
    qop = torch.empty(batch_size, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["length"]
        hit_features[i, :L] = torch.from_numpy(item["hit_features"])
        hit_s[i, :L] = torch.from_numpy(item["hit_s"])
        hit_valid[i, :L] = True
        targets = item["targets"]
        d0[i] = float(targets[0])
        z0[i] = float(targets[1])
        phi[i] = float(targets[2])
        theta[i] = float(targets[3])
        qop[i] = float(targets[4])

    inputs = {
        "hit_features": hit_features,
        "hit_s": hit_s,
        "hit_valid": hit_valid,
    }
    target_dict = {
        "d0": d0,
        "z0": z0,
        "phi": phi,
        "theta": theta,
        "qop": qop,
        "track_valid": torch.ones(batch_size, dtype=torch.bool),
    }
    return inputs, target_dict


# ============================================================================
# Length-bucketed sampler
# ============================================================================


class LengthBucketSampler(Sampler):
    """Groups tracks by length to minimize padding waste.

    Sorts by length, chunks into buckets, shuffles bucket order +
    within-bucket order each epoch.

    Parameters
    ----------
    lengths : np.ndarray
        Per-sample track lengths.
    batch_size : int
        Batch size.
    drop_last : bool
        Whether to drop the last incomplete batch.
    shuffle : bool
        Whether to shuffle.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Pre-compute sorted indices
        self._sorted_indices = np.argsort(lengths).tolist()

    def __len__(self) -> int:
        # Return number of *individual items* yielded, not batches.
        # DataLoader wraps this sampler in a BatchSampler which re-divides
        # by batch_size, so we must report sample count here.
        n = len(self.lengths)
        if self.drop_last:
            return (n // self.batch_size) * self.batch_size
        return n

    def __iter__(self):
        indices = list(self._sorted_indices)

        # Chunk into batches
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            # Shuffle batch order
            rng = np.random.default_rng()
            perm = rng.permutation(len(batches))
            batches = [batches[p] for p in perm]
            # Shuffle within each batch
            for b in batches:
                rng.shuffle(b)

        for batch in batches:
            yield from batch


# ============================================================================
# DataModule
# ============================================================================


class ColliderMLRegrDataModule(LightningDataModule):
    """Lightning DataModule for track parameter regression.

    Parameters
    ----------
    preprocessed_dir : str
        Path to preprocessed memmap shards.
    batch_size : int
        Batch size.
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin memory for GPU transfer.
    train_frac : float
        Fraction of shards for training.
    val_frac : float
        Fraction of shards for validation.
    max_hits : int
        Maximum hits per track.
    bucketed_sampling : bool
        Use length-bucketed sampling to reduce padding.
    num_shards : int
        Limit total shards (for debugging). -1 for all.
    """

    def __init__(
        self,
        preprocessed_dir: str = "/scratch/colliderml/p0_preprocessed",
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_frac: float = 0.9,
        val_frac: float = 0.05,
        max_hits: int = 50,
        bucketed_sampling: bool = True,
        num_shards: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.preprocessed_dir = Path(preprocessed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.max_hits = max_hits
        self.bucketed_sampling = bucketed_sampling
        self.num_shards_limit = num_shards

        self._train_ds: ColliderMLTrackDataset | None = None
        self._val_ds: ColliderMLTrackDataset | None = None
        self._test_ds: ColliderMLTrackDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Discover shards and split into train/val/test."""
        # Find available shards
        shard_dirs = sorted(self.preprocessed_dir.glob("shard_*"))
        shard_indices = [int(d.name.split("_")[1]) for d in shard_dirs]

        if self.num_shards_limit > 0:
            shard_indices = shard_indices[: self.num_shards_limit]

        n = len(shard_indices)
        n_train = max(1, int(n * self.train_frac))
        n_val = max(1, int(n * self.val_frac))
        n_test = max(1, n - n_train - n_val)

        # Deterministic split by shard index order
        train_shards = shard_indices[:n_train]
        val_shards = shard_indices[n_train : n_train + n_val]
        test_shards = shard_indices[n_train + n_val :]

        if stage in (None, "fit"):
            self._train_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, train_shards, max_hits=self.max_hits,
            )
            self._val_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, val_shards, max_hits=self.max_hits,
            )
        if stage in (None, "test"):
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards, max_hits=self.max_hits,
            )
        if stage == "predict":
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards, max_hits=self.max_hits,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None

        if self.bucketed_sampling:
            sampler = LengthBucketSampler(
                self._train_ds.track_lengths,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
            return DataLoader(
                self._train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_tracks,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            )

        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_tracks,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_tracks,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_tracks,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
