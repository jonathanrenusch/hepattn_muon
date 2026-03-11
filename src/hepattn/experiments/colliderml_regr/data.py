"""ColliderML track regression DataModule.

Reads the preprocessed memmap format produced by
:mod:`scripts.preprocess_colliderml` and provides PyTorch DataLoaders with:

- Track-level random access via CSR-indexed hit arrays
- Dynamic padding to batch-max length
- Deterministic train / val / test splits from a ``split.json`` file

All track selection (min/max hits, kinematics, perigee ranges) is applied
at preprocessing time.  The dataset trusts that every track stored in the
preprocessed shards passes selection and loads them unconditionally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Dataset
# ============================================================================


class ColliderMLTrackDataset(Dataset):
    """Memory-mapped dataset for track parameter regression.

    Each sample is a single selected track, yielding:
    - ``hit_features``: ``(L, 12)`` float32 — per-hit feature vectors
    - ``hit_s``: ``(L,)`` float32 — distance from IP (for sorting)
    - ``targets``: ``(5,)`` float32 — [d0, z0, phi, theta, qop]

    All track selection (min/max hits, kinematics, d0/z0 range) is applied
    at preprocessing time.  This dataset loads every track unconditionally.

    Parameters
    ----------
    preprocessed_dir : str | Path
        Root directory with ``shard_XXXX/`` subdirectories.
    shard_indices : list[int]
        Which shards to include in this dataset split.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        shard_indices: list[int],
    ):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)

        # Build global index: list of (shard_idx, local_track_idx)
        # and cache memmap references per shard
        self._shard_data: dict[int, dict[str, np.ndarray]] = {}
        self._global_index: list[tuple[int, int]] = []

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
                self._global_index.append((si, t))

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        shard_idx, local_idx = self._global_index[idx]
        data = self._shard_data[shard_idx]

        offsets = data["offsets"]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])
        hit_idx = np.array(data["hit_indices"][start:end])

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

    for i, item in enumerate(batch):
        L = item["length"]
        hit_features[i, :L] = torch.from_numpy(item["hit_features"])
        hit_s[i, :L] = torch.from_numpy(item["hit_s"])
        hit_valid[i, :L] = True

    # Vectorise target extraction
    all_targets = torch.as_tensor(
        np.stack([item["targets"] for item in batch]),
        dtype=torch.float32,
    )  # (B, 5)

    inputs = {
        "hit_features": hit_features,
        "hit_s": hit_s,
        "hit_valid": hit_valid,
    }
    target_dict = {
        "d0": all_targets[:, 0],
        "z0": all_targets[:, 1],
        "phi": all_targets[:, 2],
        "theta": all_targets[:, 3],
        "qop": all_targets[:, 4],
        "track_valid": torch.ones(batch_size, dtype=torch.bool),
    }
    return inputs, target_dict


# ============================================================================
# DataModule
# ============================================================================


class ColliderMLRegrDataModule(LightningDataModule):
    """Lightning DataModule for track parameter regression.

    Shard assignments are read from a ``split.json`` file in the preprocessed
    directory (created once by ``scripts/create_split.py``).  This guarantees
    that validation and test data always come from the same shards, regardless
    of how many training shards are actually used.

    All track selection is applied at preprocessing time.  The DataModule
    uses simple random sampling with dynamic padding.

    Parameters
    ----------
    preprocessed_dir : str
        Path to preprocessed memmap shards.
    batch_size : int
        Batch size (number of tracks per batch).
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin memory for GPU transfer.
    num_train_shards : int
        Limit the number of *training* shards loaded (for debugging).
        ``-1`` means use all training shards from the split file.
        Validation and test shards are always loaded in full.
    """

    def __init__(
        self,
        preprocessed_dir: str = "/scratch/colliderml/p0_preprocessed",
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        num_train_shards: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.preprocessed_dir = Path(preprocessed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_train_shards_limit = num_train_shards

        self._train_ds: ColliderMLTrackDataset | None = None
        self._val_ds: ColliderMLTrackDataset | None = None
        self._test_ds: ColliderMLTrackDataset | None = None

    def _load_split(self) -> dict[str, list[int]]:
        """Load shard split from ``split.json`` in the preprocessed directory."""
        split_path = self.preprocessed_dir / "split.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found at {split_path}. "
                "Create it with: python scripts/create_split.py "
                f"--preprocessed-dir {self.preprocessed_dir}"
            )
        with open(split_path) as f:
            data = json.load(f)
        for key in ("train", "val", "test"):
            if key not in data:
                raise ValueError(f"split.json missing required key '{key}'")
        return {k: data[k] for k in ("train", "val", "test")}

    def setup(self, stage: str | None = None) -> None:
        """Load split file and create datasets for the requested stage."""
        split = self._load_split()

        train_shards = split["train"]
        val_shards = split["val"]
        test_shards = split["test"]

        # Optionally limit training shards (for debugging)
        if self.num_train_shards_limit > 0:
            train_shards = train_shards[: self.num_train_shards_limit]

        if stage in (None, "fit"):
            self._train_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, train_shards,
            )
            self._val_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, val_shards,
            )
        if stage in (None, "test"):
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards,
            )
        if stage == "predict":
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
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
