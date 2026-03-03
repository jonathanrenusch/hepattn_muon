"""Per-track data loading for Mamba-based track parameter regression.

This module provides a wrapper around AtlasMuonDataset that extracts individual
tracks with their associated hits, sorts by radial distance, and batches them
for the Mamba regression model.

The key difference from the standard data loading:
- Standard: Batches events, each with multiple tracks and all hits
- Per-track: Batches individual tracks, each with only its assigned hits
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hepattn.experiments.atlas_muon.data import AtlasMuonDataset


class PerTrackAtlasMuonDataset(Dataset):
    """Dataset that yields individual tracks with their associated hits.
    
    Wraps AtlasMuonDataset and extracts individual tracks from events,
    using ground truth hit-to-track assignments.
    
    Parameters
    ----------
    base_dataset : AtlasMuonDataset
        The underlying event-based dataset.
    min_hits_per_track : int
        Minimum number of hits required for a track to be included.
    max_hits_per_track : int
        Maximum number of hits per track (for padding/truncation).
    hit_fields : list[str]
        List of hit feature fields to include.
    sort_by : str
        Field to sort hits by. Options: 'r' (radial distance), 's' (arc length).
        Default is 's' which makes more physical sense for forward detector regions.
    """
    
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        min_hits_per_track: int = 3,
        max_hits_per_track: int = 200,
        hit_fields: list[str] | None = None,
        event_max_num_particles: int = 10,
        sort_by: str = 's',
    ):
        super().__init__()
        
        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.min_hits_per_track = min_hits_per_track
        self.max_hits_per_track = max_hits_per_track
        self.event_max_num_particles = event_max_num_particles
        self.sort_by = sort_by
        
        # Validate sort_by
        if sort_by not in ['r', 's']:
            raise ValueError(f"sort_by must be 'r' or 's', got '{sort_by}'")
        
        # Default hit fields if not specified
        if hit_fields is None:
            hit_fields = [
                'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
                'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
                'spacePoint_driftR', 'spacePoint_channel', 'spacePoint_layer',
                'spacePoint_stationPhi', 'spacePoint_stationEta', 'spacePoint_stationIndex',
                'spacePoint_technology', 'r', 's', 'theta', 'phi', 'eta'
            ]
        self.hit_fields = hit_fields
        
        # Create the base dataset for loading events
        self.base_dataset = AtlasMuonDataset(
            dirpath=dirpath,
            inputs=inputs,
            targets=targets,
            num_events=num_events,
            event_max_num_particles=event_max_num_particles,
        )
        
        # Build or load index of all tracks across all events
        self._load_or_build_track_index()
        
    def _get_cache_filename(self) -> Path:
        """Generate cache filename based on dataset parameters."""
        # Include relevant parameters in filename to avoid stale cache
        return self.dirpath / f"track_index_minHits{self.min_hits_per_track}_nEvents{len(self.base_dataset)}.npy"
    
    def _load_or_build_track_index(self):
        """Load track index from cache if available, otherwise build and save it."""
        cache_file = self._get_cache_filename()
        
        if cache_file.exists():
            print(f"Loading cached track index from {cache_file}")
            track_index_array = np.load(cache_file)
            # Convert numpy array back to list of tuples
            self.track_index = [tuple(row) for row in track_index_array]
            print(f"Loaded {len(self.track_index):,} tracks with >= {self.min_hits_per_track} hits")
        else:
            print(f"No cached track index found, building...")
            self._build_track_index()
            # Save to cache (track_index is list of 4-tuples: event_idx, particle_idx, pid, num_tracks)
            track_index_array = np.array(self.track_index, dtype=np.int64)
            np.save(cache_file, track_index_array)
            print(f"Saved track index to {cache_file}")
        
    def _build_track_index(self):
        """Build an index mapping track_idx -> (event_idx, particle_idx, particle_id, num_tracks).
        
        Only counts hits within the valid range [0:num_hits] to avoid counting padding.
        Stores num_tracks to validate particle_idx is still valid when retrieving.
        """
        self.track_index = []
        
        print("Building per-track index...")
        for event_idx in tqdm(range(len(self.base_dataset)), desc="Indexing tracks", dynamic_ncols=True):
            # Load event to get track info
            hits, particles, num_hits, num_tracks = self.base_dataset.load_event(event_idx)
            
            # Only consider valid hits (not padding) - use num_hits to slice
            truth_links = hits['spacePoint_truthLink'][:num_hits]  # Only valid hits
            particle_ids = particles['particle_id'][:num_tracks]  # Only valid particles
            
            # Validate we have valid data
            if num_tracks == 0:
                continue
            
            for particle_idx, pid in enumerate(particle_ids):
                # Count hits belonging to this particle (within valid range only)
                num_track_hits = np.sum(truth_links == pid)
                if num_track_hits >= self.min_hits_per_track:
                    # Store event_idx, particle_idx, particle_id, and num_tracks for validation
                    self.track_index.append((event_idx, particle_idx, pid, num_tracks))
        
        print(f"Found {len(self.track_index):,} tracks with >= {self.min_hits_per_track} hits")
    
    def __len__(self):
        return len(self.track_index)
    
    def __getitem__(self, idx):
        """Get a single track with its hits.
        
        Returns
        -------
        dict with keys:
            - hit_features: (num_hits, num_features) tensor of hit features
            - hit_r: (num_hits,) tensor of radial distances for sorting
            - num_hits: int, actual number of hits
            - eta: float, true eta
            - phi: float, true phi  
            - pt: float, true pT (normalized)
            - charge: float, true charge (-1 or 1)
        """
        # Unpack index - handle both old (3-tuple) and new (4-tuple) formats for backward compatibility
        track_info = self.track_index[idx]
        if len(track_info) == 4:
            event_idx, particle_idx, particle_id, expected_num_tracks = track_info
        else:
            # Old format - no validation possible
            event_idx, particle_idx, particle_id = track_info
            expected_num_tracks = None
        
        # Load the event
        hits, particles, num_hits, num_tracks = self.base_dataset.load_event(event_idx)
        
        # Validate particle_idx is within bounds
        if particle_idx >= num_tracks:
            raise ValueError(
                f"Track index corrupt: particle_idx={particle_idx} >= num_tracks={num_tracks} "
                f"for event {event_idx}. Index may be stale or built with different data."
            )
        
        # Validate num_tracks matches if we have that info
        if expected_num_tracks is not None and num_tracks != expected_num_tracks:
            raise ValueError(
                f"Track index corrupt: num_tracks changed from {expected_num_tracks} to {num_tracks} "
                f"for event {event_idx}. Rebuild the track index."
            )
        
        # Get hits belonging to this track - ONLY consider valid hits (not padding)
        truth_links = hits['spacePoint_truthLink'][:num_hits]  # Only valid hits
        hit_mask = truth_links == particle_id
        
        # Extract hit features for this track
        # Apply mask to get indices of hits belonging to this track
        hit_indices = np.where(hit_mask)[0]
        
        # Build track_hits dictionary with only valid, non-padded hit data
        track_hits = {}
        for field in self.hit_fields:
            if field in hits:
                # Only take valid hits - apply mask within valid range
                track_hits[field] = hits[field][:num_hits][hit_mask]
        
        # Get sorting field
        if self.sort_by not in track_hits:
            raise ValueError(f"Missing '{self.sort_by}' field in hit data for track {idx}")
        
        sort_values = track_hits[self.sort_by]
        num_track_hits = len(sort_values)
        
        # Sanity check: we should have found hits for this track
        if num_track_hits == 0:
            raise ValueError(
                f"No hits found for track {idx} (particle_id={particle_id}, event={event_idx}). "
                f"Index may be corrupt. Rebuild track index."
            )
        
        # Stack hit features into tensor
        feature_list = []
        for field in self.hit_fields:
            if field in track_hits:
                feature_list.append(track_hits[field])
        
        hit_features = np.stack(feature_list, axis=-1)  # (num_hits, num_features)
        
        # Get track parameters
        eta = particles['truthMuon_eta'][particle_idx]
        phi = particles['truthMuon_phi'][particle_idx]
        pt = particles['truthMuon_pt'][particle_idx]  # Raw pt in GeV (not normalized)
        charge = particles['truthMuon_q'][particle_idx]
        
        return {
            'hit_features': torch.from_numpy(hit_features).float(),
            'sort_values': torch.from_numpy(sort_values).float(),
            'num_hits': num_track_hits,
            'eta': torch.tensor(eta, dtype=torch.float32),
            'phi': torch.tensor(phi, dtype=torch.float32),
            'pt': torch.tensor(pt, dtype=torch.float32),
            'charge': torch.tensor(charge, dtype=torch.float32),
            'event_idx': event_idx,
            'particle_idx': particle_idx,
            'track_idx': idx,  # Global track index for sample_id in PredictionWriter
        }


class PerTrackCollator:
    """Collator for per-track batching.
    
    Handles:
    - Sorting hits by configured field (ascending, inner to outer)
    - Prepending CLS token placeholder
    - Padding sequences to max length
    - Converting charge labels from (-1, 1) to (0, 1) for BCE
    
    Parameters
    ----------
    max_hits : int
        Maximum number of hits per track (excluding CLS token).
    num_features : int
        Number of features per hit.
    """
    
    def __init__(self, max_hits: int = 200, num_features: int = 18):
        self.max_hits = max_hits
        self.num_features = num_features
    
    def __call__(self, batch: list[dict]) -> tuple[dict, dict]:
        """Collate a batch of tracks.
        
        Returns
        -------
        inputs : dict
            - hit_features: (B, max_hits+1, num_features) - +1 for CLS token
            - hit_mask: (B, max_hits+1) - True for valid positions
            - sequence_lengths: (B,) - actual sequence lengths including CLS
        targets : dict
            - eta: (B,)
            - phi: (B,)
            - pt: (B,)
            - charge: (B,) - converted to 0/1 for BCE
            - charge_original: (B,) - original -1/1 values
        """
        batch_size = len(batch)
        
        # Find max hits in this batch (capped at self.max_hits)
        max_hits_in_batch = min(
            max(item['num_hits'] for item in batch),
            self.max_hits
        )
        
        # Total sequence length includes CLS token at position 0
        seq_len = max_hits_in_batch + 1
        
        # Initialize tensors
        hit_features = torch.zeros(batch_size, seq_len, self.num_features)
        hit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Target tensors
        eta = torch.zeros(batch_size)
        phi = torch.zeros(batch_size)
        pt = torch.zeros(batch_size)
        charge = torch.zeros(batch_size)
        charge_original = torch.zeros(batch_size)
        
        for i, item in enumerate(batch):
            # Get hits and sort by configured field (ascending)
            features = item['hit_features']  # (num_hits, num_features)
            sort_values = item['sort_values']
            num_hits = min(item['num_hits'], self.max_hits)
            
            # Sort by sort field (inner to outer)
            sort_indices = torch.argsort(sort_values)[:num_hits]
            sorted_features = features[sort_indices]
            
            # Position 0 is CLS token (leave as zeros, will be replaced by learnable embedding)
            # Positions 1 to num_hits are the sorted hits
            hit_features[i, 1:num_hits+1] = sorted_features
            hit_mask[i, :num_hits+1] = True  # CLS + hits are valid
            sequence_lengths[i] = num_hits + 1
            
            # Targets
            eta[i] = item['eta']
            phi[i] = item['phi']
            pt[i] = item['pt']
            charge_original[i] = item['charge']
            # Convert charge: -1 -> 0, +1 -> 1 for BCE
            charge[i] = (item['charge'] + 1) / 2
        
        # Build sample_id tensor - each track needs a unique ID across the dataset
        # Use the track's global index as sample_id (passed from __getitem__)
        sample_ids = torch.tensor([item.get('track_idx', i) for i, item in enumerate(batch)], dtype=torch.int64)
        
        # Build num_hits tensor for filtering during evaluation
        num_hits_tensor = torch.tensor([min(item['num_hits'], self.max_hits) for item in batch], dtype=torch.int64)
        
        inputs = {
            'hit_features': hit_features,
            'hit_mask': hit_mask,
            'sequence_lengths': sequence_lengths,
        }
        
        targets = {
            'eta': eta,
            'phi': phi,
            'pt': pt,
            'charge': charge,
            'charge_original': charge_original,
            'sample_id': sample_ids,  # Required by PredictionWriter
            'num_hits': num_hits_tensor,  # Required for evaluation filtering
        }
        
        return inputs, targets


class PerTrackDataModule(LightningDataModule):
    """Lightning DataModule for per-track data loading.
    
    Parameters
    ----------
    train_dir : str
        Path to training data directory.
    val_dir : str
        Path to validation data directory.
    test_dir : str, optional
        Path to test data directory.
    batch_size : int
        Number of tracks per batch.
    num_workers : int
        Number of data loading workers.
    num_train : int
        Number of training events (-1 for all).
    num_val : int
        Number of validation events (-1 for all).
    num_test : int
        Number of test events (-1 for all).
    min_hits_per_track : int
        Minimum hits required per track.
    max_hits_per_track : int
        Maximum hits per track.
    hit_fields : list[str], optional
        List of hit feature fields.
    sort_by : str
        Field to sort hits by. Options: 'r' (radial distance), 's' (arc length).
        Default is 's' which makes more physical sense for forward detector regions.
    """
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        batch_size: int = 256,
        pin_memory: bool = True,
        min_hits_per_track: int = 3,
        max_hits_per_track: int = 200,
        hit_fields: list[str] | None = None,
        event_max_num_particles: int = 10,
        sort_by: str = 's',
        **kwargs,
    ):
        super().__init__()
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.min_hits_per_track = min_hits_per_track
        self.max_hits_per_track = max_hits_per_track
        self.hit_fields = hit_fields
        self.event_max_num_particles = event_max_num_particles
        self.sort_by = sort_by
        self.kwargs = kwargs
        
        # These will be set from config
        self.inputs = kwargs.get('inputs', {'hit': []})
        self.targets = kwargs.get('targets', {'particle': []})
    
    def setup(self, stage: str):
        """Setup datasets for each stage."""
        
        dataset_kwargs = {
            'inputs': self.inputs,
            'targets': self.targets,
            'min_hits_per_track': self.min_hits_per_track,
            'max_hits_per_track': self.max_hits_per_track,
            'hit_fields': self.hit_fields,
            'event_max_num_particles': self.event_max_num_particles,
        }
        
        if stage == "fit" or stage == "test":
            self.train_dataset = PerTrackAtlasMuonDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                **dataset_kwargs,
            )
        
        if stage == "fit" or stage == "validate":
            self.val_dataset = PerTrackAtlasMuonDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                **dataset_kwargs,
            )
        
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Training tracks: {len(self.train_dataset):,}")
            print(f"Validation tracks: {len(self.val_dataset):,}")
        
        if stage == "test":
            assert self.test_dir is not None, "No test directory specified"
            self.test_dataset = PerTrackAtlasMuonDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                **dataset_kwargs,
            )
            print(f"Test tracks: {len(self.test_dataset):,}")
    
    def _get_collator(self):
        """Get the collator with correct number of features."""
        num_features = len(self.hit_fields) if self.hit_fields else 18
        return PerTrackCollator(
            max_hits=self.max_hits_per_track,
            num_features=num_features,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._get_collator(),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._get_collator(),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._get_collator(),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
