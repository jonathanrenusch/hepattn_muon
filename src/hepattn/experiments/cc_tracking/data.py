"""
Data module for connected component based particle tracking.

This module creates training targets for adjacency matrix prediction:
1. Outward adjacency: hit[i] -> hit[i+1] sorted by r (directed, sparse)
2. Bidirectional adjacency: outward | outward.T (undirected chains)
3. Full adjacency: all pairs of hits on the same track (undirected, dense)

Each type supports optional self-connections (diagonal).

Track extraction uses connected components - no Hungarian matching needed.
"""

from pathlib import Path
from typing import Literal
import yaml
import numpy as np
import torch
from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import h5py
from hepattn.utils.tensor_utils import pad_to_size


# Type alias for adjacency matrix types
AdjacencyType = Literal["outward", "bidirectional", "full", "knn_outward", "station_outward"]


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class CCTrackingDataset(Dataset):
    """
    Dataset for connected component based particle tracking.
    
    Creates adjacency matrix targets for hit-to-hit correlation prediction:
    - outward: Sparse directed edges hit[i] -> hit[i+1] sorted by r
    - bidirectional: Symmetric chain edges (outward | outward.T)
    - full: Dense symmetric adjacency where M[i,j]=1 if hits i,j on same track
    - knn_outward: Each hit connects to k nearest hits with larger r (across stations/layers)
    - station_outward: Within station: chain by r; between stations: connect to innermost hit of next station
    
    The model learns to predict these matrices, and we use connected components
    for track extraction at inference time.
    
    Args:
        dirpath: Path to the data directory.
        inputs: Dictionary specifying input features per object type.
        targets: Dictionary specifying target fields per object type.
        num_events: Number of events to use (-1 for all).
        event_max_num_particles: Maximum number of particles per event (for padding).
        adjacency_type: Type of adjacency matrix to build ("outward", "bidirectional", "full", "knn_outward").
        self_connections: Whether to include self-connections in the adjacency matrix (diagonal).
        knn_k: Number of nearest neighbors for knn_outward adjacency type (default 5).
        dummy_testing: If True, filter to only valid hits (for debugging).
    """
    
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        event_max_num_particles: int = 6,
        adjacency_type: AdjacencyType = "outward",
        self_connections: bool = False,
        knn_k: int = 5,
        dummy_testing: bool = False,
    ):
        super().__init__()
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)

        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.dummy_testing = dummy_testing
        self.adjacency_type = adjacency_type
        self.self_connections = self_connections
        self.knn_k = knn_k
        
        # Validate adjacency type
        valid_types = ["outward", "bidirectional", "full", "knn_outward", "station_outward"]
        if adjacency_type not in valid_types:
            raise ValueError(f"adjacency_type must be one of {valid_types}, got {adjacency_type}")
        
        # Load metadata
        with open(self.dirpath / 'metadata.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        self.hit_features = self.metadata['hit_features']
        self.track_features = self.metadata['track_features']
        
        # Load efficient index arrays
        self.file_indices = np.load(self.dirpath / 'event_file_indices.npy')
        self.row_indices = np.load(self.dirpath / 'event_row_indices.npy')

        # Calculate number of events to use
        num_events_available = len(self.row_indices)
        
        if num_events > num_events_available:
            raise ValueError(f"Requested {num_events} events, but only {num_events_available} are available.")
        
        if num_events == -1:
            num_events = num_events_available
            
        if num_events == 0:
            raise ValueError("num_events must be greater than 0")
        
        self.num_events = num_events
        self.event_max_num_particles = event_max_num_particles
        
        self_conn_str = "with" if self_connections else "without"
        print(f"Created CCTracking dataset with {self.num_events:,} events")
        knn_str = f", k={knn_k}" if adjacency_type == "knn_outward" else ""
        print(f"  Adjacency type: {adjacency_type}{knn_str} ({self_conn_str} self-connections)")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        inputs = {}
        targets = {}
 
        # Load the event
        hits, particles, num_hits, num_tracks = self.load_event(idx)
        
        # Build the input hits
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).unsqueeze(0)

        # ===== BUILD OUTWARD EDGE TARGETS =====
        # For each track, sort hits by r and create edges pointing outward
        particle_ids = hits["spacePoint_truthLink"]
        unique_pids = np.unique(particle_ids[particle_ids >= 0])
        
        source_nodes = []
        target_nodes = []
        
        for pid in unique_pids:
            mask = particle_ids == pid
            hit_indices = np.where(mask)[0]
            
            if len(hit_indices) < 2:
                continue
            
            # Sort by r (radial distance from IP)
            r_values = hits["r"][hit_indices]
            sorted_order = np.argsort(r_values)
            sorted_indices = hit_indices[sorted_order]
            
            # Create edges: hit[i] -> hit[i+1]
            for i in range(len(sorted_indices) - 1):
                source_nodes.append(sorted_indices[i])
                target_nodes.append(sorted_indices[i + 1])
        
        # Store as sparse edge index [2, E]
        if len(source_nodes) > 0:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        targets["outward_edge_index"] = edge_index.unsqueeze(0)  # [1, 2, E]
        targets["num_outward_edges"] = torch.tensor([len(source_nodes)], dtype=torch.long)
        
        # ===== BUILD OUTWARD ADJACENCY MATRIX =====
        # Dense [N, N] matrix where M[i,j]=1 if edge i->j exists
        outward_adj = torch.zeros((1, num_hits, num_hits), dtype=torch.bool)
        if len(source_nodes) > 0:
            outward_adj[0, source_nodes, target_nodes] = True
        targets["outward_adjacency"] = outward_adj
        
        # ===== BUILD BIDIRECTIONAL ADJACENCY MATRIX =====
        # Symmetric version of outward: outward | outward.T
        bidirectional_adj = outward_adj | outward_adj.transpose(-2, -1)
        targets["bidirectional_adjacency"] = bidirectional_adj
        
        # ===== BUILD FULL ADJACENCY MATRIX =====
        # M[i,j]=1 if hits i and j are on the same track (symmetric, excludes diagonal)
        full_adj = torch.zeros((1, num_hits, num_hits), dtype=torch.bool)
        for pid in unique_pids:
            mask = particle_ids == pid
            hit_indices = np.where(mask)[0]
            # Create all pairs (excluding diagonal)
            for i in hit_indices:
                for j in hit_indices:
                    if i != j:
                        full_adj[0, i, j] = True
        targets["full_adjacency"] = full_adj
        
        # ===== BUILD KNN OUTWARD ADJACENCY MATRIX =====
        # Each hit connects to k nearest hits with strictly larger r value
        # - Cannot connect backwards in r (only forwards/outward)
        # - Cannot connect to hits in same station AND same layer
        knn_outward_adj = torch.zeros((1, num_hits, num_hits), dtype=torch.bool)
        layers = hits["spacePoint_layer"]
        stations = hits["spacePoint_stationIndex"]
        r_values = hits["r"]
        x_coords = hits["spacePoint_globEdgeLowX"]
        y_coords = hits["spacePoint_globEdgeLowY"]
        z_coords = hits["spacePoint_globEdgeLowZ"]
        
        for pid in unique_pids:
            mask = particle_ids == pid
            hit_indices = np.where(mask)[0]
            
            if len(hit_indices) < 2:
                continue
            
            # For each hit, find k nearest hits with larger r (excluding same station+layer)
            for src_idx in hit_indices:
                src_r = r_values[src_idx]
                src_station = stations[src_idx]
                src_layer = layers[src_idx]
                src_x, src_y, src_z = x_coords[src_idx], y_coords[src_idx], z_coords[src_idx]
                
                # Find candidate targets: same track, larger r, not (same station AND same layer)
                candidates = []
                for dst_idx in hit_indices:
                    if dst_idx == src_idx:
                        continue
                    dst_r = r_values[dst_idx]
                    dst_station = stations[dst_idx]
                    dst_layer = layers[dst_idx]
                    
                    # Must have strictly larger r
                    if dst_r <= src_r:
                        continue
                    
                    # Cannot be in same station AND same layer
                    if dst_station == src_station and dst_layer == src_layer:
                        continue
                    
                    # Compute distance
                    dst_x, dst_y, dst_z = x_coords[dst_idx], y_coords[dst_idx], z_coords[dst_idx]
                    dist = np.sqrt((src_x - dst_x)**2 + (src_y - dst_y)**2 + (src_z - dst_z)**2)
                    candidates.append((dist, dst_idx))
                
                # Sort by distance and take k nearest
                candidates.sort(key=lambda x: x[0])
                k = min(self.knn_k, len(candidates))
                for i in range(k):
                    dst_idx = candidates[i][1]
                    knn_outward_adj[0, src_idx, dst_idx] = True
        
        targets["knn_outward_adjacency"] = knn_outward_adj
        
        # ===== BUILD STATION OUTWARD ADJACENCY MATRIX =====
        # Within station: chain hits by r (strictly outward, never same layer)
        # Between stations: outermost hit of station connects to innermost hit of next station only
        station_outward_adj = torch.zeros((1, num_hits, num_hits), dtype=torch.bool)
        
        for pid in unique_pids:
            mask = particle_ids == pid
            hit_indices = np.where(mask)[0]
            
            if len(hit_indices) < 2:
                continue
            
            # Group hits by station
            track_stations = stations[hit_indices]
            unique_stations = np.unique(track_stations)
            
            # Sort stations by their minimum r value (innermost to outermost)
            station_min_r = {}
            for station in unique_stations:
                station_mask = track_stations == station
                station_hits = hit_indices[station_mask]
                station_min_r[station] = np.min(r_values[station_hits])
            sorted_stations = sorted(unique_stations, key=lambda s: station_min_r[s])
            
            # Process each station
            for station_idx, station in enumerate(sorted_stations):
                station_mask = track_stations == station
                station_hits = hit_indices[station_mask]
                
                # Sort hits within station by r
                station_r = r_values[station_hits]
                sorted_order = np.argsort(station_r)
                sorted_hits = station_hits[sorted_order]
                
                # Within station: connect consecutive hits (by r), but not if same layer
                for i in range(len(sorted_hits) - 1):
                    src_idx = sorted_hits[i]
                    dst_idx = sorted_hits[i + 1]
                    
                    # Skip if same layer within station
                    if layers[src_idx] == layers[dst_idx]:
                        continue
                    
                    station_outward_adj[0, src_idx, dst_idx] = True
                
                # Between stations: connect outermost hit to innermost hit of next station
                if station_idx < len(sorted_stations) - 1:
                    next_station = sorted_stations[station_idx + 1]
                    next_station_mask = track_stations == next_station
                    next_station_hits = hit_indices[next_station_mask]
                    
                    # Outermost hit of current station (largest r)
                    outermost_hit = sorted_hits[-1]
                    
                    # Innermost hit of next station (smallest r)
                    next_station_r = r_values[next_station_hits]
                    innermost_next = next_station_hits[np.argmin(next_station_r)]
                    
                    station_outward_adj[0, outermost_hit, innermost_next] = True
        
        targets["station_outward_adjacency"] = station_outward_adj
        
        # ===== SELECT TARGET ADJACENCY BASED ON TYPE =====
        if self.adjacency_type == "outward":
            target_adj = outward_adj.clone()
        elif self.adjacency_type == "bidirectional":
            target_adj = bidirectional_adj.clone()
        elif self.adjacency_type == "full":
            target_adj = full_adj.clone()
        elif self.adjacency_type == "knn_outward":
            target_adj = knn_outward_adj.clone()
        elif self.adjacency_type == "station_outward":
            target_adj = station_outward_adj.clone()
        else:
            raise ValueError(f"Unknown adjacency type: {self.adjacency_type}")
        
        # ===== ADD SELF-CONNECTIONS IF REQUESTED =====
        if self.self_connections:
            # Add diagonal for valid hits that are part of a track
            valid_on_track = hits["on_valid_particle"]
            for i in range(num_hits):
                if valid_on_track[i]:
                    target_adj[0, i, i] = True
        
        targets["cc_adjacency"] = target_adj
        
        # ===== BUILD ANCHOR MASK (innermost hits) =====
        # Used for contrastive learning - anchors are innermost hits
        min_r_indices = []
        valid_particle_ids = []
        
        for pid in unique_pids:
            mask = particle_ids == pid
            if not np.any(mask):
                continue
            hit_indices = np.where(mask)[0]
            r_vals = hits["r"][mask]
            min_local_idx = np.argmin(r_vals)
            min_r_indices.append(hit_indices[min_local_idx])
            valid_particle_ids.append(pid)
        
        targets["anchor_mask"] = torch.zeros((1, num_hits), dtype=torch.bool)
        if len(min_r_indices) > 0:
            min_r_indices_tensor = torch.tensor(min_r_indices, dtype=torch.long)
            targets["anchor_mask"][0, min_r_indices_tensor] = True
        
        # ===== PARTICLE HIT CORRELATION (for contrastive loss) =====
        # For each anchor, mark which hits belong to its track
        targets["particle_hit_corr"] = torch.zeros((1, num_hits, num_hits), dtype=torch.bool)
        for i, min_r_idx in enumerate(min_r_indices):
            pid = valid_particle_ids[i]
            same_particle_mask = particle_ids == pid
            targets["particle_hit_corr"][0, min_r_idx, :] = torch.from_numpy(same_particle_mask)
        
        # ===== HIT VALIDITY =====
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)
        
        # ===== METADATA =====
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)
        targets["num_true_tracks"] = torch.tensor([num_tracks], dtype=torch.long)
        
        # ===== PARTICLE REGRESSION TARGETS =====
        if "particle" in self.targets:
            particle_ids_tensor = torch.from_numpy(particles["particle_id"])
            for field in self.targets["particle"]:
                x = torch.full((self.event_max_num_particles,), torch.nan)
                if field in particles:
                    x[:num_tracks] = torch.from_numpy(particles[field][:self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        """Load a single event from compound HDF5 files."""
        file_idx = self.file_indices[idx]
        row_idx = self.row_indices[idx]
        
        chunk = self.metadata['event_mapping']['chunk_summary'][file_idx]
        h5_file_path = self.dirpath / chunk['h5_file']
        
        try:
            with h5py.File(h5_file_path, 'r') as f:
                num_hits = f['num_hits'][row_idx]
                num_tracks = f['num_tracks'][row_idx]
                hits_array = f['hits'][row_idx, :num_hits]
                tracks_array = f['tracks'][row_idx, :num_tracks]
                
        except Exception as e:
            raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}")

        # Convert hits array to dictionary
        hits_dict = {}
        for i, feature_name in enumerate(self.hit_features):
            hits_dict[feature_name] = hits_array[:, i]
        
        # Scaling (matching selfatt_muon/data.py)
        hits = {
            'spacePoint_globEdgeHighX': hits_dict['spacePoint_globEdgeHighX'] * 0.001,
            'spacePoint_globEdgeHighY': hits_dict['spacePoint_globEdgeHighY'] * 0.001,
            'spacePoint_globEdgeHighZ': hits_dict['spacePoint_globEdgeHighZ'] * 0.001,
            'spacePoint_globEdgeLowX': hits_dict['spacePoint_globEdgeLowX'] * 0.001,
            'spacePoint_globEdgeLowY': hits_dict['spacePoint_globEdgeLowY'] * 0.001,
            'spacePoint_globEdgeLowZ': hits_dict['spacePoint_globEdgeLowZ'] * 0.001,
            'spacePoint_time': hits_dict['spacePoint_time'] * 0.00001,
            'spacePoint_driftR': hits_dict['spacePoint_driftR'],
            'spacePoint_covXX': hits_dict['spacePoint_covXX'] * 0.000001,
            'spacePoint_covXY': hits_dict['spacePoint_covXY'] * 0.000001,
            'spacePoint_covYX': hits_dict['spacePoint_covYX'] * 0.000001,
            'spacePoint_covYY': hits_dict['spacePoint_covYY'] * 0.000001,
            'spacePoint_channel': hits_dict['spacePoint_channel'] * 0.001,
            'spacePoint_layer': hits_dict['spacePoint_layer'],
            'spacePoint_stationPhi': hits_dict['spacePoint_stationPhi'],
            'spacePoint_stationEta': hits_dict['spacePoint_stationEta'],
            'spacePoint_technology': hits_dict['spacePoint_technology'],
            'spacePoint_stationIndex': hits_dict['spacePoint_stationIndex'] * 0.1,
            'spacePoint_truthLink': hits_dict['spacePoint_truthLink'],
        }
        
        # Derived features
        hits["r"] = np.sqrt(hits["spacePoint_globEdgeLowX"]**2 + hits["spacePoint_globEdgeLowY"]**2)
        hits["s"] = np.sqrt(hits["spacePoint_globEdgeLowX"]**2 + hits["spacePoint_globEdgeLowY"]**2 + hits["spacePoint_globEdgeLowZ"]**2)
        hits["theta"] = np.arccos(np.clip(hits["spacePoint_globEdgeLowZ"] / hits["s"], -1, 1))
        hits["phi"] = np.arctan2(hits["spacePoint_globEdgeLowY"], hits["spacePoint_globEdgeLowX"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2.0))
        hits["on_valid_particle"] = hits["spacePoint_truthLink"] >= 0

        # For debugging
        if self.dummy_testing:
            for k in hits:
                hits[k] = hits[k][hits["on_valid_particle"]]
            num_hits = np.sum(hits["on_valid_particle"])

        # Tracks
        tracks_dict = {}
        for i, feature_name in enumerate(self.track_features):
            tracks_dict[feature_name] = tracks_array[:, i]

        particles = {
            'particle_id': np.unique(hits["spacePoint_truthLink"][hits["on_valid_particle"]]),
            'truthMuon_pt': tracks_dict['truthMuon_pt'] / 200,
            'truthMuon_eta': tracks_dict['truthMuon_eta'],
            'truthMuon_phi': tracks_dict['truthMuon_phi'],
            'truthMuon_q': tracks_dict['truthMuon_q'],
            "truthMuon_qpt": tracks_dict['truthMuon_q'] / tracks_dict['truthMuon_pt'],
        }
        
        return hits, particles, num_hits, num_tracks


class CCTrackingCollator:
    """Collator for batching CC tracking data with variable-size events."""
    
    def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_targets
        self.max_num_obj = max_num_obj

    def __call__(self, batch):
        inputs, targets = zip(*batch, strict=False)
        
        # Get max hit size across batch
        hit_max_sizes = {}
        for input_name in self.dataset_inputs:
            hit_max_sizes[input_name] = max(event[f"{input_name}_valid"].shape[-1] for event in inputs)
        
        batched_inputs = {}
        batched_targets = {}
        
        # Batch inputs
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)
            batched_targets[k] = batched_inputs[k]
            
            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)
        
        # Batch adjacency matrices
        hit_size = hit_max_sizes["hit"]
        
        # CC adjacency [B, N, N] - the main target based on adjacency_type
        batched_targets["cc_adjacency"] = pad_and_concat(
            [t["cc_adjacency"] for t in targets], 
            (hit_size, hit_size), False
        )
        
        # Also batch the individual adjacency types for metrics/evaluation
        batched_targets["outward_adjacency"] = pad_and_concat(
            [t["outward_adjacency"] for t in targets], 
            (hit_size, hit_size), False
        )
        
        batched_targets["bidirectional_adjacency"] = pad_and_concat(
            [t["bidirectional_adjacency"] for t in targets], 
            (hit_size, hit_size), False
        )
        
        # Full adjacency [B, N, N]  
        batched_targets["full_adjacency"] = pad_and_concat(
            [t["full_adjacency"] for t in targets],
            (hit_size, hit_size), False
        )
        
        # KNN outward adjacency [B, N, N]
        batched_targets["knn_outward_adjacency"] = pad_and_concat(
            [t["knn_outward_adjacency"] for t in targets],
            (hit_size, hit_size), False
        )
        
        # Station outward adjacency [B, N, N]
        batched_targets["station_outward_adjacency"] = pad_and_concat(
            [t["station_outward_adjacency"] for t in targets],
            (hit_size, hit_size), False
        )
        
        # Particle hit correlation [B, N, N]
        batched_targets["hit_particle_hit_corr"] = pad_and_concat(
            [t["particle_hit_corr"] for t in targets],
            (hit_size, hit_size), False
        )
        
        # Anchor mask [B, N]
        batched_targets["particle_valid"] = pad_and_concat(
            [t["anchor_mask"] for t in targets],
            (hit_size,), False
        )
        
        # Hit on valid particle [B, N]
        batched_targets["hit_on_valid_particle"] = pad_and_concat(
            [t["hit_on_valid_particle"] for t in targets],
            (hit_size,), False
        )
        
        # Metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)
        batched_targets["num_true_tracks"] = torch.cat([t["num_true_tracks"] for t in targets], dim=-1)
        
        # Store edge indices as list (variable length per event)
        batched_targets["outward_edge_indices"] = [t["outward_edge_index"].squeeze(0) for t in targets]
        batched_targets["num_outward_edges"] = torch.cat([t["num_outward_edges"] for t in targets], dim=-1)
        
        return batched_inputs, batched_targets


class CCTrackingDataModule(LightningDataModule):
    """Lightning DataModule for connected component based tracking."""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        batch_size: int = 100,
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
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            self.train_dataset = CCTrackingDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                **self.kwargs,
            )

        if stage == "fit" or stage == "validate":
            self.val_dataset = CCTrackingDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                **self.kwargs,
            )
            
        if stage == "fit" and self.trainer is not None and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dataset):,} events")
            print(f"Created validation dataset with {len(self.val_dataset):,} events")
            
        if stage == "test":
            assert self.test_dir is not None, "No test file specified"
            self.test_dataset = CCTrackingDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: CCTrackingDataset, shuffle: bool, prefetch_factor: int = 8):
        actual_prefetch_factor = None if self.num_workers == 0 else prefetch_factor
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=CCTrackingCollator(
                dataset.inputs, 
                dataset.targets, 
                dataset.event_max_num_particles
            ),
            sampler=None,
            num_workers=self.num_workers,
            prefetch_factor=actual_prefetch_factor,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self, shuffle=False):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=shuffle)
