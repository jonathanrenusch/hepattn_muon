from pathlib import Path
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
import pyarrow.parquet as pq
import h5py
from hepattn.utils.tensor_utils import pad_to_size


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class SmartBatchSampler(Sampler):
    """
    A sampler that groups events with similar numbers of hits together to minimize padding.
    This improves efficiency by reducing the amount of padding needed within each batch.
    Includes memory safety by limiting maximum batch memory usage.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, 
                 max_hits_per_batch: int = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get hit counts for all events to enable smart batching
        self.hit_counts = dataset.num_hits_per_event[:len(dataset)]
        self.indices = np.arange(len(dataset))
        
        # Memory safety: limit maximum hits per batch to prevent OOM
        if max_hits_per_batch is None:
            # Conservative default: assume we can handle 50k hits per batch
            self.max_hits_per_batch = 50000
        else:
            self.max_hits_per_batch = max_hits_per_batch
        
    def __iter__(self):
        # Sort indices by hit count for better batching
        if self.shuffle:
            # Add some randomness while still keeping similar-sized events together
            # Sort into bins of similar hit counts, then shuffle within bins
            bin_size = self.batch_size * 4  # Group into larger bins first
            sorted_indices = self.indices[np.argsort(self.hit_counts)]
            
            batched_indices = []
            for i in range(0, len(sorted_indices), bin_size):
                bin_indices = sorted_indices[i:i + bin_size]
                np.random.shuffle(bin_indices)  # Shuffle within bin
                batched_indices.extend(bin_indices)
            indices = batched_indices
        else:
            # For validation, just sort by hit count for maximum efficiency
            indices = self.indices[np.argsort(self.hit_counts)]
        
        # Create memory-safe batches
        current_batch = []
        for idx in indices:
            current_batch.append(idx)
            batch_hit_counts = [self.hit_counts[i] for i in current_batch]
            max_hits_in_batch = max(batch_hit_counts)
            total_padded_hits = max_hits_in_batch * len(current_batch)
            if total_padded_hits > self.max_hits_per_batch:
                # yielding batch
                yield current_batch[:-1]
                current_batch = [current_batch[-1]]
        
        # Handle remaining events
        if current_batch and (not self.drop_last or len(current_batch) == self.batch_size):
            yield current_batch
    
    def __len__(self):
        # Conservative estimate: assume normal batching most of the time
        # This gives a reasonable approximation for progress bars and schedulers
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# class AtlasMuonCollator:
#     def __init__(self, 
#                  dataset_inputs, 
#                  dataset_targets, 
#                  max_num_particles, 
#                  verbose: bool = False):
#         self.dataset_inputs = dataset_inputs
#         self.dataset_targets = dataset_targets
#         self.max_num_particles = max_num_particles
#         self.verbose = verbose
#         self.batch_count = 0
#         self.total_padding_ratio = 0.0

#     def __call__(self, batch):
#         inputs, targets = zip(*batch, strict=False)
#         self.batch_count += 1

#         # Find the maximum number of hits across all events in the batch
#         num_hits = [input["hit_x"].shape[-1] for input in inputs]
#         max_num_hits = max(num_hits)
#         batched_inputs = {}
#         batched_targets = {}
        
#         # Batch the input features (hits)
#         for input_name, fields in self.dataset_inputs.items():
#             # Batch the validity masks
#             k = f"{input_name}_valid"
#             batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (max_num_hits,), False)

#             # Some tasks might require to know hit padding info for loss masking
#             batched_targets[k] = batched_inputs[k]

#             # Batch each field for this input type
#             for field in fields:
#                 k = f"{input_name}_{field}"
#                 batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (max_num_hits,), 0.0)

#         # Batch the target features
#         for target_name, fields in self.dataset_targets.items():
#             if target_name == "particle":
#                 size = (self.max_num_particles, max_num_hits)
#             else:
#                 size = (self.max_num_particles,)

#             # Batch the validity masks for targets
#             k = f"{target_name}_valid"
#             batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)
            
#             for field in fields:
#                 k = f"{target_name}_{field}"
#                 batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)

#         # Batch the metadata
#         batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)
#         print("Batched inputs:", batched_inputs.keys())
#         print("Batched targets:", batched_targets.keys())
#         return batched_inputs, batched_targets


class AtlasMuonCollator:
    def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_targets
        self.max_num_obj = max_num_obj

    def __call__(self, batch):
        inputs, targets = zip(*batch, strict=False)
        # print(targets[0].keys())
        # print(type(inputs))
        # print(type(targets))

        hit_max_sizes = {}
        # print(self.dataset_inputs)
        for input_name in self.dataset_inputs:
            hit_max_sizes[input_name] = max(event[f"{input_name}_valid"].shape[-1] for event in inputs)

        batched_inputs = {}
        batched_targets = {}
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k].unsqueeze(0) for i in inputs], (hit_max_sizes[input_name],), False)

            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k].unsqueeze(0) for i in inputs], (hit_max_sizes[input_name],), 0.0)
        if "particle_hit_valid" in targets[0].keys():
            size = (self.max_num_obj, hit_max_sizes["hit"])
            batched_targets["particle_hit_valid"] = pad_and_concat([t["particle_hit_valid"].unsqueeze(0) for t in targets], size, torch.nan)
        
        for target_name, fields in self.dataset_targets.items():
            # print("This is target:", target_name)
            # print("Fields:", fields)
            if target_name == "particle":
                size = (self.max_num_obj,)
                k = f"{target_name}_valid"
                batched_targets[k] = pad_and_concat([t[k].unsqueeze(0) for t in targets], size, False)
            elif target_name == "hit":
                size = (hit_max_sizes[target_name],)
                # print(size)


            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k].unsqueeze(0) for t in targets], size, torch.nan)
            # print(batched_targets.keys())
        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

        return batched_inputs, batched_targets

class AtlasMuonDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_eval_path: str | None = None,
        event_max_num_particles: int = 6,  # Typically fewer tracks per event in muon data
    ):
        super().__init__()
        # print("We got the hit_eval_path:", hit_eval_path)
        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)

        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.hit_eval_path = hit_eval_path
        
        # Setup hit eval file if specified
        if self.hit_eval_path:
            print(f"Using hit eval dataset {self.hit_eval_path}")
        
        # Load metadata
        with open(self.dirpath / 'metadata.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        self.hit_features = self.metadata['hit_features']
        self.track_features = self.metadata['track_features']
        
        # Load efficient index arrays
        # self.global_event_ids = np.load(self.dirpath / 'event_global_ids.npy')
        self.file_indices = np.load(self.dirpath / 'event_file_indices.npy')
        self.row_indices = np.load(self.dirpath / 'event_row_indices.npy')
        # self.num_hits_per_event = np.load(self.dirpath / 'event_num_hits.npy')
        # self.num_tracks_per_event = np.load(self.dirpath / 'event_num_tracks.npy')
        # self.chunk_info = np.load(self.dirpath / 'chunk_info.npy', allow_pickle=True)
        
        # Calculate number of events to use
        num_events_available = len(self.row_indices)
        
        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available."
            raise ValueError(msg)
        
        if num_events == -1:
            num_events = num_events_available
            
        if num_events == 0:
            raise ValueError("num_events must be greater than 0")
        
        self.num_events = num_events
        
        self.event_max_num_particles = event_max_num_particles
        
        print(f"Created ATLAS muon dataset with {self.num_events:,} events")


    def __len__(self):
        return self.num_events

    # def __getitem__(self, idx):
    #     inputs = {}
    #     targets = {}
 
    #     # DEBUG: Print self.inputs to see if it's corrupted
    #     print(f"DEBUG: Event {idx} - self.inputs: {self.inputs}")
    #     print(f"DEBUG: Event {idx} - self.inputs type: {type(self.inputs)}")
    #     for k, v in self.inputs.items():
    #         print(f"DEBUG: self.inputs[{k}] = {v}, type: {type(v)}")

    #     # Load the event
    #     hits, particles, num_hits, num_tracks = self.load_event(idx)

    #     # DEBUG: Print hits dictionary info before processing
    #     print(f"DEBUG: Event {idx} - hits type: {type(hits)}")
    #     print(f"DEBUG: Event {idx} - hits keys: {list(hits.keys())}")
    #     print(f"DEBUG: Event {idx} - hits keys types: {[type(k) for k in hits.keys()]}")
        
    #     # Build the input hits - using same structure as TrackML
    #     for feature, fields in self.inputs.items():
    #         print(f"DEBUG: Processing feature: {feature}, fields: {fields}")
            
    #         # SAFETY CHECK: Ensure fields is a list/tuple of strings
    #         if not isinstance(fields, (list, tuple)):
    #             print(f"ERROR: fields is not a list/tuple! Type: {type(fields)}")
    #             print(f"Expected fields to be list of strings, got: {fields}")
    #             raise TypeError(f"self.inputs[{feature}] should be a list of field names, got {type(fields)}")
            
    #         inputs[f"{feature}_valid"] = torch.full((num_hits,), True)
    #         targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            
    #         for field in fields:
    #             print(f"DEBUG: Processing field: {field}, type: {type(field)}")
                
    #             # SAFETY CHECK: Ensure field is a string
    #             if not isinstance(field, str):
    #                 print(f"ERROR: field is not a string! Type: {type(field)}")
    #                 print(f"Expected field to be string, got: {field}")
    #                 raise TypeError(f"Field name should be a string, got {type(field)}")
                
    #             # DEBUG: Check if field exists in hits
    #             if field not in hits:
    #                 print(f"ERROR: Field '{field}' not found in hits dictionary!")
    #                 print(f"Available keys: {list(hits.keys())}")
    #                 raise KeyError(f"Field '{field}' not found in hits")
                
    #             # DEBUG: Check the value type
    #             hit_value = hits[field]
    #             print(f"DEBUG: hits[{field}] type: {type(hit_value)}, shape: {getattr(hit_value, 'shape', 'no shape')}")
                
    #             try:
    #                 inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field])
    #             except Exception as e:
    #                 print(f"ERROR: Failed to convert hits[{field}] to tensor: {e}")
    #                 print(f"hits[{field}] = {hits[field]}")
    #                 raise


    def __getitem__(self, idx):
        inputs = {}
        targets = {}
 
        # Load the event
        hits, particles, num_hits, num_tracks = self.load_event(idx)

        # If a hit eval file was specified, read in the predictions from it to use the hit filtering
        
        
        # Build the input hits - using same structure as TrackML
        # print("inputs", hits.keys())
        # print("self.inputs.items():", self.inputs.items())
        for feature, fields in self.inputs.items():
            # print("Fields:", fields)
            # print("feature:", feature)
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True)
            # inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                # print(field)
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field])
                # inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).half()
                # inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field])

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_tracks] = True
        targets["particle_valid"] = targets["particle_valid"]
        # print("Particle valid shape:", targets["particle_valid"].shape)
        # print("Particle valid:", targets["particle_valid"])
        
        
        message = f"Event {idx} has {num_tracks} particles, but limit is {self.event_max_num_particles}"
        assert num_tracks <= self.event_max_num_particles, message

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"])

        # Fill in empty slots with -999s and get the IDs of the particle on each hit
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))]).type(torch.int32)
        # print("Particle IDs:", particle_ids)
        # print("Particle IDs:", particle_ids.shape)
        hit_particle_ids = torch.from_numpy(hits["spacePoint_truthLink"])

        # Create the mask targets
        # print("particle_ids.unsqueeze(-1):", particle_ids.unsqueeze(-1))
        # print("particle_ids.unsqueeze(-1).shape:", particle_ids.unsqueeze(-1).shape)
        # print("hit_particle_ids.unsqueeze(-2):", hit_particle_ids.unsqueeze(-2))
        # print("hit_particle_ids.unsqueeze(-2).shape:", hit_particle_ids.unsqueeze(-2).shape)
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2))
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"].shape)
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"])
        # Create the hit filter targets (all hits are valid for now)
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"])
        # targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)
        # print("Targets hit_on_valid_particle:", targets["hit_on_valid_particle"].shape)
        # print("Targets particle_hit_valid:", targets["hit_on_valid_particle"])

        # Add sample ID
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

         

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                x = torch.full((self.event_max_num_particles,), torch.nan)
                if field in particles:
                    x[:num_tracks] = torch.from_numpy(particles[field][:self.event_max_num_particles])
                targets[f"particle_{field}"] = x
                # targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        """Load a single event from compound HDF5 files using count-based slicing."""
        # Get file and row info using efficient indexing
        file_idx = self.file_indices[idx]
        row_idx = self.row_indices[idx]  # This is the row within the compound arrays
        
        # Get chunk info
        chunk = self.metadata['event_mapping']['chunk_summary'][file_idx]

        # Load from HDF5 file
        h5_file_path = self.dirpath / chunk['h5_file']
        
        try:
            with h5py.File(h5_file_path, 'r') as f:
                # Load feature names from file attributes
                # self.hit_features = [name.decode() if isinstance(name, bytes) else name 
                #                 for name in f.attrs['self.hit_features']]
                # self.track_features = [name.decode() if isinstance(name, bytes) else name 
                #                     for name in f.attrs['self.track_features']]
                
                # Get actual counts for this event
                num_hits = f['num_hits'][row_idx]
                num_tracks = f['num_tracks'][row_idx]
                
                # Load single event from compound arrays using count-based slicing
                hits_array = f['hits'][row_idx, :num_hits]  # Shape: [num_actual_hits, num_features]
                tracks_array = f['tracks'][row_idx, :num_tracks]  # Shape: [num_actual_tracks, num_features]
                
        except Exception as e:
            raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}")

        # Post-processing
        # Convert hits array to dictionary
        hits_dict = {}
        for i, feature_name in enumerate(self.hit_features):
            hits_dict[feature_name] = hits_array[:, i]
            # check here for nans: 
            # print("type of hits_dict[feature_name]:", type(hits_dict[feature_name]))
            # print(hits_dict[feature_name])
            # print(hits_dict[feature_name].shape)
            if np.isnan(hits_dict[feature_name]).any():
                print(f"WARNING: NaN values found in hits for feature '{feature_name}'")
            if np.isinf(hits_dict[feature_name]).any():
                print(f"WARNING: Inf values found in hits for feature '{feature_name}'")
            if hits_dict[feature_name].size == 0:
                print(f"WARNING: Empty hits array for feature '{feature_name}'")
        # TODO: Put proper normalization here instead of scale factors
        # Some scaling:
        hits = {
            # 'spacePoint_globEdgeHighX': hits_dict['spacePoint_globEdgeHighX'],
            # 'spacePoint_globEdgeHighY': hits_dict['spacePoint_globEdgeHighY'],
            # 'spacePoint_globEdgeHighZ': hits_dict['spacePoint_globEdgeHighZ'],
            # 'spacePoint_globEdgeLowX': hits_dict['spacePoint_globEdgeLowX'] ,
            # 'spacePoint_globEdgeLowY': hits_dict['spacePoint_globEdgeLowY'] ,
            # 'spacePoint_globEdgeLowZ': hits_dict['spacePoint_globEdgeLowZ'] ,
            'spacePoint_globEdgeHighX': hits_dict['spacePoint_globEdgeHighX'] * 0.001,
            'spacePoint_globEdgeHighY': hits_dict['spacePoint_globEdgeHighY'] * 0.001,
            'spacePoint_globEdgeHighZ': hits_dict['spacePoint_globEdgeHighZ'] * 0.001,
            'spacePoint_globEdgeLowX': hits_dict['spacePoint_globEdgeLowX'] * 0.001,
            'spacePoint_globEdgeLowY': hits_dict['spacePoint_globEdgeLowY'] * 0.001,
            'spacePoint_globEdgeLowZ': hits_dict['spacePoint_globEdgeLowZ'] * 0.001,
            # 'spacePoint_time': hits_dict['spacePoint_time'] ,
            'spacePoint_time': hits_dict['spacePoint_time'] * 0.00001,
            'spacePoint_driftR': hits_dict['spacePoint_driftR'],
            # Add covariance information
            # 'spacePoint_covXX': hits_dict['spacePoint_covXX'] ,
            # 'spacePoint_covXY': hits_dict['spacePoint_covXY'] ,
            # 'spacePoint_covYX': hits_dict['spacePoint_covYX'] ,
            # 'spacePoint_covYY': hits_dict['spacePoint_covYY'] ,
            'spacePoint_covXX': hits_dict['spacePoint_covXX'] * 0.000001,
            'spacePoint_covXY': hits_dict['spacePoint_covXY'] * 0.000001,
            'spacePoint_covYX': hits_dict['spacePoint_covYX'] * 0.000001,
            'spacePoint_covYY': hits_dict['spacePoint_covYY'] * 0.000001,
            # Add detector information
            # 'spacePoint_channel': hits_dict['spacePoint_channel'],
            'spacePoint_channel': hits_dict['spacePoint_channel']* 0.001,
            'spacePoint_layer': hits_dict['spacePoint_layer'],
            'spacePoint_stationPhi': hits_dict['spacePoint_stationPhi'],
            'spacePoint_stationEta': hits_dict['spacePoint_stationEta'],
            'spacePoint_technology': hits_dict['spacePoint_technology'],
            # Add truth information
            'spacePoint_truthLink': hits_dict['spacePoint_truthLink'],
        }
        
        # Add derived hit fields (vectorized numpy operations)
        hits["r"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2)

        hits["s"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2 + hits["spacePoint_globEdgeLowZ"] ** 2)

        hits["theta"] = np.arccos(np.clip(hits["spacePoint_globEdgeLowZ"] / hits["s"], -1, 1))
        hits["phi"] = np.arctan2(hits["spacePoint_globEdgeLowY"], hits["spacePoint_globEdgeLowX"])
        hits["on_valid_particle"] = hits["spacePoint_truthLink"] >= 0

        # Convert tracks array to dictionary
        tracks_dict = {}
        for i, feature_name in enumerate(self.track_features):
            tracks_dict[feature_name] = tracks_array[:, i]
        
        
        if self.hit_eval_path is not None:
            with h5py.File(self.hit_eval_path, "r") as hit_eval_file:
                assert str(idx) in hit_eval_file, f"Key {idx} not found in file {self.hit_eval_path}"

                hit_filter_pred = hit_eval_file[f"{idx}/preds/final/hit_filter/hit_on_valid_particle"][0]
                for k in hits:
                    hits[k] = hits[k][hit_filter_pred]
            num_hits = np.sum(hit_filter_pred) 

        particles = {
            'particle_id': np.unique(hits["spacePoint_truthLink"][hits["on_valid_particle"]]),  # Sequential IDs
            'truthMuon_pt': tracks_dict['truthMuon_pt'],
            'truthMuon_eta': tracks_dict['truthMuon_eta'],
            'truthMuon_phi': tracks_dict['truthMuon_phi'],
            'truthMuon_q': tracks_dict['truthMuon_q'],
        }
        return hits, particles, num_hits, num_tracks 


    # def load_event(self, idx):
    #     """Load a single event from HDF5 files using the 2D array format - optimized with dictionaries."""
    #     # Get file and row info using efficient indexing
    #     file_idx = self.file_indices[idx]
    #     row_idx = self.row_indices[idx]
        
    #     # Get chunk info
    #     chunk = self.chunk_info[file_idx]
        
    #     # Load from HDF5 file (much faster than parquet for single events)
    #     h5_file_path = self.dirpath / chunk['h5_file']
        
    #     try:
    #         with h5py.File(h5_file_path, 'r') as f:
    #             # Direct access to specific event group
    #             event_group = f[f'event_{row_idx}']
                
    #             # Load feature names from file attributes
    #             self.hit_features = f.attrs['self.hit_features']
    #             self.track_features = f.attrs['self.track_features']
                
    #             # Load hits as 2D array: [num_hits, num_features]
    #             hits_matrix = event_group['hits'][:]
                
    #             # Load tracks as 2D array: [num_tracks, num_features]  
    #             tracks_matrix = event_group['tracks'][:]
                
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}")
        
    #     # Convert hits matrix to dictionary (much faster than DataFrame)

    #     hits_dict = {}
    #     for i, feature_name in enumerate(self.hit_features):
    #         hits_dict[feature_name] = hits_matrix[:, i]
        
    #     # Convert coordinates to meters (if they're in mm)
    #     scale_factor = 0.001  # mm to m
        
    #     # Create hits dictionary with standard naming
    #     hits = {
    #         'x': hits_dict['spacePoint_PositionX'] * scale_factor,
    #         'y': hits_dict['spacePoint_PositionY'] * scale_factor,
    #         'z': hits_dict['spacePoint_PositionZ'] * scale_factor,
    #         'particle_id': hits_dict['spacePoint_truthLink'].astype(int),
    #         # Add covariance information
    #         'cov_xx': hits_dict['spacePoint_covXX'],
    #         'cov_xy': hits_dict['spacePoint_covXY'],
    #         'cov_yy': hits_dict['spacePoint_covYY'],
    #         # Add detector information
    #         'channel': hits_dict['spacePoint_channel'],
    #         'drift_r': hits_dict['spacePoint_driftR'],
    #         'layer': hits_dict['spacePoint_layer'],
    #         'station_phi': hits_dict['spacePoint_stationPhi'],
    #         'station_eta': hits_dict['spacePoint_stationEta'],
    #         'technology': hits_dict['spacePoint_technology'],
    #     }
        
    #     # Add derived hit fields (vectorized numpy operations - much faster)
    #     hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    #     hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
    #     hits["theta"] = np.arccos(np.clip(hits["z"] / hits["s"], -1, 1))
    #     hits["phi"] = np.arctan2(hits["y"], hits["x"])
    #     hits["on_valid_particle"] = hits["particle_id"] >= 0

    #     tracks_dict = {}
    #     for i, feature_name in enumerate(self.track_features):
    #         tracks_dict[feature_name] = tracks_matrix[:, i]
        
    #     particles = {
    #         'particle_id': np.arange(len(tracks_matrix)),  # Sequential IDs
    #         'pt': tracks_dict['truthMuon_pt'],
    #         'eta': tracks_dict['truthMuon_eta'],
    #         'phi': tracks_dict['truthMuon_phi'],
    #         'q': tracks_dict['truthMuon_q'],
    #     }

        
    #     return hits, particles

    # def load_event(self, idx):
    #     # Get file and row info using efficient indexing
    #     file_idx = self.file_indices[idx]
    #     row_idx = self.row_indices[idx]
        
    #     # Get chunk info
    #     chunk = self.chunk_info[file_idx]
        
    #     # Load hits and tracks from parquet files
    #     hits_file_path = self.dirpath / chunk['hits_file']
    #     tracks_file_path = self.dirpath / chunk['tracks_file']
        
    #     # Load specific row from parquet files
    #     hits_table = pq.read_table(hits_file_path, filters=[('row_number', '==', row_idx)])
    #     tracks_table = pq.read_table(tracks_file_path, filters=[('row_number', '==', row_idx)])
        
    #     hits_df = hits_table.to_pandas()
    #     tracks_df = tracks_table.to_pandas()

    #     # Convert DataFrames to dictionaries efficiently
    #     hits_data = hits_df.to_dict('list')  # Fast conversion to dict of lists
    #     tracks_data = tracks_df.to_dict('list')
        
    #     # Create hits DataFrame
    #     hits = pd.DataFrame({
    #         'x': hits_data['spacePoint_PositionX'],
    #         'y': hits_data['spacePoint_PositionY'],
    #         'z': hits_data['spacePoint_PositionZ'],
    #         'particle_id': hits_data['spacePoint_truthLink'],
    #         # Add covariance information
    #         'cov_xx': hits_data['spacePoint_covXX'],
    #         'cov_xy': hits_data['spacePoint_covXY'],
    #         'cov_yy': hits_data['spacePoint_covYY'],
    #         # Add detector information
    #         'channel': hits_data['spacePoint_channel'],
    #         'drift_r': hits_data['spacePoint_driftR'],
    #         'layer': hits_data['spacePoint_layer'],
    #         'station_phi': hits_data['spacePoint_stationPhi'],
    #         'station_eta': hits_data['spacePoint_stationEta'],
    #         'technology': hits_data['spacePoint_technology'],
    #     })
        
    #     # Create particles DataFrame
    #     particles = pd.DataFrame({
    #         'particle_id': range(self.num_tracks_per_event[idx]),  # Sequential IDs
    #         'pt': tracks_data['truthMuon_pt'],
    #         'eta': tracks_data['truthMuon_eta'],
    #         'phi': tracks_data['truthMuon_phi'],
    #         'q': tracks_data['truthMuon_q'],
    #     })
        
    #     # Convert coordinates to meters (if they're in mm)
    #     scale_factor = 0.001  # mm to m
    #     for coord in ['x', 'y', 'z']:
    #         hits[coord] *= scale_factor
        
    #     # Add derived hit fields (similar to TrackML)
    #     hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    #     hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        
    #     # Avoid division by zero
    #     hits["theta"] = np.arccos(np.clip(hits["z"] / hits["s"], -1, 1))
    #     hits["phi"] = np.arctan2(hits["y"], hits["x"])
    #     hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
        
    #     # Avoid division by zero for u, v
    #     # r_squared = hits["x"] ** 2 + hits["y"] ** 2
    #     # hits["u"] = np.where(r_squared > 0, hits["x"] / r_squared, 0)
    #     # hits["v"] = np.where(r_squared > 0, hits["y"] / r_squared, 0)
        
    #     # # Add derived particle fields
    #     # particles["px"] = particles["pt"] * np.cos(particles["phi"])
    #     # particles["py"] = particles["pt"] * np.sin(particles["phi"])
    #     # particles["pz"] = particles["pt"] * np.sinh(particles["eta"])
    #     # particles["p"] = particles["pt"] * np.cosh(particles["eta"])
    #     # particles["qopt"] = particles["q"] / particles["pt"]
    #     # particles["theta"] = 2 * np.arctan(np.exp(-particles["eta"]))
    #     # particles["costheta"] = np.cos(particles["theta"])
    #     # particles["sintheta"] = np.sin(particles["theta"])
    #     # particles["cosphi"] = np.cos(particles["phi"])
    #     # particles["sinphi"] = np.sin(particles["phi"])
        
    #     # Mark which hits are on valid particles (simplified for now)
    #     hits["on_valid_particle"] = hits["particle_id"] >= 0
    #     # Handle noise hits (particle_id == -1)
    #     hits.loc[hits["particle_id"] == -1, "on_valid_particle"] = False

        
        
    #     return hits, particles
    # def load_event(self, idx):
    #     """Load a single event from HDF5 files using the 2D array format - optimized with dictionaries."""
    #     # Get file and row info using efficient indexing
    #     file_idx = self.file_indices[idx]
    #     row_idx = self.row_indices[idx]
        
    #     # Get chunk info
    #     chunk = self.chunk_info[file_idx]
        
    #     # Load from HDF5 file (much faster than parquet for single events)
    #     h5_file_path = self.dirpath / chunk['h5_file']
    #     try:
    #         with h5py.File(h5_file_path, 'r') as f:
    #             # Direct access to specific event group
    #             event_group = f[f'event_{row_idx}']
    #             # print("Loading event group:", event_group)
    #             # Load feature names from file attributes
    #             self.hit_features = f.attrs['self.hit_features']
    #             self.track_features = f.attrs['self.track_features']
                
    #             # Load hits as 2D array: [num_hits, num_features]
    #             hits_matrix = event_group['hits'][:]
                
    #             # Load tracks as 2D array: [num_tracks, num_features]  
    #             tracks_matrix = event_group['tracks'][:]

    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}")
    #     # Convert hits matrix to dictionary (much faster than DataFrame)

    #     hits_dict = {}
    #     for i, feature_name in enumerate(self.hit_features):
    #         hits_dict[feature_name] = hits_matrix[:, i]
        
    #     # Convert coordinates to meters (if they're in mm)
    #     scale_factor = 0.001  # mm to m
        
    #     # Create hits dictionary with standard naming
    #     hits = {
    #         'x': hits_dict['spacePoint_PositionX'] * scale_factor,
    #         'y': hits_dict['spacePoint_PositionY'] * scale_factor,
    #         'z': hits_dict['spacePoint_PositionZ'] * scale_factor,
    #         'particle_id': hits_dict['spacePoint_truthLink'].astype(int),
    #         # Add covariance information
    #         'cov_xx': hits_dict['spacePoint_covXX'],
    #         'cov_xy': hits_dict['spacePoint_covXY'],
    #         'cov_yy': hits_dict['spacePoint_covYY'],
    #         # Add detector information
    #         'channel': hits_dict['spacePoint_channel'],
    #         'drift_r': hits_dict['spacePoint_driftR'],
    #         'layer': hits_dict['spacePoint_layer'],
    #         'station_phi': hits_dict['spacePoint_stationPhi'],
    #         'station_eta': hits_dict['spacePoint_stationEta'],
    #         'technology': hits_dict['spacePoint_technology'],
    #     }
            
    #     # Add derived hit fields (vectorized numpy operations - much faster)
    #     hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    #     hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
    #     hits["theta"] = np.arccos(np.clip(hits["z"] / hits["s"], -1, 1))
    #     hits["phi"] = np.arctan2(hits["y"], hits["x"])
    #     hits["on_valid_particle"] = hits["particle_id"] >= 0

        
    #     # Convert tracks matrix to dictionary

    #     tracks_dict = {}
    #     for i, feature_name in enumerate(self.track_features):
    #         tracks_dict[feature_name] = tracks_matrix[:, i]
        
    #     particles = {
    #         'particle_id': np.arange(len(tracks_matrix)),  # Sequential IDs
    #         'pt': tracks_dict['truthMuon_pt'],
    #         'eta': tracks_dict['truthMuon_eta'],
    #         'phi': tracks_dict['truthMuon_phi'],
    #         'q': tracks_dict['truthMuon_q'],
    #     }
        
    #     return hits, particles
    # def load_event(self, idx):
    #     # Get file and row info using efficient indexing
    #     file_idx = self.file_indices[idx]
    #     row_idx = self.row_indices[idx]
        
    #     # Get chunk info
    #     chunk = self.chunk_info[file_idx]
        
    #     # Load hits and tracks from parquet files
    #     hits_file_path = self.dirpath / chunk['hits_file']
    #     tracks_file_path = self.dirpath / chunk['tracks_file']
        
    #     # Load the specific row (event) from parquet files without row_number filter

    #     hits_table = pq.read_table(hits_file_path)
    #     tracks_table = pq.read_table(tracks_file_path)

    #     # Convert to pandas and extract the specific event row
    #     hits_df = hits_table.to_pandas()
    #     tracks_df = tracks_table.to_pandas()
        
    #     # Check bounds
    #     if row_idx >= len(hits_df):
    #         raise IndexError(f"Event index {row_idx} out of bounds for hits data")
        
    #     # Extract the specific event (row) which contains lists of hit data
    #     hits_row = hits_df.iloc[row_idx]
    #     tracks_row = tracks_df.iloc[row_idx] if row_idx < len(tracks_df) else None
        
    #     # Convert list data to individual hit records
        
    #     # Create hits DataFrame from the list data
    #     hits = pd.DataFrame({
    #         'x': hits_row['spacePoint_PositionX'],
    #         'y': hits_row['spacePoint_PositionY'], 
    #         'z': hits_row['spacePoint_PositionZ'],
    #         'particle_id': hits_row['spacePoint_truthLink'],
    #         # Add covariance information
    #         'cov_xx': hits_row['spacePoint_covXX'],
    #         'cov_xy': hits_row['spacePoint_covXY'],
    #         'cov_yy': hits_row['spacePoint_covYY'],
    #         # Add detector information
    #         'channel': hits_row['spacePoint_channel'],
    #         'drift_r': hits_row['spacePoint_driftR'],
    #         'layer': hits_row['spacePoint_layer'],
    #         'station_phi': hits_row['spacePoint_stationPhi'],
    #         'station_eta': hits_row['spacePoint_stationEta'],
    #         'technology': hits_row['spacePoint_technology'],
    #     })
        
    #     # Create particles DataFrame from tracks data
    #     if tracks_row is not None and self.num_tracks_per_event[idx] > 0:
    #         # Assuming tracks data also has list format similar to hits
    #         particles = pd.DataFrame({
    #             'particle_id': range(self.num_tracks_per_event[idx]),  # Sequential IDs
    #             'pt': tracks_row['truthMuon_pt'] if 'truthMuon_pt' in tracks_row.index else [],
    #             'eta': tracks_row['truthMuon_eta'] if 'truthMuon_eta' in tracks_row.index else [],
    #             'phi': tracks_row['truthMuon_phi'] if 'truthMuon_phi' in tracks_row.index else [],
    #             'q': tracks_row['truthMuon_q'] if 'truthMuon_q' in tracks_row.index else [],
    #         })
    #     else:
    #         # Handle case where no track data is available
    #         particles = pd.DataFrame({
    #             'particle_id': [],
    #             'pt': [],
    #             'eta': [],
    #             'phi': [],
    #             'q': [],
    #         })
        
    #     # Convert coordinates to meters (if they're in mm)
    #     scale_factor = 0.001  # mm to m
    #     for coord in ['x', 'y', 'z']:
    #         hits[coord] = hits[coord] * scale_factor
        
    #     # Add derived hit fields (similar to TrackML)
    #     hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    #     hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        
    #     # Avoid division by zero
    #     hits["theta"] = np.arccos(np.clip(hits["z"] / hits["s"], -1, 1))
    #     hits["phi"] = np.arctan2(hits["y"], hits["x"])
    #     # hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
        
    #     # # Add u, v coordinates (avoiding division by zero)
    #     # r_squared = hits["x"] ** 2 + hits["y"] ** 2
    #     # safe_r = np.where(r_squared > 1e-10, np.sqrt(r_squared), 1e-5)
    #     # hits["u"] = hits["x"] / safe_r
    #     # hits["v"] = hits["y"] / safe_r
        
    #     # Add the "on_valid_particle" field (assuming all hits are valid for now)
    #     hits["on_valid_particle"] = np.ones(len(hits), dtype=bool)
        
    #     return hits, particles

def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)

# def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
#     """Pads each tensor in items to target_size, then stacks along a new batch dimension."""
#     padded = []
#     for item in items:
#         # If item is 1D and target_size is 1D, pad to target_size
#         # If item is 1D and target_size is 2D, unsqueeze first
#         if item.dim() == len(target_size):
#             padded.append(pad_to_size(item, target_size, pad_value))
#         elif item.dim() + 1 == len(target_size):
#             padded.append(pad_to_size(item.unsqueeze(0), target_size, pad_value))
#         else:
#             raise ValueError(f"pad_and_concat: item shape {item.shape} incompatible with target_size {target_size}")
#     return torch.cat(padded, dim=0)


# class AtlasMuonCollator:
#     def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
#         self.dataset_inputs = dataset_inputs
#         self.dataset_targets = dataset_targets
#         self.max_num_obj = max_num_obj

#     def __call__(self, batch):
#         inputs, targets = zip(*batch, strict=False)
        
#         # print("This is the AtlasMuonCollator __call__ method")
#         # print("Inputs:", type(inputs), len(inputs), inputs[0].keys())
#         # print("Targets:", targets)
#         # print("size of every single input:", inputs[0]["hit_valid"].shape)
#         # print("hit_x_shape:", inputs[0]["hit_x"].shape)
#         # print("particle:", targets[0]["particle_valid"].shape)
#         # Find the maximum number of hits across all events in the batch
#         hit_max_sizes = {}
#         # print("Dataset inputs:", self.dataset_inputs)
#         for input_name in self.dataset_inputs:
#             hit_max_sizes[input_name] = max(event[f"{input_name}"].shape[-1] for event in inputs)
#         # print("Hit max sizes:", hit_max_sizes)

#         batched_inputs = {}
#         batched_targets = {}
        
#         # Batch the input hits with padding
#         for input_name, fields in self.dataset_inputs.items():
#             k = f"{input_name}_valid"
#             batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)

#             # Some tasks might require to know hit padding info for loss masking
#             batched_targets[k] = batched_inputs[k]

#             for field in fields:
#                 k = f"{input_name}_{field}"
#                 batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)

#         # Batch the targets with padding
#         for target_name, fields in self.dataset_targets.items():
#             if target_name == "particle":
#                 size = (self.max_num_obj,)
#             elif target_name.startswith("particle_") and "_" in target_name:
#                 # For particle-hit association targets like "particle_hit_valid"
#                 hit = target_name.split("_")[1] if len(target_name.split("_")) > 2 else list(self.dataset_inputs.keys())[0]
#                 if hit in hit_max_sizes:
#                     size = (self.max_num_obj, hit_max_sizes[hit])
#                 else:
#                     size = (self.max_num_obj, max(hit_max_sizes.values()))
#             elif target_name.startswith("hit_"):
#                 # For hit-level targets like "hit_on_valid_particle"
#                 input_name = list(self.dataset_inputs.keys())[0]  # Assume first input type for hit targets
#                 size = (hit_max_sizes[input_name],)
#             else:
#                 # Fallback for other target types
#                 size = (self.max_num_obj,)

#             k = f"{target_name}_valid"
#             if k in targets[0]:  # Only process if this target exists
#                 batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

#             for field in fields:
#                 k = f"{target_name}_{field}"
#                 if k in targets[0]:  # Only process if this field exists
#                     batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)

#         # Batch the metadata
#         batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

#         return batched_inputs, batched_targets


class AtlasMuonDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int = 1,
        test_dir: str | None = None,
        pin_memory: bool = True,
        use_smart_batching: bool = False,  # Disable smart batching by default
        drop_last: bool = False,
        max_hits_per_batch: int = 50000,  # Memory safety limit
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test  
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.use_smart_batching = use_smart_batching
        self.drop_last = drop_last
        self.max_hits_per_batch = max_hits_per_batch
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            self.train_dataset = AtlasMuonDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dataset = AtlasMuonDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                hit_eval_path=self.hit_eval_val,
                **self.kwargs,
            )

        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dataset):,} events")
            print(f"Created validation dataset with {len(self.val_dataset):,} events")
            if self.use_smart_batching:
                print(f"Using smart batching with batch size {self.batch_size}, max {self.max_hits_per_batch:,} hits per batch")

        if stage == "test":
            assert self.test_dir is not None, "No test directory specified"
            self.test_dataset = AtlasMuonDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_val,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: AtlasMuonDataset, shuffle: bool):
        # Create appropriate sampler and batch_sampler
        if self.use_smart_batching:
            batch_sampler = SmartBatchSampler(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=self.drop_last,
                max_hits_per_batch=self.max_hits_per_batch
            )
            return DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                collate_fn=AtlasMuonCollator(
                    dataset.inputs, 
                    dataset.targets, 
                    dataset.event_max_num_particles
                ),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # Fallback to standard batching
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=AtlasMuonCollator(
                    dataset.inputs, 
                    dataset.targets, 
                    dataset.event_max_num_particles
                ),
                sampler=None,
                num_workers=self.num_workers,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
            )
    # def get_dataloader(self, stage: str, dataset: AtlasMuonDataset, shuffle: bool):
    #     # Disable multiprocessing when using hit filtering to avoid race conditions
    #     # num_workers = 0 if dataset.hit_eval_path is not None else self.num_workers
    #     # num_workers = 10
        
    #     # Create appropriate sampler and batch_sampler
    #     if self.use_smart_batching:
    #         batch_sampler = SmartBatchSampler(
    #             dataset=dataset,
    #             batch_size=self.batch_size,
    #             shuffle=shuffle,
    #             drop_last=self.drop_last,
    #             max_hits_per_batch=self.max_hits_per_batch
    #         )
    #         return DataLoader(
    #             dataset=dataset,
    #             batch_sampler=batch_sampler,
    #             collate_fn=AtlasMuonCollator(
    #                 dataset.inputs, 
    #                 dataset.targets, 
    #                 dataset.event_max_num_particles
    #             ),
    #             num_workers=num_workers,
    #             pin_memory=self.pin_memory,
    #         )
    #     else:
    #         # Fallback to standard batching
    #         return DataLoader(
    #             dataset=dataset,
    #             batch_size=self.batch_size,
    #             collate_fn=AtlasMuonCollator(
    #                 dataset.inputs, 
    #                 dataset.targets, 
    #                 dataset.event_max_num_particles
    #             ),
    #             sampler=None,
    #             num_workers=num_workers,
    #             shuffle=shuffle,
    #             pin_memory=self.pin_memory,
    #         )


    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
