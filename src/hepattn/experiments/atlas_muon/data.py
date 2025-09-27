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



class AtlasMuonDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_eval_path: str | None = None,
        event_max_num_particles: int = 6,  # Typically fewer tracks per event in muon data
        dummy_testing: bool = False,
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
        self.dummy_testing = dummy_testing
        
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
            # inputs[f"{feature}_valid"] = torch.full((num_hits,), True)
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                # print(field)
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).unsqueeze(0)
                # inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).unsqueeze(0).half()
                # inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).half()
                # inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field])

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_tracks] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        # print("Particle valid shape:", targets["particle_valid"].shape)
        # print("sum:", targets["particle_valid"].sum())
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
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"].shape)
        # print("sum:", targets["particle_hit_valid"].sum())
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"])
        # Create the hit filter targets (all hits are valid for now)
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)

        # targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)
        # print("Targets hit_on_valid_particle:", targets["hit_on_valid_particle"].shape)
        # print("sum:", targets["hit_on_valid_particle"].sum())
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
                # targets[f"particle_{field}"] = x
                targets[f"particle_{field}"] = x.unsqueeze(0)

        # for key in inputs.keys():
        #     print(key, inputs[key])
        # for key in targets.keys():
        #     print(key, targets[key])
        # print("sample_id", targets["sample_id"])
        # inputs["spacePoint_truthLink"] = torch.from_numpy(hits["spacePoint_truthLink"]).unsqueeze(0)
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
            'spacePoint_stationIndex': hits_dict['spacePoint_stationIndex'] * 0.1,
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
                # TODO: print out the average match between hit filter and the ground truth you are loading here, 
                # for a quick debugging gut check
                # print("-"*20)
                # print("true positives:", np.sum(hit_filter_pred & hits["on_valid_particle"]))
                # print("false positives:", (np.sum(hit_filter_pred) - np.sum(hits["on_valid_particle"])))
                # print("number of predicted hits:", np.sum(hit_filter_pred))
                # print("number of true hits:", np.sum(hits["on_valid_particle"]))
                # print("-"*20)
                # print("true positives:", np.sum(hit_filter_pred & hits["on_valid_particle"]) / np.sum(hits["on_valid_particle"]))
                # how many did we lose:
                # print("precision:", np.sum(hit_filter_pred & hits["on_valid_particle"]) / np.sum(hit_filter_pred))
                # # how many did we miss:
                # print("false positives:", (np.sum(hit_filter_pred) - np.sum(hits["on_valid_particle"]))/ np.sum(hit_filter_pred))
                # false positives:
                # print("false positives:", np.sum(hit_filter_pred & ~hits["on_valid_particle"]) / np.sum(~hits["on_valid_particle"]))
                for k in hits:
                    hits[k] = hits[k][hit_filter_pred]
            num_hits = np.sum(hit_filter_pred) 


        # This example starts of how this is being used in the Trackml don't run this
        # if self.hit_eval_path:
        #     with h5py.File(self.hit_eval_path, "r") as hit_eval_file:
        #         assert str(self.sample_ids[idx]) in hit_eval_file, f"Key {self.sample_ids[idx]} not found in file {self.hit_eval_path}"

        #         # The dataset has shape (1, num_hits)
        #         hit_filter_pred = hit_eval_file[f"{self.sample_ids[idx]}/preds/final/hit_filter/hit_on_valid_particle"][0]
        #         hits = hits[hit_filter_pred]
        # This example is over!!!!


        # For debugging purposes
        if self.dummy_testing:
            for k in hits: 
                hits[k] = hits[k][hits["on_valid_particle"]]
            num_hits = np.sum(hits["on_valid_particle"])


        particles = {
            'particle_id': np.unique(hits["spacePoint_truthLink"][hits["on_valid_particle"]]),  # Sequential IDs
            'truthMuon_pt': tracks_dict['truthMuon_pt'],
            'truthMuon_eta': tracks_dict['truthMuon_eta'],
            'truthMuon_phi': tracks_dict['truthMuon_phi'],
            'truthMuon_q': tracks_dict['truthMuon_q'],
            "truthMuon_qpt": tracks_dict['truthMuon_q'] / tracks_dict['truthMuon_pt'],
        }
        return hits, particles, num_hits, num_tracks 
    

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
        # print(hit_max_sizes)
        batched_inputs = {}
        batched_targets = {}
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)

            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)
        if "particle_hit_valid" in targets[0].keys():
            size = (self.max_num_obj, hit_max_sizes["hit"])
            batched_targets["particle_hit_valid"] = pad_and_concat([t["particle_hit_valid"] for t in targets], size, False)
        
        for target_name, fields in self.dataset_targets.items():
            # print("This is target:", target_name)
            # print("Fields:", fields)
            if target_name == "particle":
                size = (self.max_num_obj,)
                
            elif target_name == "hit":
                size = (hit_max_sizes[target_name],)
                # print(size)
            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)
            # print(batched_targets.keys())
        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)
        # for key in batched_inputs.keys():
            # print(f"Input {key} shape: {batched_inputs[key].shape}")
            # print(f"Input {key}: {batched_inputs[key]}")
        # for key in batched_targets.keys():
        #     print(f"Target {key}: {batched_targets[key]}")
        return batched_inputs, batched_targets

class AtlasMuonDataModule(LightningDataModule):
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
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
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
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            self.train_dataset = AtlasMuonDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                **self.kwargs,
            )

        if stage == "fit" or stage == "validate":
            self.val_dataset = AtlasMuonDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                hit_eval_path=self.hit_eval_val,
                **self.kwargs,
            )
        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dataset):,} events")
            print(f"Created validation dataset with {len(self.val_dataset):,} events")
            
        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dataset = AtlasMuonDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: AtlasMuonDataset, shuffle: bool, 
    prefetch_factor: int = 8):
        # Set prefetch_factor to None when num_workers=0 to avoid ValueError
        actual_prefetch_factor = None if self.num_workers == 0 else prefetch_factor
        
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
            prefetch_factor=actual_prefetch_factor,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)

