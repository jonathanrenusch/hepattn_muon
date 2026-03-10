from pathlib import Path
import yaml
import numpy as np
import torch
from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import h5py
from hepattn.utils.tensor_utils import pad_to_size


# Scale factors applied to raw hit features to bring values into a reasonable
# numerical range for the network. Features not listed here are used as-is.
FEATURE_SCALES: dict[str, float] = {
    "spacePoint_globEdgeHighX": 0.001,
    "spacePoint_globEdgeHighY": 0.001,
    "spacePoint_globEdgeHighZ": 0.001,
    "spacePoint_globEdgeLowX":  0.001,
    "spacePoint_globEdgeLowY":  0.001,
    "spacePoint_globEdgeLowZ":  0.001,
    "spacePoint_time":          1e-5,
    "spacePoint_covXX":         1e-6,
    "spacePoint_covXY":         1e-6,
    "spacePoint_covYX":         1e-6,
    "spacePoint_covYY":         1e-6,
    "spacePoint_channel":       0.001,
    "spacePoint_stationIndex":  0.1,
}


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
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)

        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.hit_eval_path = hit_eval_path
        self.dummy_testing = dummy_testing

        if self.hit_eval_path:
            print(f"Using hit eval dataset {self.hit_eval_path}")

        # Load metadata
        with open(self.dirpath / 'metadata.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)

        self.hit_features = self.metadata['hit_features']
        self.track_features = self.metadata['track_features']

        # Load efficient index arrays
        self.file_indices = np.load(self.dirpath / 'event_file_indices.npy')
        self.row_indices = np.load(self.dirpath / 'event_row_indices.npy')

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

        hits, particles, num_hits, num_tracks = self.load_event(idx)

        # Build the input hits
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).unsqueeze(0)

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_tracks] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        message = f"Event {idx} has {num_tracks} particles, but limit is {self.event_max_num_particles}"
        assert num_tracks <= self.event_max_num_particles, message

        # Build the hit-to-particle mask targets
        particle_ids = torch.from_numpy(particles["particle_id"])
        # Fill empty slots with -999
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))]).type(torch.int32)
        hit_particle_ids = torch.from_numpy(hits["spacePoint_truthLink"])
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # Hit filter targets
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)

        # Sample ID for evaluation
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                x = torch.full((self.event_max_num_particles,), torch.nan)
                if field in particles:
                    x[:num_tracks] = torch.from_numpy(particles[field][:self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        """Load a single event from compound HDF5 files using count-based slicing."""
        file_idx = self.file_indices[idx]
        row_idx = self.row_indices[idx]

        chunk = self.metadata['event_mapping']['chunk_summary'][file_idx]
        h5_file_path = self.dirpath / chunk['h5_file']

        try:
            with h5py.File(h5_file_path, 'r') as f:
                num_hits = f['num_hits'][row_idx]
                num_tracks = f['num_tracks'][row_idx]
                hits_array = f['hits'][row_idx, :num_hits]    # Shape: [num_actual_hits, num_features]
                tracks_array = f['tracks'][row_idx, :num_tracks]  # Shape: [num_actual_tracks, num_features]
        except Exception as e:
            raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}")

        # Convert hits array to dictionary and apply feature scaling
        hits_dict = {}
        for i, feature_name in enumerate(self.hit_features):
            hits_dict[feature_name] = hits_array[:, i]
            if np.isnan(hits_dict[feature_name]).any():
                print(f"WARNING: NaN values found in hits for feature '{feature_name}'")
            if np.isinf(hits_dict[feature_name]).any():
                print(f"WARNING: Inf values found in hits for feature '{feature_name}'")
            if hits_dict[feature_name].size == 0:
                print(f"WARNING: Empty hits array for feature '{feature_name}'")

        # Apply feature scaling using FEATURE_SCALES; unscaled features are passed through as-is
        hits = {name: arr * FEATURE_SCALES.get(name, 1.0) for name, arr in hits_dict.items()}

        # Add derived hit fields (vectorized numpy operations)
        hits["r"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2)
        hits["s"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2 + hits["spacePoint_globEdgeLowZ"] ** 2)
        hits["theta"] = np.arccos(np.clip(hits["spacePoint_globEdgeLowZ"] / hits["s"], -1, 1))
        hits["phi"] = np.arctan2(hits["spacePoint_globEdgeLowY"], hits["spacePoint_globEdgeLowX"])
        hits["on_valid_particle"] = hits["spacePoint_truthLink"] >= 0

        # Convert tracks array to dictionary
        tracks_dict = {feature_name: tracks_array[:, i] for i, feature_name in enumerate(self.track_features)}

        # For debugging purposes: restrict to only valid-particle hits when dummy_testing is set
        if self.dummy_testing:
            valid_mask = hits["on_valid_particle"]
            for k in hits:
                hits[k] = hits[k][valid_mask]
            num_hits = int(valid_mask.sum())

        particles = {
            'particle_id': np.unique(hits["spacePoint_truthLink"][hits["on_valid_particle"]]),
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

        hit_max_sizes = {
            input_name: max(event[f"{input_name}_valid"].shape[-1] for event in inputs)
            for input_name in self.dataset_inputs
        }
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
            if target_name == "particle":
                size = (self.max_num_obj,)
            elif target_name == "hit":
                size = (hit_max_sizes[target_name],)
            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)

        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)
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

    def test_dataloader(self, shuffle=False):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=shuffle)

