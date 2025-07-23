from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class TrackMLDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_hits=3,
        event_max_num_particles=1000,
        hit_eval_path: str | None = None,
    ):
        super().__init__()

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # Get a list of event names
        event_names = [Path(file).stem.replace("-parts", "") for file in Path(dirpath).glob("event*-parts.parquet")]

        # Calculate the number of events that will actually be used
        num_events_available = len(event_names)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_events < 0:
            num_events = num_events_available

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        # Metadata
        self.dirpath = Path(dirpath)
        self.hit_eval_path = hit_eval_path
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]
        self.sample_ids = [int(name.removeprefix("event")) for name in self.event_names]

        # Setup hit eval file if specified
        if self.hit_eval_path:
            print(f"Using hit eval dataset {self.hit_eval_path}")

        # Hit level cuts
        self.hit_volume_ids = hit_volume_ids

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

    def __len__(self):
        return int(self.num_events)

    def __getitem__(self, idx):
        inputs = {}
        targets = {}

        # Load the event
        hits, particles = self.load_event(idx)
        num_particles = len(particles)
        print("This is number of particles:", num_particles)

        # Build the input hits
        for feature, fields in self.inputs.items():
            print("This is feature:", feature)
            print("This is length of hits:", len(hits))
            inputs[f"{feature}_valid"] = torch.full((len(hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field].values).unsqueeze(0).half()
        # print("Inputs:", inputs) # Inputs: {'hit_valid': tensor([[True, True, True,  ..., True, True, True]]), 'hit_x': tensor([[-0.9937, -0.7490, -0.6821,  ..., -1.5156, -1.3838, -1.4102]],
    #    dtype=torch.float16), 'hit_y': tensor([[-0.0850, -0.0156, -0.0550,  ...,  0.1144,  0.0817,  0.0264]],
    #    dtype=torch.float16), 'hit_z': tensor([[-15.0234, -15.0234, -15.0234,  ...,  14.9766,  14.9766,  14.9766]],
    #    dtype=torch.float16), 'hit_r': tensor([[0.9971, 0.7495, 0.6846,  ..., 1.5195, 1.3857, 1.4102]],
    #    dtype=torch.float16), 'hit_s': tensor([[15.0547, 15.0469, 15.0391,  ..., 15.0547, 15.0469, 15.0469]],
    #    dtype=torch.float16), 'hit_eta': tensor([[-3.4062, -3.6914, -3.7832,  ...,  2.9844,  3.0762,  3.0586]],
    #    dtype=torch.float16), 'hit_phi': tensor([[-3.0566, -3.1211, -3.0605,  ...,  3.0664,  3.0820,  3.1230]],
    #    dtype=torch.float16), 'hit_u': tensor([[-0.9990, -1.3340, -1.4561,  ..., -0.6562, -0.7202, -0.7090]],
    #    dtype=torch.float16), 'hit_v': tensor([[-0.0854, -0.0277, -0.1173,  ...,  0.0495,  0.0425,  0.0133]],
    #    dtype=torch.float16), 'hit_charge_frac': tensor([[0.2773, 0.2861, 0.3020,  ..., 0.2908, 0.1534, 0.1146]],
    #    dtype=torch.float16), 'hit_leta': tensor([[2.0918, 2.0918, 2.0918,  ..., 2.0918, 1.6230, 1.3027]],
    #    dtype=torch.float16), 'hit_lphi': tensor([[0.8442, 0.8442, 0.8442,  ..., 0.8442, 1.1523, 1.2832]],
    #    dtype=torch.float16), 'hit_lx': tensor([[0.0500, 0.0500, 0.0500,  ..., 0.0500, 0.0500, 0.0500]],
    #    dtype=torch.float16), 'hit_ly': tensor([[0.0562, 0.0562, 0.0562,  ..., 0.0562, 0.1125, 0.1687]],
    #    dtype=torch.float16), 'hit_lz': tensor([[0.3000, 0.3000, 0.3000,  ..., 0.3000, 0.3000, 0.3000]],
    #    dtype=torch.float16), 'hit_geta': tensor([[-2.0918, -2.0918, -2.0918,  ...,  2.0918,  1.6230,  1.3027]],
    #    dtype=torch.float16), 'hit_gphi': tensor([[-2.3359, -2.3359, -2.3359,  ...,  2.3691,  2.6777,  2.8066]],
    #    dtype=torch.float16)}
        # print("Targets:", targets) # Targets: {'hit_valid': tensor([[True, True, True,  ..., True, True, True]])}

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_particles] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)
        # print("Targets particle_valid:", targets["particle_valid"])
        # print("Targets particle_valid:", targets["particle_valid"].shape)
        # Targets particle_valid: tensor([[ True,  True,  True,  ..., False, False, False]])
# Targets particle_valid: torch.Size([1, 3000])

        message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
        assert num_particles <= self.event_max_num_particles, message

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"].values)


        # Fill in empty slots with -1s and get the IDs of the particle on each hit
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))])
        hit_particle_ids = torch.from_numpy(hits["particle_id"].values)

        # Create the mask targets
        # print("particle_ids:", particle_ids.unsqueeze(-1).shape)
        # print("particle_ids:", particle_ids.unsqueeze(-1))
        # print("hit_particle_ids:", hit_particle_ids.unsqueeze(-2).shape)
        # print("hit_particle_ids:", hit_particle_ids.unsqueeze(-2))
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"].shape)
        # print("Targets particle_hit_valid:", targets["particle_hit_valid"])
        # Create the hit filter targets
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"].to_numpy()).unsqueeze(0)
        # print("Targets hit_on_valid_particle:", targets["hit_on_valid_particle"].shape)
        # print("Targets hit_on_valid_particle:", targets["hit_on_valid_particle"])
        # Add sample ID
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)
        # print("Targets sample_id:", targets["sample_id"].shape)
        # Targets hit_on_valid_particle: torch.Size([1, 10000])

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                # This acts as a sanity check that we correctly mask out null slots in the loss
                x = torch.full((self.event_max_num_particles,), torch.nan)

                x[:num_particles] = torch.from_numpy(particles[field].to_numpy()[: self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        event_name = self.event_names[idx]

        particles = pd.read_parquet(self.dirpath / Path(event_name + "-parts.parquet"))
        hits = pd.read_parquet(self.dirpath / Path(event_name + "-hits.parquet"))

        # Make the detector volume selection
        if self.hit_volume_ids:
            hits = hits[hits["volume_id"].isin(self.hit_volume_ids)]

        # Scale the input coordinates to in meters so they are ~ 1
        for coord in ["x", "y", "z"]:
            hits[coord] *= 0.01

        # Add extra hit fields
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["theta"] = np.arccos(hits["z"] / hits["s"])
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # Add extra particle fields
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["qopt"] = particles["q"] / particles["pt"]
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["theta"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["costheta"] = np.cos(particles["theta"])
        particles["sintheta"] = np.sin(particles["theta"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])

        # Apply particle level cuts based on particle fields
        particles = particles[particles["pt"] > self.particle_min_pt]
        particles = particles[particles["eta"].abs() < self.particle_max_abs_eta]

        # Apply particle cut based on hit content
        counts = hits["particle_id"].value_counts()
        keep_particle_ids = counts[counts >= self.particle_min_num_hits].index.to_numpy()
        particles = particles[particles["particle_id"].isin(keep_particle_ids)]

        # Mark which hits are on a valid / reconstructable particle, for the hit filter
        hits["on_valid_particle"] = hits["particle_id"].isin(particles["particle_id"])

        # If a hit eval file was specified, read in the predictions from it to use the hit filtering
        if self.hit_eval_path:
            with h5py.File(self.hit_eval_path, "r") as hit_eval_file:
                assert str(self.sample_ids[idx]) in hit_eval_file, f"Key {self.sample_ids[idx]} not found in file {self.hit_eval_path}"

                # The dataset has shape (1, num_hits)
                hit_filter_pred = hit_eval_file[f"{self.sample_ids[idx]}/preds/final/hit_filter/hit_on_valid_particle"][0]
                hits = hits[hit_filter_pred]

        # TODO: Add back truth based hit filtering

        # Sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["particle_id"].nunique() == len(particles), "Non-unique particle ids"

        # Check that all hits have different phi
        # This is necessary as the fast sorting algorithm used by pytorch can be non-stable
        # if two values are equal, which could cause subtle bugs
        # msg = f"Only {hits['phi'].nunique()} of the {len(hits)} have unique phi"
        # assert hits["phi"].nunique() == len(hits), msg
        # print("This is type of hits:", type(hits))
        # print("This is len of hits:", len(hits))
        return hits, particles


class TrackMLDataModule(LightningDataModule):
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
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
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
            self.train_dataset = TrackMLDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dataset = TrackMLDataset(
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

            self.test_dataset = TrackMLDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: TrackMLDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
