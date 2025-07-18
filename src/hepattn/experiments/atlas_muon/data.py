from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import uproot


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class MuonTracking(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        particle_min_pt: float = 5.0,
        particle_max_abs_eta: float = 2.7,
        particle_min_num_hits=3,
        event_max_num_particles=1000,
        hit_eval_path: str | None = None,
    ):

        self.sampling_seed = 42
        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.hit_volume_ids = hit_volume_ids
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits
        self.event_max_num_particles = event_max_num_particles
        self.hit_eval_path = hit_eval_path


         # Calculate the number of events that will actually be used
        print("-" * 30, f" Calculating number of events for {dirpath} ", "-" * 30)
        self.files = self._get_files()
        self.events_per_file = self._count_events_per_file()
        # self.file_indices, self.local_indices = self._create_event_mapping()
        total_events_available = sum(self.events_per_file.values())
        
        if self.num_events == -1:
            self.num_events = total_events_available
        elif self.num_events == 0:
            raise ValueError("num_events must be greater than 0")

        if self.num_events > total_events_available:
            raise ValueError(f"num_events ({self.num_events}) exceeds available events ({total_events_available})")
        else:
            print(f"Using {self.num_events} events out of {total_events_available} available events.")


        

    def _get_files(self) -> list[Path]:
        files = list(self.dirpath.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No ROOT files found in {self.dirpath}")
        # Filter out invalid files (empty or corrupted)
        valid_files = [f for f in files if is_valid_file(f)]
        if not valid_files:
            raise FileNotFoundError(f"No valid ROOT files found in {self.dirpath}")
        return sorted(valid_files)  # Sort for consistent ordering

    def _create_event_mapping(self):
        """Create a memory-efficient event mapping using numpy arrays while selecting only events with muons that pass the selection criteria."""

        total_events = sum(self.events_per_file.values())
        
        # Use numpy arrays for memory efficiency
        file_indices = np.empty(len(self.events_per_file), dtype=np.int16)  # Assumes < 32k files
        local_indices = np.empty(total_events, dtype=np.int32)
        
        global_idx = 0
        for file_idx, file_path in enumerate(self.files):
            # apply selection criteria: 
            with uproot.open(file_path) as root_file:
                # Using the first tree
                tree_keys = [key for key in root_file.keys() if ';' in key]
                if not tree_keys:
                    continue
                tree_name = tree_keys[0].split(';')[0]
                tree = root_file[tree_name]
                # checking for true tracks and hits per track: 
                truth_track_ass = tree.arrays("spacePoint_truthLink", library="np", entry_start=0, entry_stop=-1)
                for truth, in truth_track_ass:
                    
                    print(truth_track_ass, "-"*40)

                    # checking if there is any tracks higher than min_pt: 

                    # checking if there is any tracks with => 3 hits

            num_events = self.events_per_file[file_path]
            file_indices[global_idx:global_idx + num_events] = file_idx
            local_indices[global_idx:global_idx + num_events] = np.arange(num_events)
            global_idx += num_events
        
        return file_indices, local_indices

    



    def _count_events_per_file(self):
        """Count events in each ROOT file using uproot."""
        events_per_file = {}

        for file_path in self.files:
            try:
                with uproot.open(file_path) as root_file:
                    # Using the first tree
                    tree_keys = [key for key in root_file.keys() if ';' in key]
                    if tree_keys:
                        tree = tree_keys[0].split(';')[0]
                        events_per_file[file_path] = root_file[tree].num_entries
                    else:
                        events_per_file[file_path] = 0
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                events_per_file[file_path] = 0

        return events_per_file

    def __len__(self):
        """Return the total number of events."""
        return self.num_events

    def load_event(self, idx):
        file_idx = self.file_indices[idx]
        local_event_idx = self.local_indices[idx]
        file_path = self.files[file_idx]
        with uproot.open(file_path) as root_file:
            # Using the first tree
            tree_keys = [key for key in root_file.keys() if ';' in key]
            if not tree_keys:
                raise ValueError(f"No valid trees found in {file_path}")
            tree_name = tree_keys[0].split(';')[0]
            tree = root_file[tree_name]
            hits = tree.arrays(library="pd", entry_start=local_event_idx, entry_stop=local_event_idx + 1)
            particles = tree.arrays(library="pd", entry_start=local_event_idx, entry_stop=local_event_idx + 1, filter_name="particles")

        # ...rest of implementation...
    def inspect_root_files(self):
        """Inspect the structure of ROOT files to understand their contents."""
        for file_path in self.files[:1]:  # Just inspect the first file
            print(f"Inspecting file: {file_path}")
            with uproot.open(file_path) as root_file:
                print(f"File keys: {list(root_file.keys())}")
                
                for key in root_file.keys():
                    if ';' in key:
                        tree_name = key.split(';')[0]
                        tree = root_file[tree_name]
                        print(f"\nTree: {tree_name}")
                        print(f"Number of entries: {tree.num_entries}")
                        print(f"Number of branches: {len(tree.keys())}")
                        
                        print("Branches:")
                        for branch_name in sorted(tree.keys()):
                            branch = tree[branch_name]
                            print(f"  {branch_name:<30} {branch.typename:<20} {branch.title}")
                        
                        # Show a sample of data for first few branches
                        print("\nSample data (first 3 entries):")
                        sample_branches = list(tree.keys())[:5]  # First 5 branches
                        sample_data = tree.arrays(sample_branches, library="np", entry_start=0, entry_stop=3)
                        for branch in sample_branches:
                            print(f"  {branch}: {sample_data[branch]}")
            break  # Only inspect first file

    def __getitem__(self, idx):
        inputs = {}
        targets = {}

        # Load the event
        hits, particles = self.load_event(idx)
        num_particles = len(particles)

        # Build the input hits
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((len(hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field].values).unsqueeze(0).half()

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_particles] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)
        message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
        assert num_particles <= self.event_max_num_particles, message

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"].values)

        # Fill in empty slots with -1s and get the IDs of the particle on each hit
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))])
        hit_particle_ids = torch.from_numpy(hits["particle_id"].values)

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # Create the hit filter targets
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"].to_numpy()).unsqueeze(0)

        # Add sample ID
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)

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

        return hits, particles
    
    # TODO: 
    #    - mask for matching of hits to particles 
    #    - mask for valid tracks 
    #    - cut on number of hits per particle, eta possibly and pt
    #    - enable to load only a subset of events in a ddp friendly way





class MuonTrackingTripplets(MuonTracking):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        particle_min_pt: float = 5.0,
        particle_max_abs_eta: float = 2.7,
        particle_min_num_hits=3,
        event_max_num_particles=1000,
        hit_eval_path: str | None = None,
    ):
        super().__init__(dirpath, inputs, 
                         targets, num_events, 
                         hit_volume_ids, 
                         particle_min_pt, 
                         particle_max_abs_eta, 
                         particle_min_num_hits, 
                         event_max_num_particles, 
                         hit_eval_path)
        
    