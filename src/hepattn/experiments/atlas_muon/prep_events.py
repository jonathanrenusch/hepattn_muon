import os
import glob
import pandas as pd
import uproot
import numpy as np
from typing import List, Dict, Any  
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm



def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

class rootFilter:
    def __init__(self, input_dir: str, output_dir: str, num_events_per_file: int,
                 pt_threshold: float, eta_threshold: float, num_hits_threshold: int
                 ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_events_per_file = num_events_per_file
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold
        self.hits_chunk= []
        self.tracks_chunk = []
        self.saved_event_count = 0
        self.last_saved_event_count = 0
        self.excluded_tracks_count = 0
        self.excluded_events_count = 0
        self.files = self._get_files()
        self.hit_features = ["spacePoint_PositionX", 
                         "spacePoint_PositionY", 
                         "spacePoint_PositionZ",
                         "spacePoint_covXX",
                         "spacePoint_covXY",
                         "spacePoint_covYX",
                         "spacePoint_covYY",
                         "spacePoint_channel",
                         "spacePoint_driftR",
                         "spacePoint_layer",
                         "spacePoint_stationPhi",
                         "spacePoint_stationEta",
                         "spacePoint_technology"
                        ]

        self.track_features = ["truthMuon_pt", 
                           "truthMuon_eta", 
                           "truthMuon_phi", 
                           "truthMuon_q",
                          ]

    def _get_files(self) -> list[Path]:
        dirpath = Path(self.input_dir)
        files = list(dirpath.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No ROOT files found in {dirpath}")
        # Filter out invalid files (empty or corrupted)
        valid_files = [f for f in files if is_valid_file(f)]
        if not valid_files:
            raise FileNotFoundError(f"No valid ROOT files found in {dirpath}")
        return sorted(valid_files)  # Sort for consistent ordering

    def process_root_file(self, 
                          root_file: str) -> List[Dict[str, Any]]:
        
        with uproot.open(root_file) as root_file:
            # Using the first tree
            tree_keys = [key for key in root_file.keys() if ';' in key]
            if tree_keys:
                tree = tree_keys[0].split(';')[0]
                num_events = root_file[tree].num_entries
            else:
                num_events = 0
            for event_idx in range(num_events):
                truth_link = root_file[tree]['spacePoint_truthLink'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
                truth_pt = root_file[tree]['truthMuon_pt'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
                truth_eta = root_file[tree]['truthMuon_eta'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
                # Running checks on event
                unique_tracks = np.unique(truth_link)
                # Remove -1 (no track) from the unique tracks
                valid_tracks = unique_tracks[unique_tracks != -1]
                
                if len(valid_tracks) == 0:
                    self.excluded_events_count += 1
                    continue # no tracks present in this event
                
                # running checks on the minimum pT, eta, and number of hits
                exclude_tracks = []
                for track_idx in valid_tracks:

                    if truth_pt[track_idx] < self.pt_threshold or abs(truth_eta[track_idx]) > self.eta_threshold:
                        exclude_tracks.append(track_idx)
                        self.excluded_tracks_count += 1
                
                # Remove excluded tracks from valid tracks
                remaining_tracks = np.setdiff1d(valid_tracks, exclude_tracks)
                
                # If all tracks are excluded, skip the event
                if len(remaining_tracks) == 0:
                    self.excluded_events_count += 1
                    continue

                # Building event with remaining tracks
                hit2track_mask = np.isin(truth_link, remaining_tracks)
                modified_truth_link = truth_link.copy()
                modified_truth_link[~hit2track_mask] = -1
                track_mask = np.isin(valid_tracks, remaining_tracks)

                # Collecting hit and track information
                hits = {branch: root_file[tree][branch].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0] for branch in self.hit_features}
                hits["spacePoint_truthLink"] = modified_truth_link
                
                tracks = {branch: root_file[tree][branch].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0][track_mask] for branch in self.track_features}
                self.hits_chunk.append(hits)
                self.tracks_chunk.append(tracks)    

                # Increment saved event count
                self.saved_event_count += 1
                print("Saved event count:", self.saved_event_count, end='\r')  # Print current event count on the same line

                # Checking if chunk is full and saving to parquet
                if len(self.hits_chunk) == self.num_events_per_file and len(self.tracks_chunk) == self.num_events_per_file:
                    # Save chunk to parquet files
                    self._save_chunk_to_parquet()
                    # Clear the chunks for the next set of events
                    self.hits_chunk.clear()
                    self.tracks_chunk.clear()
                    self.last_saved_event_count = self.saved_event_count

        
        
        

        return self.hits_chunk, self.tracks_chunk

    def _save_chunk_to_parquet(self) -> None:
        """Save the current chunk of hits and tracks to parquet files with row groups."""
        # Prepare file paths
        hits_file = os.path.join(self.output_dir, f'hits{self.last_saved_event_count}to{self.saved_event_count}.parquet')
        tracks_file = os.path.join(self.output_dir, f'tracks{self.last_saved_event_count}to{self.saved_event_count}.parquet')

        # Process hits data
        hits_tables = []
        for hits_dict in self.hits_chunk:
            hits_df = pd.DataFrame(hits_dict)
            hits_tables.append(pa.Table.from_pandas(hits_df))
        
        # Process tracks data  
        tracks_tables = []
        for tracks_dict in self.tracks_chunk:
            tracks_df = pd.DataFrame(tracks_dict)
            tracks_tables.append(pa.Table.from_pandas(tracks_df))
        
        # Write hits parquet with row groups
        if hits_tables:
            combined_hits_table = pa.concat_tables(hits_tables)
            pq.write_table(combined_hits_table, hits_file, row_group_size=len(hits_tables[0]))
        
        # Write tracks parquet with row groups
        if tracks_tables:
            combined_tracks_table = pa.concat_tables(tracks_tables)
            pq.write_table(combined_tracks_table, tracks_file, row_group_size=len(tracks_tables[0]))

    def process_events(self):
        """Process all root files and yield events."""
        for root_file in tqdm(self.files, desc="Processing ROOT files"):
            print(f"Processing file: {root_file}")
            # try:
            self.process_root_file(root_file)
            # except Exception as e:
                # print(f"Error processing file {root_file}: {e}")
        # Save any remaining data that didn't fill a complete chunk
        # Save any remaining data that didn't fill a complete chunk
        if len(self.hits_chunk) > 0 and len(self.tracks_chunk) > 0:
            self._save_chunk_to_parquet()
            self.hits_chunk.clear()
            self.tracks_chunk.clear()
            self.last_saved_event_count = self.saved_event_count
            
        # Print summary of excluded tracks and events
        print(f"\nTotal excluded tracks: {self.excluded_tracks_count}")
        # Make line new line for better readability
        print(f"Total excluded events: {self.excluded_events_count}, out of {self.saved_event_count + self.excluded_events_count} total events processed. \n"
              f"That is {self.excluded_events_count / (self.saved_event_count + self.excluded_events_count) * 100:.2f}% of all events.")

def main():    
    parser = argparse.ArgumentParser(description="Prefilter ATLAS muon events into batched parquet files.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input root files with ATLAS muon data."
    )
    parser.add_argument(
        "-o",   
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output parquet files with split tracks."
    )
    parser.add_argument(
        "-n",
        "--num_events_per_file",
        type=int,
        default=10,
        help="Number of events to save in each parquet file as different row groups."
    )
    parser.add_argument(
        "-pt",
        "--pt_threshold",
        type=float,
        default=5.0,
        help="Minimum pT threshold for tracks to be included."
    )
    parser.add_argument(
        "-eta",
        "--eta_threshold",
        type=float,
        default=2.7,
        help="Maximum |eta| <= 2.7 threshold for tracks to be included."
    )
    parser.add_argument(
        "-nh",
        "--num_hits_threshold",
        type=int,
        default=3,
        help="Minimum number of hits required for tracks to be included."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    filter = rootFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_events_per_file=args.num_events_per_file,
        pt_threshold=args.pt_threshold,
        eta_threshold=args.eta_threshold,
        num_hits_threshold=args.num_hits_threshold
    )
    
    filter.process_events()
if __name__ == "__main__":
    main()