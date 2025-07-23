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
import yaml
import h5py



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
        self.valid_event_count = 0
        self.last_saved_event_count = 0
        self.excluded_tracks_count = 0
        self.excluded_events_count = 0
        self.valid_events_count = 0
        self.valid_tracks_count = 0
        self.event_mapping = []  # List to track which events are in which files
        # Efficient index tracking for large datasets
        self.global_event_ids = []
        self.file_indices = []
        self.row_indices = []
        self.num_hits_per_event = []
        self.num_tracks_per_event = []
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
            
            # Define chunk size for efficient loading
            chunk_size = 1000  # Adjust based on your memory constraints
            
            # Process events in chunks
            for chunk_start in range(0, num_events, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_events)

                # Load all required arrays for this chunk at once
                truth_link_chunk = root_file[tree]['spacePoint_truthLink'].array(
                    entry_start=chunk_start, entry_stop=chunk_end, library='np'
                )
                truth_pt_chunk = root_file[tree]['truthMuon_pt'].array(
                    entry_start=chunk_start, entry_stop=chunk_end, library='np'
                )
                truth_eta_chunk = root_file[tree]['truthMuon_eta'].array(
                    entry_start=chunk_start, entry_stop=chunk_end, library='np'
                )
                
                # Load all hit features for this chunk
                hit_features_chunk = {}
                for feature in self.hit_features:
                    hit_features_chunk[feature] = root_file[tree][feature].array(
                        entry_start=chunk_start, entry_stop=chunk_end, library='np'
                    )
                
                # Load all track features for this chunk
                track_features_chunk = {}
                for feature in self.track_features:
                    track_features_chunk[feature] = root_file[tree][feature].array(
                        entry_start=chunk_start, entry_stop=chunk_end, library='np'
                    )
                
                # Process each event in the chunk
                for event_idx_in_chunk in range(len(truth_link_chunk)):
                    truth_link = truth_link_chunk[event_idx_in_chunk]
                    truth_pt = truth_pt_chunk[event_idx_in_chunk]
                    truth_eta = truth_eta_chunk[event_idx_in_chunk]
                    
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
                    # save number of valid tracks
                    self.valid_tracks_count += len(remaining_tracks)
                    # Building event with remaining tracks
                    hit2track_mask = np.isin(truth_link, remaining_tracks)
                    modified_truth_link = truth_link.copy()
                    modified_truth_link[~hit2track_mask] = -1
                    track_mask = np.isin(valid_tracks, remaining_tracks)

                    # Collecting hit and track information from pre-loaded chunk data
                    hits = {branch: hit_features_chunk[branch][event_idx_in_chunk] for branch in self.hit_features}
                    hits["spacePoint_truthLink"] = modified_truth_link
                    
                    tracks = {branch: track_features_chunk[branch][event_idx_in_chunk][track_mask] for branch in self.track_features}
                    self.hits_chunk.append(hits)
                    self.tracks_chunk.append(tracks)    

                    # Increment saved event count
                    self.valid_event_count += 1
                    print("Number of valid events:", self.valid_event_count, end='\r')  # Print current event count on the same line

                    # Checking if chunk is full and saving to parquet
                    if len(self.hits_chunk) == self.num_events_per_file and len(self.tracks_chunk) == self.num_events_per_file:
                        # Save chunk to parquet files
                        self._save_chunk_to_hdf5()
                        # self._save_chunk_to_parquet()
                        # Clear the chunks for the next set of events
                        self.hits_chunk.clear()
                        self.tracks_chunk.clear()
                        self.last_saved_event_count = self.valid_event_count

        return self.hits_chunk, self.tracks_chunk
    
    # def process_root_file(self, 
    #                       root_file: str) -> List[Dict[str, Any]]:
        
    #     with uproot.open(root_file) as root_file:
    #         # Using the first tree
    #         tree_keys = [key for key in root_file.keys() if ';' in key]
    #         if tree_keys:
    #             tree = tree_keys[0].split(';')[0]
    #             num_events = root_file[tree].num_entries
    #         else:
    #             num_events = 0
    #         for event_idx in range(num_events):
    #             truth_link = root_file[tree]['spacePoint_truthLink'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
    #             truth_pt = root_file[tree]['truthMuon_pt'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
    #             truth_eta = root_file[tree]['truthMuon_eta'].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0]
    #             # Running checks on event
    #             unique_tracks = np.unique(truth_link)
    #             # Remove -1 (no track) from the unique tracks
    #             valid_tracks = unique_tracks[unique_tracks != -1]
                
    #             if len(valid_tracks) == 0:
    #                 self.excluded_events_count += 1
    #                 continue # no tracks present in this event
                
    #             # running checks on the minimum pT, eta, and number of hits
    #             exclude_tracks = []
    #             for track_idx in valid_tracks:

    #                 if truth_pt[track_idx] < self.pt_threshold or abs(truth_eta[track_idx]) > self.eta_threshold:
    #                     exclude_tracks.append(track_idx)
    #                     self.excluded_tracks_count += 1
                
    #             # Remove excluded tracks from valid tracks
    #             remaining_tracks = np.setdiff1d(valid_tracks, exclude_tracks)
                
    #             # If all tracks are excluded, skip the event
    #             if len(remaining_tracks) == 0:
    #                 self.excluded_events_count += 1
    #                 continue
    #             # save number of valid tracks
    #             self.valid_tracks_count += len(remaining_tracks)
    #             # Building event with remaining tracks
    #             hit2track_mask = np.isin(truth_link, remaining_tracks)
    #             modified_truth_link = truth_link.copy()
    #             modified_truth_link[~hit2track_mask] = -1
    #             track_mask = np.isin(valid_tracks, remaining_tracks)

    #             # Collecting hit and track information
    #             hits = {branch: root_file[tree][branch].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0] for branch in self.hit_features}
    #             hits["spacePoint_truthLink"] = modified_truth_link
                
    #             tracks = {branch: root_file[tree][branch].array(entry_start=event_idx, entry_stop=event_idx+1, library='np')[0][track_mask] for branch in self.track_features}
    #             self.hits_chunk.append(hits)
    #             self.tracks_chunk.append(tracks)    

    #             # Increment saved event count
    #             self.valid_event_count += 1
    #             print("Saved event count:", self.valid_event_count, end='\r')  # Print current event count on the same line

    #             # Checking if chunk is full and saving to parquet
    #             if len(self.hits_chunk) == self.num_events_per_file and len(self.tracks_chunk) == self.num_events_per_file:
    #                 # Save chunk to parquet files
    #                 self._save_chunk_to_hdf5()
    #                 # self._save_chunk_to_parquet()
    #                 # Clear the chunks for the next set of events
    #                 self.hits_chunk.clear()
    #                 self.tracks_chunk.clear()
    #                 self.last_saved_event_count = self.valid_event_count

        
        
        

    #     return self.hits_chunk, self.tracks_chunk



    def _save_chunk_to_hdf5(self) -> None:
        """Save the current chunk of hits and tracks to HDF5 files with each event as a separate group."""
        print(f"Saving chunk: {self.last_saved_event_count} to {self.valid_event_count}", end='\r')

        # Create data subdirectory
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Prepare file path in data subdirectory
        h5_file = os.path.join(data_dir, f'events_{self.last_saved_event_count}to{self.valid_event_count}.h5')
        
        # Record chunk info for lightweight metadata (relative paths)
        chunk_info = {
            'h5_file': f'data/events_{self.last_saved_event_count}to{self.valid_event_count}.h5',
            'event_range': {
                'start': self.last_saved_event_count,
                'end': self.valid_event_count,
                'count': len(self.hits_chunk)
            }
        }
        
        current_chunk_idx = len(self.event_mapping)
        
        # Add efficient index tracking
        for i in range(len(self.hits_chunk)):
            event_idx = self.last_saved_event_count + i
            self.global_event_ids.append(event_idx)
            self.file_indices.append(current_chunk_idx)
            self.row_indices.append(i)
            self.num_hits_per_event.append(len(self.hits_chunk[i]['spacePoint_PositionX']))
            self.num_tracks_per_event.append(len(self.tracks_chunk[i]['truthMuon_pt']))
        
        self.event_mapping.append(chunk_info)

        # Save to HDF5 with each event as a separate group
        with h5py.File(h5_file, 'w') as f:
            # Set file-level attributes
            f.attrs['chunk_start_event'] = self.last_saved_event_count
            f.attrs['chunk_end_event'] = self.valid_event_count
            f.attrs['chunk_event_count'] = len(self.hits_chunk)
            
            # Store feature names as attributes for reference
            f.attrs['hit_feature_names'] = list(self.hit_features) + ['spacePoint_truthLink']
            f.attrs['track_feature_names'] = list(self.track_features)
            
            # Save each event as a separate group
            for event_idx in range(len(self.hits_chunk)):
                event_group = f.create_group(f'event_{event_idx}')
                
                # Store event metadata
                global_event_id = self.last_saved_event_count + event_idx
                event_group.attrs['global_event_id'] = global_event_id
                event_group.attrs['num_hits'] = len(self.hits_chunk[event_idx]['spacePoint_PositionX'])
                event_group.attrs['num_tracks'] = len(self.tracks_chunk[event_idx]['truthMuon_pt'])
                
                # Store hits as 2D array: [num_hits, num_features]
                hits_dict = self.hits_chunk[event_idx]
                hit_feature_names = list(self.hit_features) + ['spacePoint_truthLink']
                
                hits_matrix = np.column_stack([
                    np.array(hits_dict[feature], dtype=np.float32) 
                    for feature in hit_feature_names
                ])
                
                event_group.create_dataset(
                    'hits',
                    data=hits_matrix,
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True,
                    fletcher32=True
                )
                
                # Store tracks as 2D array: [num_tracks, num_features]
                tracks_dict = self.tracks_chunk[event_idx]
                
                if len(tracks_dict['truthMuon_pt']) > 0:
                    tracks_matrix = np.column_stack([
                        np.array(tracks_dict[feature], dtype=np.float32)
                        for feature in self.track_features
                    ])
                    
                    event_group.create_dataset(
                        'tracks',
                        data=tracks_matrix,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True,
                        fletcher32=True
                    )
                else:
                    # Empty array for no tracks
                    event_group.create_dataset(
                        'tracks', 
                        data=np.empty((0, len(self.track_features)), dtype=np.float32)
                    )
                    
    # def _save_chunk_to_parquet(self) -> None:
    #     """Save the current chunk of hits and tracks to parquet files with one row group per file."""
    #     print("Saving chunk...", end='\r')
        
    #     # Create data subdirectory
    #     data_dir = os.path.join(self.output_dir, 'data')
    #     os.makedirs(data_dir, exist_ok=True)
        
    #     # Prepare file paths in data subdirectory
    #     hits_file = os.path.join(data_dir, f'hits{self.last_saved_event_count}to{self.valid_event_count}.parquet')
    #     tracks_file = os.path.join(data_dir, f'tracks{self.last_saved_event_count}to{self.valid_event_count}.parquet')
        
    #     # Record chunk info for lightweight metadata (relative paths)
    #     chunk_info = {
    #         'hits_file': f'data/hits{self.last_saved_event_count}to{self.valid_event_count}.parquet',
    #         'tracks_file': f'data/tracks{self.last_saved_event_count}to{self.valid_event_count}.parquet',
    #         'event_range': {
    #             'start': self.last_saved_event_count,
    #             'end': self.valid_event_count,
    #             'count': len(self.hits_chunk)
    #         }
    #     }
        
    #     current_chunk_idx = len(self.event_mapping)
        
    #     # Add efficient index tracking
    #     for i in range(len(self.hits_chunk)):
    #         event_idx = self.last_saved_event_count + i
    #         self.global_event_ids.append(event_idx)
    #         self.file_indices.append(current_chunk_idx)
    #         self.row_indices.append(i)
    #         self.num_hits_per_event.append(len(self.hits_chunk[i]['spacePoint_PositionX']))
    #         self.num_tracks_per_event.append(len(self.tracks_chunk[i]['truthMuon_pt']))
        
    #     self.event_mapping.append(chunk_info)

    #     # Process hits data - each event becomes a row with list of structs
    #     hits_rows = []
    #     for hits_dict in self.hits_chunk:
    #         # Convert arrays to lists and create a single row with lists
    #         hits_row = {}
    #         for key, value in hits_dict.items():
    #             hits_row[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
    #         hits_rows.append(hits_row)
        
    #     # Process tracks data - each event becomes a row with list of structs
    #     tracks_rows = []
    #     for tracks_dict in self.tracks_chunk:
    #         # Convert arrays to lists and create a single row with lists
    #         tracks_row = {}
    #         for key, value in tracks_dict.items():
    #             tracks_row[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
    #         tracks_rows.append(tracks_row)
        
    #     # Write hits parquet with one row group
    #     if hits_rows:
    #         hits_df = pd.DataFrame(hits_rows)
    #         hits_table = pa.Table.from_pandas(hits_df)
    #         pq.write_table(hits_table, hits_file, compression='snappy')
        
    #     # Write tracks parquet with one row group
    #     if tracks_rows:
    #         tracks_df = pd.DataFrame(tracks_rows)
    #         tracks_table = pa.Table.from_pandas(tracks_df)
    #         pq.write_table(tracks_table, tracks_file, compression='snappy')

    def _save_index_arrays(self) -> None:
        """Save event indices as compact numpy arrays for efficient dataset loading."""
        # Save metadata files at the top level (common practice)
        np.save(os.path.join(self.output_dir, 'event_global_ids.npy'), 
                np.array(self.global_event_ids, dtype=np.int32))
        np.save(os.path.join(self.output_dir, 'event_file_indices.npy'), 
                np.array(self.file_indices, dtype=np.int16))
        np.save(os.path.join(self.output_dir, 'event_row_indices.npy'),
                np.array(self.row_indices, dtype=np.int16))
        np.save(os.path.join(self.output_dir, 'event_num_hits.npy'), 
                np.array(self.num_hits_per_event, dtype=np.int16))
        np.save(os.path.join(self.output_dir, 'event_num_tracks.npy'), 
                np.array(self.num_tracks_per_event, dtype=np.int8))
        
        # Save chunk info separately (lightweight) - updated for HDF5
        chunk_info = []
        for chunk in self.event_mapping:
            chunk_info.append({
                'h5_file': chunk['h5_file'],
                'start_event': chunk['event_range']['start'],
                'end_event': chunk['event_range']['end'],
                'count': chunk['event_range']['count']
            })
        
        np.save(os.path.join(self.output_dir, 'chunk_info.npy'), chunk_info)

    # def _save_index_arrays(self) -> None:
    #     """Save event indices as compact numpy arrays for efficient dataset loading."""
    #     # Save metadata files at the top level (common practice)
    #     np.save(os.path.join(self.output_dir, 'event_global_ids.npy'), 
    #             np.array(self.global_event_ids, dtype=np.int32))
    #     np.save(os.path.join(self.output_dir, 'event_file_indices.npy'), 
    #             np.array(self.file_indices, dtype=np.int16))
    #     np.save(os.path.join(self.output_dir, 'event_row_indices.npy'),
    #             np.array(self.row_indices, dtype=np.int16))
    #     np.save(os.path.join(self.output_dir, 'event_num_hits.npy'), 
    #             np.array(self.num_hits_per_event, dtype=np.int16))
    #     np.save(os.path.join(self.output_dir, 'event_num_tracks.npy'), 
    #             np.array(self.num_tracks_per_event, dtype=np.int8))
        
    #     # Save chunk info separately (lightweight)
        #     chunk_info = []
        #     for chunk in self.event_mapping:
        #         chunk_info.append({
        #             'hits_file': chunk['hits_file'],
        #             'tracks_file': chunk['tracks_file'],
        #             'start_event': chunk['event_range']['start'],
        #             'end_event': chunk['event_range']['end'],
        #             'count': chunk['event_range']['count']
        #         })
            
        #     np.save(os.path.join(self.output_dir, 'chunk_info.npy'), chunk_info)

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
            self._save_chunk_to_hdf5()
            # self._save_chunk_to_parquet()
            self.hits_chunk.clear()
            self.tracks_chunk.clear()
            self.last_saved_event_count = self.valid_event_count
            
        # Save summary information to YAML file at top level
        dataset_info_file = os.path.join(self.output_dir, 'metadata.yaml')
        
        # Calculate percentages
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_event_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_event_count) if self.valid_event_count > 0 else 0
        
        # Create structured data for YAML
        dataset_info = {
            'processing_summary': {
                'total_excluded_tracks': self.excluded_tracks_count,
                'total_tracks_processed': total_tracks,
                'excluded_tracks_percentage': excluded_tracks_percent,
                'total_excluded_events': self.excluded_events_count,
                'total_events_processed': total_events,
                'excluded_events_percentage': excluded_events_percent,
                'valid_events': self.valid_event_count,
                'valid_tracks': self.valid_tracks_count,
                'average_tracks_per_event': avg_tracks_per_event,
                'processing_status': 'Complete'
            },
            'processing_parameters': {
                'pt_threshold': self.pt_threshold,
                'eta_threshold': self.eta_threshold,
                'num_hits_threshold': self.num_hits_threshold,
                'num_events_per_file': self.num_events_per_file
            },
            'processed_files': [str(file_path) for file_path in self.files],
            'event_mapping': {
                'description': 'Event indices stored in separate numpy files for efficient access',
                'total_events': self.valid_event_count,
                'total_chunks': len(self.event_mapping),
                'index_files': {
                    'global_event_ids': 'event_global_ids.npy',
                    'file_indices': 'event_file_indices.npy',
                    'row_indices': 'event_row_indices.npy',
                    'num_hits': 'event_num_hits.npy',
                    'num_tracks': 'event_num_tracks.npy',
                    'chunk_info': 'chunk_info.npy'
                },
                'chunk_summary': [
                    {
                        'h5_file': chunk['h5_file'],
                        'event_count': chunk['event_range']['count'],
                        'start_event': chunk['event_range']['start'],
                        'end_event': chunk['event_range']['end']
                    } for chunk in self.event_mapping
                ]
            }
        }
        
        # Save efficient index arrays
        self._save_index_arrays()
        
        # Save to YAML
        with open(dataset_info_file, 'w') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
        
        # Still print summary for immediate feedback
        print(f"\nTotal excluded tracks: {self.excluded_tracks_count}, out of {total_tracks}")
        print(f"That is {excluded_tracks_percent:.2f}% of all tracks.")
        print(f"Total excluded events: {self.excluded_events_count}, out of {total_events} total events processed.")
        print(f"That is {excluded_events_percent:.2f}% of all events.")
        print("Valid events:", self.valid_event_count)
        print("Valid tracks:", self.valid_tracks_count)
        print("Average tracks per event:", f"{avg_tracks_per_event:.2f}")
        print("Processing complete.")
        print(f"Dataset information saved to: {dataset_info_file}")

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
        default=1000,
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