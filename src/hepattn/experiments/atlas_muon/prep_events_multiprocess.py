import os
import glob
import pandas as pd
import uproot
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml
import h5py
import multiprocessing as mp
from functools import partial
import time

def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

class ParallelRootFilter:
    def __init__(self, input_dir: str, output_dir: str, num_events_per_file: int,
                 pt_threshold: float, eta_threshold: float, num_hits_threshold: int, 
                 max_events: int = -1, num_workers: int = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_events_per_file = num_events_per_file
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold
        self.max_events = max_events
        self.num_workers = num_workers or mp.cpu_count()
        
        # Global counters (will be aggregated from workers)
        self.excluded_tracks_count = 0
        self.excluded_events_count = 0
        self.valid_events_count = 0
        self.valid_tracks_count = 0
        self.event_mapping = []
        self.file_indices = []
        self.row_indices = []
        self.num_hits_per_event = []
        self.num_tracks_per_event = []
        
        self.files = self._get_files()
        
        self.hit_features = [
            "spacePoint_globEdgeHighX", "spacePoint_globEdgeHighY", "spacePoint_globEdgeHighZ",
            "spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ",
            "spacePoint_time", "spacePoint_driftR", "spacePoint_readOutSide",
            "spacePoint_covXX", "spacePoint_covXY", "spacePoint_covYX", "spacePoint_covYY",
            "spacePoint_channel", "spacePoint_layer", "spacePoint_stationPhi",
            "spacePoint_stationEta", "spacePoint_stationIndex", "spacePoint_technology",
            "spacePoint_truthLink"
        ]
        
        self.track_features = ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]

    def _get_files(self) -> list[Path]:
        dirpath = Path(self.input_dir)
        files = list(dirpath.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No ROOT files found in {dirpath}")
        
        valid_files = [f for f in files if is_valid_file(f)]
        if not valid_files:
            raise FileNotFoundError(f"No valid ROOT files found in {dirpath}")
        return sorted(valid_files)

    def _split_files(self) -> List[List[Path]]:
        """Split files into chunks for parallel processing"""
        files_per_worker = len(self.files) // self.num_workers
        remainder = len(self.files) % self.num_workers
        
        file_chunks = []
        start_idx = 0
        
        for i in range(self.num_workers):
            # Distribute remainder files among first workers
            chunk_size = files_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx < len(self.files):
                file_chunks.append(self.files[start_idx:end_idx])
            else:
                file_chunks.append([])  # Empty chunk for excess workers
            
            start_idx = end_idx
        
        return file_chunks

    def _save_final_metadata(self, processing_time: float):
        """Save aggregated metadata and index arrays"""
        print("Saving final metadata...")
        
        # Save index arrays
        np.save(os.path.join(self.output_dir, 'event_file_indices.npy'), 
                np.array(self.file_indices, dtype=np.int16))
        np.save(os.path.join(self.output_dir, 'event_row_indices.npy'),
                np.array(self.row_indices, dtype=np.int16))
        
        # Calculate summary statistics
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_events_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_events_count) if self.valid_events_count > 0 else 0
        
        # Count number of workers that actually processed files
        num_active_workers = len(self.event_mapping)
        
        # Create metadata
        dataset_info = {
            'hit_features': self.hit_features,
            'track_features': self.track_features,
            'processing_summary': {
                'total_excluded_tracks': self.excluded_tracks_count,
                'total_tracks_processed': total_tracks,
                'excluded_tracks_percentage': excluded_tracks_percent,
                'total_excluded_events': self.excluded_events_count,
                'total_events_processed': total_events,
                'excluded_events_percentage': excluded_events_percent,
                'valid_events': self.valid_events_count,
                'valid_tracks': self.valid_tracks_count,
                'average_tracks_per_event': avg_tracks_per_event,
                'processing_time_seconds': processing_time,
                'num_workers': num_active_workers,
                'processing_status': 'Complete'
            },
            'processing_parameters': {
                'pt_threshold': self.pt_threshold,
                'eta_threshold': self.eta_threshold,
                'num_hits_threshold': self.num_hits_threshold,
                'num_events_per_file': self.num_events_per_file,
                'max_events': self.max_events
            },
            'processed_files': [str(file_path) for file_path in self.files],
            'event_mapping': {
                'description': 'Event indices stored in separate numpy files for efficient access',
                'total_events': self.valid_events_count,
                'total_chunks': len(self.event_mapping),
                'index_files': {
                    'file_indices': 'event_file_indices.npy',
                    'row_indices': 'event_row_indices.npy',
                },
                'chunk_summary': [
                    {
                        'h5_file': chunk['h5_file'],
                        'source_root_file': chunk['source_root_file'],
                        'event_count': chunk['event_range']['count'],
                        'worker_id': chunk['worker_id']
                    } for chunk in self.event_mapping
                ]
            }
        }
        
        # Save metadata
        dataset_info_file = os.path.join(self.output_dir, 'metadata.yaml')
        with open(dataset_info_file, 'w') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Workers used: {num_active_workers}")
        print(f"Total excluded tracks: {self.excluded_tracks_count:,} out of {total_tracks:,} ({excluded_tracks_percent:.2f}%)")
        print(f"Total excluded events: {self.excluded_events_count:,} out of {total_events:,} ({excluded_events_percent:.2f}%)")
        print(f"Valid events: {self.valid_events_count:,}")
        print(f"Valid tracks: {self.valid_tracks_count:,}")
        print(f"Average tracks per event: {avg_tracks_per_event:.2f}")
        print(f"Total chunks created: {len(self.event_mapping)}")
        print(f"Dataset metadata saved to: {dataset_info_file}")
        print(f"{'='*60}")

    def process_events(self):
        """Main method to process events in parallel"""
        print(f"Starting parallel processing with {self.num_workers} workers...")
        print(f"Total files to process: {len(self.files)}")
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
        # Split files among workers
        file_chunks = self._split_files()
        
        # Print worker assignment
        for i, chunk in enumerate(file_chunks):
            print(f"Worker {i}: {len(chunk)} files")
        
        # Create worker arguments
        worker_args = []
        for worker_id, file_chunk in enumerate(file_chunks):
            if file_chunk:  # Only create args for workers with files
                args = (
                    worker_id, file_chunk, self.output_dir, self.num_events_per_file,
                    self.pt_threshold, self.eta_threshold, self.num_hits_threshold,
                    self.max_events, self.hit_features, self.track_features
                )
                worker_args.append(args)
        
        # Process in parallel
        start_time = time.time()
        with mp.Pool(len(worker_args)) as pool:
            results = pool.map(process_worker_files, worker_args)
        
        processing_time = time.time() - start_time
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        
        # Aggregate results from all workers
        self._aggregate_results(results)
        
        # Save final metadata
        self._save_final_metadata(processing_time)

    def _aggregate_results(self, results: List[Dict]):
        """Aggregate results from all worker processes"""
        print("Aggregating results from workers...")
        
        total_chunk_offset = 0
        
        for worker_result in results:
            if worker_result is None:
                continue
                
            # Aggregate counters
            self.excluded_tracks_count += worker_result['excluded_tracks_count']
            self.excluded_events_count += worker_result['excluded_events_count']
            self.valid_events_count += worker_result['valid_events_count']
            self.valid_tracks_count += worker_result['valid_tracks_count']
            
            # Aggregate event mapping
            self.event_mapping.extend(worker_result['event_mapping'])
            
            # Aggregate indices with proper offsets
            worker_file_indices = np.array(worker_result['file_indices']) + total_chunk_offset
            self.file_indices.extend(worker_file_indices.tolist())
            self.row_indices.extend(worker_result['row_indices'])
            self.num_hits_per_event.extend(worker_result['num_hits_per_event'])
            self.num_tracks_per_event.extend(worker_result['num_tracks_per_event'])
            
            total_chunk_offset += len(worker_result['event_mapping'])

    def _save_final_metadata(self, processing_time: float):
        """Save aggregated metadata and index arrays"""
        print("Saving final metadata...")
        
        # Save index arrays
        np.save(os.path.join(self.output_dir, 'event_file_indices.npy'), 
                np.array(self.file_indices, dtype=np.int16))
        np.save(os.path.join(self.output_dir, 'event_row_indices.npy'),
                np.array(self.row_indices, dtype=np.int16))
        
        # Calculate summary statistics
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_events_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_events_count) if self.valid_events_count > 0 else 0
        
        # Count number of workers that actually processed files
        num_active_workers = len(self.event_mapping)
        
        # Create metadata
        dataset_info = {
            'hit_features': self.hit_features,
            'track_features': self.track_features,
            'processing_summary': {
                'total_excluded_tracks': self.excluded_tracks_count,
                'total_tracks_processed': total_tracks,
                'excluded_tracks_percentage': excluded_tracks_percent,
                'total_excluded_events': self.excluded_events_count,
                'total_events_processed': total_events,
                'excluded_events_percentage': excluded_events_percent,
                'valid_events': self.valid_events_count,
                'valid_tracks': self.valid_tracks_count,
                'average_tracks_per_event': avg_tracks_per_event,
                'processing_time_seconds': processing_time,
                'num_workers': num_active_workers,
                'processing_status': 'Complete'
            },
            'processing_parameters': {
                'pt_threshold': self.pt_threshold,
                'eta_threshold': self.eta_threshold,
                'num_hits_threshold': self.num_hits_threshold,
                'num_events_per_file': self.num_events_per_file,
                'max_events': self.max_events
            },
            'processed_files': [str(file_path) for file_path in self.files],
            'event_mapping': {
                'description': 'Event indices stored in separate numpy files for efficient access',
                'total_events': self.valid_events_count,
                'total_chunks': len(self.event_mapping),
                'index_files': {
                    'file_indices': 'event_file_indices.npy',
                    'row_indices': 'event_row_indices.npy',
                },
                'chunk_summary': [
                    {
                        'h5_file': chunk['h5_file'],
                        'source_root_file': chunk['source_root_file'],
                        'event_count': chunk['event_range']['count'],
                        'worker_id': chunk['worker_id']
                    } for chunk in self.event_mapping
                ]
            }
        }
        
        # Save metadata
        dataset_info_file = os.path.join(self.output_dir, 'metadata.yaml')
        with open(dataset_info_file, 'w') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Workers used: {num_active_workers}")
        print(f"Total excluded tracks: {self.excluded_tracks_count:,} out of {total_tracks:,} ({excluded_tracks_percent:.2f}%)")
        print(f"Total excluded events: {self.excluded_events_count:,} out of {total_events:,} ({excluded_events_percent:.2f}%)")
        print(f"Valid events: {self.valid_events_count:,}")
        print(f"Valid tracks: {self.valid_tracks_count:,}")
        print(f"Average tracks per event: {avg_tracks_per_event:.2f}")
        print(f"Total chunks created: {len(self.event_mapping)}")
        print(f"Dataset metadata saved to: {dataset_info_file}")
        print(f"{'='*60}")

def process_worker_files(args: Tuple) -> Dict:
    """Worker function to process a subset of files"""
    (worker_id, file_chunk, output_dir, num_events_per_file, pt_threshold, 
     eta_threshold, num_hits_threshold, max_events, hit_features, track_features) = args
    
    if not file_chunk:
        return None
    
    print(f"Worker {worker_id}: Starting processing of {len(file_chunk)} files")
    
    # Initialize worker-specific counters and data structures
    excluded_tracks_count = 0
    excluded_events_count = 0
    valid_tracks_count = 0
    event_mapping = []
    file_indices = []
    row_indices = []
    num_hits_per_event = []
    num_tracks_per_event = []
    total_valid_events = 0
    
    for file_idx, root_file in enumerate(file_chunk):
        print(f"Worker {worker_id}: Processing file {file_idx+1}/{len(file_chunk)}: {root_file.name}")
        
        # Initialize per-file variables
        hits_chunk = []
        tracks_chunk = []
        event_numbers_chunk = []
        
        try:
            with uproot.open(root_file) as rf:
                tree_keys = [key for key in rf.keys() if ';' in key]
                if not tree_keys:
                    print(f"Worker {worker_id}: No tree found in {root_file.name}")
                    continue
                    
                tree = tree_keys[0].split(';')[0]
                num_events = rf[tree].num_entries
                chunk_size = 1000
                
                for chunk_start in range(0, num_events, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_events)
                    
                    # Load chunk data
                    hit_features_chunk = {}
                    for feature in hit_features:
                        hit_features_chunk[feature] = rf[tree][feature].array(
                            entry_start=chunk_start, entry_stop=chunk_end, library='np'
                        )
                    
                    track_features_chunk = {}
                    for feature in track_features:
                        track_features_chunk[feature] = rf[tree][feature].array(
                            entry_start=chunk_start, entry_stop=chunk_end, library='np'
                        )
                    
                    event_numbers_array = rf[tree]['eventNumber'].array(
                        entry_start=chunk_start, entry_stop=chunk_end, library='np'
                    )
                    
                    # Process each event in chunk
                    for event_idx_in_chunk in range(chunk_end - chunk_start):
                        # Check if we've reached the global limit (approximate)
                        if max_events > 0 and total_valid_events >= max_events:
                            print(f"Worker {worker_id}: Reached approximate max_events limit ({max_events})")
                            break
                        
                        unique_tracks = np.unique(hit_features_chunk['spacePoint_truthLink'][event_idx_in_chunk])
                        valid_tracks = unique_tracks[unique_tracks != -1]
                        
                        if len(valid_tracks) == 0:
                            excluded_events_count += 1
                            continue
                        
                        # Apply track filters
                        exclude_tracks = []
                        for track_idx in valid_tracks:
                            if (track_features_chunk['truthMuon_pt'][event_idx_in_chunk][track_idx] < pt_threshold or
                                abs(track_features_chunk['truthMuon_eta'][event_idx_in_chunk][track_idx]) > eta_threshold or
                                np.sum(hit_features_chunk['spacePoint_truthLink'][event_idx_in_chunk] == track_idx) < num_hits_threshold):
                                exclude_tracks.append(track_idx)
                                excluded_tracks_count += 1
                        
                        remaining_tracks = np.setdiff1d(valid_tracks, exclude_tracks)
                        
                        if len(remaining_tracks) == 0:
                            excluded_events_count += 1
                            continue
                        
                        valid_tracks_count += len(remaining_tracks)
                        
                        # Build event data
                        hit2track_mask = np.isin(hit_features_chunk['spacePoint_truthLink'][event_idx_in_chunk], remaining_tracks)
                        modified_truth_link = hit_features_chunk['spacePoint_truthLink'][event_idx_in_chunk].copy()
                        modified_truth_link[~hit2track_mask] = -1
                        track_mask = np.isin(valid_tracks, remaining_tracks)
                        
                        hits = {branch: hit_features_chunk[branch][event_idx_in_chunk] for branch in hit_features}
                        hits["spacePoint_truthLink"] = modified_truth_link
                        
                        tracks = {branch: track_features_chunk[branch][event_idx_in_chunk][track_mask] for branch in track_features}
                        
                        hits_chunk.append(hits)
                        tracks_chunk.append(tracks)
                        event_numbers_chunk.append(event_numbers_array[event_idx_in_chunk])
                    
                    # Break out of chunk loop if we hit the limit
                    if max_events > 0 and total_valid_events >= max_events:
                        break
                
                # Break out of file loop if we hit the limit  
                if max_events > 0 and total_valid_events >= max_events:
                    break
                    
        except Exception as e:
            print(f"Worker {worker_id}: Error processing file {root_file}: {e}")
            continue
        
        # Save file data if we have any valid events
        if len(hits_chunk) > 0:
            file_valid_events = len(hits_chunk)
            print(f"Worker {worker_id}: Saving {file_valid_events} events from {root_file.name}")
            
            chunk_info = save_worker_chunk_to_hdf5(
                hits_chunk, tracks_chunk, event_numbers_chunk,
                output_dir, worker_id, root_file, file_valid_events,
                hit_features, track_features
            )
            
            # Update tracking data
            current_chunk_idx = len(event_mapping)
            for i in range(len(hits_chunk)):
                file_indices.append(current_chunk_idx)
                row_indices.append(i)
                num_hits_per_event.append(len(hits_chunk[i]['spacePoint_time']))
                num_tracks_per_event.append(len(tracks_chunk[i]['truthMuon_pt']))
            
            event_mapping.append(chunk_info)
            total_valid_events += file_valid_events
        else:
            print(f"Worker {worker_id}: No valid events found in {root_file.name}")
        
        # Stop processing more files if we've reached the limit
        if max_events > 0 and total_valid_events >= max_events:
            print(f"Worker {worker_id}: Reached max_events limit, stopping file processing")
            break
    
    print(f"Worker {worker_id}: Completed processing. Valid events: {total_valid_events}")
    
    return {
        'excluded_tracks_count': excluded_tracks_count,
        'excluded_events_count': excluded_events_count,
        'valid_events_count': total_valid_events,
        'valid_tracks_count': valid_tracks_count,
        'event_mapping': event_mapping,
        'file_indices': file_indices,
        'row_indices': row_indices,
        'num_hits_per_event': num_hits_per_event,
        'num_tracks_per_event': num_tracks_per_event
    }

def save_worker_chunk_to_hdf5(hits_chunk, tracks_chunk, event_numbers_chunk, 
                             output_dir, worker_id, root_file, valid_events_count,
                             hit_features, track_features):
    """Save a chunk of data to HDF5 file"""
    data_dir = os.path.join(output_dir, 'data')
    
    # Extract root file name without extension and add event count
    root_file_stem = Path(root_file).stem
    h5_filename = f'{root_file_stem}_{valid_events_count}events.h5'
    h5_file = os.path.join(data_dir, h5_filename)
    
    chunk_info = {
        'h5_file': f'data/{h5_filename}',
        'source_root_file': str(root_file),
        'worker_id': worker_id,
        'event_range': {
            'count': len(hits_chunk)
        }
    }
    
    with h5py.File(h5_file, 'w') as f:
        num_events = len(hits_chunk)
        max_hits = max(len(hits['spacePoint_time']) for hits in hits_chunk)
        max_tracks = max(len(tracks['truthMuon_pt']) for tracks in tracks_chunk)
        num_hit_features = len(hit_features)
        num_track_features = len(track_features)
        
        # Create and fill hit arrays
        hits_array = np.full((num_events, max_hits, num_hit_features), np.nan, dtype=np.float32)
        for event_idx, hits_dict in enumerate(hits_chunk):
            num_hits = len(hits_dict['spacePoint_time'])
            for feat_idx, feature in enumerate(hit_features):
                hits_array[event_idx, :num_hits, feat_idx] = hits_dict[feature]
        
        # Create and fill track arrays
        tracks_array = np.full((num_events, max_tracks, num_track_features), np.nan, dtype=np.float32)
        for event_idx, tracks_dict in enumerate(tracks_chunk):
            num_tracks = len(tracks_dict['truthMuon_pt'])
            for feat_idx, feature in enumerate(track_features):
                tracks_array[event_idx, :num_tracks, feat_idx] = tracks_dict[feature]
        
        # Save datasets
        f.create_dataset('hits', data=hits_array, compression='gzip', compression_opts=6, shuffle=True, fletcher32=True)
        f.create_dataset('tracks', data=tracks_array, compression='gzip', compression_opts=6, shuffle=True, fletcher32=True)
        
        event_num_hits = np.array([len(hits['spacePoint_time']) for hits in hits_chunk], dtype=np.int16)
        event_num_tracks = np.array([len(tracks['truthMuon_pt']) for tracks in tracks_chunk], dtype=np.int16)
        
        f.create_dataset('num_hits', data=event_num_hits, compression='gzip', compression_opts=6)
        f.create_dataset('num_tracks', data=event_num_tracks, compression='gzip', compression_opts=6)
        f.create_dataset('event_numbers', data=np.array(event_numbers_chunk, dtype=np.int32), compression='gzip', compression_opts=6)
        
        f.attrs['num_events'] = num_events
        f.attrs['source_root_file'] = str(root_file)
        f.attrs['worker_id'] = worker_id
        f.attrs['valid_events_count'] = valid_events_count
    
    return chunk_info


def main():
    parser = argparse.ArgumentParser(description="Prefilter ATLAS muon events into batched HDF5 files using parallel processing.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input root files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output HDF5 files")
    parser.add_argument("-n", "--num_events_per_file", type=int, default=10000, help="Number of events per file")
    parser.add_argument("-pt", "--pt_threshold", type=float, default=5.0, help="Minimum pT threshold")
    parser.add_argument("-eta", "--eta_threshold", type=float, default=2.7, help="Maximum |eta| threshold")
    parser.add_argument("-nh", "--num_hits_threshold", type=int, default=3, help="Minimum number of hits")
    parser.add_argument("-max", "--max_events", type=int, default=1000, help="Maximum number of valid events each worker is allowed to process")
    parser.add_argument("-w", "--num_workers", type=int, default=10, help="Number of worker processes (default: 10)")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    filter = ParallelRootFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_events_per_file=args.num_events_per_file,
        pt_threshold=args.pt_threshold,
        eta_threshold=args.eta_threshold,
        num_hits_threshold=args.num_hits_threshold,
        max_events=args.max_events,
        num_workers=args.num_workers
    )
    
    filter.process_events()

if __name__ == "__main__":
    main()