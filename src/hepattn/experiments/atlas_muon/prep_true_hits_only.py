#!/usr/bin/env python3
"""
Prepare ATLAS muon dataset with TRUE HITS ONLY for regression studies.

This script processes ROOT files from ATLAS CERN, filters to keep only true hits
(spacePoint_truthLink != -1), and saves everything to a single uncompressed HDF5 file
for optimal read/write performance during training.

Key differences from prep_events_multiprocess.py:
1. Filters out ALL noise hits (spacePoint_truthLink == -1) immediately
2. Saves to a SINGLE uncompressed HDF5 file (not multiple compressed chunks)
3. Adds spacePoint_globPosX/Y/Z hit features for regression studies
4. Optimized for latency studies assuming perfect hit-track assignment

Since only ~0.6% of hits are true muon hits, this dramatically reduces data size
and eliminates any performance degradation due to noise in regression studies.
"""

import os
import sys
import argparse
import time
import yaml
import h5py
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class TrueHitsOnlyFilter:
    """
    Process ROOT files keeping only true hits (no noise) for regression studies.
    Outputs a single uncompressed HDF5 file for optimal I/O performance.
    """
    
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 expected_num_events_per_file: int,
                 max_events: int = -1, 
                 num_workers: int = None, 
                 no_NSW: bool = False, 
                 no_rpc: bool = False,
                 disable_track_filtering: bool = False, 
                 pt_threshold: float = 5.0, 
                 eta_threshold: float = 2.7, 
                 num_hits_threshold: int = 3,
                 max_hits_per_track: int = 100,
                 enable_baseline_filters: bool = True,
                 min_total_hits: int = 6,
                 min_stations: int = 2,
                 min_stations_with_n_hits: int = 2,
                 hits_per_station_threshold: int = 3,
                 min_eta: float = 0.1,
                 max_eta: float = 2.7):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.expected_num_events_per_file = expected_num_events_per_file
        self.max_events = max_events
        self.num_workers = num_workers or mp.cpu_count()
        self.no_NSW = no_NSW
        self.no_rpc = no_rpc
        self.disable_track_filtering = disable_track_filtering
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold
        self.max_hits_per_track = max_hits_per_track
        self.enable_baseline_filters = enable_baseline_filters
        self.min_total_hits = min_total_hits
        self.min_stations = min_stations
        self.min_stations_with_n_hits = min_stations_with_n_hits
        self.hits_per_station_threshold = hits_per_station_threshold
        self.min_eta = min_eta
        self.max_eta = max_eta
        
        # Global counters (will be aggregated from workers)
        self.excluded_tracks_count = 0
        self.excluded_events_count = 0
        self.valid_events_count = 0
        self.valid_tracks_count = 0
        
        # Detailed filtering statistics
        self.tracks_excluded_pt = 0
        self.tracks_excluded_eta = 0
        self.tracks_excluded_hits = 0
        self.tracks_excluded_max_hits = 0
        self.tracks_failed_min_hits = 0
        self.tracks_failed_eta_cuts = 0
        self.tracks_failed_pt_cuts = 0
        self.tracks_failed_station_cuts = 0
        self.events_excluded_no_tracks_after_filtering = 0
        self.events_excluded_technology_filtering = 0
        self.events_excluded_no_hits_after_technology = 0
        self.events_excluded_no_true_hits = 0
        
        # Hit statistics
        self.total_hits_before_true_filter = 0
        self.total_hits_after_true_filter = 0
        
        self.total_events_seen = 0
        
        # Data storage
        self.all_events = []  # Will store filtered events from all workers
        self.file_indices = []
        self.row_indices = []
        
        self.files = self._get_files()
        
        # Hit features - ADDED spacePoint_globPosX/Y/Z for regression
        self.hit_features = [
            "spacePoint_globEdgeHighX", "spacePoint_globEdgeHighY", "spacePoint_globEdgeHighZ",
            "spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ",
            "spacePoint_globPosX", "spacePoint_globPosY", "spacePoint_globPosZ",  # NEW features
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

    def _write_hdf5_optimized(self, output_h5_path: str, events: List[Dict], max_hits: int, max_tracks: int):
        """Write all events to HDF5 in a single optimized operation with perfect dimensions"""
        if not events:
            return
        
        num_events = len(events)
        
        # Pre-allocate arrays with exact dimensions needed
        print(f"  Allocating arrays: {num_events} events × {max_hits} hits × {len(self.hit_features)} features")
        hits_array = np.full((num_events, max_hits, len(self.hit_features)), np.nan, dtype=np.float32)
        tracks_array = np.full((num_events, max_tracks, len(self.track_features)), np.nan, dtype=np.float32)
        num_hits_array = np.zeros(num_events, dtype=np.int16)
        num_tracks_array = np.zeros(num_events, dtype=np.int16)
        event_numbers_array = np.zeros(num_events, dtype=np.int32)
        
        # Fill arrays
        print(f"  Filling arrays...")
        for i, event in enumerate(events):
            n_hits = event['num_hits']
            n_tracks = event['num_tracks']
            
            # Fill hits
            for feat_idx, feature in enumerate(self.hit_features):
                hits_array[i, :n_hits, feat_idx] = event['hits'][feature]
            
            # Fill tracks
            for feat_idx, feature in enumerate(self.track_features):
                tracks_array[i, :n_tracks, feat_idx] = event['tracks'][feature]
            
            num_hits_array[i] = n_hits
            num_tracks_array[i] = n_tracks
            event_numbers_array[i] = event['event_number']
            
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i + 1:,}/{num_events:,} events...")
        
        # Write to HDF5 in single operation
        print(f"  Writing to HDF5...")
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('hits', data=hits_array, compression=None)
            f.create_dataset('tracks', data=tracks_array, compression=None)
            f.create_dataset('num_hits', data=num_hits_array, compression=None)
            f.create_dataset('num_tracks', data=num_tracks_array, compression=None)
            f.create_dataset('event_numbers', data=event_numbers_array, compression=None)
            
            f.attrs['max_hits_per_track'] = max_hits
            f.attrs['max_tracks_per_event'] = max_tracks
            f.attrs['true_hits_only'] = True
            f.attrs['num_events'] = num_events
    
    def _process_with_memory_accumulation(self, worker_args, output_h5_path):
        """Process files in parallel, accumulate all data in memory, then write once"""
        from multiprocessing import Pool
        
        print(f"\n{'='*60}")
        print(f"Starting {len(worker_args)} workers in parallel...")
        print(f"Accumulating all filtered events in memory...")
        print(f"{'='*60}\n")
        
        # Process workers in parallel using a process pool
        with Pool(processes=self.num_workers) as pool:
            # Submit all worker tasks and collect results as they complete
            results = pool.map(process_worker_files_true_hits, worker_args)
        
        print(f"\n{'='*60}")
        print(f"All workers complete. Aggregating results...")
        print(f"{'='*60}\n")
        
        # Accumulate all filtered events in memory
        all_filtered_events = []
        
        # Process results and aggregate statistics
        for result in results:
            if result:
                worker_id = result['worker_id']
                
                # Accumulate events
                if len(result['filtered_events']) > 0:
                    all_filtered_events.extend(result['filtered_events'])
                    print(f"Worker {worker_id}: {len(result['filtered_events'])} events")
                
                # Update statistics
                self.excluded_tracks_count += result['excluded_tracks_count']
                self.excluded_events_count += result['excluded_events_count']
                self.valid_events_count += result['valid_events_count']
                self.valid_tracks_count += result['valid_tracks_count']
                self.tracks_excluded_pt += result['tracks_excluded_pt']
                self.tracks_excluded_eta += result['tracks_excluded_eta']
                self.tracks_excluded_hits += result['tracks_excluded_hits']
                self.tracks_excluded_max_hits += result.get('tracks_excluded_max_hits', 0)
                self.tracks_failed_min_hits += result.get('tracks_failed_min_hits', 0)
                self.tracks_failed_eta_cuts += result.get('tracks_failed_eta_cuts', 0)
                self.tracks_failed_pt_cuts += result.get('tracks_failed_pt_cuts', 0)
                self.tracks_failed_station_cuts += result.get('tracks_failed_station_cuts', 0)
                self.events_excluded_no_tracks_after_filtering += result['events_excluded_no_tracks_after_filtering']
                self.events_excluded_technology_filtering += result['events_excluded_technology_filtering']
                self.events_excluded_no_hits_after_technology += result['events_excluded_no_hits_after_technology']
                self.events_excluded_no_true_hits += result['events_excluded_no_true_hits']
                self.total_hits_before_true_filter += result['total_hits_before_true_filter']
                self.total_hits_after_true_filter += result['total_hits_after_true_filter']
                self.total_events_seen += result['total_events_seen']
        
        print(f"\nTotal accumulated events: {len(all_filtered_events):,}")
        
        if len(all_filtered_events) > 0:
            # Calculate perfect padding dimensions from actual data
            print(f"Calculating optimal HDF5 dimensions...")
            max_hits = max(event['num_hits'] for event in all_filtered_events)
            max_tracks = max(event['num_tracks'] for event in all_filtered_events)
            
            print(f"  Max hits per event: {max_hits}")
            print(f"  Max tracks per event: {max_tracks}")
            print(f"\nWriting {len(all_filtered_events):,} events to HDF5 in single optimized write...")
            
            # Write everything at once with perfect dimensions
            self._write_hdf5_optimized(output_h5_path, all_filtered_events, max_hits, max_tracks)
            
            print(f"HDF5 write complete!")
        else:
            print(f"No events to write.")
        
        # Save index arrays
        if len(all_filtered_events) > 0:
            filtered_file_indices = np.zeros(len(all_filtered_events), dtype=np.int16)
            filtered_row_indices = np.arange(len(all_filtered_events), dtype=np.int32)
            
            np.save(os.path.join(self.output_dir, 'event_file_indices.npy'), filtered_file_indices)
            np.save(os.path.join(self.output_dir, 'event_row_indices.npy'), filtered_row_indices)
    
    def _split_files(self) -> List[List[Path]]:
        """Split files into chunks for parallel processing"""
        files_per_worker = len(self.files) // self.num_workers
        remainder = len(self.files) % self.num_workers
        
        file_chunks = []
        start_idx = 0
        
        for i in range(self.num_workers):
            chunk_size = files_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx < len(self.files):
                file_chunks.append(self.files[start_idx:end_idx])
            else:
                file_chunks.append([])
            
            start_idx = end_idx
        
        return file_chunks

    def process_events(self):
        """Main method to process events in parallel with incremental writing"""
        print(f"Starting TRUE HITS ONLY processing with {self.num_workers} workers...")
        print(f"Total files to process: {len(self.files)}")
        print(f"Filtering: Only true hits (spacePoint_truthLink != -1)")
        print(f"Max hits per track: {self.max_hits_per_track}")
        if self.enable_baseline_filters:
            print(f"Baseline filters ENABLED:")
            print(f"  - Min total hits: {self.min_total_hits}")
            print(f"  - Min stations: {self.min_stations}")
            print(f"  - Stations with ≥{self.hits_per_station_threshold} hits: {self.min_stations_with_n_hits}")
            print(f"  - Eta range: {self.min_eta} - {self.max_eta}")
            print(f"  - Min pT: {self.pt_threshold} GeV")
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
        output_h5_path = os.path.join(self.output_dir, 'data', 'filtered_events.h5')
        
        # Split files among workers
        file_chunks = self._split_files()
        
        # Print worker assignment
        for i, chunk in enumerate(file_chunks):
            if chunk:
                print(f"Worker {i}: {len(chunk)} files")
        
        # Create worker arguments
        worker_args = []
        for worker_id, file_chunk in enumerate(file_chunks):
            if file_chunk:
                args = (
                    worker_id, file_chunk, self.expected_num_events_per_file,
                    self.max_events, self.hit_features, self.track_features,
                    self.no_NSW, self.no_rpc, self.disable_track_filtering, 
                    self.pt_threshold, self.eta_threshold, self.num_hits_threshold,
                    self.max_hits_per_track, self.enable_baseline_filters,
                    self.min_total_hits, self.min_stations, self.min_stations_with_n_hits,
                    self.hits_per_station_threshold, self.min_eta, self.max_eta
                )
                worker_args.append(args)
        
        # Process with memory accumulation and single optimized write
        start_time = time.time()
        self._process_with_memory_accumulation(worker_args, output_h5_path)
        
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        # Save metadata
        self._save_metadata(processing_time)



    def _save_metadata(self, processing_time: float):
        """Save metadata for compatibility with data.py"""
        
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_events_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_events_count) if self.valid_events_count > 0 else 0
        
        true_hit_percent = (self.total_hits_after_true_filter / self.total_hits_before_true_filter * 100) if self.total_hits_before_true_filter > 0 else 0
        
        dataset_info = {
            'hit_features': self.hit_features,
            'track_features': self.track_features,
            'processing_summary': {
                'total_excluded_tracks': int(self.excluded_tracks_count),
                'total_tracks_processed': int(total_tracks),
                'excluded_tracks_percentage': float(excluded_tracks_percent),
                'tracks_excluded_pt': int(self.tracks_excluded_pt),
                'tracks_excluded_eta': int(self.tracks_excluded_eta),
                'tracks_excluded_hits': int(self.tracks_excluded_hits),
                'total_excluded_events': int(self.excluded_events_count),
                'total_events_processed': int(total_events),
                'total_events_seen': int(self.total_events_seen),
                'excluded_events_percentage': float(excluded_events_percent),
                'events_excluded_no_hits_after_technology': int(self.events_excluded_no_hits_after_technology),
                'events_excluded_no_tracks_after_filtering': int(self.events_excluded_no_tracks_after_filtering),
                'events_excluded_no_true_hits': int(self.events_excluded_no_true_hits),
                'valid_events': int(self.valid_events_count),
                'valid_tracks': int(self.valid_tracks_count),
                'average_tracks_per_event': float(avg_tracks_per_event),
                'total_hits_before_true_filter': int(self.total_hits_before_true_filter),
                'total_hits_after_true_filter': int(self.total_hits_after_true_filter),
                'true_hit_percentage': float(true_hit_percent),
                'processing_time_seconds': float(processing_time),
                'num_workers': int(self.num_workers),
                'num_root_files': int(len(self.files)),
                'processing_status': 'Complete'
            },
            'processing_parameters': {
                'expected_number_of_events_per_file': self.expected_num_events_per_file,
                'max_events': self.max_events,
                'no_NSW': self.no_NSW,
                'no_rpc': self.no_rpc,
                'disable_track_filtering': self.disable_track_filtering,
                'pt_threshold': self.pt_threshold,
                'eta_threshold': self.eta_threshold,
                'num_hits_threshold': self.num_hits_threshold,
                'max_hits_per_track': self.max_hits_per_track,
                'enable_baseline_filters': self.enable_baseline_filters,
                'min_total_hits': self.min_total_hits,
                'min_stations': self.min_stations,
                'min_stations_with_n_hits': self.min_stations_with_n_hits,
                'hits_per_station_threshold': self.hits_per_station_threshold,
                'min_eta': self.min_eta,
                'max_eta': self.max_eta,
                'true_hits_only': True,
                'compression': False
            },
            'processed_files': [str(file_path) for file_path in self.files],
            'event_mapping': {
                'description': 'All events stored in single uncompressed HDF5 file',
                'total_events': int(self.valid_events_count),
                'total_chunks': 1,
                'index_files': {
                    'file_indices': 'event_file_indices.npy',
                    'row_indices': 'event_row_indices.npy',
                },
                'chunk_summary': [
                    {
                        'h5_file': 'data/filtered_events.h5',
                        'source_root_file': 'merged_from_all_files',
                        'event_count': int(self.valid_events_count),
                        'worker_id': -1
                    }
                ]
            }
        }
        
        metadata_file = os.path.join(self.output_dir, 'metadata.yaml')
        with open(metadata_file, 'w') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
        
        print(f"  Metadata saved to: {metadata_file}")

    def _print_final_statistics(self, processing_time: float):
        """Print final processing statistics"""
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_events_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_events_count) if self.valid_events_count > 0 else 0
        
        true_hit_percent = (self.total_hits_after_true_filter / self.total_hits_before_true_filter * 100) if self.total_hits_before_true_filter > 0 else 0
        noise_removed_percent = 100 - true_hit_percent
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY - TRUE HITS ONLY")
        print(f"{'='*60}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Total events seen: {self.total_events_seen:,}")
        print(f"")
        print(f"TRUE HIT FILTERING STATISTICS:")
        print(f"  Total hits before filtering: {self.total_hits_before_true_filter:,}")
        print(f"  Total hits after filtering: {self.total_hits_after_true_filter:,}")
        print(f"  True hit percentage: {true_hit_percent:.2f}%")
        print(f"  Noise removed: {noise_removed_percent:.2f}%")
        print(f"  Events excluded (no true hits): {self.events_excluded_no_true_hits:,}")
        print(f"")
        print(f"TRACK FILTERING STATISTICS:")
        if not self.disable_track_filtering:
            print(f"  Total excluded tracks: {self.excluded_tracks_count:,} out of {total_tracks:,} ({excluded_tracks_percent:.2f}%)")
            print(f"    - Excluded due to pT < {self.pt_threshold} GeV: {self.tracks_excluded_pt:,}")
            print(f"    - Excluded due to |eta| > {self.eta_threshold}: {self.tracks_excluded_eta:,}")
            print(f"    - Excluded due to < {self.num_hits_threshold} hits: {self.tracks_excluded_hits:,}")
            print(f"    - Excluded due to > {self.max_hits_per_track} hits (noise): {self.tracks_excluded_max_hits:,}")
        else:
            print(f"  Track filtering: DISABLED")
        
        if self.enable_baseline_filters:
            print(f"\n  BASELINE FILTERING STATISTICS:")
            print(f"    - Failed min {self.min_total_hits} hits: {self.tracks_failed_min_hits:,}")
            print(f"    - Failed eta cuts ({self.min_eta}-{self.max_eta}): {self.tracks_failed_eta_cuts:,}")
            print(f"    - Failed pT ≥ {self.pt_threshold} GeV: {self.tracks_failed_pt_cuts:,}")
            print(f"    - Failed station cuts ({self.min_stations_with_n_hits} stations w/ ≥{self.hits_per_station_threshold} hits): {self.tracks_failed_station_cuts:,}")
        
        print(f"  Valid tracks: {self.valid_tracks_count:,}")
        print(f"")
        print(f"EVENT FILTERING STATISTICS:")
        print(f"  Total excluded events: {self.excluded_events_count:,} out of {total_events:,} ({excluded_events_percent:.2f}%)")
        print(f"    - Events with no hits after technology filtering: {self.events_excluded_no_hits_after_technology:,}")
        print(f"    - Events with no true hits: {self.events_excluded_no_true_hits:,}")
        if not self.disable_track_filtering:
            print(f"    - Events with no tracks after filtering: {self.events_excluded_no_tracks_after_filtering:,}")
        print(f"  Valid events: {self.valid_events_count:,}")
        print(f"  Average tracks per event: {avg_tracks_per_event:.2f}")
        print(f"")
        print(f"OUTPUT:")
        print(f"  Single HDF5 file: {self.output_dir}/data/filtered_events.h5")
        print(f"  Compression: DISABLED (for faster I/O)")
        print(f"{'='*60}")


def process_worker_files_true_hits(args: Tuple) -> Dict:
    """Worker function to process a subset of files, keeping only TRUE HITS"""
    import uproot  # Import here for multiprocessing
    
    (worker_id, file_chunk, expected_num_events_per_file,
     max_events, hit_features, track_features, no_NSW, no_rpc,
     disable_track_filtering, pt_threshold, eta_threshold, num_hits_threshold,
     max_hits_per_track, enable_baseline_filters, min_total_hits, min_stations,
     min_stations_with_n_hits, hits_per_station_threshold, min_eta, max_eta) = args
    
    if not file_chunk:
        return None
    
    print(f"Worker {worker_id}: Starting processing of {len(file_chunk)} files (TRUE HITS ONLY)")
    sys.stdout.flush()
    
    # Initialize worker-specific counters
    excluded_tracks_count = 0
    excluded_events_count = 0
    valid_tracks_count = 0
    
    tracks_excluded_pt = 0
    tracks_excluded_eta = 0
    tracks_excluded_hits = 0
    tracks_excluded_max_hits = 0
    tracks_failed_min_hits = 0
    tracks_failed_eta_cuts = 0
    tracks_failed_pt_cuts = 0
    tracks_failed_station_cuts = 0
    events_excluded_no_tracks_after_filtering = 0
    events_excluded_technology_filtering = 0
    events_excluded_no_hits_after_technology = 0
    events_excluded_no_true_hits = 0
    
    total_hits_before_true_filter = 0
    total_hits_after_true_filter = 0
    
    total_valid_events = 0
    total_events_seen = 0
    
    filtered_events = []  # Store only filtered events (with true hits only)
    
    for file_idx, root_file in enumerate(file_chunk):
        try:
            with uproot.open(root_file) as rf:
                tree_keys = [key for key in rf.keys() if ';' in key]
                if not tree_keys:
                    print(f"Worker {worker_id}: No tree found in {root_file.name}")
                    continue
                
                tree = tree_keys[0].split(';')[0]
                num_events = rf[tree].num_entries
                chunk_size = 50
                
                for chunk_start in range(0, num_events, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_events)
                    
                    # Load chunk data
                    try:
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
                    except Exception as e:
                        print(f"Worker {worker_id}: Error loading chunk {chunk_start}-{chunk_end} from {root_file.name}: {e}")
                        sys.stdout.flush()
                        excluded_events_count += (chunk_end - chunk_start)
                        continue
                    
                    # Process each event in chunk
                    for event_idx_in_chunk in range(chunk_end - chunk_start):
                        total_events_seen += 1
                        
                        # Check max events limit
                        if max_events > 0 and total_valid_events >= max_events:
                            remaining = num_events - (chunk_start + event_idx_in_chunk)
                            excluded_events_count += remaining
                            total_events_seen += remaining - 1
                            break
                        
                        # Get hits for this event
                        hits = {branch: hit_features_chunk[branch][event_idx_in_chunk].copy() 
                               for branch in hit_features}
                        
                        # Count total hits before any filtering
                        total_hits_before_true_filter += len(hits['spacePoint_time'])
                        
                        # Apply technology filtering first (if requested)
                        technology_values = hits['spacePoint_technology']
                        keep_mask = np.ones(len(technology_values), dtype=bool)
                        
                        if no_NSW:
                            keep_mask &= ~np.isin(technology_values, [4, 5])
                        
                        if no_rpc:
                            keep_mask &= (technology_values != 2)
                        
                        # Apply technology filter
                        for branch in hit_features:
                            hits[branch] = hits[branch][keep_mask]
                        
                        if len(hits['spacePoint_time']) == 0:
                            events_excluded_no_hits_after_technology += 1
                            excluded_events_count += 1
                            continue
                        
                        # ========================================
                        # KEY FILTER: Keep only TRUE HITS
                        # ========================================
                        true_hit_mask = hits['spacePoint_truthLink'] != -1
                        
                        for branch in hit_features:
                            hits[branch] = hits[branch][true_hit_mask]
                        
                        # Count hits after true hit filtering
                        total_hits_after_true_filter += len(hits['spacePoint_time'])
                        
                        if len(hits['spacePoint_time']) == 0:
                            events_excluded_no_true_hits += 1
                            excluded_events_count += 1
                            continue
                        
                        # Get unique valid tracks from TRUE HITS
                        unique_tracks = np.unique(hits['spacePoint_truthLink'])
                        valid_tracks = unique_tracks[unique_tracks != -1]  # Should already be all valid
                        
                        if len(valid_tracks) == 0:
                            excluded_events_count += 1
                            continue
                        
                        if not disable_track_filtering:
                            # Apply track filters
                            exclude_tracks = []
                            for track_idx in valid_tracks:
                                track_excluded = False
                                
                                # Get track's true hits
                                track_hit_mask = hits['spacePoint_truthLink'] == track_idx
                                track_true_hits_count = np.sum(track_hit_mask)
                                
                                # Filter: Max hits per track (remove noise)
                                if track_true_hits_count > max_hits_per_track:
                                    tracks_excluded_max_hits += 1
                                    track_excluded = True
                                
                                # Baseline filters (if enabled)
                                if enable_baseline_filters and not track_excluded:
                                    truth_pt = track_features_chunk['truthMuon_pt'][event_idx_in_chunk][track_idx]
                                    truth_eta = track_features_chunk['truthMuon_eta'][event_idx_in_chunk][track_idx]
                                    
                                    # Pre-filter 1: Minimum total hits
                                    if track_true_hits_count < min_total_hits:
                                        tracks_failed_min_hits += 1
                                        track_excluded = True
                                    
                                    # Pre-filter 2: eta acceptance cuts
                                    if not track_excluded and (np.abs(truth_eta) < min_eta or np.abs(truth_eta) > max_eta):
                                        tracks_failed_eta_cuts += 1
                                        track_excluded = True
                                    
                                    # Pre-filter 3: pt threshold
                                    if not track_excluded and truth_pt < pt_threshold:
                                        tracks_failed_pt_cuts += 1
                                        track_excluded = True
                                    
                                    # Pre-filter 4: station requirements
                                    if not track_excluded:
                                        track_station_indices = hits['spacePoint_stationIndex'][track_hit_mask]
                                        unique_stations, station_counts = np.unique(track_station_indices, return_counts=True)
                                        
                                        # At least min_stations different stations
                                        if len(unique_stations) < min_stations:
                                            tracks_failed_station_cuts += 1
                                            track_excluded = True
                                        else:
                                            # min_stations_with_n_hits stations with >= hits_per_station_threshold hits each
                                            n_good_stations = np.sum(station_counts >= hits_per_station_threshold)
                                            if n_good_stations < min_stations_with_n_hits:
                                                tracks_failed_station_cuts += 1
                                                track_excluded = True
                                
                                # Legacy filters (if baseline disabled)
                                if not enable_baseline_filters and not track_excluded:
                                    # Check pT threshold
                                    if track_features_chunk['truthMuon_pt'][event_idx_in_chunk][track_idx] < pt_threshold:
                                        tracks_excluded_pt += 1
                                        track_excluded = True
                                    
                                    # Check eta threshold
                                    if abs(track_features_chunk['truthMuon_eta'][event_idx_in_chunk][track_idx]) > eta_threshold:
                                        tracks_excluded_eta += 1
                                        track_excluded = True
                                    
                                    # Check minimum hits threshold (on true hits only)
                                    if track_true_hits_count < num_hits_threshold:
                                        tracks_excluded_hits += 1
                                        track_excluded = True
                                
                                if track_excluded:
                                    exclude_tracks.append(track_idx)
                                    excluded_tracks_count += 1
                            
                            remaining_tracks = np.setdiff1d(valid_tracks, exclude_tracks)
                            
                            if len(remaining_tracks) == 0:
                                events_excluded_no_tracks_after_filtering += 1
                                excluded_events_count += 1
                                continue
                            
                            valid_tracks_count += len(remaining_tracks)
                            total_valid_events += 1
                            
                            # Filter hits to only those belonging to remaining tracks
                            hit2track_mask = np.isin(hits['spacePoint_truthLink'], remaining_tracks)
                            for branch in hit_features:
                                hits[branch] = hits[branch][hit2track_mask]
                            
                            # Build track data
                            track_mask = np.isin(
                                np.arange(len(track_features_chunk['truthMuon_pt'][event_idx_in_chunk])),
                                remaining_tracks
                            )
                            tracks = {branch: track_features_chunk[branch][event_idx_in_chunk][track_mask] 
                                     for branch in track_features}
                        else:
                            # No track filtering
                            valid_tracks_count += len(valid_tracks)
                            total_valid_events += 1
                            
                            track_mask = np.isin(
                                np.arange(len(track_features_chunk['truthMuon_pt'][event_idx_in_chunk])),
                                valid_tracks
                            )
                            tracks = {branch: track_features_chunk[branch][event_idx_in_chunk][track_mask] 
                                     for branch in track_features}
                        
                        # Store filtered event (only true hits, minimal memory)
                        filtered_events.append({
                            'hits': hits,
                            'tracks': tracks,
                            'num_hits': len(hits['spacePoint_time']),
                            'num_tracks': len(tracks['truthMuon_pt']),
                            'event_number': event_numbers_array[event_idx_in_chunk]
                        })
                    
                    if max_events > 0 and total_valid_events >= max_events:
                        break
                
                if max_events > 0 and total_valid_events >= max_events:
                    break
                    
        except Exception as e:
            print(f"Worker {worker_id}: Error processing file {root_file}: {e}")
            sys.stdout.flush()
            excluded_events_count += expected_num_events_per_file
            continue
        
        # Progress update per file
        print(f"Worker {worker_id}: Completed file {file_idx + 1}/{len(file_chunk)}: {root_file.name} "
              f"(Events: {total_valid_events}, True hits: {total_hits_after_true_filter:,})")
        sys.stdout.flush()
    
    print(f"\nWorker {worker_id}: FINISHED. Valid events: {total_valid_events}, "
          f"Excluded: {excluded_events_count}, "
          f"True hits: {total_hits_after_true_filter:,}/{total_hits_before_true_filter:,}")
    sys.stdout.flush()
    
    return {
        'worker_id': worker_id,
        'excluded_tracks_count': excluded_tracks_count,
        'excluded_events_count': excluded_events_count,
        'valid_events_count': total_valid_events,
        'valid_tracks_count': valid_tracks_count,
        'total_events_seen': total_events_seen,
        'tracks_excluded_pt': tracks_excluded_pt,
        'tracks_excluded_eta': tracks_excluded_eta,
        'tracks_excluded_hits': tracks_excluded_hits,
        'tracks_excluded_max_hits': tracks_excluded_max_hits,
        'tracks_failed_min_hits': tracks_failed_min_hits,
        'tracks_failed_eta_cuts': tracks_failed_eta_cuts,
        'tracks_failed_pt_cuts': tracks_failed_pt_cuts,
        'tracks_failed_station_cuts': tracks_failed_station_cuts,
        'events_excluded_no_tracks_after_filtering': events_excluded_no_tracks_after_filtering,
        'events_excluded_technology_filtering': events_excluded_technology_filtering,
        'events_excluded_no_hits_after_technology': events_excluded_no_hits_after_technology,
        'events_excluded_no_true_hits': events_excluded_no_true_hits,
        'total_hits_before_true_filter': total_hits_before_true_filter,
        'total_hits_after_true_filter': total_hits_after_true_filter,
        'filtered_events': filtered_events
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ATLAS muon dataset with TRUE HITS ONLY for regression studies."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                       help="Directory containing input ROOT files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                       help="Directory to save output HDF5 file")
    parser.add_argument("-n", "--expected_num_events_per_file", type=int, default=2000, 
                       help="Expected number of events per ROOT file")
    parser.add_argument("-max", "--max_events", type=int, default=-1, 
                       help="Maximum number of valid events to process (-1 for all)")
    parser.add_argument("-w", "--num_workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--no-NSW", action="store_true", default=False, 
                       help="Remove STGC and MM technology hits")
    parser.add_argument("--no-RPC", action="store_true", default=False, 
                       help="Remove RPC technology hits")
    parser.add_argument("--disable-track-filtering", action="store_true", default=False, 
                       help="Disable track filtering based on pt, eta, and hit count")
    parser.add_argument("--pt-threshold", type=float, default=5.0, 
                       help="Minimum pT threshold for tracks (GeV, default: 5.0)")
    parser.add_argument("--eta-threshold", type=float, default=2.7, 
                       help="Maximum |eta| threshold for tracks (default: 2.7)")
    parser.add_argument("--num-hits-threshold", type=int, default=3, 
                       help="Minimum number of TRUE hits per track (default: 3)")
    parser.add_argument("--max-hits-per-track", type=int, default=100,
                       help="Maximum hits per track - longer tracks filtered as noise (default: 100)")
    parser.add_argument("--enable-baseline-filters", action="store_true", default=True,
                       help="Enable baseline track filtering (default: True)")
    parser.add_argument("--disable-baseline-filters", action="store_true", default=False,
                       help="Disable baseline track filtering")
    parser.add_argument("--min-total-hits", type=int, default=6,
                       help="Baseline: minimum total hits per track (default: 6)")
    parser.add_argument("--min-stations", type=int, default=2,
                       help="Baseline: minimum number of different stations (default: 2)")
    parser.add_argument("--min-stations-with-n-hits", type=int, default=2,
                       help="Baseline: minimum stations with ≥N hits (default: 2)")
    parser.add_argument("--hits-per-station-threshold", type=int, default=3,
                       help="Baseline: hits per station threshold (default: 3)")
    parser.add_argument("--min-eta", type=float, default=0.1,
                       help="Baseline: minimum |eta| (default: 0.1)")
    parser.add_argument("--max-eta-baseline", type=float, default=2.7,
                       help="Baseline: maximum |eta| (default: 2.7)")

    args = parser.parse_args()
    
    # Handle baseline filters flag
    enable_baseline = args.enable_baseline_filters and not args.disable_baseline_filters
    
    # Modify output directory path based on filtering flags
    output_dir = args.output_dir
    if args.no_NSW or args.no_RPC:
        suffix = ""
        if args.no_NSW:
            suffix += "_no-NSW"
        if args.no_RPC:
            suffix += "_no-RPC"
        output_dir = output_dir + suffix
    
    # Add true_hits_only suffix
    if not output_dir.endswith("_true_hits_only"):
        output_dir = output_dir + "_true_hits_only"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*80)
    print("ATLAS MUON DATASET PREPARATION - TRUE HITS ONLY")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Number of workers: {args.num_workers if args.num_workers else 'Auto'}")
    print(f"No NSW: {args.no_NSW}")
    print(f"No RPC: {args.no_RPC}")
    print(f"Disable track filtering: {args.disable_track_filtering}")
    if not args.disable_track_filtering:
        print(f"  pT threshold: {args.pt_threshold} GeV")
        print(f"  eta threshold: {args.eta_threshold}")
        print(f"  min hits threshold: {args.num_hits_threshold}")
        print(f"  max hits threshold: {args.max_hits_per_track}")
    print(f"Baseline filters: {enable_baseline}")
    if enable_baseline:
        print(f"  Min total hits: {args.min_total_hits}")
        print(f"  Min stations: {args.min_stations}")
        print(f"  Stations with ≥{args.hits_per_station_threshold} hits: {args.min_stations_with_n_hits}")
        print(f"  Eta range: {args.min_eta} - {args.max_eta_baseline}")
    print(f"")
    print(f"KEY FEATURE: Only TRUE HITS (spacePoint_truthLink != -1) will be kept")
    print(f"COMPRESSION: Disabled for faster I/O")
    print("="*80)
    
    processor = TrueHitsOnlyFilter(
        input_dir=args.input_dir,
        output_dir=output_dir,
        expected_num_events_per_file=args.expected_num_events_per_file,
        max_events=args.max_events,
        num_workers=args.num_workers,
        no_NSW=args.no_NSW,
        no_rpc=args.no_RPC,
        disable_track_filtering=args.disable_track_filtering,
        pt_threshold=args.pt_threshold,
        eta_threshold=args.eta_threshold,
        num_hits_threshold=args.num_hits_threshold,
        max_hits_per_track=args.max_hits_per_track,
        enable_baseline_filters=enable_baseline,
        min_total_hits=args.min_total_hits,
        min_stations=args.min_stations,
        min_stations_with_n_hits=args.min_stations_with_n_hits,
        hits_per_station_threshold=args.hits_per_station_threshold,
        min_eta=args.min_eta,
        max_eta=args.max_eta_baseline
    )
    
    processor.process_events()


if __name__ == "__main__":
    main()
