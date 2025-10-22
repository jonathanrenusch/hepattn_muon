#!/usr/bin/env python3
"""
Filter preprocessed ATLAS muon dataset using hit filter predictions.

This script applies hit filtering using ML model predictions and additional
cuts (max tracks per event, max hits per event) to create a reduced dataset
optimized for the second stage of tracking model training.

The script maintains the same structure as the original dataset for 
plug-and-play compatibility with the existing data loading routines.
"""

import os
import sys
import glob
import numpy as np
import h5py
import yaml
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
import time
from sklearn.metrics import roc_curve
import warnings

warnings.filterwarnings('ignore')

def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

class HitFilterDatasetReducer:
    """
    Filter dataset using hit filter predictions and additional cuts.
    """
    
    def __init__(self, 
                 input_dir: str, 
                 eval_file: str, 
                 output_dir: str,
                 working_point: float = 0.99,
                 detection_threshold: float = None,
                 max_tracks_per_event: int = 3,
                 max_hits_per_event: int = 500,
                 max_events: int = -1,
                 num_workers: int = None,
                 disable_track_filtering: bool = False,
                 pt_threshold: float = 5.0,
                 eta_threshold: float = 2.7,
                 num_hits_threshold: int = 3):
        
        self.input_dir = Path(input_dir)
        self.eval_file = Path(eval_file)
        self.output_dir = Path(output_dir)
        self.working_point = working_point
        self.detection_threshold = detection_threshold
        self.max_tracks_per_event = max_tracks_per_event
        self.max_hits_per_event = max_hits_per_event
        self.max_events = max_events
        self.num_workers = num_workers or mp.cpu_count()
        self.disable_track_filtering = disable_track_filtering
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold
        
        # Load original metadata
        self.load_original_metadata()
        
        # Calculate detection threshold if not provided
        if self.detection_threshold is None:
            self.detection_threshold = self.calculate_detection_threshold()
            print(f"Calculated detection threshold: {self.detection_threshold:.6f} for working point {self.working_point}")
        else:
            print(f"Using provided detection threshold: {self.detection_threshold}")
        
        # Statistics tracking
        self.stats = {
            'total_events_processed': 0,
            'events_passed_hit_filter': 0,
            'events_failed_no_hits_after_filter': 0,
            'events_failed_max_tracks': 0,
            'events_failed_min_tracks': 0,
            'events_failed_max_hits': 0,
            'events_failed_min_hits': 0,
            'events_failed_eval_data_missing': 0,
            'events_failed_data_loading': 0,
            'events_failed_track_filtering': 0,
            'events_final_output': 0,
            'total_hits_before': 0,
            'total_hits_after': 0,
            'total_tracks_before': 0,
            'total_tracks_after': 0,
            'excluded_tracks_count': 0,
            # Detailed track filtering statistics
            'tracks_excluded_pt': 0,
            'tracks_excluded_eta': 0,
            'tracks_excluded_hits': 0,
            'tracks_excluded_no_true_hits': 0
        }
        
        # Data storage
        self.filtered_events = []
        self.file_indices = []
        self.row_indices = []
        
    def load_original_metadata(self):
        """Load metadata from the original dataset."""
        metadata_path = self.input_dir / 'metadata.yaml'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.original_metadata = yaml.safe_load(f)
        
        self.hit_features = self.original_metadata['hit_features']
        self.track_features = self.original_metadata['track_features']
        
        # Load index arrays
        self.original_file_indices = np.load(self.input_dir / 'event_file_indices.npy')
        self.original_row_indices = np.load(self.input_dir / 'event_row_indices.npy')
        
        print(f"Loaded original dataset with {len(self.original_file_indices)} events")
        
    def calculate_detection_threshold(self):
        """Calculate detection threshold for given working point using DataLoader approach."""
        print("Calculating detection threshold from evaluation file...")
        
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
        import yaml
        
        # We need a config file to set up the data module - look for one in the configs directory
        config_dir = Path(__file__).parent / "configs" / "NGT"
        config_files = list(config_dir.glob("*.yaml"))
        
        if not config_files:
            # Fallback to a basic configuration
            inputs = {"hit": ["spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ"]}
            targets = {"hit": ["on_valid_particle"], "particle": ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]}
        else:
            # Load the first available config
            with open(config_files[0], 'r') as f:
                config = yaml.safe_load(f)
            data_config = config.get('data', {})
            inputs = data_config.get('inputs', {"hit": ["spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ"]})
            targets = data_config.get('targets', {"hit": ["on_valid_particle"], "particle": ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]})
        
        # Set up data module for a small sample to calculate threshold
        data_module = AtlasMuonDataModule(
            train_dir=str(self.input_dir),
            val_dir=str(self.input_dir),
            test_dir=str(self.input_dir),
            num_workers=50,  # Use fewer workers for threshold calculation
            num_train=-1,  # Small sample
            num_val=-1,
            num_test=-1,  # Use only 1000 events for threshold calculation
            batch_size=1,
            inputs=inputs,
            targets=targets,
            pin_memory=True,
        )
        
        data_module.setup(stage='test')
        test_dataloader = data_module.test_dataloader()
        
        all_logits = []
        all_true_labels = []
        
        with h5py.File(self.eval_file, 'r') as eval_f:
            for batch_idx, (inputs_batch, targets_batch) in enumerate(tqdm(test_dataloader, desc="Loading evaluation data")):
                try:
                    # Get sample ID to match with evaluation file
                    if "sample_id" not in targets_batch:
                        continue
                    
                    event_idx = targets_batch["sample_id"][0].item()
                    
                    # Check if this event exists in evaluation file
                    if str(event_idx) not in eval_f:
                        continue
                    
                    # Load logits from evaluation file
                    hit_logits = eval_f[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)
                    
                    # Load true labels from dataset
                    true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(np.bool_)
                    
                    # Check dimensions match
                    if len(hit_logits) != len(true_labels):
                        print(f"Warning: Dimension mismatch for event {event_idx}: logits={len(hit_logits)}, labels={len(true_labels)}")
                        continue
                    
                    all_logits.extend(hit_logits)
                    all_true_labels.extend(true_labels)
                    
                    # Stop after collecting enough data for threshold calculation
                    # if len(all_logits) > 50000000:  # 100 Mio hits should be enough
                    #     break
                        
                except KeyError as e:
                    print(f"Warning: Could not load data for event {event_idx}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                    continue
        
        if not all_logits:
            raise ValueError("No valid logits found in evaluation file")
        
        all_logits = np.array(all_logits)
        all_true_labels = np.array(all_true_labels)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_true_labels, all_logits)
        
        # Find threshold that achieves target efficiency
        target_efficiency = self.working_point
        valid_indices = tpr >= target_efficiency
        
        if not np.any(valid_indices):
            raise ValueError(f"Cannot achieve target efficiency {target_efficiency}")
        
        threshold = thresholds[valid_indices][0]
        
        print(f"ROC calculation complete: {len(all_logits)} total hits processed")
        
        return threshold
    
    def process_events(self):
        """Main processing method with multiprocessing."""
        print(f"Starting filtered dataset creation with {self.num_workers} workers...")
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        
        # Split events among workers
        num_events = len(self.original_file_indices)
        
        # Apply max_events limit if specified
        if self.max_events > 0 and self.max_events < num_events:
            num_events = self.max_events
            print(f"Limiting processing to {num_events} events (max_events={self.max_events})")
        
        chunk_size = max(1, num_events // self.num_workers)
        
        event_chunks = []
        for i in range(0, num_events, chunk_size):
            end_idx = min(i + chunk_size, num_events)
            event_chunks.append((i, end_idx))
        
        # Limit to actual number of workers needed
        event_chunks = event_chunks[:self.num_workers]
        
        print(f"Processing {num_events} events in {len(event_chunks)} chunks")
        
        # Create worker arguments
        worker_args = []
        for worker_id, (start_idx, end_idx) in enumerate(event_chunks):
            args = (
                worker_id,
                start_idx,
                end_idx,
                str(self.input_dir),
                str(self.eval_file),
                str(self.output_dir),
                self.detection_threshold,
                self.max_tracks_per_event,
                self.max_hits_per_event,
                self.hit_features,
                self.track_features,
                self.original_metadata['event_mapping']['chunk_summary'],
                self.disable_track_filtering,
                self.pt_threshold,
                self.eta_threshold,
                self.num_hits_threshold
            )
            worker_args.append(args)
        
        # Process in parallel
        start_time = time.time()
        
        with mp.Pool(len(event_chunks)) as pool:
            results = list(tqdm(
                pool.imap(process_worker_events, worker_args),
                total=len(worker_args),
                desc="Worker progress"
            ))
        
        processing_time = time.time() - start_time
        
        # Aggregate results
        self.aggregate_results(results)
        
        # Save final dataset
        self.save_filtered_dataset(processing_time)
        
        print(f"Processing complete in {processing_time:.2f} seconds")
        
    def aggregate_results(self, results: List[Dict]):
        """Aggregate results from all workers."""
        print("Aggregating results from workers...")
        
        chunk_offset = 0
        
        for worker_result in results:
            if worker_result is None:
                continue
            
            # Aggregate statistics
            for key in self.stats:
                self.stats[key] += worker_result['stats'][key]
            
            # Aggregate event data
            self.filtered_events.extend(worker_result['filtered_events'])
            
            # Aggregate indices with proper offset
            worker_file_indices = np.array(worker_result['file_indices']) + chunk_offset
            self.file_indices.extend(worker_file_indices.tolist())
            self.row_indices.extend(worker_result['row_indices'])
            
            chunk_offset += len(worker_result['filtered_events'])
    
    def save_filtered_dataset(self, processing_time: float):
        """Save the filtered dataset with metadata."""
        print("Saving filtered dataset...")
        
        # Save single merged H5 file
        output_h5_path = self.output_dir / 'data' / 'filtered_events.h5'
        
        with h5py.File(output_h5_path, 'w') as f:
            # Store feature names as attributes
            f.attrs['hit_features'] = [name.encode() for name in self.hit_features]
            f.attrs['track_features'] = [name.encode() for name in self.track_features]
            
            # Save all events in compound arrays
            if self.filtered_events:
                max_hits = max(len(event['hits_array']) for event in self.filtered_events)
                max_tracks = max(len(event['tracks_array']) for event in self.filtered_events)
                
                # Create compound arrays with proper data types (matching original dataset)
                all_hits = np.full((len(self.filtered_events), max_hits, len(self.hit_features)), np.nan, dtype=np.float32)
                all_tracks = np.full((len(self.filtered_events), max_tracks, len(self.track_features)), np.nan, dtype=np.float32)
                all_event_numbers = np.full((len(self.filtered_events),), -1, dtype=np.int64)
                all_num_hits = np.zeros((len(self.filtered_events),), dtype=np.int16)
                all_num_tracks = np.zeros((len(self.filtered_events),), dtype=np.int16)
                
                for i, event in enumerate(self.filtered_events):
                    hits_len = len(event['hits_array'])
                    tracks_len = len(event['tracks_array'])
                    
                    all_hits[i, :hits_len, :] = event['hits_array']
                    all_tracks[i, :tracks_len, :] = event['tracks_array']
                    all_event_numbers[i] = event['event_number']
                    all_num_hits[i] = hits_len
                    all_num_tracks[i] = tracks_len
                
                # Use row-based chunking for better random access performance, no compression
                chunk_size_hits = (1, max_hits, len(self.hit_features))
                chunk_size_tracks = (1, max_tracks, len(self.track_features))
                
                f.create_dataset('hits', data=all_hits, chunks=chunk_size_hits)
                f.create_dataset('tracks', data=all_tracks, chunks=chunk_size_tracks)
                f.create_dataset('event_numbers', data=all_event_numbers)
                f.create_dataset('num_hits', data=all_num_hits)
                f.create_dataset('num_tracks', data=all_num_tracks)
        
        # Save index arrays for filtered dataset
        # All events are in a single file (index 0) with sequential row indices
        num_filtered_events = len(self.filtered_events)
        filtered_file_indices = np.zeros(num_filtered_events, dtype=np.int16)  # All point to file 0
        filtered_row_indices = np.arange(num_filtered_events, dtype=np.int16)  # Sequential 0,1,2,...
        
        np.save(self.output_dir / 'event_file_indices.npy', filtered_file_indices)
        np.save(self.output_dir / 'event_row_indices.npy', filtered_row_indices)
        
        # Create and save metadata
        self.save_metadata(processing_time)
        
        # Print final statistics
        self.print_final_statistics()
    
    def save_metadata(self, processing_time: float):
        """Save metadata for the filtered dataset."""
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Calculate statistics
        hit_filter_pass_rate = (self.stats['events_passed_hit_filter'] / 
                               max(1, self.stats['total_events_processed']) * 100)
        max_tracks_fail_rate = (self.stats['events_failed_max_tracks'] / 
                               max(1, self.stats['total_events_processed']) * 100)
        max_hits_fail_rate = (self.stats['events_failed_max_hits'] / 
                              max(1, self.stats['total_events_processed']) * 100)
        final_pass_rate = (self.stats['events_final_output'] / 
                          max(1, self.stats['total_events_processed']) * 100)
        
        hit_reduction_rate = (1 - self.stats['total_hits_after'] / 
                             max(1, self.stats['total_hits_before'])) * 100
        
        # Create new metadata
        filtered_metadata = {
            'hit_features': self.hit_features,
            'track_features': self.track_features,
            'filtering_summary': {
                'total_events_processed': int(self.stats['total_events_processed']),
                'events_passed_hit_filter': int(self.stats['events_passed_hit_filter']),
                'events_failed_no_hits_after_filter': int(self.stats['events_failed_no_hits_after_filter']),
                'events_failed_eval_data_missing': int(self.stats['events_failed_eval_data_missing']),
                'events_failed_data_loading': int(self.stats['events_failed_data_loading']),
                'events_failed_max_tracks': int(self.stats['events_failed_max_tracks']),
                'events_failed_min_tracks': int(self.stats['events_failed_min_tracks']),
                'events_failed_max_hits': int(self.stats['events_failed_max_hits']),
                'events_failed_min_hits': int(self.stats['events_failed_min_hits']),
                'events_failed_track_filtering': int(self.stats['events_failed_track_filtering']),
                'events_final_output': int(self.stats['events_final_output']),
                'excluded_tracks_count': int(self.stats['excluded_tracks_count']),
                # Detailed track filtering statistics
                'tracks_excluded_pt': int(self.stats['tracks_excluded_pt']),
                'tracks_excluded_eta': int(self.stats['tracks_excluded_eta']),
                'tracks_excluded_hits': int(self.stats['tracks_excluded_hits']),
                'tracks_excluded_no_true_hits': int(self.stats['tracks_excluded_no_true_hits']),
                'hit_filter_pass_rate_percent': float(hit_filter_pass_rate),
                'max_tracks_fail_rate_percent': float(max_tracks_fail_rate),
                'max_hits_fail_rate_percent': float(max_hits_fail_rate),
                'final_pass_rate_percent': float(final_pass_rate),
                'total_hits_before': int(self.stats['total_hits_before']),
                'total_hits_after': int(self.stats['total_hits_after']),
                'hit_reduction_rate_percent': float(hit_reduction_rate),
                'total_tracks_before': int(self.stats['total_tracks_before']),
                'total_tracks_after': int(self.stats['total_tracks_after']),
                'processing_time_seconds': float(processing_time),
                'num_workers': int(self.num_workers)
            },
            'filtering_parameters': {
                'working_point': float(self.working_point),
                'detection_threshold': float(self.detection_threshold),
                'max_tracks_per_event': int(self.max_tracks_per_event),
                'max_hits_per_event': int(self.max_hits_per_event),
                'disable_track_filtering': bool(self.disable_track_filtering),
                'pt_threshold': float(self.pt_threshold),
                'eta_threshold': float(self.eta_threshold),
                'num_hits_threshold': int(self.num_hits_threshold),
                'eval_file': str(self.eval_file),
                'source_dataset': str(self.input_dir)
            },
            'original_dataset_info': convert_numpy_types(self.original_metadata),
            'event_mapping': {
                'description': 'Filtered events stored in single H5 file',
                'total_events': int(self.stats['events_final_output']),
                'total_chunks': 1,
                'index_files': {
                    'file_indices': 'event_file_indices.npy',
                    'row_indices': 'event_row_indices.npy',
                },
                'chunk_summary': [{
                    'h5_file': 'data/filtered_events.h5',
                    'source_dataset': str(self.input_dir),
                    'event_count': int(self.stats['events_final_output']),
                    'worker_id': 'merged'
                }]
            }
        }
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(filtered_metadata, f, default_flow_style=False, sort_keys=False)
        
        print(f"Filtered dataset metadata saved to: {metadata_path}")
    
    def print_final_statistics(self):
        """Print comprehensive statistics about the filtering process."""
        print(f"\n{'='*80}")
        print(f"DATASET FILTERING SUMMARY")
        print(f"{'='*80}")
        print(f"Original dataset: {self.input_dir.name}")
        print(f"Working point: {self.working_point}")
        print(f"Detection threshold: {self.detection_threshold:.6f}")
        print(f"Max tracks per event: {self.max_tracks_per_event}")
        print(f"Max hits per event: {self.max_hits_per_event}")
        if not self.disable_track_filtering:
            print(f"Track filtering enabled:")
            print(f"  pT threshold: {self.pt_threshold} GeV")
            print(f"  |eta| threshold: {self.eta_threshold}")
            print(f"  Min hits per track: {self.num_hits_threshold}")
        else:
            print(f"Track filtering: DISABLED")
        print(f"")
        print(f"EVENT STATISTICS:")
        print(f"  Total events processed: {self.stats['total_events_processed']:,}")
        print(f"  Events passed hit filter: {self.stats['events_passed_hit_filter']:,} "
              f"({self.stats['events_passed_hit_filter']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed - no hits after filter: {self.stats['events_failed_no_hits_after_filter']:,} "
              f"({self.stats['events_failed_no_hits_after_filter']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed - eval data missing: {self.stats['events_failed_eval_data_missing']:,} "
              f"({self.stats['events_failed_eval_data_missing']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed - data loading error: {self.stats['events_failed_data_loading']:,} "
              f"({self.stats['events_failed_data_loading']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed max tracks cut: {self.stats['events_failed_max_tracks']:,} "
              f"({self.stats['events_failed_max_tracks']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed min tracks cut: {self.stats['events_failed_min_tracks']:,} "
              f"({self.stats['events_failed_min_tracks']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed max hits cut: {self.stats['events_failed_max_hits']:,} "
              f"({self.stats['events_failed_max_hits']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Events failed min hits cut: {self.stats['events_failed_min_hits']:,} "
              f"({self.stats['events_failed_min_hits']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        if not self.disable_track_filtering:
            print(f"  Events failed track filtering: {self.stats['events_failed_track_filtering']:,} "
                  f"({self.stats['events_failed_track_filtering']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"  Final events output: {self.stats['events_final_output']:,} "
              f"({self.stats['events_final_output']/max(1,self.stats['total_events_processed'])*100:.2f}%)")
        print(f"")
        print(f"HIT/TRACK STATISTICS:")
        print(f"  Total hits before: {self.stats['total_hits_before']:,}")
        print(f"  Total hits after: {self.stats['total_hits_after']:,}")
        print(f"  Hit reduction: {(1-self.stats['total_hits_after']/max(1,self.stats['total_hits_before']))*100:.2f}%")
        print(f"  Total tracks before: {self.stats['total_tracks_before']:,}")
        print(f"  Total tracks after: {self.stats['total_tracks_after']:,}")
        print(f"  Track reduction: {(1-self.stats['total_tracks_after']/max(1,self.stats['total_tracks_before']))*100:.2f}%")
        if not self.disable_track_filtering:
            print(f"  Tracks excluded by filtering: {self.stats['excluded_tracks_count']:,}")
            total_tracks = self.stats['total_tracks_before']
            if total_tracks > 0:
                print(f"    - Excluded due to pT < {self.pt_threshold} GeV: {self.stats['tracks_excluded_pt']:,} ({self.stats['tracks_excluded_pt']/total_tracks*100:.2f}%)")
                print(f"    - Excluded due to |eta| > {self.eta_threshold}: {self.stats['tracks_excluded_eta']:,} ({self.stats['tracks_excluded_eta']/total_tracks*100:.2f}%)")
                print(f"    - Excluded due to < {self.num_hits_threshold} hits: {self.stats['tracks_excluded_hits']:,} ({self.stats['tracks_excluded_hits']/total_tracks*100:.2f}%)")
                print(f"    - Excluded due to no true hits after hit filtering: {self.stats['tracks_excluded_no_true_hits']:,} ({self.stats['tracks_excluded_no_true_hits']/total_tracks*100:.2f}%)")
        print(f"")
        print(f"OUTPUT SUMMARY:")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")


def process_worker_events(args: Tuple) -> Dict:
    """Worker function to process a range of events."""
    (worker_id, start_idx, end_idx, input_dir, eval_file, output_dir,
     detection_threshold, max_tracks_per_event, max_hits_per_event,
     hit_features, track_features, chunk_summary, disable_track_filtering,
     pt_threshold, eta_threshold, num_hits_threshold) = args
    
    print(f"Worker {worker_id}: Processing events {start_idx} to {end_idx-1}")
    
    # Load indices for this worker's range
    input_path = Path(input_dir)
    file_indices = np.load(input_path / 'event_file_indices.npy')[start_idx:end_idx]
    row_indices = np.load(input_path / 'event_row_indices.npy')[start_idx:end_idx]
    
    # Initialize worker statistics
    worker_stats = {
        'total_events_processed': 0,
        'events_passed_hit_filter': 0,
        'events_failed_no_hits_after_filter': 0,
        'events_failed_max_tracks': 0,
        'events_failed_min_tracks': 0,
        'events_failed_max_hits': 0,
        'events_failed_min_hits': 0,
        'events_failed_eval_data_missing': 0,
        'events_failed_data_loading': 0,
        'events_failed_track_filtering': 0,
        'events_final_output': 0,
        'total_hits_before': 0,
        'total_hits_after': 0,
        'total_tracks_before': 0,
        'total_tracks_after': 0,
        'excluded_tracks_count': 0,
        # Detailed track filtering statistics
        'tracks_excluded_pt': 0,
        'tracks_excluded_eta': 0,
        'tracks_excluded_hits': 0,
        'tracks_excluded_no_true_hits': 0
    }
    
    filtered_events = []
    worker_file_indices = []
    worker_row_indices = []
    
    # Open evaluation file once
    with h5py.File(eval_file, 'r') as eval_f:
        
        for local_idx, (file_idx, row_idx) in enumerate(zip(file_indices, row_indices)):
            global_event_idx = start_idx + local_idx
            worker_stats['total_events_processed'] += 1
            
            try:
                # Load original event
                chunk_info = chunk_summary[file_idx]
                h5_file_path = input_path / chunk_info['h5_file']
                
                with h5py.File(h5_file_path, 'r') as h5_f:
                    # Load hit and track data
                    hits_array = h5_f['hits'][row_idx]
                    tracks_array = h5_f['tracks'][row_idx]
                    event_number = h5_f['event_numbers'][row_idx]
                    
                    # Convert to dictionary format
                    hits_dict = {}
                    for i, feature_name in enumerate(hit_features):
                        hits_dict[feature_name] = hits_array[:, i]
                    
                    tracks_dict = {}
                    for i, feature_name in enumerate(track_features):
                        tracks_dict[feature_name] = tracks_array[:, i]
                
                # Remove NaN hits (padding)
                valid_hit_mask = ~np.isnan(hits_dict[hit_features[0]])
                for feature in hit_features:
                    hits_dict[feature] = hits_dict[feature][valid_hit_mask]
                
                # Remove NaN tracks (padding)
                valid_track_mask = ~np.isnan(tracks_dict[track_features[0]])
                for feature in track_features:
                    tracks_dict[feature] = tracks_dict[feature][valid_track_mask]
                
                original_num_hits = len(hits_dict[hit_features[0]])
                original_num_tracks = len(tracks_dict[track_features[0]])
                
                worker_stats['total_hits_before'] += original_num_hits
                worker_stats['total_tracks_before'] += original_num_tracks
                
                # Apply hit filter using evaluation predictions
                try:
                    # The event index from the original dataset corresponds to the sample_id
                    # In the evaluation file, events are stored using sample_id as keys
                    sample_id = global_event_idx  # This is the sample_id used in the evaluation
                    
                    # Check if this sample exists in evaluation file
                    if str(sample_id) not in eval_f:
                        print(f"Worker {worker_id}: Warning - Could not find predictions for sample {sample_id}")
                        worker_stats['events_failed_eval_data_missing'] += 1
                        continue
                    
                    # Load logits from evaluation file using the correct path
                    logits = eval_f[f"{sample_id}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)
                    
                    # Apply threshold to get hit filter mask
                    hit_filter_mask = logits >= detection_threshold
                    
                    # Ensure mask length matches hits
                    if len(hit_filter_mask) != original_num_hits:
                        print(f"Worker {worker_id}: Warning - Mask length mismatch for sample {sample_id}: "
                              f"logits={len(logits)}, hits={original_num_hits}")
                        worker_stats['events_failed_eval_data_missing'] += 1
                        continue
                    
                except KeyError as e:
                    print(f"Worker {worker_id}: Warning - Could not load predictions for sample {global_event_idx}: {e}")
                    worker_stats['events_failed_eval_data_missing'] += 1
                    continue
                
                # Apply hit filtering
                for feature in hit_features:
                    hits_dict[feature] = hits_dict[feature][hit_filter_mask]
                
                filtered_num_hits = len(hits_dict[hit_features[0]])
                
                if filtered_num_hits == 0:
                    worker_stats['events_failed_no_hits_after_filter'] += 1
                    continue
                
                worker_stats['events_passed_hit_filter'] += 1
                
                # Apply track-level filtering similar to the preprocessing script
                unique_track_ids = np.unique(hits_dict['spacePoint_truthLink'])
                valid_track_ids = unique_track_ids[unique_track_ids >= 0]
                
                if len(valid_track_ids) == 0:
                    worker_stats['events_failed_no_hits_after_filter'] += 1
                    continue
                
                # ALWAYS filter tracks that have zero hits after hit filtering
                # This ensures consistency between hit and track data
                exclude_tracks = []
                
                if not disable_track_filtering:
                    # Apply track filters based on pT, eta, and hit count
                    for track_idx in valid_track_ids:
                        # Get track index in the tracks array (valid_track_ids contains truthLink values)
                        track_array_idx = int(track_idx)  # truthLink values correspond to track indices
                        
                        # Check if track index is valid
                        if track_array_idx >= len(tracks_dict['truthMuon_pt']):
                            exclude_tracks.append(track_idx)
                            continue
                        
                        track_excluded = False
                        exclude_reasons = []
                        
                        # Check if track has any true hits left after hit filtering
                        true_hits_count = np.sum(hits_dict['spacePoint_truthLink'] == track_idx)
                        if true_hits_count == 0:
                            exclude_reasons.append('no_true_hits')
                            worker_stats['tracks_excluded_no_true_hits'] += 1
                            track_excluded = True
                        
                        # Check pT threshold
                        if tracks_dict['truthMuon_pt'][track_array_idx] < pt_threshold:
                            exclude_reasons.append('pt')
                            worker_stats['tracks_excluded_pt'] += 1
                            track_excluded = True
                        
                        # Check eta threshold
                        if abs(tracks_dict['truthMuon_eta'][track_array_idx]) > eta_threshold:
                            exclude_reasons.append('eta')
                            worker_stats['tracks_excluded_eta'] += 1
                            track_excluded = True
                        
                        # Check minimum hits threshold (only if track has true hits)
                        if true_hits_count > 0 and true_hits_count < num_hits_threshold:
                            exclude_reasons.append('hits')
                            worker_stats['tracks_excluded_hits'] += 1
                            track_excluded = True
                        
                        if track_excluded:
                            exclude_tracks.append(track_idx)
                            worker_stats['excluded_tracks_count'] += 1
                else:
                    # Even when track filtering is disabled, we must remove tracks with no hits
                    for track_idx in valid_track_ids:
                        # Get track index in the tracks array
                        track_array_idx = int(track_idx)
                        
                        # Check if track index is valid
                        if track_array_idx >= len(tracks_dict['truthMuon_pt']):
                            exclude_tracks.append(track_idx)
                            continue
                            
                        # Check if track has any true hits left after hit filtering
                        true_hits_count = np.sum(hits_dict['spacePoint_truthLink'] == track_idx)
                        if true_hits_count == 0:
                            worker_stats['tracks_excluded_no_true_hits'] += 1
                            exclude_tracks.append(track_idx)
                            worker_stats['excluded_tracks_count'] += 1
                
                remaining_tracks = np.setdiff1d(valid_track_ids, exclude_tracks)
                
                if len(remaining_tracks) == 0:
                    worker_stats['events_failed_track_filtering'] += 1
                    continue
                
                # Filter hits to only keep those from remaining tracks
                hit2track_mask = np.isin(hits_dict['spacePoint_truthLink'], remaining_tracks)
                modified_truth_link = hits_dict['spacePoint_truthLink'].copy()
                modified_truth_link[~hit2track_mask] = -1
                
                # Apply the mask to all hit features
                hits_dict["spacePoint_truthLink"] = modified_truth_link
                
                # Filter tracks to only keep remaining tracks
                track_mask = np.isin(np.arange(len(tracks_dict['truthMuon_pt'])), remaining_tracks.astype(int))
                for feature in track_features:
                    tracks_dict[feature] = tracks_dict[feature][track_mask]
                
                valid_track_ids = remaining_tracks
                
                # Update counts after track filtering
                filtered_num_hits = len(hits_dict[hit_features[0]])
                filtered_num_tracks = len(valid_track_ids)
                
                # Apply max tracks cut
                if filtered_num_tracks > max_tracks_per_event:
                    worker_stats['events_failed_max_tracks'] += 1
                    continue
                # Apply min tracks cut
                if filtered_num_tracks < 1:
                    worker_stats['events_failed_min_tracks'] += 1
                    continue 
                
                # Apply max hits cut
                if filtered_num_hits > max_hits_per_event:
                    worker_stats['events_failed_max_hits'] += 1
                    continue

                # Convert back to arrays (tracks have already been filtered above)
                filtered_hits_array = np.column_stack([hits_dict[feature] for feature in hit_features])
                filtered_tracks_array = np.column_stack([tracks_dict[feature] for feature in track_features])
                
                # Store filtered event
                filtered_events.append({
                    'hits_array': filtered_hits_array,
                    'tracks_array': filtered_tracks_array,
                    'event_number': event_number
                })
                
                worker_file_indices.append(len(filtered_events) - 1)  # Points to position in filtered_events
                worker_row_indices.append(0)  # Always 0 since we have one event per "file"
                
                worker_stats['events_final_output'] += 1
                worker_stats['total_hits_after'] += filtered_num_hits
                worker_stats['total_tracks_after'] += filtered_num_tracks
                
            except Exception as e:
                print(f"Worker {worker_id}: Error processing event {global_event_idx}: {e}")
                worker_stats['events_failed_data_loading'] += 1
                continue
    
    print(f"Worker {worker_id}: Completed. Processed {worker_stats['total_events_processed']} events, "
          f"output {worker_stats['events_final_output']} events")
    
    return {
        'stats': worker_stats,
        'filtered_events': filtered_events,
        'file_indices': worker_file_indices,
        'row_indices': worker_row_indices
    }


def generate_output_dir_name(input_dir: str, working_point: float, 
                           max_tracks: int, max_hits: int) -> str:
    """Generate output directory name with filtering parameters in the same parent directory as input."""
    input_path = Path(input_dir)
    input_name = input_path.name
    parent_dir = input_path.parent
    
    # Create descriptive suffix
    wp_str = f"wp{working_point:.3f}".replace('.', '')
    tracks_str = f"maxtrk{max_tracks}"
    hits_str = f"maxhit{max_hits}"
    
    # Create the full path in the same parent directory as the input
    filtered_name = f"{input_name}_filtered_{wp_str}_{tracks_str}_{hits_str}"
    return str(parent_dir / filtered_name)


def main():
    parser = argparse.ArgumentParser(
        description="Filter ATLAS muon dataset using hit filter predictions and cuts"
    )
    
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Input directory with preprocessed dataset")
    parser.add_argument("--eval_file", "-e", type=str, required=True,
                       help="Path to evaluation HDF5 file with hit filter predictions")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="Output directory (auto-generated if not provided)")
    parser.add_argument("--working_point", "-wp", type=float, default=0.99,
                       help="Working point efficiency for hit filter (default: 0.99)")
    parser.add_argument("--detection_threshold", "-dt", type=float, default=None,
                       help="Detection threshold (calculated from working_point if not provided)")
    parser.add_argument("--max_tracks_per_event", "-mt", type=int, default=2,
                       help="Maximum number of tracks per event (default: 3)")
    parser.add_argument("--max_hits_per_event", "-mh", type=int, default=600,
                       help="Maximum number of hits per event after filtering (default: 500)")
    parser.add_argument("--max_events", "-me", type=int, default=-1,
                       help="Maximum number of events to process (default: -1 for all events)")
    parser.add_argument("--num_workers", "-w", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--disable-track-filtering", action="store_true", default=False,
                       help="Disable track filtering based on pt, eta, and hit count")
    parser.add_argument("--pt-threshold", type=float, default=5.0,
                       help="Minimum pT threshold for tracks (GeV, default: 5.0)")
    parser.add_argument("--eta-threshold", type=float, default=2.7,
                       help="Maximum |eta| threshold for tracks (default: 2.7)")
    parser.add_argument("--num-hits-threshold", type=int, default=3,
                       help="Minimum number of hits per track (default: 3)")
    
    args = parser.parse_args()
    
    # Generate output directory name if not provided
    if args.output_dir is None:
        args.output_dir = generate_output_dir_name(
            args.input_dir, args.working_point, 
            args.max_tracks_per_event, args.max_hits_per_event
        )
    
    print("="*80)
    print("ATLAS MUON DATASET FILTERING")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Evaluation file: {args.eval_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Working point: {args.working_point}")
    print(f"Detection threshold: {args.detection_threshold if args.detection_threshold else 'Auto-calculate'}")
    print(f"Max tracks per event: {args.max_tracks_per_event}")
    print(f"Max hits per event: {args.max_hits_per_event}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Number of workers: {args.num_workers if args.num_workers else 'Auto'}")
    print(f"Disable track filtering: {args.disable_track_filtering}")
    if not args.disable_track_filtering:
        print(f"pT threshold: {args.pt_threshold} GeV")
        print(f"Eta threshold: {args.eta_threshold}")
        print(f"Min hits per track: {args.num_hits_threshold}")
    print("="*80)
    
    # Create and run the filter
    filter_processor = HitFilterDatasetReducer(
        input_dir=args.input_dir,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        working_point=args.working_point,
        detection_threshold=args.detection_threshold,
        max_tracks_per_event=args.max_tracks_per_event,
        max_hits_per_event=args.max_hits_per_event,
        max_events=args.max_events,
        num_workers=args.num_workers,
        disable_track_filtering=args.disable_track_filtering,
        pt_threshold=args.pt_threshold,
        eta_threshold=args.eta_threshold,
        num_hits_threshold=args.num_hits_threshold
    )
    
    filter_processor.process_events()


if __name__ == "__main__":
    main()
