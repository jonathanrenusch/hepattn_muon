#!/usr/bin/env python3
"""
Filter ATLAS muon dataset based on hit filter predictions and track/hit constraints.
This script applies a working point efficiency to hit filter predictions and filters
events based on maximum track count and hit count after filtering.
"""

import argparse
import os
import sys
import h5py
import yaml
import numpy as np
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import roc_curve
import warnings

warnings.filterwarnings('ignore')

def calculate_threshold_from_working_point(logits: np.ndarray, labels: np.ndarray, working_point: float) -> float:
    """Calculate threshold for given working point efficiency."""
    fpr, tpr, thresholds = roc_curve(labels, logits)
    
    # Find threshold that gives the desired efficiency (recall)
    if not np.any(tpr >= working_point):
        print(f"Warning: Cannot achieve working point {working_point}, using highest achievable: {np.max(tpr):.4f}")
        return thresholds[0]
    
    threshold = thresholds[tpr >= working_point][0]
    actual_efficiency = tpr[tpr >= working_point][0]
    
    print(f"Working point {working_point} achieved with threshold {threshold:.6f} (actual efficiency: {actual_efficiency:.4f})")
    return threshold

def load_eval_predictions(eval_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions and labels from evaluation file."""
    print(f"Loading evaluation predictions from {eval_path}")
    
    all_logits = []
    all_labels = []
    all_event_ids = []
    
    with h5py.File(eval_path, 'r') as f:
        event_keys = [k for k in f.keys() if k.isdigit()]
        
        for event_id in tqdm(event_keys, desc="Loading predictions"):
            try:
                logits = f[f"{event_id}/preds/final/hit_filter/hit_on_valid_particle"][:]
                labels = f[f"{event_id}/targets/hit_on_valid_particle"][:]
                
                all_logits.append(logits.flatten())
                all_labels.append(labels.flatten())
                all_event_ids.extend([int(event_id)] * len(logits.flatten()))
                
            except KeyError as e:
                print(f"Warning: Missing data for event {event_id}: {e}")
                continue
    
    return (np.concatenate(all_logits), 
            np.concatenate(all_labels), 
            np.array(all_event_ids))

def get_event_predictions(eval_path: str, event_id: int) -> np.ndarray:
    """Get hit filter predictions for a specific event."""
    try:
        with h5py.File(eval_path, 'r') as f:
            if str(event_id) not in f:
                return None
            predictions = f[f"{event_id}/preds/final/hit_filter/hit_on_valid_particle"][:]
            return predictions.flatten()
    except Exception as e:
        print(f"Warning: Could not load predictions for event {event_id}: {e}")
        return None

def process_single_file(args: Tuple) -> Dict[str, Any]:
    """Process a single HDF5 file and apply filters."""
    (file_path, output_dir, eval_path, threshold, max_tracks, max_hits_after_filter, 
     hit_features, track_features, worker_id) = args
    
    print(f"Worker {worker_id}: Processing {file_path}")
    
    # Statistics tracking
    stats = {
        'total_events': 0,
        'events_passed_hit_filter': 0,
        'events_failed_track_count': 0,
        'events_failed_hit_count': 0,
        'events_final': 0,
        'total_hits_before': 0,
        'total_hits_after': 0,
        'worker_id': worker_id,
        'processed_events': []
    }
    
    # Storage for filtered events
    filtered_hits = []
    filtered_tracks = []
    filtered_event_ids = []
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get all events in this file
            event_dataset = f['events']
            track_dataset = f['tracks']
            num_events = len(event_dataset)
            
            stats['total_events'] = num_events
            
            for event_idx in range(num_events):
                # Load event data
                event_data = event_dataset[event_idx]
                track_data = track_dataset[event_idx]
                
                # Convert to dictionaries
                hits = {}
                for i, feature in enumerate(hit_features):
                    hits[feature] = event_data[i]
                
                tracks = {}
                for i, feature in enumerate(track_features):
                    tracks[feature] = track_data[i]
                
                # Get original hit count
                original_hit_count = len(hits[hit_features[0]])
                stats['total_hits_before'] += original_hit_count
                
                # Try to get event ID (assume it's stored somewhere or use index)
                event_id = event_idx  # Fallback to index if no event ID available
                
                # Apply hit filter if threshold is provided
                if threshold is not None:
                    predictions = get_event_predictions(eval_path, event_id)
                    if predictions is None:
                        continue  # Skip if no predictions available
                    
                    if len(predictions) != original_hit_count:
                        print(f"Warning: Prediction length {len(predictions)} != hit count {original_hit_count} for event {event_id}")
                        continue
                    
                    # Apply threshold to get hit filter mask
                    hit_filter_mask = predictions >= threshold
                else:
                    # No filtering, keep all hits
                    hit_filter_mask = np.ones(original_hit_count, dtype=bool)
                
                # Filter hits
                filtered_hit_count = np.sum(hit_filter_mask)
                if filtered_hit_count == 0:
                    continue  # Skip events with no hits after filtering
                
                stats['events_passed_hit_filter'] += 1
                stats['total_hits_after'] += filtered_hit_count
                
                # Apply hit filtering to all hit features
                hits_filtered = {}
                for feature in hit_features:
                    hits_filtered[feature] = hits[feature][hit_filter_mask]
                
                # Count tracks after hit filtering
                truth_links = hits_filtered['spacePoint_truthLink']
                unique_tracks = np.unique(truth_links[truth_links >= 0])
                track_count = len(unique_tracks)
                
                # Check track count constraint
                if track_count > max_tracks:
                    stats['events_failed_track_count'] += 1
                    continue
                
                # Check hit count constraint
                if filtered_hit_count > max_hits_after_filter:
                    stats['events_failed_hit_count'] += 1
                    continue
                
                # Event passed all filters
                stats['events_final'] += 1
                
                # Update truth links to be contiguous starting from 0
                track_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracks)}
                track_mapping[-1] = -1  # Keep noise hits as -1
                
                hits_filtered['spacePoint_truthLink'] = np.array([
                    track_mapping.get(link, -1) for link in hits_filtered['spacePoint_truthLink']
                ])
                
                # Filter track data to only include valid tracks
                tracks_filtered = {}
                if len(unique_tracks) > 0:
                    for feature in track_features:
                        if feature in tracks:
                            tracks_filtered[feature] = tracks[feature][:len(unique_tracks)]
                        else:
                            # Create dummy data if feature missing
                            tracks_filtered[feature] = np.zeros(len(unique_tracks))
                
                filtered_hits.append(hits_filtered)
                filtered_tracks.append(tracks_filtered)
                filtered_event_ids.append(event_id)
                stats['processed_events'].append(event_id)
    
    except Exception as e:
        print(f"Worker {worker_id}: Error processing {file_path}: {e}")
        return stats
    
    # Save filtered data if we have any events
    if len(filtered_hits) > 0:
        output_file = output_dir / f"filtered_data_worker_{worker_id}.h5"
        save_filtered_data(filtered_hits, filtered_tracks, filtered_event_ids, 
                          output_file, hit_features, track_features)
        stats['output_file'] = str(output_file)
    
    print(f"Worker {worker_id}: Processed {stats['total_events']} events, "
          f"kept {stats['events_final']} events")
    
    return stats

def save_filtered_data(hits_list: List[Dict], tracks_list: List[Dict], 
                      event_ids: List[int], output_file: Path,
                      hit_features: List[str], track_features: List[str]):
    """Save filtered data to HDF5 file."""
    print(f"Saving {len(hits_list)} events to {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Prepare data arrays
        max_hits = max(len(hits[hit_features[0]]) for hits in hits_list) if hits_list else 0
        max_tracks = max(len(tracks[track_features[0]]) for tracks in tracks_list) if tracks_list else 0
        
        # Create datasets
        events_data = []
        tracks_data = []
        
        for i, (hits, tracks) in enumerate(zip(hits_list, tracks_list)):
            # Pad hits to max_hits
            event_hits = []
            for feature in hit_features:
                hit_array = hits[feature]
                if len(hit_array) < max_hits:
                    # Pad with appropriate values
                    if feature == 'spacePoint_truthLink':
                        pad_value = -1
                    else:
                        pad_value = 0
                    padded = np.pad(hit_array, (0, max_hits - len(hit_array)), 
                                  constant_values=pad_value)
                else:
                    padded = hit_array[:max_hits]
                event_hits.append(padded)
            events_data.append(event_hits)
            
            # Pad tracks to max_tracks
            event_tracks = []
            for feature in track_features:
                if feature in tracks and len(tracks[feature]) > 0:
                    track_array = tracks[feature]
                    if len(track_array) < max_tracks:
                        padded = np.pad(track_array, (0, max_tracks - len(track_array)), 
                                      constant_values=0)
                    else:
                        padded = track_array[:max_tracks]
                else:
                    padded = np.zeros(max_tracks)
                event_tracks.append(padded)
            tracks_data.append(event_tracks)
        
        # Save to HDF5
        if events_data:
            f.create_dataset('events', data=np.array(events_data))
        if tracks_data:
            f.create_dataset('tracks', data=np.array(tracks_data))
        
        # Save event IDs
        f.create_dataset('event_ids', data=np.array(event_ids))

def merge_worker_files(output_dir: Path, worker_files: List[str], 
                      hit_features: List[str], track_features: List[str]) -> str:
    """Merge all worker output files into a single file."""
    print("Merging worker output files...")
    
    final_output = output_dir / "filtered_data.h5"
    
    all_events = []
    all_tracks = []
    all_event_ids = []
    
    for worker_file in worker_files:
        if not os.path.exists(worker_file):
            continue
            
        with h5py.File(worker_file, 'r') as f:
            if 'events' in f:
                events = f['events'][:]
                tracks = f['tracks'][:]
                event_ids = f['event_ids'][:]
                
                for i in range(len(events)):
                    all_events.append(events[i])
                    all_tracks.append(tracks[i])
                    all_event_ids.append(event_ids[i])
        
        # Clean up worker file
        os.remove(worker_file)
    
    # Save merged data
    if all_events:
        with h5py.File(final_output, 'w') as f:
            f.create_dataset('events', data=np.array(all_events))
            f.create_dataset('tracks', data=np.array(all_tracks))
            f.create_dataset('event_ids', data=np.array(all_event_ids))
    
    print(f"Merged {len(all_events)} events into {final_output}")
    return str(final_output)

def create_filtered_dataset(input_dir: str, eval_path: str, output_base_dir: str,
                           working_point: float = 0.99, threshold: float = None,
                           max_tracks: int = 3, max_hits_after_filter: int = 500,
                           num_workers: int = None) -> str:
    """Main function to create filtered dataset."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Load metadata
    metadata_path = input_path / 'metadata.yaml'
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    hit_features = metadata['hit_features']
    track_features = metadata['track_features']
    
    # Calculate threshold if not provided
    if threshold is None:
        print("Calculating threshold from working point...")
        all_logits, all_labels, _ = load_eval_predictions(eval_path)
        threshold = calculate_threshold_from_working_point(all_logits, all_labels, working_point)
    
    # Create output directory with descriptive name
    original_name = input_path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if threshold is not None:
        output_dir_name = f"{original_name}_filtered_wp{working_point:.3f}_maxtracks{max_tracks}_maxhits{max_hits_after_filter}_{timestamp}"
    else:
        output_dir_name = f"{original_name}_filtered_nofilter_maxtracks{max_tracks}_maxhits{max_hits_after_filter}_{timestamp}"
    
    output_dir = Path(output_base_dir) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data subdirectory
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}")
    print(f"Max tracks: {max_tracks}")
    print(f"Max hits after filter: {max_hits_after_filter}")
    
    # Get all HDF5 files
    data_path = input_path / 'data'
    h5_files = list(data_path.glob('*.h5'))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_path}")
    
    print(f"Found {len(h5_files)} HDF5 files to process")
    
    # Set up multiprocessing
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(h5_files))
    
    print(f"Using {num_workers} workers")
    
    # Prepare arguments for workers
    worker_args = []
    for i, h5_file in enumerate(h5_files):
        args = (h5_file, data_dir, eval_path, threshold, max_tracks, max_hits_after_filter,
                hit_features, track_features, i)
        worker_args.append(args)
    
    # Process files in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, worker_args),
            total=len(worker_args),
            desc="Processing files"
        ))
    
    # Aggregate statistics
    total_stats = {
        'total_events': 0,
        'events_passed_hit_filter': 0,
        'events_failed_track_count': 0,
        'events_failed_hit_count': 0,
        'events_final': 0,
        'total_hits_before': 0,
        'total_hits_after': 0,
    }
    
    worker_files = []
    for result in results:
        if result:
            for key in total_stats:
                total_stats[key] += result.get(key, 0)
            
            if 'output_file' in result:
                worker_files.append(result['output_file'])
    
    # Merge worker files
    final_data_file = merge_worker_files(data_dir, worker_files, hit_features, track_features)
    
    # Calculate percentages
    if total_stats['total_events'] > 0:
        hit_filter_pass_pct = (total_stats['events_passed_hit_filter'] / total_stats['total_events']) * 100
        track_fail_pct = (total_stats['events_failed_track_count'] / total_stats['total_events']) * 100
        hit_fail_pct = (total_stats['events_failed_hit_count'] / total_stats['total_events']) * 100
        final_pass_pct = (total_stats['events_final'] / total_stats['total_events']) * 100
    else:
        hit_filter_pass_pct = track_fail_pct = hit_fail_pct = final_pass_pct = 0
    
    # Create updated metadata
    filtered_metadata = metadata.copy()
    filtered_metadata.update({
        'original_dataset': str(input_path),
        'filter_timestamp': datetime.now().isoformat(),
        'filter_parameters': {
            'working_point': working_point,
            'threshold': float(threshold) if threshold is not None else None,
            'max_tracks': max_tracks,
            'max_hits_after_filter': max_hits_after_filter,
        },
        'filter_statistics': {
            'total_events_processed': total_stats['total_events'],
            'events_passed_hit_filter': total_stats['events_passed_hit_filter'],
            'events_failed_track_count': total_stats['events_failed_track_count'],
            'events_failed_hit_count': total_stats['events_failed_hit_count'],
            'events_final': total_stats['events_final'],
            'total_hits_before_filter': total_stats['total_hits_before'],
            'total_hits_after_filter': total_stats['total_hits_after'],
            'percentages': {
                'hit_filter_pass_rate': f"{hit_filter_pass_pct:.2f}%",
                'track_count_failure_rate': f"{track_fail_pct:.2f}%",
                'hit_count_failure_rate': f"{hit_fail_pct:.2f}%",
                'final_pass_rate': f"{final_pass_pct:.2f}%",
            }
        },
        'event_mapping': {
            'chunk_summary': [{
                'h5_file': 'data/filtered_data.h5',
                'event_range': {'count': total_stats['events_final']}
            }]
        }
    })
    
    # Save updated metadata
    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(filtered_metadata, f, default_flow_style=False)
    
    # Create simple index arrays for compatibility
    np.save(output_dir / 'event_file_indices.npy', np.zeros(total_stats['events_final'], dtype=int))
    np.save(output_dir / 'event_row_indices.npy', np.arange(total_stats['events_final'], dtype=int))
    
    # Print summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Original dataset: {input_path}")
    print(f"Filtered dataset: {output_dir}")
    print(f"Working point: {working_point}")
    print(f"Threshold: {threshold}")
    print(f"Max tracks: {max_tracks}")
    print(f"Max hits after filter: {max_hits_after_filter}")
    print()
    print(f"Total events processed: {total_stats['total_events']:,}")
    print(f"Events passed hit filter: {total_stats['events_passed_hit_filter']:,} ({hit_filter_pass_pct:.2f}%)")
    print(f"Events failed track count: {total_stats['events_failed_track_count']:,} ({track_fail_pct:.2f}%)")
    print(f"Events failed hit count: {total_stats['events_failed_hit_count']:,} ({hit_fail_pct:.2f}%)")
    print(f"Final events: {total_stats['events_final']:,} ({final_pass_pct:.2f}%)")
    print()
    print(f"Hits before filtering: {total_stats['total_hits_before']:,}")
    print(f"Hits after filtering: {total_stats['total_hits_after']:,}")
    if total_stats['total_hits_before'] > 0:
        hit_reduction = (1 - total_stats['total_hits_after'] / total_stats['total_hits_before']) * 100
        print(f"Hit reduction: {hit_reduction:.2f}%")
    print("="*60)
    
    return str(output_dir)

def main():
    parser = argparse.ArgumentParser(
        description='Filter ATLAS muon dataset based on hit filter predictions and constraints'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to input dataset directory')
    parser.add_argument('--eval_path', type=str, required=True,
                       help='Path to evaluation HDF5 file with hit filter predictions')
    parser.add_argument('--output_dir', type=str, default='./filtered_datasets',
                       help='Base output directory for filtered dataset')
    parser.add_argument('--working_point', type=float, default=0.99,
                       help='Working point efficiency for hit filter (default: 0.99)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Detection threshold for hit filter (overrides working_point if provided)')
    parser.add_argument('--max_tracks', type=int, default=3,
                       help='Maximum number of tracks per event (default: 3)')
    parser.add_argument('--max_hits_after_filter', type=int, default=500,
                       help='Maximum number of hits per event after filtering (default: 500)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    
    args = parser.parse_args()
    
    try:
        output_path = create_filtered_dataset(
            input_dir=args.input_dir,
            eval_path=args.eval_path,
            output_base_dir=args.output_dir,
            working_point=args.working_point,
            threshold=args.threshold,
            max_tracks=args.max_tracks,
            max_hits_after_filter=args.max_hits_after_filter,
            num_workers=args.num_workers
        )
        
        print(f"\nFiltered dataset created successfully at: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()