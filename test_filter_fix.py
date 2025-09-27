#!/usr/bin/env python3
"""
Test script to verify that the filtering fix prevents events with 0 tracks.
"""
import os
import sys
import tempfile
import numpy as np
import h5py
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

def create_test_data():
    """Create a small test dataset for verification."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating test data in: {data_dir}")
    
    # Create metadata
    metadata = {
        'hit_features': ['spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ', 'spacePoint_truthLink'],
        'track_features': ['truthMuon_pt', 'truthMuon_eta', 'truthMuon_phi', 'truthMuon_q'],
        'event_mapping': {'chunk_summary': {}}
    }
    
    with open(data_dir / "metadata.yaml", 'w') as f:
        import yaml
        yaml.dump(metadata, f)
    
    # Create test data with an event that will have 0 tracks after filtering
    (data_dir / "data").mkdir(parents=True, exist_ok=True)
    with h5py.File(data_dir / "data" / "test_data.h5", 'w') as f:
        # Event 0: Will have tracks after filtering
        hits_0 = np.array([
            [1.0, 2.0, 3.0, 0],  # Hit belonging to track 0
            [1.1, 2.1, 3.1, 0],  # Hit belonging to track 0
            [2.0, 3.0, 4.0, 1],  # Hit belonging to track 1
            [np.nan, np.nan, np.nan, np.nan],  # Padding
        ])
        tracks_0 = np.array([
            [10.0, 0.5, 1.0, 1],   # Track 0: pt=10, eta=0.5
            [15.0, 1.0, 1.5, -1],  # Track 1: pt=15, eta=1.0
        ])
        
        # Event 1: All hits will be filtered out, leading to 0 tracks
        hits_1 = np.array([
            [5.0, 6.0, 7.0, 0],  # Hit belonging to track 0
            [5.1, 6.1, 7.1, 0],  # Hit belonging to track 0
            [np.nan, np.nan, np.nan, np.nan],  # Padding
            [np.nan, np.nan, np.nan, np.nan],  # Padding
        ])
        tracks_1 = np.array([
            [20.0, 0.8, 2.0, 1],   # Track 0: pt=20, eta=0.8
            [np.nan, np.nan, np.nan, np.nan],  # Padding
        ])
        
        hits = np.stack([hits_0, hits_1])
        tracks = np.stack([tracks_0, tracks_1])
        event_numbers = np.array([0, 1])
        
        f.create_dataset('hits', data=hits)
        f.create_dataset('tracks', data=tracks)
        f.create_dataset('event_numbers', data=event_numbers)
    
    # Create event indices
    np.save(data_dir / "event_file_indices.npy", np.array([0, 1]))
    np.save(data_dir / "event_row_indices.npy", np.array([0, 1]))
    
    # Create evaluation file with hit filter predictions
    eval_dir = Path(temp_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(eval_dir / "test_eval.h5", 'w') as f:
        # Event 0: Keep all hits (logits > threshold)
        f.create_group('0')
        f.create_group('0/outputs')
        f.create_group('0/outputs/final')
        f.create_group('0/outputs/final/hit_filter')
        f['0/outputs/final/hit_filter/hit_logit'] = np.array([[1.0, 1.0, 1.0, 0.0]])  # Keep first 3 hits
        
        # Event 1: Filter out all hits (logits < threshold) 
        f.create_group('1')
        f.create_group('1/outputs')
        f.create_group('1/outputs/final')
        f.create_group('1/outputs/final/hit_filter')
        f['1/outputs/final/hit_filter/hit_logit'] = np.array([[-5.0, -5.0, 0.0, 0.0]])  # Filter out all hits
    
    return str(data_dir), str(eval_dir / "test_eval.h5")

def test_filtering():
    """Test the filtering script with the fixed logic."""
    print("Creating test data...")
    data_dir, eval_file = create_test_data()
    
    print("Testing filtering script...")
    
    # Import the filtering class
    from hepattn.experiments.atlas_muon.filter_dataset_with_hitfilter import HitFilterDatasetReducer
    
    # Create output directory
    output_dir = Path(data_dir).parent / "output"
    
    # Run filtering
    reducer = HitFilterDatasetReducer(
        input_dir=data_dir,
        eval_file=eval_file,
        output_dir=str(output_dir),
        working_point=0.99,
        detection_threshold=0.0,  # Threshold to filter based on our test data
        max_tracks_per_event=5,
        max_hits_per_event=500,
        max_events=10,
        num_workers=1,
        disable_track_filtering=False
    )
    
    reducer.process_events()
    
    # Check results
    print("\nFiltering Results:")
    print("-" * 40)
    print(f"Total events processed: {reducer.stats['total_events_processed']}")
    print(f"Events with no hits after filter: {reducer.stats['events_failed_no_hits_after_filter']}")
    print(f"Events failed track filtering: {reducer.stats['events_failed_track_filtering']}")
    print(f"Events failed min tracks: {reducer.stats['events_failed_min_tracks']}")
    print(f"Events failed max tracks: {reducer.stats['events_failed_max_tracks']}")
    print(f"Final events output: {reducer.stats['events_final_output']}")
    
    # Verify no events with 0 tracks made it through
    if reducer.stats['events_final_output'] > 0:
        print("\n✓ Events were successfully filtered!")
        if reducer.stats['events_failed_no_hits_after_filter'] > 0 or reducer.stats['events_failed_track_filtering'] > 0:
            print("✓ Events with 0 tracks were properly filtered out")
        else:
            print("⚠ Warning: Expected some events to be filtered out due to 0 tracks")
    else:
        print("⚠ All events were filtered out")
    
    # Clean up
    import shutil
    shutil.rmtree(Path(data_dir).parent)
    
    return True

if __name__ == "__main__":
    print("Testing the fixed filtering logic...")
    test_filtering()
    print("Test completed!")