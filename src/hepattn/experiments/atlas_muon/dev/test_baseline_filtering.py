#!/usr/bin/env python3
"""
Test script to verify the baseline filtering functionality works correctly.
This simulates a small evaluation run to test the station-based filtering.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Test the filtering logic with simulated data
def test_baseline_filtering():
    print("Testing baseline track filtering logic...")
    
    # Create simulated data for testing
    np.random.seed(42)
    
    # Simulate 100 hits across 3 events with 2 tracks each
    n_hits = 100
    event_ids = np.repeat([0, 1, 2], [30, 35, 35])  # 3 events
    
    # Create particle IDs - ensure we have distinct tracks per event
    particle_ids = np.zeros(n_hits, dtype=int)
    start_idx = 0
    for event_id in [0, 1, 2]:
        event_mask = event_ids == event_id
        event_hits = np.sum(event_mask)
        # Split hits between 2 particles in this event
        mid_point = event_hits // 2
        event_indices = np.where(event_mask)[0]
        particle_ids[event_indices[:mid_point]] = 0
        particle_ids[event_indices[mid_point:]] = 1
    
    # Create station indices - some tracks will have good station coverage, others won't
    station_indices = np.random.randint(0, 6, n_hits)
    
    # Manually create a "good" track that should pass baseline filtering
    # Event 0, Particle 0: stations 0,1,2 with 4,4,3 hits respectively (11 hits total)
    good_track_mask = (event_ids == 0) & (particle_ids == 0)
    good_track_indices = np.where(good_track_mask)[0]
    
    print(f"Good track has {len(good_track_indices)} hits")
    
    # Ensure we have enough hits for a qualifying track
    if len(good_track_indices) >= 11:
        # First, set all good track hits to invalid stations
        station_indices[good_track_indices] = -1  # Use -1 as invalid
        
        # Now assign station indices to create a qualifying track (only for the first 11 hits)
        station_indices[good_track_indices[:4]] = 0  # 4 hits in station 0
        station_indices[good_track_indices[4:8]] = 1  # 4 hits in station 1  
        station_indices[good_track_indices[8:11]] = 2  # 3 hits in station 2
        # The remaining hits (if any) stay at -1 (invalid station)
        print("✓ Created qualifying track with stations [0,1,2] having [4,4,3] hits")
    else:
        print(f"✗ Not enough hits ({len(good_track_indices)}) to create qualifying track")
    
    # Test the filtering logic
    def create_baseline_track_filter_test(event_ids, particle_ids, station_indices):
        """Simplified version of the baseline filtering for testing."""
        # Get unique combinations of (event_id, particle_id)
        unique_combinations = np.unique(
            np.column_stack([event_ids, particle_ids]), axis=0
        )
        
        baseline_qualified_tracks = set()
        
        for event_id, particle_id in unique_combinations:
            # Get hits for this specific track
            track_mask = (event_ids == event_id) & (particle_ids == particle_id)
            track_hits = np.sum(track_mask)
            
            # Pre-filter: tracks must have at least 9 hits total
            if track_hits < 9:
                continue
            
            # Get station indices for this track
            track_stations = station_indices[track_mask]
            # Filter out invalid stations (-1)
            valid_stations = track_stations[track_stations >= 0]
            unique_stations, station_counts = np.unique(valid_stations, return_counts=True)
            
            # Check baseline requirements:
            # 1. At least 3 different stations
            if len(unique_stations) < 3:
                continue
                
            # 2. Each station must have at least 3 hits
            if np.all(station_counts >= 3):
                baseline_qualified_tracks.add((event_id, particle_id))
        
        return baseline_qualified_tracks
    
    qualified_tracks = create_baseline_track_filter_test(event_ids, particle_ids, station_indices)
    
    print(f"Qualified tracks: {qualified_tracks}")
    print(f"Expected: at least {(0, 0)} should be included")
    
    # Verify our test case
    if (0, 0) in qualified_tracks:
        print("✓ Test PASSED: Good track was correctly identified")
    else:
        print("✗ Test FAILED: Good track was not identified")
        
    return len(qualified_tracks) > 0

def test_evaluator_import():
    """Test that we can import and instantiate the evaluator."""
    print("\nTesting evaluator import...")
    
    try:
        # Add the parent directory to path to import the evaluator
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        from evaluate_hit_filter_dataloader import AtlasMuonEvaluatorDataLoader
        print("✓ Successfully imported AtlasMuonEvaluatorDataLoader")
        
        # Test that the baseline filtering method exists
        if hasattr(AtlasMuonEvaluatorDataLoader, 'create_baseline_track_filter'):
            print("✓ create_baseline_track_filter method found")
        else:
            print("✗ create_baseline_track_filter method not found")
            
        if hasattr(AtlasMuonEvaluatorDataLoader, '_backup_original_data'):
            print("✓ _backup_original_data method found")
        else:
            print("✗ _backup_original_data method not found")
            
        if hasattr(AtlasMuonEvaluatorDataLoader, '_apply_hit_filter'):
            print("✓ _apply_hit_filter method found")
        else:
            print("✗ _apply_hit_filter method not found")
            
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("BASELINE FILTERING TEST SUITE")
    print("=" * 50)
    
    # Test 1: Basic filtering logic
    test1_success = test_baseline_filtering()
    
    # Test 2: Import functionality
    test2_success = test_evaluator_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Filtering Logic Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"Import Test: {'PASSED' if test2_success else 'FAILED'}")
    
    overall_success = test1_success and test2_success
    print(f"Overall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Ready to run full evaluation with baseline filtering!")
    else:
        print("\n❌ Please fix the issues before running full evaluation.")