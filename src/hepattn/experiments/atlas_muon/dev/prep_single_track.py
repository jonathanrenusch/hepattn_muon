"""Give dir to argparser and split every event into parts each only containing one relevant track. 
1000 tracks are then saved into in one parquet file as different row groups."""

import argparse
import os
import glob
import pandas as pd
import uproot
import numpy as np
from typing import List, Dict, Any


def process_event(event_data: Dict[str, Any], event_idx: int) -> List[Dict[str, Any]]:
    """
    Process a single event and split it into individual tracks.
    
    Args:
        event_data: Dictionary containing all branch data for the event
        event_idx: Event index for tracking
    
    Returns:
        List of dictionaries, each representing a single track
    """
    # TODO: Implement your track splitting logic here
    # This is where you'll process the event data and create individual tracks
    
    tracks = []
    
    # Example structure - you'll need to implement the actual logic
    # Based on truthMuon data and spacePoint associations
    num_truth_muons = len(event_data.get('truthMuon_pt', []))
    
    for muon_idx in range(num_truth_muons):
        # Apply thresholds if needed
        pt = event_data['truthMuon_pt'][muon_idx] if 'truthMuon_pt' in event_data else 0
        eta = abs(event_data['truthMuon_eta'][muon_idx]) if 'truthMuon_eta' in event_data else 0
        
        # TODO: Add your threshold checks here
        # if pt < pt_threshold or eta > eta_threshold:
        #     continue
        
        # TODO: Associate spacePoints with this muon using truthLink
        # TODO: Apply minimum hits threshold
        
        track = {
            'event_number': event_data.get('eventNumber', 0),
            'run_number': event_data.get('runNumber', 0),
            'muon_idx': muon_idx,
            'event_idx': event_idx,
            # Add all relevant data for this track
            'truth_pt': pt,
            'truth_eta': event_data['truthMuon_eta'][muon_idx] if 'truthMuon_eta' in event_data else 0,
            'truth_phi': event_data['truthMuon_phi'][muon_idx] if 'truthMuon_phi' in event_data else 0,
            'truth_e': event_data['truthMuon_e'][muon_idx] if 'truthMuon_e' in event_data else 0,
            'truth_q': event_data['truthMuon_q'][muon_idx] if 'truthMuon_q' in event_data else 0,
            # TODO: Add associated spacePoint data
        }
        
        tracks.append(track)
    
    return tracks


def process_root_file(file_path: str, pt_threshold: float, eta_threshold: float, 
                     num_hits_threshold: int) -> List[Dict[str, Any]]:
    """
    Process a single ROOT file and extract all tracks.
    
    Args:
        file_path: Path to the ROOT file
        pt_threshold: Minimum pT threshold
        eta_threshold: Maximum |eta| threshold  
        num_hits_threshold: Minimum number of hits
    
    Returns:
        List of all tracks from this file
    """
    print(f"Processing file: {file_path}")
    
    all_tracks = []
    
    try:
        with uproot.open(file_path) as file:
            # Assuming the tree name - you may need to adjust this
            tree_name = None
            for key in file.keys():
                if 'tree' in key.lower() or 'ntuple' in key.lower():
                    tree_name = key
                    break
            
            if tree_name is None:
                # Take the first available tree
                tree_name = list(file.keys())[0]
            
            tree = file[tree_name]
            
            # Get all branch names
            branch_names = [
                'bcid', 'bucket_index', 'bucket_spacePoints', 'bucket_stationEta',
                'bucket_stationIndex', 'bucket_stationPhi', 'bucket_xMax', 'bucket_xMin',
                'eventNumber', 'lbNumber', 'mcChannelNumber', 'mcEventWeight', 'runNumber',
                'spacePoint_PositionX', 'spacePoint_PositionY', 'spacePoint_PositionZ',
                'spacePoint_channel', 'spacePoint_covXX', 'spacePoint_covXY',
                'spacePoint_covYX', 'spacePoint_covYY', 'spacePoint_driftR',
                'spacePoint_globPosX', 'spacePoint_globPosY', 'spacePoint_globPosZ',
                'spacePoint_layer', 'spacePoint_measEta', 'spacePoint_measPhi',
                'spacePoint_nEtaInUse', 'spacePoint_nPhiInUse', 'spacePoint_secChannel',
                'spacePoint_stationEta', 'spacePoint_stationIndex', 'spacePoint_stationPhi',
                'spacePoint_technology', 'spacePoint_truthLink',
                'truthMuon_e', 'truthMuon_eta', 'truthMuon_phi', 'truthMuon_pt', 'truthMuon_q'
            ]
            
            # Filter branch names to only include those that exist in the tree
            available_branches = [branch for branch in branch_names if branch in tree.keys()]
            
            # Read data in chunks to manage memory
            chunk_size = 1000
            for chunk in tree.iterate(available_branches, step_size=chunk_size):
                for event_idx in range(len(chunk['eventNumber']) if 'eventNumber' in chunk else len(list(chunk.values())[0])):
                    # Extract data for this event
                    event_data = {}
                    for branch in available_branches:
                        if hasattr(chunk[branch], '__len__') and len(chunk[branch]) > event_idx:
                            event_data[branch] = chunk[branch][event_idx]
                    
                    # Process the event and get tracks
                    tracks = process_event(event_data, event_idx)
                    all_tracks.extend(tracks)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    print(f"Extracted {len(all_tracks)} tracks from {file_path}")
    return all_tracks


def save_tracks_to_parquet(tracks: List[Dict[str, Any]], output_dir: str, 
                          file_counter: int, num_tracks_per_file: int):
    """
    Save tracks to parquet file.
    
    Args:
        tracks: List of track dictionaries
        output_dir: Output directory
        file_counter: Current file counter
        num_tracks_per_file: Maximum tracks per file
    """
    if not tracks:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(tracks)
    
    # Save to parquet
    output_file = os.path.join(output_dir, f"tracks_{file_counter:06d}.parquet")
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(tracks)} tracks to {output_file}")


def split_events(input_dir: str, output_dir: str, num_tracks_per_file: int,
                pt_threshold: float, eta_threshold: float, num_hits_threshold: int):
    """
    Main function to split events from ROOT files into individual tracks.
    
    Args:
        input_dir: Directory containing ROOT files
        output_dir: Directory to save parquet files
        num_tracks_per_file: Number of tracks per output file
        pt_threshold: Minimum pT threshold
        eta_threshold: Maximum |eta| threshold
        num_hits_threshold: Minimum number of hits
    """
    # Find all ROOT files
    root_files = glob.glob(os.path.join(input_dir, "*.root"))
    
    if not root_files:
        print(f"No ROOT files found in {input_dir}")
        return
    
    print(f"Found {len(root_files)} ROOT files")
    
    all_tracks = []
    file_counter = 0
    
    for root_file in root_files:
        # Process the ROOT file
        tracks = process_root_file(root_file, pt_threshold, eta_threshold, num_hits_threshold)
        all_tracks.extend(tracks)
        
        # Save tracks when we have enough
        while len(all_tracks) >= num_tracks_per_file:
            tracks_to_save = all_tracks[:num_tracks_per_file]
            all_tracks = all_tracks[num_tracks_per_file:]
            
            save_tracks_to_parquet(tracks_to_save, output_dir, file_counter, num_tracks_per_file)
            file_counter += 1
    
    # Save remaining tracks
    if all_tracks:
        save_tracks_to_parquet(all_tracks, output_dir, file_counter, num_tracks_per_file)
    
    print(f"Processing complete. Saved tracks to {output_dir}")


def main():    
    parser = argparse.ArgumentParser(description="Split ATLAS muon events into individual tracks.")
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
        "--num_tracks_per_file",
        type=int,
        default=1000,
        help="Number of tracks to save in each parquet file as different row groups."
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
    
    split_events(args.input_dir, args.output_dir, args.num_tracks_per_file,
                args.pt_threshold, args.eta_threshold, args.num_hits_threshold)


if __name__ == "__main__":
    main()