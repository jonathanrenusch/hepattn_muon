#!/usr/bin/env python3
"""
Unified ATLAS Muon Tracking Model Evaluation Script

This script provides a comprehensive evaluation of the three-task tracking model:
1. Task 1: Hit-Track Assignment (track_hit_valid)
2. Task 2: Track Validity Classification (track_valid) 
3. Task 3: Regression Outputs (parameter_regression)

Features:
- Single unified data collection for all tasks
- Comparison regions: All tracks, Baseline filtered tracks, Rejected tracks
- Consistent plotting style with hit filter evaluation
- Configuration logging of average efficiencies and purities
- Timestamped output organization
"""

import os
import sys
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import stats
import argparse
from datetime import datetime
import warnings
import yaml
import json
import multiprocessing as mp
from functools import partial

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings('ignore')

# Set matplotlib style to match hit filter evaluation
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.grid': True, 
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'errorbar.capsize': 4
})


def _process_track_chunk_for_baseline(track_chunk, all_event_ids, all_particle_ids, all_particle_pts, 
                                     all_particle_etas, all_station_indices, true_hit_mask):
    """
    Worker function to process a chunk of tracks for baseline filtering.
    Adapted from hit filter evaluation for consistent baseline requirements.
    
    Args:
        track_chunk: List of (event_id, particle_id) tuples to process
        all_event_ids: Array of event IDs for all hits
        all_particle_ids: Array of particle IDs for all hits
        all_particle_pts: Array of particle pT values for all hits
        all_particle_etas: Array of particle eta values for all hits
        all_station_indices: Array of station indices for all hits
        true_hit_mask: Boolean mask for true hits
    
    Returns:
        Dictionary with qualified tracks and statistics for this chunk
    """
    chunk_stats = {
        'total_tracks_checked': 0,
        'tracks_failed_min_hits': 0,
        'tracks_failed_eta_cuts': 0,
        'tracks_failed_pt_cuts': 0,
        'tracks_failed_station_cuts': 0,
        'tracks_passed_all_cuts': 0
    }
    
    qualified_tracks = set()
    
    for event_id, particle_id in track_chunk:
        chunk_stats['total_tracks_checked'] += 1
        
        # Get hits for this specific track
        track_mask = (
            (all_event_ids == event_id) & 
            (all_particle_ids == particle_id) & 
            true_hit_mask
        )
        track_hits = np.sum(track_mask)
        
        # Pre-filter 1: tracks must have at least 9 hits total
        if track_hits < 9:
            chunk_stats['tracks_failed_min_hits'] += 1
            continue
        
        # Get particle kinematic properties for this track
        track_indices = np.where(track_mask)[0]
        if len(track_indices) == 0:
            chunk_stats['tracks_failed_min_hits'] += 1
            continue
            
        # Use the first hit to get particle properties
        first_hit_idx = track_indices[0]
        track_pt = all_particle_pts[first_hit_idx]
        track_eta = all_particle_etas[first_hit_idx]
        
        # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7
        if np.abs(track_eta) < 0.1 or np.abs(track_eta) > 2.7:
            chunk_stats['tracks_failed_eta_cuts'] += 1
            continue
            
        # Pre-filter 3: pt threshold >= 3 GeV
        if track_pt < 3.0:
            chunk_stats['tracks_failed_pt_cuts'] += 1
            continue
        
        # Get station indices for this track
        track_stations = all_station_indices[track_mask]
        unique_stations, station_counts = np.unique(track_stations, return_counts=True)
        
        # Check station requirements:
        # 1. At least 3 different stations
        if len(unique_stations) < 3:
            chunk_stats['tracks_failed_station_cuts'] += 1
            continue
            
        # 2. Each station must have at least 3 hits
        if not np.all(station_counts >= 3):
            chunk_stats['tracks_failed_station_cuts'] += 1
            continue
            
        # Track passed all criteria
        qualified_tracks.add((event_id, particle_id))
        chunk_stats['tracks_passed_all_cuts'] += 1
    
    return {
        'qualified_tracks': qualified_tracks,
        'stats': chunk_stats
    }


class UnifiedTrackingEvaluator:
    """Unified evaluator for all three tracking model tasks with comparison regions."""
    
    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.config_path = config_path
        self.max_events = max_events
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for comparison regions (like hit filter evaluation)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_filtered_dir = self.output_dir / "baseline_filtered_tracks" 
        self.rejected_tracks_dir = self.output_dir / "rejected_tracks"
        
        for dir_path in [self.all_tracks_dir, self.baseline_filtered_dir, self.rejected_tracks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.reset_data_storage()
        
        # Results storage for config logging
        self.evaluation_results = {
            'all_tracks': {},
            'baseline_filtered': {},
            'rejected_tracks': {}
        }
        
        print(f"Unified Tracking Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def reset_data_storage(self):
        """Reset data storage for clean evaluation runs."""
        # Task 1: Hit-Track Assignment data
        self.hit_predictions = []
        self.hit_true_assignments = []
        self.hit_logits = []
        self.hit_track_info = []
        
        # Task 2: Track Validity data
        self.track_predictions = []
        self.track_true_validity = []
        self.track_logits = []
        self.track_info = []
        
        # Task 3: Regression data
        self.reg_predictions = {'eta': [], 'phi': [], 'qpt': []}
        self.reg_truth = {'eta': [], 'phi': [], 'qpt': []}
        self.reg_track_info = []
        
        # Common data for baseline filtering
        self.all_event_ids = []
        self.all_particle_ids = []
        self.all_particle_pts = []
        self.all_particle_etas = []
        self.all_particle_phis = []
        self.all_station_indices = []
        self.all_true_hits = []
    
    def setup_data_module(self):
        """Initialize the AtlasMuonDataModule with proper configuration."""
        print("Setting up data module...")
        
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set evaluation parameters
        num_test_events = self.max_events if self.max_events is not None and self.max_events != -1 else -1
        
        # Use standard inputs/targets for evaluation
        inputs = {
            'hit': [
                'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
                'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
                'spacePoint_time', 'spacePoint_driftR',
                'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
                'spacePoint_channel', 'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
                'spacePoint_stationIndex', 'spacePoint_technology',
                'r', 's', 'theta', 'phi'
            ]
        }
        targets = {
            'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi', 'truthMuon_qpt']
        }
        
        # Initialize data module
        self.data_module = AtlasMuonDataModule(
            train_dir=str(self.data_dir),
            val_dir=str(self.data_dir),
            test_dir=str(self.data_dir),
            num_workers=1,
            num_train=1,
            num_val=1,
            num_test=num_test_events,
            batch_size=1,
            event_max_num_particles=2,
            inputs=inputs,
            targets=targets,
            pin_memory=False,
        )
        
        # Setup the data module
        self.data_module.setup(stage='test')
        self.test_dataloader = self.data_module.test_dataloader()
        
        print(f"DataLoader setup complete, processing {num_test_events} events")
    
    def collect_all_data(self):
        """Unified data collection for all three tasks using the proven working approach."""
        print("Collecting data for all tracking tasks...")
        
        # Setup data module
        self.setup_data_module()
        
        # Process using the working approach from individual task evaluators
        with h5py.File(self.eval_path, 'r') as pred_file:
            event_count = 0
            
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                if self.max_events and event_count >= self.max_events:
                    break
                    
                # Get event ID (assuming sequential)
                event_id = str(event_count)
                
                if event_id not in pred_file:
                    print(f"Warning: Event {event_id} not found in predictions file")
                    event_count += 1
                    continue
                
                # Get predictions from the file using the working approach
                pred_group = pred_file[event_id]
                
                # Extract all task predictions
                try:
                    # Task 1: Hit-track assignment predictions and logits
                    hit_track_pred = pred_group['preds/final/track_hit_valid/track_hit_valid'][...]  # Shape: (1, 2, num_hits)
                    hit_track_logits = pred_group['outputs/final/track_hit_valid/track_hit_logit'][...]  # Shape: (1, 2, num_hits)
                    
                    # Task 2: Track validity predictions  
                    track_valid_pred = pred_group['preds/final/track_valid/track_valid'][...]  # Shape: (1, 2)
                    
                    # Task 3: Regression predictions (for track parameters)
                    track_eta = pred_group['preds/final/parameter_regression/track_truthMuon_eta'][...]  # Shape: (1, 2)
                    track_phi = pred_group['preds/final/parameter_regression/track_truthMuon_phi'][...]  # Shape: (1, 2)
                    track_qpt = pred_group['preds/final/parameter_regression/track_truthMuon_qpt'][...]  # Shape: (1, 2)
                except Exception as e:
                    print(f"Error reading predictions for event {event_id}: {e}")
                    event_count += 1
                    continue
                
                # Get truth information from batch using the working approach
                inputs, targets = batch
                
                # Truth track parameters
                true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
                true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
                
                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.cpu().numpy()
                num_valid = valid_particles.sum()
                
                if num_valid == 0:
                    event_count += 1
                    continue
                
                # Process the data for all tasks
                batch_size, num_tracks, num_hits = hit_track_pred.shape
                
                for track_idx in range(num_tracks):
                    # Only process if this track slot has a valid prediction
                    if track_valid_pred[0, track_idx]:
                        
                        # Get track data
                        pred_hits = hit_track_pred[0, track_idx]  # Shape: (num_hits,)
                        logit_hits = hit_track_logits[0, track_idx]  # Shape: (num_hits,)
                        track_valid = track_valid_pred[0, track_idx]
                        
                        # True hit assignment for this track (if valid particle exists)
                        if track_idx < num_valid:
                            true_hits = true_hit_assignments[track_idx].cpu().numpy()  # Shape: (num_hits,)
                            
                            # Get true track parameters
                            true_pt = 1.0 / abs(track_qpt[0, track_idx]) if abs(track_qpt[0, track_idx]) > 1e-6 else 0.0
                            true_eta = track_eta[0, track_idx] 
                            true_phi = track_phi[0, track_idx]
                            
                            # Store Task 1 data (Hit-Track Assignment)
                            self.hit_predictions.extend(pred_hits)
                            self.hit_logits.extend(logit_hits)
                            self.hit_true_assignments.extend(true_hits)
                            
                            for hit_idx in range(num_hits):
                                self.hit_track_info.append({
                                    'pt': true_pt,
                                    'eta': true_eta,
                                    'phi': true_phi,
                                    'event_id': event_count,
                                    'track_id': track_idx,
                                    'hit_id': hit_idx
                                })
                            
                            # Store Task 2 data (Track Validity) 
                            self.track_predictions.append(track_valid)
                            self.track_logits.append(track_valid)  # Use same as logits
                            self.track_true_validity.append(track_idx < num_valid)  # True if valid particle
                            
                            self.track_info.append({
                                'pt': true_pt,
                                'eta': true_eta,
                                'phi': true_phi,
                                'event_id': event_count,
                                'track_id': track_idx,
                                'particle_id': track_idx + 1
                            })
                            
                            # Store Task 3 data (Regression)
                            self.reg_predictions['eta'].append(track_eta[0, track_idx])
                            self.reg_predictions['phi'].append(track_phi[0, track_idx])
                            self.reg_predictions['qpt'].append(track_qpt[0, track_idx])
                            
                            self.reg_truth['eta'].append(true_eta)
                            self.reg_truth['phi'].append(true_phi)
                            self.reg_truth['qpt'].append(1.0 / true_pt if true_pt > 0 else 0.0)
                            
                            self.reg_track_info.append({
                                'pt': true_pt,
                                'eta': true_eta,
                                'phi': true_phi,
                                'event_id': event_count,
                                'track_id': track_idx
                            })
                            
                            # Store common data for baseline filtering
                            self.all_event_ids.append(event_count)
                            self.all_particle_ids.append(track_idx + 1)
                            self.all_particle_pts.append(true_pt)
                            self.all_particle_etas.append(true_eta)
                            self.all_particle_phis.append(true_phi)
                            self.all_station_indices.append([0, 1, 2])  # Placeholder - use actual station data if available
                            self.all_true_hits.append(true_hits.sum())
                
                event_count += 1
                
        print(f"Collected data for:")
        print(f"  - {len(self.hit_predictions)} hit predictions from {event_count} events")
        print(f"  - {len(self.track_predictions)} track validity predictions") 
        print(f"  - {len(self.reg_predictions['eta'])} regression predictions")
        
        # Convert lists to numpy arrays
        self._convert_to_arrays()
    
    def _process_batch(self, batch, eval_file, batch_idx):
        """Process a single batch and extract data for all tasks."""
        event_id = batch_idx  # Simple event ID
        
        # Extract basic information from the batch
        try:
            # Batch is a tuple of (inputs, targets)
            inputs, targets = batch
            
            # Extract truth information from targets
            true_particle_valid = targets['particle_valid'][0]  # Shape: (2,)
            true_hit_assignments = targets['particle_hit_valid'][0]  # Shape: (2, num_hits)
            
            # Extract true particle parameters (only for valid particles)
            valid_particles = true_particle_valid.cpu().numpy()
            hit_true_track = true_hit_assignments.cpu().numpy()
            
            # Get particle properties from targets
            particle_pt = targets.get('particle_pt', torch.zeros(2)).cpu().numpy()
            particle_eta = targets.get('particle_eta', torch.zeros(2)).cpu().numpy()
            particle_phi = targets.get('particle_phi', torch.zeros(2)).cpu().numpy()
            particle_q_over_pt = targets.get('particle_q_over_pt', torch.zeros(2)).cpu().numpy()
            
            # Create particle_id array (valid particles have positive IDs)
            particle_id = np.arange(len(valid_particles)) + 1
            particle_id[~valid_particles] = 0  # Invalid particles get ID 0
            
            # Hit information (basic structure)
            num_hits = hit_true_track.shape[1] if len(hit_true_track.shape) > 1 else 0
            hit_station_index = np.zeros(num_hits)  # Placeholder
            truth_hit_particle_id = np.zeros(num_hits)  # Placeholder
            
        except Exception as e:
            print(f"Error extracting graph data for batch {batch_idx}: {e}")
            return
        
        # Get predictions from evaluation file using event index
        try:
            # The evaluation file is organized by event ID
            event_key = str(batch_idx)
            
            if event_key not in eval_file:
                print(f"Event {batch_idx} not found in evaluation file")
                return
                
            event_preds = eval_file[event_key]['preds']['final']
            
            # Get Task 1 predictions: Hit-Track Assignment
            if 'track_hit_valid' in event_preds:
                hit_track_valid = event_preds['track_hit_valid']['track_hit_valid'][...]
            else:
                hit_track_valid = np.array([])
                
            # Get Task 2 predictions: Track Validity
            if 'track_valid' in event_preds:
                track_valid = event_preds['track_valid']['track_valid'][...]
            else:
                track_valid = np.array([])
                
            # Get Task 3 predictions: Parameter Regression
            param_regression = []
            if 'parameter_regression' in event_preds:
                param_group = event_preds['parameter_regression']
                if all(key in param_group for key in ['track_truthMuon_eta', 'track_truthMuon_phi', 'track_truthMuon_qpt']):
                    eta_pred = param_group['track_truthMuon_eta'][...]
                    phi_pred = param_group['track_truthMuon_phi'][...]
                    qpt_pred = param_group['track_truthMuon_qpt'][...]
                    # Stack into shape (n_tracks, 3)
                    param_regression = np.column_stack([eta_pred.flatten(), phi_pred.flatten(), qpt_pred.flatten()])
                else:
                    param_regression = np.array([])
            else:
                param_regression = np.array([])
            
        except Exception as e:
            print(f"Error reading predictions for batch {batch_idx}: {e}")
            return
        
        # Store data for Task 1 (Hit-Track Assignment)
        if len(hit_track_valid) > 0:
            hit_track_valid_flat = hit_track_valid.flatten()
            hit_true_track_flat = hit_true_track.flatten()
            
            # Ensure same length
            min_len = min(len(hit_track_valid_flat), len(hit_true_track_flat))
            if min_len > 0:
                self.hit_predictions.extend(hit_track_valid_flat[:min_len])
                self.hit_true_assignments.extend(hit_true_track_flat[:min_len])
                self.hit_logits.extend(hit_track_valid_flat[:min_len])  # Use predictions as logits
                
                # Store track info for hits
                for i in range(min_len):
                    hit_particle_idx = min(i, len(truth_hit_particle_id) - 1) if len(truth_hit_particle_id) > 0 else 0
                    particle_idx = min(i, len(particle_pt) - 1) if len(particle_pt) > 0 else 0
                    
                    self.hit_track_info.append({
                        'event_id': event_id,
                        'particle_id': truth_hit_particle_id[hit_particle_idx] if len(truth_hit_particle_id) > 0 else -1,
                        'pt': particle_pt[particle_idx] if len(particle_pt) > 0 else 0,
                        'eta': particle_eta[particle_idx] if len(particle_eta) > 0 else 0,
                        'phi': particle_phi[particle_idx] if len(particle_phi) > 0 else 0
                    })
        
        # Store data for Task 2 (Track Validity)
        if len(track_valid) > 0:
            n_tracks = len(track_valid)
            self.track_predictions.extend(track_valid)
            self.track_logits.extend(track_valid)  # Use predictions as logits
            
            # True track validity (tracks with valid particle_id > 0)
            track_validity = particle_id > 0  # Valid tracks have positive particle IDs
            self.track_true_validity.extend(track_validity[:n_tracks])
            
            # Store track info for Task 2
            for i in range(n_tracks):
                particle_idx = min(i, len(particle_pt) - 1) if len(particle_pt) > 0 else 0
                
                self.track_info.append({
                    'event_id': event_id,
                    'particle_id': particle_id[particle_idx] if len(particle_id) > particle_idx else -1,
                    'pt': particle_pt[particle_idx] if len(particle_pt) > particle_idx else 0,
                    'eta': particle_eta[particle_idx] if len(particle_eta) > particle_idx else 0,
                    'phi': particle_phi[particle_idx] if len(particle_phi) > particle_idx else 0
                })
        
        # Store data for Task 3 (Regression)
        if len(param_regression) > 0 and len(param_regression.shape) > 1 and param_regression.shape[1] >= 3:
            n_reg = param_regression.shape[0]
            self.reg_predictions['eta'].extend(param_regression[:, 0])
            self.reg_predictions['phi'].extend(param_regression[:, 1])
            self.reg_predictions['qpt'].extend(param_regression[:, 2])
            
            # Truth values
            for i in range(n_reg):
                particle_idx = min(i, len(particle_pt) - 1) if len(particle_pt) > 0 else 0
                
                self.reg_truth['eta'].append(particle_eta[particle_idx] if len(particle_eta) > particle_idx else 0)
                self.reg_truth['phi'].append(particle_phi[particle_idx] if len(particle_phi) > particle_idx else 0)
                self.reg_truth['qpt'].append(particle_q_over_pt[particle_idx] if len(particle_q_over_pt) > particle_idx else 0)
                
                # Store regression track info
                self.reg_track_info.append({
                    'event_id': event_id,
                    'particle_id': particle_id[particle_idx] if len(particle_id) > particle_idx else -1,
                    'pt': particle_pt[particle_idx] if len(particle_pt) > particle_idx else 0,
                    'eta': particle_eta[particle_idx] if len(particle_eta) > particle_idx else 0,
                    'phi': particle_phi[particle_idx] if len(particle_phi) > particle_idx else 0
                })
        
        # Store common data for baseline filtering (using hit-level data)
        if len(hit_true_track) > 0:
            n_common = len(hit_true_track.flatten())
            self.all_event_ids.extend([event_id] * n_common)
            
            for i in range(n_common):
                hit_particle_idx = min(i, len(truth_hit_particle_id) - 1) if len(truth_hit_particle_id) > 0 else 0
                particle_idx = min(i, len(particle_pt) - 1) if len(particle_pt) > 0 else 0
                station_idx = min(i, len(hit_station_index) - 1) if len(hit_station_index) > 0 else 0
                
                self.all_particle_ids.append(truth_hit_particle_id[hit_particle_idx] if len(truth_hit_particle_id) > 0 else -1)
                self.all_particle_pts.append(particle_pt[particle_idx] if len(particle_pt) > 0 else 0)
                self.all_particle_etas.append(particle_eta[particle_idx] if len(particle_eta) > 0 else 0)
                self.all_particle_phis.append(particle_phi[particle_idx] if len(particle_phi) > 0 else 0)
                self.all_station_indices.append(hit_station_index.flatten()[station_idx] if len(hit_station_index.flatten()) > station_idx else 0)
                self.all_true_hits.append(hit_true_track.flatten()[i])
        
        print(f"Processed batch {batch_idx}: {len(hit_true_track.flatten()) if len(hit_true_track) > 0 else 0} hits, {len(track_valid) if len(track_valid) > 0 else 0} tracks, {len(param_regression) if len(param_regression) > 0 else 0} regression predictions")
    
    def _convert_to_arrays(self):
        """Convert collected lists to numpy arrays."""
        # Task 1 arrays
        self.hit_predictions = np.array(self.hit_predictions)
        self.hit_true_assignments = np.array(self.hit_true_assignments)
        self.hit_logits = np.array(self.hit_logits)
        
        # Task 2 arrays
        self.track_predictions = np.array(self.track_predictions)
        self.track_true_validity = np.array(self.track_true_validity)
        self.track_logits = np.array(self.track_logits)
        
        # Task 3 arrays
        for param in ['eta', 'phi', 'qpt']:
            self.reg_predictions[param] = np.array(self.reg_predictions[param])
            self.reg_truth[param] = np.array(self.reg_truth[param])
        
        # Common arrays for baseline filtering
        self.all_event_ids = np.array(self.all_event_ids)
        self.all_particle_ids = np.array(self.all_particle_ids)
        self.all_particle_pts = np.array(self.all_particle_pts)
        self.all_particle_etas = np.array(self.all_particle_etas)
        self.all_particle_phis = np.array(self.all_particle_phis)
        self.all_station_indices = np.array(self.all_station_indices)
        self.all_true_hits = np.array(self.all_true_hits, dtype=bool)
    
    def create_baseline_track_filter(self):
        """
        Create filter masks for baseline evaluation following hit filter approach.
        
        Baseline requirements:
        - At least 3 different stations
        - At least 3 hits per station (>= 9 hits total per track)
        - |eta| >= 0.1 and |eta| <= 2.7 (detector acceptance region)
        - pt >= 3 GeV (minimum pt threshold)
        
        Returns:
            baseline_mask: Boolean array for hits/tracks in baseline evaluation
            rejected_mask: Boolean array for hits/tracks in rejected tracks evaluation
            stats: Dictionary with detailed filtering statistics
        """
        print("Creating baseline track filter (>=3 stations, >=3 hits per station, eta cuts, pt cuts)...")
        
        # Only consider true hits for track evaluation
        true_hit_mask = self.all_true_hits
        print(f"Total true hits available for track evaluation: {np.sum(true_hit_mask):,}")
        
        # Get unique combinations of (event_id, particle_id) for valid tracks
        valid_event_particle_combinations = np.unique(
            np.column_stack([
                self.all_event_ids[true_hit_mask],
                self.all_particle_ids[true_hit_mask]
            ]), axis=0
        )
        print(f"Found {len(valid_event_particle_combinations)} unique tracks with truth hits")
        
        # Track statistics for detailed reporting
        stats = {
            'total_tracks_checked': 0,
            'tracks_failed_min_hits': 0,
            'tracks_failed_eta_cuts': 0,
            'tracks_failed_pt_cuts': 0,
            'tracks_failed_station_cuts': 0,
            'tracks_passed_all_cuts': 0
        }
        
        # Process tracks in parallel
        print(f"Processing {len(valid_event_particle_combinations)} tracks using parallel workers...")
        
        n_workers = min(mp.cpu_count(), max(1, len(valid_event_particle_combinations) // 100))
        n_workers = min(n_workers, 8)  # Cap at 8 workers
        print(f"Using {n_workers} parallel workers")
        
        # Split tracks into chunks for parallel processing
        chunk_size = max(1, len(valid_event_particle_combinations) // n_workers)
        track_chunks = [
            valid_event_particle_combinations[i:i + chunk_size] 
            for i in range(0, len(valid_event_particle_combinations), chunk_size)
        ]
        
        # Create worker function with pre-bound arguments
        worker_fn = partial(
            _process_track_chunk_for_baseline,
            all_event_ids=self.all_event_ids,
            all_particle_ids=self.all_particle_ids,
            all_particle_pts=self.all_particle_pts,
            all_particle_etas=self.all_particle_etas,
            all_station_indices=self.all_station_indices,
            true_hit_mask=true_hit_mask
        )
        
        # Process tracks in parallel
        baseline_qualified_tracks = set()
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, track_chunks)
        
        # Aggregate results from all workers
        for result in results:
            baseline_qualified_tracks.update(result['qualified_tracks'])
            for key in stats:
                stats[key] += result['stats'][key]
        
        # Print detailed statistics
        print(f"Baseline filtering results:")
        print(f"  Total tracks checked: {stats['total_tracks_checked']}")
        print(f"  Failed minimum hits (>=9): {stats['tracks_failed_min_hits']} ({stats['tracks_failed_min_hits']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {stats['tracks_failed_eta_cuts']} ({stats['tracks_failed_eta_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed pt cuts (pt >= 3 GeV): {stats['tracks_failed_pt_cuts']} ({stats['tracks_failed_pt_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Failed station cuts (>=3 stations, >=3 hits/station): {stats['tracks_failed_station_cuts']} ({stats['tracks_failed_station_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        print(f"  Tracks passing all cuts: {stats['tracks_passed_all_cuts']} ({stats['tracks_passed_all_cuts']/stats['total_tracks_checked']*100:.1f}%)")
        
        # Create track sets
        all_track_set = set(tuple(track) for track in valid_event_particle_combinations)
        baseline_qualified_track_set = set(baseline_qualified_tracks)
        rejected_track_set = all_track_set - baseline_qualified_track_set
        
        print(f"Baseline tracks: {len(baseline_qualified_tracks)}")
        print(f"Rejected tracks: {len(rejected_track_set)}")
        
        # Store track sets for later use
        self.baseline_tracks = baseline_qualified_track_set
        self.rejected_tracks = rejected_track_set
        self.all_tracks = all_track_set
        
        return baseline_qualified_track_set, rejected_track_set, stats
    
    def _get_track_mask_for_region(self, region_type):
        """Get boolean mask for tracks in a specific region."""
        if region_type == 'all_tracks':
            return np.ones(len(self.track_info), dtype=bool)
        
        track_mask = np.zeros(len(self.track_info), dtype=bool)
        
        for i, track_info in enumerate(self.track_info):
            # Use track_id instead of particle_id for track identification
            track_key = (track_info['event_id'], track_info['track_id'])
            
            if region_type == 'baseline_filtered':
                track_mask[i] = track_key in self.baseline_tracks
            elif region_type == 'rejected_tracks':
                track_mask[i] = track_key in self.rejected_tracks
        
        return track_mask
    
    def _get_hit_mask_for_region(self, region_type):
        """Get boolean mask for hits in a specific region."""
        if region_type == 'all_tracks':
            return np.ones(len(self.hit_track_info), dtype=bool)
        
        hit_mask = np.zeros(len(self.hit_track_info), dtype=bool)
        
        for i, hit_info in enumerate(self.hit_track_info):
            # Use track_id instead of particle_id for track identification
            track_key = (hit_info['event_id'], hit_info['track_id'])
            
            if region_type == 'baseline_filtered':
                hit_mask[i] = track_key in self.baseline_tracks
            elif region_type == 'rejected_tracks':
                hit_mask[i] = track_key in self.rejected_tracks
        
        return hit_mask
    
    def _get_reg_mask_for_region(self, region_type):
        """Get boolean mask for regression data in a specific region."""
        if region_type == 'all_tracks':
            return np.ones(len(self.reg_track_info), dtype=bool)
        
        reg_mask = np.zeros(len(self.reg_track_info), dtype=bool)
        
        for i, reg_info in enumerate(self.reg_track_info):
            # Use track_id instead of particle_id for track identification
            track_key = (reg_info['event_id'], reg_info['track_id'])
            
            if region_type == 'baseline_filtered':
                reg_mask[i] = track_key in self.baseline_tracks
            elif region_type == 'rejected_tracks':
                reg_mask[i] = track_key in self.rejected_tracks
        
        return reg_mask
    
    def evaluate_task1_hit_track_assignment(self, region_type, output_dir):
        """Evaluate Task 1: Hit-Track Assignment for a specific region."""
        print(f"Evaluating Task 1 (Hit-Track Assignment) for {region_type}...")
        
        # Get data mask for this region
        hit_mask = self._get_hit_mask_for_region(region_type)
        
        if np.sum(hit_mask) == 0:
            print(f"No hits found for {region_type}, skipping Task 1 evaluation")
            return {}
        
        # Apply mask to get data for this region
        predictions = self.hit_predictions[hit_mask]
        true_assignments = self.hit_true_assignments[hit_mask]
        logits = self.hit_logits[hit_mask]
        
        print(f"Task 1 {region_type}: {len(predictions)} hits, {np.sum(true_assignments)} true assignments")
        
        # Calculate ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(true_assignments, logits)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Task 1: Hit-Track Assignment ROC - {region_type.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            output_path = output_dir / f"task1_roc_curve_{region_type}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Task 1 ROC curve saved to {output_path}")
            
        except Exception as e:
            print(f"Error calculating ROC for Task 1 {region_type}: {e}")
            roc_auc = 0.0
        
        # Calculate efficiency and fake rate by pT bins
        efficiency_results = self._calculate_efficiency_fake_rate_by_pt(
            predictions, true_assignments, logits, hit_mask, region_type, 'Task 1'
        )
        
        # Plot efficiency vs pT
        self._plot_efficiency_vs_pt(efficiency_results, output_dir, f"task1_{region_type}", "Hit-Track Assignment")
        
        return {
            'roc_auc': roc_auc,
            'total_hits': len(predictions),
            'true_assignments': np.sum(true_assignments),
            'efficiency_results': efficiency_results
        }
    
    def evaluate_task2_track_validity(self, region_type, output_dir):
        """Evaluate Task 2: Track Validity Classification for a specific region."""
        print(f"Evaluating Task 2 (Track Validity) for {region_type}...")
        
        # Get data mask for this region
        track_mask = self._get_track_mask_for_region(region_type)
        
        if np.sum(track_mask) == 0:
            print(f"No tracks found for {region_type}, skipping Task 2 evaluation")
            return {}
        
        # Apply mask to get data for this region
        predictions = self.track_predictions[track_mask]
        true_validity = self.track_true_validity[track_mask]
        logits = self.track_logits[track_mask]
        
        print(f"Task 2 {region_type}: {len(predictions)} tracks, {np.sum(true_validity)} valid tracks")
        
        # Calculate ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(true_validity, logits)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Task 2: Track Validity ROC - {region_type.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            output_path = output_dir / f"task2_roc_curve_{region_type}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Task 2 ROC curve saved to {output_path}")
            
        except Exception as e:
            print(f"Error calculating ROC for Task 2 {region_type}: {e}")
            roc_auc = 0.0
        
        # Calculate efficiency and fake rate by pT bins
        efficiency_results = self._calculate_efficiency_fake_rate_by_pt(
            predictions, true_validity, logits, track_mask, region_type, 'Task 2'
        )
        
        # Plot efficiency vs pT
        self._plot_efficiency_vs_pt(efficiency_results, output_dir, f"task2_{region_type}", "Track Validity")
        
        # Plot logit distributions
        self._plot_logit_distributions(logits, true_validity, output_dir, f"task2_logits_{region_type}", "Track Validity")
        
        return {
            'roc_auc': roc_auc,
            'total_tracks': len(predictions),
            'valid_tracks': np.sum(true_validity),
            'efficiency_results': efficiency_results
        }
    
    def evaluate_task3_regression(self, region_type, output_dir):
        """Evaluate Task 3: Regression Outputs for a specific region."""
        print(f"Evaluating Task 3 (Regression) for {region_type}...")
        
        # Get data mask for this region
        reg_mask = self._get_reg_mask_for_region(region_type)
        
        if np.sum(reg_mask) == 0:
            print(f"No regression data found for {region_type}, skipping Task 3 evaluation")
            return {}
        
        results = {}
        
        # Evaluate each parameter
        for param in ['eta', 'phi', 'qpt']:
            pred = self.reg_predictions[param][reg_mask]
            truth = self.reg_truth[param][reg_mask]
            
            if len(pred) == 0:
                continue
            
            print(f"Task 3 {region_type} {param}: {len(pred)} predictions")
            
            # Calculate residuals
            residuals = pred - truth
            
            # Remove outliers (beyond 5 sigma)
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            outlier_mask = np.abs(residuals - residual_mean) < 5 * residual_std
            
            clean_residuals = residuals[outlier_mask]
            clean_pred = pred[outlier_mask]
            clean_truth = truth[outlier_mask]
            
            # Calculate statistics
            mean_residual = np.mean(clean_residuals)
            std_residual = np.std(clean_residuals)
            
            # Plot residual distribution (step histogram)
            self._plot_residual_distribution(clean_residuals, output_dir, f"task3_{param}_residuals_{region_type}", param)
            
            # Plot overlaid distributions
            self._plot_overlaid_distributions(clean_pred, clean_truth, output_dir, f"task3_{param}_overlaid_{region_type}", param)
            
            # Plot residuals vs truth
            self._plot_residuals_vs_truth(clean_residuals, clean_truth, output_dir, f"task3_{param}_residuals_vs_truth_{region_type}", param)
            
            results[param] = {
                'mean_residual': mean_residual,
                'std_residual': std_residual,
                'n_predictions': len(clean_pred),
                'outliers_removed': len(pred) - len(clean_pred)
            }
        
        return results
    
    def _calculate_efficiency_fake_rate_by_pt(self, predictions, true_labels, logits, data_mask, region_type, task_name):
        """Calculate efficiency and fake rate binned by pT."""
        # Get pT values for this region
        if task_name == 'Task 1':
            pt_values = np.array([info['pt'] for info in self.hit_track_info])[data_mask]
        elif task_name == 'Task 2':
            pt_values = np.array([info['pt'] for info in self.track_info])[data_mask]
        else:
            return {}
        
        # Define pT bins (linear scale, matching hit filter evaluation)
        pt_bins = np.linspace(0, 200, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate threshold for 90% efficiency
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, logits)
            target_efficiency = 0.9
            threshold = thresholds[tpr >= target_efficiency][0] if np.any(tpr >= target_efficiency) else np.median(thresholds)
        except:
            threshold = 0.5  # Fallback
        
        # Apply threshold
        pred_positive = logits >= threshold
        
        efficiencies = []
        fake_rates = []
        eff_errors = []
        fake_rate_errors = []
        
        for i in range(len(pt_bins) - 1):
            pt_mask = (pt_values >= pt_bins[i]) & (pt_values < pt_bins[i+1])
            
            if np.sum(pt_mask) == 0:
                efficiencies.append(0)
                fake_rates.append(0)
                eff_errors.append(0)
                fake_rate_errors.append(0)
                continue
            
            bin_true_labels = true_labels[pt_mask]
            bin_pred_positive = pred_positive[pt_mask]
            
            # Calculate efficiency (True Positive Rate)
            true_positives = np.sum(bin_true_labels & bin_pred_positive)
            total_positives = np.sum(bin_true_labels)
            
            if total_positives > 0:
                efficiency = true_positives / total_positives
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0
                eff_error = 0
            
            # Calculate fake rate (False Positive Rate)
            false_positives = np.sum(~bin_true_labels & bin_pred_positive)
            total_negatives = np.sum(~bin_true_labels)
            
            if total_negatives > 0:
                fake_rate = false_positives / total_negatives
                fake_rate_error = np.sqrt(fake_rate * (1 - fake_rate) / total_negatives)
            else:
                fake_rate = 0
                fake_rate_error = 0
            
            efficiencies.append(efficiency)
            fake_rates.append(fake_rate)
            eff_errors.append(eff_error)
            fake_rate_errors.append(fake_rate_error)
        
        return {
            'pt_centers': pt_centers,
            'efficiencies': np.array(efficiencies),
            'fake_rates': np.array(fake_rates),
            'eff_errors': np.array(eff_errors),
            'fake_rate_errors': np.array(fake_rate_errors),
            'threshold': threshold
        }
    
    def _plot_efficiency_vs_pt(self, efficiency_results, output_dir, prefix, task_title):
        """Plot efficiency and fake rate vs pT with step plots and error bands."""
        if not efficiency_results:
            return
        
        pt_centers = efficiency_results['pt_centers']
        efficiencies = efficiency_results['efficiencies']
        fake_rates = efficiency_results['fake_rates']
        eff_errors = efficiency_results['eff_errors']
        fake_rate_errors = efficiency_results['fake_rate_errors']
        
        # Create step plots with error bands (matching hit filter style)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Efficiency plot
        ax1.step(pt_centers, efficiencies, where='mid', color='blue', linewidth=1, label='Efficiency')
        ax1.fill_between(pt_centers, 
                        np.maximum(0, efficiencies - eff_errors),
                        np.minimum(1, efficiencies + eff_errors),
                        step='mid', alpha=0.3, color='blue')
        
        ax1.set_xlabel('Truth Muon $p_T$ [GeV]')
        ax1.set_ylabel('Efficiency')
        ax1.set_title(f'{task_title} - Efficiency vs $p_T$')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 200])
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        
        # Fake rate plot
        ax2.step(pt_centers, fake_rates, where='mid', color='red', linewidth=1, label='Fake Rate')
        ax2.fill_between(pt_centers,
                        np.maximum(0, fake_rates - fake_rate_errors),
                        fake_rates + fake_rate_errors,
                        step='mid', alpha=0.3, color='red')
        
        ax2.set_xlabel('Truth Muon $p_T$ [GeV]')
        ax2.set_ylabel('Fake Rate')
        ax2.set_title(f'{task_title} - Fake Rate vs $p_T$')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 200])
        ax2.set_ylim([0, max(0.1, np.max(fake_rates) * 1.1)])
        ax2.legend()
        
        plt.tight_layout()
        
        output_path = output_dir / f"{prefix}_efficiency_fake_rate_vs_pt.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Efficiency/fake rate plot saved to {output_path}")
    
    def _plot_logit_distributions(self, logits, true_labels, output_dir, prefix, task_title):
        """Plot logit distributions for true vs false cases."""
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        true_logits = logits[true_labels]
        false_logits = logits[~true_labels]
        
        plt.hist(false_logits, bins=50, alpha=0.6, label=f'False ({len(false_logits)})', 
                color='red', density=True, histtype='stepfilled')
        plt.hist(true_logits, bins=50, alpha=0.6, label=f'True ({len(true_logits)})', 
                color='blue', density=True, histtype='stepfilled')
        
        plt.xlabel('Logit Score')
        plt.ylabel('Density')
        plt.title(f'{task_title} - Logit Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / f"{prefix}_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Logit distributions plot saved to {output_path}")
    
    def _plot_residual_distribution(self, residuals, output_dir, prefix, param_name):
        """Plot residual distribution as step histogram."""
        plt.figure(figsize=(10, 6))
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Create step histogram with thin lines (matching hit filter style)
        counts, bins, _ = plt.hist(residuals, bins=60, alpha=0, histtype='step')
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.step(bin_centers, counts, where='mid', linewidth=1, color='blue')
        
        plt.axvline(mean_residual, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {mean_residual:.4f}')
        plt.axvline(mean_residual + std_residual, color='orange', linestyle='--', 
                   label=f'±1σ = {std_residual:.4f}')
        plt.axvline(mean_residual - std_residual, color='orange', linestyle='--')
        
        plt.xlabel(f'{param_name} Residual (Predicted - Truth)')
        plt.ylabel('Count')
        plt.title(f'Task 3: {param_name} Residual Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / f"{prefix}_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Residual distribution plot saved to {output_path}")
    
    def _plot_overlaid_distributions(self, predictions, truth, output_dir, prefix, param_name):
        """Plot overlaid distributions of predictions vs truth with transparency."""
        plt.figure(figsize=(10, 6))
        
        # Plot overlaid histograms with transparency (matching hit filter style)
        plt.hist(truth, bins=60, alpha=0.6, label=f'Truth ({len(truth)})', 
                color='blue', density=True, histtype='stepfilled')
        plt.hist(predictions, bins=60, alpha=0.6, label=f'Predicted ({len(predictions)})', 
                color='red', density=True, histtype='stepfilled')
        
        plt.xlabel(f'{param_name}')
        plt.ylabel('Density')
        plt.title(f'Task 3: {param_name} - Truth vs Predicted Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / f"{prefix}_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Overlaid distributions plot saved to {output_path}")
    
    def _plot_residuals_vs_truth(self, residuals, truth, output_dir, prefix, param_name):
        """Plot residuals vs truth values."""
        plt.figure(figsize=(10, 6))
        
        # Sample data if too many points
        if len(residuals) > 10000:
            indices = np.random.choice(len(residuals), 10000, replace=False)
            residuals = residuals[indices]
            truth = truth[indices]
        
        plt.scatter(truth, residuals, alpha=0.3, s=1)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        
        plt.xlabel(f'{param_name} Truth')
        plt.ylabel(f'{param_name} Residual (Predicted - Truth)')
        plt.title(f'Task 3: {param_name} Residuals vs Truth')
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / f"{prefix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Residuals vs truth plot saved to {output_path}")
    
    def run_full_evaluation(self):
        """Run the complete unified evaluation across all comparison regions."""
        print("=" * 80)
        print("UNIFIED TRACKING MODEL EVALUATION")
        print("=" * 80)
        
        # Step 1: Collect all data
        self.collect_all_data()
        
        # Step 2: Create baseline track filter
        baseline_tracks, rejected_tracks, baseline_stats = self.create_baseline_track_filter()
        
        # Step 3: Evaluate each task for each region
        regions = ['all_tracks', 'baseline_filtered', 'rejected_tracks']
        region_dirs = [self.all_tracks_dir, self.baseline_filtered_dir, self.rejected_tracks_dir]
        
        for region, region_dir in zip(regions, region_dirs):
            print(f"\n{'=' * 60}")
            print(f"EVALUATING REGION: {region.upper().replace('_', ' ')}")
            print("=" * 60)
            
            region_results = {}
            
            # Task 1: Hit-Track Assignment
            try:
                task1_results = self.evaluate_task1_hit_track_assignment(region, region_dir)
                region_results['task1'] = task1_results
                print(f"✅ Task 1 completed for {region}")
            except Exception as e:
                print(f"❌ Task 1 failed for {region}: {e}")
                region_results['task1'] = {}
            
            # Task 2: Track Validity
            try:
                task2_results = self.evaluate_task2_track_validity(region, region_dir)
                region_results['task2'] = task2_results
                print(f"✅ Task 2 completed for {region}")
            except Exception as e:
                print(f"❌ Task 2 failed for {region}: {e}")
                region_results['task2'] = {}
            
            # Task 3: Regression
            try:
                task3_results = self.evaluate_task3_regression(region, region_dir)
                region_results['task3'] = task3_results
                print(f"✅ Task 3 completed for {region}")
            except Exception as e:
                print(f"❌ Task 3 failed for {region}: {e}")
                region_results['task3'] = {}
            
            # Store results for this region
            self.evaluation_results[region] = region_results
            
            # Write region summary
            self._write_region_summary(region, region_results, region_dir)
        
        # Step 4: Write comparative summary and config logging
        self._write_comparative_summary()
        self._log_results_to_config()
        
        print(f"\n🎉 Unified evaluation completed successfully!")
        print(f"📂 Results saved to: {self.output_dir}")
    
    def _write_region_summary(self, region, results, output_dir):
        """Write summary file for a specific region."""
        summary_path = output_dir / f"{region}_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"# {region.replace('_', ' ').title()} Evaluation Summary\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Task 1 summary
            if 'task1' in results and results['task1']:
                t1 = results['task1']
                f.write("## Task 1: Hit-Track Assignment\n\n")
                f.write(f"- ROC AUC: {t1.get('roc_auc', 0):.4f}\n")
                f.write(f"- Total hits: {t1.get('total_hits', 0):,}\n")
                f.write(f"- True assignments: {t1.get('true_assignments', 0):,}\n")
                if 'efficiency_results' in t1:
                    eff_res = t1['efficiency_results']
                    avg_eff = np.mean(eff_res.get('efficiencies', [0]))
                    avg_fake = np.mean(eff_res.get('fake_rates', [0]))
                    f.write(f"- Average efficiency: {avg_eff:.4f}\n")
                    f.write(f"- Average fake rate: {avg_fake:.4f}\n")
                f.write("\n")
            
            # Task 2 summary
            if 'task2' in results and results['task2']:
                t2 = results['task2']
                f.write("## Task 2: Track Validity Classification\n\n")
                f.write(f"- ROC AUC: {t2.get('roc_auc', 0):.4f}\n")
                f.write(f"- Total tracks: {t2.get('total_tracks', 0):,}\n")
                f.write(f"- Valid tracks: {t2.get('valid_tracks', 0):,}\n")
                if 'efficiency_results' in t2:
                    eff_res = t2['efficiency_results']
                    avg_eff = np.mean(eff_res.get('efficiencies', [0]))
                    avg_fake = np.mean(eff_res.get('fake_rates', [0]))
                    f.write(f"- Average efficiency: {avg_eff:.4f}\n")
                    f.write(f"- Average fake rate: {avg_fake:.4f}\n")
                f.write("\n")
            
            # Task 3 summary
            if 'task3' in results and results['task3']:
                t3 = results['task3']
                f.write("## Task 3: Regression Outputs\n\n")
                for param in ['eta', 'phi', 'qpt']:
                    if param in t3:
                        param_res = t3[param]
                        f.write(f"### {param}\n")
                        f.write(f"- Mean residual: {param_res.get('mean_residual', 0):.6f}\n")
                        f.write(f"- Std residual: {param_res.get('std_residual', 0):.6f}\n")
                        f.write(f"- Predictions: {param_res.get('n_predictions', 0):,}\n")
                        f.write(f"- Outliers removed: {param_res.get('outliers_removed', 0):,}\n\n")
        
        print(f"Region summary saved to {summary_path}")
    
    def _write_comparative_summary(self):
        """Write comparative summary across all regions."""
        summary_path = self.output_dir / "UNIFIED_EVALUATION_SUMMARY.md"
        
        with open(summary_path, 'w') as f:
            f.write("# ATLAS Muon Tracking Model - Unified Evaluation Summary\n\n")
            f.write(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This unified evaluation covers three main tasks across three comparison regions:\n\n")
            f.write("### Tasks:\n")
            f.write("1. **Task 1**: Hit-Track Assignment (track_hit_valid)\n")
            f.write("2. **Task 2**: Track Validity Classification (track_valid)\n")
            f.write("3. **Task 3**: Regression Outputs (parameter_regression)\n\n")
            
            f.write("### Comparison Regions:\n")
            f.write("- **All Tracks**: Complete dataset without filtering\n")
            f.write("- **Baseline Filtered**: Tracks meeting quality criteria (≥3 stations, ≥3 hits/station, eta/pt cuts)\n")
            f.write("- **Rejected Tracks**: Tracks not meeting baseline criteria\n\n")
            
            f.write("## Results Summary\n\n")
            
            # Create summary table
            f.write("| Region | Task 1 AUC | Task 2 AUC | Task 1 Avg Eff | Task 2 Avg Eff | Eta Std | Phi Std | qpt Std |\n")
            f.write("|--------|------------|------------|----------------|----------------|---------|---------|----------|\n")
            
            for region in ['all_tracks', 'baseline_filtered', 'rejected_tracks']:
                region_name = region.replace('_', ' ').title()
                results = self.evaluation_results.get(region, {})
                
                # Extract metrics
                t1_auc = results.get('task1', {}).get('roc_auc', 0)
                t2_auc = results.get('task2', {}).get('roc_auc', 0)
                
                t1_eff = 0
                if 'task1' in results and 'efficiency_results' in results['task1']:
                    t1_eff = np.mean(results['task1']['efficiency_results'].get('efficiencies', [0]))
                
                t2_eff = 0
                if 'task2' in results and 'efficiency_results' in results['task2']:
                    t2_eff = np.mean(results['task2']['efficiency_results'].get('efficiencies', [0]))
                
                eta_std = results.get('task3', {}).get('eta', {}).get('std_residual', 0)
                phi_std = results.get('task3', {}).get('phi', {}).get('std_residual', 0)
                qpt_std = results.get('task3', {}).get('qpt', {}).get('std_residual', 0)
                
                f.write(f"| {region_name} | {t1_auc:.4f} | {t2_auc:.4f} | {t1_eff:.4f} | {t2_eff:.4f} | {eta_std:.6f} | {phi_std:.6f} | {qpt_std:.6f} |\n")
            
            f.write("\n## Directory Structure\n\n")
            f.write("```\n")
            f.write("unified_tracking_evaluation/\n")
            f.write("├── all_tracks/\n")
            f.write("│   ├── task1_roc_curve_all_tracks.png\n")
            f.write("│   ├── task1_all_tracks_efficiency_fake_rate_vs_pt.png\n")
            f.write("│   ├── task2_roc_curve_all_tracks.png\n")
            f.write("│   ├── task2_all_tracks_efficiency_fake_rate_vs_pt.png\n")
            f.write("│   ├── task3_*_residuals_all_tracks_distribution.png\n")
            f.write("│   └── all_tracks_summary.txt\n")
            f.write("├── baseline_filtered_tracks/\n")
            f.write("│   └── (same structure as all_tracks)\n")
            f.write("├── rejected_tracks/\n")
            f.write("│   └── (same structure as all_tracks)\n")
            f.write("├── evaluation_config_log.yaml\n")
            f.write("└── UNIFIED_EVALUATION_SUMMARY.md (this file)\n")
            f.write("```\n\n")
            
            f.write("## Notes\n\n")
            f.write("- All plots use consistent styling with hit filter evaluation\n")
            f.write("- Step plots with error bands for efficiency/fake rate analysis\n")
            f.write("- Transparent overlaid distributions for regression comparison\n")
            f.write("- Results logged to configuration file for easy analysis\n")
            f.write("- Baseline filtering follows same criteria as hit filter evaluation\n")
        
        print(f"Comparative summary saved to {summary_path}")
    
    def _log_results_to_config(self):
        """Log average efficiencies and other metrics to configuration file."""
        config_log = {
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_file': str(self.eval_path),
            'data_directory': str(self.data_dir),
            'max_events': self.max_events,
            'baseline_filtering_stats': {
                'total_tracks': len(self.all_tracks),
                'baseline_tracks': len(self.baseline_tracks),
                'rejected_tracks': len(self.rejected_tracks)
            },
            'results_by_region': {}
        }
        
        # Log results for each region
        for region in ['all_tracks', 'baseline_filtered', 'rejected_tracks']:
            results = self.evaluation_results.get(region, {})
            region_config = {}
            
            # Task 1 metrics
            if 'task1' in results and results['task1']:
                t1 = results['task1']
                task1_config = {
                    'roc_auc': float(t1.get('roc_auc', 0)),
                    'total_hits': int(t1.get('total_hits', 0)),
                    'true_assignments': int(t1.get('true_assignments', 0))
                }
                
                if 'efficiency_results' in t1:
                    eff_res = t1['efficiency_results']
                    task1_config.update({
                        'average_efficiency': float(np.mean(eff_res.get('efficiencies', [0]))),
                        'average_fake_rate': float(np.mean(eff_res.get('fake_rates', [0]))),
                        'efficiency_std': float(np.std(eff_res.get('efficiencies', [0]))),
                        'fake_rate_std': float(np.std(eff_res.get('fake_rates', [0])))
                    })
                
                region_config['task1_hit_track_assignment'] = task1_config
            
            # Task 2 metrics
            if 'task2' in results and results['task2']:
                t2 = results['task2']
                task2_config = {
                    'roc_auc': float(t2.get('roc_auc', 0)),
                    'total_tracks': int(t2.get('total_tracks', 0)),
                    'valid_tracks': int(t2.get('valid_tracks', 0))
                }
                
                if 'efficiency_results' in t2:
                    eff_res = t2['efficiency_results']
                    task2_config.update({
                        'average_efficiency': float(np.mean(eff_res.get('efficiencies', [0]))),
                        'average_fake_rate': float(np.mean(eff_res.get('fake_rates', [0]))),
                        'efficiency_std': float(np.std(eff_res.get('efficiencies', [0]))),
                        'fake_rate_std': float(np.std(eff_res.get('fake_rates', [0])))
                    })
                
                region_config['task2_track_validity'] = task2_config
            
            # Task 3 metrics
            if 'task3' in results and results['task3']:
                t3 = results['task3']
                task3_config = {}
                
                for param in ['eta', 'phi', 'qpt']:
                    if param in t3:
                        param_res = t3[param]
                        task3_config[param] = {
                            'mean_residual': float(param_res.get('mean_residual', 0)),
                            'std_residual': float(param_res.get('std_residual', 0)),
                            'n_predictions': int(param_res.get('n_predictions', 0)),
                            'outliers_removed': int(param_res.get('outliers_removed', 0))
                        }
                
                if task3_config:
                    region_config['task3_regression'] = task3_config
            
            config_log['results_by_region'][region] = region_config
        
        # Save configuration log
        config_path = self.output_dir / "evaluation_config_log.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_log, f, default_flow_style=False, indent=2)
        
        print(f"Configuration log saved to {config_path}")
        
        # Also save as JSON for easy programmatic access
        json_path = self.output_dir / "evaluation_config_log.json"
        with open(json_path, 'w') as f:
            json.dump(config_log, f, indent=2)
        
        print(f"JSON configuration log saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified ATLAS Muon Tracking Model Evaluation')
    parser.add_argument('--eval_path', type=str, 
                       default="/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5",
                       help='Path to evaluation HDF5 file')
    parser.add_argument('--data_dir', type=str, 
                       default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500",
                       help='Path to processed test data directory')
    parser.add_argument('--config_path', type=str, 
                       default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500/metadata.yaml",
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, 
                       default='./tracking_evaluation_results',
                       help='Output directory for plots and results')
    parser.add_argument('--max_events', "-m", type=int, default=1000,
                       help='Maximum number of events to process (for testing)')
    
    args = parser.parse_args()
    
    print("UNIFIED ATLAS MUON TRACKING MODEL EVALUATION")
    print("=" * 80)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    print("=" * 80)
    
    try:
        # Create unified evaluator
        evaluator = UnifiedTrackingEvaluator(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events
        )
        
        # Run full evaluation
        evaluator.run_full_evaluation()
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()