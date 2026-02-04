#!/usr/bin/env python3
"""Quick diagnostic to check hits per track."""

from hepattn.experiments.atlas_muon.data import AtlasMuonDataset
import numpy as np

hit_fields = [
    'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
    'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
    'spacePoint_globPosX', 'spacePoint_globPosY', 'spacePoint_globPosZ',
    'spacePoint_driftR', 'spacePoint_time', 'spacePoint_channel',
    'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
    'spacePoint_stationIndex', 'spacePoint_technology',
    'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
]

inputs = {'hit': hit_fields}
targets = {'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi']}

dataset = AtlasMuonDataset(
    dirpath='/scratch/ml_validation_data_144000_regression_true_hits_only/',
    inputs=inputs,
    targets=targets,
    num_events=100,
    event_max_num_particles=10,
)

# Check a few events
hits_per_track = []
for event_idx in range(min(50, len(dataset))):
    hits, particles, num_hits, num_tracks = dataset.load_event(event_idx)
    if num_tracks == 0:
        continue
    
    truth_links = hits['spacePoint_truthLink'][:num_hits]
    particle_ids = particles['particle_id'][:num_tracks]
    
    for pid in particle_ids:
        hit_mask = truth_links == pid
        n_track_hits = np.sum(hit_mask)
        if n_track_hits >= 3:
            hits_per_track.append(n_track_hits)

hits_per_track = np.array(hits_per_track)
print(f"Number of tracks analyzed: {len(hits_per_track)}")
print(f"Hits per track - min: {np.min(hits_per_track)}, max: {np.max(hits_per_track)}, mean: {np.mean(hits_per_track):.1f}, median: {np.median(hits_per_track)}")
print(f"Distribution (bins [3,5,10,15,20,25,30,40,50,100]): {np.histogram(hits_per_track, bins=[3,5,10,15,20,25,30,40,50,100])[0]}")
