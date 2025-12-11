#!/usr/bin/env python
"""Test script for OutwardTrackingDataset."""

import sys
sys.path.insert(0, 'src')

from hepattn.experiments.outward_tracking.data import OutwardTrackingDataset

# Define inputs/targets config - use the derived feature names
inputs = {
    "hit": ["r", "phi", "eta", "spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ"]
}
targets = {
    "hit": []  # Edge targets are built internally
}

# Test on a few events
dataset = OutwardTrackingDataset(
    dirpath='/scratch/ml_training_data_2694000_hdf5_filtered_wp0990_maxtrk2_maxhit600',
    inputs=inputs,
    targets=targets,
    num_events=5,
)

print('Dataset size:', len(dataset))
print()

# Test loading a few events
for i in range(min(3, len(dataset))):
    inputs_dict, targets_dict = dataset[i]
    print('Event', i, ':')
    print('  INPUTS:')
    for key, val in inputs_dict.items():
        if hasattr(val, 'shape'):
            print('   ', key, ':', val.shape, '(', val.dtype, ')')
        else:
            print('   ', key, ':', val)
    
    print('  TARGETS:')
    for key, val in targets_dict.items():
        if hasattr(val, 'shape'):
            print('   ', key, ':', val.shape, '(', val.dtype, ')')
        else:
            print('   ', key, ':', val)
    
    # Verify outward edges
    num_hits = inputs_dict['hit_r'].shape[-1]
    outward_adj = targets_dict['outward_adjacency']
    full_adj = targets_dict['full_adjacency']
    anchor_mask = targets_dict['anchor_mask']
    
    n_outward = outward_adj.sum().item()
    n_full = full_adj.sum().item() 
    n_anchors = anchor_mask.sum().item()
    
    print('  Summary: num_hits=', num_hits, ', outward_edges=', n_outward, 
          ', full_adj_edges=', n_full, ', anchors=', n_anchors)
    print()

print('Test passed!')
