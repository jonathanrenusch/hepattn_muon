#!/usr/bin/env python3
"""Check track index file."""
import numpy as np
track_index = np.load('/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600/track_index_minHits2_nEvents1000.npy')
print('Track index shape:', track_index.shape)
print('First 5 entries:', track_index[:5])
print('Columns are: (event_idx, particle_idx, particle_id)')
