#!/usr/bin/env python3
import h5py
import numpy as np

eval_file = '/scratch/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5'
print('Examining evaluation file structure...')

with h5py.File(eval_file, 'r') as f:
    keys = list(f.keys())
    print(f'Top-level keys (first 10): {keys[:10]}')
    print(f'Total number of top-level keys: {len(keys)}')
    
    # Try to examine the structure of the first few events
    first_key = keys[0]
    print(f'\nExamining structure of first event: {first_key}')
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f'Group: {name}')
        elif isinstance(obj, h5py.Dataset):
            print(f'Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}')
    
    f[first_key].visititems(print_structure)
    
    # Check a few more keys to understand the pattern
    print(f'\nChecking if some sample keys exist:')
    test_keys = ['0', '1', '10', '100']
    for test_key in test_keys:
        if test_key in f:
            print(f'Key {test_key}: EXISTS')
        else:
            print(f'Key {test_key}: NOT FOUND')
