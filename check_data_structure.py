#!/usr/bin/env python3
"""Check structure of test data HDF5 files."""
import h5py
from pathlib import Path

data_file = '/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600/data/filtered_events.h5'
print(f'Checking file: {data_file}')

with h5py.File(data_file, 'r') as f:
    for key in list(f.keys())[:5]:
        data = f[key]
        if hasattr(data, 'shape'):
            print(f'  {key}: {data.shape}')
        else:
            print(f'  {key}: group')
            for subkey in list(data.keys())[:10]:
                subdata = data[subkey]
                if hasattr(subdata, 'shape'):
                    print(f'    {subkey}: {subdata.shape}')
                else:
                    print(f'    {subkey}: {type(subdata)}')
