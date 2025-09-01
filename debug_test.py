#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

print("Step 1: Basic imports...")
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
from pathlib import Path

print("Step 2: sklearn import...")
from sklearn.metrics import roc_curve, auc

print("Step 3: tqdm import...")
from tqdm import tqdm

print("Step 4: Testing file paths...")
eval_path = "/shared/tracking/hepattn_muon/src/logs/ATLAS-Muon-6H100-600K_20250831-T195418/ckpts/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5"
data_path = "/scratch/ml_test_data_156000_hdf5"

print(f"Eval file exists: {os.path.exists(eval_path)}")
print(f"Data dir exists: {os.path.exists(data_path)}")

print("Step 5: Testing data loading...")
data_path_obj = Path(data_path)
metadata_file = data_path_obj / 'metadata.yaml'
print(f"Metadata file exists: {metadata_file.exists()}")

if metadata_file.exists():
    print("Step 6: Loading metadata...")
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    print(f"Metadata loaded, keys: {list(metadata.keys())}")
    
    print("Step 7: Loading indices...")
    file_indices = np.load(data_path_obj / 'event_file_indices.npy')
    row_indices = np.load(data_path_obj / 'event_row_indices.npy')
    print(f"Indices loaded: {len(file_indices)} events")

print("All steps completed successfully!")
