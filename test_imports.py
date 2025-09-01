#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

print("Testing basic imports...")
try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except ImportError as e:
    print(f"✗ matplotlib failed: {e}")

try:
    import h5py
    print("✓ h5py imported")
except ImportError as e:
    print(f"✗ h5py failed: {e}")

try:
    import yaml
    print("✓ yaml imported")
except ImportError as e:
    print(f"✗ yaml failed: {e}")

try:
    from sklearn.metrics import roc_curve, auc
    print("✓ sklearn imported")
except ImportError as e:
    print(f"✗ sklearn failed: {e}")

try:
    from tqdm import tqdm
    print("✓ tqdm imported")
except ImportError as e:
    print(f"✗ tqdm failed: {e}")

# Test file access
eval_path = "/shared/tracking/hepattn_muon/src/logs/ATLAS-Muon-6H100-600K_20250831-T195418/ckpts/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5"
data_path = "/scratch/ml_test_data_156000_hdf5"

print(f"\nTesting file access...")
print(f"Eval file exists: {os.path.exists(eval_path)}")
print(f"Data dir exists: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    metadata_path = os.path.join(data_path, "metadata.yaml")
    print(f"Metadata file exists: {os.path.exists(metadata_path)}")

print("Basic test completed!")
