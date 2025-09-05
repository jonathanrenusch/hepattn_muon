#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

from hepattn.experiments.atlas_muon.filter_dataset_with_hitfilter import generate_output_dir_name

print('Testing output directory name generation:')
print(generate_output_dir_name('/path/to/ml_test_data_150K_processed', 0.99, 3, 500))
print(generate_output_dir_name('/path/to/ml_training_data_1_2Mio_processed', 0.985, 5, 1000))
print(generate_output_dir_name('/path/to/ml_validation_data_150K_processed', 0.97, 2, 300))
