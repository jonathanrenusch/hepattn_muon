#!/usr/bin/env python3
from pathlib import Path

def generate_output_dir_name(input_dir: str, working_point: float, 
                           max_tracks: int, max_hits: int) -> str:
    """Generate output directory name with filtering parameters in the same parent directory as input."""
    input_path = Path(input_dir)
    input_name = input_path.name
    parent_dir = input_path.parent
    
    # Create descriptive suffix
    wp_str = f"wp{working_point:.3f}".replace('.', '')
    tracks_str = f"maxtrk{max_tracks}"
    hits_str = f"maxhit{max_hits}"
    
    # Create the full path in the same parent directory as the input
    filtered_name = f"{input_name}_filtered_{wp_str}_{tracks_str}_{hits_str}"
    return str(parent_dir / filtered_name)

# Test the function
print('Testing new output directory generation:')
result = generate_output_dir_name('/scratch/ml_test_data_156000_hdf5/', 0.99, 5, 1000)
print(f'Input: /scratch/ml_test_data_156000_hdf5/')
print(f'Output: {result}')

# Test another example
result2 = generate_output_dir_name('/path/to/my_dataset/', 0.985, 3, 500)
print(f'Input: /path/to/my_dataset/')
print(f'Output: {result2}')
