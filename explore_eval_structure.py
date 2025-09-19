#!/usr/bin/env python3
"""
Quick script to explore the structure of the tracking evaluation file.
"""
import h5py
import numpy as np

def explore_evaluation_file():
    eval_path = "/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5"
    
    with h5py.File(eval_path, 'r') as f:
        print("Root keys:", list(f.keys())[:10])  # First 10 event keys
        print(f"Total number of events: {len(list(f.keys()))}")
        
        # Look at the first event
        first_event_key = list(f.keys())[0]
        print(f"\nExamining event {first_event_key}:")
        event = f[first_event_key]
        print("  Event keys:", list(event.keys()))
        
        if 'preds' in event:
            print("  Preds keys:", list(event['preds'].keys()))
            
            if 'final' in event['preds']:
                print("  Final prediction keys:", list(event['preds']['final'].keys()))
                
                for task in event['preds']['final'].keys():
                    print(f"\n    Task '{task}':")
                    task_outputs = event['preds']['final'][task]
                    print(f"      Output keys: {list(task_outputs.keys())}")
                    
                    for output_key in task_outputs.keys():
                        output_data = task_outputs[output_key]
                        print(f"        {output_key}: shape={output_data.shape}, dtype={output_data.dtype}")
                        
                        # Show a small sample of the data
                        if output_data.size < 50:
                            print(f"          Sample values: {output_data[:]}")
                        else:
                            # Convert to numpy array first, then flatten
                            data_array = np.array(output_data)
                            print(f"          Sample values: {data_array.flat[:10]}")
        
        # Check if there are targets/inputs too
        if 'targets' in event:
            print("\n  Targets keys:", list(event['targets'].keys()))
            for target_key in event['targets'].keys():
                target_data = event['targets'][target_key]
                print(f"    {target_key}: shape={target_data.shape}, dtype={target_data.dtype}")
        
        if 'inputs' in event:
            print("\n  Inputs keys:", list(event['inputs'].keys()))
            for input_key in event['inputs'].keys():
                input_data = event['inputs'][input_key]
                print(f"    {input_key}: shape={input_data.shape}, dtype={input_data.dtype}")

if __name__ == "__main__":
    explore_evaluation_file()