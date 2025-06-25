import torch

def inspect_checkpoint(file_path):
    try:
        checkpoint = torch.load(file_path, map_location="cpu")
        print(f"Checkpoint: {file_path}")
        print("Contents:")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  {key}: (dict with {len(value)} keys)")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor, Shape: {value.shape}, Dtype: {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# Replace with your file path
file_path = "/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn/logs/HC-v3_20250622-T103156/ckpts/epoch=046-val_acc=0.96694.ckpt"
inspect_checkpoint(file_path)