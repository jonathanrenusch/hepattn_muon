import torch
n = torch.cuda.device_count()
print(f"CUDA devices: {n}")
for i in range(n):
    print(f"  [{i}]: {torch.cuda.get_device_name(i)}")
