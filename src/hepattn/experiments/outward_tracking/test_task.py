#!/usr/bin/env python
"""Quick test to verify OutwardEdgeTask import and basic functionality."""

import sys
sys.path.insert(0, 'src')

import torch
from hepattn.models.task import OutwardEdgeTask

print("OutwardEdgeTask imported successfully")

# Test instantiation
task = OutwardEdgeTask(
    name="edge",
    input_object="hit",
    dim=128,
    hidden_dim=64,
    threshold=0.5,
)
print("Task instantiated successfully")

# Test forward pass
B, N, D = 2, 10, 128
x = {"hit_embed": torch.randn(B, N, D)}
outputs = task.forward(x)
print(f"Forward pass successful, output shape: {outputs['hit_outward_edge_logit'].shape}")

# Test prediction
preds = task.predict(outputs)
print(f"Prediction successful, edge shape: {preds['hit_outward_edge'].shape}")

# Test loss (with dummy targets)
targets = {
    "outward_adjacency": torch.zeros(B, N, N, dtype=torch.bool),
    "hit_valid": torch.ones(B, N, dtype=torch.bool),
}
# Add a few true edges
targets["outward_adjacency"][0, 0, 1] = True
targets["outward_adjacency"][0, 1, 2] = True
targets["outward_adjacency"][1, 0, 1] = True

losses = task.loss(outputs, targets)
print(f"Loss computation successful: {losses}")

# Test track extraction
edge_probs = outputs['hit_outward_edge_logit'].sigmoid()
tracks = task.extract_tracks(edge_probs, targets["hit_valid"])
print(f"Track extraction successful: {len(tracks)} batch items")

print("\nAll tests passed!")
