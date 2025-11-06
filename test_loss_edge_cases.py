#!/usr/bin/env python
"""Test edge cases in loss computation."""

import torch

print("Testing empty tensor mean:")
empty = torch.tensor([])
print(f"  empty.mean() = {empty.mean()}")
print(f"  Is NaN: {torch.isnan(empty.mean())}")

print("\nTesting smooth_l1_loss with empty tensors:")
loss = torch.nn.functional.smooth_l1_loss(empty, empty, reduction="none")
print(f"  loss shape: {loss.shape}")
print(f"  loss.mean() = {loss.mean()}")
print(f"  Is NaN: {torch.isnan(loss.mean())}")

print("\nTesting with clamp_min on mean:")
safe_mean = empty.mean().clamp(min=0.0)
print(f"  empty.mean().clamp(min=0.0) = {safe_mean}")
print(f"  Is NaN: {torch.isnan(safe_mean)}")

print("\nTesting alternative: sum / max(count, 1):")
count = empty.numel()
safe_mean2 = empty.sum() / max(count, 1)
print(f"  empty.sum() / max(count, 1) = {safe_mean2}")
print(f"  Is NaN: {torch.isnan(safe_mean2)}")

print("\nTesting inverse of zero (pT edge case):")
pt_zero = torch.tensor([0.0])
inv_pt = 1.0 / pt_zero
print(f"  1.0 / 0.0 = {inv_pt}")
print(f"  Is Inf: {torch.isinf(inv_pt)}")

print("\nTesting inverse with clamp:")
pt_small = torch.tensor([1e-10])
inv_pt_clamped = 1.0 / pt_small.clamp(min=1e-6)
print(f"  1.0 / clamp(1e-10, min=1e-6) = {inv_pt_clamped}")
