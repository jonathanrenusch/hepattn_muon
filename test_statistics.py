#!/usr/bin/env python3
"""
Test script to verify that normalized and absolute residuals use their own statistics
"""
import numpy as np

# Simulate some test data
np.random.seed(42)
truth = np.random.uniform(10, 50, 100)  # Truth values from 10 to 50
noise = np.random.normal(0, 2, 100)     # Add some noise
predictions = truth + noise              # Predictions = truth + noise

# Calculate absolute residuals
absolute_residuals = predictions - truth
abs_mean = np.mean(absolute_residuals)
abs_std = np.std(absolute_residuals)

# Calculate normalized residuals  
normalized_residuals = absolute_residuals / np.abs(truth)
norm_mean = np.mean(normalized_residuals)
norm_std = np.std(normalized_residuals)

print("Test Statistics Verification:")
print("=" * 40)
print(f"Absolute Residuals:")
print(f"  Mean: {abs_mean:.6f}")
print(f"  Std:  {abs_std:.6f}")
print(f"")
print(f"Normalized Residuals:")
print(f"  Mean: {norm_mean:.6f}")
print(f"  Std:  {norm_std:.6f}")
print(f"")
print("These should be different values!")
print(f"Mean difference: {abs(abs_mean - norm_mean):.6f}")
print(f"Std difference:  {abs(abs_std - norm_std):.6f}")

# Test the 3-sigma range calculation
print(f"\n3-sigma ranges:")
print(f"Absolute:   [{abs_mean - 3*abs_std:.3f}, {abs_mean + 3*abs_std:.3f}]")
print(f"Normalized: [{norm_mean - 3*norm_std:.3f}, {norm_mean + 3*norm_std:.3f}]")