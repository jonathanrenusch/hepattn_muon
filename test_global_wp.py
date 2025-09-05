#!/usr/bin/env python3
"""Test script to verify global working points."""

import sys
import os
sys.path.append('/shared/tracking/hepattn_muon/src')

from hepattn.experiments.atlas_muon.evaluate_hit_filter_dataloader import DEFAULT_WORKING_POINTS

print('Global working points:', DEFAULT_WORKING_POINTS)
print('Length:', len(DEFAULT_WORKING_POINTS))
print('Type:', type(DEFAULT_WORKING_POINTS))

# Test that it has the expected values
expected = [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
print('Expected:', expected)
print('Match:', DEFAULT_WORKING_POINTS == expected)
