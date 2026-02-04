# Backward Compatibility Alignment Summary

## Date: January 31, 2026

## Overview
Updated `evaluate_mamba_regression.py` to align filtering logic with the backward-compatible version for proper comparison with historical results.

## Key Changes

### 1. Baseline Filtering Criteria (Updated)
**Previous:**
- pt >= 5.0 GeV

**Current (Backward Compatible):**
- pt >= 3.0 GeV
- All other criteria unchanged:
  - >= 9 hits per track
  - 0.1 <= |eta| <= 2.7
  - >= 3 unique stations
  - >= 3 stations with >= 3 hits each

### 2. ML Region Filtering (Added)
**New separate filtering regime:**
- >= 3 hits per track
- |eta| <= 2.7
- pt >= 5.0 GeV

### 3. Rejected Tracks Definition (Updated)
**Previous:**
- Tracks that don't pass baseline filter

**Current (Backward Compatible):**
- Tracks that don't pass ML region filter

This maintains backward compatibility where "rejected" tracks are defined by the ML region boundary (pt < 5.0 GeV) rather than the stricter baseline criteria.

### 4. Output Structure (Enhanced)
**Added:**
- `ml_region_tracks/` subdirectory with all plots
- Separate ML region statistics in output files
- Four-category evaluation: All, Baseline, ML Region, Rejected

### 5. Statistics Tracking (Enhanced)
**Baseline stats now include:**
- `tracks_failed_insufficient_stations` (< 3 stations total)
- `tracks_failed_hits_per_station` (< 3 stations with >= 3 hits)
- Granular breakdown instead of combined "station_cuts"

**ML region stats separately track:**
- Min hits (>= 3)
- Eta cuts (|eta| <= 2.7)
- pT cuts (pt >= 5.0 GeV)

## Backward Compatibility Guarantee

The changes ensure:
1. **Baseline regime**: More inclusive (pt >= 3.0 instead of 5.0), matching historical baseline
2. **ML region**: New explicit category for high-quality tracks (pt >= 5.0)
3. **Rejected tracks**: Defined by ML region boundary, not baseline
4. **Output format**: All previous plots and statistics remain, plus new ML region outputs

## Files Modified
- `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/evaluate_mamba_regression.py`

## Testing Recommendation
Run evaluation on same dataset with both old and new code to verify:
- Baseline track counts match expected behavior
- ML region tracks are subset of baseline tracks
- Rejected tracks = All tracks - ML region tracks
- Statistics align with historical results
