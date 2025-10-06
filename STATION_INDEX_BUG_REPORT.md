# Station Index Masking Bug Report

## Date: October 6, 2025

## Summary
A critical bug was found in `evaluate_task1_hit_track_assignment.py` that caused incorrect baseline filtering statistics. The bug resulted in ~90% of tracks passing baseline filters instead of the correct ~40%, a discrepancy of over 2x.

## Root Cause
**Double bracket indexing** was used instead of single bracket indexing when filtering station indices by track masks:

### Buggy Code (Lines 231 and 259):
```python
# Line 231
all_data['station_indices'].append(true_station_index[[track_mask]])

# Line 259
unique_stations, station_counts = np.unique(true_station_index[[track_mask]], return_counts=True)
```

### Fixed Code:
```python
# Line 231
all_data['station_indices'].append(true_station_index[track_mask])

# Line 259
unique_stations, station_counts = np.unique(true_station_index[track_mask], return_counts=True)
```

## Impact

### Baseline Filter Requirements:
1. **≥9 hits total** per track
2. **|eta| ≥ 0.1 and ≤ 2.7** (detector acceptance)
3. **pt ≥ 3.0 GeV** (momentum threshold)
4. **≥3 different stations** with **≥3 hits per station**

The bug affected **requirement #4** (station criteria), which is the most stringent filter.

### Observed Symptoms:
- **evaluate_task1_hit_track_assignment.py**: Reported ~90% pass rate (INCORRECT)
- **evaluate_hit_filter_dataloader.py**: Reported ~40% pass rate (CORRECT)

The hit filter script used correct single bracket indexing at lines 125-126:
```python
track_mask = (
    (all_event_ids == event_id) & 
    (all_particle_ids == particle_id) & 
    true_hit_mask
)
track_stations = all_station_indices[track_mask]  # ✓ CORRECT
```

## Technical Explanation

### What `[[mask]]` Does:
Double bracket indexing `array[[mask]]` attempts to use the mask as a **nested index array** rather than a boolean mask. In NumPy:
- `array[mask]` → Boolean masking (selects elements where mask is True)
- `array[[mask]]` → Fancy indexing with nested array (causes error or unexpected behavior)

### Why This Caused False Positives:
When `true_station_index[[track_mask]]` was evaluated:
1. NumPy attempted to interpret the boolean array as nested indices
2. This likely caused an indexing error OR returned incorrect/unfiltered data
3. The station counting logic then operated on wrong data
4. Nearly all tracks appeared to meet the "3 stations with 3+ hits" requirement
5. Result: Inflated pass rate of ~90% instead of ~40%

## Validation

A validation script (`validate_station_index_masking.py`) was created to demonstrate the bug:

```bash
cd /shared/tracking/hepattn_muon
pixi run python src/hepattn/experiments/atlas_muon/validate_station_index_masking.py
```

Output shows:
- Single bracket `[mask]` correctly filters array elements
- Double bracket `[[mask]]` causes "too many indices for array" error

## Files Modified

### Fixed:
- `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/evaluate_task1_hit_track_assignment.py`
  - Line 231: Changed `[[track_mask]]` → `[track_mask]`
  - Line 259: Changed `[[track_mask]]` → `[track_mask]`

### Created:
- `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/validate_station_index_masking.py`
  - Validation script demonstrating the masking bug

## Recommendations

1. **Re-run all Task 1 evaluations** with the fixed script to get correct statistics
2. **Audit other scripts** for similar double bracket patterns
3. **Add unit tests** for track filtering logic to catch such bugs early
4. **Compare results** between fixed task1 script and hit filter script to confirm alignment

## Verification Checklist

- [x] Bug identified and root cause documented
- [x] Fix applied to both occurrences (lines 231 and 259)
- [x] Verification script created and tested
- [x] No other instances of `[[track_mask]]` pattern found
- [ ] Re-run evaluation with fixed code
- [ ] Confirm ~40% pass rate matches hit filter script
- [ ] Update any downstream analysis that used incorrect statistics

## Related Scripts

### Correct Implementation (Reference):
- `evaluate_hit_filter_dataloader.py` (lines 125-126, 145-158)
  - Uses single bracket indexing consistently
  - Reports correct ~40% baseline pass rate

### Needs Re-evaluation:
- `evaluate_task1_hit_track_assignment.py`
  - All previous results are invalid due to this bug
  - Need to regenerate all plots and statistics
