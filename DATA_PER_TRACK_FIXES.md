# PerTrackAtlasMuonDataset Index Generation Fixes

## Date: January 31, 2026

## Critical Issues Found and Fixed

### 1. **Hit Counting Ignored Valid Range (CRITICAL BUG)**

**Problem:**
```python
# OLD - WRONG
truth_links = hits['spacePoint_truthLink']  # Includes padding!
num_track_hits = np.sum(truth_links == pid)  # Counts padding as real hits
```

The code counted **all** occurrences of `particle_id` in the `truth_links` array, including padded entries beyond `num_hits`. This caused:
- Inflated hit counts for tracks
- Tracks being included that shouldn't meet `min_hits_per_track` threshold  
- Inconsistent filtering between index build and retrieval

**Fix:**
```python
# NEW - CORRECT
truth_links = hits['spacePoint_truthLink'][:num_hits]  # Only valid hits
num_track_hits = np.sum(truth_links == pid)  # Only counts real hits
```

### 2. **No Bounds Validation on particle_idx**

**Problem:**
When retrieving a track, `particle_idx` was used to index into the particles array without checking if it's still valid. If the data changed or the index became stale, this could cause:
- IndexError crashes
- Silent data corruption (reading wrong particle)

**Fix:**
```python
# Validate particle_idx is within bounds
if particle_idx >= num_tracks:
    raise ValueError(f"Track index corrupt: particle_idx={particle_idx} >= num_tracks={num_tracks}")
```

### 3. **Inconsistent Data Between Index Build and Retrieval**

**Problem:**
The index stored `(event_idx, particle_idx, particle_id)` but didn't validate that the event still has the same structure when retrieved. If `num_tracks` changed, `particle_idx` could point to a different particle.

**Fix:**
- Store `num_tracks` in index: `(event_idx, particle_idx, particle_id, num_tracks)`
- Validate on retrieval:
```python
if num_tracks != expected_num_tracks:
    raise ValueError(f"Track index corrupt: num_tracks changed from {expected_num_tracks} to {num_tracks}")
```

### 4. **Hit Retrieval Also Used Padding**

**Problem:**
```python
# OLD - WRONG
hit_mask = truth_links == particle_id  # Includes padding
track_hits[field] = hits[field][hit_mask]  # Gets padded data
```

When extracting hit features during retrieval, the same padding issue occurred.

**Fix:**
```python
# NEW - CORRECT  
truth_links = hits['spacePoint_truthLink'][:num_hits]  # Only valid
hit_mask = truth_links == particle_id
track_hits[field] = hits[field][:num_hits][hit_mask]  # Only valid data
```

### 5. **Missing Error Handling**

**Problem:**
No validation that:
- The track actually has hits when retrieved
- Required fields exist in the data
- Empty events are handled correctly

**Fix:**
- Skip events with `num_tracks == 0` during index build
- Check `num_track_hits > 0` after retrieval
- Validate required fields exist before use

## Impact Assessment

### **Critical Impact - This Affects All Datasets**

These bugs would cause:

1. **Training Data Leakage**: Tracks with < min_hits might be included due to padding counting
2. **Inconsistent Filtering**: Different hit counts between indexing and retrieval
3. **Silent Corruption**: Wrong particle data could be retrieved without error
4. **Stale Cache Issues**: Cached indices would fail on different data without clear errors

### **Backward Compatibility**

The fix includes backward compatibility for old cached indices:
```python
if len(track_info) == 4:
    # New format with validation
    event_idx, particle_idx, particle_id, expected_num_tracks = track_info
else:
    # Old format - no validation possible
    event_idx, particle_idx, particle_id = track_info
    expected_num_tracks = None
```

Old caches will still work but won't have validation. **Recommendation: Delete old caches and rebuild.**

## Verification Steps

1. **Delete existing track index caches**:
   ```bash
   find /scratch -name "track_index_*.npy" -delete
   ```

2. **Rebuild indices** - first run will rebuild with correct logic

3. **Verify track counts** match between:
   - PerTrackAtlasMuonDataset length
   - Event-level data total tracks (with same filtering)

4. **Check for validation errors** - if you see errors about corrupt indices, the old cache was wrong

## Files Modified

- `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/data_per_track.py`
  - `_build_track_index()`: Fixed hit counting and validation
  - `__getitem__()`: Fixed hit retrieval and added bounds checking
  - `_load_or_build_track_index()`: Updated cache comments

## Recommended Actions

1. ✅ **Delete all cached track indices immediately**
2. ✅ **Rebuild indices** on next run (will happen automatically)
3. ✅ **Re-run training validation** to ensure data consistency
4. ✅ **Compare old vs new** track counts to see impact

## Example of Issues This Could Cause

**Scenario**: Dataset has events with max 600 hits padded array
- Event has `num_hits = 50` actual hits
- Hit array has 600 entries (550 are padding with truthLink=0 or similar)
- Particle ID happens to be 0

**Old behavior**: 
- Counts hits: `np.sum(truth_links == 0)` → finds 550 padded hits!
- Track incorrectly passes `min_hits_per_track` threshold
- Gets included in training despite being invalid

**New behavior**:
- Counts hits: `np.sum(truth_links[:50] == 0)` → finds only real hits
- Track correctly filtered based on actual hit count
- Clean, consistent dataset
