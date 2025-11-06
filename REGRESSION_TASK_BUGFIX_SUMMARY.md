# WeightedPoolingObjectHitRegressionTask Bug Fixes

## Summary of Issues Found and Fixed

The `WeightedPoolingObjectHitRegressionTask` had **three critical bugs** that could cause training to stagnate or fail:

### 1. **NaN Loss from Empty Valid Masks** (CRITICAL)
**Problem:** When no valid particles exist in a batch (all `particle_valid = False`), the loss computation would call `.mean()` on an empty tensor, returning NaN.

**Impact:** 
- NaN loss propagates through backpropagation, producing NaN gradients
- Optimizer updates with NaN gradients cause all parameters to become NaN
- Training completely stalls with flat loss curves

**Fix in `loss()` method:**
```python
# Before (line ~860):
return {"smooth_l1": self.loss_weight * loss.mean()}

# After:
num_valid = mask.sum()
if num_valid > 0:
    return {"smooth_l1": self.loss_weight * loss.sum() / num_valid}
else:
    # Return zero loss when no valid targets (gradient will be zero, no update)
    return {"smooth_l1": torch.tensor(0.0, device=output.device, dtype=output.dtype, requires_grad=True)}
```

### 2. **Division by Zero in Inverse Field Regression** (CRITICAL)
**Problem:** When converting pT to 1/pT for regression, if pT = 0 (or very small), we get division by zero → infinity.

**Impact:**
- Infinite target values cause extremely large or infinite losses
- Gradients become unstable or NaN
- Can cause training divergence

**Fix in `loss()` method:**
```python
# Before (line ~841):
if i in self.inverse_indices:
    target_field = 1.0 / target_field

# After:
if i in self.inverse_indices:
    # Clamp to avoid division by zero (pT should never be exactly 0 in practice)
    target_field = 1.0 / target_field.clamp(min=1e-6)
```

### 3. **Infinity Values from Empty Hit Assignments** (MODERATE)
**Problem:** When no hits are assigned to a track (all assignment probabilities < threshold), max pooling returns `-inf` and min pooling returns `+inf`.

**Impact:**
- Infinity values propagate through regression head
- Can cause NaN after operations like inf - inf
- Network learns to always output infinity in certain situations

**Fix in `latent()` method (applied to both raw and encoded features):**
```python
# After max pooling (line ~718):
raw_max = raw_for_max.max(dim=2)[0]
# NEW: Replace -inf with 0 for tracks with no assigned hits
raw_max = torch.where(torch.isinf(raw_max), torch.zeros_like(raw_max), raw_max)

# After min pooling (line ~726):
raw_min = raw_for_min.min(dim=2)[0]
# NEW: Replace +inf with 0 for tracks with no assigned hits
raw_min = torch.where(torch.isinf(raw_min), torch.zeros_like(raw_min), raw_min)

# (Same fixes applied to encoded_max and encoded_min)
```

## Testing Results

All tests pass after fixes:

1. ✓ Forward pass with random inputs
2. ✓ Loss computation with valid targets
3. ✓ Gradient computation (all non-zero, no NaN)
4. ✓ Edge case: all tracks invalid (returns 0.0 instead of NaN)
5. ✓ Prediction with inverse fields
6. ✓ No hits assigned scenario
7. ✓ Very low pT targets (high 1/pT)
8. ✓ Mixed valid/invalid across batch
9. ✓ Mini training loop (10 steps, loss decreases)

## Recommendation for Training

**Before starting full training:**

1. Monitor initial batches for NaN/Inf values
2. Check that pT values in your dataset are reasonable (> 1e-6 GeV)
3. Consider adding gradient clipping if training is still unstable
4. The loss might be initially high due to 1/pT regression - this is normal

**Expected behavior:**
- Loss should now be finite in all cases
- Gradients should flow properly
- Training should progress (loss should decrease)
- No more flat loss curves due to NaN propagation

## Files Modified

- `/shared/tracking/hepattn_muon/src/hepattn/models/task.py`:
  - Fixed `WeightedPoolingObjectHitRegressionTask.loss()` (lines ~822-869)
  - Fixed `WeightedPoolingObjectHitRegressionTask.latent()` (lines ~710-762)
