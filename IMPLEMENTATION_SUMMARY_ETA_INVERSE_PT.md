# Implementation Summary: Pseudorapidity and Inverse pT Regression

## Changes Implemented

### 1. Pseudorapidity (η) Feature - Data Preprocessing
**File:** `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/data.py`

**Change:** Added pseudorapidity as a derived hit feature
```python
# Add pseudorapidity (eta) derived from theta
# eta = -ln(tan(theta/2))
hits["eta"] = -np.log(np.tan(hits["theta"] / 2.0))
```

**Location:** After computing `theta` and `phi` (around line 272)

**Physics:**
- η = -ln(tan(θ/2))
- More linear with particle trajectories than θ
- Symmetric around 0, typically ranges from -3 to +3
- The actual physics quantity you're predicting for track η

### 2. Inverse pT Regression - Task Implementation
**File:** `/shared/tracking/hepattn_muon/src/hepattn/models/task.py`

**Changes:**

#### a) Added `inverse_fields` parameter to `__init__`:
```python
def __init__(
    self,
    ...
    inverse_fields: list[str] | None = None,
):
    self.inverse_fields = inverse_fields or []
    self.inverse_indices = [i for i, f in enumerate(fields) if f in self.inverse_fields]
```

#### b) Overrode `predict()` method:
- Network outputs 1/pT
- Predictions automatically converted back to pT
- Clamping for numerical stability

```python
def predict(self, outputs):
    for i, field in enumerate(self.fields):
        pred_value = raw_output[..., i]
        if i in self.inverse_indices:
            pred_value = 1.0 / pred_value.clamp(min=1e-6, max=1e6)
        predictions[self.output_object + "_" + field] = pred_value
    return predictions
```

#### c) Overrode `loss()` method:
- Targets converted to inverse space (pT → 1/pT)
- Loss computed in 1/pT space
- Fully transparent to user

```python
def loss(self, outputs, targets):
    for i, field in enumerate(self.fields):
        target_field = targets[self.target_object + "_" + field]
        if i in self.inverse_indices:
            target_field = 1.0 / target_field
        target_list.append(target_field)
    # Compute smooth L1 loss in inverse space
```

## Why These Changes?

### Pseudorapidity (η)
1. **Direct feature engineering**: Network sees the quantity it needs to predict
2. **Physics-motivated**: η is the natural coordinate in detector geometry
3. **Better ML properties**: More uniform distribution than θ

### Inverse pT (1/pT)
1. **Physics relationship**: Track curvature ∝ 1/pT (Lorentz force law)
2. **Better dynamic range**: pT [5, 200] GeV → 1/pT [0.005, 0.2] GeV⁻¹
3. **Improved resolution**: High-pT tracks (harder to distinguish) get more precision
4. **More uniform distribution**: 1/pT typically more Gaussian than pT

## Configuration Required

### 1. Update Data Config (add η to inputs):
```yaml
data:
  inputs:
    hit:
      # ... existing 22 fields ...
      - r
      - s
      - theta
      - phi
      - eta  # ADD THIS LINE
```

### 2. Update Task Config:
```yaml
- class_path: hepattn.models.task.WeightedPoolingObjectHitRegressionTask
  init_args:
    name: parameter_regression
    hit_fields:
      # ... existing 22 fields ...
      - theta
      - phi
      - eta  # ADD THIS (23 total)
    fields:
      - truthMuon_eta
      - truthMuon_phi
      - truthMuon_pt
      - truthMuon_q
    inverse_fields:
      - truthMuon_pt  # ADD THIS LINE
    dim: 32
```

### 3. Update InputNet (increase input_size):
```yaml
net:
  class_path: hepattn.models.Dense
  init_args:
    input_size: 23  # Changed from 22 to 23
    output_size: *dim
```

## Testing Strategy

1. **Verify η computation**: Check that `eta` values are reasonable (-3 to +3 range)
2. **Test inverse pT**: 
   - Print some predictions to verify they're in pT space (not 1/pT)
   - Check that loss is computed in 1/pT space (gradients should be different)
3. **Compare performance**: Train with and without `inverse_fields` to quantify improvement

## Expected Improvements

### For η regression:
- Direct feature access should improve predictions
- Expect better resolution especially at high |η|

### For pT regression:
- Better learning dynamics (more uniform loss landscape)
- Improved high-pT resolution (currently most challenging)
- More stable training (smaller gradients for extreme values)

## Backward Compatibility

- **Pseudorapidity**: Fully backward compatible, just add the field to configs
- **Inverse pT**: Opt-in via `inverse_fields` parameter, defaults to normal regression

## Files Modified

1. ✅ `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/data.py`
   - Added η computation

2. ✅ `/shared/tracking/hepattn_muon/src/hepattn/models/task.py`
   - Added `inverse_fields` parameter
   - Overrode `predict()` method
   - Overrode `loss()` method

3. ✅ `/shared/tracking/hepattn_muon/config_snippet_weighted_pooling_regression.yaml`
   - Updated with η and inverse_fields

4. ✅ `/shared/tracking/hepattn_muon/WEIGHTED_POOLING_REGRESSION_TASK.md`
   - Updated documentation

## Next Steps

1. Update your full config file with:
   - `eta` in `data.inputs.hit`
   - `eta` in task `hit_fields`
   - `truthMuon_pt` in task `inverse_fields`
   - Update `input_size: 23` in InputNet

2. Test on a small subset to verify:
   - η values look reasonable
   - pT predictions are in GeV (not GeV⁻¹)
   - Training converges

3. Full training run and compare metrics with baseline
