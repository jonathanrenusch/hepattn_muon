# Quick Reference: Config Changes Required

## Five Places to Update in Your Config

### 1. Data Section - Add η to inputs
**Location:** `data.inputs.hit` (around line 25-50)

```yaml
data:
  inputs:
    hit:
      # ... existing fields ...
      - r
      - s
      - theta
      - phi
      - eta  # ← ADD THIS
```

### 2. InputNet Fields - Add η to fields list
**Location:** Inside `input_nets.InputNet.fields` config (around line 145-170)

```yaml
            - class_path: hepattn.models.InputNet
              init_args:
                input_name: hit
                fields:
                  # ... existing 22 fields ...
                  - r
                  - s
                  - theta
                  - phi
                  - eta  # ← ADD THIS (makes 23 total)
```

### 3. InputNet - Increase input_size
**Location:** Inside `input_nets` config (around line 165)

```yaml
net:
  class_path: hepattn.models.Dense
  init_args:
    input_size: 23  # ← CHANGE from 22 to 23
    output_size: *dim
```

### 4. Position Encoding - Add η (OPTIONAL)
**Location:** Inside `input_nets.posenc` config (around line 185)

**Note:** This is optional - only if you want η in position encoding

```yaml
posenc:
  class_path: hepattn.models.posenc.PositionEncoder
  init_args:
    input_name: hit
    dim: *dim
    fields:
      - r
      - theta
      - phi
      - eta  # ← OPTIONALLY ADD THIS
    sym_fields:
      - phi
```

### 5. Task Section - Replace ObjectRegressionTask
**Location:** `tasks` list (around line 271)

**REMOVE:**
```yaml
- class_path: hepattn.models.task.ObjectRegressionTask
  init_args:
    name: parameter_regression
    input_object: query
    output_object: track
    target_object: particle
    fields:
    - truthMuon_eta
    - truthMuon_phi
    - truthMuon_pt
    - truthMuon_q
    loss_weight: 1.0
    dim: *dim
```

**REPLACE WITH:**
```yaml
- class_path: hepattn.models.task.WeightedPoolingObjectHitRegressionTask
  init_args:
    name: parameter_regression
    input_hit: hit
    input_object: query
    output_object: track
    target_object: particle
    hit_fields:
      - spacePoint_globEdgeHighX
      - spacePoint_globEdgeHighY
      - spacePoint_globEdgeHighZ
      - spacePoint_globEdgeLowX
      - spacePoint_globEdgeLowY
      - spacePoint_globEdgeLowZ
      - spacePoint_time
      - spacePoint_driftR
      - spacePoint_covXX
      - spacePoint_covXY
      - spacePoint_covYX
      - spacePoint_covYY
      - spacePoint_channel
      - spacePoint_layer
      - spacePoint_stationPhi
      - spacePoint_stationEta
      - spacePoint_stationIndex
      - spacePoint_technology
      - r
      - s
      - theta
      - phi
      - eta  # ← NEW
    fields:
      - truthMuon_eta
      - truthMuon_phi
      - truthMuon_pt
      - truthMuon_q
    inverse_fields:
      - truthMuon_pt  # ← NEW
    loss_weight: 1.0
    dim: *dim
    assignment_threshold: 0.1
```

## Verification Checklist

- [ ] Added `eta` to `data.inputs.hit`
- [ ] Added `eta` to InputNet `fields` list
- [ ] Changed `input_size: 23` in InputNet (was 22)
- [ ] (Optional) Added `eta` to position encoding fields
- [ ] Added `eta` to regression task `hit_fields`
- [ ] Added `inverse_fields: [truthMuon_pt]` to task
- [ ] Added regression head params (optional: has defaults)
- [ ] Task comes AFTER `ObjectHitMaskTask` in the list
- [ ] Saved config file

## Quick Test

Run a short training (1-2 epochs) and verify:
```bash
# Check that eta values are reasonable
# Should see eta in range roughly [-3, 3]

# Check that pT predictions are in GeV (not inverse)
# Should see pT predictions in range [5, 200] GeV
```

## To Disable Features (for comparison)

**Disable inverse pT regression:**
```yaml
# Simply remove or comment out:
# inverse_fields:
#   - truthMuon_pt
```

**Disable η feature:**
```yaml
# Remove from both data.inputs.hit and task.hit_fields
# And change input_size back to 22
```
