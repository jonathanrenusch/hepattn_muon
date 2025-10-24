# WeightedPoolingObjectHitRegressionTask

## Overview

A new regression task that aggregates hit information using weighted pooling based on hit-track assignment probabilities. This task directly incorporates low-level detector hit information for improved track parameter estimation.

## New Features

### 1. Pseudorapidity (η) as Input Feature
Added `eta` as a derived hit feature, calculated from `theta`:
```python
eta = -log(tan(theta / 2))
```

**Benefits:**
- Direct access to the physics quantity being predicted
- More linear relationship with particle trajectories than θ
- Better behaved for machine learning (symmetric around 0)

### 2. Inverse pT Regression
The task can regress `1/pT` instead of `pT` for better ML performance:

**Physics Motivation:**
- Track curvature in magnetic field is proportional to `1/pT`
- Better dynamic range: pT ∈ [5, 200] GeV → 1/pT ∈ [0.005, 0.2] GeV⁻¹
- High-pT tracks get more resolution in inverse space

**Implementation:**
- Network predicts `1/pT` internally
- Loss computed in inverse space
- Predictions automatically converted back to `pT`
- Metrics computed in original `pT` space
- **Fully transparent to the user!**

## Architecture

```
For each track candidate:
1. Get assignment probabilities from ObjectHitMaskTask → sigmoid(logits) [B, N, M]
2. Apply hard threshold (default 0.1) → binary mask
3. Process raw hits through task-specific network → raw_features [B, M, D/2]
4. Process encoded hits through task-specific network → encoded_features [B, M, D/2]
5. Apply pooling operations:
   - Weighted average (uses assignment probabilities as weights)
   - Weighted sum (total contribution scaled by confidence)
   - Max pooling (unweighted, geometric maximum over thresholded hits)
   - Min pooling (unweighted, geometric minimum over thresholded hits)
6. Concatenate: [query_embed, raw_pooled, encoded_pooled] → [B, N, 5*D]
7. Final regression head → [B, N, num_targets]
```

## Key Features

- **Hybrid approach**: Uses both raw detector features AND encoded embeddings
- **Hard thresholding**: Only hits with assignment probability ≥ 0.1 contribute
- **Multiple pooling**: Captures different statistical properties of hit distributions
- **Weighted pooling**: Uses assignment confidence for avg/sum operations
- **Unweighted extrema**: Max/min pooling preserves geometric boundaries

## Configuration

Replace your existing `ObjectRegressionTask` in the config with:

```yaml
- class_path: hepattn.models.task.WeightedPoolingObjectHitRegressionTask
  init_args:
    name: parameter_regression
    input_hit: hit
    input_object: query
    output_object: track
    target_object: particle
    hit_fields:
      # Raw hit features (23 total - added eta!)
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
      - eta  # NEW: Pseudorapidity
    fields:
      # Regression targets
      - truthMuon_eta
      - truthMuon_phi
      - truthMuon_pt
      - truthMuon_q
    inverse_fields:
      - truthMuon_pt  # NEW: Regress as 1/pT
    loss_weight: 1.0
    dim: 32  # Must match encoder dim
    assignment_threshold: 0.1  # Optional, defaults to 0.1
```

## Important Notes

1. **Task Order**: This task MUST come AFTER `ObjectHitMaskTask` in the task list, since it uses the assignment logits from that task

2. **Raw Hit Fields**: The `hit_fields` list must exactly match the raw hit feature names in your data (not the embedded versions). **Note:** Now includes 23 features with the new `eta` field.

3. **Dimension**: The `dim` parameter must match your encoder's embedding dimension (32 in your config)

4. **Assignment Threshold**: Default is 0.1 (same as ObjectHitMaskTask attention masking threshold). You can tune this as a hyperparameter.

5. **Inverse pT Regression**: Specify `inverse_fields: [truthMuon_pt]` to enable 1/pT regression. The network learns in inverse space, but all predictions and metrics are in original pT space (GeV).

6. **Pseudorapidity**: The `eta` feature is automatically computed in `data.py` from the `theta` angle. Make sure to add it to both:
   - `hit_fields` in the task config
   - `inputs.hit` in the data config

7. **Memory**: This task creates expanded tensors for broadcasting, so it uses more memory than the simple ObjectRegressionTask. With your settings (batch_size=200, max 2 tracks, max 600 hits), this should be fine.

## Expected Benefits

Based on your description of 90% hit assignment purity and efficiency:

1. **Better geometric precision**: Direct access to hit positions and uncertainties
2. **Reduced noise**: Hard thresholding filters out low-confidence assignments
3. **Rich feature space**: Multiple pooling operations capture different aspects of hit distributions
4. **Context awareness**: Encoded hits provide global event context
5. **Improved η regression**: Network sees pseudorapidity directly as input feature
6. **Better pT resolution**: Inverse pT regression leverages physics (curvature ∝ 1/pT) for improved learning, especially for high-pT tracks

## Example Task Configuration (Full)

```yaml
tasks:
  class_path: torch.nn.ModuleList
  init_args:
    modules:
      # 1. Track validity classification
      - class_path: hepattn.models.task.ObjectValidTask
        init_args:
          name: track_valid
          input_object: query
          output_object: track
          target_object: particle
          losses:
            object_bce: 1.0
          costs:
            object_bce: 10.0
          dim: 32
          null_weight: 1.0
          mask_queries: false
      
      # 2. Hit-to-track assignment
      - class_path: hepattn.models.task.ObjectHitMaskTask
        init_args:
          name: track_hit_valid
          input_hit: hit
          input_object: query
          output_object: track
          target_object: particle
          losses:
            mask_bce: 1.0
          costs:
            mask_bce: 1.0
          dim: 32
          null_weight: 1.0
          mask_attn: true
      
      # 3. Track parameter regression (NEW!)
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
            - eta  # NEW!
          fields:
            - truthMuon_eta
            - truthMuon_phi
            - truthMuon_pt
            - truthMuon_q
          inverse_fields:
            - truthMuon_pt  # NEW!
          loss_weight: 1.0
          dim: 32
          assignment_threshold: 0.1
```

## Hyperparameter Tuning

Consider tuning:
- `assignment_threshold`: Try 0.1 (default), 0.2, 0.3, 0.5
- `loss_weight`: Balance regression loss vs classification losses
- `inverse_fields`: Can disable by removing to compare performance
- Architecture: Could add dropout or layer norm in the regression head if needed

## Data Configuration Changes

**IMPORTANT:** You must also add `eta` to your data config's input fields:

```yaml
data:
  inputs:
    hit:
      # ... existing fields ...
      - r
      - s
      - theta
      - phi
      - eta  # ADD THIS!
```

The `eta` feature is automatically computed in `data.py` from the `theta` angle using:
```python
eta = -log(tan(theta / 2))
```

## Implementation Details

The task is implemented in `/shared/tracking/hepattn_muon/src/hepattn/models/task.py` as `WeightedPoolingObjectHitRegressionTask`.

Key methods:
- `latent()`: Performs the weighted pooling and aggregation
- `forward()`: Calls latent() and applies regression head
- Inherits `loss()`, `predict()`, and `metrics()` from `RegressionTask`
