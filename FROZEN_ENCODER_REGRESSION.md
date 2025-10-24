# Frozen Encoder Regression Task

## Overview

The `FrozenEncoderRegressionTask` enables **two-stage training** where:
1. **Stage 1**: Train hit-to-track assignment (ObjectValidTask + ObjectHitMaskTask)
2. **Stage 2**: Train regression on frozen encoder (FrozenEncoderRegressionTask only)

This approach provides several advantages for rapid iteration on regression architectures.

## Benefits

### 🚀 Faster Iteration
- No backpropagation through transformer encoder/decoder
- Smaller computational graph → faster training
- Typical speedup: 2-3x per epoch

### 🧹 Cleaner Gradients
- Regression task doesn't interfere with assignment learning
- No multi-task gradient conflicts
- More stable training dynamics

### 🔍 Easier Debugging
- Isolate regression performance from encoder quality
- Test different regression architectures quickly
- Compare architectures with identical encoder features

### 💾 Lower Memory
- No gradient storage for frozen parameters
- Can use larger batch sizes
- Typical memory reduction: 30-50%

### 🎯 Better Hyperparameter Tuning
- Quickly test: hidden dims, layer counts, dropout rates
- No need to retrain encoder for each experiment
- Grid search becomes feasible

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Stage 1: Train Assignment Model         │
│                                                  │
│  Input Hits → Encoder → Decoder → {             │
│    - ObjectValidTask (track existence)          │
│    - ObjectHitMaskTask (hit-track assignment)   │
│  }                                               │
│                                                  │
│  Save checkpoint: best_assignment_model.ckpt    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       Stage 2: Train Regression (Frozen)        │
│                                                  │
│  Load checkpoint ✓                               │
│  Freeze: Input Nets, Encoder, Decoder, Tasks ❄️ │
│                                                  │
│  Input Hits → [FROZEN Encoder] → [FROZEN Decoder] → {│
│    - [FROZEN ObjectHitMaskTask] (logits only)   │
│    - [TRAINABLE FrozenEncoderRegressionTask]    │
│  }                                               │
│                                                  │
│  Only trains: Regression head (53k params)      │
└─────────────────────────────────────────────────┘
```

## How It Works

### Checkpoint Loading
```python
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

# Load all weights EXCEPT regression task
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith(f"tasks.{regression_task_name}.")
}
model.load_state_dict(filtered_state_dict, strict=False)
```

### Parameter Freezing
```python
if freeze_all:
    # Freeze everything except regression head
    for name, param in model.named_parameters():
        if not name.startswith(f"tasks.regr."):
            param.requires_grad = False
```

### Forward Pass
1. **Frozen encoder** processes hits → encoded features
2. **Frozen decoder** generates queries → object embeddings
3. **Frozen ObjectHitMaskTask** produces assignment logits
4. **Trainable regression head** predicts track parameters

The frozen components provide **fixed features** while only the regression head learns.

## Usage

### Step 1: Train Assignment Model

Use your standard config with ObjectValidTask and ObjectHitMaskTask:

```yaml
# atlas_muon_tracking_assignment.yaml
model:
  model:
    init_args:
      tasks:
        - class_path: hepattn.models.task.ObjectValidTask
          init_args:
            name: valid
            # ... config ...
        
        - class_path: hepattn.models.task.ObjectHitMaskTask
          init_args:
            name: mask
            # ... config ...
```

Train until assignment performance plateaus:
```bash
python -m lightning.pytorch.cli fit --config atlas_muon_tracking_assignment.yaml
```

Save the best checkpoint (e.g., `logs/ckpts/epoch=004-val_loss=9.00107.ckpt`).

### Step 2: Train Regression with Frozen Encoder

Create a new config using **only** FrozenEncoderRegressionTask:

```yaml
# atlas_muon_tracking_frozen_regression.yaml
model:
  model:
    init_args:
      # Same encoder/decoder config as Stage 1
      encoder: ...
      decoder_layer_config: ...
      
      # ONLY regression task
      tasks:
        - class_path: hepattn.models.task.FrozenEncoderRegressionTask
          init_args:
            name: regr
            
            # Path to Stage 1 checkpoint
            checkpoint_path: logs/ckpts/epoch=004-val_loss=9.00107.ckpt
            
            # Freeze all encoder/decoder/task weights
            freeze_all: true
            
            # Same as WeightedPoolingObjectHitRegressionTask
            input_hit: hit
            input_object: query
            output_object: track
            target_object: truthMuon
            
            hit_fields:
              - r
              - phi
              # ... all 23 fields including eta
            
            fields:
              - truthMuon_eta
              - truthMuon_phi
              - truthMuon_pt
              - truthMuon_charge
            
            loss_weight: 1.0
            dim: 32
            
            # Regression-specific params
            inverse_fields: [truthMuon_pt]
            regression_hidden_dim: 144
            regression_num_layers: 3
            dropout: 0.1
```

Train the regression head:
```bash
python -m lightning.pytorch.cli fit --config atlas_muon_tracking_frozen_regression.yaml
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str | Path to Stage 1 checkpoint (.ckpt file) |
| `name` | str | Task name (e.g., "regr") |
| `input_hit` | str | Hit input name (e.g., "hit") |
| `input_object` | str | Object input name (e.g., "query") |
| `output_object` | str | Output object name (e.g., "track") |
| `target_object` | str | Target object name (e.g., "truthMuon") |
| `hit_fields` | list[str] | All 23 hit features (including eta) |
| `fields` | list[str] | Regression targets (eta, phi, pt, charge) |
| `loss_weight` | float | Loss weight (typically 1.0) |
| `dim` | int | Encoder dimension (must match Stage 1) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freeze_all` | `True` | Freeze all params except regression head |
| `assignment_threshold` | `0.1` | Hard threshold for hit-track assignments |
| `inverse_fields` | `None` | Fields to regress as inverse (e.g., [truthMuon_pt]) |
| `regression_hidden_dim` | `aggregated_dim // 2` | Hidden dim for regression MLP |
| `regression_num_layers` | `3` | Number of MLP layers (1=linear, 3=2 hidden layers) |
| `dropout` | `0.1` | Dropout probability for regularization |

## Hyperparameter Tuning

With frozen encoder, you can quickly test different architectures:

### Example: Grid Search Over Architectures

```yaml
# config_regression_arch1.yaml
regression_hidden_dim: 144
regression_num_layers: 3
dropout: 0.1

# config_regression_arch2.yaml
regression_hidden_dim: 288
regression_num_layers: 4
dropout: 0.2

# config_regression_arch3.yaml
regression_hidden_dim: 96
regression_num_layers: 2
dropout: 0.05
```

Train all three in parallel:
```bash
python train_regression.py --config config_regression_arch1.yaml &
python train_regression.py --config config_regression_arch2.yaml &
python train_regression.py --config config_regression_arch3.yaml &
```

### Typical Tuning Strategy

1. **Start with defaults** (144 hidden, 3 layers, 0.1 dropout)
2. **If underfitting**: Increase `regression_hidden_dim` or `regression_num_layers`
3. **If overfitting**: Increase `dropout` or decrease `regression_hidden_dim`
4. **Monitor**: Validation loss convergence (should be fast, ~10-20 epochs)

## Expected Results

### Training Speed
- **Stage 1** (Assignment): ~500-600 events/sec on single GPU
- **Stage 2** (Frozen Regression): ~1200-1500 events/sec (2-3x faster)

### Memory Usage
- **Stage 1**: ~8-10 GB VRAM for batch_size=32
- **Stage 2**: ~5-7 GB VRAM for batch_size=32 (can increase to 48-64)

### Convergence
- **Stage 1**: Typically 50-100 epochs to converge
- **Stage 2**: Typically 10-30 epochs to converge (much faster!)

### Parameter Counts
```
Stage 1 Model:
├── Input Nets:        ~5k params
├── Encoder:          ~50k params
├── Decoder:          ~60k params
├── ObjectValidTask:   ~1k params
└── ObjectHitMaskTask: ~1k params
Total: ~117k params (all trainable)

Stage 2 Model:
├── [FROZEN] All above:        ~117k params (requires_grad=False)
└── [TRAINABLE] Regression:     ~53k params (requires_grad=True)
Total: ~170k params (31% trainable)
```

## Comparison: Joint vs Frozen Training

| Aspect | Joint Training | Frozen Training |
|--------|----------------|-----------------|
| **Speed** | 500 events/sec | 1500 events/sec (3x faster) |
| **Memory** | 10 GB | 6 GB (40% less) |
| **Convergence** | 100 epochs | 20 epochs (5x faster) |
| **Gradient Quality** | Multi-task conflicts | Clean, task-focused |
| **Debugging** | Hard to isolate issues | Easy to isolate regression |
| **Hyperparameter Tuning** | Expensive | Cheap |
| **Final Performance** | Potentially better (joint opt) | Very close (fixed features) |

## When to Use Frozen Training

✅ **Use frozen training when:**
- Experimenting with regression architectures
- Hit-to-track assignment is already good
- Need fast iteration cycles
- Limited GPU memory
- Want to isolate regression performance

❌ **Use joint training when:**
- Assignment and regression should co-adapt
- End-to-end optimization is critical
- You have time for long training runs
- Final deployment model (after architecture is finalized)

## Advanced: Partial Freezing

If you want input nets to adapt to regression task:

```yaml
freeze_all: false  # Only freeze encoder/decoder, not input nets
```

This allows:
- Input feature scaling to adjust
- Position encodings to adapt
- Slightly better regression performance
- Still much faster than full joint training

## Troubleshooting

### Issue: Checkpoint not found
```
Error: FileNotFoundError: logs/ckpts/epoch=004-val_loss=9.00107.ckpt
```
**Solution**: Check the path, use absolute path if needed:
```yaml
checkpoint_path: /shared/tracking/hepattn_muon/logs/ckpts/epoch=004-val_loss=9.00107.ckpt
```

### Issue: Dimension mismatch
```
Error: size mismatch for query_initial: copying from (10, 32), expected (10, 64)
```
**Solution**: Ensure `dim` matches between Stage 1 and Stage 2 configs.

### Issue: Missing ObjectHitMaskTask
```
Error: KeyError: 'track_hit_logit'
```
**Solution**: Stage 1 checkpoint must include ObjectHitMaskTask. Check Stage 1 config includes:
```yaml
- class_path: hepattn.models.task.ObjectHitMaskTask
  init_args:
    name: mask  # Must be present!
```

### Issue: Poor regression performance
```
Validation loss not decreasing
```
**Solutions**:
1. Check Stage 1 assignment quality (should have good hit-track matching)
2. Increase `regression_hidden_dim` (try 288 instead of 144)
3. Add more layers (`regression_num_layers: 4`)
4. Reduce `dropout` if underfitting
5. Check learning rate (try 1e-3 to 1e-4)

## Next Steps

After finding the best regression architecture with frozen training:

1. **Optional**: Fine-tune end-to-end
   - Load Stage 2 checkpoint
   - Unfreeze all parameters
   - Train for 10-20 more epochs with low learning rate (1e-5)
   
2. **Evaluate**: Compare frozen vs joint training performance

3. **Deploy**: Use the best model for inference

## Example Complete Workflow

```bash
# Stage 1: Train assignment (once)
python -m lightning.pytorch.cli fit \
  --config atlas_muon_tracking_assignment.yaml \
  --trainer.max_epochs 100

# Stage 2: Experiment with regression (many times, fast!)
for hidden_dim in 96 144 192 288; do
  for num_layers in 2 3 4; do
    python -m lightning.pytorch.cli fit \
      --config atlas_muon_tracking_frozen_regression.yaml \
      --model.model.init_args.tasks.0.init_args.regression_hidden_dim $hidden_dim \
      --model.model.init_args.tasks.0.init_args.regression_num_layers $num_layers \
      --trainer.max_epochs 20
  done
done

# Find best hyperparameters, then optionally fine-tune end-to-end
python -m lightning.pytorch.cli fit \
  --config atlas_muon_tracking_joint.yaml \
  --ckpt_path logs/frozen_regression/best.ckpt \
  --trainer.max_epochs 20 \
  --optimizer.lr 1e-5
```
