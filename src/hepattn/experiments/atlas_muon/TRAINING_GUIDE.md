# ATLAS Muon Tracking Training Guide

This comprehensive guide walks you through the complete training pipeline for the ATLAS muon tracking model, from raw data to trained model evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Data Types and Format](#data-types-and-format)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Batching](#data-batching)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Evaluation](#evaluation)
9. [Step-by-Step Workflow](#step-by-step-workflow)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The ATLAS muon tracking system uses a **MaskFormer** architecture (encoder-decoder transformer) to solve multiple reconstruction tasks:
- **Track validity prediction**: Identifying valid particle tracks
- **Hit-to-track assignment**: Associating detector hits with tracks
- **Parameter regression**: Predicting track kinematics (pt, eta, phi)
- **Charge classification**: Determining particle charge

The model is trained end-to-end using multi-task learning with Hungarian matching for optimal track-to-truth assignment.

---

## Data Types and Format

### Raw Data (ROOT Format)

Raw simulation data comes from ATLAS detector simulations in ROOT format containing:

**Hit-level information:**
- Space point coordinates (global edge positions)
- Timing information (drift time, readout side)
- Detector geometry (station indices, layer, technology)
- Covariance matrices for uncertainty estimation
- Truth links connecting hits to particles

**Truth particle information:**
- Muon kinematics: pt (transverse momentum), eta (pseudorapidity), phi (azimuthal angle)
- Charge (q): ±1
- PDG codes and decay information

### Processed Data (HDF5 Format)

After preprocessing, data is stored in HDF5 format for efficient training:

**File structure:**
```
dataset.h5
├── hits/                    # (N_events, max_hits, N_hit_features)
├── particles/               # (N_events, max_tracks, N_track_features)
├── particle_hit_valid/      # (N_events, max_tracks, max_hits) - hit masks
├── particle_valid/          # (N_events, max_tracks) - track validity
├── num_hits/                # (N_events,) - actual hit count per event
└── num_particles/           # (N_events,) - actual track count per event
```

**Feature dimensions:**
- `N_hit_features = 22`: Hit properties including coordinates, detector info, derived features
- `N_track_features = 4`: pt, eta, phi, charge
- `max_hits = 500-600`: Maximum hits per event (padded)
- `max_tracks = 2-6`: Maximum tracks per event (padded)

### Hit Features (22 dimensions)

| Feature | Description | Preprocessing |
|---------|-------------|---------------|
| `spacePoint_globEdgeHighX/Y/Z` | Global coordinates (upper edge) | ×0.001 (mm → m) |
| `spacePoint_globEdgeLowX/Y/Z` | Global coordinates (lower edge) | ×0.001 (mm → m) |
| `spacePoint_time` | Drift time | ×0.00001 |
| `spacePoint_driftR` | Drift radius | ×0.001 |
| `spacePoint_covXX/XY/YX/YY` | Covariance matrix | ×0.000001 |
| `spacePoint_channel` | Detector channel | Raw integer |
| `spacePoint_layer` | Detector layer | Raw integer |
| `spacePoint_stationPhi` | Station phi index | Raw integer |
| `spacePoint_stationEta` | Station eta index | Raw integer |
| `spacePoint_stationIndex` | Unique station ID | Raw integer |
| `spacePoint_technology` | Detector technology (MDT/RPC/TGC) | Integer encoding |
| `r` | Radial distance (derived) | ×0.001 |
| `s` | 3D distance from origin (derived) | ×0.001 |
| `theta` | Polar angle (derived) | Radians |
| `phi` | Azimuthal angle (derived) | Radians |

### Track/Particle Features (4 dimensions)

| Feature | Description | Preprocessing |
|---------|-------------|---------------|
| `truthMuon_pt` | Transverse momentum | ×0.001 (MeV → GeV) |
| `truthMuon_eta` | Pseudorapidity | Raw |
| `truthMuon_phi` | Azimuthal angle | Radians |
| `truthMuon_q` | Charge | ±1 |

---

## Data Preprocessing

### Step 1: ROOT to HDF5 Conversion

**Script:** `prep_events_multiprocess.py`

This script converts raw ROOT files to HDF5 format with filtering and feature engineering.

**Key operations:**
1. **Technology filtering** (optional):
   - Remove specific detector technologies (e.g., NSW, RPC)
   - Controlled via `--no-NSW` and `--no-rpc` flags

2. **Track filtering**:
   - Minimum pt threshold (default: 1.0 GeV)
   - Maximum |eta| (default: 2.5)
   - Minimum hits per track (default: 5)

3. **Event selection**:
   - Exclude events with no valid tracks after filtering
   - Exclude events with no hits after technology filtering

4. **Feature derivation**:
   - Calculate cylindrical coordinates: r = √(x² + y²)
   - Calculate 3D distance: s = √(x² + y² + z²)
   - Calculate polar angle: θ = arctan(r/z)
   - Calculate azimuthal angle: φ = arctan2(y, x)

5. **Data compaction**:
   - Pad all events to uniform size (max_hits, max_tracks)
   - Store efficient indexing for fast random access
   - Generate metadata YAML with feature definitions

**Example usage:**
```bash
python prep_events_multiprocess.py \
    --input_dir /path/to/root/files \
    --output_dir /path/to/output/hdf5 \
    --expected_num_events_per_file 100 \
    --max_events 1000000 \
    --num_workers 16 \
    --pt_threshold 1.0 \
    --eta_threshold 2.5 \
    --num_hits_threshold 5
```

**Output:**
- `dataset.h5`: Main data file with all events
- `metadata.yaml`: Feature definitions and dataset statistics
- `event_file_indices.npy`: Mapping to original ROOT files
- `event_row_indices.npy`: Row indices within files

### Step 2: Optional Hit Filtering

**Script:** `filter_dataset_with_hitfilter.py`

Pre-filter hits using a trained hit filter model to reduce noise:

```bash
python filter_dataset_with_hitfilter.py \
    --input_dir /path/to/original/hdf5 \
    --output_dir /path/to/filtered/hdf5 \
    --checkpoint /path/to/hitfilter/model.ckpt \
    --working_point 0.99  # Hit retention threshold
```

**Benefits:**
- Reduces computational cost (fewer hits per event)
- Improves signal-to-noise ratio
- Typical retention: 70-90% of hits, >99% of signal hits

---

## Data Batching

### Dataset Class: `AtlasMuonDataset`

**File:** `data.py`

**Initialization:**
```python
dataset = AtlasMuonDataset(
    dirpath="/path/to/hdf5/data",
    inputs={
        "hit": [list of hit feature names]
    },
    targets={
        "particle": [list of target feature names]
    },
    num_events=-1,  # -1 for all events
    event_max_num_particles=2,  # Max tracks per event
    hit_eval_path=None,  # Optional pre-filtered hits
)
```

**Data loading workflow:**
1. Load event index (fast O(1) lookup)
2. Read hits and particles from HDF5
3. Apply feature scaling
4. Apply hit filtering (if `hit_eval_path` provided)
5. Return dictionary with inputs, targets, and metadata

**Returned data structure:**
```python
{
    'inputs': {
        'hit': {
            'spacePoint_globEdgeHighX': Tensor[num_hits],
            # ... (22 features total)
        }
    },
    'targets': {
        'particle_valid': Tensor[max_tracks],         # Boolean mask
        'particle_hit_valid': Tensor[max_tracks, max_hits],  # Hit assignment
        'particle': {
            'truthMuon_pt': Tensor[max_tracks],
            # ... (4 features total)
        }
    },
    'metadata': {
        'sample_id': int,  # Unique event ID
    }
}
```

### Collator Class: `AtlasMuonCollator`

**File:** `CollatorATLAS.py`

The collator handles **dynamic batch padding** for variable-length sequences:

**Padding strategy:**
1. **Hit padding**: Pad to maximum hit count in batch
   - Pad value: 0.0 for hit features
   - Allows nested tensor optimization

2. **Track padding**: Already padded to `event_max_num_particles`
   - Pad value: NaN for regression targets
   - Pad value: False for boolean masks

3. **Metadata batching**: Stack sample IDs for tracking

**Example batch shapes:**
```python
batch = {
    'inputs': {
        'hit': {
            'spacePoint_globEdgeHighX': Tensor[batch_size, max_batch_hits],
            # All hit features have same shape
        }
    },
    'targets': {
        'particle_valid': Tensor[batch_size, max_tracks],
        'particle_hit_valid': Tensor[batch_size, max_tracks, max_batch_hits],
        'particle': {
            'truthMuon_pt': Tensor[batch_size, max_tracks],
        }
    },
    'metadata': {
        'sample_id': List[batch_size],
    }
}
```

**Performance considerations:**
- Smaller batches → less padding overhead
- Larger batches → better GPU utilization
- Typical sweet spot: 200-1024 samples
- Use `num_workers=10-12` for I/O parallelism

### DataModule: `AtlasMuonDataModule`

**Lightning integration:**
```python
datamodule = AtlasMuonDataModule(
    train_dir="/path/to/train/hdf5",
    val_dir="/path/to/val/hdf5",
    test_dir="/path/to/test/hdf5",
    batch_size=200,
    num_workers=10,
    num_train=-1,  # Use all training events
    num_val=-1,
    num_test=-1,
    inputs={"hit": [feature_list]},
    targets={"particle": [target_list]},
    event_max_num_particles=2,
)
```

---

## Model Architecture

### MaskFormer Overview

The model uses a **MaskFormer** architecture adapted from object detection:

```
Input Hits → Embedding → Encoder → Decoder → Multi-Task Heads → Predictions
              ↓                                    ↑
         Position Encoding              Query Embeddings
```

### 1. Input Embedding

**Component:** `InputNet` with `Dense` network

**Process:**
1. Concatenate all 22 hit features
2. Pass through dense network: Input[22] → Hidden[44-88] → Output[dim]
3. Add learned position encoding based on (r, θ, φ)

**Configuration:**
```yaml
input_nets:
  - class_path: hepattn.models.InputNet
    init_args:
      input_name: hit
      fields: [22 hit features]
      net:
        class_path: hepattn.models.Dense
        init_args:
          input_size: 22
          output_size: 32  # Model dimension
          hidden_dim_scale: 2  # Hidden = 22 * 2 = 44
          activation: SwiGLU
      posenc:
        class_path: hepattn.models.posenc.PositionEncoder
        init_args:
          dim: 32
          fields: [r, theta, phi]
```

### 2. Encoder

**Component:** `Encoder` with FlashAttention

**Architecture:**
- Multiple transformer layers (typically 2)
- Each layer: Self-Attention → LayerNorm → FFN → LayerNorm
- **FlashAttention** for memory-efficient attention
- **HybridNorm** for improved training stability
- **Value residuals** for better gradient flow

**Attention mechanism:**
- Variable-length sequence support (sorted by phi)
- Multi-head attention (typically 8 heads)
- No positional encoding added in encoder (already in embeddings)

**Configuration:**
```yaml
encoder:
  class_path: hepattn.models.Encoder
  init_args:
    num_layers: 2
    dim: 32
    attn_type: flash-varlen  # Variable-length FlashAttention
    hybrid_norm: true
    value_residual: true
    attn_kwargs:
      num_heads: 8
```

### 3. Decoder

**Component:** Cross-attention decoder with mask attention

**Architecture:**
- Learned query embeddings (one per max track)
- Multiple decoder layers (typically 2)
- Each layer:
  1. Self-attention on queries
  2. Cross-attention to encoder outputs
  3. FFN

**Query masking:**
- Optionally mask unused query slots
- Prevents hallucination of fake tracks

**Configuration:**
```yaml
decoder:
  num_decoder_layers: 2
  num_queries: 2  # Max tracks per event
  mask_attention: true
  use_query_masks: false
  decoder_layer_config:
    dim: 32
    norm: RMSNorm
    attn_kwargs:
      num_heads: 8
```

### 4. Multi-Task Heads

The decoder outputs feed into four parallel task heads:

#### Task 1: Track Validity (`ObjectValidTask`)

**Purpose:** Predict which track slots contain valid particles

**Architecture:**
- Dense network: Query[32] → Hidden[64] → Output[1]
- Sigmoid activation for binary classification

**Loss:** Binary cross-entropy (BCE)
```python
loss = BCE(pred_valid, true_valid) * null_weight
```

**Cost for matching:** BCE cost × 10.0

**Configuration:**
```yaml
- class_path: hepattn.models.task.ObjectValidTask
  init_args:
    name: track_valid
    dim: 32
    losses:
      object_bce: 1.0
    costs:
      object_bce: 10.0
    null_weight: 1.0
```

#### Task 2: Hit Assignment (`ObjectHitMaskTask`)

**Purpose:** Assign hits to tracks (binary mask per track-hit pair)

**Architecture:**
- Hit embeddings: Encoder outputs
- Query embeddings: Decoder outputs
- Interaction: Dot product similarity between hit and query embeddings

**Losses:**
- BCE on hit masks
- Focal loss (emphasizes hard examples)
- Optional: Dice loss for segmentation

**Cost for matching:** BCE cost × 1.0

**Configuration:**
```yaml
- class_path: hepattn.models.task.ObjectHitMaskTask
  init_args:
    name: track_hit_valid
    dim: 32
    losses:
      mask_bce: 1.0
      mask_focal: 1.0
    costs:
      mask_bce: 1.0
```

#### Task 3: Parameter Regression (`ObjectRegressionTask`)

**Purpose:** Predict track kinematics (eta, phi, pt)

**Architecture:**
- Dense network: Query[32] → Hidden[64] → Output[3]
- No activation (linear regression)

**Loss:** Smooth L1 (Huber loss)
```python
loss = SmoothL1(pred_params, true_params)
```

**Cost for matching:** Smooth L1 cost × 1.0

**Configuration:**
```yaml
- class_path: hepattn.models.task.ObjectRegressionTask
  init_args:
    name: parameter_regression
    dim: 32
    fields: [truthMuon_eta, truthMuon_phi, truthMuon_pt]
    loss_weight: 1.0
    cost_weight: 1.0
```

#### Task 4: Charge Classification (`ObjectChargeClassificationTask`)

**Purpose:** Predict particle charge (±1)

**Architecture:**
- Dense network: Query[32] → Hidden[64] → Output[2]
- Softmax activation for classification

**Loss:** Cross-entropy
```python
loss = CrossEntropy(pred_charge, true_charge)
```

**Cost for matching:** Cross-entropy cost × 1.0

**Configuration:**
```yaml
- class_path: hepattn.models.task.ObjectChargeClassificationTask
  init_args:
    name: charge_classification
    dim: 32
    field: truthMuon_q
    loss_weight: 1.0
    cost_weight: 1.0
```

### 5. Hungarian Matching

**Component:** `Matcher` for optimal assignment

Before computing losses, predicted tracks are matched to ground truth tracks using the **Hungarian algorithm** (optimal bipartite matching).

**Cost matrix construction:**
```python
cost_matrix[i, j] = Σ (task_cost_weight * task_cost(pred_i, true_j))
```

Where costs come from each task:
- Track validity BCE cost
- Hit assignment BCE cost
- Parameter regression L1 cost
- Charge classification CE cost

**Solver options:**
- **scipy**: Hungarian algorithm (O(n³))
- **lapjv**: Jonker-Volgenant (faster for large problems)
- **parallel**: Multi-threaded for batch processing

**Configuration:**
```yaml
matcher:
  class_path: hepattn.models.matcher.Matcher
  init_args:
    default_solver: scipy
    adaptive_solver: false
    parallel_solver: true
    n_jobs: 16
```

---

## Training Process

### Training Configuration

**File:** `configs/NGT/smallCuts/atlas_muon_tracking_NGT_small2track_regression.yaml`

### Optimizer Configuration

**Optimizer:** Lion (improved AdamW variant)
- Better convergence on transformer architectures
- Lower memory footprint than AdamW

**Alternative:** AdamW (standard choice)

**Learning rate schedule:**
- Type: Cosine annealing with warmup
- Initial LR: 1e-5
- Max LR: 5e-5 (reached at 5% of training)
- Final LR: 1e-5 (end of training)
- Weight decay: 1e-5

**Configuration:**
```yaml
model:
  optimizer: Lion
  lrs_config:
    initial: 1e-5
    max: 5e-5
    end: 1e-5
    pct_start: 0.05  # 5% warmup
    skip_scheduler: false
    weight_decay: 1e-5
```

### Training Hyperparameters

**Epochs and batching:**
```yaml
trainer:
  max_epochs: 50
  gradient_clip_val: 0.1
  precision: bf16-mixed  # BFloat16 for memory efficiency
```

**Data configuration:**
```yaml
data:
  batch_size: 200
  num_workers: 10
  num_train: -1  # Use all training data
  num_val: -1
  num_test: -1
```

**GPU configuration:**
```yaml
trainer:
  accelerator: gpu
  devices: [1]  # Single GPU
  # For multi-GPU:
  # devices: [0, 1, 2]
  # strategy:
  #   class_path: lightning.pytorch.strategies.DDPStrategy
  #   init_args:
  #     find_unused_parameters: true
```

### Callbacks

**Essential callbacks:**

1. **Checkpointing:**
```yaml
- class_path: hepattn.callbacks.Checkpoint
  init_args:
    monitor: val/loss
    mode: min
    save_top_k: 1  # Save only best model
```

2. **Early stopping:**
```yaml
- class_path: lightning.pytorch.callbacks.EarlyStopping
  init_args:
    monitor: val/loss
    patience: 5  # Stop after 5 epochs without improvement
    min_delta: 0.0
    mode: min
```

3. **Learning rate monitoring:**
```yaml
- class_path: lightning.pytorch.callbacks.LearningRateMonitor
```

4. **Prediction saving:**
```yaml
- class_path: hepattn.callbacks.PredictionWriter
  init_args:
    write_inputs: false
    write_outputs: true
    write_preds: true
    write_targets: false
```

### Logging

**Comet ML integration:**
```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.CometLogger
    init_args:
      project_name: atlas_muon_tracking
      experiment_name: TRK-ATLAS-Muon-smallModel-better-run
      save_dir: logs
```

**Logged metrics:**

**Per-task metrics:**
- `train/loss`, `val/loss`: Total loss
- `train/{task_name}_loss`, `val/{task_name}_loss`: Per-task losses
- Example: `train/track_valid_loss`, `val/parameter_regression_loss`

**Track-level metrics:**
- `val/p{wp}_eff`: Track efficiency at matching working point (wp ∈ {0.25, 0.5, 0.75, 1.0})
- `val/p{wp}_pur`: Track purity at matching working point
- `val/track_efficiency`: Overall track reconstruction efficiency
- `val/track_fake_rate`: Fake track rate

**Matching working points:**
- A track is "matched" if at least `wp` fraction of hits are correctly assigned
- Example: `val/p0.75_eff` = efficiency requiring 75% hit purity

### Training Execution

**Command:**
```bash
python run_tracking.py fit \
    --config configs/NGT/smallCuts/atlas_muon_tracking_NGT_small2track_regression.yaml
```

**Expected behavior:**
1. Load training and validation datasets
2. Initialize model with random weights (or from checkpoint)
3. Run sanity validation check (1 batch)
4. Training loop:
   - Forward pass through encoder-decoder
   - Compute costs for Hungarian matching
   - Match predictions to ground truth
   - Compute losses for each task
   - Backward pass and optimization step
   - Log metrics every 50 steps
5. Validation loop (after each epoch):
   - Compute validation metrics
   - Save checkpoint if best val/loss
   - Check early stopping criterion
6. Save final model checkpoint

**Typical training time:**
- Single epoch (2.69M events, batch_size=200): ~30-60 minutes on A100
- Full training (50 epochs): ~25-50 hours
- With early stopping (typical ~20-30 epochs): ~15-30 hours

---

## Hyperparameter Optimization

### Optuna Framework

**Script:** `optuna_tune.py`

Optuna enables systematic hyperparameter search using Bayesian optimization (TPE sampler).

### Tunable Hyperparameters

**Architecture:**
```python
num_encoder_layers: [1, 2, 3, 4]
num_decoder_layers: [1, 2, 3, 4]
dim: [16, 32, 64]
num_heads: [4, 8, 16]
```

**Task configurations:**
```python
# For each task:
cost_weight: [0.1, 1.0, 10.0]
loss_weight: [0.1, 1.0, 10.0]

# Dense networks:
hidden_dim_scale: [1, 2, 4, 8]
# Resulting hidden sizes: [32, 64, 128, 256] for dim=32
```

**Optimization:**
```python
initial_lr: [1e-6, 1e-4] (log scale)
max_lr: [1e-5, 1e-3] (log scale)
weight_decay: [1e-6, 1e-3] (log scale)
```

### Multi-GPU Optimization

**Setup for 4 GPUs:**

**Terminal 1 (GPU 0):**
```bash
CUDA_VISIBLE_DEVICES=0 python optuna_tune.py \
    --study_name atlas_muon_optuna \
    --storage sqlite:///optuna_study.db \
    --n_trials 25
```

**Terminal 2 (GPU 1):**
```bash
CUDA_VISIBLE_DEVICES=1 python optuna_tune.py \
    --study_name atlas_muon_optuna \
    --storage sqlite:///optuna_study.db \
    --n_trials 25
```

**Terminal 3 (GPU 2):**
```bash
CUDA_VISIBLE_DEVICES=2 python optuna_tune.py \
    --study_name atlas_muon_optuna \
    --storage sqlite:///optuna_study.db \
    --n_trials 25
```

**Terminal 4 (GPU 3):**
```bash
CUDA_VISIBLE_DEVICES=3 python optuna_tune.py \
    --study_name atlas_muon_optuna \
    --storage sqlite:///optuna_study.db \
    --n_trials 25
```

**Study coordination:**
- All workers share the same SQLite database
- Optuna handles trial distribution and locking
- Each worker runs 25 trials → 100 trials total
- TPE sampler uses all completed trials to suggest next

**Objective function:**
```python
def objective(trial):
    # Sample hyperparameters
    params = {
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 4),
        'dim': trial.suggest_categorical('dim', [16, 32, 64]),
        # ... more parameters
    }
    
    # Train model with sampled config
    trainer.fit(model, datamodule)
    
    # Return objective (minimize validation loss)
    return trainer.callback_metrics['val/loss'].item()
```

### Analyzing Results

**View best trials:**
```python
import optuna
study = optuna.load_study(
    study_name='atlas_muon_optuna',
    storage='sqlite:///optuna_study.db'
)

print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

# Get top 10 trials
top_trials = sorted(study.trials, key=lambda t: t.value)[:10]
for i, trial in enumerate(top_trials):
    print(f"{i+1}. Trial {trial.number}: {trial.value:.4f}")
```

**Visualization:**
```python
import optuna.visualization as vis

# Parameter importance
vis.plot_param_importances(study).write_html('importance.html')

# Optimization history
vis.plot_optimization_history(study).write_html('history.html')

# Hyperparameter relationships
vis.plot_parallel_coordinate(study).write_html('parallel.html')
```

---

## Evaluation

### Evaluation Metrics

**Task 1: Track Validity**
- **Track efficiency**: TP / (TP + FN)
- **Fake rate**: FP / (FP + TN)

Where:
- TP: Predicted valid AND actually valid
- FP: Predicted valid BUT actually invalid
- FN: Predicted invalid BUT actually valid
- TN: Predicted invalid AND actually invalid

**Task 2: Hit Assignment**
- **Efficiency at WP**: Fraction of tracks with ≥WP hit purity
- **Purity at WP**: Fraction of predicted tracks with ≥WP hit purity

**Task 3: Parameter Regression**
- **MAE (Mean Absolute Error)** for each parameter
- **Relative error** for pt: |pred_pt - true_pt| / true_pt

**Task 4: Charge Classification**
- **Accuracy**: Correct charge predictions
- **Per-class precision/recall**

### Evaluation Scripts

**Script 1: Tracking evaluation**
```bash
python evaluate_tracking_model.py \
    --checkpoint /path/to/best/model.ckpt \
    --test_dir /path/to/test/hdf5 \
    --output_dir evaluation_results
```

**Outputs:**
- Predicted tracks saved to HDF5
- Metrics JSON file
- Optional: Event displays

**Script 2: Task-specific evaluation**

Each task has a dedicated evaluation script:

```bash
# Task 1: Track validity
python evaluate_task2_track_validity.py \
    --predictions /path/to/preds.h5 \
    --ground_truth /path/to/test/data

# Task 2: Hit assignment (covered in tracking evaluation)

# Task 3: Regression
python evaluate_task3_regression.py \
    --predictions /path/to/preds.h5 \
    --ground_truth /path/to/test/data
```

### Plotting Results

**Official ATLAS-style plots:**

```bash
# Tracking performance plots
python atlas_style_tracking_plots_official.py \
    --predictions evaluation_results/ \
    --output_dir tracking_plots/

# Hit filtering plots (if applicable)
python atlas_style_filtering_plots_official.py \
    --predictions hit_filter_results/ \
    --output_dir filtering_plots/
```

**Generated plots:**
- Track efficiency vs pt, eta
- Fake rate vs selection threshold
- Hit purity distributions
- Parameter resolution (Δpt, Δeta, Δphi)
- Confusion matrices for charge classification

---

## Step-by-Step Workflow

### Complete Training Pipeline

#### Step 1: Data Preparation

**1a. Convert ROOT to HDF5:**
```bash
python prep_events_multiprocess.py \
    --input_dir /data/simulation/root_files \
    --output_dir /data/processed/training_hdf5 \
    --expected_num_events_per_file 100 \
    --max_events 2694000 \
    --num_workers 16 \
    --pt_threshold 1.0 \
    --eta_threshold 2.5 \
    --num_hits_threshold 5
```

**1b. (Optional) Pre-filter hits:**

First, train a hit filter model (separate task), then:
```bash
python filter_dataset_with_hitfilter.py \
    --input_dir /data/processed/training_hdf5 \
    --output_dir /data/processed/training_hdf5_filtered \
    --checkpoint /models/hit_filter/best.ckpt \
    --working_point 0.99
```

**1c. Verify data:**
```bash
python -c "
import h5py
import yaml

# Check HDF5 structure
with h5py.File('/data/processed/training_hdf5/dataset.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Hits shape:', f['hits'].shape)
    print('Particles shape:', f['particles'].shape)

# Check metadata
with open('/data/processed/training_hdf5/metadata.yaml') as f:
    metadata = yaml.safe_load(f)
    print('Hit features:', metadata['hit_features'])
    print('Track features:', metadata['track_features'])
"
```

#### Step 2: Configure Training

**2a. Copy base config:**
```bash
cp configs/NGT/smallCuts/atlas_muon_tracking_NGT_small2track_regression.yaml \
   configs/my_experiment.yaml
```

**2b. Edit configuration:**

Update paths and hyperparameters in `configs/my_experiment.yaml`:
```yaml
name: my_tracking_experiment

data:
  train_dir: /data/processed/training_hdf5_filtered
  val_dir: /data/processed/validation_hdf5_filtered
  test_dir: /data/processed/test_hdf5
  batch_size: 200  # Adjust based on GPU memory
  
trainer:
  max_epochs: 50
  devices: [0]  # Your GPU ID
  
  logger:
    init_args:
      experiment_name: my_tracking_experiment
```

#### Step 3: Train Model

**3a. Start training:**
```bash
python run_tracking.py fit --config configs/my_experiment.yaml
```

**3b. Monitor training:**

**Option 1: Comet ML dashboard**
- Navigate to https://www.comet.ml/
- Find your experiment: `atlas_muon_tracking/my_tracking_experiment`
- Monitor real-time metrics, system usage, and hyperparameters

**Option 2: TensorBoard (if configured)**
```bash
tensorboard --logdir logs/
```

**Option 3: Terminal output**
- Watch for `val/loss` decreasing
- Check `val/p0.75_eff` and `val/p0.75_pur` for track performance

**3c. Training checkpoints:**

Checkpoints are saved automatically:
```
logs/
└── my_tracking_experiment_YYYYMMDD-THHMMSS/
    └── ckpts/
        ├── epoch=000-val_loss=X.XXXXX.ckpt
        ├── epoch=005-val_loss=X.XXXXX.ckpt
        └── epoch=029-val_loss=X.XXXXX.ckpt  # Best model
```

#### Step 4: Evaluate Model

**4a. Run evaluation:**
```bash
python evaluate_tracking_model.py \
    --checkpoint logs/my_tracking_experiment_*/ckpts/epoch=*-val_loss=*.ckpt \
    --test_dir /data/processed/test_hdf5 \
    --output_dir evaluation_results/my_experiment
```

**4b. Generate plots:**
```bash
python atlas_style_tracking_plots_official.py \
    --predictions evaluation_results/my_experiment \
    --output_dir tracking_plots/my_experiment
```

**4c. Analyze results:**

Check generated files:
- `evaluation_results/my_experiment/metrics.json`: Numerical metrics
- `tracking_plots/my_experiment/efficiency_vs_pt.pdf`: Performance plots
- `tracking_plots/my_experiment/resolution_eta.pdf`: Resolution plots

#### Step 5: (Optional) Hyperparameter Optimization

**5a. Prepare Optuna study:**
```bash
# Create base config for Optuna
cp configs/NGT/smallCuts/optuna_base_config.yaml \
   configs/my_optuna_experiment.yaml
```

**5b. Launch optimization (4 GPUs):**
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python optuna_tune.py \
    --study_name my_optuna_study \
    --storage sqlite:///my_optuna.db \
    --n_trials 25 &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python optuna_tune.py \
    --study_name my_optuna_study \
    --storage sqlite:///my_optuna.db \
    --n_trials 25 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python optuna_tune.py \
    --study_name my_optuna_study \
    --storage sqlite:///my_optuna.db \
    --n_trials 25 &

# Terminal 4
CUDA_VISIBLE_DEVICES=3 python optuna_tune.py \
    --study_name my_optuna_study \
    --storage sqlite:///my_optuna.db \
    --n_trials 25
```

**5c. Analyze Optuna results:**
```python
import optuna

study = optuna.load_study(
    study_name='my_optuna_study',
    storage='sqlite:///my_optuna.db'
)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

# Export best config
import yaml
with open('configs/best_params.yaml', 'w') as f:
    yaml.dump(study.best_trial.params, f)
```

**5d. Train with best parameters:**

Merge best parameters into config and retrain:
```bash
python run_tracking.py fit --config configs/my_best_config.yaml
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
```yaml
data:
  batch_size: 100  # Down from 200
```

2. Reduce model size:
```yaml
model:
  model:
    init_args:
      dim: 16  # Down from 32
      encoder:
        init_args:
          num_layers: 1  # Down from 2
```

3. Enable gradient checkpointing (if available)
4. Use smaller precision (already using `bf16-mixed`)

#### Issue 2: Slow Data Loading

**Symptoms:**
- Low GPU utilization
- Long epoch times

**Solutions:**
1. Increase data workers:
```yaml
data:
  num_workers: 16  # Up from 10
```

2. Pre-filter hits to reduce data size
3. Use SSD for data storage instead of network drives

#### Issue 3: Poor Convergence

**Symptoms:**
- Validation loss not decreasing
- Low track efficiency

**Solutions:**
1. Increase learning rate:
```yaml
model:
  lrs_config:
    max: 1e-4  # Up from 5e-5
```

2. Reduce weight decay:
```yaml
model:
  lrs_config:
    weight_decay: 1e-6  # Down from 1e-5
```

3. Increase model capacity:
```yaml
encoder:
  num_layers: 3  # Up from 2
decoder:
  num_decoder_layers: 3  # Up from 2
```

4. Check task weights (ensure no single task dominates)

#### Issue 4: NaN Losses

**Symptoms:**
```
train/loss = nan
```

**Solutions:**
1. Enable gradient clipping (should already be enabled):
```yaml
trainer:
  gradient_clip_val: 0.1
```

2. Reduce learning rate:
```yaml
model:
  lrs_config:
    max: 1e-5  # Down from 5e-5
```

3. Check for numerical instability in custom losses
4. Verify data has no NaN values

#### Issue 5: Multi-GPU Training Issues

**Symptoms:**
- Training slower on multiple GPUs than single GPU
- Inconsistent results across GPUs

**Solutions:**
1. Enable DDP strategy:
```yaml
trainer:
  devices: [0, 1, 2]
  strategy:
    class_path: lightning.pytorch.strategies.DDPStrategy
    init_args:
      find_unused_parameters: true
```

2. Ensure batch size is divisible by number of GPUs
3. Use `sync_dist=True` in logging (already enabled)

#### Issue 6: Optuna Database Locking

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
1. Use PostgreSQL instead of SQLite for large studies:
```bash
python optuna_tune.py \
    --study_name my_study \
    --storage postgresql://user:pass@localhost/optuna
```

2. Reduce concurrent workers
3. Add retry logic in Optuna script

---

## Additional Resources

### Documentation Files

- **README_OPTUNA.md**: Detailed Optuna optimization guide
- **IMPLEMENTATION_SUMMARY.md**: Architecture implementation details
- **PLOT_MODIFICATIONS_SUMMARY.md**: Plotting customization guide

### Example Configs

- `configs/NGT/smallCuts/atlas_muon_tracking_NGT_small2track_regression.yaml`: Full tracking config
- `configs/NGT/smallCuts/optuna_base_config.yaml`: Optuna optimization template
- `configs/NGT/smallCuts/atlas_muon_filtering_NGT_small.yaml`: Hit filtering config

### Key Scripts

- `run_tracking.py`: Main training entry point
- `run_filtering.py`: Hit filter training
- `data.py`: Dataset and DataModule classes
- `CollatorATLAS.py`: Batch collation logic
- `prep_events_multiprocess.py`: Data preprocessing
- `optuna_tune.py`: Hyperparameter optimization
- `evaluate_tracking_model.py`: Model evaluation

### Contact

For questions or issues, please consult the main repository README or contact the maintainers.

---

**Note:** This guide assumes familiarity with PyTorch and PyTorch Lightning. For general deep learning concepts, refer to the [PyTorch tutorials](https://pytorch.org/tutorials/) and [Lightning documentation](https://lightning.ai/docs/pytorch/stable/).
