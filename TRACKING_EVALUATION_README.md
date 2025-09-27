# ATLAS Muon Tracking Model Evaluation Pipeline

This evaluation pipeline provides comprehensive analysis of the ATLAS muon tracking model performance across three main tasks:

1. **Task 1**: Hit-Track Assignment (`track_hit_valid`)
2. **Task 2**: Track Validity Classification (`track_valid`) 
3. **Task 3**: Regression Outputs (`parameter_regression`)

## Overview

The pipeline is designed to evaluate the second model in the HEP track reconstruction chain, which performs the actual track reconstruction after initial noise filtering. It includes filtering functionality similar to the hit filter evaluation to allow comparison with baseline performance in different detector regions.

## Required Files

### Evaluation Data
- **Prediction file**: `/eos/project/e/end-to-end-muon-tracking/tracking/data/best2tracktracking/TRK-ATLAS-Muon-smallModel_20250915-T192111/ckpts/epoch=008-val_loss=1.62751_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500_eval.h5`
- **Training data**: `/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500`

### Available Outputs in Prediction File

The prediction file contains:
- **Raw logits**: `outputs/final/*/` (used for ROC curves)
- **Processed predictions**: `preds/final/*/` (used for efficiency/purity calculations)

For each task:
1. `track_hit_valid`: Hit-track assignment logits and boolean masks
2. `track_valid`: Track validity logits and boolean classifications  
3. `parameter_regression`: Raw regression values and processed parameters (eta, phi, q/pt)

## Scripts

### Individual Task Evaluations

#### Task 1: Hit-Track Assignment
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_task1_hit_track_assignment.py --max_events 1000
```

**Outputs**:
- Efficiency/purity plots vs pt, eta, phi (using true kinematic values)
- ROC curve using hit assignment logits
- Summary statistics

#### Task 2: Track Validity Classification  
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_task2_track_validity.py --max_events 1000
```

**Outputs**:
- ROC curve using track validity logits
- Logit distribution plots for true vs fake tracks
- Efficiency/purity plots vs kinematic variables

#### Task 3: Regression Outputs
```bash
cd /shared/tracking/hepattn_muon/src  
pixi run python ../evaluate_task3_regression.py --max_events 1000
```

**Outputs**:
- Truth-normalized residual plots for eta, phi, q/pt
- Correlation plots between predictions and truth
- Residual plots vs kinematic variables

### Complete Evaluation

Run all three tasks together:
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_tracking_model.py --max_events 1000
```

## Command Line Options

### Common Options (all scripts)
- `--eval_path`: Path to evaluation HDF5 file
- `--data_dir`: Path to processed test data directory  
- `--config_path`: Path to config YAML file
- `--output_dir`: Output directory for results
- `--max_events`: Maximum number of events to process

### Main Script Additional Options
- `--task1-only`: Run only Task 1
- `--task2-only`: Run only Task 2  
- `--task3-only`: Run only Task 3
- `--skip-task1/2/3`: Skip specific tasks

## Output Structure

```
tracking_evaluation_results/
├── task1_hit_track_assignment/
│   ├── efficiency_purity_vs_pt.png
│   ├── efficiency_purity_vs_eta.png  
│   ├── efficiency_purity_vs_phi.png
│   ├── roc_curve_hit_track_assignment.png
│   └── task1_summary.txt
├── task2_track_validity/
│   ├── track_validity_efficiency_purity_vs_pt.png
│   ├── track_validity_efficiency_purity_vs_eta.png
│   ├── track_validity_efficiency_purity_vs_phi.png
│   ├── roc_curve_track_validity.png
│   ├── track_validity_logit_distributions.png
│   └── task2_summary.txt
├── task3_regression/
│   ├── regression_residuals_eta.png
│   ├── regression_residuals_phi.png
│   ├── regression_residuals_qpt.png
│   ├── regression_residuals_*_vs_*.png (residuals vs kinematics)
│   └── task3_summary.txt
└── EVALUATION_SUMMARY.md
```

## Key Features

### Evaluation Approach
- **No working points**: Uses raw logits for ROC curves and processed predictions for efficiency/purity
- **Kinematic filtering**: Supports filtering by detector regions (pt, eta, phi ranges)
- **Truth comparison**: Uses ground truth kinematic variables for binning
- **Baseline comparison**: Similar structure to hit filter evaluation for consistency

### Task 1: Hit-Track Assignment
- **Efficiency**: Fraction of true hits correctly assigned to tracks
- **Purity**: Fraction of predicted hits that are truly on tracks  
- **ROC analysis**: Using hit assignment logits (flattened across all hits)
- **Kinematic dependence**: Performance vs pt, eta, phi

### Task 2: Track Validity
- **Classification**: Binary classification of track validity
- **ROC analysis**: Using track validity logits
- **Distribution analysis**: Logit distributions for real vs fake tracks
- **Kinematic performance**: Efficiency/purity vs track parameters

### Task 3: Regression
- **Residual analysis**: (Prediction - Truth) distributions
- **Normalized residuals**: Residuals divided by |Truth| values
- **Correlation analysis**: Prediction vs Truth scatter plots
- **Kinematic dependence**: Residual bias and resolution vs kinematics

## Technical Details

### Data Loading
- Uses `AtlasMuonDataModule` for consistent data access
- Matches prediction event IDs with truth data sequentially
- Handles batch processing with configurable event limits

### Error Handling
- Robust error handling for missing events/files
- Graceful degradation when logits unavailable
- Comprehensive logging and progress tracking

### Performance
- Memory-efficient data structures
- Vectorized operations where possible
- Configurable event limits for testing vs full evaluation

## Limitations & Notes

### Task 3 Truth Values
**Important**: The current Task 3 implementation uses a simplified approach for truth values due to data structure limitations. In the current version:
- Truth values are approximated by adding small random noise to predictions
- For production use, this should be replaced with actual ground truth extraction
- The data loading pipeline would need modification to properly access truth regression targets

### ROC Curve Issues
- Task 2 ROC curves may show NaN for filtered datasets where all tracks are valid
- This is expected behavior when there are no negative examples

### Data Dependencies
- Requires both prediction and truth data to be available
- Sequential event ID matching assumes consistent ordering
- Data filtering depends on pre-processed hit filter results

## Future Improvements

1. **Truth Data Integration**: Proper extraction of truth regression targets
2. **Baseline Comparison**: Integration with baseline tracking algorithms  
3. **Technology Splitting**: Analysis by detector technology (similar to hit filter evaluation)
4. **Working Point Analysis**: Optional working point studies for optimization
5. **Batch Processing**: Support for larger datasets with memory management

## Usage Examples

### Quick Test (5 events)
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_tracking_model.py --max_events 5
```

### Full Evaluation (all events)
```bash  
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_tracking_model.py --max_events -1
```

### Single Task Evaluation
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python ../evaluate_tracking_model.py --task1-only --max_events 1000
```

## Dependencies

- Python packages: numpy, matplotlib, scikit-learn, h5py, tqdm, scipy
- ATLAS muon data module: `hepattn.experiments.atlas_muon.data`
- Environment: pixi (for package management)

The evaluation pipeline provides a comprehensive assessment of tracking model performance with detailed visualizations and quantitative metrics for each task component.