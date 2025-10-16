
## Overview

Successfully implemented an independent script to generate ATLAS-style publication plots for the muon hit filtering model evaluation.

## Generated Plots

The script now generates **THREE** ATLAS-style plots:

### 1. ROC Curve (atlas_roc_curve.png)
- Shows True Positive Rate (Efficiency) vs. False Positive Rate
- Includes AUC (Area Under Curve) metric
- Compares to random classifier baseline
- **ATLAS Style**: Internal label, Simulation subtext, proper formatting

### 2. Rejection Rate vs. Purity (atlas_rejection_vs_purity.png)
- X-axis: Hit Purity (Precision) = TP / (TP + FP)
- Y-axis: Background Rejection Rate = TN / (TN + FP)
- Color-coded by efficiency working point (0.95 to 0.998)
- **Red cross marker** at the 0.99 efficiency working point (used in analysis)
- Shows how well the model rejects background while maintaining purity
- **ATLAS Style**: Internal label, proper grid and formatting

### 3. Hit Efficiency vs. Hit Purity (atlas_efficiency_vs_purity.png) ⭐ NEW
- X-axis: Hit Efficiency (Working Point) = TP / (TP + FN)  
- Y-axis: Hit Purity (Precision) = TP / (TP + FP)
- Simple line plot (no color coding - working point is already on x-axis)
- Shows the fundamental trade-off between keeping signal hits and maintaining purity
- **ATLAS Style**: Internal label, proper grid and formatting

## Working Point Sweep

All plots use a fine-grained sweep of efficiency working points:
- **Range**: 0.95 to 0.998
- **Step size**: 0.0001
- **Total points**: 481 working points

This provides high-resolution curves suitable for publication.

## Output Structure

```
CTD_plots/
└── run_YYYYMMDD_HHMMSS/
    ├── atlas_roc_curve.png                (ROC curve)
    ├── atlas_rejection_vs_purity.png      (Rejection vs Purity)
    ├── atlas_efficiency_vs_purity.png     (Efficiency vs Purity) ⭐ NEW
    ├── rejection_purity_data.csv          (Rejection data)
    └── efficiency_purity_data.csv         (Efficiency data) ⭐ NEW
```

## Test Results (100 events)

```
✓ Events processed: 100
✓ Total hits: 725,835
  - True hits: 5,414
  - Noise hits: 720,421
✓ AUC Score: 0.99833

✓ Rejection vs Purity:
  - Valid points: 481
  - Purity range: [0.0330, 0.8051]
  - Rejection range: [0.7803, 0.9983]
  - 0.99 working point: Purity=0.4606, Rejection=0.9913 (marked with red cross)

✓ Efficiency vs Purity:
  - Valid points: 481
  - Purity range: [0.0330, 0.8051]
  - Efficiency range: [0.9503, 0.9982]
  - Line plot showing efficiency (x-axis) vs purity (y-axis)
```

## Physics Interpretation

### Hit Efficiency vs. Hit Purity Plot (New)

This plot is particularly important for ATLAS publications because it shows:

1. **Precision-Recall Trade-off**: The direct relationship between efficiency (x-axis) and purity (y-axis)
2. **Model Performance**: The curve shows how purity changes as you vary the efficiency threshold
3. **Operational Choice**: The plot helps choose the optimal working point based on whether you prioritize:
   - High efficiency (don't lose signal hits) 
   - High purity (minimize contamination)
4. **Simple visualization**: Since efficiency is the working point, it's plotted directly on x-axis without redundant color coding

### Rejection vs Purity Plot

This plot shows:

1. **Background Suppression**: How well the model rejects background noise
2. **0.99 Working Point Marker**: A red cross highlights the specific working point (0.99 efficiency) used in the analysis
3. **Color-coded scatter**: Shows the full range of working points with color indicating efficiency

### Relationship Between Plots

- **ROC Curve**: Shows overall classifier performance (TPR vs FPR)
- **Rejection vs Purity**: Shows background suppression capability vs signal purity
- **Efficiency vs Purity**: Shows signal retention vs signal purity (precision-recall)

Together, these three plots provide a complete picture of the hit filtering model's performance.

## Usage for Publication

All plots are:
- ✅ 300 DPI resolution (publication quality)
- ✅ ATLAS Internal label (for internal notes/CTD)
- ✅ Proper formatting and styling
- ✅ Ready for inclusion in ATLAS documents

To use for external publication, you may need to change "Internal" to the appropriate label based on ATLAS approval status.

## Running the Script

### Quick test (100 events):
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python -m hepattn.experiments.atlas_muon.atlas_style_plots -m 100
```

### Full dataset:
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python -m hepattn.experiments.atlas_muon.atlas_style_plots
```

### With custom paths:
```bash
pixi run python -m hepattn.experiments.atlas_muon.atlas_style_plots \
    --eval_path /path/to/eval.h5 \
    --data_dir /path/to/data \
    --max_events -1
```

## Next Steps

1. ✅ Script working correctly with proper data loading
2. ✅ All three ATLAS-style plots generated
3. ✅ CSV data files saved for further analysis
4. 📋 Ready for full-scale analysis on complete dataset
5. 📋 Review plots and adjust style parameters if needed
6. 📋 Submit to ATLAS for CTD approval

## Date
October 13, 2025
