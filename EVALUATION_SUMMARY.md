# ATLAS Muon Hit Filter Evaluation Results

## Summary

This evaluation script analyzes the performance of a binary classifier transformer architecture for ATLAS muon hit filtering. The script has been successfully updated to use ROC-based working points as requested.

## Key Features Implemented

### 1. ROC Curve Analysis
- Generates ROC curve with AUC score
- AUC achieved: **0.9982** (excellent performance)

### 2. Working Point Methodology
The script now uses the correct working point logic:
```python
fpr, tpr, thresholds = roc_curve(true_labels, predicted_logits)
# For target efficiency of 0.995:
cut = thresholds[tpr > 0.995][0]
```

### 3. Efficiency and Purity Analysis
- **20 pt bins** from 5.5 to 89.7 GeV (based on actual data range)
- **Working points**: [0.96, 0.97, 0.98, 0.99, 0.995] efficiency targets
- **Error bars**: Binomial statistics for uncertainty estimation
- **Track loss calculation**: Percentage of tracks with < 3 hits retained

### 4. Output Structure
```
evaluation_results/
├── roc_curve.png                    # ROC curve with AUC
├── efficiency_plots/                # Efficiency vs pt for each working point
│   ├── efficiency_target_0.96.png
│   ├── efficiency_target_0.97.png
│   ├── efficiency_target_0.98.png
│   ├── efficiency_target_0.99.png
│   └── efficiency_target_0.995.png
├── purity_plots/                    # Purity vs pt for each working point
│   ├── purity_target_0.96.png
│   ├── purity_target_0.97.png
│   ├── purity_target_0.98.png
│   ├── purity_target_0.99.png
│   └── purity_target_0.995.png
└── evaluation_report.txt            # Comprehensive summary

```

## Results Summary (from 100 events test)

### Overall Performance
- **Total hits analyzed**: 725,835
- **True muon hits**: 5,414 (0.7%)
- **Noise hits**: 720,421 (99.3%)
- **ROC AUC**: 0.9982

### Working Point Performance
| Target Efficiency | Achieved Efficiency | Purity | Threshold |
|-------------------|-------------------|---------|-----------|
| 0.96              | 0.960            | 0.774   | -0.9180   |
| 0.97              | 0.970            | 0.707   | -1.9375   |
| 0.98              | 0.980            | 0.567   | -3.6250   |
| 0.99              | 0.990            | 0.316   | -6.8125   |
| 0.995             | 0.995            | 0.116   | -9.6875   |

## Usage

### Basic Usage
```bash
cd /shared/tracking/hepattn_muon
pixi run python src/hepattn/experiments/atlas_muon/evaluate_hit_filter.py \
  --max-events 1000 \
  --output-dir ./evaluation_results
```

### Full Dataset
```bash
# Process all events (86,526 events)
pixi run python src/hepattn/experiments/atlas_muon/evaluate_hit_filter.py \
  --output-dir ./evaluation_results_full
```

### Custom Parameters
```bash
pixi run python src/hepattn/experiments/atlas_muon/evaluate_hit_filter.py \
  --eval-path /path/to/eval.h5 \
  --data-path /path/to/raw/data \
  --output-dir /path/to/output \
  --max-events 5000
```

## Fixed Issues

1. **Working Point Logic**: Now correctly uses ROC curve thresholds instead of fixed probability thresholds
2. **Variable Naming**: Fixed `predicted_true_all` → `predicted_in_bin` error
3. **PT Range**: Automatically detects actual pt range (5.5-89.7 GeV) from data
4. **Efficiency Calculation**: Proper handling of particle-to-hit mapping using `spacePoint_truthLink`
5. **Track Loss**: Correctly counts tracks with insufficient hits after filtering

## Next Steps

The script is now ready for production use. For final publication-quality results, run with the full dataset:

```bash
pixi run python src/hepattn/experiments/atlas_muon/evaluate_hit_filter.py \
  --output-dir ./final_evaluation_results
```

This will process all 86,526 events and provide comprehensive statistics for your analysis.
