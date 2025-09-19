# ATLAS Muon Tracking Model Evaluation Summary

**Generated at:** 2025-09-19 15:04:25

## Overview

This evaluation covers three main tasks of the ATLAS muon tracking model:

1. **Task 1**: Hit-Track Assignment (track_hit_valid)
2. **Task 2**: Track Validity Classification (track_valid)
3. **Task 3**: Regression Outputs (parameter_regression)

## Task Results

### Task 1: Hit-Track Assignment: ✅ COMPLETED


### Task 2: Track Validity Classification: ✅ COMPLETED


### Task 3: Regression Outputs: ✅ COMPLETED


## Directory Structure

```
tracking_evaluation_results/
├── task1_hit_track_assignment/
│   ├── efficiency_purity_vs_*.png
│   ├── roc_curve_hit_track_assignment.png
│   └── task1_summary.txt
├── task2_track_validity/
│   ├── track_validity_efficiency_purity_vs_*.png
│   ├── roc_curve_track_validity.png
│   ├── track_validity_logit_distributions.png
│   └── task2_summary.txt
├── task3_regression/
│   ├── regression_residuals_*.png
│   ├── regression_residuals_*_vs_*.png
│   └── task3_summary.txt
└── EVALUATION_SUMMARY.md (this file)
```

## Notes

- All plots are saved as PNG files with 150 DPI resolution
- Efficiency and purity calculations use true kinematic variables
- ROC curves use raw logits where available
- Regression analysis includes normalized residuals and correlation plots
- The evaluation follows similar filtering approaches to the hit filter evaluation
