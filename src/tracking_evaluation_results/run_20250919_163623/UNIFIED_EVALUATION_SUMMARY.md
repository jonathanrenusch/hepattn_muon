# ATLAS Muon Tracking Model - Unified Evaluation Summary

**Generated at:** 2025-09-19 16:36:27

## Overview

This unified evaluation covers three main tasks across three comparison regions:

### Tasks:
1. **Task 1**: Hit-Track Assignment (track_hit_valid)
2. **Task 2**: Track Validity Classification (track_valid)
3. **Task 3**: Regression Outputs (parameter_regression)

### Comparison Regions:
- **All Tracks**: Complete dataset without filtering
- **Baseline Filtered**: Tracks meeting quality criteria (≥3 stations, ≥3 hits/station, eta/pt cuts)
- **Rejected Tracks**: Tracks not meeting baseline criteria

## Results Summary

| Region | Task 1 AUC | Task 2 AUC | Task 1 Avg Eff | Task 2 Avg Eff | Eta Std | Phi Std | qpt Std |
|--------|------------|------------|----------------|----------------|---------|---------|----------|
| All Tracks | 0.9224 | nan | 0.2293 | 0.2500 | nan | nan | 0.058289 |
| Baseline Filtered | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 |
| Rejected Tracks | 0.0000 | nan | 0.0000 | 0.2500 | 0.000000 | 0.000000 | 0.000000 |

## Directory Structure

```
unified_tracking_evaluation/
├── all_tracks/
│   ├── task1_roc_curve_all_tracks.png
│   ├── task1_all_tracks_efficiency_fake_rate_vs_pt.png
│   ├── task2_roc_curve_all_tracks.png
│   ├── task2_all_tracks_efficiency_fake_rate_vs_pt.png
│   ├── task3_*_residuals_all_tracks_distribution.png
│   └── all_tracks_summary.txt
├── baseline_filtered_tracks/
│   └── (same structure as all_tracks)
├── rejected_tracks/
│   └── (same structure as all_tracks)
├── evaluation_config_log.yaml
└── UNIFIED_EVALUATION_SUMMARY.md (this file)
```

## Notes

- All plots use consistent styling with hit filter evaluation
- Step plots with error bands for efficiency/fake rate analysis
- Transparent overlaid distributions for regression comparison
- Results logged to configuration file for easy analysis
- Baseline filtering follows same criteria as hit filter evaluation
