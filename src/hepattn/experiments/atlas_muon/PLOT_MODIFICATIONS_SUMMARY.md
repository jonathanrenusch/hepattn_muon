# ATLAS Style Plots - Plot Modifications Summary

## Date
October 13, 2025

## Changes Implemented

### 1. Hit Efficiency vs. Hit Purity Plot - Redesigned ✓

**Previous Design:**
- X-axis: Hit Purity (Precision)
- Y-axis: Hit Efficiency (Recall)
- Scatter plot with color gradient by working point
- Trend line overlay

**New Design:**
- **X-axis: Hit Efficiency (Working Point)** ⭐ SWAPPED
- **Y-axis: Hit Purity (Precision)** ⭐ SWAPPED
- **Simple line plot** (no scatter, no color coding)
- Cleaner visualization since working point is already on x-axis

**Rationale:**
- Eliminates redundant color coding (working point is now the x-axis itself)
- More direct interpretation: "As efficiency increases, how does purity change?"
- Cleaner, publication-ready plot without unnecessary visual elements

### 2. Rejection vs. Purity Plot - Enhanced with 0.99 Marker ✓

**Previous Design:**
- Scatter plot with color gradient by working point
- Color bar showing efficiency

**New Design:**
- Same scatter plot with color gradient
- **Red cross (×) marker at 0.99 efficiency working point** ⭐ NEW
- Marker properties:
  - Size: 200
  - Line width: 0.8
  - Color: Red
  - Z-order: 5 (on top)
  - Legend label showing actual working point value

**Rationale:**
- Highlights the specific working point (0.99 efficiency) used in the analysis
- Makes it easy for readers to identify the operational point in publications
- Maintains all other plot elements for context

## Test Results (100 events)

### Rejection vs Purity Plot
```
0.99 working point: Purity=0.4606, Rejection=0.9913
```
The red cross marker successfully highlights this point on the plot.

### Efficiency vs Purity Plot
```
Efficiency range: [0.9503, 0.9982]
Purity range: [0.0330, 0.8051]
```
The line plot now clearly shows purity as a function of efficiency.

## File Changes

### Modified Files:
1. `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/atlas_style_plots.py`
   - Updated `plot_atlas_efficiency_vs_purity()` method
   - Updated `plot_atlas_rejection_vs_purity()` method

2. `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/IMPLEMENTATION_SUMMARY.md`
   - Updated plot descriptions
   - Updated test results

3. `/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/ATLAS_STYLE_PLOTS_README.md`
   - Updated technical details section

## Visual Comparison

### Efficiency vs Purity Plot

**Before:**
```
   Efficiency (y)
        ^
        |  • scatter points (colored by WP)
        |  — trend line
        +---------------> Purity (x)
```

**After:**
```
   Purity (y)
        ^
        |  ──── simple line
        |
        +---------------> Efficiency (x) = Working Point
```

### Rejection vs Purity Plot

**Before:**
```
   Rejection (y)
        ^
        |  • • • scatter (colored)
        |
        +---------------> Purity (x)
```

**After:**
```
   Rejection (y)
        ^
        |  • • × • • scatter (colored)
        |      ↑
        |   0.99 WP (red cross)
        +---------------> Purity (x)
```

## Output Files Generated

All three plots maintain ATLAS style conventions:
1. ✅ `atlas_roc_curve.png` - Unchanged
2. ✅ `atlas_rejection_vs_purity.png` - Now with 0.99 marker
3. ✅ `atlas_efficiency_vs_purity.png` - Redesigned with swapped axes and line plot
4. ✅ `rejection_purity_data.csv` - Unchanged
5. ✅ `efficiency_purity_data.csv` - Unchanged

## Code Details

### Efficiency vs Purity - Key Code Changes

```python
# Sort by efficiency (now on x-axis)
sort_idx = np.argsort(efficiencies)
sorted_eff = efficiencies[sort_idx]
sorted_pur = purities[sort_idx]

# Simple line plot (efficiency on x-axis, purity on y-axis)
ax.plot(sorted_eff, sorted_pur, 'b-', linewidth=2, zorder=2)

# Swapped axis labels
ax.set_xlabel('Hit Efficiency (Working Point)', fontsize=14)
ax.set_ylabel('Hit Purity (Precision)', fontsize=14)
```

### Rejection vs Purity - Key Code Changes

```python
# Find point closest to 0.99
wp_target = 0.99
idx_099 = np.argmin(np.abs(valid_wps - wp_target))
purity_099 = purities[idx_099]
rejection_099 = rejections[idx_099]

# Plot red cross marker
ax.scatter(purity_099, rejection_099, marker='x', s=200, 
          linewidth=0.8, color='red', zorder=5, 
          label=f'Working Point {actual_wp_099:.4f}')

# Add legend
ax.legend(loc='lower right', fontsize=12)
```

## Verification

✅ Script runs successfully
✅ All three plots generated
✅ 0.99 marker appears on rejection plot
✅ Efficiency vs purity has swapped axes
✅ Line plot (no scatter) for efficiency vs purity
✅ ATLAS style maintained on all plots
✅ CSV data files generated

## Publication Readiness

Both modified plots are now ready for ATLAS CTD (Computing Technical Design) documents:
- Clear, uncluttered visualizations
- Operational working point (0.99) clearly marked
- Direct interpretation of efficiency-purity relationship
- Professional ATLAS style formatting
- High resolution (300 DPI)

## Usage

No changes to command-line interface:
```bash
cd /shared/tracking/hepattn_muon/src
pixi run python -m hepattn.experiments.atlas_muon.atlas_style_plots -m 100  # Test
pixi run python -m hepattn.experiments.atlas_muon.atlas_style_plots          # Full run
```
