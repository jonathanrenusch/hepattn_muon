# Background Rejection vs. Purity Plot - Explanation

## What This Plot Shows

The **Background Rejection vs. Purity** plot displays the fundamental trade-off in classification performance for the muon hit filtering model.

## Understanding the Axes

### X-axis: Signal Purity (Precision)
- **Definition**: Purity = True Positives / (True Positives + False Positives)
- **Meaning**: Of all hits we accept as "signal", what fraction are actually true signal hits?
- **Higher is better**: More pure means fewer false positives contaminating our signal

### Y-axis: Background Rejection Rate
- **Definition**: Rejection = True Negatives / (True Negatives + False Positives)
- **Meaning**: Of all background hits, what fraction do we correctly reject?
- **Higher is better**: More rejection means we're successfully filtering out noise

## The Trade-off Curve

The curve shows operating points at different **signal efficiency** (recall) thresholds from 0.96 to 0.999.

### Physical Interpretation:

1. **High Efficiency (0.999) → Bottom-Left**
   - We want to capture 99.9% of true signal hits
   - To achieve this, we must be very lenient with our cuts
   - Result: Lower purity (~1.5%), Lower rejection (~50%)
   - We accept many false positives to avoid missing any signal

2. **Low Efficiency (0.96) → Top-Right**
   - We only need to capture 96% of true signal hits
   - We can afford to be more selective
   - Result: Higher purity (~77%), Higher rejection (~99.8%)
   - We reject more aggressively, losing some signal but gaining cleanliness

### Why Does Rejection Decrease with Increasing Purity?

This might seem counterintuitive at first, but it makes sense:

- **Low Purity Region**: We're accepting almost everything (high efficiency target). Even though purity is low, we're rejecting very few background hits because our threshold is lenient.

- **High Purity Region**: We're being more selective. We achieve high purity by accepting fewer hits overall, but this means we also reject more background.

Wait, this explanation has an error. Let me reconsider...

Actually, looking at the data:
- At efficiency 0.96 (selective): Purity = 77.4%, Rejection = 99.8%
- At efficiency 0.999 (lenient): Purity = 1.5%, Rejection = 50.5%

This makes perfect sense!

### Correct Interpretation:

As we move from **left to right** on the plot (increasing purity):
- We become more **selective** (lower efficiency targets)
- We reject **more background** (higher rejection rate)
- We achieve **higher purity** (fewer false positives get through)

This is the expected and correct behavior! The curve shows that:
- **High purity comes with high rejection** (top-right: very selective cuts)
- **Low purity comes with low rejection** (bottom-left: very lenient cuts)

## Choosing an Operating Point

The choice of operating point depends on your physics goals:

- **Need high signal efficiency?** → Accept lower purity and rejection (bottom-left)
- **Need clean sample?** → Accept lower signal efficiency for high purity and rejection (top-right)
- **Balanced approach?** → Choose a point in the middle of the curve

## ATLAS Style Conventions

The plot follows ATLAS publication guidelines:
- Simple, clear markers and lines
- No color coding (black/white printable)
- Clear axis labels and annotations
- Professional typography
- Annotated with efficiency values at key points

## Technical Notes

- Working points swept: 0.96 to 0.999 in steps of 0.001 (40 points)
- Each point represents a different classification threshold
- Data saved to CSV for further analysis
- Plot is publication-ready for ATLAS internal notes
