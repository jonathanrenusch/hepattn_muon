# Where to Add `eta` - Visual Guide

## ⚠️ CRITICAL: `eta` must be added in 5 places (4 required + 1 optional)

```
Your Config File Structure:
│
├── data:
│   └── inputs:
│       └── hit:                           ← 1. ADD eta HERE (line ~45)
│           - theta
│           - phi
│           - eta  ← ADD THIS
│
└── model:
    └── model:
        └── init_args:
            │
            ├── input_nets:
            │   └── - InputNet:
            │       ├── fields:            ← 2. ADD eta HERE (line ~165)
            │       │   - theta
            │       │   - phi
            │       │   - eta  ← ADD THIS
            │       │
            │       └── net:
            │           └── init_args:
            │               └── input_size: 23  ← 3. CHANGE from 22 to 23
            │
            │       └── posenc:            ← 4. OPTIONALLY ADD eta HERE
            │           └── fields:
            │               - r
            │               - theta
            │               - phi
            │               - eta  ← OPTIONAL
            │
            └── tasks:
                └── - WeightedPoolingObjectHitRegressionTask:
                    └── hit_fields:        ← 5. ADD eta HERE (line ~290)
                        - theta
                        - phi
                        - eta  ← ADD THIS
```

## Detailed Locations:

### 1️⃣ Data Inputs (REQUIRED)
**Line ~45** in your config
```yaml
data:
  inputs:
    hit:
      # ... 20 existing fields ...
      - theta
      - phi
      - eta  # ADD THIS - 21st field
```

### 2️⃣ InputNet Fields (REQUIRED)
**Line ~165** in your config
```yaml
            - class_path: hepattn.models.InputNet
              init_args:
                input_name: hit
                fields:
                  # ... 20 existing fields ...
                  - theta
                  - phi
                  - eta  # ADD THIS - makes 21 input fields
```

### 3️⃣ InputNet Size (REQUIRED)
**Line ~172** in your config
```yaml
                net:
                  class_path: hepattn.models.Dense
                  init_args:
                    input_size: 23  # CHANGE from 22 to 23
                    output_size: *dim
```

### 4️⃣ Position Encoding (OPTIONAL)
**Line ~185** in your config
```yaml
                posenc:
                  class_path: hepattn.models.posenc.PositionEncoder
                  init_args:
                    input_name: hit
                    dim: *dim
                    fields:
                      - r
                      - theta
                      - phi
                      - eta  # OPTIONAL - only if you want eta in posenc
                    sym_fields:
                      - phi
```

### 5️⃣ Regression Task hit_fields (REQUIRED)
**Line ~290** in your config (in the task definition)
```yaml
            - class_path: hepattn.models.task.WeightedPoolingObjectHitRegressionTask
              init_args:
                # ... other params ...
                hit_fields:
                  # ... 22 existing fields ...
                  - theta
                  - phi
                  - eta  # ADD THIS - 23rd hit field
```

## Count Check:
- **data.inputs.hit**: Should have 21 items (was 20)
- **InputNet.fields**: Should have 21 items (was 20) 
- **InputNet.net.input_size**: Should be **23** (was 22)
  - Note: input_size = 21 fields + 2 from position encoding = 23
- **Task.hit_fields**: Should have 23 items (was 22)

## Why 23 for input_size but only 21 fields?

The `input_size: 23` comes from:
- 21 raw hit features (including eta)
- +2 position encoded features (r, theta encoded)
- = 23 total

## Common Mistakes to Avoid:

❌ **Mistake 1**: Adding eta to data.inputs but forgetting InputNet.fields
❌ **Mistake 2**: Adding eta to fields but forgetting to update input_size
❌ **Mistake 3**: Wrong input_size calculation (should be 23, not 21)
❌ **Mistake 4**: Adding eta to task.hit_fields but not to data.inputs

✅ **Correct**: Add eta to ALL 4 required places + update input_size to 23
