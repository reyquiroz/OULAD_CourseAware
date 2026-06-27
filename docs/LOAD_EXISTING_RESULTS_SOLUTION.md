# Load Existing Results - Corrected Solution

## Problem

The notebook expects columns: `week`, `model`, `feature_group`, `AUROC`  
But baseline results have: `Week`, `Model`, `Features`, `AUROC_mean`

## Corrected Solution for Cells 16-17

Replace cells 16 and 17 with this code:

```python
# ===================================================================
# LOAD EXISTING RESULTS (instead of generating from scratch)
# ===================================================================

import pandas as pd

# Load existing baseline results
baseline_path = RESULTS_DIR / "baseline" / "baseline_results_detailed.csv"
if baseline_path.exists():
    results_df = pd.read_csv(baseline_path)
    
    # Rename columns to match expected format
    results_df = results_df.rename(columns={
        'Week': 'week',
        'Model': 'model', 
        'Features': 'feature_group',
        'AUROC_mean': 'AUROC'
    })
    
    # Add split column
    results_df["split"] = "random"
    
    print(f"✓ Loaded baseline results: {len(results_df)} rows")
    print(f"  Columns: {list(results_df.columns)[:8]}...")
    print(f"  Weeks: {sorted(results_df['week'].unique())}")
    print(f"  Models: {results_df['model'].unique()}")
    print(f"  Feature groups: {results_df['feature_group'].unique()}")
else:
    print(f"✗ Baseline results not found at: {baseline_path}")
    results_df = pd.DataFrame()

# Load existing LCPO results
lcpo_path = RESULTS_DIR / "lcpo" / "lcpo_results_detailed.csv"
if lcpo_path.exists():
    course_df = pd.read_csv(lcpo_path)
    
    # Check what columns LCPO has and rename if needed
    print(f"\n✓ Loaded LCPO results: {len(course_df)} rows")
    print(f"  Columns: {list(course_df.columns)[:8]}...")
    
    # Rename if needed (check actual column names first)
    if 'Week' in course_df.columns:
        course_df = course_df.rename(columns={
            'Week': 'week',
            'Model': 'model',
            'Features': 'feature_group',
            'AUROC_mean': 'AUROC'
        })
    
    course_df["split"] = "lcpo"
else:
    print(f"✗ LCPO results not found at: {lcpo_path}")
    course_df = pd.DataFrame()

print(f"\n{'='*60}")
print("Results loaded successfully!")
print(f"{'='*60}")
```

## Then Cell 17 (Analysis) Will Work:

```python
# Build random AUROC (from overall results)
if not results_df.empty:
    random_df = (
        results_df[results_df["split"] == "random"]
        .groupby(["week", "model", "feature_group"], as_index=False)["AUROC"]
        .mean()
        .rename(columns={"AUROC": "random"})
    )
    print(f"✓ Created random_df: {len(random_df)} rows")
    print(random_df.head())
else:
    print("⚠ Cannot create random_df: results_df is empty")
    random_df = pd.DataFrame()
```

## Column Mapping Reference

| Baseline CSV Column | Expected Column | Description |
|---------------------|-----------------|-------------|
| `Week` | `week` | Prediction window (2, 4, 6, 8) |
| `Model` | `model` | Model name |
| `Features` | `feature_group` | Feature group name |
| `AUROC_mean` | `AUROC` | Mean AUROC score |
| `AUPRC_mean` | `AUPRC` | Mean AUPRC score |
| `F1_mean` | `F1` | Mean F1 score |

## Verification

After running the corrected code, verify:

```python
# Add this verification cell
print("Verification:")
print(f"  results_df shape: {results_df.shape}")
print(f"  results_df columns: {list(results_df.columns)}")
print(f"\nSample data:")
print(results_df[['week', 'model', 'feature_group', 'AUROC', 'split']].head())
```

Expected output:
```
Verification:
  results_df shape: (68, 17)
  results_df columns: ['week', 'model', 'feature_group', 'N_features', 'AUROC', ...]

Sample data:
   week               model feature_group     AUROC  split
0     2            Majority  All_features  0.500000  random
1     2  LogisticRegression  All_features  0.615211  random
...
```

---

*Created: 2026-06-21*  
*Issue: Column name mismatch between baseline results and notebook expectations*