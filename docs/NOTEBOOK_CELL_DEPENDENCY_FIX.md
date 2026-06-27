# Fix for Cell 17: "results_df is not defined"

## Error

```python
NameError: name 'results_df' is not defined
```

## Root Cause

Cell 17 tries to use `results_df`, but cell 16 didn't create it because:
1. `overall_results` list was empty (no results generated)
2. Cell 16 tried to concatenate empty `course_results` and failed
3. The error stopped execution before `results_df` could be created

## Solution: Fix Both Cells 16 and 17

### Cell 16: Add Error Handling

Replace cell 16 with:

```python
# Create results dataframes with error handling
if overall_results:
    results_df = pd.DataFrame(overall_results)
    results_df.to_csv(RESULTS_DIR / "overall_results.csv", index=False)
    print(f"✓ Saved overall results: {len(results_df)} rows")
    print(f"  Columns: {list(results_df.columns)}")
else:
    print("⚠ WARNING: No overall results generated!")
    print("  Creating empty dataframe to prevent downstream errors")
    results_df = pd.DataFrame()  # Create empty dataframe

if course_results:
    course_df = pd.concat(course_results, ignore_index=True)
    course_df.to_csv(RESULTS_DIR / "per_course_results.csv", index=False)
    print(f"✓ Saved per-course results: {len(course_df)} rows")
else:
    print("⚠ WARNING: No per-course results generated!")
    print("  Creating empty dataframe to prevent downstream errors")
    course_df = pd.DataFrame()  # Create empty dataframe

print(f"\nResults directory: {RESULTS_DIR}")
```

### Cell 17: Add Safety Check

Replace cell 17 with:

```python
# Build random AUROC (from overall results) - with safety check
if not results_df.empty and "split" in results_df.columns:
    random_df = (
        results_df[results_df["split"] == "random"]
        .groupby(["week", "model", "feature_group"], as_index=False)["AUROC"]
        .mean()
        .rename(columns={"AUROC": "random"})
    )
    print(f"✓ Created random_df: {len(random_df)} rows")
else:
    print("⚠ Cannot create random_df: results_df is empty or missing 'split' column")
    print("  This means the evaluation loop (cell 15) didn't generate any results")
    print("\nTroubleshooting steps:")
    print("  1. Check if previous cells ran successfully")
    print("  2. Verify 'df' dataframe exists and has required columns")
    print("  3. Check for error messages in cell 15 output")
    print("  4. Ensure 'weeks', 'feature_groups', and 'models' are defined")
    random_df = pd.DataFrame()  # Create empty to prevent further errors
```

## Why Results Are Empty

The evaluation loop (likely cell 15) didn't generate any results. Common causes:

### 1. Missing Variables

Check if these are defined before the loop:
```python
# Add this diagnostic cell before cell 15
print("Checking required variables:")
print(f"  'df' defined: {'df' in globals()}")
print(f"  'weeks' defined: {'weeks' in globals()}")
print(f"  'feature_groups' defined: {'feature_groups' in globals()}")
print(f"  'models' defined: {'models' in globals()}")

if 'df' in globals():
    print(f"\nDataframe info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")
```

### 2. Missing Columns

The loop skips feature groups with missing columns. Check output for:
```
Skipping group=demographics (missing ['at_risk', 'code_module'])
```

### 3. All Iterations Failed

Check for error messages in cell 15 output:
```
ERROR:
week=2, group=demographics, model=logreg
<error details>
```

### 4. Functions Not Defined

Ensure these functions exist:
- `run_random_split(df, features, model, week)`
- `run_lcpo(df, features, model, week)`

## Complete Diagnostic Approach

Add this cell **before** cell 15 (the evaluation loop):

```python
# === DIAGNOSTIC CELL ===
print("=" * 60)
print("PRE-EVALUATION DIAGNOSTICS")
print("=" * 60)

# Check variables
required_vars = ['df', 'weeks', 'feature_groups', 'models']
for var in required_vars:
    exists = var in globals()
    print(f"✓ {var}: {'EXISTS' if exists else '✗ MISSING'}")
    
# Check dataframe
if 'df' in globals():
    print(f"\nDataframe 'df':")
    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['at_risk', 'code_module', 'code_presentation', 'week']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing required columns: {missing}")
    else:
        print(f"  ✓ All required columns present")
else:
    print("\n✗ ERROR: 'df' not defined! Cannot proceed with evaluation.")
    
# Check weeks
if 'weeks' in globals():
    print(f"\nWeeks to evaluate: {weeks}")
    
# Check feature groups
if 'feature_groups' in globals():
    print(f"\nFeature groups: {list(feature_groups.keys())}")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features")
        
# Check models
if 'models' in globals():
    print(f"\nModels: {list(models.keys())}")
    
# Check functions
print(f"\nFunctions:")
print(f"  run_random_split: {'EXISTS' if 'run_random_split' in globals() else '✗ MISSING'}")
print(f"  run_lcpo: {'EXISTS' if 'run_lcpo' in globals() else '✗ MISSING'}")

print("=" * 60)
print("If any items show ✗ MISSING, run the cells that define them first!")
print("=" * 60)
```

## Quick Fix: Use Existing Results

If you can't get the notebook to generate results, use the existing standalone script results:

```python
# Load existing results instead
import pandas as pd
from pathlib import Path

# Load baseline results (random split)
baseline_path = RESULTS_DIR / "baseline" / "baseline_results_detailed.csv"
if baseline_path.exists():
    results_df = pd.read_csv(baseline_path)
    results_df["split"] = "random"  # Add split column
    print(f"✓ Loaded baseline results: {len(results_df)} rows")
else:
    print(f"✗ Baseline results not found at: {baseline_path}")
    results_df = pd.DataFrame()

# Load LCPO results
lcpo_path = RESULTS_DIR / "lcpo" / "lcpo_results_detailed.csv"
if lcpo_path.exists():
    course_df = pd.read_csv(lcpo_path)
    course_df["split"] = "lcpo"  # Add split column
    print(f"✓ Loaded LCPO results: {len(course_df)} rows")
else:
    print(f"✗ LCPO results not found at: {lcpo_path}")
    course_df = pd.DataFrame()
```

---

*Created: 2026-06-21*  
*Issue: Cascading errors from empty results*