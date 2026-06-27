# Fix for "No objects to concatenate" Error

## Error Description

```python
ValueError: No objects to concatenate
```

This occurs in cell 16 when trying to concatenate `course_results`:

```python
course_df = pd.concat(course_results, ignore_index=True)
```

## Root Cause

The `course_results` list is empty, meaning no LCPO results were generated. This happens when:
1. All iterations hit exceptions in the try/except block
2. Missing columns caused all feature groups to be skipped
3. Required variables (`weeks`, `feature_groups`, `models`, or `df`) are not defined

## Solution

Replace cell 16 content with this error-handling version:

```python
# Create results dataframes with error handling
if overall_results:
    results_df = pd.DataFrame(overall_results)
    results_df.to_csv(RESULTS_DIR / "overall_results.csv", index=False)
    print(f"✓ Saved overall results: {len(results_df)} rows")
else:
    print("⚠ No overall results to save")

if course_results:
    course_df = pd.concat(course_results, ignore_index=True)
    course_df.to_csv(RESULTS_DIR / "per_course_results.csv", index=False)
    print(f"✓ Saved per-course results: {len(course_df)} rows")
else:
    print("⚠ No per-course results to save")

print(f"\nResults saved to: {RESULTS_DIR}")
```

## Debugging Steps

If you see "⚠ No per-course results to save", check:

### 1. Check if previous cells ran successfully
```python
# Add this before the loop to verify variables exist
print(f"weeks defined: {'weeks' in globals()}")
print(f"feature_groups defined: {'feature_groups' in globals()}")
print(f"models defined: {'models' in globals()}")
print(f"df defined: {'df' in globals()}")

if 'df' in globals():
    print(f"df shape: {df.shape}")
    print(f"df columns: {list(df.columns)}")
```

### 2. Check for missing columns
The loop skips feature groups with missing columns. Add this after the loop:

```python
# Check what happened
print(f"\nResults summary:")
print(f"  overall_results: {len(overall_results)} items")
print(f"  course_results: {len(course_results)} items")

if len(course_results) == 0:
    print("\n⚠ No course results were generated!")
    print("Check the output above for 'Skipping group=' messages")
    print("or 'ERROR:' messages that indicate what went wrong")
```

### 3. Common Issues

**Issue: Missing required columns**
```
Skipping group=demographics (missing ['at_risk', 'code_module'])
```
**Fix:** Ensure the dataframe `df` has all required columns before the loop

**Issue: All iterations failed**
```
ERROR:
week=2, group=demographics, model=logreg
...
```
**Fix:** Check the error messages to see what's failing in `run_random_split()` or `run_lcpo()`

**Issue: Variables not defined**
```
NameError: name 'weeks' is not defined
```
**Fix:** Run all previous cells in order, especially those defining `weeks`, `feature_groups`, `models`, and `df`

## Prevention

Add this at the start of the results-saving cell:

```python
# Validate we have results before trying to save
assert len(overall_results) > 0, "No overall results generated! Check previous cells for errors."
assert len(course_results) > 0, "No course results generated! Check previous cells for errors."
```

This will give a clear error message if results are missing.

---

*Created: 2026-06-21*  
*Issue: Empty results list causing concat error*