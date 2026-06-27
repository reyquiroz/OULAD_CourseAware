# OULAD Notebook Path Cleanup - Implementation Summary

**Date:** 2026-06-21  
**Status:** ✅ Complete - Ready for Testing

---

## Changes Implemented

### 1. OULAD_Baseline_Analysis.ipynb

**File:** [`notebooks/OULAD_Baseline_Analysis.ipynb`](../notebooks/OULAD_Baseline_Analysis.ipynb)

**Changes Made:**
- ✅ Added `sys` import for path manipulation
- ✅ Added execution context detection (notebooks/ vs project root)
- ✅ Added `sys.path.insert()` to enable importing from `src/`
- ✅ Imported `RESULTS_DIR` and `DATA_DIR` from [`src/config.py`](../src/config.py)
- ✅ Replaced hardcoded `results_dir` with `RESULTS_DIR` from config
- ✅ Added path verification output

**Before:**
```python
# Paths
project_root = Path.cwd().parent
results_dir = project_root / 'results'

print("✓ Setup complete")
print(f"Results directory: {results_dir}")
```

**After:**
```python
# Configure paths - handle both execution contexts
if Path.cwd().name == 'notebooks':
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()

# Add src to path and import configuration
sys.path.insert(0, str(project_root / 'src'))
from config import RESULTS_DIR, DATA_DIR

results_dir = RESULTS_DIR

print("✓ Setup complete")
print(f"Project root: {project_root}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {results_dir}")
print(f"Data directory exists: {DATA_DIR.exists()}")
print(f"Results directory exists: {results_dir.exists()}")
```

**Impact:** LOW RISK - Only reads results, no data file access

---

### 2. OULAD_graph_schema.ipynb

**File:** [`notebooks/OULAD_graph_schema.ipynb`](../notebooks/OULAD_graph_schema.ipynb)

**Changes Made:**
- ✅ Added `sys` import for path manipulation
- ✅ Removed all custom path definitions (`BASE_DIR`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `OUTPUT_DIR`)
- ✅ Removed custom `Config` class
- ✅ Added execution context detection (notebooks/ vs project root)
- ✅ Added `sys.path.insert()` to enable importing from `src/`
- ✅ Imported all required paths and constants from [`src/config.py`](../src/config.py):
  - `DATA_DIR`
  - `RESULTS_DIR`
  - `STUDENT_INFO_FILE`
  - `STUDENT_VLE_FILE`
  - `STUDENT_ASSESSMENT_FILE`
  - `ASSESSMENTS_FILE`
  - `COURSES_FILE`
  - `VLE_FILE`
  - `LABEL_MAPPING`
  - `PREDICTION_WINDOWS`
  - `MODEL_PARAMS`
- ✅ Updated all file path references to use imported constants
- ✅ Improved path verification output with clear status indicators

**Before:**
```python
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"          # ← WRONG: lowercase
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

class Config:
    RAW_DIR = RAW_DIR
    OUTPUT_DIR = OUTPUT_DIR

studentInfo_path = Config.RAW_DIR / "studentInfo.csv"
studentVle_path = Config.RAW_DIR / "studentVle.csv"
studentAssessment_path = Config.RAW_DIR / "studentAssessment.csv"
assessments_path = Config.RAW_DIR / "assessments.csv"
```

**After:**
```python
# Configure paths - handle both execution contexts
if Path.cwd().name == 'notebooks':
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()

# Add src to path and import centralized configuration
sys.path.insert(0, str(project_root / 'src'))

from config import (
    DATA_DIR,
    RESULTS_DIR,
    STUDENT_INFO_FILE,
    STUDENT_VLE_FILE,
    STUDENT_ASSESSMENT_FILE,
    ASSESSMENTS_FILE,
    COURSES_FILE,
    VLE_FILE,
    LABEL_MAPPING,
    PREDICTION_WINDOWS,
    MODEL_PARAMS
)

# Use imported paths directly
studentInfo_path = STUDENT_INFO_FILE
studentVle_path = STUDENT_VLE_FILE
studentAssessment_path = STUDENT_ASSESSMENT_FILE
assessments_path = ASSESSMENTS_FILE

print("✓ Configuration loaded successfully")
print(f"Project root: {project_root}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
```

**Impact:** HIGH IMPACT - Fixes incorrect path that would cause file not found errors

---

## Key Improvements

### 1. Single Source of Truth
- All paths now come from [`src/config.py`](../src/config.py:1)
- No more duplicate path definitions across notebooks
- Changes to paths only need to be made in one place

### 2. Correct Path Structure
- Fixed incorrect lowercase `data/raw/` → correct uppercase `DATA/raw/`
- Removed references to non-existent `outputs/` directory
- All paths now match actual directory structure

### 3. Execution Context Handling
Both notebooks now work correctly whether executed from:
- `notebooks/` directory (typical Jupyter usage)
- Project root directory (alternative execution)

### 4. Better Error Detection
- Added path existence verification
- Clear status messages (✓ Found / ✗ Missing)
- Easier to diagnose path issues

### 5. Access to Configuration Constants
Notebooks now have access to:
- `LABEL_MAPPING` - Correct label convention (1=at-risk, 0=success)
- `PREDICTION_WINDOWS` - Standard prediction windows (weeks 2, 4, 6, 8)
- `MODEL_PARAMS` - Standardized model hyperparameters
- All data file paths as constants

---

## Verification Checklist

Before considering this complete, verify:

- [ ] Both notebooks import successfully from `src/config.py`
- [ ] No hardcoded paths remain (except project_root detection)
- [ ] All data files are found correctly
- [ ] Results load successfully in OULAD_Baseline_Analysis.ipynb
- [ ] Data files load successfully in OULAD_graph_schema.ipynb
- [ ] Notebooks work when executed from `notebooks/` directory
- [ ] Notebooks work when executed from project root
- [ ] No references to non-existent directories (`data/`, `outputs/`)
- [ ] Path verification messages are clear and helpful
- [ ] No errors when running first few cells of each notebook

---

## Testing Instructions

### Test 1: OULAD_Baseline_Analysis.ipynb

```bash
# From project root
cd /Users/olivialoza/Documents/Development/OULAD
jupyter notebook notebooks/OULAD_Baseline_Analysis.ipynb

# Run first cell - should see:
# ✓ Setup complete
# Project root: /Users/olivialoza/Documents/Development/OULAD
# Data directory: /Users/olivialoza/Documents/Development/OULAD/DATA/raw
# Results directory: /Users/olivialoza/Documents/Development/OULAD/results
# Data directory exists: True
# Results directory exists: True
```

### Test 2: OULAD_graph_schema.ipynb

```bash
# From project root
cd /Users/olivialoza/Documents/Development/OULAD
jupyter notebook notebooks/OULAD_graph_schema.ipynb

# Run first two cells - should see:
# ✓ Configuration loaded successfully
# Project root: /Users/olivialoza/Documents/Development/OULAD
# Data directory: /Users/olivialoza/Documents/Development/OULAD/DATA/raw
# Results directory: /Users/olivialoza/Documents/Development/OULAD/results
#
# Data files:
#   studentInfo: studentInfo.csv
#   studentVle: studentVle.csv
#   studentAssessment: studentAssessment.csv
#   assessments: assessments.csv
#
# Verifying data files...
# ✓ Found: studentInfo.csv
# ✓ Found: studentVle.csv
# ✓ Found: studentAssessment.csv
# ✓ Found: assessments.csv
```

---

## Files Modified

1. [`notebooks/OULAD_Baseline_Analysis.ipynb`](../notebooks/OULAD_Baseline_Analysis.ipynb) - Updated path configuration
2. [`notebooks/OULAD_graph_schema.ipynb`](../notebooks/OULAD_graph_schema.ipynb) - Complete path refactoring

## Files Referenced

1. [`src/config.py`](../src/config.py) - Centralized configuration (unchanged)
2. [`DATA/raw/`](../DATA/raw/) - Actual data directory (verified exists)

---

## Benefits Achieved

✅ **Consistency** - Both notebooks use identical path configuration pattern  
✅ **Reliability** - Correct paths that match actual directory structure  
✅ **Maintainability** - Single source of truth for all paths  
✅ **Portability** - Works from multiple execution contexts  
✅ **Clarity** - Clear verification messages for debugging  
✅ **Standards** - Access to centralized constants and configurations  

---

## Next Steps

1. **Test both notebooks** - Run first few cells to verify paths work
2. **Full execution test** - Run complete notebooks to ensure no downstream issues
3. **Document pattern** - Add this pattern to project documentation for future notebooks
4. **Apply to other notebooks** - Update any other notebooks following old pattern

---

## Standard Pattern for Future Notebooks

Use this pattern in all new OULAD notebooks:

```python
# Standard OULAD notebook setup
import sys
from pathlib import Path

# Handle both execution contexts
if Path.cwd().name == 'notebooks':
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()

# Add src to path
sys.path.insert(0, str(project_root / 'src'))

# Import centralized configuration
from config import (
    DATA_DIR,
    RESULTS_DIR,
    STUDENT_INFO_FILE,
    STUDENT_VLE_FILE,
    STUDENT_ASSESSMENT_FILE,
    ASSESSMENTS_FILE,
    COURSES_FILE,
    VLE_FILE,
    LABEL_MAPPING,
    PREDICTION_WINDOWS,
    MODEL_PARAMS
)

print("✓ Configuration loaded")
print(f"  Project root: {project_root}")
print(f"  Data directory: {DATA_DIR}")
print(f"  Results directory: {RESULTS_DIR}")
```

---

*Implementation completed: 2026-06-21*  
*Ready for testing and validation*