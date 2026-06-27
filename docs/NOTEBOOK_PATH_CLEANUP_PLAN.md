# OULAD Notebook Path Standardization Plan

## Executive Summary

This document outlines the plan to standardize path configurations across OULAD notebooks to ensure consistency with the project's centralized configuration system.

## Current State Analysis

### Actual Directory Structure (Verified)
```
/Users/olivialoza/Documents/Development/OULAD/
├── DATA/                          # ← UPPERCASE, at project root
│   ├── raw/                       # ← Contains actual CSV files
│   │   ├── studentInfo.csv
│   │   ├── studentVle.csv
│   │   ├── studentAssessment.csv
│   │   ├── assessments.csv
│   │   ├── courses.csv
│   │   └── vle.csv
│   └── processed/
├── src/
│   └── config.py                  # ← Centralized configuration
├── results/
│   ├── baseline/
│   └── lcpo/
└── notebooks/
    ├── OULAD_Baseline_Analysis.ipynb
    └── OULAD_graph_schema.ipynb
```

### Standard Configuration (src/config.py)
```python
PROJECT_ROOT = Path(__file__).parent.parent  # /Users/olivialoza/Documents/Development/OULAD
DATA_DIR = PROJECT_ROOT / "DATA/raw"         # Correct: DATA/raw (uppercase)
RESULTS_DIR = PROJECT_ROOT / "results"
```

**Key Files Defined:**
- `STUDENT_INFO_FILE = DATA_DIR / "studentInfo.csv"`
- `STUDENT_VLE_FILE = DATA_DIR / "studentVle.csv"`
- `STUDENT_ASSESSMENT_FILE = DATA_DIR / "studentAssessment.csv"`
- `ASSESSMENTS_FILE = DATA_DIR / "assessments.csv"`
- `COURSES_FILE = DATA_DIR / "courses.csv"`
- `VLE_FILE = DATA_DIR / "vle.csv"`

---

## Problem Analysis

### Notebook 1: OULAD_Baseline_Analysis.ipynb

**Current Approach:**
```python
project_root = Path.cwd().parent
results_dir = project_root / 'results'
```

**Issues:**
1. ❌ Does NOT import from `src/config.py`
2. ❌ Only defines `results_dir`, no data paths
3. ❌ Assumes notebook is in `notebooks/` subdirectory
4. ✅ Correctly uses `results/` for loading results

**Impact:** Low risk - only reads results, doesn't access raw data

---

### Notebook 2: OULAD_graph_schema.ipynb

**Current Approach:**
```python
BASE_DIR = Path.cwd()  # Assumes running from project root
DATA_DIR = BASE_DIR / "data"        # ← WRONG: lowercase "data"
RAW_DIR = DATA_DIR / "raw"          # ← Creates data/raw (doesn't exist)
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

class Config:
    RAW_DIR = RAW_DIR
    OUTPUT_DIR = OUTPUT_DIR
```

**Issues:**
1. ❌ Does NOT import from `src/config.py`
2. ❌ Uses lowercase `data/` instead of uppercase `DATA/`
3. ❌ Creates custom `Config` class instead of importing
4. ❌ Assumes notebook runs from project root (not `notebooks/`)
5. ❌ References non-existent `data/raw/` directory
6. ❌ Creates `outputs/` directory (not in standard structure)

**Impact:** HIGH risk - will fail to find data files

---

## Root Cause Analysis

### Why the Inconsistency Exists

1. **Historical Development**: Notebooks were created before `src/config.py` was standardized
2. **Different Authors/Times**: Different path conventions used at different stages
3. **Case Sensitivity**: Mix of `DATA/` (actual) vs `data/` (assumed)
4. **Working Directory Assumptions**: Some assume running from project root, others from `notebooks/`

---

## Standardization Strategy

### Design Principles

1. **Single Source of Truth**: All paths come from `src/config.py`
2. **Notebook-Friendly**: Handle both execution contexts (project root or notebooks/)
3. **Fail-Fast**: Clear error messages if paths don't exist
4. **Minimal Changes**: Preserve existing notebook logic where possible

### Standard Notebook Setup Pattern

```python
# Standard setup for all OULAD notebooks
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
    DATA_DIR,           # PROJECT_ROOT / "DATA/raw"
    RESULTS_DIR,        # PROJECT_ROOT / "results"
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

print(f"✓ Configuration loaded")
print(f"  Project root: {project_root}")
print(f"  Data directory: {DATA_DIR}")
print(f"  Results directory: {RESULTS_DIR}")
```

---

## Implementation Plan

### Phase 1: Update OULAD_Baseline_Analysis.ipynb

**Changes Required:**
1. Add `sys.path` setup to import from `src/`
2. Import `RESULTS_DIR` from config
3. Replace `project_root / 'results'` with `RESULTS_DIR`
4. Add verification that paths exist

**Risk Level:** LOW (only reads results, no data access)

**Code Changes:**
```python
# OLD
project_root = Path.cwd().parent
results_dir = project_root / 'results'

# NEW
import sys
from pathlib import Path

if Path.cwd().name == 'notebooks':
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()

sys.path.insert(0, str(project_root / 'src'))
from config import RESULTS_DIR

results_dir = RESULTS_DIR
```

---

### Phase 2: Update OULAD_graph_schema.ipynb

**Changes Required:**
1. Remove custom path definitions (`BASE_DIR`, `DATA_DIR`, `RAW_DIR`, etc.)
2. Remove custom `Config` class
3. Add `sys.path` setup
4. Import all paths from `src/config.py`
5. Update all file references to use imported constants

**Risk Level:** HIGH (accesses raw data files)

**Code Changes:**
```python
# OLD (REMOVE ALL OF THIS)
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

class Config:
    RAW_DIR = RAW_DIR
    OUTPUT_DIR = OUTPUT_DIR

studentInfo_path = Config.RAW_DIR / "studentInfo.csv"
studentVle_path = Config.RAW_DIR / "studentVle.csv"
# ... etc

# NEW (REPLACE WITH THIS)
import sys
from pathlib import Path

if Path.cwd().name == 'notebooks':
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()

sys.path.insert(0, str(project_root / 'src'))

from config import (
    DATA_DIR,
    RESULTS_DIR,
    STUDENT_INFO_FILE,
    STUDENT_VLE_FILE,
    STUDENT_ASSESSMENT_FILE,
    ASSESSMENTS_FILE,
    COURSES_FILE,
    VLE_FILE
)

# Use imported paths directly
studentInfo_path = STUDENT_INFO_FILE
studentVle_path = STUDENT_VLE_FILE
studentAssessment_path = STUDENT_ASSESSMENT_FILE
assessments_path = ASSESSMENTS_FILE
```

---

### Phase 3: Verification & Testing

**Test Cases:**
1. ✅ Run from `notebooks/` directory
2. ✅ Run from project root directory
3. ✅ Verify all data files are found
4. ✅ Verify results are loaded correctly
5. ✅ Check no hardcoded paths remain

**Verification Script:**
```python
# Add to first cell of each notebook
import os
print(f"Current working directory: {Path.cwd()}")
print(f"Project root: {project_root}")
print(f"Data directory exists: {DATA_DIR.exists()}")
print(f"Results directory exists: {RESULTS_DIR.exists()}")

# Verify key files
for file_path in [STUDENT_INFO_FILE, STUDENT_VLE_FILE, STUDENT_ASSESSMENT_FILE]:
    print(f"  {file_path.name}: {'✓' if file_path.exists() else '✗ MISSING'}")
```

---

## Migration Checklist

### OULAD_Baseline_Analysis.ipynb
- [ ] Add sys.path setup for src/ import
- [ ] Import RESULTS_DIR from config
- [ ] Replace hardcoded results_dir with RESULTS_DIR
- [ ] Add path verification
- [ ] Test from notebooks/ directory
- [ ] Test from project root
- [ ] Verify results load correctly

### OULAD_graph_schema.ipynb
- [ ] Remove custom BASE_DIR, DATA_DIR, RAW_DIR definitions
- [ ] Remove custom Config class
- [ ] Add sys.path setup for src/ import
- [ ] Import all required paths from config
- [ ] Update studentInfo_path to use STUDENT_INFO_FILE
- [ ] Update studentVle_path to use STUDENT_VLE_FILE
- [ ] Update studentAssessment_path to use STUDENT_ASSESSMENT_FILE
- [ ] Update assessments_path to use ASSESSMENTS_FILE
- [ ] Remove all references to non-existent data/ directory
- [ ] Remove OUTPUT_DIR references (use RESULTS_DIR instead)
- [ ] Add path verification
- [ ] Test from notebooks/ directory
- [ ] Test from project root
- [ ] Verify all data files load correctly

---

## Benefits of Standardization

1. **Single Source of Truth**: All paths defined in one place (`src/config.py`)
2. **Consistency**: All notebooks use identical path structure
3. **Maintainability**: Path changes only need to be made in config.py
4. **Reliability**: No more missing file errors due to incorrect paths
5. **Portability**: Works regardless of execution context
6. **Documentation**: Clear, centralized path definitions

---

## Post-Cleanup Validation

After implementing changes, verify:

1. ✅ Both notebooks import from `src/config.py`
2. ✅ No hardcoded paths remain (except project_root detection)
3. ✅ All data files are found correctly
4. ✅ Results load successfully
5. ✅ Notebooks work from both `notebooks/` and project root
6. ✅ No references to non-existent directories (`data/`, `outputs/`)
7. ✅ Path verification messages are clear and helpful

---

## Next Steps

1. **Review this plan** with the team
2. **Implement Phase 1** (OULAD_Baseline_Analysis.ipynb) - LOW RISK
3. **Implement Phase 2** (OULAD_graph_schema.ipynb) - HIGH IMPACT
4. **Test thoroughly** from both execution contexts
5. **Document** the standard pattern for future notebooks
6. **Update** any other notebooks following the same pattern

---

## Notes

- The actual directory is `DATA/` (uppercase), not `data/` (lowercase)
- Raw data files are in `DATA/raw/`, not `data/raw/`
- Standard results go to `results/`, not `outputs/`
- Always use `src/config.py` as the single source of truth for paths
- Handle both execution contexts (notebooks/ and project root)

---

*Created: 2026-06-21*  
*Purpose: Standardize path configuration across OULAD notebooks*  
*Status: Ready for implementation*