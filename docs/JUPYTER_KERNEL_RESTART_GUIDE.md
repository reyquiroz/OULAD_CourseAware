# Jupyter Kernel Restart Guide

## Issue: ImportError after config.py changes

If you see this error after we updated `src/config.py`:

```
ImportError: cannot import name 'RANDOM_STATE' from 'config'
```

**This is a Jupyter kernel caching issue.** The notebook has cached the old version of the config module.

## Solution: Restart the Jupyter Kernel

### Option 1: Restart Kernel (Recommended)
1. In Jupyter, click **Kernel** → **Restart Kernel**
2. Re-run all cells from the beginning

### Option 2: Restart Kernel & Clear Output
1. In Jupyter, click **Kernel** → **Restart Kernel and Clear All Outputs**
2. Re-run all cells from the beginning

### Option 3: Force Reload (Alternative)
Add this cell before importing config:

```python
import sys
import importlib

# Remove cached config module
if 'config' in sys.modules:
    del sys.modules['config']
    
# Now import will get fresh version
from config import RANDOM_STATE, ...
```

## Verification

After restarting, the first cell should output:

```
✓ Configuration loaded successfully
Project root: /Users/olivialoza/Documents/Development/OULAD
Data directory: /Users/olivialoza/Documents/Development/OULAD/DATA/raw
Results directory: /Users/olivialoza/Documents/Development/OULAD/results

Data files:
  studentInfo: studentInfo.csv
  studentVle: studentVle.csv
  studentAssessment: studentAssessment.csv
  assessments: assessments.csv
```

And `RANDOM_STATE` should be available with value `42`.

## Why This Happens

Python/Jupyter caches imported modules for performance. When you modify a `.py` file that's already been imported, the notebook continues using the cached version until you restart the kernel.

This is normal behavior and happens whenever you:
- Modify a Python module that's been imported
- Add new constants or functions to an imported module
- Change function/class definitions in an imported module

## Prevention

For development, you can use auto-reload magic:

```python
%load_ext autoreload
%autoreload 2
```

Add this at the top of your notebook to automatically reload modules when they change.

---

*Created: 2026-06-21*  
*Issue: Jupyter kernel caching after config.py update*