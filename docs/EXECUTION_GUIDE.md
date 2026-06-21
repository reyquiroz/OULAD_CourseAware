# OULAD Evaluation Execution Guide

**Purpose**: Step-by-step guide to execute all evaluations and generate complete results  
**Date**: June 20, 2026  
**Estimated Total Time**: 6-10 hours (depending on computational resources)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Execution Steps](#execution-steps)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Expected Outputs](#expected-outputs)

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: At least 5GB free space
- **CPU**: Multi-core processor recommended for faster execution

### Data Requirements

Ensure all OULAD datasets are in `data/raw/`:
- ✅ `studentInfo.csv`
- ✅ `studentVle.csv`
- ✅ `studentAssessment.csv`
- ✅ `assessments.csv`
- ✅ `vle.csv`
- ✅ `courses.csv`

**Verify data exists**:
```bash
ls -lh data/raw/*.csv
```

---

## Environment Setup

### Step 1: Activate Virtual Environment

```bash
# Navigate to project root
cd /Users/olivialoza/Documents/Development/OULAD

# Activate virtual environment
source oulad_env/bin/activate

# Verify Python version
python --version  # Should be 3.8+
```

### Step 2: Verify Dependencies

```bash
# Check if all packages are installed
pip list | grep -E "pandas|numpy|scikit-learn|xgboost|lightgbm"

# If any are missing, install from requirements.txt
pip install -r requirements.txt
```

### Step 3: Verify Configuration

```bash
# Test configuration import
python -c "from src.config import DATA_DIR, RESULTS_DIR; print(f'Data: {DATA_DIR}'); print(f'Results: {RESULTS_DIR}')"
```

Expected output:
```
Data: /Users/olivialoza/Documents/Development/OULAD/data/raw
Results: /Users/olivialoza/Documents/Development/OULAD/results
```

---

## Execution Steps

### Phase 1: Baseline Evaluation (2-3 hours)

**Purpose**: Random 5-fold cross-validation across all prediction windows

```bash
cd src
python baseline_evaluation.py
```

**What it does**:
- Loads OULAD data with correct label mapping (1=at-risk, 0=success)
- Creates features for weeks 4, 8, 12, 16 with temporal leakage prevention
- Trains 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Performs 5-fold stratified cross-validation
- Saves detailed results to `results/baseline/baseline_results_detailed.csv`

**Progress indicators**:
```
Loading OULAD data...
✓ Loaded 32,593 students
Creating features for week 4...
✓ Created 87 features for 32,593 students
Training Logistic Regression...
  Fold 1/5... AUROC: 0.723
  Fold 2/5... AUROC: 0.718
  ...
```

**Expected runtime**: 30-45 minutes per prediction window × 4 windows = 2-3 hours

---

### Phase 2: LCPO Evaluation (2-3 hours)

**Purpose**: Leave-Course-Presentation-Out cross-validation for cross-course generalization

```bash
cd src
python lcpo_evaluation.py
```

**What it does**:
- Identifies 22 unique course-presentation combinations
- For each course-presentation:
  - Trains on remaining 21 courses
  - Tests on held-out course
- Evaluates all 4 models across all prediction windows
- Saves results to `results/lcpo/lcpo_results_detailed.csv`

**Progress indicators**:
```
Starting LCPO Evaluation...
Found 22 unique course-presentations
Week 4, Course 1/22: AAA-2013J
  Training on 21 courses, testing on AAA-2013J
  Logistic Regression: AUROC=0.682
  Random Forest: AUROC=0.701
  ...
```

**Expected runtime**: 5-8 minutes per course × 22 courses × 4 weeks = 2-3 hours

---

### Phase 3: Future-Presentation Evaluation (1-2 hours)

**Purpose**: Temporal generalization - train on past presentations, test on future

```bash
cd src
python future_presentation_evaluation.py
```

**What it does**:
- Sorts presentations chronologically
- For each course module:
  - Trains on earlier presentations
  - Tests on later presentations
- Evaluates temporal generalization capability
- Saves results to `results/cross_course/future_presentation_results.csv`

**Progress indicators**:
```
Starting Future-Presentation Evaluation...
Processing course: AAA
  Training on: 2013J, 2013B
  Testing on: 2014J, 2014B
  Week 4 - Logistic Regression: AUROC=0.695
  ...
```

**Expected runtime**: 1-2 hours

---

### Phase 4: Feature Group Comparison (2-3 hours)

**Purpose**: Compare performance across different feature groups

**Option A: Create dedicated script**

Create `src/feature_group_evaluation.py`:

```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR, LABEL_MAPPING, RANDOM_STATE

def load_and_prepare_data():
    """Load OULAD data"""
    student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
    student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
    student_assessment = pd.read_csv(DATA_DIR / "studentAssessment.csv")
    assessments = pd.read_csv(DATA_DIR / "assessments.csv")
    vle = pd.read_csv(DATA_DIR / "vle.csv")
    
    student_info["target"] = student_info["final_result"].map(LABEL_MAPPING)
    
    return student_info, student_vle, student_assessment, assessments, vle

def create_feature_groups(features_df):
    """Define feature groups"""
    all_cols = features_df.columns.tolist()
    
    # Demographic features
    demo_features = [c for c in all_cols if any(x in c for x in 
                    ['gender_', 'region_', 'education_', 'imd_', 'age_', 'disability_', 
                     'num_of_prev_attempts', 'studied_credits'])]
    
    # VLE features
    vle_features = [c for c in all_cols if 'vle_' in c]
    
    # Assessment features
    assess_features = [c for c in all_cols if 'assess_' in c]
    
    return {
        'Demographics Only': demo_features,
        'VLE Only': vle_features,
        'Assessment Only': assess_features,
        'Combined': demo_features + vle_features + assess_features
    }

def evaluate_feature_group(X, y, feature_cols, model_name, model):
    """Evaluate a specific feature group"""
    X_group = X[feature_cols]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aurocs = []
    auprcs = []
    
    for train_idx, test_idx in skf.split(X_group, y):
        X_train, X_test = X_group.iloc[train_idx], X_group.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        aurocs.append(roc_auc_score(y_test, y_pred_proba))
        auprcs.append(average_precision_score(y_test, y_pred_proba))
    
    return {
        'AUROC_mean': np.mean(aurocs),
        'AUROC_std': np.std(aurocs),
        'AUPRC_mean': np.mean(auprcs),
        'AUPRC_std': np.std(auprcs)
    }

def main():
    print("="*80)
    print("FEATURE GROUP COMPARISON EVALUATION")
    print("="*80)
    
    # Load data and create features for week 8
    print("\nLoading data and creating features...")
    # [Add feature creation code here - similar to baseline_evaluation.py]
    
    # Define feature groups
    feature_groups = create_feature_groups(features_df)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    }
    
    # Evaluate each combination
    results = []
    for group_name, feature_cols in feature_groups.items():
        print(f"\n{group_name} ({len(feature_cols)} features):")
        
        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ")
            metrics = evaluate_feature_group(X, y, feature_cols, model_name, model)
            print(f"AUROC: {metrics['AUROC_mean']:.3f}±{metrics['AUROC_std']:.3f}")
            
            results.append({
                'Feature_Group': group_name,
                'Model': model_name,
                'Num_Features': len(feature_cols),
                **metrics
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "feature_group_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
```

**Run it**:
```bash
cd src
python feature_group_evaluation.py
```

**Option B: Use consolidated notebook**

Open `notebooks/OULAD_Consolidated_Analysis.ipynb` and run the feature group comparison cells.

---

## Verification

### Check Result Files

```bash
# List all result files
find results -name "*.csv" -type f

# Expected files:
# results/baseline/baseline_results_detailed.csv
# results/lcpo/lcpo_results_detailed.csv
# results/cross_course/future_presentation_results.csv
# results/feature_group_comparison.csv
```

### Verify Result Contents

```bash
# Check baseline results
head -5 results/baseline/baseline_results_detailed.csv
wc -l results/baseline/baseline_results_detailed.csv  # Should be ~81 rows (4 models × 4 weeks × 5 folds + header)

# Check LCPO results
head -5 results/lcpo/lcpo_results_detailed.csv
wc -l results/lcpo/lcpo_results_detailed.csv  # Should be ~89 rows (22 courses × 4 models + header)

# Check future-presentation results
head -5 results/cross_course/future_presentation_results.csv
```

### Quick Analysis

```python
# In Python or Jupyter
import pandas as pd

# Load baseline results
baseline = pd.read_csv('results/baseline/baseline_results_detailed.csv')
print("Baseline Results:")
print(baseline.groupby('model')['AUROC'].agg(['mean', 'std']))

# Load LCPO results
lcpo = pd.read_csv('results/lcpo/lcpo_results_detailed.csv')
print("\nLCPO Results:")
print(lcpo.groupby('model')['AUROC'].agg(['mean', 'std']))

# Compare
print("\nPerformance Drop (Baseline → LCPO):")
baseline_mean = baseline.groupby('model')['AUROC'].mean()
lcpo_mean = lcpo.groupby('model')['AUROC'].mean()
print(baseline_mean - lcpo_mean)
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'config'`

**Solution**:
```bash
# Ensure you're in the src directory
cd src
python baseline_evaluation.py

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/olivialoza/Documents/Development/OULAD/src"
python baseline_evaluation.py
```

---

### Issue 2: FileNotFoundError for data

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/studentInfo.csv'`

**Solution**:
```bash
# Check if data exists
ls -lh data/raw/

# If data is in different location, update config.py
# Or create symlink
ln -s /path/to/actual/data data/raw
```

---

### Issue 3: Memory Error

**Error**: `MemoryError` or system becomes unresponsive

**Solution**:
1. Close other applications
2. Reduce batch size in evaluation scripts
3. Process one prediction window at a time
4. Use a machine with more RAM

---

### Issue 4: XGBoost Feature Name Error

**Error**: `ValueError: feature_names may not contain [, ] or <`

**Solution**: Already fixed in code with `clean_feature_names()` function. If still occurs:
```python
# In feature engineering code
df.columns = df.columns.str.replace('[', '_', regex=False)
df.columns = df.columns.str.replace(']', '_', regex=False)
df.columns = df.columns.str.replace('<', '_lt_', regex=False)
df.columns = df.columns.str.replace('>', '_gt_', regex=False)
```

---

### Issue 5: Slow Execution

**Symptoms**: Evaluation taking much longer than expected

**Solutions**:
1. **Use fewer folds**: Change `n_splits=5` to `n_splits=3` in StratifiedKFold
2. **Reduce estimators**: Change `n_estimators=100` to `n_estimators=50`
3. **Skip some models**: Comment out slower models (Random Forest, XGBoost)
4. **Process fewer weeks**: Test with just week 8 first
5. **Use parallel processing**: Add `n_jobs=-1` to model parameters

---

## Expected Outputs

### Baseline Results Structure

**File**: `results/baseline/baseline_results_detailed.csv`

**Columns**:
- `model`: Model name (Logistic Regression, Random Forest, XGBoost, LightGBM)
- `prediction_week`: Week number (4, 8, 12, 16)
- `fold`: Fold number (1-5)
- `AUROC`: Area under ROC curve
- `AUPRC`: Area under precision-recall curve
- `Precision`: Precision for at-risk class
- `Recall`: Recall for at-risk class
- `F1`: F1 score for at-risk class

**Expected rows**: ~81 (4 models × 4 weeks × 5 folds + 1 header)

**Sample**:
```csv
model,prediction_week,fold,AUROC,AUPRC,Precision,Recall,F1
Logistic Regression,4,1,0.723,0.456,0.512,0.678,0.584
Logistic Regression,4,2,0.718,0.449,0.508,0.671,0.579
...
```

---

### LCPO Results Structure

**File**: `results/lcpo/lcpo_results_detailed.csv`

**Columns**:
- `model`: Model name
- `prediction_week`: Week number
- `test_course`: Course-presentation held out for testing
- `train_size`: Number of training samples
- `test_size`: Number of test samples
- `AUROC`: Area under ROC curve
- `AUPRC`: Area under precision-recall curve
- `Precision`: Precision for at-risk class
- `Recall`: Recall for at-risk class
- `F1`: F1 score for at-risk class

**Expected rows**: ~89 (22 courses × 4 models + 1 header)

**Sample**:
```csv
model,prediction_week,test_course,train_size,test_size,AUROC,AUPRC,Precision,Recall,F1
Logistic Regression,8,AAA-2013J,28543,1250,0.682,0.423,0.489,0.645,0.556
Random Forest,8,AAA-2013J,28543,1250,0.701,0.445,0.503,0.667,0.574
...
```

---

### Feature Group Comparison Structure

**File**: `results/feature_group_comparison.csv`

**Columns**:
- `Feature_Group`: Group name (Demographics Only, VLE Only, Assessment Only, Combined)
- `Model`: Model name
- `Num_Features`: Number of features in group
- `AUROC_mean`: Mean AUROC across folds
- `AUROC_std`: Standard deviation of AUROC
- `AUPRC_mean`: Mean AUPRC across folds
- `AUPRC_std`: Standard deviation of AUPRC

**Expected rows**: ~17 (4 feature groups × 4 models + 1 header)

**Sample**:
```csv
Feature_Group,Model,Num_Features,AUROC_mean,AUROC_std,AUPRC_mean,AUPRC_std
Demographics Only,Logistic Regression,23,0.645,0.012,0.389,0.015
VLE Only,Logistic Regression,8,0.712,0.009,0.441,0.011
Assessment Only,Logistic Regression,7,0.698,0.011,0.428,0.013
Combined,Logistic Regression,38,0.756,0.008,0.478,0.010
...
```

---

## Performance Benchmarks

### Expected AUROC Ranges

**Baseline (Random 5-fold CV)**:
- Week 4: 0.70-0.75
- Week 8: 0.75-0.80
- Week 12: 0.78-0.83
- Week 16: 0.80-0.85

**LCPO (Cross-Course)**:
- Week 4: 0.60-0.68
- Week 8: 0.65-0.73
- Week 12: 0.68-0.76
- Week 16: 0.70-0.78

**Future-Presentation (Temporal)**:
- Week 4: 0.62-0.70
- Week 8: 0.67-0.75
- Week 12: 0.70-0.78
- Week 16: 0.72-0.80

**Feature Groups (Week 8)**:
- Demographics Only: 0.64-0.68
- VLE Only: 0.70-0.74
- Assessment Only: 0.68-0.72
- Combined: 0.75-0.80

---

## Post-Execution Tasks

### 1. Generate Summary Report

```bash
cd notebooks
jupyter notebook OULAD_Consolidated_Analysis.ipynb
# Run all cells to generate visualizations and summary
```

### 2. Create Comparison Tables

```python
import pandas as pd

# Load all results
baseline = pd.read_csv('results/baseline/baseline_results_detailed.csv')
lcpo = pd.read_csv('results/lcpo/lcpo_results_detailed.csv')
future = pd.read_csv('results/cross_course/future_presentation_results.csv')

# Create comparison table
comparison = pd.DataFrame({
    'Evaluation': ['Baseline', 'LCPO', 'Future-Presentation'],
    'Mean_AUROC': [
        baseline['AUROC'].mean(),
        lcpo['AUROC'].mean(),
        future['AUROC'].mean()
    ],
    'Std_AUROC': [
        baseline['AUROC'].std(),
        lcpo['AUROC'].std(),
        future['AUROC'].std()
    ]
})

print(comparison)
comparison.to_csv('results/evaluation_comparison.csv', index=False)
```

### 3. Update Documentation

Update `docs/CROSS_COURSE_EVALUATION_REPORT.md` with actual results:
- Replace placeholder values with real metrics
- Add insights from actual data
- Update visualizations

---

## Execution Checklist

Use this checklist to track your progress:

### Pre-Execution
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Data files verified
- [ ] Configuration tested

### Execution
- [ ] Baseline evaluation completed
- [ ] LCPO evaluation completed
- [ ] Future-presentation evaluation completed
- [ ] Feature group comparison completed

### Verification
- [ ] All result CSVs generated
- [ ] Result files have expected row counts
- [ ] No error messages in logs
- [ ] Performance metrics in expected ranges

### Post-Execution
- [ ] Summary report generated
- [ ] Comparison tables created
- [ ] Visualizations saved
- [ ] Documentation updated

---

## Estimated Timeline

| Task | Duration | Cumulative |
|------|----------|------------|
| Environment setup | 15 min | 0:15 |
| Baseline evaluation | 2-3 hours | 2:15-3:15 |
| LCPO evaluation | 2-3 hours | 4:15-6:15 |
| Future-presentation | 1-2 hours | 5:15-8:15 |
| Feature group comparison | 2-3 hours | 7:15-11:15 |
| Verification & reporting | 1 hour | 8:15-12:15 |
| **Total** | **8-12 hours** | |

**Recommendation**: Run evaluations overnight or over a weekend.

---

## Next Steps After Execution

1. **Analyze Results**: Review performance metrics and identify patterns
2. **Update Reports**: Fill in actual results in documentation
3. **Communicate Findings**: Send follow-up to Dr. Singh with completed results
4. **Plan Phase 2**: Begin graph neural network implementation
5. **Prepare Presentation**: Create slides summarizing findings

---

## Support

If you encounter issues not covered in this guide:

1. Check error messages carefully
2. Review `docs/LEAKAGE_PREVENTION.md` for feature engineering issues
3. Consult `docs/EVALUATION_SPLITS.md` for evaluation strategy questions
4. Check `NOTEBOOK_ERROR_ANALYSIS.md` for common notebook errors

---

**Document Version**: 1.0  
**Last Updated**: June 20, 2026  
**Status**: Ready for Execution