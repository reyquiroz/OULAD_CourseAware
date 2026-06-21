# OULAD Baseline Analysis - Implementation Guide

## Overview
This guide provides instructions for running the enhanced OULAD baseline analysis that addresses all requirements from the research plan.

## Environment Setup

### 1. Activate Virtual Environment
```bash
source oulad_env/bin/activate
```

### 2. Verify Installation
```bash
python -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)"
```

## Running the Analysis

### Option A: Run Python Script
```bash
python OULAD_baseline_analysis_v5.py
```

This will:
- Load OULAD data from `DATA/` directory
- Create feature sets for weeks 2, 4, 6, 8
- Evaluate 5 models (Majority, LogisticRegression, RandomForest, XGBoost, LightGBM)
- Test 4 feature subsets (VLE-only, Assessment-only, VLE+Assessment, All features)
- Generate results with 5-fold cross-validation
- Save outputs:
  - `baseline_results_detailed.csv` - Full results with all metrics
  - `baseline_results_table.csv` - Formatted summary table
  - `baseline_results_plot.png` - Visualization of all metrics

### Option B: Convert to Jupyter Notebook
```bash
# Install jupytext if needed
pip install jupytext

# Convert Python script to notebook
jupytext --to notebook OULAD_baseline_analysis_v5.py

# Launch Jupyter
jupyter notebook OULAD_baseline_analysis_v5.ipynb
```

## What's Implemented

### ✅ Completed Features

1. **Baseline Results Table (Weeks 2, 4, 6, 8)**
   - All required metrics: AUROC, AUPRC, F1, Precision, Recall, Balanced Accuracy
   - 5-fold stratified cross-validation
   - Mean ± standard deviation for each metric

2. **Enhanced Model Suite**
   - Majority class baseline
   - Logistic Regression (max_iter=1000)
   - Random Forest (100 trees)
   - XGBoost (100 estimators)
   - **LightGBM** (100 estimators) ← NEW

3. **Reference Baselines**
   - VLE-only features
   - Assessment-only features
   - VLE + Assessment combined
   - All features (full model)

4. **Comprehensive Evaluation**
   - Cross-validation for robust estimates
   - Confidence intervals (std)
   - Feature subset comparison

### 🔄 Next Steps (Not Yet Implemented)

5. **Leave-Course-Presentation-Out (LCPO) Evaluation**
   - Requires additional script (see below)

6. **Course-Level Performance Analysis**
   - Per-course metrics
   - Difficulty analysis

7. **Underfitting Diagnosis Solutions**
   - Feature scaling
   - Hyperparameter tuning
   - Feature engineering for early windows

## Expected Output

### Console Output
```
================================================================================
OULAD BASELINE ANALYSIS - ENHANCED VERSION
================================================================================
Loading OULAD data...
Loaded 32593 students
Target distribution:
1    17208
0    15385
Name: target, dtype: int64

Building features for week 2...
  Week 2: 28081 samples, 19 features
...

============================================================
EVALUATING WEEK 2
============================================================

Majority:
  All_features: AUROC=0.500±0.000, F1=0.000±0.000

LogisticRegression:
  VLE_only: AUROC=0.XXX±0.XXX, F1=0.XXX±0.XXX
  Assessment_only: AUROC=0.XXX±0.XXX, F1=0.XXX±0.XXX
  ...
```

### Output Files

1. **baseline_results_detailed.csv**
   - Columns: Week, Model, Features, N_features, AUROC_mean, AUROC_std, AUPRC_mean, ...
   - ~80 rows (5 models × 4 feature subsets × 4 weeks)

2. **baseline_results_table.csv**
   - Formatted table with mean±std for each metric
   - Focus on "All_features" results
   - Ready for publication/reporting

3. **baseline_results_plot.png**
   - 2×3 grid showing all 6 metrics
   - Bar charts comparing models across weeks
   - High-resolution (300 DPI)

## Troubleshooting

### Issue: "FileNotFoundError: DATA/studentInfo.csv"
**Solution**: Ensure DATA directory exists with all CSV files:
```bash
ls DATA/
# Should show: studentInfo.csv, studentVle.csv, studentAssessment.csv, assessments.csv
```

### Issue: "ModuleNotFoundError: No module named 'lightgbm'"
**Solution**: Reinstall in virtual environment:
```bash
source oulad_env/bin/activate
pip install lightgbm>=4.0.0
```

### Issue: Script runs slowly
**Expected**: Full evaluation takes 10-30 minutes depending on hardware
- 4 weeks × 5 models × 4 feature subsets × 5 CV folds = 400 model fits

## Next Implementation Phase

To implement LCPO evaluation and course-level analysis, run:
```bash
python OULAD_lcpo_evaluation.py  # (to be created)
```

## GitHub Integration

Once results are generated:
```bash
git add baseline_results_*.csv baseline_results_plot.png
git add OULAD_baseline_analysis_v5.py
git add OULAD_IMPLEMENTATION_GUIDE.md
git commit -m "Add comprehensive OULAD baseline analysis with LightGBM"
git push origin main
```

Repository: https://github.com/BioAI-Systems-Lab/CourseAware

## Contact

For questions or issues, refer to the main project documentation or contact the research team.