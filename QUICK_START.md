# OULAD Analysis - Quick Start Guide

**Last Updated**: June 20, 2026

This guide provides the exact commands to run all analysis scripts in the correct order.

---

## Prerequisites

### 1. Activate Virtual Environment
```bash
cd /Users/olivialoza/Documents/Development/OULAD
source oulad_env/bin/activate
```

### 2. Verify Data Exists
```bash
ls -lh data/raw/*.csv
```

You should see:
- studentInfo.csv
- studentVle.csv
- studentAssessment.csv
- assessments.csv
- vle.csv
- courses.csv

---

## Phase 1: Run Base Evaluations (6-8 hours total)

### Step 1: Baseline Evaluation (2-3 hours)
```bash
cd src
python baseline_evaluation.py
```

**What it does**: Random 5-fold CV across weeks 2, 4, 6, 8

**Outputs**:
- `baseline_results_detailed.csv`
- `baseline_results_table.csv`
- `baseline_results_plot.png`

---

### Step 2: LCPO Evaluation (2-3 hours)
```bash
cd src
python lcpo_evaluation.py
```

**What it does**: Leave-Course-Presentation-Out cross-validation

**Outputs**:
- `results/lcpo/lcpo_results_detailed.csv`
- `results/lcpo/lcpo_summary.csv`

---

### Step 3: Future-Presentation Evaluation (1-2 hours)
```bash
cd src
python future_presentation_evaluation.py
```

**What it does**: Temporal generalization testing

**Outputs**:
- `results/cross_course/future_presentation_results.csv`

---

## Phase 2: Run Advanced Analysis (4-6 hours total)

### Step 4: Per-Course LCPO Analysis (30 minutes)
```bash
cd src
python lcpo_course_analysis.py
```

**Prerequisites**: Step 2 must be complete

**Outputs**:
- `results/lcpo_analysis/course_difficulty_analysis.csv`
- `results/lcpo_analysis/model_consistency.csv`
- `results/lcpo_analysis/outlier_courses.csv`
- `results/lcpo_analysis/module_performance.csv`
- `results/lcpo_analysis/lcpo_course_analysis.png`
- `results/lcpo_analysis/lcpo_heatmap.png`
- `results/lcpo_analysis/lcpo_course_analysis_report.md`

---

### Step 5: Feature Importance Analysis (1-2 hours)
```bash
cd src
python feature_importance_analysis.py
```

**Optional**: For SHAP values, first install:
```bash
pip install shap
```

**Outputs**:
- `results/feature_importance/random_forest_importance.csv`
- `results/feature_importance/xgboost_importance.csv`
- `results/feature_importance/lightgbm_importance.csv`
- `results/feature_importance/permutation_importance.csv`
- `results/feature_importance/shap_importance.csv` (if SHAP installed)
- `results/feature_importance/feature_importance_comparison.png`
- `results/feature_importance/category_importance.png`
- `results/feature_importance/shap_summary.png` (if SHAP installed)
- `results/feature_importance/feature_importance_report.md`

---

### Step 6: Threshold Optimization (1-2 hours)
```bash
cd src
python threshold_optimization.py
```

**Outputs**:
- `results/threshold_optimization/random_forest_threshold_analysis.csv`
- `results/threshold_optimization/xgboost_threshold_analysis.csv`
- `results/threshold_optimization/lightgbm_threshold_analysis.csv`
- `results/threshold_optimization/random_forest_optimal_thresholds.csv`
- `results/threshold_optimization/xgboost_optimal_thresholds.csv`
- `results/threshold_optimization/lightgbm_optimal_thresholds.csv`
- `results/threshold_optimization/precision_recall_curves.png`
- `results/threshold_optimization/threshold_impact_analysis.png`
- `results/threshold_optimization/threshold_optimization_report.md`

---

### Step 7: GNN Training (2-3 hours)

**First, install PyTorch and PyTorch Geometric**:
```bash
# For CPU-only (faster to install)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric
```

**Then run**:
```bash
cd src
python gnn_model.py
```

**Outputs**:
- `results/gnn/best_model.pt`
- Console output with training progress

---

## Quick Commands (Copy-Paste)

### Run All Base Evaluations
```bash
cd /Users/olivialoza/Documents/Development/OULAD/src
python baseline_evaluation.py && \
python lcpo_evaluation.py && \
python future_presentation_evaluation.py
```

### Run All Advanced Analysis
```bash
cd /Users/olivialoza/Documents/Development/OULAD/src
python lcpo_course_analysis.py && \
python feature_importance_analysis.py && \
python threshold_optimization.py
```

### Run Everything (10-14 hours total)
```bash
cd /Users/olivialoza/Documents/Development/OULAD/src
python baseline_evaluation.py && \
python lcpo_evaluation.py && \
python future_presentation_evaluation.py && \
python lcpo_course_analysis.py && \
python feature_importance_analysis.py && \
python threshold_optimization.py
```

---

## Alternative: Use Jupyter Notebook

If you prefer interactive execution:

```bash
cd /Users/olivialoza/Documents/Development/OULAD
jupyter notebook notebooks/OULAD_Consolidated_Analysis.ipynb
```

Then run cells sequentially.

---

## Monitoring Progress

### Check if scripts are running
```bash
ps aux | grep python
```

### Monitor output files
```bash
# Watch for new result files
watch -n 5 'find results -name "*.csv" -mmin -5'
```

### Check disk space
```bash
df -h .
```

---

## Troubleshooting

### If script fails with "ModuleNotFoundError"
```bash
# Ensure you're in src directory
cd /Users/olivialoza/Documents/Development/OULAD/src

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/olivialoza/Documents/Development/OULAD/src"
```

### If script is too slow
Edit the script and reduce:
- Number of folds (5 → 3)
- Number of estimators (100 → 50)
- Number of weeks (test with just week 8)

### If memory error
Close other applications and try again, or process one week at a time.

---

## Verification

After running all scripts, verify outputs:

```bash
# Count result files
find results -name "*.csv" | wc -l

# List all result directories
ls -R results/

# Check file sizes
du -sh results/*
```

Expected:
- ~20-30 CSV files
- ~10-15 PNG visualizations
- ~5-7 markdown reports
- Total size: 50-200 MB

---

## Next Steps After Execution

1. **Review Results**:
   ```bash
   # Open reports in VS Code
   code results/lcpo_analysis/lcpo_course_analysis_report.md
   code results/feature_importance/feature_importance_report.md
   code results/threshold_optimization/threshold_optimization_report.md
   ```

2. **View Visualizations**:
   ```bash
   # Open images
   open results/lcpo_analysis/*.png
   open results/feature_importance/*.png
   open results/threshold_optimization/*.png
   ```

3. **Analyze CSVs**:
   ```python
   import pandas as pd
   
   # Load and explore results
   baseline = pd.read_csv('results/baseline/baseline_results_detailed.csv')
   lcpo = pd.read_csv('results/lcpo/lcpo_results_detailed.csv')
   
   print(baseline.groupby('model')['AUROC'].mean())
   print(lcpo.groupby('model')['AUROC'].mean())
   ```

---

## Estimated Timeline

| Phase | Duration | Can Run Overnight? |
|-------|----------|-------------------|
| Baseline Evaluation | 2-3 hours | Yes |
| LCPO Evaluation | 2-3 hours | Yes |
| Future-Presentation | 1-2 hours | Yes |
| LCPO Analysis | 30 min | No (quick) |
| Feature Importance | 1-2 hours | Yes |
| Threshold Optimization | 1-2 hours | Yes |
| GNN Training | 2-3 hours | Yes |
| **Total** | **10-14 hours** | **Yes** |

**Recommendation**: Start all base evaluations before leaving for the day, then run advanced analysis the next morning.

---

## Support

If you encounter issues:
1. Check `docs/EXECUTION_GUIDE.md` for detailed troubleshooting
2. Review error messages carefully
3. Verify data files exist and are readable
4. Ensure virtual environment is activated

---

**Ready to start? Copy and paste the commands above!**