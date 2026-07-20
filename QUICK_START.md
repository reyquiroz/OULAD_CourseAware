# OULAD Analysis — Quick Start Guide

**Last Updated**: July 2026

---

## Fresh Clone Setup

Steps to reproduce all results from a clean clone:

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd OULAD

# 2. Python 3.11.11 is pinned via .python-version — pyenv picks it up automatically
pyenv install 3.11.11   # skip if already installed

# 3. Create virtual environment and install dependencies
python -m venv oulad_env
source oulad_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch and PyTorch Geometric (CPU)
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1: replace URL with https://download.pytorch.org/whl/cu121

# 5. Download the OULAD dataset and place CSV files in data/raw/
#    Download URL: https://analyse.kmi.open.ac.uk/open_dataset
#
#    Required files (all 7 must be present):
#
#    File                      Size        Tracked in git?
#    ─────────────────────────────────────────────────────
#    studentInfo.csv           ~3 MB       ✓ yes
#    studentVle.csv            ~433 MB     ✗ NO — gitignored, MUST DOWNLOAD
#    studentAssessment.csv     ~6 MB       ✓ yes
#    assessments.csv           ~10 KB      ✓ yes
#    courses.csv               ~1 KB       ✓ yes
#    vle.csv                   ~500 KB     ✓ yes
#    studentRegistration.csv   ~1.5 MB     ✓ yes
#
#    Verify all 7 files are present:
python src/check_data.py

# 6. Run the complete evaluation pipeline (all 3 split strategies, ~10-15 min)
python src/run_evaluation.py

# 7. Build the Week 8 graph and run validation (~6 s)
python src/run_graph_pipeline.py --week 8

# 8. Open the canonical graph analysis notebook
jupyter lab notebooks/OULAD_Graph_Analysis_Final.ipynb
```

Validation report: `results/graph/validation/week08_validation_summary.txt`
Full validation details: `docs/validation_report_week8.md`

---

## Prerequisites

### 1. Activate Virtual Environment
```bash
source oulad_env/bin/activate
```

### 2. Verify Data Files
```bash
python src/check_data.py
```

Expected output — all 7 files marked ✓:
```
  ✓ studentInfo.csv              ~3 MB
  ✓ studentVle.csv               ~433 MB    (gitignored — download separately)
  ✓ studentAssessment.csv        ~6 MB
  ✓ assessments.csv              ~10 KB
  ✓ courses.csv                  ~1 KB
  ✓ vle.csv                      ~500 KB
  ✓ studentRegistration.csv      ~1.5 MB
```

---

## Run All Evaluations (single command)

```bash
# From project root — runs all 3 split strategies, saves all result CSVs
python src/run_evaluation.py
```

**Outputs** (all under `results/`):
- `baseline/baseline_results_detailed.csv` — random-student 5-fold CV
- `baseline/baseline_results_table.csv`
- `lcpo/lcpo_results_detailed.csv` — 22-fold LCPO
- `lcpo/random_vs_lcpo_comparison.csv`
- `lcpo/course_presentation_difficulty.csv`
- `lcpo/course_difficulty_chart.png`
- `cross_course/future_presentation_results.csv`
- `comparison/all_splits_comparison.csv` — unified 4 weeks × 5 models × 3 splits
- `overall_summary.csv`

---

## Build All-Week Graphs

Build graphs for all four prediction windows (14 / 28 / 42 / 56 days).
Each run takes ~5 s and ~1 GB peak memory.

```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
```

**What it does**: Runs all 7 pipeline stages (load → filter → nodes → edges →
enrollments → validate → persist) for the specified prediction window.
Assessment filtering uses the strictly leakage-free dual guard:
`due_date ≤ window` **AND** `date_submitted ≤ window` (Strategy B).

**Outputs per week** (gitignored — regenerate locally):
- `results/graph/artifacts/week{N}_*.parquet` (10 files)
- `results/graph/artifacts/week{N}_metadata.json`
- `results/graph/validation/week{N}_validation_summary.txt`
- `results/graph/validation/week{N}_integrity.json`

**After building all four weeks**, generate split definitions and the
multi-week summary:

```bash
python src/save_graph_splits.py        # saves per-week train/val/test + LCPO splits
python src/summarize_graph_weeks.py    # saves results/graph/validation/all_weeks_summary.csv
```

**Documentation**:
- Week 8 detailed audit: `docs/validation_report_week8.md`
- Multi-week comparison: `docs/graph_validation_summary.md`
- Schema reference: `docs/GRAPH_SCHEMA.md`

**Then re-execute the canonical notebook** to refresh all embedded outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/OULAD_Graph_Analysis_Final.ipynb
```

> GNN training (GraphSAGE) is planned for the next iteration.

---

## Advanced Analysis (optional)

### Feature Importance
```bash
python src/feature_importance_analysis.py
```

### Threshold Optimization
```bash
python src/threshold_optimization.py
```

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

### Step 7: Build All-Week Graphs (< 30 minutes total)

```bash
# From project root
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
python src/save_graph_splits.py
python src/summarize_graph_weeks.py
```

**What it does**: Builds the leakage-safe enrollment-centric graph for each of
the four prediction windows. Assessment filtering uses **dual guard (Strategy B)**:
`due_date ≤ window` AND `date_submitted ≤ window`.

**Outputs per week** (gitignored — regenerate locally):
- `results/graph/artifacts/week{N}_*.parquet` (10 files)
- `results/graph/artifacts/week{N}_metadata.json`
- `results/graph/validation/week{N}_validation_summary.txt`
- `results/graph/evaluation/week{N}/splits/` (4 split definition files)

**Summary outputs** (committed):
- `results/graph/validation/all_weeks_summary.csv`
- `docs/graph_validation_summary.md`

> GNN training (GraphSAGE) is planned for the following iteration.

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