# Evaluation Split Strategies for OULAD

## Overview

This document describes the three evaluation split strategies used to assess model performance
and generalization in the OULAD student success prediction task. All three strategies are
implemented in `src/evaluation_pipeline.py` and can be reproduced end-to-end by running:

```bash
python src/run_evaluation.py
```

## Label Convention

**All evaluations use the corrected label convention:**
- **1 = at-risk** (Fail/Withdrawn) — positive class, students needing intervention
- **0 = success** (Pass/Distinction) — negative class, students on track

Metrics (precision, recall, F1, AUPRC) refer to identifying at-risk students.

---

## Split Strategy 1: Random Student Split (5-fold GroupKFold CV)

### Description

5-fold cross-validation where splits are made at the **student level**, guaranteeing that the
same student never appears in both train and test within a fold.

### Implementation

```python
from evaluation_pipeline import run_random_student_evaluation
from oulad_data import create_datasets, load_oulad_data

student_info, student_vle, student_assess, assessments = load_oulad_data()
datasets = create_datasets(student_info, student_vle, student_assess, assessments)
random_df = run_random_student_evaluation(datasets)
```

The function uses `GroupKFold` on `id_student` — students are shuffled with `random_state=42`
and assigned to folds round-robin. This correctly prevents any student appearing in both train
and test within a fold.

### Characteristics

- **Split Unit**: Unique students (not enrollments)
- **Method**: 5-fold `GroupKFold` on `id_student`
- **Folds**: 5
- **Feature subsets evaluated**: VLE_only, Assessment_only, VLE+Assessment, All_features

### Results (Week 8, All Features)

| Model | AUROC | F1 | Precision | Recall |
|-------|-------|----|-----------|--------|
| LightGBM | 0.865±0.006 | 0.785±0.008 | 0.837±0.006 | 0.739±0.011 |
| XGBoost | 0.858±0.006 | — | — | — |
| RandomForest | 0.857±0.005 | — | — | — |

### Results Location

- Detailed (per fold): `results/baseline/baseline_results_detailed.csv`
- Summary (mean±std): `results/baseline/baseline_results_table.csv`

---

## Split Strategy 2: Leave-Course-Presentation-Out (LCPO)

### Description

For each of the 22 unique `(code_module, code_presentation)` pairs in OULAD, hold it out as
the test set and train on all remaining enrollments. Uses the canonical `lcpo_split()` from
`src/oulad_data.py`.

### Implementation

```python
from evaluation_pipeline import run_lcpo_evaluation

lcpo_df = run_lcpo_evaluation(datasets)
```

The function iterates all 22 course-presentations, calls `lcpo_split()` from `oulad_data.py`
for each, and evaluates all 4 models (LR, RF, XGBoost, LightGBM) with All_features.

### Characteristics

- **Split Unit**: Course-presentation combinations (e.g., AAA-2013B, BBB-2014J)
- **Number of Folds**: 22 (one per course-presentation)
- **Train Size**: ~21/22 of data per fold (~31,000 enrollments)
- **Test Size**: ~1/22 of data per fold (~350–2,500 enrollments)

### Course-Presentation Combinations

| Module | Presentations | Total |
|--------|---------------|-------|
| AAA | 2013J, 2014J | 2 |
| BBB | 2013B, 2013J, 2014B, 2014J | 4 |
| CCC | 2014B, 2014J | 2 |
| DDD | 2013B, 2013J, 2014B, 2014J | 4 |
| EEE | 2013J, 2014B, 2014J | 3 |
| FFF | 2013B, 2013J, 2014B, 2014J | 4 |
| GGG | 2013J, 2014B, 2014J | 3 |

### Results (Week 8, All Features)

| Model | AUROC | F1 | Balanced Acc |
|-------|-------|----|--------------|
| LightGBM | 0.838±0.077 | 0.737±0.103 | — |
| XGBoost | ~0.833 | — | — |

### Results Location

- Detailed (per course-presentation): `results/lcpo/lcpo_results_detailed.csv`
- Random vs LCPO comparison: `results/lcpo/random_vs_lcpo_comparison.csv`
- Course difficulty ranking: `results/lcpo/course_presentation_difficulty.csv`
- Course difficulty chart: `results/lcpo/course_difficulty_chart.png`

---

## Split Strategy 3: Future-Presentation Split

### Description

Temporal split where models are trained on earlier presentations (2013B, 2013J, 2014B) and
tested on the latest presentation (2014J), simulating deployment to future course offerings.

### Implementation

```python
from evaluation_pipeline import run_future_presentation_evaluation

future_df = run_future_presentation_evaluation(datasets)
```

Train presentations: `["2013B", "2013J", "2014B"]`  
Test presentation: `["2014J"]`

### Characteristics

- **Split Unit**: Presentation year/semester
- **Temporal Order**: Strictly enforced — no 2014J data in training
- **Train size** (Week 8): ~21,333 enrollments
- **Test size** (Week 8): ~11,260 enrollments

### Presentation Timeline

```
2013B (Feb 2013) ─┐
2013J (Oct 2013) ─┤ TRAIN
2014B (Feb 2014) ─┘
                   │
2014J (Oct 2014) ──  TEST
```

### Results (Week 8, All Features)

| Model | AUROC | F1 | Precision | Recall |
|-------|-------|----|-----------|--------|
| RandomForest | 0.846 | 0.771 | 0.768 | 0.775 |
| LightGBM | 0.836 | — | — | — |

### Results Location

- Full results: `results/cross_course/future_presentation_results.csv`

---

## Comparison of Split Strategies (Week 8, LightGBM)

| Aspect | Random-Student | LCPO | Future-Presentation |
|--------|---------------|------|---------------------|
| **AUROC** | 0.865±0.006 | 0.838±0.077 | 0.836 |
| **Realism** | Low | Medium | High |
| **Generalization Test** | Within-distribution | Cross-course | Temporal |
| **Variance** | Low | High (±0.077) | None (single eval) |
| **Computation** | Medium (5-fold CV) | Slow (22 folds) | Fast |
| **Use Case** | Baseline upper bound | New course offerings | Future cohorts |

---

## Course Difficulty Analysis

> **Two output files are produced** when `analyze_course_difficulty()` is called:
>
> | File | Description |
> |------|-------------|
> | `results/lcpo/course_presentation_difficulty.csv` | Aggregated: one row per course-presentation, AUROC mean±std across all weeks and models. Use for a quick overall hardness ranking. |
> | `results/lcpo/course_difficulty_by_week_model.csv` | Long-format: one row per `(Course_Presentation, Week, Model)` — up to 440 rows. Use for per-week or per-model analysis. |


LCPO reveals significant variation across course-presentations. AUROC is aggregated across
all 4 models and 4 prediction windows (16 measurements per course-presentation).

### Hardest to Generalize To (AUROC < 0.70)

| Course | AUROC mean | AUROC std |
|--------|-----------|-----------|
| GGG/2013J | 0.628 | 0.016 |
| GGG/2014B | 0.639 | 0.018 |
| GGG/2014J | 0.665 | 0.015 |

GGG consistently has the lowest generalization performance across all models and windows.
This suggests course-specific characteristics that don't transfer well from other modules.

### Easiest to Generalize To (AUROC > 0.84)

| Course | AUROC mean | AUROC std |
|--------|-----------|-----------|
| CCC/2014B | 0.850 | 0.047 |
| FFF/2014J | 0.844 | 0.035 |
| EEE/2014J | 0.836 | 0.041 |

Full ranking: `results/lcpo/course_presentation_difficulty.csv`  
Chart: `results/lcpo/course_difficulty_chart.png`

---

## Unified Comparison Table

A single CSV covering all 4 weeks × 5 models × 3 splits is saved at:

```
results/comparison/all_splits_comparison.csv
```

Columns: `Week, Model, Split, AUROC_mean, AUROC_std, F1_mean, F1_std, Precision_mean,
Precision_std, Recall_mean, Recall_std, Balanced_Acc_mean, Balanced_Acc_std`

---

## Result File Index

| File | Description |
|------|-------------|
| `results/baseline/baseline_results_detailed.csv` | Random split: all weeks × models × feature subsets × folds |
| `results/baseline/baseline_results_table.csv` | Random split summary (mean±std) |
| `results/lcpo/lcpo_results_detailed.csv` | LCPO: all weeks × models × course-presentations |
| `results/lcpo/random_vs_lcpo_comparison.csv` | Random vs LCPO per model (Week 8) |
| `results/lcpo/course_presentation_difficulty.csv` | Aggregated: per-course AUROC mean±std (all weeks × models), sorted hardest first |
| `results/lcpo/course_difficulty_by_week_model.csv` | Long-format: AUROC per `(Course_Presentation, Week, Model)` — 440 rows |
| `results/lcpo/course_difficulty_chart.png` | Boxplot of per-course AUROC distribution |
| `results/cross_course/future_presentation_results.csv` | Future-presentation: all weeks × models |
| `results/comparison/all_splits_comparison.csv` | Unified: all weeks × models × all 3 splits |
| `results/overall_summary.csv` | Top-level summary: best model per split for Week 8 |

---

## Reproducing All Results

```bash
# From the project root
python src/run_evaluation.py
```

This regenerates all result CSVs and the course difficulty chart from scratch. Runtime is
approximately 10–15 minutes on a standard laptop.

---

## Preprocessing (Common to All Splits)

All splits use the same shared utilities from `src/oulad_data.py`:

### Temporal Filtering

`filter_window(vle, assess, assessments, window)` filters:
- VLE interactions: `date <= window`
- Assessments — dual guard (Strategy B):
  - Guard 1: `assessments.date` (due date) `<= window`
  - Guard 2: `date_submitted <= window` — 28.8% of OULAD submissions are submitted
    after their due date; Guard 2 removes these to prevent future leakage

### Feature Engineering

`build_features(vle_w, assess_w, student_info)` produces one row per enrollment with:
- **VLE features**: `vle_total`, `vle_mean`, `vle_std` (aggregated sum_click)
- **Assessment features**: `assess_mean`, `assess_max`, `assess_count` (aggregated score)
- **Demographics**: gender, region, highest_education, imd_band, age_band,
  num_of_prev_attempts, disability
- Missing values: numeric → 0, categorical → "Unknown"

### Models

| Name | Class | Key Params |
|------|-------|-----------|
| Majority | DummyClassifier | strategy="most_frequent" |
| LogisticRegression | LogisticRegression | max_iter=1000, random_state=42 |
| RandomForest | RandomForestClassifier | n_estimators=100, n_jobs=-1, random_state=42 |
| XGBoost | XGBClassifier | n_estimators=100, eval_metric="logloss", random_state=42 |
| LightGBM | LGBMClassifier | n_estimators=100, verbose=-1, random_state=42 |

---

## Evaluation Metrics

All splits report the same 6 metrics (positive class = 1 = at-risk):

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve — overall discrimination |
| AUPRC | Area Under Precision-Recall Curve — imbalanced class performance |
| F1 | Harmonic mean of precision and recall |
| Precision | Fraction of at-risk predictions that are correct |
| Recall | Fraction of actual at-risk students identified |
| Balanced_Acc | Average of sensitivity and specificity |
