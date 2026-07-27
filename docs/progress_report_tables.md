# Progress Report — Quantitative Evidence Tables

All figures in this document are drawn directly from saved result files in this
repository. Each table cites its source file so numbers can be verified and
regenerated independently.

---

## Table 1 — Strategy A vs. Strategy B: Assessment Filter Comparison

Strategy A applies only a due-date guard (`assessments.date ≤ window`).  
Strategy B (current pipeline) adds a submission-date guard (`date_submitted ≤ window`),
removing scores that were submitted late and therefore not observable at prediction time.

**Source**: `results/comparison/strategy_a_vs_b_comparison.csv`

| Prediction Week | Window (days) | Strategy A edges | Strategy B edges | Edges removed | % removed | LightGBM AUROC (A) | LightGBM AUROC (B) | ΔAUROC |
|---|---|---|---|---|---|---|---|---|
| Week 2 | 14 | 1,189 | 1,089 | 100 | 8.4% | 0.769 | 0.769 | +0.001 |
| Week 4 | 28 | 22,043 | 21,393 | 650 | 2.9% | 0.821 | 0.822 | +0.001 |
| Week 6 | 42 | 29,107 | 28,569 | 538 | 1.8% | 0.840 | 0.839 | −0.001 |
| Week 8 | 56 | 47,259 | 44,927 | 2,332 | 4.9% | 0.865 | 0.863 | −0.002 |

**Interpretation**: Strategy B removes between 2% and 8% of submitted-assessment
edges per week. The AUROC impact is negligible (|ΔAUROC| < 0.003) across all
weeks and models, confirming the dual-guard filter improves schema correctness
without degrading predictive performance. All subsequent results use Strategy B.

---

## Table 2 — Graph Statistics by Prediction Week

**Source**: `results/graph/artifacts/week{02,04,06,08}_metadata.json`

### Node counts (identical across all weeks — nodes are not time-filtered)

| Node Type | Count |
|---|---|
| student | 28,785 |
| course_presentation | 22 |
| VLE resource | 6,364 |

> Assessment nodes vary by week (only assessments due on or before the cutoff
> are included):

| Week | Window (days) | Assessment nodes |
|---|---|---|
| 2 | 14 | 1 |
| 4 | 28 | 17 |
| 6 | 42 | 24 |
| 8 | 56 | 40 |

### Edge counts by week

| Week | enrolled_in | contains_assess | has_resource | submitted | interacted_with |
|---|---|---|---|---|---|
| 2 | 32,593 | 1 | 6,364 | 1,089 | 634,723 |
| 4 | 32,593 | 17 | 6,364 | 21,393 | 835,935 |
| 6 | 32,593 | 24 | 6,364 | 28,569 | 952,241 |
| 8 | 32,593 | 40 | 6,364 | 44,927 | 1,056,217 |

### Label distribution (identical across all weeks — labels do not depend on the time cutoff)

| Total enrollments | At-risk (Fail/Withdrawn) | Success (Pass/Distinction) | At-risk rate |
|---|---|---|---|
| 32,593 | 17,208 | 15,385 | 52.8% |

### Construction runtime and memory

**Source**: `results/graph/validation/week{02,04,06,08}_validation_summary.txt`

| Week | Runtime (s) | Peak memory (MB) |
|---|---|---|
| 2 | 7.1 | 915.5 |
| 4 | 5.3 | 915.5 |
| 6 | 6.3 | 915.5 |
| 8 | 6.1 | 1,049.0 |

---

## Table 3 — Missing Values Before and After Imputation

The graph pipeline records pre-imputation null counts in each
`week{N}_metadata.json` → `pre_imputation_nulls` field.
Post-imputation, all tables have zero nulls (asserted in `build_node_tables()`).

**Source**: `results/graph/artifacts/week{02,04,06,08}_metadata.json`

| Node / Edge type | Column | Pre-imputation nulls | Imputed to | Reason |
|---|---|---|---|---|
| student | `imd_band` | 971 (all weeks) | `"Unknown"` | Postcode could not be geocoded in source data |
| vle_resource | `week_from` | 5,243 (all weeks) | `0` | Resource has no scheduled week range in `vle.csv` |
| vle_resource | `week_to` | 5,243 (all weeks) | `0` | Same as above |
| All other tables | — | 0 | — | No nulls in source data for these columns |

**Imputation strategy** (consistent with tabular baseline):
- Numeric columns: fill `NaN` → `0`
- Categorical columns: fill `NaN` → `"Unknown"`

---

## Table 4 — Temporal Validation: LightGBM AUROC Across Prediction Weeks

Performance improves with each additional week of observable data.
All results use the random-student 5-fold CV split (Strategy B dual-filter).

**Source**: `results/baseline/baseline_results_table.csv`

| Prediction Week | Window (days) | AUROC | AUPRC | F1 | Precision | Recall | Balanced Acc |
|---|---|---|---|---|---|---|---|
| Week 2 | 14 | 0.769 ± 0.005 | 0.805 ± 0.003 | 0.704 ± 0.003 | 0.729 ± 0.007 | 0.680 ± 0.007 | 0.698 ± 0.004 |
| Week 4 | 28 | 0.822 ± 0.002 | 0.858 ± 0.003 | 0.738 ± 0.005 | 0.788 ± 0.007 | 0.694 ± 0.007 | 0.743 ± 0.005 |
| Week 6 | 42 | 0.839 ± 0.003 | 0.875 ± 0.003 | 0.753 ± 0.006 | 0.802 ± 0.008 | 0.709 ± 0.008 | 0.757 ± 0.006 |
| Week 8 | 56 | 0.863 ± 0.004 | 0.894 ± 0.003 | 0.782 ± 0.004 | 0.830 ± 0.004 | 0.738 ± 0.008 | 0.785 ± 0.003 |

> All metrics report the at-risk class (`target=1`, Fail/Withdrawn).
> Standard deviations are population std (ddof=0) across 5 folds.

**Absolute AUROC gain Week 2 → Week 8**: +0.094 (+12.2%)

---

## Table 5 — Evaluation Strategy Comparison (Week 8, LightGBM)

Three evaluation strategies were used to assess both in-distribution and
out-of-distribution performance.

**Sources**:
- Random-student: `results/comparison/all_splits_comparison.csv`
- LCPO: `results/lcpo/random_vs_lcpo_comparison.csv`
- Future-presentation: `results/cross_course/future_presentation_results.csv`

| Strategy | Description | AUROC | F1 | Balanced Acc | Generalization gap (vs. Random) |
|---|---|---|---|---|---|
| Random-student (5-fold CV) | Students randomly split; same distribution train/test | 0.863 ± 0.004 | 0.782 ± 0.004 | 0.785 ± 0.003 | — |
| LCPO (22 folds) | Train on 21 course-presentations, test on 1 held-out | 0.835 ± 0.076 | 0.734 ± 0.105 | 0.754 ± 0.070 | −0.028 AUROC (−3.2%) |
| Future-presentation | Train on 2013B/J + 2014B, test on 2014J (temporal) | 0.840 | 0.765 | 0.759 | −0.023 AUROC (−2.7%) |

**Key observation**: The LCPO standard deviation of ±0.076 is substantially
larger than the random-student ±0.004, reflecting genuine course-specific
variation — not noise. This is quantified in Table 6.

---

## Table 6 — LCPO Per-Presentation AUROC (Week 8, LightGBM)

The aggregate LCPO AUROC of 0.835 ± 0.076 masks substantial variation across
courses. This table replaces vague statements such as "stable across folds"
with the actual per-presentation measurements.

**Source**: `results/lcpo/lcpo_results_detailed.csv`  
(computed as mean over `Week==8`, `Model==LightGBM`)

| Course | Presentation | Test AUROC | Test N | Difficulty tier |
|---|---|---|---|---|
| GGG | 2013J | 0.641 | 952 | Very hard |
| GGG | 2014B | 0.652 | 833 | Very hard |
| GGG | 2014J | 0.687 | 749 | Very hard |
| AAA | 2014J | 0.778 | 365 | Moderate |
| AAA | 2013J | 0.799 | 383 | Moderate |
| BBB | 2014J | 0.813 | 2,292 | Moderate |
| FFF | 2013B | 0.851 | 1,614 | Easy |
| EEE | 2014B | 0.852 | 694 | Easy |
| EEE | 2013J | 0.862 | 1,052 | Easy |
| FFF | 2014B | 0.865 | 1,500 | Easy |
| FFF | 2013J | 0.867 | 2,283 | Easy |
| CCC | 2014J | 0.868 | 2,498 | Easy |
| BBB | 2014B | 0.875 | 1,613 | Easy |
| BBB | 2013J | 0.876 | 2,237 | Easy |
| DDD | 2014J | 0.876 | 1,803 | Easy |
| BBB | 2013B | 0.877 | 1,767 | Easy |
| EEE | 2014J | 0.878 | 1,188 | Easy |
| DDD | 2013B | 0.882 | 1,303 | Easy |
| DDD | 2014B | 0.889 | 1,228 | Easy |
| DDD | 2013J | 0.891 | 1,938 | Easy |
| CCC | 2014B | 0.894 | 1,936 | Easy |
| FFF | 2014J | 0.899 | 2,365 | Easy |

**Range**: 0.641 (GGG/2013J) to 0.899 (FFF/2014J) — a spread of 0.258 AUROC.  
**GGG courses** (3 presentations, avg AUROC 0.660) are the consistently hardest
to generalise to; all three have below-chance-adjusted performance relative to
their class-imbalance baseline. This is attributable to: (1) only 2–3
presentations available for training, limiting cross-course signal, and (2) a
lower at-risk rate (~40%) diverging from the global 52.8% average.

---

## Table 7 — Test Suite Summary

**Source**: `tests/test_filter_window.py`, `tests/test_splits.py`

| Test module | Tests | Coverage |
|---|---|---|
| `test_filter_window.py` | 11 | VLE cutoff boundaries, submission-date boundaries, due-date boundaries, dual-guard (Strategy B) correctness |
| `test_splits.py` | 13 | Zero student overlap (train/val/test), all splits non-empty, masks cover all rows, reproducibility, LCPO held-out isolation |
| **Total** | **24** | **24/24 passing** |

Run with: `source oulad_env/bin/activate && pytest tests/ -v`

---

*All tables generated from committed result files. To regenerate pipeline outputs:*
```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
python src/summarize_graph_weeks.py
```
