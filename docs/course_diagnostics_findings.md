# Course-Level Diagnostic Findings

> **Status**: partial results (quick-run, 2 of 22 folds).  
> Placeholders marked **TBD** will be filled after the full 5-seed × 22-fold run (Sub-Task 1).

---

## Overview

`src/course_diagnostics.py` computes six explanatory factors per LCPO fold and
correlates each with the GNN AUROC mean across model seeds.  The goal is to
separate *model instability* (high `gnn_auroc_std`, cured by more seeds) from
*genuine course difficulty* (low `n_test`, extreme class imbalance, sparse VLE
data, or absent student overlap across presentations).

Diagnostic CSV: `results/graph/course_diagnostics.csv`  
Figure: `results/graph/figures/fig_course_diagnostics.png`

---

## Six Diagnostic Factors

| Factor | Column | Source |
|---|---|---|
| Sample size | `n_test` | `week08_lcpo_folds.csv` |
| Class prevalence (at-risk rate) | `at_risk_rate` | `week08_enrollments.parquet` |
| Assessment availability | `n_assessments` | `week08_edges_contains_assess.parquet` |
| VLE activity density | `mean_clicks_per_student` | `week08_edges_interacted_with.parquet` |
| Student overlap across presentations | `student_overlap_rate` | `week08_enrollments.parquet` |
| Distribution shift (optional) | `dist_shift` | `week08_nodes_student.parquet` (requires torch) |

---

## Partial Findings — Quick-Run (2 Folds)

The quick-run produced results for folds 0 and 1 (module **AAA**, presentations
2013J and 2014J).  Both are small (`n_test` ≈ 370–383), have moderate at-risk
rates, and share a high student overlap (AAA has only two presentations so any
student in one is likely in the other).

| held_out | n_test | at_risk_rate | n_assessments | mean_clicks | gnn_auroc_mean | lgbm_auroc | auroc_delta |
|---|---|---|---|---|---|---|---|
| AAA 2013J | 383 | 0.274 | 2 | 746 | 0.671 | 0.802 | −0.131 |
| AAA 2014J | 365 | 0.307 | 2 | 813 | 0.734 | 0.775 | −0.041 |

Both AAA folds show **negative** `auroc_delta`: LightGBM outperforms the GNN on
these small presentations (n ≈ 370).  Note that `gnn_auroc_std = 0.0` because
the quick-run used only 1 model seed per fold — instability cannot be assessed
until the full 5-seed run completes.

---

## Correlation Analysis — TBD After Full Run

**Which factor shows the strongest correlation with GNN AUROC?**

> TBD after full run.  Hypothesis: `n_test` (sample size) will show the
> strongest positive correlation — small folds (AAA, CCC) are expected to have
> higher AUROC variance and potentially lower mean performance.

**Does model instability or course difficulty better predict low-AUROC folds?**

> TBD after full run.  Discriminating criterion: if `gnn_auroc_std` (seed
> variance) is high in the same folds where `n_test` is low, instability and
> difficulty are confounded and cannot be separated from quick-run data alone.
> The full 5-seed run is required.

**Pearson r values (all factors vs. `gnn_auroc_mean`):**

| Factor | r | direction |
|---|---|---|
| `n_test` | TBD | — |
| `at_risk_rate` | TBD | — |
| `n_assessments` | TBD | — |
| `mean_clicks_per_student` | TBD | — |
| `student_overlap_rate` | TBD | — |
| `dist_shift` | TBD | — |

---

## Notable Outlier Courses — TBD After Full Run

> TBD.  Candidates based on prior course variation results:
> - Folds where `auroc_delta` (GNN − LightGBM) is most negative (LightGBM wins
>   by the largest margin) — these are the folds where the graph structure
>   provides no advantage and course-design features may explain why.
> - Folds where `gnn_auroc_std` across seeds exceeds 0.05 — flagged as
>   *unstable* rather than intrinsically difficult.

---

## Interpretation Framework

Three disjoint explanations for low GNN AUROC on a given fold:

1. **Model instability** — `gnn_auroc_std` is high; more seeds or a longer
   training schedule would improve the mean.  Does not require a different model
   architecture.
2. **Genuine course difficulty** — small `n_test`, extreme `at_risk_rate`
   (very high or very low), or few assessments.  The signal is weak for *any*
   model and the fold should be flagged in the reporting narrative.
3. **Distribution shift** — held-out students are demographically distinct from
   training students (`dist_shift` high).  May indicate that the graph
   representation over-fits to a specific demographic profile present only in
   the training modules.

---

## Next Steps

1. Run the full 5-seed × 22-fold LCPO experiment (Sub-Task 1).
2. Re-run `python src/course_diagnostics.py --week 8` to populate all 22 rows.
3. Inspect `results/graph/figures/fig_course_diagnostics.png` for the 2×3
   scatter grid; note panels where the trendline slope is steepest.
4. Fill in the "TBD" placeholders in this document.
5. Add a subsection to `docs/summer_report_draft.md` Section 5 citing these
   findings (Sub-Task 3).
