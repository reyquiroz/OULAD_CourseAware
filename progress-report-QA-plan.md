# Progress Report Q&A Plan
## Latest Developments — OULAD GraphSAGE At-Risk Prediction

---

## A. What Have I Done?

### 1. Filled the full AUROC results matrix
Populated `AUROC_Results_Matrix_by_Dr_Guo.xlsx` (6 sheets: AUROC, AUPRC, F1, Balanced Accuracy, Precision, Recall) with **94 verified cells** drawn directly from CSV experiment files. No estimates — every value is traceable to a source file.

**Models covered:**
- Majority classifier, Logistic Regression, Random Forest, XGBoost, LightGBM (all features, matched protocol, VLE-only, assessment-only, VLE+assessment)
- MLP — all enrollment features (matched protocol, 5 seeds)
- GraphSAGE base (weighted + unweighted), + interaction attributes
- GraphSAGE architectural variants: enrollment summaries, enrollment-node, edge-aware message passing, temporal behavior features
- GraphSAGE component ablations: −edge attributes, −temporal, −course features, −VLE, −assessment
- Zenodo external validation (random + LCPO)

**Splits and weeks covered:**
- Random-Student: weeks 2/4/6/8 for tabular, GNN base, MLP
- LCPO: weeks 2/4/6/8 for tabular and MLP; all 4 weeks for GNN base (new)
- Future-Presentation: weeks 2/4/6/8 for tabular models
- Zenodo external: week 8 (random + LCPO)

---

### 2. Extended GNN LCPO to all four prediction windows
Previously LCPO was only evaluated at week 8. Ran the full 22-fold × 5-seed LCPO experiment for **weeks 2, 4, and 6**, giving a complete multi-window generalisation picture for the GNN.

**Command used:**
```bash
python src/run_gnn_experiment.py --weeks 2 4 6 --lcpo-only
```

---

### 3. Ran the MLP matched-protocol baseline
Implemented and ran `MLPClassifier(128, 64)` on the same feature set, splits, seeds, and temporal guard as the GNN, for all 4 weeks and both random-student and LCPO splits (5 seeds each). Results written to `results/matched/mlp_matched_results.csv`.

**Command used:**
```bash
python src/run_matched_comparison.py --models mlp --weeks 2 4 6 8 --seeds 42 123 7 17 99 --splits random lcpo
```

---

### 4. Ran four GNN architectural variants
Evaluated four new GNN conditions at week 8 (random-student, 5 seeds each), all previously unrun:

| Condition flag | Description |
|---|---|
| `with_enrollment_summaries` | Adds 6 VLE behavioral summary features to enrollment edges |
| `enrollment_node` | Introduces an explicit enrollment node in the graph |
| `edge_aware_mp` | Edge-weighted message passing on interaction edges |
| `temporal_features` | Time-ordered VLE interaction encoding |

---

### 5. Fixed two bugs in `run_gnn_experiment.py`

**Bug 1 — `lcpo_summary.csv` was overwritten on each run:**
```python
# Before (line 652–653): overwrote existing data
summary_df.to_csv(summary_path, index=False)

# After: appends and deduplicates
_append_or_create_csv(summary_df, summary_path, dedup_keys=["week", "fold_idx"])
```

**Bug 2 — `random_student_results.csv` dedup key missing `condition`:**
```python
# Before: different conditions silently overwrote each other
dedup_keys=["week", "seed", "loss_weighting"]

# After: each condition is kept separately
dedup_keys=["week", "seed", "loss_weighting", "condition"]
```
The second bug caused `with_enrollment_summaries` and `enrollment_node` to be silently dropped on the first run. Both conditions had to be re-run after the fix was applied.

---

### 6. Updated the technical progress report
Rewrote `graphsage-at-risk-prediction-progress-report.html` with all new results, a new GNN Variants section, updated KPIs (best GNN now 0.888), MLP integrated into all comparison tables, and revised key findings.

---

## B. What Results Do I Obtain?

### GNN Architectural Variants — Week 8, Random-Student (AUROC, 5 seeds, weighted)

| Model | AUROC | F1 | Bal. Acc. |
|---|---|---|---|
| GraphSAGE + temporal features | **0.888 ± 0.002** | 0.811 | 0.797 |
| GraphSAGE + enrollment summaries | 0.884 ± 0.001 | 0.807 | 0.793 |
| GraphSAGE — edge-aware MP | 0.884 ± 0.002 | 0.807 | 0.792 |
| GraphSAGE — enrollment-node | 0.883 ± 0.001 | 0.802 | 0.794 |
| GraphSAGE + interaction attrs | 0.846 ± 0.003 | 0.769 | 0.746 |
| **GraphSAGE base (weighted)** | 0.838 ± 0.001 | 0.760 | 0.751 |
| *LightGBM (best tabular)* | *0.863 ± 0.004* | *0.782* | *0.785* |
| *MLP (matched protocol)* | *0.857 ± 0.004* | *0.782* | *0.759* |

**Key result:** Four of the five GNN variants exceed the best tabular model (LightGBM 0.863). This is the first point in the project where GNN architecture clearly surpasses the tabular ceiling.

---

### MLP vs. LightGBM — All Weeks (AUROC)

| Split | Model | W2 | W4 | W6 | W8 |
|---|---|---|---|---|---|
| Random-Student | LightGBM | 0.769 | 0.822 | 0.839 | 0.863 |
| Random-Student | MLP | 0.749 | 0.810 | 0.828 | 0.857 |
| LCPO | LightGBM | 0.752 | 0.797 | 0.813 | 0.835 |
| LCPO | MLP | 0.734 | 0.781 | 0.803 | 0.829 |

**Key result:** MLP trails LightGBM by only 0.006–0.014 AUROC across all protocols and windows, confirming feature engineering is the dominant driver of tabular performance.

---

### GNN LCPO — All Prediction Windows (AUROC, macro-mean ± std, 22 folds)

| Week | GNN (weighted) | LightGBM | MLP |
|---|---|---|---|
| 2 | 0.722 ± 0.043 | 0.752 ± 0.052 | 0.734 ± 0.048 |
| 4 | 0.751 ± 0.054 | 0.797 ± 0.066 | 0.781 ± 0.068 |
| 6 | 0.768 ± 0.055 | 0.813 ± 0.076 | 0.803 ± 0.080 |
| 8 | 0.788 ± 0.063 | 0.835 ± 0.076 | 0.829 ± 0.084 |

**Key result:** Under LCPO, the GNN consistently trails both tabular models at every window by 0.030–0.046 AUROC. The graph structure provides no LCPO advantage — tabular models generalise more cleanly to unseen course–presentation pairs.

---

### GNN Ablations — Week 8 (AUROC, 5 seeds, weighted)

| Condition | AUROC | Δ vs base |
|---|---|---|
| Base | 0.838 ± 0.001 | — |
| − Edge attributes | 0.835 ± 0.005 | −0.003 |
| − Temporal features | 0.830 ± 0.004 | −0.008 |
| − Course features | 0.830 ± 0.002 | −0.008 |
| − VLE features | 0.824 ± 0.004 | −0.014 |
| − Assessment features | 0.809 ± 0.008 | −0.029 |

Assessment features are the most critical component (−0.029); VLE is second (−0.014).

---

### Zenodo External Validation — Week 8

| Protocol | GNN AUROC |
|---|---|
| Random-Student (5 seeds) | 0.554 ± 0.033 |
| LCPO (6 folds) | 0.499 ± 0.013 |

Near-random performance. The model has not learned features that transfer across institutional datasets.

---

## C. What Problems Have I Encountered?

### Problem 1 — `lcpo_summary.csv` silently overwritten on multi-week runs
**What happened:** The function `run_lcpo_experiment()` wrote the fold summary with `summary_df.to_csv(path, index=False)` — a hard overwrite. When `--weeks 2 4 6` was run, each subsequent week erased the previous week's summary rows. After week 6 completed, only week 6 data survived; weeks 2 and 4 were lost.

**How it was discovered:** After running `--weeks 2 4 6`, `lcpo_summary.csv` showed only 22 rows (week 6) instead of 66.

**Fix:** Changed to `_append_or_create_csv(summary_df, summary_path, dedup_keys=["week", "fold_idx"])`, which appends new weeks and deduplicates on `(week, fold_idx)` so re-runs are safe.

---

### Problem 2 — GNN condition variants silently dropped by dedup
**What happened:** `_append_or_create_csv` for `random_student_results.csv` used `dedup_keys=["week", "seed", "loss_weighting"]`. When a new condition (e.g. `with_enrollment_summaries`) was run for the same week/seed/weighting as an existing row (`full` condition), the new row was treated as a duplicate and dropped — keeping the old `full` row. The conditions appeared to run successfully (no error) but produced no new CSV rows.

**How it was discovered:** After the `with_enrollment_summaries` and `enrollment_node` runs completed (exit 0), checking the CSV showed those conditions were absent; only `edge_aware_mp` (which ran last, after the old `full` rows were already keyed differently) survived.

**Fix:** Added `"condition"` to the dedup key: `dedup_keys=["week", "seed", "loss_weighting", "condition"]`. Both conditions were re-run after the fix.

---

### Problem 3 — Arch variants run as a chained shell command caused ordering issues
**What happened:** The four conditions were run as a single chained background command (`cmd1 && cmd2 && cmd3 && cmd4`). The dedup bug meant the first two conditions wrote rows that were immediately overwritten by the third and fourth. By the time the bug was found, the chain had already completed, requiring targeted re-runs.

**How it was discovered:** Only `edge_aware_mp` and `temporal_features` appeared in the CSV after the full chain finished.

**Resolution:** After fixing the dedup key, `with_enrollment_summaries` and `enrollment_node` were re-run individually as separate commands.

---

### Problem 4 — Long compute times for full LCPO runs
**What happened:** Each LCPO fold with 5 seeds at early prediction windows (weeks 2/4/6) took ~7 minutes on the local MacBook (vs. ~3 minutes for week 8). 22 folds × 5 seeds × 3 weeks ≈ 9+ hours of wall-clock time per week in practice.

**Impact:** The LCPO multi-week job ran for approximately 24 hours total. No data was lost, but iteration speed is a constraint for future LCPO experiments.

**Mitigation:** Future multi-week LCPO runs should be submitted to the TACC LS6 HPC cluster using the existing `scripts/ls6_gnn_lcpo.sh` job scripts, which support parallelisation across folds.

---

### Problem 5 — MLP behavioral summaries variant (row 17) not implemented
**What happened:** `src/update_results_matrix.py` references a `MLP_behavioral_summaries` model for row 17, and the fill plan listed it as fillable. However, `run_matched_comparison.py` only implements `MLP_all_features` — the behavioral summaries variant was never coded.

**Impact:** Row 17 of the matrix remains empty (`—`).

**Resolution options:**
1. Implement the 6-feature behavioral summary MLP in `run_matched_comparison.py` and run it.
2. Accept row 17 as permanently `—` if the variant is out of scope.
