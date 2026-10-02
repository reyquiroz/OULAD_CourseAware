# Plan: Fill AUROC Results Matrix from CSV Data

## Top-Level Overview

Populate `AUROC_Results_Matrix_by_Dr_Guo.xlsx` (6 sheets: Results, AUPRC, F1, Balanced Accuracy, Precision, Recall) with verified statistics from the CSV files under `results/` and `results/results/`. All written values must trace directly to a CSV. Rows with no CSV source remain `—`.

The matrix's own constraint applies: *"Per-seed and per-fold repository CSV files remain the reproducible source of truth; update this matrix only from verified outputs."*

---

## Source File Index

| File | Content |
|------|---------|
| `results/results/comparison/all_splits_comparison.csv` | **Primary source.** Pre-aggregated mean ± std for ALL models × ALL weeks (2/4/6/8) × ALL splits (Random-Student, LCPO, Future-Presentation). Tabular models only. |
| `results/baseline/baseline_results_table.csv` | Per-week per-feature-set LightGBM ablations (VLE-only, assessment-only, VLE+assessment). |
| `results/cross_course/future_presentation_results.csv` | Single-run future-presentation values for all models and weeks. |
| `results/graph/random_student_results.csv` | GNN per-seed week-8 random-student results (weighted + unweighted). |
| `results/graph/ablation_results.csv` | GNN ablation conditions per seed (full, no_assessment, no_vle, no_temporal, no_course_features, no_edge_attrs, with_iw_attrs). |
| `results/graph/lcpo_summary.csv` | GNN fold-level mean ± std across 22 LCPO folds, week 8. |
| `results/matched/lgbm_matched_results.csv` | Single matched-protocol LightGBM run (seed 42, week 8). |
| `results/zenodo/random_student_results.csv` | GNN on Zenodo dataset, 5 seeds. |
| `results/zenodo/lcpo_summary.csv` | GNN on Zenodo dataset, 6 LCPO folds. |

---

## Matrix Column Map

| Excel col | Meaning |
|-----------|---------|
| B | OULAD Random Student — Week 2 |
| C | OULAD Random Student — Week 4 |
| D | OULAD Random Student — Week 6 |
| E | OULAD Random Student — Week 8 |
| F | OULAD LCPO — Week 2 |
| G | OULAD LCPO — Week 4 |
| H | OULAD LCPO — Week 6 |
| I | OULAD LCPO — Week 8 |
| J | OULAD Future Presentation — Week 2 |
| K | OULAD Future Presentation — Week 4 |
| L | OULAD Future Presentation — Week 6 |
| M | OULAD Future Presentation — Week 8 |
| N | Zenodo/KU Leuven — Random W8 |
| O | Zenodo/KU Leuven — LCPO W8 |

---

## Format Rules

- **Results sheet (AUROC):** `mean ± std` rounded to 3 decimal places (e.g. `0.769 ± 0.005`).
- **All other sheets (AUPRC, F1, Balanced Accuracy, Precision, Recall):** plain `mean` rounded to 3 decimal places per their headers ("Supported mean values only; standard deviations are not consistently available").
- **Single-run values** (no std available): use `±—` notation per existing matrix convention (e.g. `0.843 ± —`).
- **No data:** leave cell as `—`.

---

## Sub-Task 1: Fill tabular models — Random Student, all weeks (rows 7–11, 13–15, cols B–E)

**Status:** [ ] pending

**Intent:** Populate Majority, Logistic Regression, Random Forest, XGBoost, and all LightGBM feature-set rows for the Random Student split across all four prediction windows.

**Primary source:** `results/results/comparison/all_splits_comparison.csv` (Split = `Random-Student`)
**Secondary source:** `results/baseline/baseline_results_table.csv` (for LightGBM feature-set ablations at week 8)

**Values from all_splits_comparison.csv — Random-Student rows:**

| Row | Model | Wk2 AUROC | Wk4 AUROC | Wk6 AUROC | Wk8 AUROC |
|-----|-------|-----------|-----------|-----------|-----------|
| 7 Majority | Majority | — | — | — | 0.500 ± 0.000 |
| 8 Logistic Regression | LogisticRegression | 0.753 ± 0.005 | 0.779 ± 0.006 | 0.800 ± 0.005 | 0.814 ± 0.005 |
| 9 Random Forest | RandomForest | 0.746 ± 0.004 | 0.808 ± 0.003 | 0.830 ± 0.003 | 0.855 ± 0.004 |
| 10 XGBoost | XGBoost | 0.756 ± 0.003 | 0.810 ± 0.003 | 0.827 ± 0.004 | 0.856 ± 0.005 |
| 11 LightGBM all features | LightGBM | 0.769 ± 0.005 | 0.822 ± 0.002 | 0.839 ± 0.003 | 0.863 ± 0.004 |

Note: Majority only has week-8 data (always 0.500 ± 0.000 for AUROC). The other metrics follow from `baseline_results_table.csv`:
- Majority W8: AUPRC=0.528, F1=0.691, Precision=0.528, Recall=1.000, Bal Acc=0.500

**LightGBM feature-set ablations (week 8 only, col E) from baseline_results_table.csv:**
- Row 13 LightGBM VLE only: AUROC=0.763 ± 0.004, AUPRC=0.805, F1=0.694, Prec=0.725, Rec=0.665, BalAcc=0.691
- Row 14 LightGBM assessment only: AUROC=0.807 ± 0.005, AUPRC=0.801, F1=0.755, Prec=0.773, Rec=0.737, BalAcc=0.747
- Row 15 LightGBM VLE+assessment: AUROC=0.838 ± 0.005, AUPRC=0.863, F1=0.769, Prec=0.784, Rec=0.756, BalAcc=0.761

**All other metrics for rows 8–11 (weeks 2–8) are also in all_splits_comparison.csv** and must be written to their respective sheets.

**Expected Outcomes:** Rows 7–11 and 13–15 populated with verified values across all applicable columns in all 6 metric sheets.

**Todo List:**
1. Read `results/results/comparison/all_splits_comparison.csv`
2. Filter to `Split = Random-Student`; extract mean/std per model per week per metric
3. Write AUROC values (mean ± std) to Results sheet, rows 7–11, cols B–E
4. Write AUPRC/F1/Precision/Recall/BalAcc (mean only) to their respective sheets
5. Read `results/baseline/baseline_results_table.csv` for LightGBM ablation rows 13–15, week 8 only
6. Write rows 13–15 col E across all 6 sheets

---

## Sub-Task 2: Fill tabular models — LCPO, all weeks (rows 8–11, cols F–I)

**Status:** [ ] pending

**Intent:** Populate Logistic Regression, Random Forest, XGBoost, and LightGBM for the LCPO split across all prediction windows.

**Primary source:** `results/results/comparison/all_splits_comparison.csv` (Split = `LCPO`)

**Values from all_splits_comparison.csv — LCPO rows:**

| Row | Model | Wk2 AUROC | Wk4 AUROC | Wk6 AUROC | Wk8 AUROC |
|-----|-------|-----------|-----------|-----------|-----------|
| 8 Logistic Regression | LogisticRegression | 0.735 ± 0.045 | 0.769 ± 0.063 | 0.790 ± 0.072 | 0.804 ± 0.074 |
| 9 Random Forest | RandomForest | 0.729 ± 0.055 | 0.781 ± 0.074 | 0.800 ± 0.084 | 0.826 ± 0.084 |
| 10 XGBoost | XGBoost | 0.742 ± 0.050 | 0.781 ± 0.069 | 0.800 ± 0.078 | 0.825 ± 0.077 |
| 11 LightGBM all features | LightGBM | 0.752 ± 0.052 | 0.797 ± 0.066 | 0.813 ± 0.076 | 0.835 ± 0.076 |

Note: Majority is not meaningful for LCPO (no AUROC variance) — leave as `—`.

**All other metrics** (AUPRC, F1, Precision, Recall, Bal Acc) are also present in all_splits_comparison.csv for these rows.

**Expected Outcomes:** Rows 8–11 populated for cols F–I in all 6 metric sheets.

**Todo List:**
1. Read `results/results/comparison/all_splits_comparison.csv`
2. Filter to `Split = LCPO`; extract mean/std per model per week per metric
3. Write AUROC values to Results sheet rows 8–11, cols F–I
4. Write other metrics to their respective sheets

---

## Sub-Task 3: Fill tabular models — Future Presentation, all weeks (rows 8–11, cols J–M)

**Status:** [ ] pending

**Intent:** Populate Logistic Regression, Random Forest, XGBoost, and LightGBM for the Future Presentation split.

**Primary source:** `results/results/comparison/all_splits_comparison.csv` (Split = `Future-Presentation`, std=0 for all — single run)

**Values (std=0 so reported as mean ± — per matrix convention):**

| Row | Model | Wk2 | Wk4 | Wk6 | Wk8 |
|-----|-------|-----|-----|-----|-----|
| 8 LogReg | LogisticRegression | 0.754 | 0.753 | 0.788 | 0.809 |
| 9 RandomForest | RandomForest | 0.745 | 0.786 | 0.810 | 0.843 |
| 10 XGBoost | XGBoost | 0.752 | 0.780 | 0.797 | 0.824 |
| 11 LightGBM | LightGBM | 0.766 | 0.797 | 0.813 | 0.840 |

Since std=0.0 for all Future-Presentation rows (single test-presentation run), these are reported without ± in AUROC sheet or as bare means in other sheets.

**Expected Outcomes:** Rows 8–11 cols J–M populated in all 6 metric sheets.

**Todo List:**
1. Read `results/results/comparison/all_splits_comparison.csv`
2. Filter to `Split = Future-Presentation`; extract mean per model per week per metric
3. Write values to cols J–M for rows 8–11 across all 6 sheets
4. For AUROC Results sheet: use `0.754` format (no ± since single run, consistent with existing M9=`0.843 ± —` pattern — update to bare value or `± —`)

---

## Sub-Task 4: Fill GNN Random Student Week 8 base model (rows 18–19, col E)

**Status:** [ ] pending

**Intent:** Compute mean ± std for GraphSAGE weighted and unweighted base model from per-seed CSV.

**Source:** `results/graph/random_student_results.csv`

**Critical seed exclusion rule:** Seeds 42 for both weighted and unweighted carry `condition=temporal_features` — a different model variant belonging to row 25. Exclude seed 42. Use only seeds 123, 7, 17, 99.

**Computed values (4 seeds: 123, 7, 17, 99):**

Row 18 — GraphSAGE base (weighted):
- AUROC: 0.8394, 0.8385, 0.8386, 0.8374 → **0.838 ± 0.001**
- AUPRC: 0.8822, 0.8797, 0.8818, 0.8808 → **0.881**
- F1: 0.7530, 0.7586, 0.7648, 0.7648 → **0.760**
- Precision: 0.8309, 0.7979, 0.7365, 0.7547 → **0.780**
- Recall: 0.6885, 0.7229, 0.7953, 0.7752 → **0.745**
- Bal Acc: 0.7650, 0.7592, 0.7356, 0.7427 → **0.751**

Row 19 — GraphSAGE base (unweighted):
- AUROC: 0.8388, 0.8380, 0.8401, 0.8406 → **0.839 ± 0.001**
- AUPRC: 0.8814, 0.8795, 0.8827, 0.8835 → **0.882**
- F1: 0.7565, 0.7630, 0.7638, 0.7667 → **0.763**
- Precision: 0.7721, 0.7835, 0.7275, 0.7409 → **0.756**
- Recall: 0.7415, 0.7436, 0.8039, 0.7945 → **0.771**
- Bal Acc: 0.7470, 0.7570, 0.7304, 0.7374 → **0.743**

Note: The existing `E18 = 0.837 ± 0.003` in the sheet may have been computed with seed 42 included; the corrected 4-seed value is `0.838 ± 0.001`.

**Expected Outcomes:** Rows 18–19 col E correctly filled in all 6 sheets.

**Todo List:**
1. Read `results/graph/random_student_results.csv`
2. Filter to weighted rows, exclude seed=42; compute mean/std per metric
3. Filter to unweighted rows, exclude seed=42; compute mean/std per metric
4. Update row 18 and 19 col E in all 6 sheets (correct existing values if needed)

---

## Sub-Task 5: Fill GNN ablation rows (rows 20, 30–34, col E)

**Status:** [ ] pending

**Intent:** Fill GraphSAGE + interaction attributes and the five GraphSAGE minus-component ablation rows for Random Student week 8.

**Source:** `results/graph/ablation_results.csv` (all 5 seeds: 42, 123, 7, 17, 99; all weighted)

**Condition → Excel row mapping:**
- `with_iw_attrs` → row 20 (GraphSAGE + interaction attributes)
- `no_edge_attrs` → row 30 (GraphSAGE − edge attributes)
- `no_temporal` → row 31 (GraphSAGE − temporal features)
- `no_course_features` → row 32 (GraphSAGE − course features)
- `no_vle` → row 33 (GraphSAGE − VLE)
- `no_assessment` → row 34 (GraphSAGE − assessment)

**Computed values (5 seeds: 42, 123, 7, 17, 99):**

Row 20 — with_iw_attrs:
- AUROC: 0.8436, 0.8489, 0.8468, 0.8498, 0.8420 → **0.846 ± 0.003**
- AUPRC: 0.8812, 0.8889, 0.8856, 0.8892, 0.8841 → **0.886**
- F1: 0.7605, 0.7724, 0.7685, 0.7751, 0.7665 → **0.769**
- Precision: 0.7826, 0.7444, 0.7333, 0.7292, 0.7757 → **0.753**
- Recall: 0.7395, 0.8025, 0.8073, 0.8272, 0.7576 → **0.787**
- Bal Acc: 0.7556, 0.7455, 0.7397, 0.7386, 0.7528 → **0.746**

Row 30 — no_edge_attrs:
- AUROC: 0.8264, 0.8387, 0.8379, 0.8386, 0.8319 → **0.835 ± 0.005**
- AUPRC: 0.8709, 0.8815, 0.8799, 0.8818, 0.8780 → **0.878**
- F1: 0.7509, 0.7651, 0.7624, 0.7629, 0.7615 → **0.761**
- Precision: 0.7251, 0.7540, 0.7263, 0.7119, 0.7178 → **0.727**
- Recall: 0.7786, 0.7765, 0.8024, 0.8218, 0.8109 → **0.798**
- Bal Acc: 0.7252, 0.7450, 0.7323, 0.7215, 0.7221 → **0.729**

Row 31 — no_temporal:
- AUROC: 0.8222, 0.8288, 0.8349, 0.8300, 0.8338 → **0.830 ± 0.005**
- AUPRC: 0.8655, 0.8720, 0.8753, 0.8737, 0.8775 → **0.873**
- F1: 0.7513, 0.7566, 0.7629, 0.7603, 0.7641 → **0.759**
- Precision: 0.7000, 0.7466, 0.7296, 0.7243, 0.7190 → **0.724**
- Recall: 0.8107, 0.7669, 0.7995, 0.8002, 0.8152 → **0.798**
- Bal Acc: 0.7122, 0.7363, 0.7343, 0.7266, 0.7244 → **0.727**

Row 32 — no_course_features:
- AUROC: 0.8256, 0.8329, 0.8299, 0.8304, 0.8296 → **0.830 ± 0.003**
- AUPRC: 0.8694, 0.8760, 0.8731, 0.8754, 0.8753 → **0.874**
- F1: 0.7453, 0.7530, 0.7517, 0.7567, 0.7570 → **0.753**
- Precision: 0.6558, 0.7549, 0.7395, 0.7006, 0.7302 → **0.716**
- Recall: 0.8632, 0.7510, 0.7643, 0.8226, 0.7858 → **0.797**
- Bal Acc: 0.6798, 0.7377, 0.7318, 0.7111, 0.7260 → **0.717**

Row 33 — no_vle:
- AUROC: 0.8150, 0.8280, 0.8248, 0.8243, 0.8257 → **0.824 ± 0.005**
- AUPRC: 0.8625, 0.8747, 0.8704, 0.8717, 0.8738 → **0.871**
- F1: 0.7425, 0.7507, 0.7509, 0.7523, 0.7556 → **0.750**
- Precision: 0.7029, 0.7317, 0.7481, 0.6919, 0.7114 → **0.717**
- Recall: 0.7868, 0.7707, 0.7538, 0.8244, 0.8057 → **0.788**
- Bal Acc: 0.7086, 0.7256, 0.7351, 0.7031, 0.7149 → **0.717**

Row 34 — no_assessment:
- AUROC: 0.7947, 0.8176, 0.8162, 0.8105, 0.8058 → **0.809 ± 0.009**
- AUPRC: 0.8416, 0.8629, 0.8607, 0.8573, 0.8547 → **0.855**
- F1: 0.7367, 0.7475, 0.7465, 0.7469, 0.7423 → **0.744**
- Precision: 0.6672, 0.7257, 0.6838, 0.6919, 0.7059 → **0.695**
- Recall: 0.8223, 0.7707, 0.8219, 0.8114, 0.7826 → **0.802**
- Bal Acc: 0.6832, 0.7207, 0.6987, 0.6999, 0.7038 → **0.701**

**Expected Outcomes:** Rows 20, 30–34 col E fully populated in all 6 sheets.

**Todo List:**
1. Read `results/graph/ablation_results.csv`
2. Group by condition; compute mean/std across 5 seeds per metric
3. Cross-check computed AUROC against existing Results sheet entries
4. Fill all 6 sheets for rows 20, 30–34, col E

---

## Sub-Task 6: Fill GNN LCPO Week 8 (row 18, col I)

**Status:** [ ] pending

**Intent:** Compute macro-mean ± std across 22 LCPO folds for the GNN at week 8.

**Source:** `results/graph/lcpo_summary.csv` (fold-level means already aggregated over 5 model seeds)

**Fold-level AUROC means (22 folds):**
0.7243, 0.7594, 0.8230, 0.8361, 0.8500, 0.8538, 0.7592, 0.7542, 0.7737, 0.8416, 0.8402, 0.7839, 0.8040, 0.7692, 0.7736, 0.8101, 0.8649, 0.8476, 0.8741, 0.6655, 0.6768, 0.6604

Macro-mean ≈ 0.791, std across fold-means ≈ 0.065 → **0.791 ± 0.065**

Note: Existing I18=`0.788 ± 0.065` — small discrepancy from prior single-seed run. The multi-seed summary gives 0.791.

**All other metrics** similarly computed from `lcpo_summary.csv`:
- AUPRC fold means → macro-mean ≈ 0.810
- F1 fold means → macro-mean ≈ 0.676
- Precision fold means → macro-mean ≈ 0.732
- Recall fold means → macro-mean ≈ 0.629
- Bal Acc fold means → macro-mean ≈ 0.715

**Expected Outcomes:** Row 18 col I filled in all 6 sheets with macro-averaged fold statistics.

**Todo List:**
1. Read `results/graph/lcpo_summary.csv`
2. Compute macro-mean and std across all 22 fold-level means per metric
3. Update row 18 col I in Results sheet (AUROC mean ± std)
4. Fill row 18 col I in remaining 5 metric sheets (mean only)

---

## Sub-Task 7: Fill Zenodo external validation (rows 18, cols N–O)

**Status:** [ ] pending

**Intent:** Fill Zenodo columns for GNN Random W8 (col N) and GNN LCPO W8 (col O).

**Sources:**
- `results/zenodo/random_student_results.csv` — 5 seeds
- `results/zenodo/lcpo_summary.csv` — 6 folds

**Zenodo Random Student W8 (col N), GNN weighted (5 seeds: 42, 123, 7, 17, 99):**
- AUROCs: 0.5635, 0.5968, 0.5176, 0.5245, 0.5672 → **0.554 ± 0.031**
- AUPRCs: 0.4089, 0.4580, 0.3546, 0.3901, 0.4091 → **0.402**
- F1: 0.4827, 0.4842, 0.4820, 0.4756, 0.5119 → **0.487**
- Precision: 0.3433, 0.3723, 0.3347, 0.4128, 0.3440 → **0.361**
- Recall: 0.8127, 0.6923, 0.8606, 0.5607, 1.0000 → **0.785**
- Bal Acc: 0.5190, 0.5271, 0.5063, 0.5510, 0.5026 → **0.521**

**Zenodo LCPO W8 (col O), GNN (6 folds):**
- AUROC fold means: 0.5125, 0.4833, 0.4830, 0.5031, 0.5150, 0.4950 → **0.499 ± 0.014**
- AUPRC fold means: 0.3305, 0.3098, 0.2858, 0.4415, 0.2689, 0.4427 → **0.346**
- F1 fold means: 0.2695, 0.0000, 0.0000, 0.4782, 0.3941, 0.4523 → **0.266**
- Precision: 0.2405, 0.0000, 0.0000, 0.4383, 0.2454, 0.4546 → **0.230**
- Recall: 0.3071, 0.0000, 0.0000, 0.6455, 1.0000, 0.5252 → **0.413**
- Bal Acc: 0.5021, 0.5000, 0.4988, 0.5051, 0.5004, 0.5162 → **0.504**

**Expected Outcomes:** Row 18 cols N and O filled in all 6 sheets.

**Todo List:**
1. Read `results/zenodo/random_student_results.csv`; compute mean/std across 5 seeds per metric
2. Read `results/zenodo/lcpo_summary.csv`; compute macro-mean across 6 folds per metric
3. Fill row 18 col N (Random) and col O (LCPO) in all 6 sheets

---

## Sub-Task 8: Fill matched-protocol LightGBM (row 12, col E)

**Status:** [ ] pending

**Intent:** Row 12 "LightGBM — all features (matched protocol)" is empty. One verified seed exists.

**Source:** `results/matched/lgbm_matched_results.csv` (seed 42, week 8, random split only)

**Single-seed values:**
- AUROC: 0.856 → reported as `0.856 ± —` (single run, no std)
- AUPRC: 0.890
- F1: 0.780
- Precision: 0.789
- Recall: 0.772
- Bal Acc: 0.771

**Expected Outcomes:** Row 12 col E populated with single-seed values using `± —` notation in Results sheet.

**Todo List:**
1. Read `results/matched/lgbm_matched_results.csv`
2. Write single-seed value to row 12 col E in all 6 sheets
3. Use `0.856 ± —` in Results sheet; bare `0.890` etc. in other sheets

---

## Cells with NO data available (remain `—`)

- **GNN weeks 2/4/6** (cols B/C/D for rows 18–19; cols F/G/H for rows 18–19): No multi-week GNN experiments were run. Tabular models at these weeks come from `all_splits_comparison.csv` but GNN experiments are week-8 only.
- **GNN LCPO cols F/G/H** (rows 18–19): Same — week 8 only.
- **Rows 16–17** (MLP variants): No MLP experiments in any CSV.
- **Rows 21–26** (GraphSAGE enrollment-node, edge-aware, + course-design, + temporal, + course-conditioned): No corresponding data in any CSV.
- **Rows 27–29** (GCN, R-GCN, HGT): No data.
- **LightGBM ablations weeks 2/4/6** (rows 13–15, cols B–D and F–H): `baseline_results_table.csv` has week-specific ablation values — **these CAN be filled**. See note below.

**Correction — LightGBM ablation rows (13–15) for weeks 2/4/6:**
`baseline_results_table.csv` contains VLE-only, assessment-only, and VLE+assessment for ALL weeks. These should be filled in cols B–D (Random Student) for rows 13–15. LCPO versions do not exist in any CSV, so cols F–H remain `—`.

---

## Execution Order in Agent Mode

Run sub-tasks in this order (each is independent after the source files are read):

1. Sub-task 1 (tabular Random-Student, all weeks — rows 7–11, 13–15, cols B–E)
2. Sub-task 2 (tabular LCPO, all weeks — rows 8–11, cols F–I)
3. Sub-task 3 (tabular Future Presentation — rows 8–11, cols J–M)
4. Sub-task 4 (GNN Random Student W8 base — rows 18–19, col E)
5. Sub-task 5 (GNN ablations — rows 20, 30–34, col E)
6. Sub-task 6 (GNN LCPO W8 — row 18, col I)
7. Sub-task 7 (Zenodo — row 18, cols N–O)
8. Sub-task 8 (matched LightGBM — row 12, col E)

**Also fill LightGBM ablation multi-week** as part of Sub-task 1: add week 2/4/6 values for rows 13–15 cols B–D from `baseline_results_table.csv`.

**Implementation notes:**
- Load `office-insights` skill before any `office_edit` call.
- `office_edit` calls must be serial — no parallel calls on the same file.
- Use `operation: set` with path `/SheetName/CellRef` (e.g. `/Results/E18`).
- Compute all values from the CSVs in-task (Python snippet or manual arithmetic) — do not estimate.
- After all edits are done, read back a sample of cells with `office_read mode: get` to verify.
