# Plan: Fill Remaining Matrix Gaps

## Current State

79 cells filled across 6 sheets. The remaining empty cells fall into five distinct groups,
each requiring a different action.

---

## Gap Inventory

### Column reference
| Col | Meaning |
|-----|---------|
| B–E | OULAD Random-Student, weeks 2/4/6/8 |
| F–I | OULAD LCPO, weeks 2/4/6/8 |
| J–M | OULAD Future-Presentation, weeks 2/4/6/8 |
| N   | Zenodo Random W8 |
| O   | Zenodo LCPO W8 |

---

## Group 1 — GNN LCPO Weeks 2/4/6  (rows 18–19, cols F/G/H)
**Effort: LOW — run existing script, ~30 min compute per week per variant**

The `run_gnn_experiment.py` runner already supports `--weeks` and `--lcpo-only`.
LCPO for week 8 (col I) is already in `results/graph/lcpo_summary.csv`.
Weeks 2/4/6 simply have not been run yet.

**Scope:**
- Row 18 (weighted) cols F, G, H — weeks 2, 4, 6
- Row 19 (unweighted) cols F, G, H — weeks 2, 4, 6 *(note: unweighted LCPO has never been run for any week)*

**Commands:**
```bash
source oulad_env/bin/activate

# Weighted LCPO weeks 2, 4, 6 (uses default --model-seeds 42 123 7 17 99)
python src/run_gnn_experiment.py --weeks 2 4 6 --lcpo-only

# Unweighted: there is no direct --unweighted flag in the LCPO path.
# Check gnn_model.py to confirm whether loss_weighting is configurable for LCPO,
# or whether LCPO always runs weighted only.
# If unweighted LCPO is not supported, leave row 19 cols F/G/H as —.
```

**Output written to:** `results/graph/lcpo_summary.csv` (appended, keyed by week+fold)

**Matrix write:** After run, recompute macro-mean ± std per week from the new rows and
write to rows 18–19, cols F/G/H in all 6 sheets using the same pattern as col I.

---

## Group 2 — MLP (rows 16–17, cols B–I)
**Effort: LOW — script already exists (`run_matched_comparison.py`), just needs to be run**

`src/run_matched_comparison.py` already implements:
- **Row 16:** `MLP_all_features` — MLPClassifier(128, 64) on all enrollment features
- **Row 17:** `MLP_behavioral_summaries` — MLPClassifier on 6 VLE behavioral summary features

The output CSV `results/matched/mlp_matched_results.csv` does not yet exist.
`src/update_results_matrix.py` already knows how to read it and write to rows 16–17.

**Command:**
```bash
source oulad_env/bin/activate

# Run MLP for both random and LCPO splits, all weeks, 5 seeds
python src/run_matched_comparison.py --models mlp --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 --splits random lcpo
```

**Fills:**
- Row 16: cols B–I (Random-Student W2/4/6/8 + LCPO W2/4/6/8)
- Row 17: cols B–I (same)
- Future-Presentation (cols J–M) is NOT supported by `run_matched_comparison.py` —
  those cells remain `—`.
- Zenodo cols N–O: not applicable — remain `—`.

**Matrix write:** After run, call `python src/update_results_matrix.py` which already
has the MLP read/write logic, OR compute manually from the CSV and write with openpyxl.

---

## Group 3 — GNN Architectural Variants (rows 21–26, col E)
**Effort: MEDIUM — conditions are implemented in gnn_model.py but need to be run**

The following conditions are supported by `--condition` in `run_gnn_experiment.py`
but have never been run and have no results CSV:

| Row | Matrix label | `--condition` value | Implemented? |
|-----|-------------|---------------------|--------------|
| 21 | GraphSAGE + six enrollment summaries (matched protocol) | `with_enrollment_summaries` | ✅ |
| 22 | GraphSAGE — enrollment-node representation | `enrollment_node` | ✅ |
| 23 | GraphSAGE — edge-aware message passing | `edge_aware_mp` | ✅ |
| 24 | GraphSAGE + explicit course-design features | *(check if implemented)* | ❓ |
| 25 | GraphSAGE + temporal behavior features | `temporal_features` | ✅ (seed 42 only, in random_student_results.csv) |
| 26 | GraphSAGE + course-conditioned prediction | *(check if implemented)* | ❓ |

**Note on row 25:** `temporal_features` has seed 42 in `random_student_results.csv`
already (this was why seed 42 was excluded from the base model rows). Seeds 123/7/17/99
have never been run for this condition.

**Commands (for confirmed-implemented conditions):**
```bash
source oulad_env/bin/activate

# Row 21
python src/run_gnn_experiment.py --week 8 --random-only \
    --condition with_enrollment_summaries --seeds 42 123 7 17 99

# Row 22
python src/run_gnn_experiment.py --week 8 --random-only \
    --condition enrollment_node --seeds 42 123 7 17 99

# Row 23
python src/run_gnn_experiment.py --week 8 --random-only \
    --condition edge_aware_mp --seeds 42 123 7 17 99

# Row 25 — temporal_features (seed 42 already exists, add remaining seeds)
python src/run_gnn_experiment.py --week 8 --random-only \
    --condition temporal_features --seeds 123 7 17 99
```

**For rows 24 and 26:** Before running, verify implementation:
```bash
grep -n "course_design\|course_conditioned\|course_condition" src/run_gnn_experiment.py src/gnn_model.py
```
If the condition flag is not mapped in `run_gnn_experiment.py`, it needs to be wired up
or the cell stays `—`.

**Output written to:** `results/graph/random_student_results.csv` (appended, with
`condition` column set to the variant name)

**Matrix write:** For each condition, filter the CSV to that condition + 5 seeds,
compute mean ± std, write col E only (week 8, random-student).

---

## Group 4 — Alternative GNN Architectures (rows 27–29: GCN, R-GCN, HGT)
**Effort: HIGH — these architectures are NOT implemented in gnn_model.py**

A search of `src/gnn_model.py` found no `GCN`, `RGCN`, or `HGT` class or import.
These would require implementing three new model classes from scratch using
`torch_geometric.nn` (GCNConv, RGCNConv, HGTConv).

**Recommendation:** Out of scope unless explicitly requested. Leave cols B–O as `—`.

If implementing, the plan would be:
1. Add `GCNEnrollmentModel`, `RGCNEnrollmentModel`, `HGTEnrollmentModel` to `gnn_model.py`
2. Add `--arch {graphsage,gcn,rgcn,hgt}` flag to `run_gnn_experiment.py`
3. Run week 8 random-student (5 seeds) for each architecture
4. Optionally run LCPO for the best-performing one

---

## Group 5 — GNN Ablations Multi-Week (rows 20, 30–34, cols B/C/D)
**Effort: HIGH — requires running 6 ablation conditions × 3 weeks × 5 seeds = 90 runs**

Currently ablations exist only for week 8 (col E). Extending to weeks 2/4/6 would use:
```bash
source oulad_env/bin/activate

# Example for one condition
python src/run_ablation.py --weeks 2 4 6 --condition no_assessment --seeds 42 123 7 17 99
```

**But:** Check whether `src/run_ablation.py` supports `--weeks`:
```bash
python src/run_ablation.py --help
```

**Recommendation:** Lower priority. The week 8 ablation already shows the feature
importance ordering clearly. Multi-week ablations add depth but are not essential for
the primary comparison. Run only if time permits after Groups 1–3.

---

## Group 6 — Remaining Structural Gaps (always `—`)

These cells have no feasible data source and should permanently remain `—`:

| Rows | Cols | Reason |
|------|------|--------|
| 12 (LightGBM matched) | B–D, F–I, J–M, N–O | Matched protocol only designed for W8 |
| 13–15 (LightGBM ablations) | F–I (LCPO), J–M (Future-Pres), N–O (Zenodo) | No LCPO/FP ablation experiments exist |
| 16–17 (MLP) | J–M, N–O | Future-Pres and Zenodo not in run_matched_comparison.py |
| 18–20, 30–34 (GNN variants) | J–M | No GNN future-presentation pipeline |
| 19, 21–26, 30–34 (GNN variants) | O | Zenodo LCPO only evaluated for base weighted model |
| 7 (Majority) | B–D, F–I, J–M, N | Majority classifier only meaningful at W8 |

---

## Execution Order (by value vs effort)

| Priority | Group | Cells added | Compute time | Prerequisite |
|----------|-------|-------------|--------------|--------------|
| 1 | GNN LCPO W2/4/6 (Group 1) | 12 × 6 sheets = 72 | ~2–3 hrs | None |
| 2 | MLP rows 16–17 (Group 2) | 16 × 6 sheets = 96 | ~30 min | None |
| 3 | GNN arch variants rows 21–25 (Group 3) | 5 × 6 sheets = 30 | ~2 hrs | Verify condition flags |
| 4 | GNN ablations multi-week (Group 5) | 18 × 6 sheets = 108 | ~4–6 hrs | Verify run_ablation.py --weeks |
| 5 | GCN/R-GCN/HGT (Group 4) | 3 × 6 sheets = 18 | 1–2 days dev + compute | New implementation |

---

## Deliverable After Each Group

After each group's experiments complete, run:
```bash
python src/update_results_matrix.py
# or the targeted openpyxl write script used previously
```
Then commit:
```bash
git add AUROC_Results_Matrix_by_Dr_Guo.xlsx results/graph/ results/matched/
git commit -m "Fill matrix: <group description>"
```
