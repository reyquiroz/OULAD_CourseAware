# Research Next Steps Plan

## Confirmed Design Decisions

| Decision | Choice |
|---|---|
| **Sub-Task 1 first run** | Use `--quick` mode (2 folds) to validate the pipeline end-to-end before launching the full 5-seed × 22-fold run. |
| **Sub-Task 3 manuscript** | Produce both a tracked-changes revision of `docs/summer_report_draft.md` (preserving section numbering) AND a clean rewrite as `docs/summer_report_reframed.md` so both versions can be compared. |

**Goal**: Advance the OULAD GraphSAGE dissertation chapter through four parallel work streams: course-level variation analysis, manuscript reframing, external dataset validation, and reporting standards.  
**Scope**: `src/`, `results/graph/`, `docs/`, manuscript draft (`docs/summer_report_draft.md`).  
**Context**: Multi-seed LCPO (5 seeds × 22 folds) is implemented and ready to run. The current draft frames the contribution as "GraphSAGE outperforms LightGBM"; the chapter needs reframing as a leakage-safe evaluation framework. KU Leuven collaboration has not been initiated; feasibility must be assessed from published documentation before any implementation.

---

## Sub-Task 1 — Run multi-seed LCPO experiments and produce module-level summaries

**Status**: `[ ] pending`

### Intent
The multi-seed LCPO infrastructure (5 model seeds × 22 folds) is in place but has not been run since the correctness fixes. Re-running produces the definitive fold-level variance estimates needed for every downstream analysis. Module-level aggregation (across the 7 modules: AAA, BBB, CCC, DDD, EEE, FFF, GGG) must also be produced because the 22 folds are not independent — multiple presentations per module share students and course design.

### Expected Outcomes
- `results/graph/lcpo_results.csv` contains 110 rows (22 folds × 5 seeds) with `model_seed` column.
- `results/graph/lcpo_summary.csv` contains 22 fold-level rows with `{metric}_mean` and `{metric}_std`.
- New `src/summarize_lcpo_modules.py` produces `results/graph/lcpo_module_summary.csv` with one row per module, reporting mean ± std AUROC / AUPRC / F1 aggregated across that module's presentations and seeds, for both GNN and LightGBM.
- Within-module vs. across-module generalization gap is quantified: for each fold, record whether the test presentation's module appears in the training set (it always does under LCPO with >1 presentation per module) — and flag modules with only one presentation (AAA has 2; GGG has 3; all others have ≥3).

### Todo List
1. From the project root with the virtual environment active, run:
   ```bash
   python src/run_gnn_experiment.py --weeks 8 --seeds 42 123 7 --random-only
   python src/run_gnn_experiment.py --weeks 8  # LCPO with default 5 model seeds
   python src/compare_gnn_lgbm.py --weeks 8
   ```
2. Verify `results/graph/lcpo_results.csv` has 110 rows and a `model_seed` column.
3. Write `src/summarize_lcpo_modules.py`:
   - Read `lcpo_results.csv` and `lcpo_summary.csv`.
   - Group by `(code_module, model)` — derive `code_module` from `held_out_module` column.
   - Compute mean ± std across folds × seeds for AUROC, AUPRC, F1, Balanced_Acc.
   - Add `n_presentations` (number of unique presentations per module) and `n_students_total` (sum of `n_test` across that module's folds).
   - Write `results/graph/lcpo_module_summary.csv`.
4. Extend `src/generate_report_figures.py` to produce `fig_module_summary.png` — a horizontal bar chart of per-module GNN vs. LightGBM AUROC ± std.
5. Run `python src/verify_results.py` to confirm all table values are consistent.

### Relevant Context
- `src/run_gnn_experiment.py` — `run_lcpo_experiment(model_seeds=[42,123,7,17,99])` is implemented
- `results/graph/evaluation/week08/splits/week08_lcpo_folds.csv` — 22 folds, 7 modules: AAA(2), BBB(4), CCC(2), DDD(4), EEE(3), FFF(4), GGG(3)
- `src/course_variation.py` — existing per-fold GNN vs LightGBM delta (extend, do not duplicate)
- `src/compare_gnn_lgbm.py` — already aligned on features, splits, and threshold procedure

---

## Sub-Task 2 — Course-level diagnostic analysis

**Status**: `[ ] pending`

### Intent
The current `src/course_variation.py` reports only AUROC / F1 delta between GNN and LightGBM per fold. It does not explain *why* performance varies. The diagnostic must test each of the six proposed explanatory factors against the per-fold GNN AUROC: sample size, class prevalence, assessment availability, VLE activity density, and student overlap between presentations. This turns the course-level results from a descriptive table into an explanatory analysis.

### Expected Outcomes
- New `src/course_diagnostics.py` produces `results/graph/course_diagnostics.csv` with one row per fold and columns for all six diagnostic factors plus per-fold GNN AUROC (mean across seeds).
- A correlation table and scatter plot are produced: each diagnostic factor vs. GNN AUROC, annotated by module.
- Model stability (std AUROC across seeds) is included as a column — folds where the GNN is unstable are flagged separately from folds where LightGBM genuinely wins.
- Distribution shift proxy: for each held-out presentation, compute the mean cosine distance between its enrolled student feature vectors and the training set's enrolled student feature vectors (using the already-built node feature tensors from the artifact parquet).
- Student overlap: for each held-out module, what fraction of students in the held-out presentation also appear in at least one training presentation of the same module?

### Todo List
1. Write `src/course_diagnostics.py` with a `build_course_diagnostics(week=8)` function:
   - **Sample size**: read `n_test` from `week08_lcpo_folds.csv`.
   - **Class prevalence**: read `results/graph/artifacts/week08_enrollments.parquet`; compute at-risk rate per `(code_module, code_presentation)`.
   - **Assessment availability**: read `results/graph/artifacts/week08_edges_contains_assess.parquet`; count assessments per `(code_module, code_presentation)` — join to course_presentation node index.
   - **VLE activity density**: read `results/graph/artifacts/week08_edges_interacted_with.parquet`; compute median clicks per enrolled student per fold's presentation.
   - **Student overlap**: for each module, compute the fraction of held-out students who also appear in that module's other presentations (read enrollments, group by module).
   - **Distribution shift**: for each held-out fold, compute mean cosine distance between held-out student node feature vectors and training student vectors (read `week08_nodes_student.parquet`, one-hot encoded features).
   - **GNN AUROC mean and std**: join from `lcpo_summary.csv` (after Sub-Task 1).
2. Produce `results/graph/course_diagnostics.csv`.
3. Add to `src/generate_report_figures.py`:
   - `fig_course_diagnostics.png`: 2×3 scatter grid — each panel shows one diagnostic factor vs. GNN AUROC, with module colour coding and a linear trendline.
   - A `table_course_diagnostics.csv` + `.md` in `results/graph/tables/`.
4. Write a short findings summary to `docs/course_diagnostics_findings.md`: which factor shows the strongest correlation with GNN AUROC, and whether model instability or genuine course difficulty better explains low-AUROC folds.

### Relevant Context
- `src/course_variation.py` — per-fold GNN vs. LightGBM delta; extend, do not replace
- `results/graph/artifacts/week08_*.parquet` — node and edge tables with all required data
- `results/graph/evaluation/week08/splits/week08_lcpo_folds.csv` — fold definitions
- `results/graph/lcpo_summary.csv` — per-fold AUROC mean/std (available after Sub-Task 1)
- `src/gnn_model.py` — `_onehot()` helper available for feature encoding

---

## Sub-Task 3 — Manuscript reframing

**Status**: `[ ] pending`

### Intent
`docs/summer_report_draft.md` currently positions the paper as "GraphSAGE outperforms LightGBM." The dissertation chapter contribution must be reframed as: *a leakage-safe, temporally aligned, and course-design-aware evaluation framework for graph and tabular models under cross-course generalization.* Every section of the draft needs to reflect this repositioning without discarding the empirical results.

### Expected Outcomes
- Revised `docs/summer_report_draft.md` with updated title, abstract, introduction, methods framing, and conclusions.
- Three new research questions replace or augment the current three:
  1. Can a leakage-safe graph representation improve early at-risk prediction beyond a feature-matched tabular baseline?
  2. How does cross-course generalization (LCPO) performance vary as a function of course-design factors?
  3. What is the relative contribution of graph structure vs. enrollment-scoped edge attributes to predictive performance?
- Contributions section (new) itemises: leakage prevention methodology, temporal alignment, enrollment-centric graph schema, matched LightGBM comparison, LCPO diagnostic framework.
- Results section retains all tables and figures but narrative emphasises framework quality rather than margin size.
- Limitations section explicitly addresses: single-dataset evaluation, OULAD anonymisation of module identities, performance variance across seeds, and the fact that the GNN margin under LCPO is modest (0.020 AUROC).

### Todo List
1. Revise the title to reflect the framework framing (suggested: *"A Leakage-Safe Evaluation Framework for Graph and Tabular Models in Cross-Course At-Risk Prediction"* — or equivalent agreed with supervisor).
2. Rewrite the Abstract: lead with the framework contribution; retain the empirical summary but contextualise the margin.
3. Revise Section 1 (Introduction): add a paragraph on evaluation methodology as a contribution; cite the importance of leakage prevention and temporal alignment in learning analytics.
4. Add a new Section 3.x (Evaluation Design): describe leakage prevention (Strategy B dual guard), enrollment-centric supervision, LCPO protocol, multi-seed variance estimation, and matched LightGBM comparison — these were implemented but not narrated.
5. Revise Section 5 (Results): add a subsection on course-level variation (from Sub-Tasks 1 and 2); add a subsection on model stability (within-fold seed std); frame win counts as "18 of 22 folds" (corrected from the earlier reported 19).
6. Revise Section 6 (Discussion / Conclusions): reframe the main claim; add a paragraph on what the framework enables for future datasets.
7. Add Section 7 (Limitations): single dataset, module anonymisation, LCPO independence assumption, modest LCPO margin.
8. Write this revision directly into `docs/summer_report_draft.md`, preserving the existing section numbers where possible.

### Relevant Context
- `docs/summer_report_draft.md` — current draft (title through conclusions)
- `docs/LEAKAGE_PREVENTION.md` — canonical leakage methodology documentation
- `docs/EVALUATION_SPLITS.md` — LCPO and random-student split definitions
- `results/graph/tables/table_main_comparison.csv` — authoritative numbers
- Corrected win count: 18/22 folds (not 19) — confirmed by `src/verify_results.py`

---

## Sub-Task 4 — KU Leuven dataset feasibility report

**Status**: `[ ] pending`

### Intent
Before investing any implementation effort in a second dataset, a structured feasibility report must assess whether the KU Leuven Blackboard dataset (published by Hlosta et al. or the KU Leuven MOOC/blended dataset) can support the same evaluation protocol used on OULAD. The report will inform the decision of whether to proceed with a full benchmark study.

### Expected Outcomes
- New document `docs/ku_leuven_feasibility.md` assessing the KU Leuven dataset against six criteria, with a go / no-go / conditional recommendation.
- The report is based on published dataset descriptions, papers, and any available data dictionaries — no data access is assumed.
- A separate section assesses KDD Cup 2015 / XuetangX against the same criteria.
- The report concludes with a prioritised recommendation: which dataset (if any) to pursue first, and what pre-conditions must be met.

### Todo List
1. Search published sources for the KU Leuven Blended / Blackboard learning analytics dataset (likely: Hlosta et al., or the Open University spin-off). Document: publisher, license, access procedure, contact, DOI.
2. Assess against the six feasibility criteria — for each criterion record: satisfied / partially / not satisfied / unknown, with evidence:
   - **Early prediction windows**: Can weekly cutoffs (≤8 weeks) be applied? Is date-of-interaction available at sufficient granularity?
   - **Risk label**: Is a binary at-risk label (fail/withdraw vs. pass) or equivalent derivable from the published outcome fields?
   - **Course/year transfer**: Does the dataset span multiple courses or cohort-years suitable for LCPO-style evaluation?
   - **Graph schema compatibility**: Can the OULAD heterogeneous schema (student, course, assessment, VLE resource nodes; enrolled_in, submitted, interacted_with edges) be instantiated? What fields map to what?
   - **Tabular schema compatibility**: Can the 9 tabular features used by the LightGBM baseline be replicated?
   - **Sample size and class balance**: Are there enough enrollments (>1,000) and sufficient at-risk prevalence (>20%) for meaningful evaluation?
3. Repeat criteria assessment for KDD Cup 2015 / XuetangX — focus on multi-course evaluation and comparable outcome definition.
4. Write a 1-page recommendation section: go / no-go for each dataset, rationale, and any pre-conditions (e.g., data access request, schema mapping work estimate).
5. Save to `docs/ku_leuven_feasibility.md`.

### Relevant Context
- `docs/DATA_POLICY.md` — current dataset governance documentation (OULAD only)
- `docs/keitha_pearce_data_request_draft.md` — Canvas data request template; field requirements listed there mirror what would be needed for any external dataset
- OULAD schema reference: `docs/GRAPH_SCHEMA.md` — defines the target schema that any new dataset must map to
- LightGBM feature list: `src/compare_gnn_lgbm.py` `_FEATURE_COLS` — 9 features that must be replicable

---

## Sub-Task 5 — Dataset-specific reporting structure

**Status**: `[ ] pending`

### Intent
Once a second dataset is confirmed feasible (Sub-Task 4), a reporting skeleton must be in place so that results are never pooled across datasets and comparisons are explicit. This sub-task creates the reporting infrastructure — directory structure, CSV schema, figure templates, and a results section template in the draft — so that adding a second dataset later requires only filling in data, not redesigning the output.

### Expected Outcomes
- `results/` top-level directory is namespaced by dataset: `results/oulad/` and `results/{dataset2}/` (with `results/graph/` → `results/oulad/graph/` migration plan documented but not yet executed — migration is deferred until a second dataset is confirmed).
- New `src/reporting_schema.py` defines the canonical column schema for all cross-dataset comparison tables, with a `dataset` column as the primary grouping key.
- `docs/summer_report_draft.md` gains a placeholder Section 5.x (External Validation) with the reporting template: per-dataset table, cross-dataset comparison table, and explicit acknowledgement of dataset-specific differences.
- `src/generate_report_figures.py` gains a stub `make_cross_dataset_figure(df)` that accepts a combined DataFrame with a `dataset` column and produces a faceted comparison figure — callable once a second dataset's results are available.

### Todo List
1. Document the migration plan in `docs/reporting_structure.md`: describes the intended `results/{dataset}/` structure, the `dataset` column in all result CSVs, and the principle that no aggregate is ever computed across datasets.
2. Add `dataset` column to `results/graph/comparison_results.csv` — value `"oulad"` for all existing rows. Update `src/verify_results.py` to treat `dataset` as a grouping key.
3. Add placeholder Section 5.x to `docs/summer_report_draft.md` with empty tables labelled "OULAD results" and "External dataset results — pending feasibility confirmation".
4. Add `make_cross_dataset_figure()` stub to `src/generate_report_figures.py`.
5. Write `docs/reporting_structure.md`.

### Relevant Context
- `src/verify_results.py` — consistency checker; needs updating when `dataset` column added
- `results/graph/comparison_results.csv` — current unified results file
- Sub-Task 4 recommendation will determine which second dataset to name in the placeholders

---

## Execution Order and Dependencies

```
Sub-Task 1 (multi-seed LCPO + module summaries)
    │
    ├──► Sub-Task 2 (course diagnostics)   ← needs lcpo_summary.csv from Sub-Task 1
    │
    └──► Sub-Task 3 (manuscript reframing)  ← needs corrected numbers and module summary
              │
              └──► Sub-Task 5 (reporting structure)  ← extends the draft from Sub-Task 3

Sub-Task 4 (KU Leuven feasibility)   ← independent; can run in parallel with 1–3
    │
    └──► Sub-Task 5 (reporting structure)  ← depends on Sub-Task 4 recommendation
```

Sub-Tasks 1, 2, and 3 must be completed before the dissertation chapter draft is submitted for supervisor review. Sub-Task 4 can proceed in parallel. Sub-Task 5 is the integrating step and should follow both Sub-Task 3 and Sub-Task 4.

---

## Immediate Next Actions (ordered)

1. Run multi-seed LCPO (`python src/run_gnn_experiment.py --weeks 8`) — Sub-Task 1 step 1.
2. Write `src/summarize_lcpo_modules.py` — Sub-Task 1 step 3.
3. Write `src/course_diagnostics.py` — Sub-Task 2 step 1.
4. Write `docs/ku_leuven_feasibility.md` — Sub-Task 4 (parallel).
5. Revise `docs/summer_report_draft.md` — Sub-Task 3 (after Sub-Tasks 1 and 2 produce final numbers).
6. Add `dataset` column and reporting skeleton — Sub-Task 5 (after Sub-Task 4 recommendation).
