# Pipeline Correctness & External Dataset Validation Plan

## Overview

This plan addresses four categories of work requested by the supervisor:

1. **Bug fixes** — Two hard-coded seed resets in `gnn_model.py` make all model-seed runs
   produce identical outputs. The GNN results in `results/graph/lcpo_results.csv` (only
   2 of 22 folds, all 5 seeds identical per fold) are therefore invalid and must be discarded.

2. **Design questions** — The supervisor raised four open questions: training-only
   normalization (already correct — no change needed), inductive vs. transductive setting
   (already inductive — document clearly), use of `interacted_with` edge attributes
   (currently topology-only — add an ablation flag), and LightGBM feature matching
   (already matched — document clearly).

3. **Full experiments** — After fixing bugs, run the complete 22-fold LCPO evaluation,
   multi-seed random-student split, and ablation study. Regenerate all tables, figures,
   and manuscript sections. Old results must be archived and clearly distinguished.

4. **External dataset** — Assess the Zenodo dataset (https://zenodo.org/records/17087849)
   for OULAD schema compatibility, build the same pipeline on it if compatible, and produce
   matched GNN + LightGBM comparison tables.

---

## Sub-Task 1 — Fix Seed Independence Bugs

**Status:** `[x] done`

### Intent
Remove the two hard-coded `torch.manual_seed(SEED)` / `np.random.seed(SEED)` calls in
`src/gnn_model.py` that override the per-seed setup done by callers. After this fix,
each model-seed will produce genuinely independent weight initializations and training
trajectories.

### Root Cause
- `EnrollmentGNN.__init__` (line 603 of `src/gnn_model.py`) calls
  `torch.manual_seed(SEED)` unconditionally every time a model is constructed, so every
  model regardless of which seed the caller set will initialize to the same weights.
- `run_training_loop` (lines 792–793 of `src/gnn_model.py`) resets both seeds to `SEED`
  at the top of the function, overwriting the seed the caller set before invoking it.

The callers (`run_lcpo_experiment` and `run_random_split_experiment` in
`src/run_gnn_experiment.py`) already correctly call `torch.manual_seed(mseed)` and
`np.random.seed(mseed)` immediately before constructing the model and calling the training
loop. The two internal resets undo this work.

### Expected Outcomes
- All five model seeds (42, 123, 7, 17, 99) produce distinct val AUROC values and
  distinct metrics for each fold — verifiable by re-running `--quick` (2 folds) and
  checking that `results/graph/lcpo_results.csv` rows differ across seeds.
- The existing test suite continues to pass (the `run_overfit_check` function in
  `gnn_model.py` lines 957–958 and the `__main__` block at lines 1009–1010 may each
  retain their own `torch.manual_seed(SEED)` calls as those are self-contained
  diagnostics).

### Todo List
- [x] Remove `torch.manual_seed(SEED)` from `EnrollmentGNN.__init__` (line 603 of
      `src/gnn_model.py`)
- [x] Remove `torch.manual_seed(SEED)` and `np.random.seed(SEED)` from the top of
      `run_training_loop` (lines 792–793 of `src/gnn_model.py`)
- [x] Run `pytest tests/ -v` — all tests must pass
- [x] Run `python src/run_gnn_experiment.py --quick --week 8` and confirm that the
      5 model seeds now produce different `best_val_auroc` values for at least one fold in
      the printed output (visual check before committing)

### Relevant Context
- `src/gnn_model.py` lines 600–605 (`EnrollmentGNN.__init__`)
- `src/gnn_model.py` lines 780–795 (`run_training_loop`)
- `src/run_gnn_experiment.py` lines 491–496 (LCPO model-seed loop — already correct)
- `src/run_gnn_experiment.py` lines 308–312 (random-split seed setup — already correct)

---

## Sub-Task 2 — Add `interacted_with` Attribute Ablation Flag

**Status:** `[x] done`

### Intent
The `interacted_with` edges (student → vle_resource, carrying total_clicks and
n_interactions) currently contribute only graph topology. Add a boolean flag
`use_interacted_with_attrs` to `EnrollmentGNN` and its callers so that the ablation study
can compare topology-only vs. attribute-enriched message passing. This avoids a
hard architectural change while letting the data decide.

### Expected Outcomes
- `EnrollmentGNN` accepts a `use_interacted_with_attrs: bool = False` constructor
  argument (default preserves current behavior)
- When `True`, the `interacted_with` edge features are projected and concatenated into
  the source student node embedding during message passing (or passed to the head — whichever is architecturally cleaner given the current `ei_attr_proj` pattern)
- `run_random_split_experiment` and `run_ablation.py` can pass this flag through
- A new ablation condition `"with_iw_attrs"` (or similar) is added to `CONDITIONS` in
  `src/run_ablation.py`
- Results tables show both settings; the manuscript reports whichever variant is stronger
  as the main result (with the other as ablation evidence)

### Todo List
- [x] Add `use_interacted_with_attrs: bool = False` parameter to
      `EnrollmentGNN.__init__` in `src/gnn_model.py`
- [x] When `True`, construct a `nn.Linear` for `interacted_with` edge attr projection
      and apply it in `forward()` alongside `enrolled_in` attr projection
- [x] Thread the flag through `_build_model_and_optimizer` in
      `src/run_gnn_experiment.py`
- [x] Add `"with_iw_attrs"` to `CONDITIONS` in `src/run_ablation.py`
- [x] Run `pytest tests/ -v` — all tests must pass with the default (`False`) value
- [x] Document the flag in the model docstring

### Relevant Context
- `src/gnn_model.py` lines 590–655 (`EnrollmentGNN` class definition and `forward`)
- `src/run_gnn_experiment.py` lines 66–78 (`_build_model_and_optimizer`)
- `src/run_ablation.py` lines 23–30 (`CONDITIONS` list)

---

## Sub-Task 3 — Document Already-Correct Design Decisions

**Status:** `[x] done`

### Intent
The supervisor asked about training-only normalization and inductive vs. transductive
evaluation. Both are already implemented correctly. This sub-task adds explicit
documentation in code comments and a short section in `docs/` so reviewers can verify
without reading through all the code.

### Findings (no code change required)
- **Training-only normalization**: `_normalize_numeric_features` in `src/gnn_model.py`
  accepts a `train_edge_mask` and computes mean/std exclusively from training edges/nodes,
  then applies those statistics to the full graph. It is called after the split mask is
  created in both the random-split and LCPO experiments.
- **Inductive evaluation**: `build_train_subgraph` in `src/gnn_model.py` filters
  `enrolled_in`, `submitted`, and `interacted_with` edges to training rows only. Node
  features for all nodes are retained, but test-enrollment edges are never visible during
  training message passing.
- **LightGBM feature matching**: `compare_gnn_lgbm.py` uses `random_student_split` with
  the same seed and fractions, `filter_window` with `submission_date_guard=True`, and
  `build_features` aligned to the same enrollment table. The LCPO protocol uses the same
  fold definitions from `results/graph/evaluation/week{N}/splits/`.

### Expected Outcomes
- A new section in `docs/DESIGN_DECISIONS.md` (create if absent) enumerating: (1)
  training-only normalization protocol, (2) inductive graph construction, (3) LightGBM
  feature matching, with file + line references for each
- Key code comments added inline at the call sites in `src/run_gnn_experiment.py` (one
  comment per design decision at the point where the decision is enacted)

### Todo List
- [x] Add inline comment at normalization call sites in `src/run_gnn_experiment.py`
      (lines ~331 and ~480) stating "train_edge_mask ensures stats computed from training
      data only — prevents leakage into test normalization"
- [x] Add inline comment at `build_train_subgraph` call in `src/run_gnn_experiment.py`
      confirming inductive setting
- [x] Create or update `docs/DESIGN_DECISIONS.md` with a section for each of the three
      confirmed-correct decisions, citing the relevant function and line numbers

### Relevant Context
- `src/gnn_model.py` lines 54–180 (`_normalize_numeric_features`)
- `src/gnn_model.py` lines 201–299 (`build_train_subgraph`)
- `src/run_gnn_experiment.py` lines 325–335 and 465–485 (call sites)
- `src/compare_gnn_lgbm.py` lines 130–180 (`_build_lgbm_features`)

---

## Sub-Task 4 — Archive Old Results and Run Full Corrected Experiments

**Status:** `[x] done`

### Intent
Old results produced by the buggy pipeline (seed-collapsed LCPO, 2-fold quick run) must
be archived before new results overwrite them. Then the full corrected pipeline is run:
22-fold LCPO, multi-seed random-student split, and ablation study. All outputs are clearly
stamped as "corrected pipeline" in filenames or a `pipeline_version` column.

### Expected Outcomes
- `results/graph/lcpo_results_v1_buggy.csv` — renamed copy of the current
  `lcpo_results.csv` before any new run
- New `results/graph/lcpo_results.csv` containing 22 folds × 5 seeds = 110 rows, with
  distinct metric values across seeds within each fold
- New `results/graph/random_student_results.csv` from multi-seed run using seeds
  42, 123, 7, 17, 99 (same set as LCPO model seeds — confirmed)
- New `results/graph/ablation_results.csv` including the new `with_iw_attrs` condition
- All new CSVs contain a `pipeline_version` column set to `"v2_corrected"` to
  distinguish them from any legacy rows

### Todo List
- [ ] Archive: rename `results/graph/lcpo_results.csv` →
      `results/graph/lcpo_results_v1_buggy.csv` (and similarly for
      `lcpo_summary.csv`, `random_student_results.csv`, `ablation_results.csv`,
      `comparison_results.csv`)
- [ ] Add a `pipeline_version` column to the output-writing code in
      `src/run_gnn_experiment.py` and `src/run_ablation.py` (value `"v2_corrected"`)
- [ ] Run full 22-fold LCPO:
      `python src/run_gnn_experiment.py --week 8 --random-only` first to verify seed
      independence, then
      `python src/run_gnn_experiment.py --week 8` for both experiments
- [ ] Run multi-seed random-student split:
      `python src/run_gnn_experiment.py --week 8 --random-only --seeds 42 123 7 17 99`
- [ ] Run ablation:
      `python src/run_ablation.py --week 8 --seeds 42 123 7 17 99`
- [ ] Run LightGBM comparison:
      `python src/compare_gnn_lgbm.py --weeks 8 --seeds 42 123 7 17 99`
- [ ] Verify that `lcpo_results.csv` has 110 rows (22 folds × 5 seeds) with seed
      variation within each fold

### Relevant Context
- `src/run_gnn_experiment.py` — main experiment runner
- `src/run_ablation.py` — ablation conditions
- `src/compare_gnn_lgbm.py` — LightGBM comparison
- `results/graph/` — all output CSVs live here

---

## Sub-Task 5 — Regenerate Tables and Figures

**Status:** `[x] done`

### Intent
Regenerate all manuscript tables and figures from the corrected experiment outputs.
The existing figures in `results/graph/figures/` were produced from the buggy quick run
and must be replaced.

### Expected Outcomes
- All figures in `results/graph/figures/` updated from new CSVs
- `results/graph/tables/` updated with new LaTeX/CSV summary tables
- The `make_cross_dataset_figure` stub in `src/generate_report_figures.py` remains a
  stub until Sub-Task 7 (external dataset) produces results

### Todo List
- [ ] Run `python src/generate_report_figures.py` — confirm no errors (cross-dataset
      figure stub is expected to be skipped/stubbed)
- [ ] Run `python src/summarize_lcpo_modules.py` to regenerate per-module summaries
- [ ] Run `python src/course_diagnostics.py` to regenerate course-level diagnostics
- [ ] Run `python src/compare_gnn_lgbm.py --from-csv` to regenerate comparison tables
      from the new CSVs without re-running training
- [ ] Visually inspect key figures (LCPO boxplot, random-split AUROC, ablation bar chart)
      for plausibility — seed variation should now be visible in error bars

### Relevant Context
- `src/generate_report_figures.py` — figure generation entry point
- `src/summarize_lcpo_modules.py` — per-module LCPO summaries
- `src/compare_gnn_lgbm.py --from-csv` — regenerates tables without re-running models
- `results/graph/figures/` and `results/graph/tables/`

---

## Sub-Task 6 — Assess Zenodo Dataset for OULAD Schema Compatibility

**Status:** `[x] done — CONDITIONAL GO verdict issued; all pre-conditions satisfied (license confirmed, supervisor approved, OULAD VLE-only baseline run → results/graph/comparison_results_vle_only.csv)`

### Intent
Before building any pipeline code, systematically assess the Zenodo dataset
(https://zenodo.org/records/17087849) against the six OULAD feasibility criteria defined
in `docs/ku_leuven_feasibility.md`. Produce a schema mapping document. Only if the
dataset passes a minimum threshold (criteria C1–C3 satisfied, C4 partially satisfied)
does implementation proceed in Sub-Task 7.

### Expected Outcomes
- `docs/zenodo_dataset_feasibility.md` — assessment document covering:
  - Dataset identification (name, institution, DOI, license, scale)
  - Six-criterion assessment table (same format as `docs/ku_leuven_feasibility.md`)
  - Schema mapping: OULAD node/edge types vs. Zenodo dataset equivalents
  - Go / No-Go verdict with explicit pre-conditions if conditional
- `docs/DATA_POLICY.md` updated with the Zenodo dataset entry (source URL, access date,
  license, re-identification risk statement)

### Todo List
- [ ] Download the Zenodo record metadata and file listing (no code needed — manual
      inspection of the record page and any attached data descriptor)
- [ ] Apply the six criteria from `docs/ku_leuven_feasibility.md` §2 to the Zenodo
      dataset and record the assessment
- [ ] Produce the schema mapping tables (node mapping, edge mapping, tabular feature
      reconstruction) in the same format as §3.3 / §4.3 of `ku_leuven_feasibility.md`
- [ ] Write the Go / No-Go verdict and pre-conditions
- [ ] Write `docs/zenodo_dataset_feasibility.md`
- [ ] Update `docs/DATA_POLICY.md` with the new dataset entry
- [ ] If verdict is CONDITIONAL GO: document all gaps and pre-conditions in
      `docs/zenodo_dataset_feasibility.md`; do NOT proceed to Sub-Task 7 without
      explicit supervisor approval
- [ ] If verdict is NO-GO: stop here and report to supervisor; do not proceed to
      Sub-Task 7

### Relevant Context
- `docs/ku_leuven_feasibility.md` — template for assessment format and criteria
- `docs/DATA_POLICY.md` — data provenance registry
- `docs/GRAPH_SCHEMA.md` — canonical OULAD graph schema to map against

---

## Sub-Task 7 — Build External Dataset Pipeline (If Compatible)

**Status:** `[x] complete`
*Prerequisite: Sub-Task 6 verdict is GO or CONDITIONAL GO **AND** supervisor has
reviewed and approved corrected OULAD results from Sub-Task 5*

### Intent
Build a matched pipeline for the Zenodo dataset that mirrors the OULAD pipeline as
closely as the source data permits. The goal is to produce GNN and LightGBM results on
the external dataset that can be placed in the same comparison tables as the OULAD results.

Any structural gaps identified in Sub-Task 6 (missing node types, missing features) must
be explicitly handled and documented as dataset-specific limitations, not silently omitted.

### Expected Outcomes
- `src/zenodo_data.py` — data loader analogous to `src/oulad_data.py`, exposing the same
  public API (`load_raw_tables`, `filter_window`, `build_features`, split utilities)
- `src/run_zenodo_pipeline.py` — CLI entry point to build graph artifacts for the
  external dataset, stored under `results/zenodo/`
- `results/zenodo/artifacts/` — parquet graph artifacts, mirroring OULAD's structure
- `results/zenodo/lcpo_results.csv` and `results/zenodo/random_student_results.csv` from
  GNN experiments
- `results/zenodo/comparison_results.csv` from LightGBM comparison
- `src/generate_report_figures.py::make_cross_dataset_figure` implemented (the current
  stub replaced with real code) to produce `results/graph/figures/fig_cross_dataset.png`

### Todo List
- [ ] Write `src/zenodo_data.py` following the public API of `src/oulad_data.py`;
      document all schema gaps as `# GAP: <explanation>` comments
- [ ] Run `python src/check_data.py`-equivalent verification for the Zenodo data files
- [ ] Build graph artifacts: `python src/run_zenodo_pipeline.py`
- [ ] Generate splits: extend `src/save_graph_splits.py` or write a Zenodo-specific
      equivalent
- [ ] Run GNN experiments on Zenodo: `python src/run_gnn_experiment.py --dataset zenodo`
      (or a separate script if the dataset flag is too invasive)
- [ ] Run LightGBM comparison on Zenodo using `src/compare_gnn_lgbm.py`
- [ ] Implement `make_cross_dataset_figure` in `src/generate_report_figures.py`
- [ ] Update `docs/DESIGN_DECISIONS.md` to describe where the Zenodo pipeline diverges
      from OULAD and why

### Relevant Context
- `src/oulad_data.py` — template for data loader public API
- `src/graph_pipeline.py` — staged pipeline to replicate
- `src/compare_gnn_lgbm.py` — LightGBM comparison template
- `src/generate_report_figures.py` lines 463–481 — cross-dataset figure stub
- `docs/zenodo_dataset_feasibility.md` — schema mapping from Sub-Task 6
- `docs/GRAPH_SCHEMA.md` — canonical node/edge schema

---

## Sub-Task 8 — Zenodo Dataset Results Notebooks

**Status:** `[ ] pending`
*Prerequisite: Sub-Task 7 complete (external dataset pipeline and results CSVs exist)*

### Intent
Produce two canonical, reproducible Jupyter notebooks for the Zenodo dataset that mirror
the structure of the existing OULAD canonical notebooks
(`notebooks/OULAD_Graph_Analysis_Final.ipynb` and
`notebooks/OULAD_Baseline_Analysis.ipynb`). These notebooks serve as the reproducible
record of the external dataset results and are the primary artefacts shared with the
supervisor for interpretation.

Two notebooks are needed because the data exploration/graph construction work is
distinct from the model evaluation work — keeping them separate matches the OULAD
notebook convention and makes each notebook self-contained and reviewable independently.

### Expected Outcomes

**Notebook 1 — `notebooks/Zenodo_Dataset_Graph_Analysis.ipynb`**
- Purpose: pipeline construction and data characterisation for the Zenodo dataset
- Contents:
  - Dataset overview: scale, class balance, course-presentation count vs. OULAD
  - Schema gap commentary (any node/edge types absent vs. OULAD, with `# GAP` callouts)
  - Graph artifact build (calls `src/run_zenodo_pipeline.py` or equivalent)
  - Integrity validation output (mirrors the validation section in
    `OULAD_Graph_Analysis_Final.ipynb`)
  - Split summary: LCPO fold count, random-split class balance per split
  - Side-by-side summary table comparing Zenodo graph stats vs. OULAD (node counts,
    edge counts, class imbalance ratio)

**Notebook 2 — `notebooks/Zenodo_Dataset_Results.ipynb`**
- Purpose: model evaluation results and cross-dataset comparison
- Contents:
  - GNN random-student split results (AUROC, AUPRC, F1, Balanced Acc) with seed
    variation displayed as mean ± std
  - GNN LCPO results across all held-out course-presentations with per-fold breakdown
  - LightGBM comparison table (same metrics, same splits)
  - Cross-dataset comparison: Zenodo GNN and LightGBM vs. OULAD GNN and LightGBM
    (side-by-side table and the figure produced by `make_cross_dataset_figure`)
  - Ablation results for the `with_iw_attrs` condition on Zenodo
  - Limitations section: any schema gaps that constrain comparability, with direct
    references to `docs/zenodo_dataset_feasibility.md`

### Todo List
- [ ] Create `notebooks/Zenodo_Dataset_Graph_Analysis.ipynb` following the section
      structure of `notebooks/OULAD_Graph_Analysis_Final.ipynb`; call into
      `src/zenodo_data.py` and `src/run_zenodo_pipeline.py`
- [ ] Add a side-by-side dataset comparison table (Zenodo vs. OULAD graph stats)
      as a pandas DataFrame display cell near the end of Notebook 1
- [ ] Create `notebooks/Zenodo_Dataset_Results.ipynb` following the section
      structure of `notebooks/OULAD_Baseline_Analysis.ipynb`; load results from
      `results/zenodo/*.csv`
- [ ] Add a cross-dataset comparison section in Notebook 2 that loads both
      `results/graph/comparison_results.csv` (OULAD) and
      `results/zenodo/comparison_results.csv` (Zenodo) and displays them in a
      combined table
- [ ] Embed the `fig_cross_dataset.png` figure (produced in Sub-Task 7) in Notebook 2
- [ ] Run both notebooks top-to-bottom with a clean kernel and confirm no errors;
      commit with all cell outputs saved

### Relevant Context
- `notebooks/OULAD_Graph_Analysis_Final.ipynb` — structural template for Notebook 1
- `notebooks/OULAD_Baseline_Analysis.ipynb` — structural template for Notebook 2
- `results/zenodo/` — all Zenodo result CSVs produced in Sub-Task 7
- `results/graph/figures/fig_cross_dataset.png` — cross-dataset figure from Sub-Task 7
- `docs/zenodo_dataset_feasibility.md` — schema gap reference for limitations section

---

## Execution Order

```
Sub-Task 1 (fix seed bug)
    ↓
Sub-Task 2 (add iw-attr ablation flag)
    ↓
Sub-Task 3 (document design decisions)
    ↓
Sub-Task 4 (archive + run full experiments) ──────── Sub-Task 6 (assess Zenodo dataset)
    ↓                                                      ↓
Sub-Task 5 (regenerate tables/figures)        [report verdict to supervisor]
    ↓                                                      │
    └──────────── supervisor review of OULAD results ──────┘
                              ↓
             Sub-Task 7 (external dataset pipeline)
             — only with GO verdict AND supervisor approval
                              ↓
             Sub-Task 8 (Zenodo results notebooks)
```

Sub-Tasks 1–3 are code-only changes with no long-running compute.
Sub-Task 4 is the main compute step (22 folds × 5 seeds, expected several hours on CPU).
Sub-Task 6 begins in parallel with Sub-Task 4 — it is documentation/inspection work and
requires no running code.
Sub-Task 7 is gated on both the Sub-Task 6 feasibility verdict AND explicit supervisor
approval of the corrected OULAD results from Sub-Task 5.
Sub-Task 8 begins only after Sub-Task 7 results CSVs exist; the two notebooks are the
final deliverable for the external dataset comparison.

---

### Confirmed Decisions
- **Random-student split seeds**: `42 123 7 17 99` (same as LCPO model seeds)
- **Sub-Task 7 gate**: requires both (a) Sub-Task 6 GO/CONDITIONAL-GO verdict AND
  (b) explicit supervisor approval of corrected OULAD results from Sub-Task 5;
  no automatic proceed on conditional go
- **Sub-Tasks 4/5 and Sub-Task 6 run in parallel** — Sub-Task 6 is documentation
  work requiring no running code, so it does not block the compute in Sub-Task 4
