# CourseAware Review — Remediation Plan

**Source:** `CourseAware_Code_Review_2026-10-16.md` (Xuan Guo, Oct 16 2026)  
**Scope:** Fix the evaluation methodology, re-establish model comparisons on equal footing,
and begin the course-information experiment.
**Approach:** One sub-task per logical change, introduced in order so each intermediate
result is preserved and traceable. Run weeks 2/4/6/8 throughout, but always validate on
week 8 first.

---

## Sub-Task 1 — Exclude already-unregistered enrollments

**Status:** [x] done (code + unit test; week-8 artifacts not yet rebuilt — requires full OULAD data)

### Intent
Students whose `date_unregistration` is on or before the prediction-window cutoff have a
known outcome at prediction time and should not be included in any model's evaluation.
The review showed 6,071 of 32,593 week-8 enrollments (18.6%) fall into this category and
inflate every AUROC score.

### Expected Outcomes
- A filter function `exclude_early_unregistered(enrollments_df, cutoff_day)` exists in
  `src/oulad_data.py` that returns the filtered DataFrame and a count of removed rows.
- `src/graph_pipeline.py` calls this filter per prediction week after loading registrations,
  and logs the removal count.
- All four weeks report the number of removed enrollments in console output.
- Re-running the pipeline produces artifact parquet files that no longer contain
  already-unregistered enrollments.

### Todo List
1. Add `exclude_early_unregistered(enrollments_df, student_registration_df, cutoff_day)`
   to `src/oulad_data.py`. It should join on `(id_student, code_module, code_presentation)`,
   filter rows where `date_unregistration <= cutoff_day`, and return
   `(filtered_df, n_removed)`.
2. Call the new function in `src/graph_pipeline.py` inside `apply_window_cutoff()` (or
   its equivalent), after `studentRegistration` is loaded. Log removal counts.
3. Update `src/run_matched_comparison.py` to apply the same filter before building
   tabular features, so LightGBM/MLP results are on the same population.
4. Add a unit test in `tests/` that verifies the function removes exactly the enrollments
   whose `date_unregistration` ≤ cutoff and preserves the rest.
5. Rebuild week-8 graph artifacts. Record the new enrollment count in a comment or log.

### Relevant Context
- Registration data loaded at `src/oulad_data.py` line 102.
- `src/graph_pipeline.py` lines 59–78 load supplementary tables; the student_registration
  table is available there but not consumed for filtering.
- `PREDICTION_WINDOWS` in `src/config.py` maps week → day (week 8 → day 56).
- The architecture rule: prediction unit is the enrollment triple
  `(id_student, code_module, code_presentation)`.

---

## Sub-Task 2 — Align LightGBM / MLP inputs with GraphSAGE

**Status:** [x] done (code + unit test; actual model runs require full OULAD data and LS6 — deferred)

### Intent
GraphSAGE receives student demographics (`gender`, `region`, `highest_education`,
`imd_band`, `disability`) and course identity (one-hot `code_module`,
`code_presentation`) via node features, while the matched baseline in
`run_matched_comparison.py` receives only six behavioral summaries, `num_of_prev_attempts`,
`studied_credits`, and `age_band`. This asymmetry means the comparison is not apples-to-apples.

The review proposes running five input configurations:
- A: current matched set (baseline)
- B: A + demographics
- C: A + course identity
- D: A + demographics + course identity
- E: D + the four temporal features

### Expected Outcomes
- `src/run_matched_comparison.py` accepts a `--feature-set {A,B,C,D,E}` argument (or
  equivalent named conditions) that controls which columns are added.
- Results for all five configurations are written to `results/matched/` with a `feature_set`
  column identifying each run. Existing results are not overwritten (dedup by
  `week + seed + feature_set`).
- Week-8 numbers for all five configurations appear in the results matrix.

### Todo List
1. Extend `build_tabular_features()` in `src/run_matched_comparison.py` with optional
   arguments: `add_demographics=False`, `add_course_identity=False`,
   `add_temporal=False`.
2. Add a `--feature-set` CLI argument mapping to the five configurations above.
3. Ensure `feature_set` is written as a column in every output row, and add it to the
   deduplication key in `_append_or_create_csv()` calls.
4. Run all five configurations for all four weeks (prioritize week 8 first).
5. Update the results matrix to show the five rows alongside the GraphSAGE numbers from
   the review (section 1.1 table).

### Relevant Context
- Current base feature list: `src/run_matched_comparison.py` lines 67–76.
- `age_band` one-hot added at line 151; demographics and course identity columns need the
  same one-hot treatment.
- `_append_or_create_csv()` dedup pattern is in `src/run_gnn_experiment.py` — follow that
  pattern here.
- Temporal features (`weeks_active`, `recency`, `activity_trend`, `max_inactive_gap`) are
  computed in `src/graph_pipeline.py` and already land in the enrollment parquet artifacts;
  they just need to be pulled into the tabular DataFrame.

---

## Sub-Task 3 — Make enrollment nodes an explicit loader option

**Status:** [x] done

### Intent
`GraphDataLoader.load()` in `src/gnn_model.py` currently adds enrollment nodes whenever
`week08_nodes_enrollment.parquet` exists on disk (line 589, file-existence check). This
means all variants inadvertently included enrollment nodes whenever the artifacts were
present, making the ablations invalid. The loader needs an explicit boolean flag.

### Expected Outcomes
- `GraphDataLoader.__init__()` accepts `use_enrollment_nodes: bool = False`.
- Loading enrollment node artifacts is gated on this flag, not on file existence.
- `run_gnn_experiment.py` and `run_ablation.py` expose a `--use-enrollment-nodes` CLI flag
  and pass it through to the loader.
- Rebuilt artifacts with the flag disabled produce graphs without enrollment nodes.
- The six-summaries, edge-aware, and temporal variants are re-run without enrollment nodes
  (unless explicitly enabled), producing fresh result rows with `condition` labels.

### Todo List
1. Add `use_enrollment_nodes: bool = False` parameter to `GraphDataLoader.__init__()`.
2. Replace the `os.path.exists(enr_node_path)` condition at line 589 with
   `self.use_enrollment_nodes and os.path.exists(enr_node_path)`.
3. Add `--use-enrollment-nodes` flag to `src/run_gnn_experiment.py` and
   `src/run_ablation.py`.
4. Rebuild week-8 artifacts with the default (flag off) and verify enrollment node edge
   types are absent from the graph.
5. Re-run the cumulative variant sequence (base → +summaries → +edge-attrs → +temporal)
   one change at a time, writing each result with the appropriate `condition` label.

### Relevant Context
- `GraphDataLoader.load()` enrollment node block: `src/gnn_model.py` lines 587–620.
- `EnrollmentGNN` always registers enrollment edge types (they are no-ops when the node
  type has no features, but they still participate in message passing structure).
- Architecture rule: graph construction and split creation are fully decoupled — do not
  couple the flag to split logic.

---

## Sub-Task 4 — Fix LCPO results keys and rerun all variants

**Status:** [x] done (code changes only; LCPO re-runs deferred — require full data)

### Intent
In `src/run_gnn_experiment.py`, `lcpo_results.csv` and `lcpo_summary.csv` are deduplicated
on `["week", "fold_idx", "model_seed"]` only — `condition` is not part of the key.
Running LCPO for a variant overwrites the base model rows. `condition` must be added to
every written record and to all deduplication keys before any LCPO variant runs are
attempted.

### Expected Outcomes
- `condition` is present in every row written to `lcpo_results.csv`, `lcpo_summary.csv`,
  and the main GNN results file.
- Deduplication keys for all three files include `condition`.
- LCPO is run for every variant (base, +summaries, +edge-attrs, +temporal,
  +enrollment-nodes) at all four weeks.
- The results matrix includes LCPO AUROC ± SD and Wilcoxon p-value for each variant.

### Todo List
1. In `src/run_gnn_experiment.py`, add `condition` to the record dict before the first
   `_append_or_create_csv()` call (before the column is later appended separately).
2. Update dedup key lists at lines 631, 653, and 815 to include `"condition"`.
3. Verify `lcpo_summary.csv` aggregation still groups correctly after the key change.
4. Run LCPO for all variants at weeks 2, 4, 6, 8 (week 8 first). Keep results from each
   run separate via the `condition` key.
5. Produce a paired fold-level comparison table (Wilcoxon signed-rank) for each variant
   vs. matched LightGBM, at each week.

### Relevant Context
- Dedup key locations: `src/run_gnn_experiment.py` lines 631, 653, 815.
- `condition` is assigned at line 778/796 but only added to the dataframe after the write
  helper is first called.
- Architecture rule: two-level seed separation — `--seeds` (split seeds) and
  `--model-seeds` (init seeds) must remain independent.
- 22 LCPO folds correspond to unique (code_module, code_presentation) pairs.

---

## Sub-Task 5 — Run the six-summaries MLP control and fix loss reporting

**Status:** [x] done

### Intent
The review requests a "six-summaries MLP" control (the MLP with only the six behavioral
summaries, no demographics or course identity) to isolate the contribution of architecture
vs. inputs. Additionally, each variant currently has 10 rows (5 seeds × 2 loss settings),
and the mean pools both, which conflates loss weighting with architecture. These should be
reported separately.

### Expected Outcomes
- A dedicated MLP run using only the six behavioral summaries (`vle_total`, `vle_mean`,
  `vle_std`, `assess_mean`, `assess_max`, `assess_count`) exists in the results.
- The results matrix reports both loss settings (weighted and unweighted) separately, with
  explicit `loss_weighting` labels.
- The six-summaries MLP result appears in the results matrix under a distinct `condition`.

### Todo List
1. Run `src/run_matched_comparison.py` (or the MLP experiment script) with feature-set A
   (six summaries only) for all four weeks.
2. Confirm the `loss_weighting` column exists in all GNN/MLP result files.
3. Split GNN variant summary tables by `loss_weighting` value before computing mean ± SD;
   report both settings separately in the matrix (e.g., two rows per variant: weighted and
   unweighted).
4. Add the six-summaries MLP row to the results matrix.

### Relevant Context
- MLP matched results: `results/matched/mlp_matched_results.csv` and
  `results/results/matched/mlp_matched_results.csv` (duplicate folder issue — see
  Sub-Task 7).
- Confirmed from the review: base GraphSAGE has only 4 seeds (seed 42 missing at all
  weeks); fix or document this when running new seeds.

---

## Sub-Task 6 — Fix recency for inactive enrollments

**Status:** [x] done

### Intent
In `src/graph_pipeline.py` line 487, `recency` is set to `0.0` via `fillna(0.0)` for
enrollments with no VLE activity. A value of 0 means "active on the cutoff day," which is
the opposite of what we want for inactive students. The correct sentinel is
`window_days + 1` (i.e., never active within the window), or an explicit boolean
`no_activity` indicator column.

### Expected Outcomes
- Enrollments with no VLE activity receive `recency = window_days + 1` (or equivalent
  sentinel), not `0.0`.
- An `active_in_window` boolean column is added as an explicit no-activity indicator.
- A unit test verifies that a synthetic enrollment with zero clicks gets the correct
  recency value.
- Graph artifacts are rebuilt after this fix.

### Todo List
1. In `src/graph_pipeline.py`, replace the blanket `fillna(0.0)` on `recency` with
   `fillna(window_days + 1)` (the window length + 1 sentinel).
2. Add an `active_in_window` boolean column (`True` if the enrollment has at least one
   click in the window, `False` otherwise) to the enrollment edge features.
3. Update `_normalize_numeric_features()` in `src/gnn_model.py` to include
   `active_in_window` in the categorical/binary feature group (skip z-scoring it).
4. Add a unit test in `tests/` for the recency sentinel behavior.
5. Rebuild artifacts; verify the change in the parquet schema.

### Relevant Context
- `fillna(0.0)` location: `src/graph_pipeline.py` line 487.
- `_compute_temporal_features()` computes `recency = float(window_days - max_d)` at
  line 295 — students present in this function always have activity; NaN only arises from
  the left-join at line 480.
- Architecture rule: normalization scoped to training statistics.

---

## Sub-Task 7 — Consolidate results folder and update the matrix

**Status:** [ ] pending

### Intent
`results/` and `results/results/` are mirror trees containing different versions of the
same files (e.g., `random_student_results.csv` in `results/results/graph/` contains a
temporal run that stopped at epoch 4). Using both as sources has led to inconsistent
values in the matrix (e.g., single-seed matched LightGBM from the older file vs.
five-seed value from the canonical one). There should be one authoritative results tree,
and the matrix must be built from it.

Additionally, the report compared GraphSAGE against historical LightGBM (0.863/0.835)
rather than the matched LightGBM (0.857/0.826) that already exists in `results/matched/`.
The Zenodo LightGBM baseline (~0.65 random, ~0.61 LCPO) also needs to be added.

### Expected Outcomes
- The content of `results/results/` is backed up to `results/results_bk/` first.
- `results/results/` becomes the single authoritative results path, documented in
  `src/config.py` constants. All scripts write to `results/results/`.
- The old top-level `results/` sub-directories (graph, matched, zenodo, etc.) are either
  redirected or left as archives with a README note.
- The results matrix uses matched LightGBM (five seeds, all four weeks) as the comparison
  baseline throughout.
- The Zenodo LightGBM row appears in the matrix.
- Incompletely-trained runs (e.g., epoch-4 run) are removed from canonical CSVs.

### Todo List
1. Copy `results/results/` → `results/results_bk/` as a snapshot before any changes.
2. Audit any files present in top-level `results/` sub-directories that are NOT in
   `results/results/`; move them into `results/results/` to consolidate.
3. Update `src/config.py` to define canonical path constants pointing to `results/results/`
   sub-directories (graph, matched, zenodo, lcpo, etc.).
4. Update any hardcoded `"results/..."` path strings in `src/gnn_model.py`,
   `src/run_gnn_experiment.py`, and other scripts to use the new `src/config.py` constants.
5. Update the results matrix to use `results/results/matched/lgbm_matched_results.csv`
   (five seeds) as the LightGBM comparison baseline.
6. Add the Zenodo LightGBM results from `results/results/zenodo/comparison_results.csv`
   to the matrix.
7. Remove or flag any result rows where `best_epoch` < 10 (incompletely trained runs).

### Relevant Context
- `results/results/matched/lgbm_matched_results.csv` is the canonical five-seed matched
  LightGBM file; verify the top-level `results/matched/` copy is identical.
- Architecture rule: all persistent paths must use `src/config.py` constants.
  `gnn_model.py` and `run_gnn_experiment.py` currently hardcode path strings — this
  sub-task is the moment to fix that pattern.

---

## Sub-Task 8 — Fix enrollment-node normalization to use training statistics

**Status:** [ ] pending

### Intent
When enrollment nodes are loaded, their numeric features are normalized using global
statistics (all enrollments), not training-only statistics. This is a minor but principled
leak that violates the training-only normalization rule enforced elsewhere in the pipeline.

### Expected Outcomes
- `_normalize_numeric_features()` in `src/gnn_model.py` normalizes enrollment node
  features using statistics computed from training enrollment nodes only.
- The train mask for enrollment nodes is derived from `train_edge_mask` (the set of
  `enrolled_in` edges in the training split).

### Todo List
1. In `_normalize_numeric_features()`, identify where enrollment node features would be
   normalized. Add a branch that uses `train_edge_mask` to select training enrollment
   nodes and compute mean/std from them only.
2. Verify the fix does not change behavior when `use_enrollment_nodes=False` (from
   Sub-Task 3).
3. Add a comment citing the architecture rule about training-only normalization.

### Relevant Context
- `_normalize_numeric_features()`: `src/gnn_model.py` lines 54–194.
- Architecture rule: normalization is scoped to training statistics; `train_edge_mask`
  is the canonical mechanism for this.

---

## Sub-Task 9 — Course-information experiment

**Status:** [ ] pending

### Intent
The central hypothesis of the paper is that course-design features (assessment types,
weights, schedule, length, resource mix) allow a model to generalize to new courses where
a course ID cannot. This sub-task implements the three-condition × three-evaluation-setting
experiment proposed in the review (section 2).

| Course information | Random-student | LCPO | Leave-module-out |
|---|---|---|---|
| None | baseline | baseline | baseline |
| Course ID only | expected: helps | expected: helps a little | expected: does not help |
| Course design features | expected: helps | expected: helps | expected: helps |

### Expected Outcomes
- A `leave_module_out_split()` function exists in `src/oulad_data.py` that holds out all
  presentations of a given module.
- LightGBM and GraphSAGE are each run under three course-information conditions and three
  evaluation settings, on the still-enrolled (unregistered-excluded) population.
- Results are reported with Wilcoxon paired tests for LCPO and leave-module-out
  comparisons.
- If course design features help under leave-module-out where course ID does not, this is
  identified as the central finding.

### Todo List
1. Add `leave_module_out_split(enrollments_df, held_out_module)` to `src/oulad_data.py`.
   It should mask all enrollments where `code_module == held_out_module`, regardless of
   presentation.
2. Add `--eval-setting {random,lcpo,lmo}` CLI flag to `src/run_gnn_experiment.py` and
   `src/run_matched_comparison.py`.
3. Define three course-information conditions in both pipelines:
   - `no_course`: remove `code_module`, `code_presentation`, and course-design columns.
   - `course_id`: include `code_module` and `code_presentation` one-hots only.
   - `course_design`: include only course-design features (`n_assessments`,
     `total_assessment_weight`, `prop_cma`, `prop_tma`, `prop_exam`, `n_vle_resources`,
     `module_presentation_length`), not the identity columns.
4. Run all 3 × 3 = 9 combinations for LightGBM and GraphSAGE at week 8.
5. Produce a summary table with AUROC ± SD and Wilcoxon p-values comparing `course_design`
   vs. `course_id` under each evaluation setting.
6. Extend to weeks 2, 4, 6 after week-8 results are reviewed.

### Relevant Context
- `lcpo_split()` in `src/oulad_data.py` lines 326–342: uses both module AND presentation
  as the hold-out condition. Leave-module-out needs only the module condition.
- Architecture rule: `enrolled_in` / `rev_enrolled_in` edges are never filtered during
  LCPO. The same must hold for leave-module-out: course_presentation nodes stay visible.
- Course-presentation node features (one-hot code_module/code_presentation + design
  features) are built in `src/gnn_model.py` lines 456–465.

---

## Sub-Task 10 — Zenodo pipeline improvements

**Status:** [ ] pending

### Intent
The Zenodo GraphSAGE currently uses constant-1.0 student and enrollment features, and the
interaction edge attributes are built but not passed to the model. This means its
near-random result reflects model setup rather than institutional differences. The Zenodo
pipeline also lacks class balance and graph statistics that were requested for the paper.

### Expected Outcomes
- The Zenodo GNN uses actual interaction edge attributes (`total_clicks`, `n_interactions`,
  `first_day`, `last_day`, `active_days`) in message passing (either by enabling
  `use_interacted_with_attrs=True` or switching SAGEConv to sum aggregation).
- Existing Zenodo LightGBM results from `results/zenodo/comparison_results.csv` are added
  to the results matrix.
- Dataset statistics (class balance, graph node/edge counts) are computed and documented.

### Todo List
1. In `src/run_zenodo_pipeline.py`, add `use_interacted_with_attrs=True` to the GNN
   initialization (or expose it as a CLI flag and set the default to `True`).
2. Alternatively, switch the Zenodo GNN's SAGEConv aggregation from `mean` to `sum` so
   interaction frequency is naturally captured.
3. Add a reporting block that prints class balance (fraction at-risk) and graph statistics
   (node counts by type, edge counts by type) after graph construction.
4. Add the existing LightGBM Zenodo results to the results matrix with a `dataset=zenodo`
   label.
5. Document in comments why student features are constants in Zenodo (no demographics
   available) and what information the model actually has access to.

### Relevant Context
- Zenodo student features: `src/run_zenodo_pipeline.py` lines 458–460 (const 1.0).
- Zenodo enrollment edge attrs: lines 481–485 (const 1.0).
- Interaction edge attrs are built but `use_interacted_with_attrs` is not set.
- `results/zenodo/comparison_results.csv` contains LightGBM Zenodo results (~0.65 / ~0.61).

---

## Execution Order

```
Sub-Task 1  →  Sub-Task 6  →  Sub-Task 3  →  Sub-Task 4
     ↓                              ↓
Sub-Task 2  →  Sub-Task 5  →  Sub-Task 7
                                    ↓
                               Sub-Task 8  →  Sub-Task 9  →  Sub-Task 10
```

Sub-Tasks 1 and 6 must be completed first because they change the enrollment population
and feature values; all subsequent results depend on these fixes. Sub-Task 3 (enrollment
node flag) must precede Sub-Task 4 (LCPO rerun), because the variants cannot be run
cleanly until the loader is fixed. Sub-Task 7 (folder consolidation) — which includes
backing up `results/results/` and making it the canonical path — should happen before
Sub-Tasks 9 and 10 to avoid writing new results into the wrong location.
