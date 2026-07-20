# Pipeline Quality Improvements — Plan

## Top-Level Overview

**Goal**: Seven targeted correctness and reproducibility improvements to the OULAD
student success prediction pipeline, building on the completed `graph-pipeline-fixes-plan.md`
work. All changes are additive or corrective — no architectural rewrites.

**Key context**:
- The `filter_window()` dual-guard (Strategy B) is already implemented and results regenerated.
- The graph pipeline's 7-stage API is stable. The submission edge (`submitted`) carries
  `score` but not `date_submitted` — the submission date is consumed by the filter and
  then discarded, so the validation cannot currently check it post-filter.
- `num_of_prev_attempts` and `studied_credits` are enrollment-scoped attributes that
  vary per (student, course, presentation) but are currently collapsed to a single row
  via `drop_duplicates("id_student")`, silently losing values for multi-course students.
- The Strategy A vs Strategy B AUROC comparison exists only as a comment in source code.
- `analyze_course_difficulty()` averages AUROC over all weeks and models, hiding
  per-week and per-model variation.
- Std values in comparison CSVs are derived from pandas `.std()` (ddof=1, sample std),
  but the convention is never stated.

**Scope**: 7 sub-tasks, each independently reviewable.

**Confirmed design decisions**:
- Sub-Tasks 2 and 4 share the `extra_metadata` mechanism and are implemented in one pass.
- Sub-Task 6 (Strategy A vs B comparison) runs as a separate one-off script
  (`src/run_strategy_comparison.py`), not as part of the main `run_evaluation.py` runner.
- Sub-Task 7: the `Week` column exists in `lcpo_df` — confirmed.

**Not in scope**: GNN training, new model architectures, graph schema beyond the
`enrolled_in` edge attribute addition, notebook re-execution.

---

## Sub-Tasks

---

### Sub-Task 1: Update stale documentation describing due-date-only filtering

**Status**: `[ ] pending`

**Intent**
Several documentation files and code comments still describe the **old** Strategy A
(filter by `assessments.date` only) even though Strategy B (dual guard: due date AND
`date_submitted`) has been implemented. These stale references mislead readers into
thinking only due-date filtering is applied.

**Expected Outcomes**
- `docs/LEAKAGE_PREVENTION.md` — Assessment filtering section updated to describe both
  guards. "Best Practice #2" updated from "Use due dates, not submission dates" to
  reflect the dual-guard approach.
- `README.md` — "Key Design Decisions" table row for assessment temporal filter updated
  to mention both the due-date guard and the submission-date guard.
- `docs/validation_report_week8.md` — "Key Changes" section already mentions the fix
  but does not include the submission-date guard in the temporal compliance table.
  Add a `max_date_submitted` row to the Temporal Compliance section.
- `src/graph_pipeline.py` — The module docstring's "Typical call sequence" comment
  says "assessment *due date* (not submission date)". Update to "assessment due date
  AND submission date (dual guard, Strategy B)".
- No functional code changes in this sub-task.

**Todo List**
1. Read `docs/LEAKAGE_PREVENTION.md` sections 3 (Assessment Features) and the
   "Best Practices" section. Update the filtering description and code snippet to show
   both guard conditions. Remove the "Critical Note" that says only due date is used.
2. Read `README.md` "Key Design Decisions" table. Update the "Assessment temporal filter"
   row description to include both guards.
3. Read `docs/validation_report_week8.md` Temporal Compliance section. Add a
   `date_submitted` row showing `max(date_submitted) ≤ 56` once Task 2 is complete
   (coordinate wording; can add a placeholder noting the validation check is added in
   Sub-Task 2).
4. Read `src/graph_pipeline.py` lines 1–55 (module docstring). Update the comment
   about assessment filtering in the Typical call sequence block.
5. Do a project-wide grep for "due date, not submission" or "not submission date" to
   catch any additional stale references.

**Relevant Context**
- `docs/LEAKAGE_PREVENTION.md:84-102` — current assessment filtering description
- `docs/LEAKAGE_PREVENTION.md:264-267` — "Best Practices" item #2
- `README.md:218` — Key Design Decisions table
- `src/graph_pipeline.py:20-21` — module docstring comment about filter strategy
- `src/oulad_data.py:107-158` — the implemented dual-guard in `filter_window()` (ground truth)

---

### Sub-Task 2: Add `max_date_submitted` validation check to graph validation

**Status**: `[ ] pending`

**Intent**
The graph validation (`src/graph_validation.py`) currently confirms temporal compliance
via `interacted_with_max_last_day` and `assessment_max_due_date`, but does **not**
check that the maximum `date_submitted` in the submitted-score data is ≤ the prediction
cutoff. Since the `submitted` edge artifact does not retain `date_submitted` (it was
consumed and dropped by the filter), this check must be surfaced via a different
mechanism — either by adding `date_submitted` as an edge attribute, or by recording
the max at pipeline construction time and embedding it in the metadata JSON.

The chosen approach is to record `max_date_submitted` in the `week{N}_metadata.json`
at pipeline construction time (Stage 7, `materialize_graph_artifacts`) and have
`graph_validation.py` read it from there and report it as a temporal compliance check.

**Expected Outcomes**
- `src/graph_pipeline.py:build_edge_tables()` — after building the `submitted` edge,
  compute `max_date_submitted` from the filtered `student_assess` and store it in a
  variable passed to `materialize_graph_artifacts()`.
- `src/graph_pipeline.py:materialize_graph_artifacts()` — writes `max_date_submitted`
  into `week{N}_metadata.json` alongside the existing `window_days` field.
- `src/graph_validation.py:run_validation()` — reads `max_date_submitted` from
  `week{N}_metadata.json` and adds a new `submitted_max_date_submitted` check to
  `temporal["checks"]` with `compliant = (value <= window_days)`.
- `src/graph_validation.py:_build_summary()` — the temporal compliance section
  prints the new check alongside the existing ones.
- After re-running `python src/run_graph_pipeline.py --week 8`, the summary shows:
  `[✓] submitted_max_date_submitted   value=56 (threshold=56)`
- `docs/validation_report_week8.md` Temporal Compliance table gains the new row.

**Todo List**
1. Read `src/graph_pipeline.py:build_edge_tables()` — confirm where the `submitted`
   edge is built and identify where to compute `max_date_submitted`.
2. Compute `max_date_submitted = int(filtered["student_assess"]["date_submitted"].max())`
   after building the `submitted` edge and return it alongside `edges` (or via a
   separate dict that gets passed to `materialize_graph_artifacts()`).
3. Read `src/graph_pipeline.py:materialize_graph_artifacts()`. Update its signature to
   accept an optional `extra_metadata: dict` parameter and merge it into the JSON
   written to `week{N}_metadata.json`.
4. Update `src/run_graph_pipeline.py` to thread `max_date_submitted` from
   `build_edge_tables()` through to `materialize_graph_artifacts()`.
5. Read `src/graph_validation.py:run_validation()` — find the temporal compliance section
   (lines ~205–270). Add a block that reads `max_date_submitted` from
   `meta.get("max_date_submitted")` and adds it to `temporal["checks"]`.
6. Read `src/graph_validation.py:_build_summary()` — confirm the temporal section
   already iterates `temporal["checks"]` generically, so no changes needed there.
7. Re-run `python src/run_graph_pipeline.py --week 8` and confirm the new check appears
   in `results/graph/validation/week08_validation_summary.txt`.
8. Update `docs/validation_report_week8.md` to add the new row to the Temporal
   Compliance table.

**Relevant Context**
- `src/graph_pipeline.py:303-312` — where `submitted` edge is built
- `src/graph_pipeline.py:466-600` — `materialize_graph_artifacts()` and metadata JSON writing
- `src/graph_validation.py:205-270` — temporal compliance check section
- `src/graph_validation.py:455-471` — `_build_summary()` temporal section
- `results/graph/artifacts/week08_metadata.json` — current metadata JSON structure

---

### Sub-Task 3: Add unit tests for submissions at, before, and after the cutoff

**Status**: `[ ] pending`

**Intent**
There are currently no tests for the temporal filtering logic in `filter_window()`.
The existing test file (`tests/test_splits.py`) covers only the split utilities.
This sub-task adds a new `tests/test_filter_window.py` with boundary tests for the
submission-date guard (Strategy B, Guard 2): submissions at exactly the cutoff, one day
before, and one day after the cutoff — ensuring the guard is tight.

**Expected Outcomes**
- `tests/test_filter_window.py` created with a `TestFilterWindow` class.
- Tests use synthetic DataFrames (no real data files required).
- Tests cover:
  - Submission exactly at cutoff (`date_submitted == window`) → included
  - Submission one day before cutoff (`date_submitted == window - 1`) → included
  - Submission one day after cutoff (`date_submitted == window + 1`) → excluded
  - Due-date guard (Guard 1): assessment due after window → excluded regardless of
    submission date
  - VLE boundary: interaction at exactly `window` → included; at `window + 1` →
    excluded
- `pytest tests/test_filter_window.py -v` passes all new tests.
- Existing `pytest tests/test_splits.py -v` still passes 13/13.

**Todo List**
1. Read `src/oulad_data.py:107-158` (`filter_window()`) to confirm the exact filter
   conditions and the column names expected in each input DataFrame.
2. Create `tests/test_filter_window.py` with the `sys.path` setup pattern matching
   `tests/test_splits.py`.
3. Write a `@pytest.fixture` that builds minimal synthetic DataFrames for `vle`,
   `assess` (studentAssessment), and `assessments` (metadata with due dates).
4. Write `TestFilterWindow` class with test methods for each boundary case listed above.
5. Run `pytest tests/ -v` and confirm all new tests pass alongside existing 13.

**Relevant Context**
- `src/oulad_data.py:107-158` — `filter_window()` implementation (both guards)
- `tests/test_splits.py:1-20` — `sys.path` setup and fixture patterns to mirror
- The `assess` fixture must include columns: `id_student`, `id_assessment`,
  `date_submitted`, `score`, `code_module`, `code_presentation`
- The `assessments` fixture must include: `id_assessment`, `code_module`,
  `code_presentation`, `date` (due date)
- The `vle` fixture must include: `id_student`, `id_site`, `code_module`,
  `code_presentation`, `date`, `sum_click`

---

### Sub-Task 4: Record pre-imputation missing value counts in graph artifacts

**Status**: `[ ] pending`

**Intent**
The pipeline already prints pre-imputation null counts to stdout and asserts 0 post-imputation,
but these counts are never persisted to a file. The validation summary currently reads zero
for all fields because `graph_validation.py` reads the *already-imputed* saved artifacts.
The actual pre-imputation counts (971 for `imd_band`, 5,243 for `week_from`/`week_to`)
are only in code comments.

This sub-task records the actual pre-imputation null counts in `week{N}_metadata.json`
(written during pipeline construction) and has `graph_validation.py` display them in
the "pre-imputation audit" section of the validation summary, replacing the current
hardcoded expected-null lookup.

**Expected Outcomes**
- `src/graph_pipeline.py:build_node_tables()` — the imputation loop collects
  `{artifact: {column: count}}` into a `pre_imputation_nulls` dict and returns it
  alongside `nodes` (or passes it to `materialize_graph_artifacts()` via `extra_metadata`).
- `src/graph_pipeline.py:materialize_graph_artifacts()` — writes
  `pre_imputation_nulls` into `week{N}_metadata.json`.
- `src/graph_validation.py:run_validation()` — reads `pre_imputation_nulls` from
  metadata JSON and uses it to populate the `data_quality` pre-imputation section
  instead of reading the post-imputation saved artifacts.
- `src/graph_validation.py:_build_summary()` — the "pre-imputation audit" section
  shows the actual source counts from metadata JSON.
- After re-running `python src/run_graph_pipeline.py --week 8`, the validation summary
  shows:
  ```
  nodes_student.imd_band          971  [expected, resolved by imputation]
  nodes_vle_resource.week_from  5,243  [expected, resolved by imputation]
  nodes_vle_resource.week_to    5,243  [expected, resolved by imputation]
  ```
- Post-imputation confirmation still shows 0 for all node types (read from artifacts).

**Todo List**
1. Read `src/graph_pipeline.py:209-227` — the imputation loop that already computes
   `pre_total` per node type. Extend it to accumulate per-column counts.
2. Collect results into `pre_imputation_nulls: dict` keyed as
   `{node_type: {column_name: count}}` (only non-zero columns recorded).
3. Update `build_node_tables()` to return `(nodes, pre_imputation_nulls)` instead of
   just `nodes`. Update the caller in `src/run_graph_pipeline.py` accordingly.
4. Pass `pre_imputation_nulls` to `materialize_graph_artifacts()` via `extra_metadata`
   (introduced in Sub-Task 2) and confirm it is written to `week{N}_metadata.json`.
5. Read `src/graph_validation.py:385-415` — the pre-imputation audit section.
   Replace the hardcoded `_expected_nulls` lookup with a read from
   `meta.get("pre_imputation_nulls", {})`.
6. Re-run `python src/run_graph_pipeline.py --week 8` and confirm the summary shows
   the real counts.
7. Update `docs/validation_report_week8.md` Null Handling table with the actual counts.

**Relevant Context**
- `src/graph_pipeline.py:186-227` — imputation loop in `build_node_tables()`
- `src/graph_validation.py:385-415` — pre-imputation audit section
- `results/graph/artifacts/week08_metadata.json` — metadata JSON (will gain new key)
- `docs/validation_report_week8.md:59-66` — Null Handling table (currently manual)
- Sub-Task 2 introduces `extra_metadata` in `materialize_graph_artifacts()` — coordinate
  implementation so both sub-tasks use the same mechanism

---

### Sub-Task 5: Move `num_of_prev_attempts` and `studied_credits` to `enrolled_in` edge

**Status**: `[ ] pending`

**Intent**
Both `num_of_prev_attempts` and `studied_credits` are recorded once per
(id_student, code_module, code_presentation) in `studentInfo.csv`. They are
enrollment-scoped, not student-scoped. The current pipeline calls
`drop_duplicates("id_student")` when building the student node, which silently
picks an arbitrary row for students enrolled in multiple courses — discarding the
correct values for all other enrollments.

Moving these two columns to the `enrolled_in` edge (student → course_presentation)
preserves the correct per-enrollment values and keeps the student node features
truly student-level.

**Expected Outcomes**
- `src/graph_pipeline.py:build_node_tables()` — `num_of_prev_attempts` and
  `studied_credits` are removed from `student_cols`.
- `src/graph_pipeline.py:build_edge_tables()` — the `enrolled_in` edge gains two
  new attribute columns: `num_of_prev_attempts` (int) and `studied_credits` (int).
  These are joined from `filtered["student_info"]` on
  `(id_student, code_module, code_presentation)`.
- The `enrolled_in` edge DataFrame changes from `[src, dst]` to
  `[src, dst, num_of_prev_attempts, studied_credits]`.
- `docs/GRAPH_SCHEMA.md` — `student` node features updated (two columns removed);
  `enrolled_in` edge updated to show the two new attribute columns.
- `README.md` — Graph Schema table updated for `student` node features and
  `enrolled_in` edge features.
- `docs/validation_report_week8.md` — Edge Counts and node feature description
  updated to reflect the new schema.
- `pytest tests/ -v` still passes all tests (no existing tests cover this schema).
- Re-running `python src/run_graph_pipeline.py --week 8` produces correct artifacts
  with `num_of_prev_attempts` and `studied_credits` on the edge.

**Todo List**
1. Read `src/graph_pipeline.py:144-158` (`student_cols` definition). Remove
   `"num_of_prev_attempts"` and `"studied_credits"` from the list.
2. Read `src/graph_pipeline.py:270-279` (`enrolled_in` edge construction). After
   creating the `ei` DataFrame, join `filtered["student_info"][["id_student",
   "code_module", "code_presentation", "num_of_prev_attempts", "studied_credits"]]`
   on the three key columns and add the two columns to the edge.
3. Update the `edges["enrolled_in"]` assignment to include the new columns:
   `ei[["src", "dst", "num_of_prev_attempts", "studied_credits"]]`.
4. Read `docs/GRAPH_SCHEMA.md` and update the student node features table and the
   enrolled_in edge features table.
5. Update `README.md` Graph Schema table (student node `features` column and
   enrolled_in edge `features` column).
6. Update `docs/validation_report_week8.md` if it lists specific student node columns.
7. Re-run `python src/run_graph_pipeline.py --week 8` and verify the saved
   `week08_edges_enrolled_in.parquet` has the two new columns.
8. Run `pytest tests/ -v` to confirm all tests still pass.

**Relevant Context**
- `src/graph_pipeline.py:144-158` — `student_cols` in `build_node_tables()`
- `src/graph_pipeline.py:270-279` — `enrolled_in` edge construction in `build_edge_tables()`
- `docs/GRAPH_SCHEMA.md` — schema reference document
- `README.md:154-167` — Graph Schema tables in README
- `studentInfo.csv` columns: `id_student, code_module, code_presentation,
  gender, region, highest_education, imd_band, age_band, num_of_prev_attempts,
  studied_credits, disability, final_result` (per-enrollment, not per-student)

---

### Sub-Task 6: Strategy A vs B comparison CSV and standardize std to population (ddof=0)

**Status**: `[ ] pending`

**Intent**
**Part A — Strategy comparison CSV**: The AUROC delta between Strategy A (due-date
only) and Strategy B (dual guard) is currently documented only in a code comment.
A reproducible `results/comparison/strategy_a_vs_b_comparison.csv` should capture
the full comparison across all 5 models × 4 prediction weeks (20 rows).

This runs as a **separate one-off script** (`src/run_strategy_comparison.py`) — it
is not invoked by the main `run_evaluation.py` runner.

The cleanest approach: add a `submission_date_guard: bool = True` parameter to
`filter_window()`, defaulting to True (no behaviour change to existing callers), and
pass `False` when running Strategy A.

**Part B — Std convention**: All comparison CSVs and the `random_vs_lcpo_comparison.csv`
report mean±std but never specify whether std is population or sample. Switch all
aggregation calls that compute std from ddof=1 (pandas default) to ddof=0 (population
std, numpy default convention) by passing `ddof=0` explicitly. Update README and any
CSV metadata comments.

**Expected Outcomes**
- `src/oulad_data.py:filter_window()` — new optional `submission_date_guard: bool = True`
  parameter. Guard 2 is applied only when `True`.
- `src/evaluation_pipeline.py` — new `run_strategy_comparison(data_dir=None)` function
  that runs evaluation twice (with and without Guard 2) and returns a merged DataFrame
  with columns: `Week, Model, Strategy_A_AUROC_mean, Strategy_A_AUROC_std,
  Strategy_B_AUROC_mean, Strategy_B_AUROC_std, Delta_AUROC_mean, Rows_Dropped,
  Rows_Dropped_Pct`
- New `src/run_strategy_comparison.py` — standalone CLI script that calls
  `run_strategy_comparison()` and saves the result to
  `results/comparison/strategy_a_vs_b_comparison.csv`. Not called by
  `run_evaluation.py`.
- All existing std aggregations across the codebase use `ddof=0` explicitly (population std).
  The `random_vs_lcpo_comparison.csv` header or a README footnote states
  "std = population std (ddof=0)".
- `README.md` — Key Design Decisions table gains a "Std convention" row:
  "Population std (ddof=0)".

**Todo List**
1. Read `src/oulad_data.py:107-158` — add `submission_date_guard: bool = True` param
   to `filter_window()`. Wrap Guard 2 in `if submission_date_guard:`.
2. Read `src/evaluation_pipeline.py` — identify where `filter_window()` is called
   (indirectly via `create_datasets()` in `oulad_data.py`). Trace how to pass
   the new parameter through `create_datasets()`.
3. Add `submission_date_guard: bool = True` parameter to `create_datasets()` in
   `src/oulad_data.py` and forward it to `filter_window()`.
4. Write `run_strategy_comparison(data_dir=None)` in `src/evaluation_pipeline.py`:
   - Loads datasets twice: once with `submission_date_guard=True` (Strategy B) and
     once with `submission_date_guard=False` (Strategy A)
   - Runs `run_random_student_evaluation()` for each strategy
   - Merges results into a 20-row DataFrame and computes delta and rows-dropped counts
5. Create `src/run_strategy_comparison.py` as a standalone CLI script with a `main()`
   that calls `run_strategy_comparison()` and saves the CSV. Add a `if __name__ == "__main__":`
   guard. Do not wire it into `run_evaluation.py`.
6. Search for all `.std()` calls in `src/evaluation_pipeline.py` and `src/oulad_data.py`.
   Change to `.std(ddof=0)` (or `np.std(..., ddof=0)` where numpy is used).
7. Add a `README.md` Key Design Decisions row for std convention.
8. Run `python src/run_strategy_comparison.py` and confirm
   `results/comparison/strategy_a_vs_b_comparison.csv` is saved with 20 data rows.

**Relevant Context**
- `src/oulad_data.py:107-158` — `filter_window()` (Guard 2 is line 156)
- `src/oulad_data.py:161-190` — `build_features()` and `create_datasets()` (caller chain)
- `src/evaluation_pipeline.py:486-560` — `analyze_course_difficulty()` (pattern to mirror
  for structuring `run_strategy_comparison()`)
- `src/run_evaluation.py` — reference for CLI script structure to mirror in
  `run_strategy_comparison.py`
- `results/comparison/all_splits_comparison.csv` — existing comparison CSV (columns to mirror)
- `results/lcpo/random_vs_lcpo_comparison.csv` — uses ± notation (std convention to clarify)

---

### Sub-Task 7: Report course difficulty separately by prediction week and model

**Status**: `[ ] pending`

**Intent**
`analyze_course_difficulty()` in `src/evaluation_pipeline.py` aggregates AUROC across
all models AND all prediction weeks into a single mean per course-presentation. This
hides the fact that course difficulty rankings shift between early (Week 2) and late
(Week 8) windows, and that some models disagree more than others on which courses are
hard. A long-format CSV (`course_difficulty_by_week_model.csv`) with 440 rows
(22 courses × 4 weeks × 5 models) enables downstream filtering and plotting.

The existing `course_presentation_difficulty.csv` (aggregated) is retained unchanged
for backward compatibility.

**Expected Outcomes**
- `src/evaluation_pipeline.py:analyze_course_difficulty()` — saves an additional
  long-format CSV to `results/lcpo/course_difficulty_by_week_model.csv` with columns:
  `Course_Presentation, Week, Model, AUROC`
- The existing `course_presentation_difficulty.csv` (aggregated) is still saved as before.
- `src/run_evaluation.py` (or wherever `analyze_course_difficulty()` is called) does
  not need changes — the new CSV is a side effect of the same function call.
- `docs/EVALUATION_SPLITS.md` — "Course Difficulty" section updated to mention both
  output files and explain the difference.
- `README.md` Results section updated to reference the new CSV.

**Todo List**
1. Read `src/evaluation_pipeline.py:486-560` — `analyze_course_difficulty()` in full
   to understand the input `lcpo_df` columns. Confirm it contains `Test_Module`,
   `Test_Presentation`, `AUROC`, and the `Week` and `Model` columns needed for the
   long-format output.
2. Read `src/run_evaluation.py` — confirm how `analyze_course_difficulty()` is called
   and what `lcpo_df` looks like at call time.
3. Inside `analyze_course_difficulty()`, after saving the aggregated CSV, add:
   ```python
   # Long-format: one row per (course_presentation, week, model)
   long_df = lcpo_df[["Course_Presentation", "Week", "Model", "AUROC"]].copy()
   long_csv = output_dir / "course_difficulty_by_week_model.csv"
   long_df.to_csv(long_csv, index=False)
   ```
4. Read `docs/EVALUATION_SPLITS.md` — find the "Course Difficulty" section. Add a
   paragraph describing the new long-format CSV and when to use each file.
5. Update `README.md` Results section to reference `course_difficulty_by_week_model.csv`.
6. Run `python src/run_evaluation.py` (or the relevant portion) and confirm both CSVs
   are saved to `results/lcpo/`.

**Relevant Context**
- `src/evaluation_pipeline.py:486-560` — full `analyze_course_difficulty()` function
- `src/run_evaluation.py` — entry point that calls `analyze_course_difficulty()`
- `results/lcpo/course_presentation_difficulty.csv` — existing aggregated output
- `docs/EVALUATION_SPLITS.md` — course difficulty documentation section
- The `lcpo_df` DataFrame returned by `run_lcpo_evaluation()` contains:
  `Week, Model, Feature_Subset, Test_Module, Test_Presentation, AUROC, F1, ...` columns

---

## Summary of Changes per File

| File | Sub-Tasks | Change Type |
|------|-----------|-------------|
| `docs/LEAKAGE_PREVENTION.md` | 1 | Documentation update |
| `README.md` | 1, 5, 6, 7 | Documentation update |
| `docs/validation_report_week8.md` | 1, 2, 4 | Documentation update |
| `docs/GRAPH_SCHEMA.md` | 5 | Documentation update |
| `docs/EVALUATION_SPLITS.md` | 7 | Documentation update |
| `src/graph_pipeline.py` | 1, 2, 4, 5 | Code change |
| `src/graph_validation.py` | 2, 4 | Code change |
| `src/run_graph_pipeline.py` | 2, 4 | Code change (threading new metadata) |
| `src/oulad_data.py` | 3, 6 | Code change |
| `src/evaluation_pipeline.py` | 6, 7 | Code change |
| `src/run_strategy_comparison.py` | 6 | New standalone CLI script |
| `tests/test_filter_window.py` | 3 | New test file |
| `results/comparison/strategy_a_vs_b_comparison.csv` | 6 | New generated artifact |
| `results/lcpo/course_difficulty_by_week_model.csv` | 7 | New generated artifact |

## Execution Order

Sub-tasks can be executed in this order to minimize rework:

1. **Sub-Task 1** — pure documentation, zero code risk
2. **Sub-Task 3** — new test file, pure addition, no pipeline dependency
3. **Sub-Tasks 2 & 4** — implemented in one pass (confirmed); both modify
   `build_node_tables()`, `materialize_graph_artifacts()`, and `graph_validation.py`
   via the shared `extra_metadata` mechanism. Re-run `run_graph_pipeline.py --week 8`
   once at the end of this combined pass.
4. **Sub-Task 5** — graph schema change; re-run `run_graph_pipeline.py --week 8` after.
5. **Sub-Task 6** — adds `submission_date_guard` param and the standalone
   `run_strategy_comparison.py` script; run it separately to generate the CSV.
   Also updates all std calls to `ddof=0` across the evaluation pipeline.
6. **Sub-Task 7** — smallest eval change; re-run `run_evaluation.py` (or the LCPO
   portion) to regenerate `course_difficulty_by_week_model.csv`.
