# Graph Finalization Plan

## Top-Level Overview

**Goal**: Complete the graph pipeline from a Week-8-only prototype to a
fully documented, multi-week, reproducible system with per-week split
definitions and a corrected canonical notebook.

**Current state**:
- Week 8 graph artifacts exist and are validated.
- Weeks 2, 4, 6 have no artifacts, metadata, validation reports, or split files.
- `course_difficulty_by_week_model.csv` has not been generated yet (requires a
  full `run_evaluation.py` re-run after Sub-Task 6 changes in the prior session).
- The canonical notebook has stale markdown (old filtering description, old
  validation output text in embedded results).
- `GRAPH_EVALUATION_DIR` (`results/graph/evaluation/`) is empty.
- No split definition files exist anywhere.

**Scope**: 5 sub-tasks. Sub-Tasks 1 and 2 are prerequisites for 3 and 4.
Sub-Task 5 is independent and can run in parallel with 1–4.

**Not in scope**: GNN training, model evaluation, tabular baseline re-runs,
changes to the tabular evaluation pipeline.

---

## Sub-Tasks

---

### Sub-Task 1: Finalize and document the graph schema

**Status**: `[x] complete`

**Intent**
The current `docs/GRAPH_SCHEMA.md` still contains aspirational features that
are not implemented (e.g. `num_students`, `pass_rate`, `avg_vle_clicks` on
course-presentation nodes; `date_normalized`, `week_due`, `is_early`, `is_final`
on assessment nodes; `avg_clicks`, `total_clicks` on VLE resource nodes).
The implemented schema must be documented precisely, including the enrollment-
specific attributes now on `enrolled_in` edges, the temporal filtering rules
per edge type, the target definition, and the null imputation strategy.

This is a documentation-only sub-task.

**Expected Outcomes**
- `docs/GRAPH_SCHEMA.md` accurately reflects what `build_node_tables()` and
  `build_edge_tables()` actually produce — no aspirational columns.
- Each node type section lists exactly the columns in the saved parquet files,
  with types, source table, and null handling noted.
- Each edge type section lists columns, source table, enrollment-scope note
  where applicable, and temporal filtering rule.
- Target definition section: `target = 1` (Fail/Withdrawn), `target = 0`
  (Pass/Distinction), attached to the enrollment supervision table
  (`week{N}_enrollments.parquet`), not to the student node.
- Null imputation section: numeric → 0, categorical → "Unknown"; expected sources
  (student.imd_band = 971, vle_resource.week_from/week_to = 5243 each).
- Ground-truth column names sourced from `week08_metadata.json` (node_schema,
  edge_schema keys).

**Todo List**
1. Read `results/graph/artifacts/week08_metadata.json` fully — use `node_schema`
   and `edge_schema` as the authoritative column lists.
2. Read the current `docs/GRAPH_SCHEMA.md` fully and identify every section that
   describes a column not present in the metadata JSON.
3. Rewrite the Node Types section to match exactly: student (6 feature cols + 
   node_idx), course_presentation (3 cols + node_idx), assessment (5 cols +
   node_idx), vle_resource (5 cols + node_idx).
4. Rewrite the Edge Types section: enrolled_in (src, dst, num_of_prev_attempts,
   studied_credits), contains_assess (src, dst), has_resource (src, dst),
   submitted (src, dst, score), interacted_with (src, dst, total_clicks,
   n_interactions, first_day, last_day, active_days).
5. Add a "Target and Supervised Unit" section: enrollment-level supervision,
   binary label, class distribution (52.8% at-risk).
6. Add a "Temporal Filtering Rules" section describing the dual guard for
   assessment submissions and VLE interaction date filter.
7. Add a "Null Imputation" section.
8. Remove or clearly mark as "not-yet-implemented" any aspirational features
   that are not in the current parquet files (derived features, stats).
9. Update the "Graph Statistics" section with actual Week 8 counts.

**Relevant Context**
- `results/graph/artifacts/week08_metadata.json` — authoritative column lists
- `src/graph_pipeline.py:build_node_tables()` (lines 125–230)
- `src/graph_pipeline.py:build_edge_tables()` (lines 234–350)
- `docs/GRAPH_SCHEMA.md` — current document (needs rewrite of node/edge sections)
- `README.md:149–167` — Graph Schema tables (already updated in prior session;
  verify these are consistent after GRAPH_SCHEMA.md is updated)

---

### Sub-Task 2: Generate graph datasets for Weeks 2, 4, 6, and 8

**Status**: `[x] complete`

**Intent**
Run `run_graph_pipeline.py` for Weeks 2, 4, and 6 to produce artifacts,
metadata JSON, integrity JSON, and validation reports for all four prediction
windows. Week 8 already exists and does not need to be re-run.

Each week uses the same leakage-safe rules:
- VLE interactions: `date <= window_days`
- Assessment submissions: `due_date <= window_days` AND `date_submitted <=
  window_days` (Strategy B, dual guard)
- Null imputation: numeric → 0, categorical → "Unknown"

Expected artifact counts will differ across weeks because fewer assessments
are due before earlier cutoffs, and fewer VLE interactions and submissions fall
within shorter windows.

**Expected Outcomes**
- `results/graph/artifacts/week02_*.parquet` (10 files) and `week02_metadata.json`
- `results/graph/artifacts/week04_*.parquet` (10 files) and `week04_metadata.json`
- `results/graph/artifacts/week06_*.parquet` (10 files) and `week06_metadata.json`
- `results/graph/validation/week02_integrity.json`, `week02_validation.json`,
  `week02_validation_summary.txt` (same for 04 and 06)
- All 6 new validation summaries show: zero duplicate nodes/edges, zero dangling
  edges, zero post-imputation nulls, overall temporal compliance ✓,
  `submitted_max_date_submitted` ≤ window_days ✓.
- Each metadata JSON includes `max_date_submitted` and `pre_imputation_nulls`.

**Todo List**
1. Run `python src/run_graph_pipeline.py --week 2` and confirm completion.
2. Run `python src/run_graph_pipeline.py --week 4` and confirm completion.
3. Run `python src/run_graph_pipeline.py --week 6` and confirm completion.
4. Verify each week produced exactly 10 parquet artifacts, a metadata JSON,
   an integrity JSON, a validation JSON, and a validation summary text file.
5. Spot-check that node counts are consistent across weeks (student, course_
   presentation, and vle_resource counts should be the same for all weeks;
   assessment counts should be ≤ Week 8's 40 for earlier weeks).
6. Spot-check that `max_date_submitted` ≤ window_days in each week's metadata.
7. Run `pytest tests/ -q` to confirm 24/24 tests still pass.

**Relevant Context**
- `src/run_graph_pipeline.py` — CLI entry point (accepts `--week 2|4|6|8`)
- `src/config.py:PREDICTION_WINDOWS` — 14/28/42/56 days for weeks 2/4/6/8
- `results/graph/artifacts/week08_metadata.json` — Week 8 reference for
  expected schema and counts
- All 4 weeks share the same node schema (student, course_presentation,
  vle_resource are not time-filtered); only assessment, submitted, and
  interacted_with edges shrink in earlier windows.

---

### Sub-Task 3: Produce a multi-week graph statistics summary

**Status**: `[x] complete`

**Intent**
After all 4 weeks are built (Sub-Task 2), produce a single comparison table
that shows how node/edge counts, label distribution, runtime, and temporal
compliance vary across prediction windows. This table becomes the primary
reference for understanding graph scale at each window.

Save it as `results/graph/validation/all_weeks_summary.csv` (machine-readable)
and update `docs/validation_report_week8.md` to become
`docs/graph_validation_summary.md` (a multi-week report).

**Expected Outcomes**
- `results/graph/validation/all_weeks_summary.csv` with columns:
  `Week, Window_Days, N_students, N_course_presentations, N_assessments,
   N_vle_resources, N_enrolled_in, N_contains_assess, N_has_resource,
   N_submitted, N_interacted_with, N_enrollments, At_risk_count,
   At_risk_rate, Max_date_submitted, Pre_null_imd_band,
   Pre_null_week_from, Pre_null_week_to, Runtime_s, Peak_memory_MB,
   All_temporal_compliant`
- New file `docs/graph_validation_summary.md` covering all 4 weeks with a
  table of the above metrics. The existing `docs/validation_report_week8.md`
  is retained as a week-specific audit trail.
- A new helper script `src/summarize_graph_weeks.py` reads the 4 metadata
  JSONs and 4 validation JSONs and writes the CSV + markdown table. This is
  a short standalone script (no new imports beyond json/pathlib/csv).

**Todo List**
1. Confirm all 4 metadata JSONs and 4 validation JSONs exist (Sub-Task 2
   must be complete first).
2. Create `src/summarize_graph_weeks.py`:
   - Reads `week{N}_metadata.json` for each week N in [2, 4, 6, 8].
   - Reads `week{N}_validation.json` for each week.
   - Extracts the fields listed in Expected Outcomes above.
   - Writes `all_weeks_summary.csv`.
   - Prints a formatted ASCII table showing the comparison.
3. Run `python src/summarize_graph_weeks.py` and verify the CSV is saved.
4. Create `docs/graph_validation_summary.md`:
   - Header: project context, pipeline version, temporal filtering rule.
   - Per-week section for each of the 4 weeks (can reuse the structure from
     `docs/validation_report_week8.md`).
   - Summary comparison table using the CSV data.
5. Update `README.md` to reference both `docs/validation_report_week8.md`
   (for deep Week 8 detail) and the new `docs/graph_validation_summary.md`
   (for multi-week overview).

**Relevant Context**
- `results/graph/artifacts/week{N}_metadata.json` — per-week metadata
- `results/graph/validation/week{N}_validation.json` — per-week full report
- `docs/validation_report_week8.md` — structure to mirror for the new doc
- `src/config.py:GRAPH_ARTIFACTS_DIR`, `GRAPH_VALIDATION_DIR` — paths

---

### Sub-Task 4: Save per-week split definitions

**Status**: `[x] complete`

**Intent**
GNN training will need reproducible splits that can be loaded from disk
without re-running the full split-generation logic. Since the enrollment
supervision table is the same for all weeks (it derives from studentInfo,
not from time-filtered data), each week's split directory will contain:

1. A copy of the week's enrollment supervision table (parquet) — identical
   content across weeks, but per-week copies make each week self-contained.
2. A `random_student_split.parquet` — the enrollment table with three boolean
   columns added: `is_train`, `is_val`, `is_test` (seed=42, val_frac=0.1,
   test_frac=0.2).
3. A `lcpo_folds.csv` — one row per LCPO fold with columns:
   `fold_idx, held_out_module, held_out_presentation, n_train, n_test`.
4. A `splits_config.json` documenting the split parameters:
   `random_seed, val_frac, test_frac, n_random_folds=1,
    n_lcpo_folds=22, future_train_presentations, future_test_presentations`.

All files land under `results/graph/evaluation/week{N}/splits/`.

**Expected Outcomes**
- `results/graph/evaluation/week02/splits/` — 4 files as described above
- `results/graph/evaluation/week04/splits/` — 4 files
- `results/graph/evaluation/week06/splits/` — 4 files
- `results/graph/evaluation/week08/splits/` — 4 files
- A new `src/save_graph_splits.py` script that:
  - Accepts `--week` argument (default: all four weeks).
  - Loads enrollments from `week{N}_enrollments.parquet`.
  - Calls `random_student_split(enrollments, val_frac=0.1, test_frac=0.2, seed=42)`.
  - Iterates all 22 course-presentations and calls `lcpo_split()` for each.
  - Saves the 4 output files per week.
  - Prints a summary: total enrollments, split sizes, LCPO fold sizes.
- All saved boolean masks are verifiable: `is_train | is_val | is_test` == True
  for every row, no student appears in more than one partition.

**Todo List**
1. Read `src/oulad_data.py` `random_student_split()` and `lcpo_split()` to
   confirm exact signatures and return types.
2. Create `src/save_graph_splits.py`:
   - For each week in [2, 4, 6, 8] (or `--week N` if specified):
     a. Load `week{N}_enrollments.parquet` from `GRAPH_ARTIFACTS_DIR`.
     b. Apply `random_student_split(enrollments, val_frac=0.1,
        test_frac=0.2, seed=42)` → three boolean masks.
     c. Add `is_train`, `is_val`, `is_test` columns to the enrollment table.
     d. Save as `week{N}_random_split.parquet` in the split dir.
     e. Iterate all unique (code_module, code_presentation) pairs; call
        `lcpo_split(enrollments, module, pres)` for each; collect
        (fold_idx, module, pres, n_train, n_test) rows.
     f. Save `lcpo_folds.csv`.
     g. Save `splits_config.json` with the documented parameters.
3. Run `python src/save_graph_splits.py` and confirm all 16 files are created
   (4 per week × 4 weeks).
4. Spot-check Week 8 splits:
   - `is_train | is_val | is_test` is True for all rows.
   - No student appears in more than one partition.
   - `lcpo_folds.csv` has exactly 22 rows, one per course-presentation.
5. Update `QUICK_START.md` to include the step:
   `python src/save_graph_splits.py` after building all graph weeks.

**Relevant Context**
- `src/oulad_data.py` — `random_student_split()`, `lcpo_split()`
- `results/graph/artifacts/week{N}_enrollments.parquet` — source (after Sub-Task 2)
- `src/config.py:GRAPH_EVALUATION_DIR` — `results/graph/evaluation/`
- `src/run_evaluation.py:_TRAIN_PRESENTATIONS`, `_TEST_PRESENTATIONS` — for
  `future_train_presentations` and `future_test_presentations` in splits_config.json

---

### Sub-Task 5: Update canonical notebook and reproduction instructions

**Status**: `[x] complete`

**Intent**
The canonical notebook has two types of stale content:
1. Markdown cells describing the old due-date-only filtering.
2. Embedded code-output cells showing the old validation summary (pre-imputation
   audit section shows all zeros instead of real counts).

Update the stale markdown cells to reflect the dual-guard (Strategy B) and
the multi-week graph pipeline. Clear the stale embedded outputs in code cells
that read and print the validation summary — they will be refreshed by
re-execution after the user builds all 4 weeks. Add a note directing users
to re-run after generating all week artifacts.

Also update `QUICK_START.md`:
- Add steps for building all 4 weeks (`--week 2`, `--week 4`, `--week 6`).
- Add the `save_graph_splits.py` step.
- Add reference to `docs/graph_validation_summary.md` (created in Sub-Task 3).
- Add a note that the notebook should be re-executed after all 4 weeks are built.

**Expected Outcomes**
- `notebooks/OULAD_Graph_Analysis_Final.ipynb`:
  - Cell 1 markdown (Section 1 header): "Assessment filtering uses **dual guard
    (Strategy B)**: due date ≤ 56 **and** `date_submitted` ≤ 56."
  - Cell 3 (validation summary cell): output cleared; cell source remains
    unchanged (it reads from the saved txt file).
  - A new markdown cell added after the validation summary cell explaining that
    outputs were cleared pending re-execution with all 4 weeks built.
- `QUICK_START.md`:
  - Graph pipeline section expanded to cover all 4 weeks with a loop or 4
    explicit commands.
  - New step: `python src/save_graph_splits.py` for split definitions.
  - New step: `python src/summarize_graph_weeks.py` for multi-week comparison.
  - Reference to `docs/graph_validation_summary.md`.
  - Note: re-execute the notebook after all steps complete.

**Todo List**
1. Read the notebook cell at index 3 (markdown, Section 1 header) — identify
   the exact stale string "not submission date".
2. Update that markdown cell source to: "Assessment filtering uses **dual guard
   (Strategy B)**: due date ≤ window **and** `date_submitted` ≤ window."
3. Find the validation summary code cell (the one that reads
   `week08_validation_summary.txt`) and clear its `outputs` array in the
   notebook JSON.
4. Insert a new markdown cell immediately after that code cell explaining that
   outputs were cleared and directing the user to re-run after building all 4
   weeks.
5. Read `QUICK_START.md` lines 107–130 (the graph pipeline section).
6. Expand the graph pipeline section to include:
   - Commands for weeks 2, 4, 6, 8.
   - `python src/save_graph_splits.py`
   - `python src/summarize_graph_weeks.py`
7. Add a reference to `docs/graph_validation_summary.md` in the Outputs section.
8. Add a note: "Re-execute the canonical notebook after all 4 weeks are built
   to refresh embedded outputs."

**Relevant Context**
- `notebooks/OULAD_Graph_Analysis_Final.ipynb` — Cell 3 markdown, Cell 4 code
  (validation summary), structure described above
- `QUICK_START.md:107–130` — current graph pipeline section
- `docs/graph_validation_summary.md` — created in Sub-Task 3 (coordinate)

---

## Summary of Changes per File

| File | Sub-Tasks | Change Type |
|------|-----------|-------------|
| `docs/GRAPH_SCHEMA.md` | 1 | Rewrite (remove aspirational, match parquet schema) |
| `docs/graph_validation_summary.md` | 3 | New document (multi-week validation summary) |
| `docs/validation_report_week8.md` | 3 | Unchanged (retained as week-specific audit trail) |
| `README.md` | 1, 3 | Add reference to graph_validation_summary.md |
| `src/summarize_graph_weeks.py` | 3 | New script |
| `src/save_graph_splits.py` | 4 | New script |
| `results/graph/artifacts/week02_*.parquet` + JSON | 2 | New generated artifacts |
| `results/graph/artifacts/week04_*.parquet` + JSON | 2 | New generated artifacts |
| `results/graph/artifacts/week06_*.parquet` + JSON | 2 | New generated artifacts |
| `results/graph/validation/week02_*` through `week06_*` | 2 | New generated artifacts |
| `results/graph/validation/all_weeks_summary.csv` | 3 | New generated artifact |
| `results/graph/evaluation/week{N}/splits/*` (16 files) | 4 | New generated artifacts |
| `notebooks/OULAD_Graph_Analysis_Final.ipynb` | 5 | Fix stale markdown, clear stale outputs |
| `QUICK_START.md` | 5 | Add multi-week build steps + new scripts |

## Execution Order

1. **Sub-Task 1** — documentation only, no dependencies
2. **Sub-Task 2** — generate weeks 2, 4, 6 (prerequisite for 3 and 4)
3. **Sub-Tasks 3 and 4** — can run in parallel after Sub-Task 2
4. **Sub-Task 5** — independent of 1–4; can run any time
