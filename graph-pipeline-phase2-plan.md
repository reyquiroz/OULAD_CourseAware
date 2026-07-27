# Graph Pipeline Phase 2 — Finalization Plan

## Overview

This plan addresses six deliverables required before the graph pipeline can be
considered finalized and ready for GNN training. The tasks are ordered by
dependency: schema documentation must be complete before the consolidated
statistics table is produced; the pipeline must be fully validated before the
reproducibility document is written.

**Scope**: documentation, validation, and verification only — no new graph
features, no model training, no architecture changes.

**Out of scope**: GNN training (next iteration), Canvas/IRB submissions
(tracked externally, task 7 is handled via institutional channels, not code).

---

## Sub-Task 1 — Finalize and document the heterogeneous graph schema

**Status**: `[ ] pending`

### Intent

`docs/GRAPH_SCHEMA.md` currently documents node types, edge types, column
names, null imputation, and temporal filtering rules. Several items are either
missing or described as "Planned Extensions" but never resolved:

- **Prediction cutoff semantics**: Days are defined relative to "course start"
  but the schema does not state what "course start" means in terms of the
  OULAD calendar or `studentRegistration.date_registration`.
- **Timestamp columns on edges**: `interacted_with` stores `first_day` /
  `last_day` (relative days); `submitted` stores nothing (submission date is
  dropped after filtering). This is correct but not explicitly justified in the
  schema document.
- **Label derivation edge cases**: The mapping Fail/Withdrawn → 1,
  Pass/Distinction → 0 is stated but the rationale (both Fail and Withdrawn
  indicate non-completion) and the implication that "Withdrawn" dominates the
  at-risk class are not documented.
- **Registration dates**: Marked as "Planned Extension" but never implemented.
  The schema must explicitly state why `date_unregistration` is excluded
  (it reveals the target — withdrawal is part of `target=1`) and why
  `date_registration` is not currently used.
- **`studentVle.csv` aggregation semantics**: The `interacted_with` edge
  aggregates raw per-day rows into one edge per (student, resource, enrollment).
  The grouping key and aggregated columns need to be explicitly stated.

### Expected Outcomes

- `docs/GRAPH_SCHEMA.md` contains a new **Prediction Cutoff Semantics** section
  explaining that day 0 = course start date, and all window days are measured
  from the first presentation date of that `(code_module, code_presentation)`.
- The **submitted edge** section explicitly states that `date_submitted` is
  used only as a filter guard (Strategy B) and is not stored in the artifact.
- The **interacted_with edge** section explicitly states the grouping key
  `(id_student, id_site, code_module, code_presentation)` and lists all five
  aggregated columns with their derivation formula.
- The **label derivation** section documents the Withdrawn ≡ Fail rationale and
  notes the 52.8% at-risk rate.
- The **studentRegistration exclusion** section documents why
  `date_unregistration` is a leakage risk (it reveals the target) and why
  `date_registration` is not currently included (no evidence of predictive
  value over existing enrollment-time features; can be added later without
  pipeline restructuring).
- All "Planned Extensions" entries that are definitively out of scope for this
  iteration are marked as such with a reason, not left open-ended.

### Todo List

1. Add **Prediction Cutoff Semantics** subsection to `docs/GRAPH_SCHEMA.md`:
   explain day-0 convention and note that OULAD does not provide absolute
   calendar dates for course start in the public dataset — day numbers are
   already relative in the source CSVs.
2. Update **submitted edge** section: add explicit note that `date_submitted`
   is dropped from the saved artifact; cite the dual-guard filter.
3. Update **interacted_with edge** section: add grouping key and per-column
   derivation (e.g., `total_clicks = sum(sum_click)`,
   `n_interactions = count of raw rows`, `first_day = min(date)`,
   `last_day = max(date)`, `active_days = nunique(date)`).
4. Add **Label Derivation** subsection: document the Withdrawn ≡ Fail decision
   with rationale, note 17,208 at-risk / 15,385 success / 32,593 total.
5. Add **Excluded Source Columns** subsection: document why `date_unregistration`
   is excluded (label leakage — withdrawal IS the target), why
   `date_registration` is not currently used, and update "Planned Extensions"
   to reflect current decisions.
6. Commit updated `docs/GRAPH_SCHEMA.md`.

### Relevant Context

- `docs/GRAPH_SCHEMA.md` — current schema reference
- `src/graph_pipeline.py` — `build_edge_tables()` lines 276–391 (edge construction)
- `src/graph_pipeline.py` — `build_enrollment_supervision()` (label derivation)
- `docs/LEAKAGE_PREVENTION.md` — already documents `date_unregistration` as
  a removed feature

---

## Sub-Task 2 — Regenerate and validate graph datasets for all four weeks

**Status**: `[ ] pending`

### Intent

The graph artifacts for all four prediction weeks already exist on disk (parquet
files, metadata JSON, validation reports). However, they were generated before
the schema correction in Sub-Task 2 of the previous plan (which moved `age_band`
from the student node to the `enrolled_in` edge). The artifacts on disk reflect
the corrected schema (they were regenerated as part of that fix), but the
`all_weeks_summary.csv` file in `results/graph/validation/` was not regenerated
— it still shows runtime figures from the pre-correction runs.

This sub-task re-runs the pipeline for all four weeks from a single `make`-style
command, verifies zero integrity failures across all weeks, and ensures every
committed validation file matches the current pipeline output.

### Expected Outcomes

- `python src/run_graph_pipeline.py --week 2`, `4`, `6`, `8` all complete with
  exit code 0 and print "All integrity checks: PASS".
- `results/graph/validation/all_weeks_summary.csv` is regenerated by
  `python src/summarize_graph_weeks.py` and matches the current artifact counts
  (submitted edges: 1,089 / 21,393 / 28,569 / 44,927).
- The four `week{N}_metadata.json` files committed to the repo reflect the
  correct enrolled_in edge schema: `['src', 'dst', 'age_band',
  'num_of_prev_attempts', 'studied_credits']`.
- All four `week{N}_validation_summary.txt` files show the same pass/fail
  summary and are committed.
- A `results/graph/validation/last_regenerated.txt` file is written containing
  the Python version, pipeline git commit hash, and timestamp of the last
  successful regeneration run (to make staleness detectable).

### Todo List

1. Run `python src/run_graph_pipeline.py --week 2` — confirm pass.
2. Run `python src/run_graph_pipeline.py --week 4` — confirm pass.
3. Run `python src/run_graph_pipeline.py --week 6` — confirm pass.
4. Run `python src/run_graph_pipeline.py --week 8` — confirm pass.
5. Run `python src/summarize_graph_weeks.py` — regenerate
   `results/graph/validation/all_weeks_summary.csv`.
6. Write `results/graph/validation/last_regenerated.txt` with Python version,
   git commit hash, and ISO timestamp.
7. Confirm all four metadata JSON files show the corrected enrolled_in schema
   (contains `age_band`).
8. Commit updated validation files and `last_regenerated.txt`.

### Relevant Context

- `src/run_graph_pipeline.py` — CLI entry point (`--week {2,4,6,8}`)
- `src/summarize_graph_weeks.py` — generates `all_weeks_summary.csv`
- `results/graph/artifacts/week{02,04,06,08}_metadata.json` — committed
- `results/graph/validation/` — validation reports

---

## Sub-Task 3 — Produce consolidated graph-statistics table

**Status**: `[ ] pending`

### Intent

`docs/graph_validation_summary.md` already contains per-week tables for node
counts, edge counts, label distribution, and runtime. However, it is spread
across multiple tables and does not include:
- Missingness (pre-imputation null counts) per week in the same view
- Temporal-boundary check results (max VLE date, max assessment due date,
  max date_submitted) in a compact summary row

The supervisor expects one consolidated table per prediction window. The existing
`all_weeks_summary.csv` already contains all the raw numbers; this sub-task
formats it into a single human-readable document.

### Expected Outcomes

- A new file `docs/graph_consolidated_stats.md` contains one section per
  prediction week (Weeks 2, 4, 6, 8), each with a single reference table
  covering: window (days), node counts (4 types), edge counts (5 types),
  enrollment count, label distribution (at-risk N and %), pre-imputation null
  counts (3 columns), temporal-boundary check (max VLE date, max assess
  due_date, max date_submitted vs. window), construction runtime, peak memory.
- A summary row at the top shows totals/ranges across all four weeks.
- Every number in the document is sourced from `all_weeks_summary.csv` or
  the `week{N}_metadata.json` files — no hand-typed values.
- `docs/graph_validation_summary.md` is updated to reference the new
  consolidated document rather than duplicating content.

### Todo List

1. Read `results/graph/validation/all_weeks_summary.csv` and all four
   `week{N}_metadata.json` files.
2. Generate `docs/graph_consolidated_stats.md` with one section per week
   (script or manual — numbers must match the CSV exactly).
3. Add a preamble explaining how to regenerate: `python src/summarize_graph_weeks.py`
   followed by the plan sub-task 2 pipeline runs.
4. Update `docs/graph_validation_summary.md`: add a one-line pointer to
   `docs/graph_consolidated_stats.md` at the top.
5. Commit both files.

### Relevant Context

- `results/graph/validation/all_weeks_summary.csv` — machine-readable source
- `results/graph/artifacts/week{N}_metadata.json` — per-week metadata
- `docs/graph_validation_summary.md` — existing multi-table document
- `docs/progress_report_tables.md` — Tables 2 and 3 contain partial versions
  of this content (reference, do not duplicate)

---

## Sub-Task 4 — Verify and document split definitions with overlap/leakage checks

**Status**: `[ ] pending`

### Intent

Three split strategies are implemented: random-student CV, LCPO, and
future-presentation. Each has a dedicated config JSON and CSV in
`results/graph/evaluation/week{N}/splits/`. However, no single document
currently:
- States the formal definition of each split (what is the unit? what is the
  guarantee? what leakage is prevented?)
- Provides a verification table showing that no overlap exists between train and
  test for any fold of any strategy
- Cross-references the unit tests that enforce these guarantees

This sub-task produces `docs/EVALUATION_SPLITS_VERIFICATION.md` as that
single document, and also verifies computationally that all four week split
files satisfy the overlap/leakage invariants.

### Expected Outcomes

- `docs/EVALUATION_SPLITS_VERIFICATION.md` contains:
  - Formal definition of each split strategy (supervised unit, split key,
    train/test boundary rule, leakage guarantee)
  - A table showing for each strategy × week: train N, test N, unique students
    in train, unique students in test, and confirmed overlap count (must be 0
    for random-student and LCPO)
  - A note that future-presentation overlap is intentionally non-zero at the
    student level (the same student may appear in both train and test across
    presentations) but is zero at the enrollment level within any single
    presentation
  - Cross-references to the unit tests in `tests/test_splits.py` that enforce
    each guarantee
- A verification script `src/verify_splits.py` that loads all 12 split files
  (3 strategies × 4 weeks), checks each one for the relevant invariant, and
  prints PASS/FAIL per combination — exit code 0 iff all pass.
- All 12 split files verified and PASS logged.

### Todo List

1. Write `src/verify_splits.py`:
   - Load `week{N}_random_split.parquet`, `week{N}_lcpo_folds.csv`,
     `week{N}_future_split.parquet` for weeks 2, 4, 6, 8.
   - For random-student: verify train/val/test student sets are disjoint.
   - For LCPO: for each fold, verify that held-out (module, presentation) does
     not appear in the train rows.
   - For future-presentation: verify that no 2014J enrollment appears in the
     train rows and no 2013B/J/2014B enrollment appears in the test rows.
   - Print PASS/FAIL per split and exit with code 0 if all pass.
2. Run `python src/verify_splits.py` — confirm all 12 pass.
3. Write `docs/EVALUATION_SPLITS_VERIFICATION.md` with formal definitions,
   verification table (sourced from running the script), and test cross-references.
4. Commit `src/verify_splits.py` and `docs/EVALUATION_SPLITS_VERIFICATION.md`.

### Relevant Context

- `results/graph/evaluation/week{N}/splits/` — 4 split files per week
- `src/oulad_data.py` — `random_student_split()`, `lcpo_split()` (verified
  implementations with in-function assertions)
- `tests/test_splits.py` — 13 unit tests covering overlap and coverage
- `docs/EVALUATION_SPLITS.md` — existing high-level description (do not
  duplicate; cross-reference)
- `results/graph/evaluation/week08/splits/week08_splits_config.json` —
  canonical split parameters (seed=42, val_frac=0.1, test_frac=0.2,
  LCPO 22 folds, future train=2013B/J/2014B test=2014J)

---

## Sub-Task 5 — Run the complete test suite and pipeline from a clean environment

**Status**: `[ ] pending`

### Intent

There is no committed document proving that a new collaborator can reproduce all
results from a fresh clone. `QUICK_START.md` has setup commands but no recorded
output. This sub-task produces `REPRODUCIBILITY.md` — a verified record of the
exact commands, environment information, and outputs produced by running
everything from scratch in the existing `oulad_env` (without destroying it).

### Expected Outcomes

- `REPRODUCIBILITY.md` at the repository root contains:
  - **Environment**: Python version (`python --version`), OS, all installed
    package versions (`pip freeze` output, trimmed to project-relevant packages)
  - **Step-by-step commands** for a fresh clone, verbatim, copy-pasteable
  - **Test suite output**: verbatim `pytest tests/ -v` result (24/24 passing,
    runtime)
  - **Pipeline output** for each of the 4 weeks: key lines from
    `run_graph_pipeline.py` output (node/edge counts, integrity pass, runtime)
  - **Split verification output**: output from `src/verify_splits.py` (Sub-Task 4)
  - **Git commit hash** of the run, so the record is reproducible to a specific
    version
- All commands in `QUICK_START.md` are reviewed for accuracy against the current
  repo state; any stale steps are updated.

### Todo List

1. Capture `python --version`, `pip freeze`, platform info.
2. Run `pytest tests/ -v` — capture full output.
3. Run `python src/run_graph_pipeline.py --week {2,4,6,8}` — capture key
   summary lines (node/edge counts, integrity pass, runtime).
4. Run `python src/verify_splits.py` — capture output (Sub-Task 4 must be
   done first).
5. Review `QUICK_START.md` for any stale commands (e.g., old test count,
   old file paths) — update in place.
6. Write `REPRODUCIBILITY.md` with all captured outputs, environment info,
   and git commit hash.
7. Commit `REPRODUCIBILITY.md` and any `QUICK_START.md` updates.

### Relevant Context

- `QUICK_START.md` — existing setup guide (verify and update)
- `requirements.txt` — Python dependencies
- `.python-version` — pins Python 3.11.11 via pyenv
- Sub-Task 4 must be complete before step 4 (needs `src/verify_splits.py`)

---

## Sub-Task 6 — Canvas authorization and IRB — repository safeguards

**Status**: `[ ] pending`

### Intent

The repository uses only the publicly available OULAD dataset (Kuzilek et al.
2017, CC-BY 4.0). There is no Canvas data, no proprietary institutional data,
and no identifiable information currently in the repository (the
`CompletionReport-2023.csv` file was purged in the previous plan). However,
there is no committed policy document explicitly stating this, and no automated
safeguard that would catch a future accidental commit of institutional data.

The Canvas authorization and IRB submissions are institutional processes handled
outside the repository. This sub-task's scope is limited to: (1) documenting
the data policy in the repository so any reviewer can immediately confirm what
data is and is not present, and (2) ensuring the `.gitignore` and
`CONTRIBUTING.md` safeguards from the previous plan are sufficient and
discoverable.

### Expected Outcomes

- `docs/DATA_POLICY.md` is created, stating:
  - The repository contains only the public OULAD dataset (citation, license,
    download URL)
  - No Canvas data, no institutional student records, no proprietary data is or
    will be committed
  - Any data collected under IRB approval or Canvas authorization must be stored
    outside the repository (absolute path in a local `.env` or `local_config.py`
    that is gitignored)
  - The `.gitignore` rules that enforce this (`data/*Completion*.csv`,
    `data/*Report*.csv`) are cross-referenced
- `CONTRIBUTING.md` Data Directory Policy section (added in the previous plan)
  is updated to reference `docs/DATA_POLICY.md`.
- A brief note is added to `README.md` (in the Dataset section) stating that
  all data used is from the public OULAD dataset and pointing to
  `docs/DATA_POLICY.md` for the full policy.

### Todo List

1. Write `docs/DATA_POLICY.md` covering: dataset identity (OULAD, CC-BY 4.0),
   what is excluded and why, where IRB/Canvas data should live if ever
   collected, and `.gitignore` safeguards already in place.
2. Update `CONTRIBUTING.md` Data Directory Policy section: add one sentence
   pointing to `docs/DATA_POLICY.md`.
3. Add a one-sentence data-policy note to `README.md` in the Dataset/Setup
   section.
4. Commit all three files.

### Relevant Context

- `CONTRIBUTING.md` — Data Directory Policy added in previous plan
- `.gitignore` — `data/*Completion*.csv` and `data/*Report*.csv` rules added
  in previous plan
- `README.md` — Dataset section (lines ~80–100)
- OULAD license: CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- OULAD citation: Kuzilek J., Hlosta M., Zdrahal Z. (2017). Scientific Data 4:170171.

---

## Dependency Order

Sub-Tasks 1 and 2 are independent and can run in parallel.
Sub-Task 3 depends on Sub-Task 2 (needs regenerated `all_weeks_summary.csv`).
Sub-Task 4 is independent (splits already exist on disk).
Sub-Task 5 depends on Sub-Tasks 2 and 4 (needs the pipeline to be clean and `verify_splits.py` to exist).
Sub-Task 6 is independent.

```
ST1 (schema docs) ─────────────────────────────────────────┐
ST2 (regenerate) → ST3 (consolidated stats table) ─────────┤→ ST5 (REPRODUCIBILITY.md)
ST4 (splits verification) ─────────────────────────────────┘
ST6 (data policy) — independent
```
