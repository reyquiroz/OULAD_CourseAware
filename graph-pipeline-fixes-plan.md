# Graph Pipeline Fixes — Plan

## Top-Level Overview

**Goal**: Finalize the graph pipeline by addressing five correctness and usability issues:
assessment filtering leakage, missing-value documentation, split utility discoverability,
canonical notebook confirmation, and fresh-clone reproducibility.

**Key finding from analysis**: Assessment filtering in the current pipeline uses only
`assessments.date` (due date). Empirical comparison shows that also filtering by
`date_submitted ≤ window` (Strategy B) is strictly leakage-free. The AUROC delta is
≤0.0024 across all windows (within noise), but Strategy B is scientifically cleaner and
is the adopted approach.

**Scope**:
- Modified: `src/oulad_data.py` — dual filter in `filter_window()` + graph split docstring
- Modified: `src/graph_pipeline.py` — inline imputation documentation
- Modified: `src/graph_validation.py` — pre/post imputation sections in validation report
- Modified: `QUICK_START.md` — explicit file list with sizes
- New: `src/check_data.py` — preflight data check
- Regenerated: all baseline, LCPO, future-presentation results and graph artifacts (Week 8)
- Not in scope: README restructure, GNN training, new model architectures

**Key Design Decisions (confirmed)**:
- Strategy B (dual filter): include assessment score only if `due_date ≤ window`
  **AND** `date_submitted ≤ window`. No null `date_submitted` values exist in the dataset.
- Split utilities remain in `src/oulad_data.py` — no new file for graph splits.
- Validation report updated to show both pre-imputation audit counts and
  post-imputation confirmation.

---

## Sub-Tasks

---

### Sub-Task 1: Add dual assessment filter to `filter_window()` and regenerate results

**Status**: `[ ] pending`

**Intent**
Update `filter_window()` in `src/oulad_data.py` to apply a second guard:
`date_submitted ≤ window`. This removes the 4.9% of Week 8 submissions that were
submitted after the prediction cutoff. Because `filter_window()` is shared by both the
tabular baseline and the graph pipeline, this one-line change propagates consistently
to all evaluation paths.

After the code change, regenerate all evaluation results (via `src/run_evaluation.py`)
and the Week 8 graph artifacts (via `src/run_graph_pipeline.py --week 8`).

**Expected Outcomes**
- `filter_window()` applies `date_submitted ≤ window` as a second condition on
  the student assessment rows
- The docstring is updated to reflect both filter conditions and their rationale
- `python src/run_evaluation.py` completes without error
- `python src/run_graph_pipeline.py --week 8` completes without error
- All result CSVs in `results/baseline/`, `results/lcpo/`, `results/cross_course/`,
  and `results/comparison/` are regenerated
- New AUROC values remain within ±0.005 of prior values (confirmed by empirical test)

**Todo List**
1. Read `src/oulad_data.py` lines 54–70 (`filter_window`) to confirm exact current code
2. Update `filter_window()`:
   - After joining `student_assess` with `assessments` on `id_assessment`, add second
     filter: `assess_w = assess_w[assess_w["date_submitted"] <= window]`
   - Update docstring to document both filter conditions and note that
     `date_submitted` is always populated in OULAD (no nulls)
3. Run `pytest tests/ -v` — confirm all 13 unit tests still pass
4. Run `python src/run_evaluation.py` — regenerate all tabular results
5. Run `python src/run_graph_pipeline.py --week 8` — regenerate graph artifacts
6. Verify LightGBM Week 8 AUROC in new `results/overall_summary.csv` is within
   ±0.005 of 0.865 (expected: ~0.863 based on empirical test)

**Relevant Context**
- `src/oulad_data.py:54-70` — `filter_window()` current implementation
- `src/oulad_data.py:128-134` — `create_datasets()` calls `filter_window()`
- `src/graph_pipeline.py:106-113` — `apply_window_cutoff()` calls `filter_window()`
- `tests/test_splits.py` — 13 unit tests (split functions, not filter; should still pass)
- Empirical test result: Strategy B AUROC = 0.8630±0.0038 at Week 8 (delta = −0.0024)

---

### Sub-Task 2: Document missing values — inline code + validation report format

**Status**: `[ ] pending`

**Intent**
Two classes of pre-imputation nulls are expected from raw OULAD source data:
- `nodes_student.imd_band`: 971 nulls (students with unknown deprivation band)
- `nodes_vle_resource.week_from` / `week_to`: 10,486 nulls (VLE resources with no
  scheduled week in `vle.csv`)

The pipeline already imputes all of these (numeric→0, categorical→"Unknown") and
asserts 0 nulls post-imputation. However, the validation summary report shows ⚠
flags with no explanation, making it look like a bug. This sub-task adds inline code
comments explaining the source of each expected null and updates the validation report
to clearly separate the pre-imputation audit from the post-imputation confirmation.

**Expected Outcomes**
- `src/graph_pipeline.py` `build_node_tables()` has an inline comment block explaining
  expected nulls per node type and the imputation strategy
- `src/graph_validation.py` validation summary includes a new "Post-imputation null
  check" section that shows 0 for all node types and labels the pre-imputation ⚠
  counts as "expected from source data — resolved by imputation"
- Regenerating `results/graph/validation/week08_validation_summary.txt` (via
  `run_graph_pipeline.py`) reflects the new format
- `docs/validation_report_week8.md` is updated to document the two expected null
  sources and their resolution

**Todo List**
1. Read `src/graph_pipeline.py` lines 178–210 (`build_node_tables` imputation section)
2. Add inline comments above the imputation loop explaining:
   - `imd_band` nulls: ~971 students in OULAD have no recorded deprivation band;
     imputed to "Unknown" (categorical)
   - `week_from` / `week_to` nulls: ~5,243 VLE resources have no scheduled week
     in `vle.csv`; imputed to 0 (numeric)
3. Read `src/graph_validation.py` — find where the data quality section is printed
4. Update the validation summary output to:
   - Label pre-imputation counts as `[pre-imputation, expected]`
   - Add a new "Post-imputation null check" section that confirms 0 nulls per node type
5. Read `docs/validation_report_week8.md` and add a "Known null sources" section
   documenting both fields, their origin, and the imputation applied
6. Re-run `python src/run_graph_pipeline.py --week 8` to regenerate the validation
   report with the new format
7. Confirm `results/graph/validation/week08_validation_summary.txt` shows the
   pre/post separation

**Relevant Context**
- `src/graph_pipeline.py:178-210` — `build_node_tables()` imputation loop
- `src/graph_validation.py` — data quality section of validation report
- `results/graph/validation/week08_validation_summary.txt` — current report output
- `results/graph/validation/week08_validation.json` — machine-readable report
- `docs/validation_report_week8.md` — human-readable audit trail

---

### Sub-Task 3: Document graph split utilities in `src/oulad_data.py`

**Status**: `[ ] pending`

**Intent**
`random_student_split()` and `lcpo_split()` in `src/oulad_data.py` are the canonical
split utilities for both the tabular baseline and the graph/GNN pipeline. Their graph
context is currently only demonstrated in the notebook. This sub-task adds a clear
module-level section and docstring examples showing how they are used in the graph
context — operating on the enrollment supervision table from
`build_enrollment_supervision()`.

**Expected Outcomes**
- `src/oulad_data.py` module docstring (or a dedicated section comment block) has a
  "Graph pipeline usage" example showing how to call `random_student_split()` and
  `lcpo_split()` on the output of `build_enrollment_supervision()`
- `random_student_split()` docstring has an `Examples` section illustrating graph usage
- `lcpo_split()` docstring has an `Examples` section illustrating graph usage
- No new files created; no behaviour change

**Todo List**
1. Read `src/oulad_data.py` fully — note current module docstring and function
   docstrings for `random_student_split()` and `lcpo_split()`
2. Add or update the module-level docstring to include a "Graph pipeline usage" section
   showing the full call chain:
   ```python
   from graph_pipeline import build_enrollment_supervision
   from oulad_data import random_student_split, lcpo_split

   enrollments = build_enrollment_supervision(filtered)
   train_mask, val_mask, test_mask = random_student_split(enrollments)
   # masks index directly into the enrollment supervision table
   ```
3. Add an `Examples` block to `random_student_split()` docstring showing graph context
4. Add an `Examples` block to `lcpo_split()` docstring showing graph context
5. Run `python -c "import oulad_data; help(oulad_data.random_student_split)"` to
   confirm the docstring renders correctly

**Relevant Context**
- `src/oulad_data.py:141-235` — `random_student_split()` and `lcpo_split()`
- `src/graph_pipeline.py:332-359` — `build_enrollment_supervision()` produces the
  DataFrame that the split utilities operate on
- `notebooks/OULAD_Graph_Analysis_Final.ipynb` cells 16–19 — existing demo of split
  on graph enrollment table (reference for the docstring example)

---

### Sub-Task 4: Add preflight data check (`src/check_data.py`) and update `QUICK_START.md`

**Status**: `[ ] pending`

**Intent**
A fresh clone requires `studentVle.csv` (433 MB) to be manually downloaded. Currently,
running the pipeline without it produces an unhelpful `FileNotFoundError`. This sub-task
adds a preflight check function that validates all required data files exist before the
pipeline runs, prints a clear actionable error if any are missing, and updates
`QUICK_START.md` with the exact expected file list including sizes.

**Expected Outcomes**
- `src/check_data.py` exists with a `check_data_files()` function that:
  - Lists all required CSV files with their expected minimum sizes
  - Prints ✓/✗ for each file
  - Raises `FileNotFoundError` with the OULAD download URL if any required file
    is missing
- `src/run_evaluation.py` calls `check_data_files()` at the top of `main()`
- `src/run_graph_pipeline.py` calls `check_data_files()` at the top of `main()`
- `QUICK_START.md` step 5 lists all 7 required CSV files with sizes and notes which
  is gitignored

**Todo List**
1. Create `src/check_data.py` with `check_data_files(data_dir=None)`:
   - Required files: `studentInfo.csv`, `studentVle.csv`, `studentAssessment.csv`,
     `assessments.csv`, `courses.csv`, `vle.csv`, `studentRegistration.csv`
   - Include approximate file sizes for user guidance
   - Print ✓/✗ status for each file
   - Raise `FileNotFoundError` with OULAD download URL if any file is missing
2. Add `from check_data import check_data_files` + `check_data_files()` call at the
   top of `src/run_evaluation.py` `main()`
3. Add the same call at the top of `src/run_graph_pipeline.py` `main()`
4. Read `QUICK_START.md` step 5 (data files section)
5. Update step 5 to list all 7 CSVs with approximate sizes and note which is gitignored
6. Test: rename `data/raw/studentVle.csv` temporarily, run `python src/check_data.py`,
   confirm clear error message, restore the file

**Relevant Context**
- `src/config.py:13` — `DATA_DIR = PROJECT_ROOT / "data/raw"`
- `src/run_evaluation.py` — `main()` entry point
- `src/run_graph_pipeline.py` — `main()` entry point
- `QUICK_START.md` — existing fresh-clone instructions
- Required files and sizes (from repo):
  - `studentInfo.csv` ~3 MB
  - `studentVle.csv` ~433 MB (**gitignored — must download**)
  - `studentAssessment.csv` ~6 MB
  - `assessments.csv` ~10 KB
  - `courses.csv` ~1 KB
  - `vle.csv` ~500 KB
  - `studentRegistration.csv` ~1.5 MB

---

### Sub-Task 5: Final end-to-end validation and push

**Status**: `[ ] pending`

**Intent**
After all four code changes are in place, run the full pipeline end-to-end from a clean
state to confirm everything works together. Verify the canonical notebook still executes
cleanly with the regenerated results. Commit and push all changes.

**Expected Outcomes**
- `pytest tests/ -v` passes 13/13
- `python src/check_data.py` prints ✓ for all 7 files
- `python src/run_evaluation.py` completes and saves all 9 result CSVs
- `python src/run_graph_pipeline.py --week 8` completes and saves graph artifacts
  and updated validation summary
- `notebooks/OULAD_Graph_Analysis_Final.ipynb` executes clean top-to-bottom
- All changes committed and pushed to both `origin` and `lab` remotes

**Todo List**
1. Run `pytest tests/ -v` — confirm 13/13 pass
2. Run `python src/check_data.py` — confirm all ✓
3. Run `python src/run_evaluation.py` — confirm completion and check
   `results/overall_summary.csv` for expected AUROC values
4. Run `python src/run_graph_pipeline.py --week 8` — confirm new validation summary
   shows pre/post imputation sections
5. Run the canonical notebook: `jupyter nbconvert --to notebook --execute
   --inplace notebooks/OULAD_Graph_Analysis_Final.ipynb`
6. Confirm notebook executes without errors
7. Commit all changes with a clear message referencing each fix
8. Push to `origin main` and `lab main`

**Relevant Context**
- All prior sub-tasks
- `results/overall_summary.csv` — expected LightGBM Week 8 AUROC ~0.863
- `results/graph/validation/week08_validation_summary.txt` — should show new format

---

## Summary of Changes per File

| File | Change |
|------|--------|
| `src/oulad_data.py` | `filter_window()`: add `date_submitted ≤ window` guard; update docstring + add graph usage examples to split utilities |
| `src/graph_pipeline.py` | `build_node_tables()`: inline comments documenting expected null sources and imputation |
| `src/graph_validation.py` | Validation summary: separate pre-imputation audit from post-imputation confirmation |
| `src/check_data.py` | New: preflight data file check with clear error messages |
| `src/run_evaluation.py` | Add `check_data_files()` call at top of `main()` |
| `src/run_graph_pipeline.py` | Add `check_data_files()` call at top of `main()` |
| `QUICK_START.md` | Step 5: explicit file list with sizes, note on gitignored file |
| `docs/validation_report_week8.md` | Add "Known null sources" section |
| All result CSVs under `results/` | Regenerated with Strategy B filter |
| `results/graph/artifacts/week08_*` | Regenerated with Strategy B filter |
| `results/graph/validation/week08_*` | Regenerated with new validation format |
