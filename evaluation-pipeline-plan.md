# Evaluation Pipeline — Plan

## Top-Level Overview

**Goal**: Consolidate the three existing evaluation scripts (`baseline_evaluation.py`,
`lcpo_evaluation.py`, `future_presentation_evaluation.py`) into a single reusable shared
module (`src/evaluation_pipeline.py`), verify correctness of each split strategy, regenerate
all result tables from scratch, produce a unified 4-week × 4-model × 3-split comparison CSV,
add a course-level difficulty analysis with chart, and update the notebook and documentation
to reflect the clean pipeline.

**Scope**:
- New: `src/evaluation_pipeline.py` — shared module with all split/evaluation logic
- New: `src/run_evaluation.py` — orchestrator that runs all three evaluations end-to-end
- Updated: `notebooks/OULAD_Graph_Analysis_Final.ipynb` — uses the new shared module
- Updated: `docs/EVALUATION_SPLITS.md` — references new pipeline and result files
- New result files: `results/comparison/all_splits_comparison.csv`,
  `results/lcpo/course_presentation_difficulty.csv`,
  `results/lcpo/course_difficulty_chart.png`
- Deleted (after verification): `src/baseline_evaluation.py`, `src/lcpo_evaluation.py`,
  `src/future_presentation_evaluation.py`

**Non-goals**:
- GNN training or graph pipeline changes
- Changes to `src/oulad_data.py` split functions (already correct)
- New model architectures or feature engineering changes
- A `--fast` flag or any run-time limiting mechanism in the runner

**Key Design Decisions (confirmed)**:
- Random split uses **5-fold GroupKFold CV** on `id_student` (not a single 80/10/10 split) —
  consistent with existing baseline and provides better variance estimation
- No run-time limits: the runner always executes all 4 weeks × 4 models × 4 feature subsets
  for all three splits

---

## Sub-Tasks

---

### Sub-Task 1: Audit the three existing split implementations

**Status**: `[x] done`

**Intent**  
Before replacing the three evaluation scripts, audit each one to confirm: (a) random split
enforces student-level separation, (b) LCPO holds out exactly one course-presentation at a
time with no leakage, (c) future-presentation split uses only 2013B/2013J/2014B as train and
2014J as test with strict temporal ordering. Document any bugs or inconsistencies found.

**Expected Outcomes**  
- Written audit notes (added to this plan below each sub-task) confirming correctness or
  describing any issues found
- Confidence that `evaluation_pipeline.py` can be modelled on the correct logic
- Any leakage or split errors surfaced before regeneration

**Todo List**
1. Read `src/baseline_evaluation.py` fully — confirm it uses `GroupKFold` on `id_student`
   (student-level separation) and not `StratifiedKFold` on enrollments
2. Read `src/lcpo_evaluation.py` fully — confirm it calls `lcpo_split()` from `oulad_data.py`
   for each unique `(code_module, code_presentation)` pair and does not mix train/test
3. Read `src/future_presentation_evaluation.py` fully — confirm it filters train set to
   presentations in `{2013B, 2013J, 2014B}` and test set to `{2014J}` without temporal leakage
4. Cross-check that all three scripts use the same `build_features()` and `filter_window()`
   from `oulad_data.py` (not local copies)
5. Cross-check that all three use the same label convention: 1 = at-risk (Fail/Withdrawn),
   0 = success
6. Record any discrepancies in the audit notes section below

**Relevant Context**
- `src/oulad_data.py:141-195` — `random_student_split()`
- `src/oulad_data.py:198-235` — `lcpo_split()`
- `src/baseline_evaluation.py`
- `src/lcpo_evaluation.py`
- `src/future_presentation_evaluation.py`
- `tests/test_splits.py` — 13 unit tests for split functions

**Audit Notes** *(filled in during implementation)*

### Random split (`src/baseline_evaluation.py`)

**Correct and reusable — with one critical bug noted.**

- ✅ Uses `GroupKFold` on `id_student` via `evaluate_model_student_grouped_cv()`. Students are
  shuffled with a fixed seed (`RANDOM_STATE = 42`) and assigned fold labels with `i % n_folds`,
  then `GroupKFold(n_splits=5).split(X, y, groups=groups)` is used — this correctly prevents any
  student appearing in both train and test within a fold.
- ✅ Label convention: `target = 1` (at-risk / Fail+Withdrawn), `0` (success) — explicitly stated
  in the module docstring.
- ✅ Imports `build_features`, `filter_window`, `evaluate_metrics`, `load_oulad_data`,
  `sanitize_feature_names`, and `create_datasets` from `oulad_data.py` — no local copies.
- ✅ Feature subsets: `VLE_only`, `Assessment_only`, `VLE+Assessment`, `All_features` — derived
  dynamically from column names; these 4 subsets are the canonical ablation set.
- ✅ Models: `Majority` (DummyClassifier), `LogisticRegression`, `RandomForest`, `XGBoost`,
  `LightGBM` — matching `config.MODELS`.
- ✅ Model params match `config.MODEL_PARAMS`:
  - LR: `max_iter=1000, random_state=42`
  - RF: `n_estimators=100, random_state=42` (note: local `get_models()` omits `n_jobs=-1`; config
    has `n_jobs: -1` — **minor discrepancy**, new pipeline should use `n_jobs=-1` from config)
  - XGBoost: `n_estimators=100, random_state=42, eval_metric="logloss"`
  - LightGBM: `n_estimators=100, random_state=42, verbose=-1`
- 🐛 **BUG**: `run_baseline_evaluation()` (line 234) calls `evaluate_model_cv(model, ...)` which
  is **not defined anywhere in the file** and is **not imported**. The correct grouped-CV function
  defined in the same file is `evaluate_model_student_grouped_cv()`. This means `main()` would
  raise a `NameError` at runtime. The new `evaluation_pipeline.py` must call the correct
  grouped function.
- ⚠️ **Local `sanitize_feature_names` shadow**: A local copy of `sanitize_feature_names()` is
  defined at line 174, shadowing the import from `oulad_data.py` at line 57. They are functionally
  identical, but the duplicate should not be carried into the new pipeline.
- ✅ `results/baseline/` output path used via `BASELINE_RESULTS_DIR` from `config.py`.
- ✅ Fold assignments CSV (`id_student`, `fold`) is saved as a side-effect — this is a nice audit
  trail but is not required by the new pipeline spec.

### LCPO split (`src/lcpo_evaluation.py`)

**Functionally correct, but does NOT call `lcpo_split()` from `oulad_data.py` — uses inline
split logic instead. Also has local copies of shared utilities.**

- ✅ Label convention: 1 = at-risk (Fail/Withdrawn), 0 = success — stated in module docstring.
- ✅ Iterates all unique `(code_module, code_presentation)` pairs (line 165), holds out each one
  as the test set (`test_mask = (df["code_module"] == module) & (df["code_presentation"] ==
  presentation)`), trains on `~test_mask`. Logic is identical to `lcpo_split()` in
  `oulad_data.py` — just inlined.
- ✅ No train/test leakage: the mask is built before feature preparation; features are computed on
  the full `df` (window-filtered) before the split, which is correct — VLE/assessment aggregates
  are built from the same window-filtered snapshot for all rows.
- ⚠️ **Does NOT call `lcpo_split()` from `oulad_data.py`**. The new pipeline must call the
  canonical `lcpo_split(enrollments_df, held_out_module, held_out_presentation)`.
- 🐛 **Local copies of shared utilities**: `load_oulad_data`, `filter_window`, `build_features`,
  `sanitize_feature_names`, `evaluate_metrics` are all re-defined locally (lines 40–150). These
  are **not imports** from `oulad_data.py`. Functionally they are nearly identical, with one
  difference:
  - Local `build_features()` aggregates `assess_count` on `"date"` column (`.agg({"score":
    ["mean","max"], "date": "count"})`), whereas `oulad_data.build_features()` counts
    `id_assessment` (`.agg({"score": ["mean","max"], "id_assessment": "count"})`). These produce
    the same row count but `"date"` could have NaNs where `id_assessment` cannot — **minor
    discrepancy** that won't materially change results.
  - Local `evaluate_metrics()` calls `f1_score` **without** `zero_division=0`, while
    `oulad_data.evaluate_metrics()` passes `zero_division=0`. The canonical `oulad_data` version
    is safer and must be used.
- ✅ Models and params: identical to `baseline_evaluation.py` (LR, RF, XGBoost, LightGBM with
  same hyperparameters) — no `Majority` baseline here.
- ✅ `main()` only runs week 8; the `lcpo_evaluation()` function accepts a `week` parameter so
  all 4 weeks can be run.
- ✅ Output saved to `LCPO_RESULTS_DIR` via `config.py`.

### Future-presentation split (`src/future_presentation_evaluation.py`)

**Correct temporal split logic. Has same local-copy pattern as lcpo_evaluation.py.**

- ✅ Label convention: 1 = at-risk (Fail/Withdrawn), 0 = success — stated in module docstring.
- ✅ Train presentations: `["2013B", "2013J", "2014B"]`; test presentations: `["2014J"]` — exactly
  as specified. Temporal ordering is correct: 2013B < 2013J < 2014B < 2014J.
- ✅ No leakage: `train_mask = df["code_presentation"].isin(train_presentations)` and
  `test_mask = df["code_presentation"].isin(test_presentations)` are disjoint; features are
  pre-computed for the full window before the split (identical pattern to LCPO).
- ✅ Validation guards: skips if test set < 50 samples or has only one class (lines 223–229).
- 🐛 **Local copies of shared utilities**: same pattern as `lcpo_evaluation.py` — `load_oulad_data`,
  `filter_window`, `build_features`, `sanitize_feature_names`, `evaluate_metrics` all re-defined
  locally (lines 40–150), not imported from `oulad_data.py`. Same `"date"` vs `"id_assessment"`
  discrepancy in `assess_count`; same missing `zero_division=0` in local `evaluate_metrics`.
- ⚠️ `get_models()` here does **not** include `Majority` baseline (4 models only: LR, RF,
  XGBoost, LightGBM). For consistency, the new pipeline should optionally include Majority.
- ✅ All 4 prediction windows (2, 4, 6, 8) are run in `main()`.
- ✅ Output saved to `CROSS_COURSE_RESULTS_DIR` via `config.py`.

### Shared utilities (`src/oulad_data.py`)

- ✅ `random_student_split()` (lines 141–195): splits on unique students (not enrollments),
  returns boolean masks, includes assertions verifying no train/test overlap. **Correct.**
- ✅ `lcpo_split()` (lines 198–235): straightforward boolean mask on
  `(code_module, code_presentation)` with ValueError guards for empty splits. **Correct.**
- ✅ `build_features()` (lines 73–102): starts from `student_info` (all enrollments) and left-
  joins VLE and assessment aggregates — inactive students kept with zero features. Uses
  `id_assessment` for `assess_count`. **Canonical version.**
- ✅ `filter_window()` (lines 54–70): filters on `assessments.date` (due date), not submission
  date, preventing future leakage. **Correct.**
- ✅ `evaluate_metrics()` (lines 116–125): includes `zero_division=0` on all relevant metrics.
  **Canonical version.**

### Label convention (all scripts)

- ✅ **Uniform across all three scripts**: `target = 1` for Fail/Withdrawn; `target = 0` for
  Pass/Distinction. All module docstrings explicitly state this. `config.LABEL_MAPPING` also
  reflects this correctly.

### Canonical params and feature subsets for `evaluation_pipeline.py`

**Models** (source: `config.MODELS` + `config.MODEL_PARAMS`):
- `Majority`: `DummyClassifier(strategy="most_frequent")`
- `LogisticRegression`: `max_iter=1000, random_state=42`
- `RandomForest`: `n_estimators=100, random_state=42, n_jobs=-1`
- `XGBoost`: `n_estimators=100, random_state=42, eval_metric="logloss"`
- `LightGBM`: `n_estimators=100, random_state=42, verbose=-1`

**Feature subsets** (source: `baseline_evaluation.get_feature_subsets()`):
- `VLE_only`: columns containing `"vle_"`
- `Assessment_only`: columns containing `"assess_"`
- `VLE+Assessment`: union of above two
- `All_features`: all non-metadata columns (VLE + Assessment + demographics)

**Prediction windows** (source: `config.PREDICTION_WINDOWS`):
- week 2 = 14 days, week 4 = 28 days, week 6 = 42 days, week 8 = 56 days

**Presentation codes** for future-presentation split:
- Train: `["2013B", "2013J", "2014B"]`
- Test: `["2014J"]`

### Summary of bugs / inconsistencies to fix in new pipeline

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| 1 | `baseline_evaluation.py:234` | Calls undefined `evaluate_model_cv()` — **NameError at runtime** | Use `evaluate_model_student_grouped_cv()` (the correct function defined in the same file) |
| 2 | `lcpo_evaluation.py`, `future_presentation_evaluation.py` | Local copies of `load_oulad_data`, `filter_window`, `build_features`, `sanitize_feature_names`, `evaluate_metrics` — not imported from `oulad_data.py` | New pipeline imports canonical versions from `oulad_data.py` only |
| 3 | Local `build_features()` in LCPO/FP scripts | `assess_count` computed on `"date"` column; canonical uses `"id_assessment"` | Use canonical `oulad_data.build_features()` |
| 4 | Local `evaluate_metrics()` in LCPO/FP scripts | Missing `zero_division=0` on `f1_score` | Use canonical `oulad_data.evaluate_metrics()` |
| 5 | `baseline_evaluation.py:174` | Local `sanitize_feature_names()` shadows the import | Remove local copy; use import from `oulad_data.py` |
| 6 | `baseline_evaluation.py` `get_models()` | `RandomForest` missing `n_jobs=-1` present in `config.MODEL_PARAMS` | Add `n_jobs=-1` in new pipeline |
| 7 | `lcpo_evaluation.py` | Does not call `lcpo_split()` from `oulad_data.py` — uses inlined mask | New pipeline calls `lcpo_split()` directly |

---

### Sub-Task 2: Build `src/evaluation_pipeline.py`

**Status**: `[x] done`

**Intent**  
Create a single shared module that contains all reusable functions needed to run any of the
three split evaluations. This becomes the single source of truth, replacing the three existing
evaluation scripts. Each function should be independently callable so the notebook and runner
can use them selectively.

**Expected Outcomes**
- `src/evaluation_pipeline.py` exists and is importable
- Contains: `get_models()`, `evaluate_split()` (generic train/test evaluator),
  `run_random_student_evaluation()`, `run_lcpo_evaluation()`,
  `run_future_presentation_evaluation()`, `build_unified_comparison_table()`,
  `analyze_course_difficulty()` (AUROC per course-presentation, sorted hardest to easiest)
- All functions share the same `build_features()`, `filter_window()`, `evaluate_metrics()`
  from `src/oulad_data.py`
- All functions use the same label convention, model params from `src/config.py`, and
  prediction windows from `src/config.py`
- Module has a clear docstring documenting the public API

**Todo List**
1. Create `src/evaluation_pipeline.py` with module-level docstring
2. Add `get_models()` — returns the same 4 sklearn-compatible model instances used across
   all three existing scripts (LR, RF, XGBoost, LightGBM) with params from `config.py`
3. Add `evaluate_split(model, X_train, y_train, X_test, y_test)` — trains a single model
   on train, predicts on test, returns a dict of all 6 metrics via `evaluate_metrics()`
4. Add `run_random_student_evaluation(enrollments_df, windows, n_seeds)` — runs 5-fold
   GroupKFold CV on student IDs across all windows and models, returns a DataFrame
5. Add `run_lcpo_evaluation(enrollments_df, windows)` — iterates all 22 course-presentations,
   calls `lcpo_split()` from `oulad_data.py`, runs all models, returns a DataFrame
6. Add `run_future_presentation_evaluation(enrollments_df, windows)` — applies temporal split
   (train: 2013B/2013J/2014B, test: 2014J), runs all models, returns a DataFrame
7. Add `build_unified_comparison_table(random_df, lcpo_df, future_df)` — merges all three
   into a single DataFrame keyed by `(Week, Model, Split)` with all 6 metrics
8. Add `analyze_course_difficulty(lcpo_df)` — aggregates AUROC per course-presentation
   across all models and weeks, computes mean ± std, sorts ascending (hardest first),
   and generates a matplotlib boxplot saved to `results/lcpo/course_difficulty_chart.png`
9. Write unit-level docstrings for every public function
10. Verify the module is importable from `src/` with no side effects on import

**Relevant Context**
- `src/oulad_data.py` — `build_features()`, `filter_window()`, `evaluate_metrics()`,
  `random_student_split()`, `lcpo_split()`
- `src/config.py` — `MODEL_PARAMS`, `PREDICTION_WINDOWS`, `MODELS`, `METRICS`,
  `BASELINE_RESULTS_DIR`, `LCPO_RESULTS_DIR`, `CROSS_COURSE_RESULTS_DIR`
- Existing model params and feature sets from `baseline_evaluation.py` — audit output from
  Sub-Task 1 confirms the canonical params to use
- `results/lcpo/lcpo_results_detailed.csv` — example of expected LCPO output schema
- `results/cross_course/future_presentation_results.csv` — example of expected FP output schema

---

### Sub-Task 3: Build `src/run_evaluation.py` (orchestrator)

**Status**: `[x] done`

**Intent**  
Create a single runnable script that calls all three evaluation functions from
`evaluation_pipeline.py`, saves all result CSVs, and saves the unified comparison table and
course difficulty chart. Running this script from the command line should fully reproduce all
results from scratch.

**Expected Outcomes**
- `src/run_evaluation.py` runs end-to-end via `python src/run_evaluation.py` with no errors
- Saves/overwrites:
  - `results/baseline/baseline_results_detailed.csv`
  - `results/baseline/baseline_results_table.csv`
  - `results/lcpo/lcpo_results_detailed.csv`
  - `results/lcpo/random_vs_lcpo_comparison.csv`
  - `results/cross_course/future_presentation_results.csv`
  - `results/comparison/all_splits_comparison.csv` *(new)*
  - `results/lcpo/course_presentation_difficulty.csv` *(new)*
  - `results/lcpo/course_difficulty_chart.png` *(new)*
  - `results/overall_summary.csv` *(updated with Future-Presentation row)*
- Prints progress to stdout and a completion summary

**Todo List**
1. Create `src/run_evaluation.py` with `main()` function
2. Load OULAD data via `load_oulad_data()` from `oulad_data.py`
3. Call `run_random_student_evaluation()` → save detailed and summary CSVs to
   `results/baseline/`
4. Call `run_lcpo_evaluation()` → save to `results/lcpo/`
5. Call `run_future_presentation_evaluation()` → save to `results/cross_course/`
6. Call `build_unified_comparison_table()` → save to `results/comparison/all_splits_comparison.csv`
7. Call `analyze_course_difficulty()` → save CSV and PNG to `results/lcpo/`
8. Regenerate `results/overall_summary.csv` with all three split rows for Week 8
9. Add `if __name__ == "__main__": main()` guard
10. Test that the script runs without errors on the actual OULAD data

**Relevant Context**
- `src/evaluation_pipeline.py` (Sub-Task 2 output)
- `src/oulad_data.py:load_oulad_data()`
- `src/config.py` — result directories
- Existing result file schemas (Sub-Task 1 audit output)

---

### Sub-Task 4: Verify results and delete old evaluation scripts

**Status**: `[x] done`

**Intent**  
Run the new pipeline, compare output metrics to the previously saved results to confirm
reproducibility, and once confirmed, delete the three old evaluation scripts to establish
`evaluation_pipeline.py` as the single source of truth.

**Expected Outcomes**
- `python src/run_evaluation.py` completes without error
- Key metrics match previously saved results within tolerance (±0.002 AUROC for random split;
  LCPO and future-presentation results may differ slightly if old scripts had issues identified
  in the Sub-Task 1 audit)
- `src/baseline_evaluation.py`, `src/lcpo_evaluation.py`,
  `src/future_presentation_evaluation.py` are deleted
- `results/comparison/all_splits_comparison.csv` exists with 4 weeks × 4 models × 3 splits
  = 48 rows
- `results/lcpo/course_presentation_difficulty.csv` exists with 22 rows (one per
  course-presentation)
- `results/lcpo/course_difficulty_chart.png` exists

**Todo List**
1. Run `python src/run_evaluation.py` and capture stdout/stderr
2. Compare `results/baseline/baseline_results_detailed.csv` (new) to prior results — confirm
   LightGBM Week 8 AUROC is within ±0.002 of 0.835
3. Compare `results/lcpo/lcpo_results_detailed.csv` (new) — confirm LightGBM Week 8 mean
   AUROC is within ±0.01 of 0.804
4. Compare `results/cross_course/future_presentation_results.csv` (new) — confirm LightGBM
   Week 8 AUROC is within ±0.01 of 0.7998
5. Verify `results/comparison/all_splits_comparison.csv` has the expected schema and row count
6. Verify `results/lcpo/course_presentation_difficulty.csv` has 22 rows and is sorted by AUROC
7. Verify `results/lcpo/course_difficulty_chart.png` renders correctly
8. Delete `src/baseline_evaluation.py`, `src/lcpo_evaluation.py`,
   `src/future_presentation_evaluation.py`

**Relevant Context**
- `results/overall_summary.csv` — existing Week 8 benchmarks to compare against
- `results/lcpo/random_vs_lcpo_comparison.csv` — existing LCPO benchmarks

---

### Sub-Task 5: Update the notebook and documentation

**Status**: `[x] done`

**Intent**  
Update `notebooks/OULAD_Graph_Analysis_Final.ipynb` to import from `evaluation_pipeline.py`,
display the new unified comparison table, and show the course difficulty chart. Update
`docs/EVALUATION_SPLITS.md` to reference the new pipeline, runner, and all output CSV files.

**Expected Outcomes**
- Notebook imports from `evaluation_pipeline.py` (no inline duplicated logic)
- Notebook displays the unified 3-split comparison table
- Notebook displays the course difficulty chart
- Notebook runs top-to-bottom without errors
- `docs/EVALUATION_SPLITS.md` references `src/evaluation_pipeline.py`,
  `src/run_evaluation.py`, and all CSV/PNG output files
- `docs/EVALUATION_SPLITS.md` includes a "course difficulty" section with the hardest and
  easiest course-presentations identified

**Todo List**
1. Open `notebooks/OULAD_Graph_Analysis_Final.ipynb` and identify existing cells that
   duplicate logic now in `evaluation_pipeline.py`
2. Replace those cells with imports from `evaluation_pipeline` and calls to the public
   functions
3. Add a new section "Unified Split Comparison" that loads and displays
   `results/comparison/all_splits_comparison.csv`
4. Add a new section "Course-Level Difficulty" that loads
   `results/lcpo/course_presentation_difficulty.csv`, displays it as a table, and shows the
   course difficulty chart from `results/lcpo/course_difficulty_chart.png`
5. Run all cells top-to-bottom and confirm no errors
6. Update `docs/EVALUATION_SPLITS.md`:
   - Update the "Running the Evaluation" section to reference `src/run_evaluation.py`
   - Add a table listing all output CSV files with their paths and descriptions
   - Add a "Course Difficulty Analysis" section describing the per-course AUROC findings

**Relevant Context**
- `notebooks/OULAD_Graph_Analysis_Final.ipynb` — 44 cells, currently fully executed
- `docs/EVALUATION_SPLITS.md` — existing documentation
- `results/comparison/all_splits_comparison.csv` (Sub-Task 3 output)
- `results/lcpo/course_presentation_difficulty.csv` (Sub-Task 3 output)

---

## Result File Index (final state)

| File | Description |
|------|-------------|
| `results/baseline/baseline_results_detailed.csv` | Random split: all weeks × models × feature subsets |
| `results/baseline/baseline_results_table.csv` | Random split summary by week |
| `results/lcpo/lcpo_results_detailed.csv` | LCPO: all weeks × models × course-presentations |
| `results/lcpo/random_vs_lcpo_comparison.csv` | Random vs LCPO per model (Week 8) |
| `results/lcpo/course_presentation_difficulty.csv` | Per-course AUROC mean ± std, sorted hardest first |
| `results/lcpo/course_difficulty_chart.png` | Boxplot of per-course AUROC distribution |
| `results/cross_course/future_presentation_results.csv` | Future-presentation: all weeks × models |
| `results/comparison/all_splits_comparison.csv` | Unified: all weeks × models × all 3 splits |
| `results/overall_summary.csv` | Top-level summary: all 3 splits for Week 8 |
