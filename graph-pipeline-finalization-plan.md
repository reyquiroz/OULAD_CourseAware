# Graph Pipeline Finalization and Reproducibility Audit — Plan

## Top-Level Overview

**Goal**: Harden the existing graph-construction pipeline so that it is fully reproducible from a fresh clone, all evaluation code is correct and honest, and the lab repository reflects the current state of work.

**Scope**: Covers `src/oulad_data.py`, `src/graph_pipeline.py`, `requirements.txt`, the GNN evaluation notebook(s), and reusable split utilities. Does **not** include training or evaluating GraphSAGE — that is left for the following week.

**Target definition (fixed throughout)**:
- `1 = at-risk` → Fail or Withdrawn (positive class)
- `0 = success` → Pass or Distinction (negative class)
- All reported Precision, Recall, F1, and AUPRC refer to the at-risk class (class 1).

**Non-goals**:
- Training or benchmarking GraphSAGE / any GNN model.
- Extending or replacing `src/gnn_model.py` (deprecated; leave in place).
- Hyperparameter tuning.

---

## Sub-Tasks

---

### Sub-Task 1 — Fix Assessment Filtering to Use Due Date

**Status**: `[ ] pending`

**Intent**

The canonical leakage-safe rule (documented in `docs/LEAKAGE_PREVENTION.md`) is: *include an assessment only if its **due date** falls on or before the prediction cutoff*. The shared utility `filter_window()` in `src/oulad_data.py` currently filters on `date_submitted <= window` (line 63), which leaks future submission behaviour. `src/graph_pipeline.py` already applies the correct due-date filter on `assessments["date"]` (line 115) but still delegates to `filter_window()` for `student_assess`, creating an inconsistency. The fix aligns both callers on due-date filtering.

**Expected Outcomes**

- `filter_window()` in `src/oulad_data.py` filters `student_assess` rows by `due_date <= window` (via a join to `assessments["date"]`), not `date_submitted`.
- `src/baseline_evaluation.py` header comment (Task 1 note, line 9) is updated to reflect the corrected rule.
- All four prediction windows (Week 2/4/6/8) still produce valid non-empty feature tables after the change.

**Todo List**

1. In `src/oulad_data.py`, change line 63 from `assess_with_dates["date_submitted"] <= window` to `assess_with_dates["date"] <= window` so filtering uses the due date column that was joined in on line 59.
2. Update the Task 1 comment in `src/baseline_evaluation.py` (line 9) to read: `assessment filtering uses due_date (assessments.date) <= window`.
3. Verify `src/graph_pipeline.py` `apply_window_cutoff()` remains consistent — its docstring already says "due date" and line 115 applies `assessments["date"] <= window_days`, which is correct and unchanged.

**Relevant Context**

- [`src/oulad_data.py:54-65`](src/oulad_data.py) — `filter_window()` to edit.
- [`src/baseline_evaluation.py:9`](src/baseline_evaluation.py) — comment to update.
- [`src/graph_pipeline.py:88-118`](src/graph_pipeline.py) — `apply_window_cutoff()`, reference for correct pattern.
- [`docs/LEAKAGE_PREVENTION.md`](docs/LEAKAGE_PREVENTION.md) — canonical rule.

---

### Sub-Task 2 — Handle Missing Graph Features Explicitly

**Status**: `[ ] pending`

**Intent**

The graph pipeline (`src/graph_pipeline.py`) builds node feature tables but does not document or assert how null/NaN values in those tables are handled. Validation already flags expected nulls (971 in `imd_band`, 10 486 in VLE `week_from`/`week_to`). For downstream GNN training these nulls must be resolved before tensors are constructed. This sub-task makes the handling explicit and deterministic so no silently NaN features enter future training.

**Expected Outcomes**

- A helper function (or an explicit imputation block within `build_node_tables()`) fills or marks missing values in each node-feature table before the table is returned.
- Strategy: numeric nulls → 0, categorical nulls → `"Unknown"` (mirrors `build_features()` in `oulad_data.py:94-95`).
- Validation report (`week08_validation_summary.txt`) notes that nulls are handled and the post-imputation null counts are zero for all feature columns that will be tensor-encoded.

**Todo List**

1. In `src/graph_pipeline.py` `build_node_tables()`, after building each node DataFrame, apply the same imputation pattern used in `oulad_data.py:94-95`: fill numeric columns with `0`, fill categorical columns with `"Unknown"`.
2. Update `src/graph_validation.py` (or the validation stage in `run_pipeline()`) to separately report pre-imputation and post-imputation null counts so the audit trail is clear.
3. Confirm the Week 8 validation summary reflects zero post-imputation nulls for feature columns.

**Relevant Context**

- [`src/graph_pipeline.py:125-186`](src/graph_pipeline.py) — `build_node_tables()` to edit.
- [`src/oulad_data.py:91-95`](src/oulad_data.py) — imputation pattern to mirror.
- [`src/graph_validation.py`](src/graph_validation.py) — validation reporting.
- Expected nulls documented in exploration: `imd_band` (971), `week_from`/`week_to` (10 486).

---

### Sub-Task 3 — Correct Data and Output Paths

**Status**: `[ ] pending`

**Intent**

`src/config.py` defines `DATA_DIR = PROJECT_ROOT / "DATA/raw"` (uppercase `DATA`). The actual data directory must match. Similarly, all output paths must be reachable relative to the project root from both `src/` (when scripts are run directly) and the notebook directory. Any hardcoded absolute paths in notebooks must be removed.

**Expected Outcomes**

- `src/config.py` `DATA_DIR` matches the real data directory name on disk (case-sensitive).
- `src/run_graph_pipeline.py` `--data-dir` default derives from `config.DATA_DIR`, not a hardcoded string.
- Notebooks reference data and output via relative paths or `config.py` constants rather than absolute paths.
- Running `python src/run_graph_pipeline.py --week 8` from the project root succeeds without path errors.

**Todo List**

1. Check the actual data directory name on disk and reconcile with `config.py` `DATA_DIR`.
2. Audit `src/run_graph_pipeline.py` for hardcoded paths; replace with `config.DATA_DIR` / `config.GRAPH_ARTIFACTS_DIR` etc.
3. Audit the GNN notebook(s) for absolute paths and replace with relative paths or `config` imports.
4. Verify `results/graph/artifacts/` and `results/graph/validation/` exist or are created automatically.

**Relevant Context**

- [`src/config.py:1-30`](src/config.py) — all path constants.
- [`src/run_graph_pipeline.py`](src/run_graph_pipeline.py) — CLI entry point to audit.
- `data/` directory (gitignored) — confirm case.

---

### Sub-Task 4 — Switch to Python 3.12 via pyenv, Rebuild venv, Fix requirements.txt

**Status**: `[ ] pending`

**Intent**

The current `oulad_env` is built on **Homebrew Python 3.14**, which has two blockers:

1. **PyTorch has no official Python 3.14 wheels** — `torch` and `torch-geometric` cannot be installed in the current environment. These are needed for GNN work.
2. **`pyarrow` is missing** — `src/graph_pipeline.py` calls `df.to_parquet()`, which requires `pyarrow`. The pipeline will crash without it.

The decision is to switch the project to **Python 3.12 via pyenv**, which is the latest PyTorch-compatible Python version with full wheel support. This sub-task: pins the project to Python 3.12 with a `.python-version` file, rebuilds `oulad_env` from scratch under Python 3.12, installs all dependencies including `pyarrow`, `torch`, and `torch-geometric`, and updates `requirements.txt` to match.

**Expected Outcomes**

- `pyenv install 3.12` has been run and Python 3.12 is available.
- A `.python-version` file at the project root contains `3.12` (or the exact patch version chosen).
- The old `oulad_env/` is deleted and a new one is created with `pyenv local 3.12 && python -m venv oulad_env`.
- All packages from `requirements.txt` are installed cleanly in the new venv.
- `pyarrow>=14.0.0` is added to `requirements.txt`.
- A `# Graph neural network` section is added to `requirements.txt` with `torch>=2.0.0` and `torch-geometric>=2.3.0`, plus a comment with the PyTorch wheel index URL (`https://download.pytorch.org/whl/cpu` for CPU-only, or CUDA equivalent).
- `torch` and `torch-geometric` are installed in the rebuilt `oulad_env` and importable.
- `python -c "import torch; import torch_geometric; print(torch.__version__)"` succeeds.
- `requirements.txt` version bounds are updated to reflect the newly installed versions.

**Todo List**

1. In `requirements.txt`: add `pyarrow>=14.0.0`; add a `# Graph neural network` section with `torch>=2.0.0` and `torch-geometric>=2.3.0` and a wheel-index comment.
2. Create `.python-version` at the project root containing `3.12` (pyenv will select the latest installed 3.12.x).
3. Run `pyenv install 3.12` (if not already installed) and confirm `pyenv versions` lists it.
4. Delete the existing `oulad_env/` directory.
5. Run `pyenv local 3.12` then `python -m venv oulad_env` to create the new venv under Python 3.12.
6. Activate the venv and run `pip install -r requirements.txt`.
7. Install PyTorch and PyTorch Geometric separately using the wheel index: `pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu` and `pip install torch-geometric`.
8. Verify: `python -c "import torch; import torch_geometric; import pyarrow; print('ok')"`.
9. Update `requirements.txt` version lower bounds to match the versions that resolved during install.

**Relevant Context**

- [`oulad_env/pyvenv.cfg`](oulad_env/pyvenv.cfg) — current Homebrew Python 3.14 config to be replaced.
- [`requirements.txt`](requirements.txt) — file to extend and update.
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — calls `to_parquet()`, requires pyarrow.
- [`src/gnn_model.py`](src/gnn_model.py) — torch imports; will be importable after switch.
- `.gitignore` already excludes `oulad_env/` (lines 9, 68) — the rebuilt venv will not be tracked.

---

### Sub-Task 5 — Consolidate and Clean the GNN Notebook

**Status**: `[ ] pending`

**Intent**

The two existing GNN notebooks (`notebooks/OULAD_Graph_Analysis.ipynb`, `notebooks/OULAD_Graph_Analysis2.ipynb`) contain: random label/feature placeholders (`torch.randn(...)`), toy metric printouts, and obsolete troubleshooting cells. These cells produce misleading numbers. Both notebooks are consolidated into a single `notebooks/OULAD_Graph_Analysis_Final.ipynb` with all placeholder and debug content removed.

The final notebook should contain only: data loading, graph construction (delegating to `src/graph_pipeline.py`), integrity-check display, and demonstrations of the split functions added in Sub-Task 6. It must not train any model or report model metrics (those are for next week).

**Expected Outcomes**

- A single `notebooks/OULAD_Graph_Analysis_Final.ipynb` exists; the two old notebooks are deleted.
- All `torch.randn(...)` feature placeholders are removed.
- All cells printing toy/placeholder metrics (e.g., random-label AUROC ~0.5 demos) are removed.
- All troubleshooting / diagnostic cells are removed.
- Remaining cells run top-to-bottom without errors from a fresh kernel.
- The notebook imports `run_pipeline`, `random_student_split`, and `lcpo_split` from `src/` via a `sys.path` insert at the top.

**Todo List**

1. Create `notebooks/OULAD_Graph_Analysis_Final.ipynb` with cells drawn from the best parts of both existing notebooks.
2. Strip all `torch.randn(...)` cells, placeholder metric loops, and troubleshooting markers from the consolidated content.
3. Retain: imports + `sys.path` setup, `run_pipeline()` call (Week 8), validation summary display, `random_student_split` demo, `lcpo_split` demo.
4. Delete `notebooks/OULAD_Graph_Analysis.ipynb` and `notebooks/OULAD_Graph_Analysis2.ipynb`.
5. Restart kernel and run all cells top-to-bottom; confirm no errors.

**Relevant Context**

- `notebooks/OULAD_Graph_Analysis.ipynb` — placeholder feature cells at lines ~2716, 2735, 2762, 2783; `lcpo_split()` stub at ~2481.
- `notebooks/OULAD_Graph_Analysis2.ipynb` — `torch.randn` feature cells at lines ~986-988.
- Deleted: `notebooks/OULAD_graph_schema.ipynb` (already removed from git).

---

### Sub-Task 6 — Implement Reusable random-student and LCPO Split Functions

**Status**: `[ ] pending`

**Intent**

There are no canonical, tested split functions in `src/`. `baseline_evaluation.py` implements student-grouped cross-validation inline, and the notebooks have ad-hoc `lcpo_split()` stubs. This sub-task extracts two reusable functions into `src/oulad_data.py` (or a new `src/splits.py`) so they can be imported by the notebook and by next week's GNN training code.

Requirements:
- `random_student_split(df, val_frac, test_frac, seed)` → `(train_mask, val_mask, test_mask)` where splits are on *unique students*, not rows. The same student must not appear in more than one split.
- `lcpo_split(df, held_out_presentation)` → `(train_mask, test_mask)` where the test set is all enrollments in `held_out_presentation` and train is the rest.
- Both functions must return non-empty train, val (where applicable), and test sets.
- Both functions must work on the enrollment-level supervision DataFrame produced by `build_enrollment_supervision()`.

**Expected Outcomes**

- Functions are defined in `src/oulad_data.py` (or `src/splits.py`).
- A short test (`pytest` or an assertion block) confirms: (a) no student overlap between train and test in `random_student_split`, (b) non-empty splits for all 22 course-presentations in `lcpo_split`.
- The cleaned notebook imports and demonstrates both functions on the Week 8 supervision table.

**Todo List**

1. Add `random_student_split(enrollments_df, val_frac=0.1, test_frac=0.2, seed=42)` to `src/oulad_data.py`. Implementation: get unique student IDs, shuffle with `seed`, slice into train/val/test, return boolean masks on the DataFrame index.
2. Add `lcpo_split(enrollments_df, held_out_module, held_out_presentation)` to `src/oulad_data.py`. Implementation: test mask = rows where `code_module == held_out_module AND code_presentation == held_out_presentation`; train mask = complement.
3. Assert both functions return non-empty partitions and raise a clear error if not.
4. Add a brief test in `tests/test_splits.py` (or equivalent) verifying the no-overlap guarantee for `random_student_split` and non-empty guarantee for `lcpo_split` across all 22 presentations.
5. Replace the ad-hoc `lcpo_split()` stub in the notebook with an import of the new function.

**Relevant Context**

- [`src/oulad_data.py`](src/oulad_data.py) — file to extend.
- [`src/baseline_evaluation.py:78-123`](src/baseline_evaluation.py) — GroupKFold pattern to reference (student-level separation already proven there).
- [`docs/EVALUATION_SPLITS.md`](docs/EVALUATION_SPLITS.md) — specification for each split strategy.
- 22 course-presentations confirmed by graph pipeline (`code_presentation` node table, 22 rows).

---

### Sub-Task 7 — Rebuild Week 8 Graph and Validation Outputs

**Status**: `[ ] pending`

**Intent**

After Sub-Tasks 1–6 change `filter_window()`, imputation logic, and paths, the stored Week 8 artifacts in `results/graph/artifacts/` and `results/graph/validation/` are stale. They must be regenerated to reflect the corrected pipeline.

**Expected Outcomes**

- `results/graph/artifacts/week08_*.parquet` / `*.csv` files are regenerated locally.
- `results/graph/validation/week08_validation_summary.txt` reflects the post-imputation null counts and updated assessment filtering.
- Enrollment count remains 32 593 (the cutoff fix changes only assessment feature values, not enrollment counts).
- Validation passes all integrity checks (zero duplicates, zero dangling edges, label distribution ~52.8% at-risk).
- `results/graph/` artifact files are listed in `.gitignore` (large binary/parquet files are not committed); only the text validation summary and metadata JSON are committed.

**Todo List**

1. From a clean working directory, activate the virtual environment.
2. Run `python src/run_graph_pipeline.py --week 8` from the project root.
3. Inspect `results/graph/validation/week08_validation_summary.txt` — confirm all checks pass.
4. Add `results/graph/artifacts/*.parquet` and `results/graph/artifacts/*.csv` to `.gitignore`; keep `results/graph/validation/` and `results/graph/artifacts/*_metadata.json` tracked.

**Relevant Context**

- [`src/run_graph_pipeline.py`](src/run_graph_pipeline.py) — CLI to invoke.
- [`src/config.py:20-27`](src/config.py) — output path constants.
- Previous validation baseline: 32 593 enrollments, 52.8% at-risk, runtime ~30 s, peak memory ~1 362 MB.

---

### Sub-Task 8 — Full Pipeline Reproducibility Run and Documentation

**Status**: `[ ] pending`

**Intent**

The repository must be reproducible from a fresh clone. This sub-task documents and verifies the exact sequence of commands needed, and produces a short validation report committed to the repository.

**Expected Outcomes**

- `QUICK_START.md` (or a new `docs/REPRODUCIBILITY.md`) contains the exact shell commands to: clone, install, run the graph pipeline, and open the notebook.
- A `docs/validation_report_week8.md` (or similar) records: pipeline version, cutoff, node/edge counts, label distribution, null-handling strategy, integrity check results, and runtime/memory.
- Running those commands from a directory with only a fresh clone + data files produces no errors and regenerates matching artifacts.

**Todo List**

1. Write a step-by-step reproducibility section in `QUICK_START.md` or a new `docs/REPRODUCIBILITY.md` covering: `git clone`, `pip install -r requirements.txt`, placing data files in `data/`, running `python src/run_graph_pipeline.py --week 8`.
2. Write `docs/validation_report_week8.md` summarising: graph schema version, Week 8 cutoff (56 days), node counts (students: 28 785, courses: 22, assessments: 40, vle: 6 364), edge counts, label distribution, integrity results, and peak memory.
3. Verify by running the documented commands in a new terminal from the project root (with `oulad_env` deactivated first).
4. Commit both documents.

**Relevant Context**

- [`QUICK_START.md`](QUICK_START.md) — existing quick-start to extend.
- [`results/graph/validation/`](results/graph/validation/) — validation summaries to source numbers from.
- [`docs/`](docs/) — existing documentation home.

---

### Sub-Task 9 — Push All Changes to Lab Repository

**Status**: `[ ] pending`

**Intent**

The git status shows uncommitted modifications to `src/baseline_evaluation.py` and `src/oulad_data.py`, a deleted notebook, and untracked directories. All corrected code, regenerated artifacts, and new documentation must be committed and pushed so the lab can verify the pipeline.

**Expected Outcomes**

- All modified source files are committed with descriptive messages.
- Large parquet/CSV artifacts are gitignored; validation summaries, metadata JSON, and `tests/` are committed.
- The consolidated notebook (`notebooks/OULAD_Graph_Analysis_Final.ipynb`) is committed; the two old notebooks are deleted in git.
- The deleted `notebooks/OULAD_graph_schema.ipynb` deletion is staged.
- `git status` is clean on `main` after the push.
- The pushed commit history is linear and readable (one commit per logical sub-task is acceptable).

**Todo List**

1. Stage `src/oulad_data.py`, `src/baseline_evaluation.py`, `src/graph_pipeline.py`, `requirements.txt`.
2. Stage `notebooks/OULAD_Graph_Analysis_Final.ipynb`; git-remove the two old notebooks.
3. Stage `.gitignore` changes (artifacts excluded, validation summaries and metadata included).
4. Stage `results/graph/validation/`, `results/graph/artifacts/*_metadata.json`, `tests/test_splits.py`.
5. Stage `docs/validation_report_week8.md` and updated `QUICK_START.md`.
6. Commit with message: `fix: graph pipeline finalization and reproducibility audit`.
7. `git push origin main`.

**Relevant Context**

- Git status at conversation start: `src/baseline_evaluation.py` and `src/oulad_data.py` modified; `notebooks/OULAD_graph_schema.ipynb` deleted; `.bob/`, `.specify/`, `AGENTS.md`, `notebooks/output/` untracked.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — commit message conventions.
