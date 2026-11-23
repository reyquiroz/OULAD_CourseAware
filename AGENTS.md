# AGENTS.md

This file provides guidance to agents when working with code in this repository.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

## Setup and commands

- Python 3.11.11 (`.python-version`), venv in `oulad_env/` (`source oulad_env/bin/activate`).
- Install `torch` and `torch-geometric` in **two separate** `pip install` calls with `--index-url` for torch (see comments in `requirements.txt`). `pip install -r requirements.txt` alone fails for them.
- Run pytest from the **project root**: `pytest tests/ -v`, `pytest tests/test_splits.py::TestRandomStudentSplit::test_no_overlap_train_test -v`, `pytest tests/ -k filter_window`. `tests/test_gnn_data_flow.py` reads `GRAPH_ARTIFACTS_DIR` from `src/config.py` and skips itself when the week-8 parquet is absent.
- No `black`/`flake8`/pytest config files exist, so defaults apply.
- Style (from `CONTRIBUTING.md`, not enforced by tooling): PEP 8, max line length 100, type hints where appropriate, Google-style docstrings on all functions and classes.
- `studentVle.csv` (~433 MB) is gitignored; download it into `data/raw/` and verify with `python src/check_data.py`.
- Full pipeline: `bash scripts/reproduce_all.sh`. Graph artifacts for one week: `python src/run_graph_pipeline.py --week 8`. Tabular baseline: `python src/run_evaluation.py`.
- `src/` modules use flat imports (`from config import ...`). Most scripts self-insert `src/` into `sys.path`, so `python src/<script>.py` works from the root; test files do the same.

## Results directory trap

- `src/config.py` sets `RESULTS_DIR = results/results/`, and `results/README.md` calls it the only canonical tree. The top-level `results/graph/`, `results/matched/` etc. are legacy copies, and `results/results_bk/` is a frozen backup. Never write to either.
- `.gitignore`, `scripts/reproduce_all.sh` and `tests/test_gnn_data_flow.py` still reference `results/graph/...`, whereas the code reads `GRAPH_ARTIFACTS_DIR` = `results/results/graph/artifacts`. Parquet artifacts currently exist in both trees. If a GNN script or test cannot find artifacts, check which tree it reads before regenerating anything.
- Always use `pathlib.Path` constants from `src/config.py` (`GRAPH_ARTIFACTS_DIR`, `MATCHED_RESULTS_DIR`, ...) and `get_results_path()`. Never hardcode result paths.

## Correctness rules (violations silently corrupt results)

- **Prediction unit is the enrollment** `(id_student, code_module, code_presentation)`. Never deduplicate or label at student level.
- **Label 1 = at-risk** (Fail/Withdrawn), 0 = Pass/Distinction (`LABEL_MAPPING` in `src/config.py`).
- **Dual temporal guard**: assessment data needs both `due_date ≤ window` and `date_submitted ≤ window` (`filter_window()` in `src/oulad_data.py`). `submission_date_guard=False` is the legacy Strategy A, which leaks.
- **Splits return boolean Series** aligned to the enrollment index: `random_student_split` returns `(train, val, test)`, `lcpo_split` returns `(train, test)`. Do not convert them to index arrays.
- **Weeks, not days**: pass 2/4/6/8 to `--week` and `create_datasets()` (which does `week * 7` internally). Day values live in `PREDICTION_WINDOWS` (keys `"week_2"`, ...).
- Call **`sanitize_feature_names()`** after every `pd.get_dummies()` before XGBoost/LightGBM. Column names with `[ ] < >` crash them.

## GNN specifics

- Logits must equal the number of `enrolled_in` edges (26,522 at week 8 after `exclude_early_unregistered`; the count differs by week); the loss is computed per enrollment edge.
- LCPO masking (`run_gnn_experiment.py`) filters `submitted`, `interacted_with`, `contains_assess` and `has_resource` by **destination node**. `enrolled_in`/`rev_enrolled_in` are intentionally never filtered. Where a forward edge tensor is filtered, update its `rev_*` counterpart too.
- Pass `train_edge_mask` to `_normalize_numeric_features()` in `src/gnn_model.py`. Without it, normalization silently uses global statistics and leaks test data.
- Two seed types: `--seeds` is the random-student split seed (default `[42]`); `--model-seeds` (default `42 123 7 17 99`) is the init seed and applies only to LCPO folds.
- `--condition` selects the variant: `full`, `with_enrollment_summaries`, `with_iw_attrs`, `enrollment_node`, `edge_aware_mp`, `temporal_features` (alias `temporal`).
- `_append_or_create_csv()` (in both `run_gnn_experiment.py` and `run_matched_comparison.py`) merges with `drop_duplicates(keep="last")` on the keys you pass. Omitting a distinguishing key such as `condition` makes new rows silently overwrite old ones. Re-runs are idempotent by design.
- Graph artifacts are parquet (pyarrow) and gitignored, except `*_metadata.json`.
- Importing `src/evaluation_pipeline.py` calls `matplotlib.use("Agg")`, which is permanent for that kernel.
