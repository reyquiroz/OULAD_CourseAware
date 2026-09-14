# AGENTS.md

This file provides guidance to agents when working with code in this repository.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

## Stack

Python 3.11.11 (pinned via `.python-version`, managed with pyenv). Virtual env lives in `oulad_env/` at project root.  
ML stack: scikit-learn, XGBoost, LightGBM, PyTorch 2.13.0, PyTorch Geometric 2.8.0.post1.

## Setup

```bash
source oulad_env/bin/activate
```

PyTorch / PyG must be installed via index-URL — plain `pip install -r requirements.txt` fails for those two packages:

```bash
pip install torch==2.13.0 torch-geometric==2.8.0.post1 --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1: use https://download.pytorch.org/whl/cu121
```

## Running Tests

Tests must be run from the **project root** (not from `src/` or `tests/`):

```bash
pytest tests/ -v                                     # all tests
pytest tests/test_splits.py -v                       # single file
pytest tests/test_splits.py::test_no_overlap -v      # single test by name
pytest tests/ -k "test_filter_window" -v             # single test by pattern
pytest tests/test_gnn_data_flow.py -v                # needs artifacts; some tests auto-skip on fresh clone
```

`sys.path.insert(0, .../src)` is done inside each test file — no `PYTHONPATH` export needed.

Tests that require parquet artifacts (`results/graph/artifacts/week08_*.parquet`) are decorated with `@pytest.mark.skipif(not ARTIFACTS_PRESENT, ...)` and skip safely on a fresh clone.

## Key Architecture Constraints

- **Prediction unit is the enrollment** (`id_student, code_module, code_presentation`), NOT the student. One student can appear in multiple enrollments with different outcomes. Never aggregate labels at the student level.
- **Label convention** (in `src/config.py`): `1 = at-risk (Fail/Withdrawn)`, `0 = success (Pass/Distinction)`. This is inverted from some datasets — do not assume 1 = positive class means success.
- **Temporal leakage guard (Strategy B, dual-guard)**: `filter_window()` in `src/oulad_data.py` requires BOTH `due_date ≤ window` AND `date_submitted ≤ window` for assessments. Implementing only one guard is a bug.
- **Split functions** (`random_student_split`, `lcpo_split`) return boolean pandas masks indexing the enrollment DataFrame directly — not integer indices or DataFrames. Shared by both the tabular baseline and GNN pipelines.
- **Graph artifacts are gitignored** — `results/graph/artifacts/week*.parquet` must be regenerated locally by running `python src/run_graph_pipeline.py --week 8` etc.
- `studentVle.csv` (~433 MB) is gitignored and must be downloaded separately from https://analyse.kmi.open.ac.uk/open_dataset. Verify with `python src/check_data.py`.

## Module Imports

`src/` modules import each other as flat names (e.g., `from config import ...`, `from oulad_data import ...`) because scripts are expected to be run with `src/` on the path. When running from the project root, set `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or use the `sys.path.insert` pattern already present in all test files.

## End-to-End Reproducibility

```bash
bash scripts/reproduce_all.sh   # full pipeline (requires graph artifacts + GPU/CPU torch)
python src/run_evaluation.py    # tabular baseline only (~10–15 min)
python src/run_graph_pipeline.py --week 8   # build week-8 graph (~6 s, ~1 GB peak RAM)
```

## Code Style

- `black` + `flake8` are in requirements as dev tools but no config files exist — defaults apply.
- All file paths use `pathlib.Path` via constants from `src/config.py`. Never use string concatenation for paths.
- `RANDOM_STATE = 42` is the canonical seed; GNN experiments accept `--seeds` CLI argument.

## Non-Obvious Gotchas

- **Never aggregate at student level** — prediction target lives on the enrollment triple `(id_student, code_module, code_presentation)`. Adding a student-level label column or deduplicating by student is wrong.
- **`sanitize_feature_names()` is mandatory before XGBoost/LightGBM** — call it after `pd.get_dummies()` every time. XGBoost rejects column names with special characters (brackets, `<`, `>`) which appear naturally in OULAD one-hot encoded categoricals.
- **`gnn_model.py` hardcodes string paths** (`ARTIFACT_DIR = "results/graph/artifacts"`) and `run_gnn_experiment.py` hardcodes `RESULTS_DIR = "results/graph"` — do not follow this pattern in new code; always use the `Path` constants from `src/config.py`.
- **`run_ablation.py` adds `src/` to `sys.path` via `sys.path.insert(0, os.path.dirname(__file__))`** — this is the only script that does this self-insertion; all others rely on caller setting `PYTHONPATH`.
- **GNN has two distinct seed types** — `--seeds` controls the random-student *split*; `--model-seeds` (default `42 123 7 17 99`) controls model *initialisation* and applies only to LCPO folds. Conflating them produces non-reproducible comparisons.
- **GNN prediction head logit count must equal enrolled_in edge count** (32,593 for week 8). `enrolled_in` / `rev_enrolled_in` edges are intentionally NOT filtered during LCPO masking — only `submitted`, `interacted_with`, `contains_assess`, and `has_resource` edges are filtered by destination node. Changing this breaks train/test mask alignment.
- **Reversed edges must be kept in sync** — the heterogeneous graph stores both `enrolled_in` and `rev_enrolled_in` (and `submitted`/`interacted_with`). When masking edges for LCPO, both forward and reverse edge tensors must be updated.
- **Normalization must use train-subset statistics** — always pass `train_edge_mask` to `_normalize_numeric_features()` in `gnn_model.py`. Omitting it silently uses global statistics, leaking test-set distribution into normalization.
- **`matplotlib.use("Agg")` is set at import time** in `evaluation_pipeline.py` — importing it in interactive notebooks will switch the backend to non-interactive.
- **Week numbers map to days** — week 2 → 14, week 4 → 28, week 6 → 42, week 8 → 56. Always look these up via `PREDICTION_WINDOWS` in `src/config.py`; never hardcode day values.
- **`create_datasets()` uses `week * 7` for day conversion internally** — pass week integers (2, 4, 6, 8), not day counts.
- **`_append_or_create_csv()` in `run_gnn_experiment.py`** deduplicates on provided keys (keep="last") when appending to results CSVs — re-running an experiment updates existing rows rather than creating duplicates.
- **Graph artifacts are parquet, not CSV** — `materialize_graph_artifacts()` writes `.parquet` files via pyarrow. Do not convert to CSV for intermediate steps.
