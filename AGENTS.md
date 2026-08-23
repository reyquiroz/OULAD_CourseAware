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
pytest tests/ -v                            # all tests
pytest tests/test_splits.py -v              # single file
pytest tests/test_gnn_data_flow.py -v       # needs artifacts; some tests auto-skip on fresh clone
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
