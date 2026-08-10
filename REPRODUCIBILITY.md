# Reproducibility Guide

This document records the exact commands, environment, and outputs needed to
reproduce all pipeline results from a fresh clone of this repository. All
commands were verified on the environment described below.

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.11.11 (pinned via `.python-version` — pyenv picks it up automatically) |
| Platform | macOS 15.6, arm64 |
| Git commit | `77765ab` |

### Key package versions

| Package | Version |
|---------|---------|
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.6.0 |
| xgboost | 3.2.0 |
| torch | 2.12.1 |
| torch-geometric | 2.8.0 |
| pyarrow | 24.0.0 |
| networkx | 3.6.1 |
| matplotlib | 3.11.0 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |

---

## Step-by-Step Reproduction

### 1. Clone and set up the environment

```bash
git clone https://github.com/BioAI-Systems-Lab/CourseAware.git
cd OULAD_CourseAware

# Python 3.11.11 is pinned in .python-version
# If using pyenv: pyenv install 3.11.11 (if not already installed)
python -m venv oulad_env
source oulad_env/bin/activate   # Windows: oulad_env\Scripts\activate

pip install -r requirements.txt

# PyTorch + PyTorch Geometric (CPU build)
pip install torch==2.12.1 torch-geometric==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu
# For CUDA 12.1: replace URL with https://download.pytorch.org/whl/cu121
```

### 2. Download the OULAD data

All OULAD CSVs except `studentVle.csv` are already tracked in the repository.
Download `studentVle.csv` (433 MB) separately:

```
https://analyse.kmi.open.ac.uk/open_dataset
```

Place it at `data/raw/studentVle.csv`. Verify all files are present:

```bash
python src/check_data.py
```

Expected output: `All required data files present. ✓`

### 3. Run the test suite

```bash
source oulad_env/bin/activate
pytest tests/ -v
```

**Expected output** (verified output):
```
============================= test session starts ==============================
platform darwin -- Python 3.11.11, pytest-9.1.1, pluggy-1.6.0
collected 24 items

tests/test_filter_window.py::TestVleBoundary::test_interaction_at_cutoff_included PASSED
tests/test_filter_window.py::TestVleBoundary::test_interaction_before_cutoff_included PASSED
tests/test_filter_window.py::TestVleBoundary::test_interaction_after_cutoff_excluded PASSED
tests/test_filter_window.py::TestSubmissionDateBoundary::test_submitted_at_cutoff_included PASSED
tests/test_filter_window.py::TestSubmissionDateBoundary::test_submitted_one_day_before_cutoff_included PASSED
tests/test_filter_window.py::TestSubmissionDateBoundary::test_submitted_one_day_after_cutoff_excluded PASSED
tests/test_filter_window.py::TestDueDateBoundary::test_due_at_cutoff_included PASSED
tests/test_filter_window.py::TestDueDateBoundary::test_due_one_day_before_cutoff_included PASSED
tests/test_filter_window.py::TestDueDateBoundary::test_due_one_day_after_cutoff_excluded PASSED
tests/test_filter_window.py::TestDualGuard::test_due_in_window_submitted_after_excluded PASSED
tests/test_filter_window.py::TestDualGuard::test_both_guards_satisfied_included PASSED
tests/test_splits.py::TestRandomStudentSplit::test_no_overlap_train_test PASSED
tests/test_splits.py::TestRandomStudentSplit::test_no_overlap_train_val PASSED
tests/test_splits.py::TestRandomStudentSplit::test_no_overlap_val_test PASSED
tests/test_splits.py::TestRandomStudentSplit::test_all_splits_nonempty PASSED
tests/test_splits.py::TestRandomStudentSplit::test_masks_cover_all_rows PASSED
tests/test_splits.py::TestRandomStudentSplit::test_reproducibility PASSED
tests/test_splits.py::TestRandomStudentSplit::test_different_seed_different_split PASSED
tests/test_splits.py::TestRandomStudentSplit::test_raises_when_too_few_students PASSED
tests/test_splits.py::TestLcpoSplit::test_test_set_is_held_out_presentation PASSED
tests/test_splits.py::TestLcpoSplit::test_train_excludes_held_out PASSED
tests/test_splits.py::TestLcpoSplit::test_masks_are_complement PASSED
tests/test_splits.py::TestLcpoSplit::test_all_presentations_give_nonempty_splits PASSED
tests/test_splits.py::TestLcpoSplit::test_raises_for_unknown_presentation PASSED

============================== 24 passed in 1.85s ==============================
```

### 4. Build the graph datasets for all four prediction weeks

```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
```

Each run prints a validation summary. Key expected lines per week:

| Week | submitted edges | interacted_with edges | Runtime | Memory |
|------|----------------|-----------------------|---------|--------|
| 2 | 1,089 | 634,723 | ~5 s | 915.5 MB |
| 4 | 21,393 | 835,935 | ~5 s | 915.5 MB |
| 6 | 28,569 | 952,241 | ~6 s | 915.5 MB |
| 8 | 44,927 | 1,056,217 | ~6 s | 1,049.0 MB |

All weeks print `Overall temporal compliance: ✓` and show zero duplicates and
zero dangling edges. Committed validation outputs are in
`results/graph/validation/`.

### 5. Regenerate the multi-week summary

```bash
python src/summarize_graph_weeks.py
```

Overwrites `results/graph/validation/all_weeks_summary.csv` and
`docs/graph_validation_summary.md`.

### 6. Verify split integrity

```bash
python src/verify_splits.py
```

**Expected output** (verified):
```
=== Random-Student Split ===
  [PASS] Week 2 random-student: train∩val student overlap = 0
  [PASS] Week 2 random-student: train∩test student overlap = 0
  [PASS] Week 2 random-student: val∩test student overlap = 0
  [PASS] Week 2 random-student: all enrollment rows covered by a split
  [PASS] Week 2 random-student: train is non-empty
  [PASS] Week 2 random-student: val is non-empty
  [PASS] Week 2 random-student: test is non-empty
  ... (identical PASS for Weeks 4, 6, 8)

=== LCPO Split ===
  [PASS] Week 2 LCPO: 22 folds present (found 22)
  [PASS] Week 2 LCPO: no held-out presentation found in any train set
  ... (identical PASS for Weeks 4, 6, 8)

=== Future-Presentation Split ===
  [PASS] Week 2 future-presentation: 2014J in train set = 0 (must be 0)
  [PASS] Week 2 future-presentation: 2013B/J/2014B in test set = 0 (must be 0)
  [PASS] Week 2 future-presentation: train is non-empty
  [PASS] Week 2 future-presentation: test is non-empty
  ... (identical PASS for Weeks 4, 6, 8)

RESULT: ALL CHECKS PASS
```

56 / 56 checks pass (3 strategies × 4 weeks).

---

## What Gets Generated vs. What Is Already Committed

| File type | Location | Status |
|-----------|----------|--------|
| Graph parquet artifacts (`week{N}_*.parquet`) | `results/graph/artifacts/` | Gitignored — regenerate with Step 4 |
| Metadata JSON (`week{N}_metadata.json`) | `results/graph/artifacts/` | **Committed** |
| Validation JSON + TXT | `results/graph/validation/` | **Committed** |
| Split parquet + CSV files | `results/graph/evaluation/` | **Committed** |
| Evaluation result CSVs | `results/baseline/`, `results/lcpo/`, etc. | **Committed** |
| Python environment (`oulad_env/`) | Root | Gitignored — recreate with Step 1 |
| `studentVle.csv` | `data/raw/` | Gitignored — download from OULAD |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pandas'`**
→ Virtual environment not activated. Run `source oulad_env/bin/activate`.

**`FileNotFoundError: data/raw/studentVle.csv`**
→ Download from https://analyse.kmi.open.ac.uk/open_dataset and place at
`data/raw/studentVle.csv`.

**`git filter-repo: command not found`**
→ Install via pip: `pip install git-filter-repo`

**PyTorch install fails**
→ Try the CPU-only wheel explicitly:
`pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu`
