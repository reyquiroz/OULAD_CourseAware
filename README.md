# OULAD Student Success Prediction

## Overview

This repository implements an end-to-end student at-risk prediction pipeline using the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset). The project progresses through two phases:

- **Phase 1 — Tabular Baselines**: Strong LightGBM/XGBoost baselines evaluated under random-student and Leave-Course-Presentation-Out (LCPO) splits across four temporal prediction windows (Weeks 2, 4, 6, 8).
- **Phase 2 — Graph Pipeline** *(current)*: A leakage-safe, enrollment-centric heterogeneous graph built from the same OULAD tables. Graph artifacts, reusable split utilities, and 12 visualizations are committed. GraphSAGE training and GNN vs. LightGBM comparison are planned for the next iteration.

---

## Target Definition

| Label | Meaning | Final Result |
|---|---|---|
| `1 = at-risk` | Positive class — requires intervention | Fail **or** Withdrawn |
| `0 = success` | Negative class | Pass **or** Distinction |

**All reported Precision, Recall, F1, and AUPRC refer to the at-risk class (class 1).**

Class distribution: **52.8% at-risk**, 47.2% success (32,593 total enrollments).

---

## Repository Structure

```
OULAD/
├── src/
│   ├── config.py                    # Centralized paths, hyperparameters
│   ├── oulad_data.py                # Shared data utilities:
│   │                                #   load_oulad_data, filter_window,
│   │                                #   build_features, evaluate_metrics,
│   │                                #   random_student_split, lcpo_split
│   ├── graph_pipeline.py            # 7-stage heterogeneous graph pipeline
│   ├── run_graph_pipeline.py        # CLI entry point: --week {2,4,6,8}
│   ├── graph_validation.py          # Graph integrity + statistics reporting
│   ├── baseline_evaluation.py       # Random-student 5-fold CV (5 models × 4 feature sets)
│   ├── lcpo_evaluation.py           # Leave-Course-Presentation-Out evaluation
│   ├── feature_importance_analysis.py
│   ├── threshold_optimization.py
│   ├── future_presentation_evaluation.py
│   └── gnn_model.py                 # GNN architecture stub (next iteration)
├── notebooks/
│   └── OULAD_Graph_Analysis_Final.ipynb   # Canonical graph notebook (44 cells, 12 charts)
├── tests/
│   └── test_splits.py               # 13 unit tests for split utilities (all passing)
├── data/
│   ├── raw/                         # Place OULAD CSVs here (studentVle.csv gitignored, 433 MB)
│   └── processed/
├── results/
│   ├── baseline/                    # Baseline CV results + plots
│   ├── lcpo/                        # LCPO results
│   ├── feature_importance/
│   ├── threshold_optimization/
│   └── graph/
│       ├── artifacts/               # Graph PNGs + metadata JSON (parquet/csv gitignored)
│       └── validation/              # week08_validation_summary.txt + JSON reports
│   └── comparison/
│       ├── all_splits_comparison.csv              # Unified 4W×5M×3S table
│       └── strategy_a_vs_b_comparison.csv         # Strategy A vs B (run separately)
├── docs/
│   ├── validation_report_week8.md   # Week 8 graph audit trail
│   ├── LEAKAGE_PREVENTION.md
│   ├── EVALUATION_SPLITS.md
│   └── GRAPH_SCHEMA.md
├── .python-version                  # pyenv 3.11.11
├── requirements.txt
└── QUICK_START.md                   # Fresh-clone setup commands
```

---

## Setup

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) with Python 3.11.11 installed
- `studentVle.csv` downloaded from [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) and placed in `data/raw/`

### Install

```bash
git clone https://github.com/BioAI-Systems-Lab/CourseAware.git
cd CourseAware

# Python 3.11.11 is pinned via .python-version — pyenv picks it up automatically
python -m venv oulad_env
source oulad_env/bin/activate
pip install -r requirements.txt

# PyTorch + PyTorch Geometric (CPU)
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1: replace URL with https://download.pytorch.org/whl/cu121

# Place studentVle.csv in data/raw/ (all other CSVs are already in the repo)
# Download from: https://analyse.kmi.open.ac.uk/open_dataset
```

### Build Week 8 Graph

```bash
python src/run_graph_pipeline.py --week 8
```

Outputs: `results/graph/validation/week08_validation_summary.txt` and `results/graph/artifacts/week08_metadata.json`.

---

## Phase 1 — Tabular Baselines

### Approach

- **Supervised unit**: enrollment `(id_student, code_module, code_presentation)`
- **Assessment filtering**: Dual guard (Strategy B) — `assessments.date ≤ window` (due date) **and** `date_submitted ≤ window` (submission date). Both guards prevent temporal leakage; 28.8% of OULAD submissions are submitted after their due date.
- **Student split**: `GroupKFold` on `id_student` — same student cannot appear in both train and test
- **5 models × 4 feature subsets × 4 temporal windows**

### Week 8 Results — All Features, Random-Student 5-Fold CV

| Model | AUROC | AUPRC | F1 | Precision | Recall | Bal. Acc |
|---|---|---|---|---|---|---|
| Majority | 0.500 | 0.527 | 0.690 | 0.527 | 1.000 | 0.500 |
| Logistic Regression | 0.772 | 0.769 | 0.730 | 0.694 | 0.770 | 0.696 |
| Random Forest | 0.825 | 0.806 | 0.777 | 0.741 | 0.816 | 0.749 |
| XGBoost | 0.824 | 0.809 | 0.775 | 0.737 | 0.818 | 0.746 |
| **LightGBM** | **0.835** | **0.823** | **0.788** | **0.740** | **0.842** | **0.757** |

### LCPO Results — LightGBM Week 8

| Split | AUROC | F1 | Bal. Acc |
|---|---|---|---|
| Random-student | 0.835 ± 0.005 | 0.788 ± 0.004 | 0.757 ± 0.005 |
| LCPO | 0.804 ± 0.087 | 0.758 ± 0.066 | 0.726 ± 0.074 |

3–4% AUROC drop from random to LCPO reflects realistic cross-course generalization. High LCPO variance (±0.087) indicates course-specific difficulty — GGG courses AUROC ~0.60–0.63, DDD/FFF/EEE >0.85.

### Temporal Progression (LightGBM, All Features)

| Window | AUROC |
|---|---|
| Week 2 | 0.714 |
| Week 4 | 0.781 |
| Week 6 | 0.812 |
| Week 8 | 0.835 |

---

## Phase 2 — Heterogeneous Graph Pipeline

### Graph Schema (Week 8, cutoff = 56 days)

**Node types**

| Type | Count | Features |
|---|---|---|
| `student` | 28,785 | gender, region, education, imd_band, age_band, disability |
| `course_presentation` | 22 | module, presentation, length |
| `assessment` | 40 | type (TMA/CMA/Exam), weight, due_date |
| `vle_resource` | 6,364 | activity_type, week_from, week_to |

**Edge types**

| Type | Count | Features |
|---|---|---|
| `enrolled_in` | 32,593 | num_of_prev_attempts, studied_credits |
| `contains_assess` | 40 | — |
| `has_resource` | 6,364 | — |
| `submitted` | 47,259 | score |
| `interacted_with` | 1,056,217 | total_clicks, n_interactions, first_day, last_day, active_days |

### Pipeline Stages

```
load_raw_tables()
  → apply_window_cutoff()      # assessments filtered on due_date ≤ window
  → build_node_tables()        # explicit null imputation: numeric→0, categorical→"Unknown"
  → build_edge_tables()
  → build_enrollment_supervision()
  → validate_graph_integrity()
  → materialize_graph_artifacts()
```

**Week 8 validation**: zero duplicates, zero dangling edges, zero post-imputation nulls, 52.8% at-risk rate, runtime ~6.6 s. Full report: [`docs/validation_report_week8.md`](docs/validation_report_week8.md).

### Split Utilities (`src/oulad_data.py`)

```python
# Student-level split — same student never appears in more than one partition
train_mask, val_mask, test_mask = random_student_split(
    enrollments, val_frac=0.1, test_frac=0.2, seed=42
)
# Week 8: 22,801 train / 3,280 val / 6,512 test rows — zero student overlap verified

# Leave-Course-Presentation-Out
train_mask, test_mask = lcpo_split(enrollments, "BBB", "2013J")
# All 22 course-presentations yield non-empty splits — verified by test suite
```

### Tests

```bash
source oulad_env/bin/activate
pytest tests/test_splits.py -v   # 13/13 passing
```

### Canonical Notebook

`notebooks/OULAD_Graph_Analysis_Final.ipynb` — 44 cells, fully executed, 12 embedded charts:

**Statistical charts**: at-risk rate by course, class balance, split quality, enrollments per student, graph scale, VLE degree/click distributions, LCPO split comparison, daily VLE activity.

**Network charts**: heterogeneous graph schema, course × VLE activity-type bipartite, sampled student–course enrollment graph, assessment score vs. VLE clicks scatter.

---

## Key Design Decisions

| Decision | Rule |
|---|---|
| Assessment temporal filter | Dual guard (Strategy B): `assessments.date ≤ window` (due date) **AND** `date_submitted ≤ window` — both required to exclude unobservable scores |
| Null imputation | Numeric → `0`, categorical → `"Unknown"` (consistent across tabular + graph) |
| Supervised unit | Enrollment `(id_student, code_module, code_presentation)` — not student node |
| Student overlap | Guaranteed zero overlap between train and test via `random_student_split` |
| Metrics positive class | At-risk (`target=1`) throughout — Precision, Recall, F1, AUPRC all refer to class 1 |
| Python version | 3.11.11 via pyenv (PyTorch-compatible; pinned in `.python-version`) |
| Std convention | Population std (ddof=0) throughout all comparison CSVs and summary tables |

---

## Next Steps

- [ ] Train GraphSAGE on Week 8 graph using `random_student_split` and LCPO splits
- [ ] Compare GNN vs. LightGBM fairly under identical evaluation conditions
- [ ] Extend graph pipeline to Weeks 2, 4, 6

---

## Citation

```bibtex
@misc{oulad_courseaware_2026,
  title  = {OULAD Student Success Prediction: Tabular Baselines and Heterogeneous Graph Pipeline},
  author = {BioAI Systems Lab},
  year   = {2026},
  url    = {https://github.com/BioAI-Systems-Lab/CourseAware}
}
```

Dataset citation:
> Kuzilek J., Hlosta M., Zdrahal Z. (2017) Open University Learning Analytics dataset. *Scientific Data* 4:170171. doi:10.1038/sdata.2017.171

---

## License

MIT License — see [LICENSE](LICENSE).
