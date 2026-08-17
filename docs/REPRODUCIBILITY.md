# Reproducibility Guide

This document describes how to recreate the committed GraphSAGE vs. LightGBM results for the OULAD project.

## Environment setup

- Python: 3.11.x
- Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

For the pinned PyTorch packages in [`requirements.txt`](../requirements.txt), use the PyTorch wheel index noted in that file if plain pip resolution does not find compatible wheels.

## Key hyperparameters

| Parameter | Value |
| --- | --- |
| hidden_dim | 64 |
| learning_rate | 1e-3 |
| patience | 20 |
| lcpo_patience | 50 |
| max_epochs | 200 |
| random seeds | 42, 123, 7 |
| ablation seeds | 42 |

## Split file locations

- Random/LCPO split root: [`results/graph/evaluation/`](../results/graph/evaluation/)
- Week-specific split directories:
  - [`results/graph/evaluation/week02/splits/`](../results/graph/evaluation/week02/splits/)
  - [`results/graph/evaluation/week04/splits/`](../results/graph/evaluation/week04/splits/)
  - [`results/graph/evaluation/week06/splits/`](../results/graph/evaluation/week06/splits/)
  - [`results/graph/evaluation/week08/splits/`](../results/graph/evaluation/week08/splits/)
- LCPO fold definitions are stored as `weekXX_lcpo_folds.csv` inside each split directory.

## Exact regeneration commands

Run the full pipeline with [`scripts/reproduce_all.sh`](../scripts/reproduce_all.sh):

```bash
bash scripts/reproduce_all.sh
```

Equivalent step-by-step commands:

```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
python src/save_graph_splits.py --weeks 2 4 6 8
python src/run_gnn_experiment.py --weeks 8 --seeds 42 123 7
python src/run_gnn_experiment.py --weeks 2 4 6 --seeds 42 123 7 --random-only
python src/compare_gnn_lgbm.py --weeks 2 4 6 8 --seeds 42 123 7
python src/run_ablation.py --week 8 --seeds 42
python src/course_variation.py --week 8
python src/generate_report_figures.py
```

## Result files produced

Core regenerated outputs live under [`results/graph/`](../results/graph/), including:

- `random_student_results.csv`
- `lcpo_results.csv`
- `lcpo_summary.csv`
- `comparison_results.csv`
- `comparison_summary.md`
- `course_variation.csv`
- `ablation_results.csv`
- [`results/graph/figures/`](../results/graph/figures/)
- [`results/graph/tables/`](../results/graph/tables/)

## Committed vs. regenerated artifacts

Committed to the repository:

- Result CSV/Markdown files in [`results/graph/`](../results/graph/)
- Split definitions in [`results/graph/evaluation/`](../results/graph/evaluation/)
- Scripts and documentation under [`src/`](../src/) and [`docs/`](../docs/)

Must be regenerated from raw data if missing or stale:

- Graph artifacts under [`results/graph/artifacts/`](../results/graph/artifacts/)
- Training curve `.npz` files
- Any figures/tables after upstream result changes

The raw OULAD CSV files are expected in `data/raw/` and are not recreated by this repository.