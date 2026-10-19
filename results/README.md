# Results Directory Structure

This directory contains experimental results, graph artifacts, and model evaluation metrics for the OULAD project.

## Canonical Results Directory

- **Authoritative Path**: `results/results/` is the single canonical and authoritative results tree.
- All evaluation scripts, experiments, and pipelines read from and write to `results/results/` (and its subdirectories) via the `pathlib.Path` constants defined in `src/config.py`.

### Structure of `results/results/`:
- `results/results/graph/`: GNN experiment outputs, metrics, training curves, and graph artifacts (`artifacts/`).
- `results/results/baseline/`: Tabular baseline models (RandomForest, LightGBM, MLP, etc.) and metrics.
- `results/results/matched/`: Matched comparison experiments across five seeds (`lgbm_matched_results.csv`, `mlp_matched_results.csv`).
- `results/results/zenodo/`: Replications and baselines for the Zenodo benchmark dataset.
- `results/results/lcpo/`: Leave-Course-Presentation-Out (LCPO) evaluation metrics.
- `results/results/threshold_optimization/`: Threshold search and optimization results.
- `results/results/feature_importance/`: Feature importance analyses.
- `results/results/comparison/`: Comparative summaries between tabular and graph models.

## Legacy / Archive Directories

- Sub-directories located directly at the top level of `results/` (e.g. `results/graph/`, `results/matched/`, `results/zenodo/`, etc.) are legacy/archive copies preserved for reference.
- `results/results_bk/` is an immutable backup snapshot taken during consolidation.
- **Do not write new results to the top-level sub-directories.** All new outputs must be written to `results/results/`.
