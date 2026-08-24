# Reporting Structure

All cross-dataset result CSVs must include a `dataset` column, with [`dataset`](src/reporting_schema.py:8) as the primary grouping key. Results are reported separately per dataset; aggregates must never pool rows from different datasets.

Planned structure (migration deferred until a second dataset exists): `results/oulad/`, `results/{dataset2}/`, each containing graph artifacts, tables, figures, and comparison outputs. Current OULAD outputs remain under `results/graph/` until migration is needed.

To add a new dataset: (1) populate the `dataset` column in that dataset’s outputs, (2) run the dataset-specific pipeline, (3) append the new rows to `results/graph/comparison_results.csv`, (4) run [`main()`](src/verify_results.py:192) via `python src/verify_results.py`, and (5) regenerate figures/tables with [`main()`](src/generate_report_figures.py:484).

Rule: no pooled mean, standard deviation, win count, or figure may be computed before grouping by dataset.
