# Week 8 Graph Validation Report

## Pipeline Version

- Source: `src/graph_pipeline.py` (staged 7-function API)
- Entry point: `src/run_graph_pipeline.py --week 8`
- Prediction window: **56 days** from course start (Week 8)

## Key Changes vs. Previous Run

1. **Assessment filtering corrected** — `filter_window()` now applies a dual guard (Strategy B): `assessments.date` (due date) ≤ window **AND** `date_submitted` ≤ window. Both conditions are required to exclude scores that were not yet observable at prediction time.
2. **Explicit null imputation** — `build_node_tables()` fills numeric nulls → 0 and categorical nulls → `"Unknown"` before returning node tables. Pre-imputation null counts are logged for audit.
3. **Data path fixed** — `config.py` `DATA_DIR` corrected from `DATA/raw` to `data/raw` (case-sensitive).
4. **Python environment** — switched from Homebrew Python 3.14 to **pyenv Python 3.11.11**; `pyarrow`, `torch 2.12.1`, and `torch-geometric 2.8.0` installed.

## Node Counts

| Node Type | Count |
|---|---|
| student | 28,785 |
| course_presentation | 22 |
| assessment | 40 |
| vle_resource | 6,364 |

## Edge Counts

| Edge Type | Count |
|---|---|
| enrolled_in | 32,593 |
| contains_assess | 40 |
| has_resource | 6,364 |
| submitted | 47,259 |
| interacted_with | 1,056,217 |

## Enrollment Supervision

- Total enrollments: **32,593**
- At-risk (`target=1`, Fail/Withdrawn): **17,208** (52.8%)
- Success (`target=0`, Pass/Distinction): **15,385** (47.2%)

## Integrity Checks

| Check | Result |
|---|---|
| Duplicate nodes (all types) | 0 ✓ |
| Duplicate edges (all types) | 0 ✓ |
| Duplicate enrollments | 0 ✓ |
| Dangling edge endpoints | 0 ✓ |
| Post-imputation nulls in node features | 0 ✓ |

## Temporal Compliance (cutoff = 56 days)

| Check | Value | Status |
|---|---|---|
| Max VLE interaction date | 56 | ✓ |
| Max assessment due date | 54 | ✓ |
| Max date_submitted | 56 | ✓ |
| Submitted score range | 0.0–100.0 | ✓ |

## Null Handling (pre-imputation, expected from raw data)

| Column | Nulls | Imputed to |
|---|---|---|
| `student.imd_band` | 971 | `"Unknown"` |
| `vle_resource.week_from` | 5,243 | `0` |
| `vle_resource.week_to` | 5,243 | `0` |

## Performance

- Runtime: **6.6 s**
- Peak memory: **1,048.7 MB**

## Artifacts

Saved to `results/graph/artifacts/` (Parquet, gitignored — regenerate with `run_graph_pipeline.py`):

- `week08_nodes_student.parquet`
- `week08_nodes_course_presentation.parquet`
- `week08_nodes_assessment.parquet`
- `week08_nodes_vle_resource.parquet`
- `week08_edges_enrolled_in.parquet`
- `week08_edges_contains_assess.parquet`
- `week08_edges_has_resource.parquet`
- `week08_edges_submitted.parquet`
- `week08_edges_interacted_with.parquet`
- `week08_enrollments.parquet`
- `week08_metadata.json` (**committed**)

Validation reports saved to `results/graph/validation/` (**committed**):

- `week08_integrity.json`
- `week08_validation.json`
- `week08_validation_summary.txt`
