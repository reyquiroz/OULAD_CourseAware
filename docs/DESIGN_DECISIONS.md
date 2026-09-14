# Design Decisions

This document records three design decisions that are already implemented correctly in the
codebase. Each section explains what the decision is, why it was made, where it is enacted,
and what evidence in the code confirms it. These notes exist so that reviewers can verify
correctness without reading through the full implementation.

---

## 1. Training-Only Normalization

### What

`_normalize_numeric_features()` in [`src/gnn_model.py`](../src/gnn_model.py) (function
definition at line 54) standardizes numeric columns in node and edge feature tensors.
When a `train_edge_mask` is provided, mean and standard deviation are computed **from
training edges and training student nodes only**; those statistics are then applied to the
full graph (train + val + test).

### Why

Computing normalization statistics from the full graph (including test rows) would
constitute feature leakage: the model would implicitly receive information about the test
distribution during training. Restricting statistics to the training partition ensures
that the test set is a genuinely held-out evaluation.

### Where

- **Function definition**: [`src/gnn_model.py`](../src/gnn_model.py) lines 54–194
  (`_normalize_numeric_features`). The `train_edge_mask` parameter is documented in the
  function docstring (lines 68–74). The inner `_standardize` helper at line 95 accepts an
  optional `ref_rows` mask; when provided, `mean` and `std` are computed from
  `tensor[ref_rows]` only (lines 119–121).
- **Random-split call site**: [`src/run_gnn_experiment.py`](../src/run_gnn_experiment.py)
  line 347 — called immediately after the split masks are created, passing
  `train_edge_mask=train_mask`.
- **LCPO call site**: [`src/run_gnn_experiment.py`](../src/run_gnn_experiment.py) line 498
  — called after the per-fold train mask is assembled, before the held-out edges are masked
  out.

### Evidence

- `_normalize_numeric_features` signature at line 54: `train_edge_mask: torch.BoolTensor = None`.
- `_standardize` inner function (line 95): `ref_rows` parameter; line 119:
  `ref_col = col[ref_rows] if ref_rows is not None else col`; lines 120–121: stats computed
  on `ref_col` only.
- `enrolled_in` edge attrs use `train_edge_mask` directly (line 170: `ref_rows=train_edge_mask`).
- Student node features use a node-level boolean mask derived from training enrollment
  student indices (lines 131–137; applied at line 147).
- `submitted` and `interacted_with` edges derive their reference mask from the same
  training student set (lines 173–188).

---

## 2. Inductive Graph Evaluation

### What

`build_train_subgraph()` in [`src/gnn_model.py`](../src/gnn_model.py) (function definition
at line 201) constructs a new `HeteroData` object that contains **only training enrollment
edges** for message passing. Test enrollment edges are entirely absent from the subgraph
used during training forward passes.

### Why

A transductive setting would allow the GNN to aggregate over test student nodes during
training, effectively leaking the graph structure of test enrollments into the learned
representations. Removing test edges enforces the inductive setting: the model must
generalize to new student–course relationships at inference time without having seen those
connections during training.

### Where

- **Function definition**: [`src/gnn_model.py`](../src/gnn_model.py) lines 201–299
  (`build_train_subgraph`). The docstring (lines 202–224) explicitly states that
  `enrolled_in` / `rev_enrolled_in` are filtered to training rows; `submitted` /
  `interacted_with` are filtered to training students; all node feature tensors are
  retained at their global indices.
- **Random-split call site**: [`src/run_gnn_experiment.py`](../src/run_gnn_experiment.py)
  line 351 — called after normalization, passing the full data object and `train_mask`.
- **LCPO call site**: [`src/run_gnn_experiment.py`](../src/run_gnn_experiment.py) line 505
  — called on `data_masked` (which has the held-out course-presentation's edges already
  removed) plus `train_mask` to further restrict to within-fold training rows.

### Evidence

- `build_train_subgraph` docstring, line 220: "enrolled_in / rev_enrolled_in filtered to
  training rows only."
- Line 250: `sub[ei_key].edge_index = ei_store.edge_index[:, train_mask]` — only training
  columns of the edge index are retained.
- Line 258: `rev_enrolled_in` is recomputed as the transpose of the already-filtered
  `enrolled_in` edge index, keeping forward and reverse edges in sync.
- Lines 264–289: `submitted` and `interacted_with` edges are filtered by checking whether
  their source student index appears in `train_student_nodes` (the set derived from the
  filtered `enrolled_in` edge index).
- Lines 240–246: all node feature tensors (`store.x`) are copied as-is — global node
  indices are preserved so message-passing from non-enrolled structural nodes is unaffected.

---

## 3. LightGBM Feature and Split Matching

### What

The LightGBM comparison pipeline in [`src/compare_gnn_lgbm.py`](../src/compare_gnn_lgbm.py)
uses the **same split function, same seeds, same prediction-window filter (Strategy B
dual-guard), and same fold definitions** as the GNN pipeline. This ensures that GNN vs.
LightGBM comparisons are made on identical data partitions.

### Why

A fair model comparison requires that both models are trained and evaluated on exactly the
same samples under exactly the same temporal constraints. Differences in split strategy,
seed, or feature window would conflate data-setup differences with architectural differences,
making the comparison misleading.

### Where

- **Feature construction**: [`src/compare_gnn_lgbm.py`](../src/compare_gnn_lgbm.py)
  `build_tabular_features()` at line 128. It calls `filter_window()` (line 152) with
  `submission_date_guard=True` (line 154) — the Strategy B dual-guard that filters on both
  `due_date ≤ window` AND `date_submitted ≤ window`.
- **Random-split protocol**: `run_lgbm_random_split()` at line 207. It imports and calls
  `random_student_split()` (line 224 / line 238) with the same `seed` argument passed by
  the caller — mirrors the GNN's `run_random_split_experiment` which calls the identical
  function with the same seed.
- **LCPO protocol**: `run_lgbm_lcpo()` at line 268. It reads fold definitions from
  `results/graph/evaluation/week{N}/splits/week{N:02d}_lcpo_folds.csv` (line 290) — the
  same CSV produced by `src/save_graph_splits.py` and consumed by the GNN LCPO loop.

### Evidence

- `build_tabular_features` line 154: `submission_date_guard=True` — confirms Strategy B
  dual temporal guard is active (see also `filter_window()` docstring in
  [`src/oulad_data.py`](../src/oulad_data.py)).
- `run_lgbm_random_split` line 238: `train_s, val_s, test_s = _random_student_split(enrollments, seed=seed)` — same utility as the GNN path; both use 70/10/20 fractions (the
  default in `random_student_split`).
- `run_lgbm_lcpo` line 290: fold CSV path mirrors the GNN LCPO fold path constant in
  `src/run_gnn_experiment.py`, so both models see identical held-out course-presentations
  per fold.
- `run_lgbm_random_split` line 232–234: features are aligned to the canonical enrollment
  ordering from `week{N:02d}_enrollments.parquet` before split masks are applied — the same
  parquet artifact used by the GNN graph loader.

---

## 4. Zenodo Dataset — Schema Divergences and Pipeline Adaptations

### What

The KU Leuven Zenodo dataset (Tiukhova et al., 2024; record 17087849) has a structurally
different schema from OULAD. The pipeline in [`src/run_zenodo_pipeline.py`](../src/run_zenodo_pipeline.py)
and [`src/zenodo_data.py`](../src/zenodo_data.py) implements OULAD-compatible abstractions
with four specific divergences documented here.

### Divergence 1 — No assessment data → single temporal guard

**OULAD** applies a dual temporal guard in `filter_window()` ([`src/oulad_data.py`](../src/oulad_data.py)):
both `due_date ≤ window` AND `date_submitted ≤ window` must hold, preventing label leakage
via early assessment submissions.

**Zenodo** has no `studentAssessment` or `assessments` table at all. The `filter_window_zenodo()`
function in [`src/zenodo_data.py`](../src/zenodo_data.py) therefore applies only a single
guard: `day_offset ≤ window_days`. This is not a leakage risk for Zenodo (there is nothing
to leak), but it means the two datasets cannot be compared on assessment-derived features.
The `submitted` and `contains_assess` edge types are absent from the Zenodo graph entirely.

**Evidence**: [`src/zenodo_data.py`](../src/zenodo_data.py) comment at line 264:
`# GAP: no dual guard for submission-date (no assessment submissions exist).`

### Divergence 2 — Course-section name normalisation

The Zenodo `course_participation` table records the 2021 Global economics cohort as two
separate sections (`Global economics 1`, `Global economics 2`). The `course_content` and
`log_activity` tables use the base name `Global economics`. If not normalised, 676 of 4,292
enrollment rows would fail to map to a `course_presentation` node.

**Fix applied**: [`src/zenodo_data.py`](../src/zenodo_data.py) `build_oulad_compatible_tables()`
normalises `student_info["code_module"]` via `str.replace(r"\s+\d+$", "", regex=True)`,
matching the normalisation already applied to `vle` and `courses`. Twelve students enrolled
in both sections have identical `target` values; `drop_duplicates` on the enrollment key
retains one row per student.

**Evidence**: `zenodo_data.py` line 199 (`"code_module": part["COURSE_ID"].str.replace(...)`)
and line 214 (`vle["code_module"] = vle["code_module"].str.replace(...)`).

### Divergence 3 — No student demographic features → constant node features

OULAD student nodes carry 11 demographic and academic-history features (gender, age_band,
IMD band, disability, etc.). The Zenodo `course_participation` table contains no such fields.

**Fix applied**: `_build_zenodo_hetero_data()` in [`src/run_zenodo_pipeline.py`](../src/run_zenodo_pipeline.py)
sets `data["student"].x = torch.ones(n_stu, 1, dtype=torch.float32)` — a single constant
placeholder so the GNN's student embedding layer has a valid input dimension. Similarly,
`enrolled_in` edge attributes are a constant 1.0 vector (no age_band, credit load, etc.).

**Evidence**: `run_zenodo_pipeline.py` comment: `# GAP: student nodes have no feature
attributes — a single constant (1.0) is used as a placeholder`.

### Divergence 4 — Timestamps are calendar datetimes, not day offsets

OULAD precomputes day offsets from module registration dates. Zenodo logs contain ISO
datetime strings; day offsets must be computed at query time from `course_info.json` start dates:
`(TIMESTAMP.date() − course_start).days`.

**Fix applied**: `filter_window_zenodo()` in [`src/zenodo_data.py`](../src/zenodo_data.py)
reads the per-(year, course) start date from the `start_dates` lookup table and computes
`day_offset` dynamically at line 283:
`df["day_offset"] = (df["TIMESTAMP"] - course_start).dt.days`.

**Evidence**: `zenodo_data.py` docstring comment: `# GAP: timestamps are calendar datetimes,
not day-offsets — day offsets are computed from course_info.json start dates`.

### Impact on comparability

These divergences mean the Zenodo GNN uses a reduced graph (3 edge types vs 5 in OULAD;
no student features) and a weaker feature set (VLE-only vs VLE + assessment). The cross-dataset
figure (`results/graph/figures/fig_cross_dataset.png`) shows OULAD GNN AUROC ~0.84 vs Zenodo
GNN ~0.55 (random-split), consistent with the smaller dataset size (4,280 enrollments vs
32,593) and missing feature modalities.

---
