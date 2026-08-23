# GNN Correctness & Reproducibility Plan

**Goal**: Address all eight correctness, parity, and reproducibility issues in the OULAD GNN pipeline.
**Scope**: `src/gnn_model.py`, `src/run_gnn_experiment.py`, `src/compare_gnn_lgbm.py`, `src/generate_report_figures.py`, `tests/`, `scripts/reproduce_all.sh`.
**Approach**: Each sub-task is self-contained, ordered so later tasks depend on stable interfaces from earlier ones.

## Confirmed Design Decisions

| # | Decision |
|---|---|
| **Graph separation** | Inductive subgraph: held-out / val / test student `enrolled_in` edges are excluded from training message-passing. Val/test inference uses the full graph (transductive). Logit indices are local to the subgraph; test inference is separate on the full graph using `test_mask`. |
| **Student representation** | Remove the three `pyg_scatter` injections. `submitted` and `interacted_with` edges remain in the graph and contribute via message-passing topology only — their attributes no longer aggregate into the student node. `enrolled_in` edge attributes are projected and concatenated in the edge prediction head. |
| **LCPO model seeds** | 5 model seeds per fold (`[42, 123, 7, 17, 99]`). Val-student sampling RNG keyed on fold index only (decoupled from model seed). |

---

## Sub-Task 1 — Training-only normalisation

**Status**: `[ ] pending`

### Intent
`_normalize_numeric_features()` currently computes mean/std over the full graph (all 32 593 enrollments), leaking validation and test statistics into training. Normalization parameters must be estimated from training-enrolled edges/nodes only and then applied to the full graph.

### Expected Outcomes
- `_normalize_numeric_features` accepts optional `train_mask` (boolean tensor aligned to `enrolled_in` edges).
- When `train_mask` is supplied, mean/std for each numeric column are computed on training rows only; the computed parameters are then applied to all rows.
- Node feature statistics that cannot be indexed by enrollment (student, course_presentation, assessment, vle_resource) are computed from the node subset that participates in training enrollments (student nodes whose `node_id` appears in the training `enrolled_in` src indices; all course_presentation / assessment / vle_resource nodes are structure nodes and can be normalized globally — document this decision).
- Existing call site in `GraphDataLoader.load()` passes `train_mask=None` (backward compatible, computes global stats as before); callers that know the split pass the mask.
- New unit test `tests/test_normalization.py` verifies that statistics differ when computed on a 50% train subset vs. the full set.

### Todo List
1. Refactor `_normalize_numeric_features(data, train_edge_mask=None)` in `src/gnn_model.py`:
   - For `enrolled_in` edge attrs: if `train_edge_mask` provided, compute mean/std on `edge_attr[train_edge_mask]`, apply to all rows.
   - For `submitted` and `interacted_with` edge attrs: derive a student-level boolean from the train enrollment student indices; use only edges originating from training students.
   - For node features: derive training student node indices from `train_edge_mask`; compute student node stats on that subset; compute structural node stats globally.
   - Return the normalised `data` plus a `NormStats` dict of `{tensor_key: (means, stds)}` for reproducibility.
2. Update `run_random_split_experiment()` in `src/run_gnn_experiment.py`:
   - Pass `train_mask` (the boolean enrollment mask) into `GraphDataLoader.load()` or call `_normalize_numeric_features` again after load with the mask.
   - Preferred: add `train_mask` parameter to `GraphDataLoader.load()` so loading + normalisation are atomic.
3. Update `run_lcpo_experiment()` similarly — pass `train_mask_np` after it is computed.
4. Write `tests/test_normalization.py` covering: (a) stats differ with vs. without mask, (b) all-data stats equal old behaviour when mask is `None`, (c) normalised output is zero-mean unit-variance on training slice.

### Relevant Context
- `src/gnn_model.py` lines 55–110: `_normalize_numeric_features`
- `src/gnn_model.py` lines 200–340: `GraphDataLoader.load()` — line 337 calls `_normalize_numeric_features`
- `src/run_gnn_experiment.py` lines 260–270: where `GraphDataLoader(week).load()` is called

---

## Sub-Task 2 — Inductive subgraph isolation (held-out students excluded from message-passing)

**Status**: `[ ] pending`

### Intent
The full graph (including held-out/validation/test student nodes and edges) currently passes through the GNN during training forward passes. Training must use an inductive subgraph that excludes held-out `enrolled_in` edges and the student-to-resource/assessment edges of held-out students. Validation/test inference can then be done transductively on the full graph (standard GraphSAGE inductive setup) since node features are fixed at load time.

### Expected Outcomes
- A new function `build_train_subgraph(data, train_mask)` in `src/gnn_model.py` returns a `HeteroData` copy where:
  - `enrolled_in` / `rev_enrolled_in` edges are filtered to only train-mask rows.
  - `submitted` and `interacted_with` edges are filtered to only student nodes in the training set.
  - Node feature tensors and structural edges (`contains_assess`, `has_resource`, `rev_*`) are kept intact so course-structure context is preserved.
  - The `y` tensor and `enrollment_idx` on `enrolled_in` are re-indexed to match the filtered edge set.
- `run_training_loop()` uses `train_subgraph` for the forward pass + loss; the original full `data` is used only for validation/test inference.
- `run_random_split_experiment()` and `run_lcpo_experiment()` are updated accordingly.
- Existing test `test_lcpo_mask_does_not_strip_cross_course_edges` still passes.
- New unit test `tests/test_subgraph.py` verifies that held-out student indices do not appear in the train subgraph's `enrolled_in` src tensor.

### Todo List
1. Implement `build_train_subgraph(data: HeteroData, train_mask: torch.BoolTensor) -> HeteroData` in `src/gnn_model.py`.
   - Filter `enrolled_in` to rows where `train_mask` is True; recompute `rev_enrolled_in`.
   - Collect training student node indices from filtered `enrolled_in` src.
   - Filter `submitted` src to training student nodes; filter `interacted_with` src to training student nodes.
   - Keep all node feature tensors and structural edges (course_presentation, assessment, vle_resource) unchanged.
   - Attach `y = data[ei_key].y[train_mask]` and `enrollment_idx = data[ei_key].enrollment_idx[train_mask]` to the subgraph's `enrolled_in`.
2. Refactor `run_training_loop(model, data, train_subgraph, val_data, ...)`:
   - Forward pass for loss uses `model(train_subgraph)` on full subgraph (no masking inside the loop — all edges in subgraph are training edges).
   - Validation pass uses `model(data)` on full graph, then masks to val indices.
   - Remove the `logits[train_mask]` pattern — the subgraph's logits are all training logits.
3. Update `run_random_split_experiment()`:
   - After building masks, call `build_train_subgraph(data, train_mask)`.
   - Pass subgraph to `run_training_loop`; pass full `data` for val/test inference.
4. Update `run_lcpo_experiment()`:
   - After `_mask_held_out_edges()`, call `build_train_subgraph(data_masked, train_mask)` to produce the training subgraph.
   - Inference on test set uses the full (unmasked) `data` as before.
5. Write `tests/test_subgraph.py`: (a) held-out student nodes absent from subgraph `enrolled_in` src, (b) subgraph `y` length equals `train_mask.sum()`, (c) full graph used for val inference still contains all enrolled_in edges.

### Relevant Context
- `src/run_gnn_experiment.py` lines 538–660: `run_training_loop` and its callers
- `src/run_gnn_experiment.py` line 94: `_mask_held_out_edges` — LCPO edge masking (unchanged; subgraph is built on top)
- `src/gnn_model.py` lines 480–531: `EnrollmentGNN.forward()` — receives a `HeteroData` and returns logits for every `enrolled_in` edge; this is unchanged

---

## Sub-Task 3 — Remove cross-course student aggregation (enrollment-level representation)

**Status**: `[ ] pending`

### Intent
The three `pyg_scatter` injections in `EnrollmentGNN.forward()` aggregate edge attributes from ALL of a student's enrollments into their shared student-node representation, mixing course-specific information and potentially future course information. These injections must be removed. The edge prediction head already concatenates the student-node embedding with the course_presentation-node embedding per enrolled_in edge; passing `enrolled_in` edge attributes directly into the prediction head (after a linear projection) achieves enrollment-level conditioning without any cross-course aggregation.

### Expected Outcomes
- `EnrollmentGNN` no longer has `enrollment_attr_proj`, `submitted_attr_proj`, or `interacted_with_attr_proj` modules.
- The edge prediction head (`edge_head`) accepts `[h_src ‖ h_dst ‖ proj(ei_attr)]` (concatenation of student embedding, course embedding, and projected enrolled_in edge attributes) — shape `(E, 2*hidden + hidden)` → `Linear → 1`.
- `submitted` and `interacted_with` edge attributes that were previously scattered into the student node are now excluded from the GNN (they enter via message-passing topology, which is still present); or, if retained, they are aggregated per-enrollment (not per-student) — document chosen approach.
- The model still passes the existing forward-pass shape test (`test_edge_attr_reaches_prediction_head`) after updating expected dimensions.
- A new unit test `tests/test_no_cross_course.py` verifies that changing one student's non-training enrollment edge attributes does not alter the logit for another enrollment of the same student.

### Todo List
1. In `src/gnn_model.py`, `EnrollmentGNN.__init__`:
   - Remove `self.enrollment_attr_proj`, `self.submitted_attr_proj`, `self.interacted_with_attr_proj`.
   - Add `self.ei_attr_proj = nn.Linear(n_enrolled_in_attr, hidden_dim)` for per-enrollment edge attribute projection.
   - Change `self.edge_head` input dimension from `hidden_dim * 2` to `hidden_dim * 3` (student + course + projected ei_attr).
2. In `EnrollmentGNN.forward()`:
   - Remove the three `pyg_scatter` injection blocks (lines 485–517).
   - After both conv layers, project `enrolled_in` edge attributes: `ei_proj = self.ei_attr_proj(data[ei_key].edge_attr)` → `(E, hidden_dim)`.
   - Concatenate: `edge_repr = torch.cat([h_src, h_dst, ei_proj], dim=1)`.
   - Pass to `self.edge_head`.
3. Update `_build_model_and_optimizer()` in `src/run_gnn_experiment.py` if it passes `n_submitted_attr` or `n_interacted_with_attr` to `EnrollmentGNN` — remove those arguments.
4. Update `test_edge_attr_reaches_prediction_head` expected dimensions.
5. Write `tests/test_no_cross_course.py`.

### Relevant Context
- `src/gnn_model.py` lines 410–531: `EnrollmentGNN.__init__` and `forward()`
- `src/run_gnn_experiment.py` lines 52–80: `_build_model_and_optimizer()`
- `tests/test_gnn_data_flow.py` lines 178–205: `test_edge_attr_reaches_prediction_head` — needs dimension update

---

## Sub-Task 4 — LightGBM feature parity and threshold alignment

**Status**: `[ ] pending`

### Intent
LightGBM currently receives 7 features and uses train+val combined for fitting with a fixed 0.5 threshold. It must receive the same enrollment-scoped features used by the GNN's `enrolled_in` edge attributes (adding `age_band`, `studied_credits`) and use the same val-set F1-maximising threshold procedure as the GNN.

### Expected Outcomes
- `_FEATURE_COLS` in `src/compare_gnn_lgbm.py` extended with `age_band` (encoded) and `studied_credits`.
- `build_tabular_features()` returns these additional columns from the `enrolled_in` edge parquet (not re-engineered from raw data — read from the already-built artifact).
- `run_lgbm_random_split()` trains on train only (not train+val); uses val set for threshold selection via `select_threshold()` (imported from `gnn_model.py`); evaluates on test.
- `run_lgbm_lcpo()` similarly uses a 10% val draw (same protocol as GNN LCPO) for threshold selection per fold.
- The existing `_compute_metrics` in `compare_gnn_lgbm.py` accepts a threshold parameter.
- All other hyperparameters (`n_estimators=100`, `random_state=42`) remain unchanged.

### Todo List
1. In `src/compare_gnn_lgbm.py`:
   - Add a `build_enrolled_in_features(week)` helper that reads `results/graph/artifacts/week{N}_edges_enrolled_in.parquet` and returns the `age_band`, `studied_credits`, `num_of_prev_attempts` columns (already present in the artifact from `GraphDataLoader.load()`).
   - Merge with `build_tabular_features(week)` output on the enrollment key triple.
   - Extend `_FEATURE_COLS` to include `age_band` (one-hot columns, or raw categorical — match GNN encoding), `studied_credits`.
2. Refactor `run_lgbm_random_split()`:
   - Train on `train_mask` only (remove `| val_s`).
   - After training, call `model.predict_proba(X_val)` and `select_threshold()` (import from `gnn_model`).
   - Evaluate on test with tuned threshold.
3. Refactor `run_lgbm_lcpo()`:
   - For each fold, draw a 10% val set using the same `rng = np.random.default_rng(SEED + fold_idx)` pattern from `run_lcpo_experiment()`.
   - Train on train-only, tune threshold on val, evaluate on test.
4. Update `_compute_metrics()` to accept `threshold` kwarg (default 0.5 for backward compatibility).
5. Update the comparison table builder if column schema changes.

### Relevant Context
- `src/compare_gnn_lgbm.py` lines 70–200: feature building and split logic
- `src/gnn_model.py` lines 663–682: `select_threshold()` — to be imported
- `results/graph/artifacts/week{N}_edges_enrolled_in.parquet` — contains `age_band`, `num_of_prev_attempts`, `studied_credits`, `src`, `dst`

---

## Sub-Task 5 — LCPO multi-seed model initialisation

**Status**: `[ ] pending`

### Intent
Each LCPO fold currently trains exactly one model (seed = `SEED + fold_idx`). The split seed (controlling val-student sampling) and the model initialisation seed must be decoupled, and 5 independent model seeds must be run per fold. The fold result is the mean ± std across those 5 seeds.

### Expected Outcomes
- `run_lcpo_experiment()` accepts a `model_seeds: list[int]` parameter (default `[42, 123, 7, 17, 99]`).
- The val-student draw uses a fixed fold-specific RNG (keyed on fold index only, not on model seed) so the same val set is used across all model seeds within a fold.
- Per-seed metrics are stored; the fold row in `lcpo_results.csv` reports mean ± std across seeds.
- Training curve filenames include both fold index and model seed.
- The LCPO summary CSV reports mean ± std correctly propagated over all folds × seeds.

### Todo List
1. In `src/run_gnn_experiment.py`, `run_lcpo_experiment()`:
   - Add parameter `model_seeds: list[int] = [42, 123, 7, 17, 99]`.
   - Move val-student sampling RNG construction to use only `fold_idx` as entropy: `rng = np.random.default_rng(fold_idx)` (separate from model seed).
   - Inner loop over `model_seeds`: set `torch.manual_seed(mseed); np.random.seed(mseed)` before `_build_model_and_optimizer`.
   - Collect per-seed metric dicts; compute fold-level mean ± std.
   - Append one row per seed to `lcpo_results.csv` (with a `model_seed` column) and a summary row to `lcpo_summary.csv`.
2. Update training-curve save path to include fold_idx and model_seed: `f"training_curves_lcpo_fold{fold_idx:02d}_seed{mseed}.npz"`.
3. Expose `--model-seeds` CLI argument in the `argparse` block.
4. Update `run_lgbm_lcpo()` in `compare_gnn_lgbm.py` to match: LightGBM is deterministic given `random_state`, so one run per fold suffices — document this parity note.

### Relevant Context
- `src/run_gnn_experiment.py` lines 332–490: `run_lcpo_experiment()`
- `src/run_gnn_experiment.py` line 344–345 and 439–440: current seed reset locations

---

## Sub-Task 6 — Training-curve filename namespacing + reproduction-script safety

**Status**: `[ ] pending`

### Intent
Training curve `.npz` files for the random-split experiment omit week and seed, causing multi-week or multi-seed runs to silently overwrite earlier outputs. The reproduction script must never overwrite existing result CSVs. All output filenames must be namespaced by week, seed (or fold), and loss-weighting.

### Expected Outcomes
- Random-split training curves saved as `training_curves_random_student_week{N}_seed{S}_{weighting}.npz`.
- LCPO training curves (from Sub-Task 5) already include fold and seed.
- Result CSVs (`random_student_results.csv`, `lcpo_results.csv`, `comparison_results.csv`) are appended to (not overwritten) if they already exist, or a timestamp-namespaced run directory is used.
- `scripts/reproduce_all.sh` checks that required graph artifact parquet files exist before attempting GNN experiments and prints an actionable error if they are absent.
- README / QUICK_START notes that `data/raw/studentVle.csv` is gitignored and must be downloaded.

### Todo List
1. In `src/run_gnn_experiment.py` line 293:
   - Change `curves_path` to `f"training_curves_random_student_week{week:02d}_seed{seed}_{loss_weighting}.npz"`.
2. In `src/run_gnn_experiment.py` main result-CSV write (lines around 610):
   - Switch from `to_csv(path)` (overwrite) to read-existing-then-concat-append pattern: if file exists, load it, concatenate new rows, deduplicate on `(week, seed, loss_weighting)`, then write.
   - Same pattern for `lcpo_results.csv`.
3. In `scripts/reproduce_all.sh`:
   - Add a preflight check before step 3 that verifies `results/graph/artifacts/week08_enrollments.parquet` exists; abort with message if not.
   - Add `--no-overwrite` flag convention or timestamp the run.
4. Update `QUICK_START.md` to note the `studentVle.csv` download requirement.

### Relevant Context
- `src/run_gnn_experiment.py` lines 291–294: current curve path
- `src/run_gnn_experiment.py` lines 600–620: CSV write block
- `scripts/reproduce_all.sh` line 18–19: GNN experiment calls

---

## Sub-Task 7 — Authoritative result file + figure/table regeneration

**Status**: `[x] done`

### Intent
Multiple result CSVs (`random_student_results.csv`, `lcpo_results.csv`, `lcpo_summary.csv`, `comparison_results.csv`) can diverge if regenerated independently. All reported numbers in manuscript tables and figures must trace to a single canonical regeneration of result CSVs. A validation script must confirm that every number in `results/graph/tables/` matches the source CSV.

### Expected Outcomes
- `src/generate_report_figures.py` reads exclusively from `comparison_results.csv` (the combined authoritative file) — not independently from `random_student_results.csv` or `lcpo_results.csv`.
- `comparison_results.csv` is rebuilt by `compare_gnn_lgbm.py` by re-loading `random_student_results.csv` and `lcpo_results.csv` as inputs (rather than re-running the GNN), ensuring one source of truth.
- `src/generate_report_figures.py` includes a consistency check: for every number in the figures/tables output, the value can be recomputed from `comparison_results.csv` alone.
- Course-level win/loss count (previously reported as 19/22 but CSV shows 18/22) is corrected by rerunning from the authoritative CSV.
- A `src/verify_results.py` script (or added to `generate_report_figures.py`) prints a diff of any value that differs between `tables/` and the source CSV.

### Todo List
1. Audit `src/generate_report_figures.py`: list every `pd.read_csv()` call and the columns used.
2. Decide and implement: either (a) `generate_report_figures.py` reads only `comparison_results.csv`, or (b) it reads the two primary CSVs and re-derives the combined table in-process — never from a stale file.
3. Update `compare_gnn_lgbm.py` to accept `--from-csv` flag: instead of re-running experiments, load existing `random_student_results.csv` and `lcpo_results.csv`, combine them with fresh LightGBM results, and write `comparison_results.csv`.
4. Add `src/verify_results.py`:
   - Recomputes every aggregate (mean, std, win counts) from `comparison_results.csv`.
   - Diffs against values in `results/graph/tables/*.csv`.
   - Exits non-zero if any value differs by more than 1e-4.
5. Run `verify_results.py` after regeneration and fix any discrepancies in the course-level win count.

### Relevant Context
- `src/generate_report_figures.py` lines 17–20: input paths
- `results/graph/tables/`: existing table CSVs to validate against
- `src/compare_gnn_lgbm.py` lines 352–580: `build_combined_csv` and its write

---

## Sub-Task 8 — Expanded test suite

**Status**: `[ ] pending`

### Intent
The existing 29 tests do not cover the risks introduced by this plan. New tests must cover: training-only normalization, held-out graph isolation, edge-attribute perturbation invariance, validation fallback behaviour, label alignment, and a lightweight full-model training smoke test.

### Expected Outcomes
- `tests/test_normalization.py`: training-only normalisation (from Sub-Task 1).
- `tests/test_subgraph.py`: inductive subgraph isolation (from Sub-Task 2).
- `tests/test_no_cross_course.py`: enrollment-level representation — perturbing held-out edge attrs leaves training logits unchanged (from Sub-Task 3).
- `tests/test_label_alignment.py`: label order matches `enrolled_in` edge order in the loaded graph; after subgraph filtering, label order is preserved.
- `tests/test_training_smoke.py`: run 3 epochs on a toy graph (5 students, 2 courses); verify loss decreases and logit count matches enrolled_in edge count.
- `tests/test_validation_fallback.py`: `run_lcpo_experiment` fallback val-sampling logic (currently inline in `run_gnn_experiment.py`) triggers correctly when val positives < 20, and the resulting val mask still has disjoint students from test.
- All new tests use toy in-memory graphs (no parquet files required) and run in < 10 seconds total.

### Todo List
1. Extract the LCPO fallback val-sampling logic from `run_lcpo_experiment()` into a standalone function `_sample_lcpo_val(enroll_df, train_mask, y, fold_idx, min_val_pos=20)` — makes it unit-testable.
2. Write `tests/test_label_alignment.py`:
   - Build a toy graph with `_build_toy_graph()` and verify `data[ei_key].y[i]` matches the `i`-th enrollment's target.
   - Verify that after `build_train_subgraph(data, train_mask)`, the subgraph's y tensor is `data[ei_key].y[train_mask]`.
3. Write `tests/test_training_smoke.py`:
   - Build a toy 5-student graph.
   - Run `run_training_loop` for 3 epochs.
   - Assert `train_losses[-1] < train_losses[0]` (loss decreasing).
   - Assert `logits.shape[0] == train_subgraph[ei_key].edge_index.shape[1]`.
4. Write `tests/test_validation_fallback.py` using the extracted `_sample_lcpo_val`.
5. Fill in `tests/test_normalization.py` and `tests/test_subgraph.py` and `tests/test_no_cross_course.py` as specified in Sub-Tasks 1–3.
6. Ensure all 8 test files pass from project root: `pytest tests/ -v`.

### Relevant Context
- `tests/test_gnn_data_flow.py` lines 38–146: `_build_toy_graph()` — reuse for new tests
- `src/run_gnn_experiment.py` lines 397–424: fallback val-sampling logic to extract
- `src/gnn_model.py` lines 595–660: `run_training_loop()` signature

---

## Execution Order

```
Sub-Task 1 (normalization)
    ↓
Sub-Task 2 (subgraph isolation)   ← depends on normalisation mask API
    ↓
Sub-Task 3 (remove cross-course aggregation)   ← independent of 1 & 2 but best after
    ↓
Sub-Task 4 (LightGBM parity)   ← depends on Sub-Task 3 model interface being stable
    ↓
Sub-Task 5 (LCPO multi-seed)   ← depends on Sub-Task 2 training loop
    ↓
Sub-Task 6 (filename namespacing + script safety)   ← depends on Sub-Task 5 curve paths
    ↓
Sub-Task 7 (authoritative results)   ← depends on Sub-Tasks 4–6 for correct numbers
    ↓
Sub-Task 8 (expanded tests)   ← depends on all refactored APIs being stable
```
