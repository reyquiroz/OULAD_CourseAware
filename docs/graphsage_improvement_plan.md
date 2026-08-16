# GraphSAGE Improvement Plan

## Overview

The current two-layer `EnrollmentGNN` (GraphSAGE) in `src/gnn_model.py` is an appropriate
baseline, but several gaps prevent a fair evaluation:

1. **Edge attributes are loaded but not consumed** — `enrolled_in` edge attributes
   (age_band, num_of_prev_attempts, studied_credits) are stored in `edge_attr` tensors
   but `SAGEConv` never sees them; numeric features are also un-normalized.
2. **LCPO removes behavioral data from the wrong students** — the current
   `_mask_held_out_edges` strips submitted/interacted_with edges for every student
   enrolled in the held-out course, including their activity in *other* courses.
3. **Validation contamination in LCPO** — the 10% validation sample is drawn randomly
   from the full non-test enrollment pool, so a student can appear in both the
   validation mask and the training graph's message-passing edges.
4. **Training diagnostics are incomplete** — no overfit sanity check, no saved loss
   curves, threshold is hard-coded at 0.5, and no comparison between weighted and
   unweighted loss.
5. **Comparison is incomplete** — not all 22 LCPO folds are run with multiple seeds,
   reported metrics don't include mean ± std, LightGBM uses different folds, and test
   coverage is thin.

The plan below addresses all five areas without changing the model architecture or
moving to a different GNN family.

---

## Sub-task 1 — Incorporate enrollment edge attributes via linear projection into message passing, and normalize numeric features

**Status:** `[x] complete`

### Intent
The `enrolled_in` edge carries three behaviorally meaningful attributes:
age_band (categorical, already one-hot), num_of_prev_attempts (numeric), and
studied_credits (numeric). Currently the `ei_attr` tensor is stored on the edge but
`forward()` never passes it to `SAGEConv` or any other layer, so these attributes
have zero effect on node embeddings or predictions.

**Design decision (confirmed):** enrollment edge attributes will be incorporated via a
**separate linear projection** that embeds them into the message-passing process. A
small MLP (`enrollment_attr_proj`) will project the enrollment edge attribute vector
into `hidden_dim` space. For each enrolled_in edge at prediction time, the projected
attribute embedding will be **added to** (or concatenated with) the student embedding
before the second SAGEConv layer, so that the enrollment context shapes the student
representation that is ultimately combined with the course embedding in the prediction
head. This keeps the edge-level enrollment context inside the GNN's representational
bottleneck rather than bypassing it entirely.

Additionally, all numeric features (node and edge) should be normalized before entering
the model.

### Expected Outcomes
- `enrolled_in` edge attributes are projected into `hidden_dim` and folded into the
  per-enrollment student representation before the prediction head.
- All numeric node and edge features are zero-mean / unit-variance (log-scale where
  appropriate, e.g., `total_clicks`, `n_interactions`) before entering any layer.
- No change to the output shape: logits remain a vector of length 32,593 (one per
  enrolled_in edge).

### Todo List
1. Add a `_normalize_numeric_features(data)` function in `src/gnn_model.py` that
   standardizes every numeric column across node and edge feature tensors (compute
   mean and std over the full graph, not per-split). Apply it inside
   `GraphDataLoader.load()` before returning `data`.
2. Add an `enrollment_attr_proj` linear layer to `EnrollmentGNN.__init__`:
   `nn.Linear(n_enrolled_in_attr, hidden_dim)`. `n_enrolled_in_attr` is a new
   constructor argument computed from `data[ei_key].edge_attr.shape[1]`.
3. In `EnrollmentGNN.forward()`, after layer 1 completes, gather the enrollment
   edge attributes for each enrolled_in edge, project them with `enrollment_attr_proj`,
   and add the result to the per-edge student embedding slice before passing into
   layer 2: `h_student_per_edge = h_dict["student"][src_idx] + self.enrollment_attr_proj(ei_attr)`.
   Then scatter-aggregate (mean) back to student node positions so the updated
   representation can feed into layer 2.
4. Update `_build_model_and_optimizer()` in `src/run_gnn_experiment.py` to read
   `n_enrolled_in_attr` from `data[ei_key].edge_attr.shape[1]` and pass it to
   `EnrollmentGNN`.

### Relevant Context
- [`src/gnn_model.py:GraphDataLoader.load()`](src/gnn_model.py:76) — where edge attrs are built
- [`src/gnn_model.py:EnrollmentGNN.forward()`](src/gnn_model.py:322) — where the injection point sits
- [`src/gnn_model.py:EnrollmentGNN.__init__()`](src/gnn_model.py:266) — where `enrollment_attr_proj` is added
- [`src/run_gnn_experiment.py:_build_model_and_optimizer()`](src/run_gnn_experiment.py:50) — where model is constructed
- Numeric edge columns to normalize: `num_of_prev_attempts`, `studied_credits`,
  `score` (submitted), `total_clicks`, `n_interactions`, `first_day`, `last_day`,
  `active_days` (interacted_with)
- Numeric node columns to normalize: `module_presentation_length`, `weight`,
  `date` (assessment), `week_from`, `week_to` (vle_resource)
- `scatter_mean` from `torch_scatter` (already available via PyG) can aggregate
  per-edge projections back to per-node space

---

## Sub-task 2 — Fix LCPO held-out edge masking to preserve cross-course student activity

**Status:** `[ ] pending`

### Intent
`_mask_held_out_edges()` currently removes all submitted and interacted_with edges for
every student enrolled in the held-out course-presentation, even if those same edges
connect the student to a *different* course's assessments or VLE resources. This
over-filters training signal and can make the held-out student nodes information-poor,
which is not representative of how the model will be deployed. The correct rule is:
remove only the edges that belong to the held-out course-presentation, not all edges for
any student who happens to be enrolled in it.

Concretely:
- A `submitted` edge (student → assessment) should be removed only when that assessment
  belongs to the held-out course-presentation (i.e., is reachable via
  `contains_assess` from the held-out CP node).
- An `interacted_with` edge (student → vle_resource) should be removed only when that
  resource belongs to the held-out course-presentation (i.e., is reachable via
  `has_resource` from the held-out CP node).

### Expected Outcomes
- After masking, held-out students still retain submitted/interacted_with edges
  that belong to their other enrolled courses.
- Held-out CP's assessment and VLE resource nodes remain in the graph (for inference),
  but no messages flow through them during training.
- The comment at line 106–113 of `src/run_gnn_experiment.py` is updated to reflect
  the new logic.

### Todo List
1. In `_mask_held_out_edges()`, replace the student-index-based filter for
   `submitted` edges with an assessment-node-based filter: build the set of
   assessment node indices belonging to the held-out CP (via `contains_assess`
   edge_index on `data`), then keep only submitted edges whose destination is NOT
   in that set.
2. Similarly replace the student-index-based filter for `interacted_with` edges
   with a vle_resource-node-based filter: build the set of vle_resource node
   indices belonging to the held-out CP (via `has_resource` edge_index), then keep
   only interacted_with edges whose destination is NOT in that set.
3. Remove the `ho_student_node_indices` computation (lines 88–93 of
   `src/run_gnn_experiment.py`) as it is no longer needed.
4. Update inline comments to describe the corrected masking logic.

### Relevant Context
- [`src/run_gnn_experiment.py:_mask_held_out_edges()`](src/run_gnn_experiment.py:69) — function to rewrite
- Relevant edge types: `contains_assess`, `has_resource`, `submitted`, `interacted_with`
- The enrolled_in edges are intentionally kept intact (already noted in code)

---

## Sub-task 3 — Correct validation split isolation for LCPO and align GNN/LightGBM folds

**Status:** `[ ] pending`

### Intent
The current LCPO loop draws the validation subset randomly from the full non-test
enrollment pool (line 300–303 of `src/run_gnn_experiment.py`). Because message-passing
uses the whole graph, a student in the validation mask still passes messages to training
nodes, meaning validation labels can indirectly influence training embeddings. The fix
is to group the validation split by student (same guarantee as the random split), so
that no student appears in both the validation set and the message-passing neighborhood
of training-only nodes (to the extent possible without mini-batching).

Additionally, LightGBM in `src/compare_gnn_lgbm.py` builds its own train/test split
from the enrollment parquet independently; it should load the *same* pre-saved
random-student split parquet that the GNN uses, ensuring identical folds.

### Expected Outcomes
- LCPO validation set is drawn by grouping on `id_student`, with no student straddling
  both train and val masks.
- `compare_gnn_lgbm.py` random-student experiment reads from the same
  `week08_random_split.parquet` file that the GNN experiment uses.
- Both GNN and LightGBM LCPO experiments iterate over the same 22 fold definitions
  from `week08_lcpo_folds.csv`.

### Todo List
1. In `run_lcpo_experiment()`, replace the random-index val draw with a student-grouped
   draw: get the unique student IDs from the non-test enrollments, sample 10% of
   student IDs, then expand to all their enrollment rows.
2. In `compare_gnn_lgbm.py`'s random-split path, load
   `results/graph/evaluation/week08/splits/week08_random_split.parquet` to obtain
   is_train / is_val / is_test columns, and use these masks instead of constructing
   a fresh split.
3. Confirm that LightGBM LCPO also iterates over `week08_lcpo_folds.csv` (same 22
   folds, same held-out module+presentation per row) rather than re-deriving its own
   folds.

### Relevant Context
- [`src/run_gnn_experiment.py:run_lcpo_experiment()`](src/run_gnn_experiment.py:245) — val draw to fix (lines 300–310)
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — random-split loading to align
- Split files live under `results/graph/evaluation/week08/splits/`

---

## Sub-task 4 — Add training diagnostics: overfit check, loss curves, threshold tuning, weighted vs. unweighted comparison

**Status:** `[ ] pending`

### Intent
Before trusting test-set numbers, we need to know whether the model can fit the
training data at all, and whether it converges cleanly. We also need a
data-driven threshold rather than the hard-coded 0.5.

### Expected Outcomes
- A small-subset overfit smoke test (e.g., 64 or 128 training enrollments, 200 epochs)
  can be run via a CLI flag and prints train loss at convergence; if loss does not
  approach 0.1 or below, a warning is printed.
- `run_training_loop()` returns (and optionally saves) per-epoch train and validation
  loss arrays alongside the existing return values.
- The best classification threshold is selected on the validation set by maximizing
  F1 (or balanced accuracy) over a grid of candidate thresholds, and is stored
  alongside probabilities in the results CSV.
- Unweighted loss (no pos_weight) is run alongside the default weighted loss for the
  random-student experiment; results for both are saved to a single CSV with a
  `loss_weighting` column.
- Numeric features are verified to be normalized before training (assertion check).

### Todo List
1. Add a `run_overfit_check(data, n_samples, max_epochs)` function in
   `src/gnn_model.py` that selects the first `n_samples` training enrollments,
   trains until convergence, and returns the final train loss.
2. Modify `run_training_loop()` to return `(best_val_auroc, best_epoch, train_losses,
   val_losses)` — append the current train loss and val loss/auroc per epoch to lists.
3. Add a `select_threshold(probs, labels)` function in `src/gnn_model.py` that sweeps
   thresholds [0.05, 0.10, …, 0.95] and returns the one maximizing F1 on the
   provided validation probabilities and labels.
4. In `run_random_split_experiment()`, call `select_threshold()` on val probs/labels
   and pass the result to `compute_metrics()` (replace the hard-coded 0.5).
5. Add a `weighted` boolean parameter to `run_random_split_experiment()` and run it
   twice (weighted=True, weighted=False); save both rows to `random_student_results.csv`.
6. Save per-epoch loss arrays to
   `results/graph/training_curves_random_student.npz` using `np.savez`.

### Relevant Context
- [`src/gnn_model.py:run_training_loop()`](src/gnn_model.py:391) — return signature to extend
- [`src/gnn_model.py:compute_metrics()`](src/gnn_model.py:457) — threshold parameter to add
- [`src/run_gnn_experiment.py:run_random_split_experiment()`](src/run_gnn_experiment.py:186) — where to call new utilities

---

## Sub-task 5 — Complete reproducible LCPO comparison with multiple seeds and full metrics

**Status:** `[ ] pending`

### Intent
A single-seed, single-run result is not sufficient for a reproducible comparison.
We need all 22 LCPO folds completed, multiple seeds for the random-student experiment,
and a standard reporting format that matches LightGBM's output structure so the two
can be directly compared.

### Expected Outcomes
- All 22 LCPO folds are run for both GNN and LightGBM.
- The random-student experiment is run with at least 3 seeds; fold-level AUROC,
  AUPRC, F1, precision, recall, and balanced accuracy are saved with seed column.
- `compare_gnn_lgbm.py` produces a single combined CSV with columns:
  model, split, fold/seed, auroc, auprc, f1, precision, recall, balanced_acc.
- Summary statistics (mean ± std across folds or seeds) are printed and saved to a
  Markdown table at `results/graph/comparison_summary.md`.
- A `--seeds` CLI argument (e.g., `--seeds 42 123 7`) controls which seeds are run.

### Todo List
1. Add a `--seeds` argument to both `run_gnn_experiment.py` and
   `compare_gnn_lgbm.py`.
2. In `run_random_split_experiment()`, loop over seeds; for each seed, call
   `random_student_split()` (or load seed-specific split if pre-saved) and record
   metrics with a `seed` column.
3. Ensure `run_lcpo_experiment()` runs all 22 folds by default
   (max_folds=None is already the default; verify the `--quick` flag does not affect
   a production run).
4. In `compare_gnn_lgbm.py`, produce the combined per-fold CSV and write the
   Markdown summary table.
5. Verify the reported metric set is identical between GNN and LightGBM code paths
   (AUROC, AUPRC, F1, precision, recall, balanced_acc).

### Relevant Context
- [`src/run_gnn_experiment.py`](src/run_gnn_experiment.py) — seeds loop
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — combined CSV and Markdown table
- [`src/oulad_data.py:random_student_split()`](src/oulad_data.py) — seed parameter already supported

---

## Sub-task 6 — Add tests for graph loader, edge-feature use, split isolation, and prediction alignment

**Status:** `[ ] pending`

### Intent
The existing tests cover split utilities and temporal filtering but not the GNN-specific
data flow: whether edge attributes reach the model, whether held-out masking is correct,
whether split masks align to enrollment indices, and whether the prediction vector has
the right length. These tests act as regression guards for the changes made in
sub-tasks 1–5.

### Expected Outcomes
- A new test file `tests/test_gnn_data_flow.py` with at minimum the following tests:
  - `test_enrolled_in_edge_attr_shape`: `data[ei_key].edge_attr` has the expected
    number of columns after loading.
  - `test_edge_attr_reaches_prediction_head`: after a forward pass, the logit vector
    length equals the number of enrolled_in edges (32,593 for week 8 if artifacts
    are present, else skip if artifacts absent).
  - `test_lcpo_mask_does_not_strip_cross_course_edges`: construct a minimal toy graph
    and verify that masking the held-out CP removes only its own submitted/interacted_with
    edges, not edges belonging to other courses for the same student.
  - `test_split_mask_length_matches_enrollments`: the boolean masks returned by
    `load_split_masks` have length equal to the number of rows in the enrollments
    parquet.
  - `test_no_student_overlap_in_lcpo_val_draw`: after the student-grouped val draw,
    no student ID appears in both train and val masks.

### Todo List
1. Create `tests/test_gnn_data_flow.py`.
2. Use `pytest.importorskip` or a fixture with `pytest.mark.skipif` to skip tests
   that require artifact files if they are not present (so CI does not fail on a
   fresh clone).
3. Write a synthetic toy graph builder (5 students, 2 CPs, 3 assessments, 4 VLE
   resources) to use as a fixture for masking and alignment tests — avoids reading
   real parquet files for unit tests.
4. Implement the five tests listed above.
5. Confirm all tests pass alongside the existing suite (`tests/test_splits.py`,
   `tests/test_filter_window.py`).

### Relevant Context
- [`tests/test_splits.py`](tests/test_splits.py) — existing test pattern to follow
- [`src/gnn_model.py:GraphDataLoader`](src/gnn_model.py:53) — data loader to test
- [`src/run_gnn_experiment.py:_mask_held_out_edges()`](src/run_gnn_experiment.py:69) — masking to test

---

## Implementation Order

The sub-tasks should be implemented in this order:

```
Sub-task 1 (edge attrs + normalization)
  → Sub-task 2 (LCPO masking fix)
    → Sub-task 3 (val isolation + fold alignment)
      → Sub-task 4 (diagnostics + threshold)
        → Sub-task 5 (full comparison)
          → Sub-task 6 (tests)
```

Sub-tasks 1 and 2 change the data flow; sub-task 3 changes the split logic; sub-tasks 4
and 5 depend on both. Sub-task 6 can be done in parallel with 4–5 but should be
finalized last to cover all changes.
