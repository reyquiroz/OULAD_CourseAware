# Final GraphSAGE Completion Plan

## Overview

This plan covers the remaining work to produce a complete, reproducible GraphSAGE
vs. LightGBM comparison and a summer-report-ready manuscript draft. It assumes the
improvements implemented in `docs/graphsage_improvement_plan.md` are already in place:
enrollment edge-attribute projection, corrected LCPO masking, student-grouped
validation, training diagnostics, multi-seed random-split runs, and the five
`test_gnn_data_flow.py` tests. The current state before this plan begins:

- **GNN LCPO:** Only 2 of 22 folds completed (quick-mode runs only)
- **GNN multi-week:** Artifacts exist for weeks 2, 4, 6, 8 but GNN has only run on week 8
- **Edge attributes:** `submitted` (score) and `interacted_with` (5 click features) are
  loaded and normalized but not consumed in message passing — only `enrolled_in` attrs
  are projected
- **Ablations:** No infrastructure exists
- **Report:** No manuscript draft exists

The eleven user requirements map to eight sub-tasks below.

---

## Sub-task 1 — Consume submitted and interacted_with edge attributes in message passing

**Status:** `[x] done`

### Intent
Requirement 1 asks that assessment and VLE-interaction attributes be used in
predictions. The `enrolled_in` edge attributes are already projected via
`enrollment_attr_proj` and added to student embeddings between conv1 and conv2.
The same injection pattern should be applied to `submitted` (score) and
`interacted_with` (total_clicks, n_interactions, first_day, last_day, active_days)
edge attributes. Projecting both into `hidden_dim` and scatter-aggregating to student
nodes — in the same way `enrollment_attr_proj` already works — makes the behavioral
information available to the second SAGEConv layer and the prediction head.

### Expected Outcomes
- A `submitted_attr_proj` linear layer (`nn.Linear(1, hidden_dim)`) exists in
  `EnrollmentGNN.__init__`.
- An `interacted_with_attr_proj` linear layer (`nn.Linear(5, hidden_dim)`) exists.
- In `forward()`, after conv1, both projections are scatter-mean'd to student nodes
  and added to `h_dict["student"]` before conv2, alongside the existing enrollment
  projection.
- `n_submitted_attr` and `n_interacted_with_attr` are constructor arguments with
  defaults of 0 (disabled when those edge types are absent, e.g., week 2 for
  submitted).
- No change to output shape.

### Todo List
1. Add `submitted_attr_proj` and `interacted_with_attr_proj` linear layers to
   `EnrollmentGNN.__init__`, gated by `n_submitted_attr > 0` and
   `n_interacted_with_attr > 0`.
2. In `forward()`, after the `enrollment_attr_proj` scatter, add analogous scatter
   projections for submitted and interacted_with edges (src = student index; use
   submitted's `edge_index[0]` and interacted_with's `edge_index[0]` to map back to
   student nodes).
3. Update `_build_model_and_optimizer()` in `src/run_gnn_experiment.py` to read
   `n_submitted_attr` and `n_interacted_with_attr` from the respective `edge_attr`
   shapes and pass them to `EnrollmentGNN`.
4. Handle the case where submitted edges are absent (week 2): if
   `n_submitted_attr == 0` or the edge type is not present, skip that projection.
5. Run `pytest tests/ -v` and the quick smoke test to confirm no regressions.

### Relevant Context
- [`src/gnn_model.py:EnrollmentGNN.__init__()`](src/gnn_model.py:366) — `enrollment_attr_proj` pattern
  to replicate
- [`src/gnn_model.py:EnrollmentGNN.forward()`](src/gnn_model.py:411) — injection point after conv1
- [`src/run_gnn_experiment.py:_build_model_and_optimizer()`](src/run_gnn_experiment.py:58) — constructor
  call site
- `submitted` edge_attr dim: 1 (score, already normalized)
- `interacted_with` edge_attr dim: 5 (total_clicks, n_interactions, first_day,
  last_day, active_days — log1p + normalized)

---

## Sub-task 2a — Fix LCPO early-stopping before the full run

**Status:** `[x] done`

### Intent
Both completed LCPO folds stop at **epoch 4 out of 200** with best_val_auroc ≈ 0.69.
This is a diagnostic failure: the model is not converging, so the test-set results
are not representative of a trained model. The root cause is almost certainly that the
student-grouped 10% validation draw produces too few at-risk examples to give a stable
AUROC signal for a small course (AAA has ~370–380 test enrollments; 10% of the remaining
~29,000 training students is plentiful in count but the validation set may contain
very few at-risk students after grouping). The fix is to ensure the validation set
always contains a minimum number of at-risk examples, and to raise early-stopping
patience so the training loop does not terminate prematurely.

This must be resolved before running all 22 folds, otherwise the full LCPO run will
produce unreliable results.

### Expected Outcomes
- `run_lcpo_experiment()` validates the validation set has a minimum of `min_val_pos`
  positive examples (default 20) before starting training; if not, it falls back to
  a random (non-student-grouped) sample that meets the minimum.
- Early-stopping patience in the LCPO path is raised to 50 (from 20) to give the
  optimizer more time to find improvements on out-of-distribution folds.
- When re-run on folds 0 and 1, best_epoch is no longer 4 — the model trains for
  substantially more epochs before stopping.
- `results/graph/lcpo_results.csv` is updated with the corrected 2-fold results.

### Todo List
1. In `run_lcpo_experiment()` in `src/run_gnn_experiment.py`, after building
   `val_mask_np`, check that `y[val_mask].sum() >= min_val_pos` (default 20).
   If not, fall back: sample a random 10% of non-test enrollment indices (not
   student-grouped) and keep retrying with increasing sample sizes until the
   minimum is met, up to 30% of training enrollments.
2. Add a `lcpo_patience` parameter (default 50) separate from the random-split
   patience (default 20) to `run_lcpo_experiment()`, and pass it to
   `run_training_loop()`.
3. Re-run folds 0 and 1 with the fix:
   `python src/run_gnn_experiment.py --quick --random-only` (verify random split
   still works), then run 2 LCPO folds and confirm best_epoch > 10.
4. Update the `PATIENCE` constant or pass `lcpo_patience=50` explicitly in the
   `__main__` block.

### Relevant Context
- [`src/run_gnn_experiment.py:run_lcpo_experiment()`](src/run_gnn_experiment.py:311) — val draw and
  training loop call (lines ~365–400)
- [`results/graph/lcpo_results.csv`](results/graph/lcpo_results.csv) — currently shows best_epoch=4 for
  both folds; this is the symptom to fix
- The student-grouped draw is correct in principle; the issue is minimum at-risk
  count, not the grouping strategy itself

---

## Sub-task 2b — Run all feasible LCPO folds and multi-week experiments with multiple seeds

**Status:** `[x] done`

### Intent
Requirements 2, 5, and 6 all depend on having complete LCPO results. Currently only
2 of 22 folds have been run (quick mode). This sub-task — which must follow Sub-task
2a — runs the full 22-fold LCPO for week 8, extends the GNN experiment to weeks 2, 4,
and 6, and ensures all random splits also use multiple seeds. Results must include
fold-level metrics, means, and standard deviations.

### Expected Outcomes
- `results/graph/lcpo_results.csv` contains 22 rows (one per fold) for week 8.
- `results/graph/lcpo_summary.csv` has mean ± std across all 22 folds.
- Random-student results exist for all four weeks (2, 4, 6, 8) with seeds 42, 123, 7.
- LCPO results exist for all four weeks (capacity permitting — run week 8 fully first,
  then extend to earlier weeks).
- `results/graph/comparison_results.csv` is regenerated to include all folds (not
  just the 2 quick-mode folds in the current file).
- `results/graph/comparison_summary.md` reflects the full 22-fold summary.

### Todo List
1. Run the full GNN LCPO experiment for week 8:
   `python src/run_gnn_experiment.py --week 8 --seeds 42 123 7`
   (this runs random-split with 3 seeds and all 22 LCPO folds).
2. Extend to weeks 2, 4, 6 — add a `--weeks` multi-value argument to
   `src/run_gnn_experiment.py` (nargs="+", default=[8]) so all weeks can be run in
   one command: `python src/run_gnn_experiment.py --weeks 2 4 6 8 --seeds 42 123 7`.
   Save per-week results with a `week` column.
3. Run the full LightGBM comparison for all weeks and seeds:
   `python src/compare_gnn_lgbm.py --weeks 2 4 6 8 --seeds 42 123 7`
   (add the same `--weeks` argument to `compare_gnn_lgbm.py`).
4. Regenerate `comparison_results.csv` and `comparison_summary.md` to include all
   weeks, all folds, and all seeds.
5. Confirm LCPO val draw is student-grouped for every fold (already implemented; verify
   it still holds when iterating all 22 folds).

### Relevant Context
- [`src/run_gnn_experiment.py`](src/run_gnn_experiment.py) — `--week` already exists; add `--weeks`
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — add `--weeks` argument
- Artifacts for weeks 2, 4, 6, 8 confirmed present in `results/graph/artifacts/`
- Split files confirmed present in `results/graph/evaluation/week{02|04|06|08}/splits/`
- GNN LCPO is the slowest part (22 folds × 200 epochs each); run week 8 first to
  validate, then extend
- **Depends on Sub-task 2a** (early-stopping fix must be in place first)

---

## Sub-task 3 — Build ablation infrastructure for feature-group masking

**Status:** `[x] done`

### Intent
Requirement 7 asks for focused ablations covering assessment information, VLE activity,
temporal information, course-presentation information, and edge attributes. No ablation
infrastructure currently exists. The ablations should be implemented as a feature-group
mask passed to `GraphDataLoader` at load time, so the graph structure is identical
across conditions but specific feature columns are zeroed. This is the minimal change
that produces interpretable ablations without rebuilding the graph.

**Ablation groups to support:**
- `no_assessment`: zero out assessment node features and submitted edge attrs
- `no_vle`: zero out vle_resource node features and interacted_with edge attrs
- `no_temporal`: zero out temporal numeric features (first_day, last_day, active_days
  from interacted_with; date from assessment nodes)
- `no_course_features`: zero out course_presentation node features
- `no_edge_attrs`: zero out all edge attribute tensors (enrolled_in, submitted,
  interacted_with) — isolates the effect of edge attributes added in Sub-tasks 1–2

### Expected Outcomes
- `GraphDataLoader.load()` accepts a `feature_mask: list[str] = None` argument.
  When provided, the named feature groups are zeroed in the returned `HeteroData`.
- A `src/run_ablation.py` script runs all five ablation conditions for week 8 random
  split (and optionally LCPO), saves per-condition metrics to
  `results/graph/ablation_results.csv`.
- Ablation results include a `condition` column and the same metric columns as
  the main experiments.

### Todo List
1. Define a `FEATURE_GROUPS` dict in `src/gnn_model.py` mapping condition names to
   the node types / edge types and column slice indices to zero out.
2. Add `feature_mask: list[str] = None` parameter to `GraphDataLoader.load()`;
   after `_normalize_numeric_features(data)`, apply zeroing for any requested groups.
3. Create `src/run_ablation.py`:
   - Loops over the five ablation conditions plus a "full" baseline.
   - For each condition, calls `GraphDataLoader(week, feature_mask=[condition]).load()`,
     trains a fresh model with `run_random_split_experiment()`, and records metrics.
   - Saves all rows (condition, week, seed, split, metrics) to
     `results/graph/ablation_results.csv`.
   - Add a `--week` and `--seeds` CLI argument.
4. Run all ablations for week 8: `python src/run_ablation.py --week 8 --seeds 42`.

### Relevant Context
- [`src/gnn_model.py:GraphDataLoader.load()`](src/gnn_model.py:112) — where zeroing is applied
- [`src/run_gnn_experiment.py:run_random_split_experiment()`](src/run_gnn_experiment.py:213) — reuse for
  each ablation condition
- Zeroing is preferred over dropping edge types because it preserves graph structure
  and makes the ablation results directly comparable

---

## Sub-task 4 — Course-level variation analysis: GNN vs. LightGBM

**Status:** `[x] done`

### Intent
Requirement 8 asks for an analysis identifying where GraphSAGE performs better or
worse than LightGBM at the course level. The existing `comparison_results.csv` already
has per-fold LCPO results for LightGBM across all 22 folds; once Sub-task 2 provides
GNN LCPO results for all 22 folds, a per-course comparison can be produced.

### Expected Outcomes
- A `results/graph/course_variation.csv` with columns:
  `code_module, code_presentation, gnn_auroc, lgbm_auroc, auroc_delta, gnn_f1,
  lgbm_f1, f1_delta, n_test`
- Courses ranked by `auroc_delta` (GNN − LightGBM), flagging where GNN outperforms
  and where it lags.
- A summary noting which course characteristics correlate with GNN advantage
  (e.g., course size, class balance, assessment count).

### Todo List
1. After Sub-task 2 completes, join GNN and LightGBM LCPO fold results on
   `(held_out_module, held_out_presentation)`.
2. Compute per-fold deltas for AUROC, AUPRC, F1.
3. Save to `results/graph/course_variation.csv`.
4. Add a short analysis function in `src/compare_gnn_lgbm.py` (or a standalone
   `src/course_variation.py`) that produces this CSV and prints a ranked table.
5. Note any course where GNN AUROC > LightGBM AUROC and hypothesize why (e.g.,
   graph structure helps on larger courses or courses with denser VLE interaction).

### Relevant Context
- [`results/graph/comparison_results.csv`](results/graph/comparison_results.csv) — source data for join
- Course variation is already visible in LightGBM LCPO: GGG courses score ~0.64–0.69
  AUROC vs. BBB/CCC/DDD courses scoring 0.86–0.88
- Depends on Sub-task 2 (GNN LCPO 22 folds)

---

## Sub-task 5 — Figures and tables for the report

**Status:** `[x] done`

### Intent
Requirements 4, 6, and 9 ask for clear tables and figures. The manuscript needs
four specific visualizations: early-prediction performance (across weeks), random vs.
LCPO comparison, course-level variation, and available ablation results.

### Expected Outcomes
- `results/graph/figures/fig_week_performance.png`: AUROC vs. prediction week (2, 4,
  6, 8) for GNN and LightGBM, side-by-side bar chart with error bars (mean ± std
  across seeds).
- `results/graph/figures/fig_random_vs_lcpo.png`: Grouped bar chart comparing random-
  student and LCPO AUROC for GNN and LightGBM at week 8.
- `results/graph/figures/fig_course_variation.png`: Per-course AUROC for GNN and
  LightGBM (22 courses), ordered by LightGBM AUROC, with a line showing the delta.
- `results/graph/figures/fig_ablation.png`: Bar chart of AUROC by ablation condition
  (full model, no_assessment, no_vle, no_temporal, no_course_features, no_edge_attrs).
- A `results/graph/tables/` directory containing Markdown and CSV versions of all
  result tables referenced in the report.

### Todo List
1. Create `src/generate_report_figures.py` — a single script that reads the result
   CSVs produced by Sub-tasks 2, 3, and 4 and generates all four figures.
2. Use `matplotlib` / `seaborn` for all plots; save as PNG at 150 dpi.
3. Generate tables:
   - Main comparison table (GNN vs. LightGBM × random/LCPO × all metrics, week 8)
   - Early-prediction table (week 2, 4, 6, 8 × GNN vs. LightGBM × AUROC ± std)
   - Ablation table (condition × AUROC, AUPRC, F1)
   - Course-variation table (top 5 GNN wins, top 5 GNN losses)
4. Run after Sub-tasks 2, 3, and 4 are complete.

### Relevant Context
- Depends on Sub-tasks 2, 3, 4
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — existing figure generation can be extended
- Existing visualizations in `results/graph/artifacts/` can be reused for background
  figures (class balance, graph scale, etc.)

---

## Sub-task 6 — Reproducibility materials

**Status:** `[x] done`

### Intent
Requirement 10 asks for complete reproducibility materials. The following are missing
or incomplete: PyTorch/PyG version pins in requirements.txt, a single end-to-end
re-run script, and a clear seed/split/config documentation file.

### Expected Outcomes
- `requirements.txt` includes pinned versions of `torch`, `torch-geometric`,
  `torch-scatter`, and `torch-sparse` matching the installed environment.
- A `scripts/reproduce_all.sh` shell script that, given the OULAD raw data, runs
  the full pipeline in order: graph construction → splits → GNN experiments →
  LightGBM comparison → ablations → figures.
- `docs/REPRODUCIBILITY.md` documents: seeds used (42, 123, 7), split file locations,
  config parameters (hidden_dim=64, lr=1e-3, patience=20, max_epochs=200), and the
  exact commands to regenerate every result file.
- All result CSVs and split parquets remain committed to the repository (already the
  case; verify no large files are gitignored by accident).

### Todo List
1. Audit `requirements.txt` — add or update torch, torch-geometric, torch-scatter,
   torch-sparse with exact installed versions (run `pip show` to verify).
2. Write `scripts/reproduce_all.sh`:
   ```bash
   python src/run_graph_pipeline.py --week 2 4 6 8
   python src/save_graph_splits.py --weeks 2 4 6 8
   python src/run_gnn_experiment.py --weeks 2 4 6 8 --seeds 42 123 7
   python src/compare_gnn_lgbm.py --weeks 2 4 6 8 --seeds 42 123 7
   python src/run_ablation.py --week 8 --seeds 42
   python src/generate_report_figures.py
   ```
3. Write `docs/REPRODUCIBILITY.md` with the parameter table and command reference.
4. Verify `tests/test_gnn_data_flow.py` covers the new Sub-task 1 changes
   (submitted/interacted_with projection); add tests if needed.

### Relevant Context
- [`requirements.txt`](requirements.txt) — torch pinning missing or incomplete
- [`tests/test_gnn_data_flow.py`](tests/test_gnn_data_flow.py) — may need extension for Sub-task 1 changes

---

## Sub-task 7 — Manuscript draft (summer report)

**Status:** `[x] done`

### Intent
Requirement 11 asks for a manuscript draft with explicit research questions,
contributions, related work, methods, results, limitations, and a preliminary title
and abstract. This is a writing sub-task that synthesizes the results from all prior
sub-tasks.

### Expected Outcomes
- A single Markdown file `docs/summer_report_draft.md` structured as a manuscript
  draft with the following sections:
  - **Title and Abstract** (preliminary)
  - **1. Introduction** — motivation, problem statement, research questions
  - **2. Related Work** — graph-based learning analytics, LightGBM baselines for
    at-risk prediction, enrollment-level prediction in MOOCs
  - **3. Data and Preprocessing** — OULAD description, graph construction, window
    cutoffs, label convention
  - **4. Methods** — GraphSAGE architecture, enrollment-edge projection, training
    protocol, evaluation splits (random vs. LCPO), threshold selection
  - **5. Results** — main comparison table, early-prediction performance, course-level
    variation, ablation results
  - **6. Discussion** — where GNN helps and where it does not, course-level insights,
    limitations of transductive evaluation
  - **7. Conclusion and Future Work**
  - **References** (stubs with key papers)
- The abstract is ≤250 words and states the research question, approach, key result,
  and implication.
- All tables referenced in Section 5 match the CSVs produced by Sub-tasks 2–5.

### Todo List
1. Draft the title and abstract first — forces clarity on the main finding.
2. Write sections 1–4 (context, data, methods) before results are finalized; these do
   not depend on Sub-task 2 completion.
3. Fill in Section 5 tables once Sub-tasks 2–5 are complete.
4. Write Section 6 after reviewing fold-level and course-level results from Sub-task 4.
5. Collect reference stubs for: OULAD paper (Kuzilek et al., 2017), GraphSAGE
   (Hamilton et al., 2017), LightGBM (Ke et al., 2017), at least two prior OULAD
   prediction papers.

### Relevant Context
- Does not depend on Sub-tasks 2–5 for sections 1–4; can start immediately
- Research questions to address: (1) Does a graph representation provide an advantage
  over tabular LightGBM for at-risk prediction? (2) How does the performance gap change
  across prediction windows (weeks 2, 4, 6, 8)? (3) Which feature groups contribute
  most to GraphSAGE performance?
- Key result to report once available: current week-8 random-split AUROC is GNN 0.847
  vs. LightGBM 0.842 (3 seeds); LCPO gap is larger (only 2 folds so far: GNN ~0.74 vs.
  LightGBM ~0.76)

---

## Implementation Order

Sub-tasks have the following dependencies:

```
Sub-task 1  (submitted/interacted_with attrs)
  → Sub-task 2a (fix LCPO early-stopping)
    → Sub-task 2b (full LCPO + multi-week)
      → Sub-task 4  (course-level variation)
        → Sub-task 5  (figures)
          → Sub-task 7 (manuscript — results section)

Sub-task 3 (ablation infrastructure)
  → Sub-task 5 (ablation figure)

Sub-task 6 (reproducibility) — depends on Sub-tasks 1, 2a, 2b, 3 being complete

Sub-task 7 (manuscript sections 1–4) — can start immediately in parallel
```

Priority order:
1. Sub-task 1  — closes the last data-flow gap (submitted/interacted_with attrs)
2. Sub-task 2a — fix LCPO early-stopping (blocker for the full run)
3. Sub-task 2b — produces the core results table (all 22 folds, all weeks)
4. Sub-task 3  — ablations (time-permitting; run in parallel with 2b if capacity allows)
5. Sub-task 4  — course analysis (depends on Sub-task 2b)
6. Sub-task 5  — figures (depends on 2b, 3, 4)
7. Sub-task 6  — reproducibility (depends on 1, 2a, 2b, 3)
8. Sub-task 7  — manuscript (sections 1–4 start now; sections 5–6 after Sub-task 2b)
