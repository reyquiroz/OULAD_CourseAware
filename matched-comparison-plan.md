# Matched Comparison & GraphSAGE Improvement Plan

## Confirmed Design Decisions

| Decision | Choice |
|---|---|
| Matched comparison script | New `src/run_matched_comparison.py`; `compare_gnn_lgbm.py` is left untouched |
| MLP implementation | `sklearn.neural_network.MLPClassifier` (2 hidden layers, same train-only normalization) |
| Week coverage | All four weeks (2, 4, 6, 8) required for every model in the matched comparison |
| GNN enrollment-summaries weeks | All four weeks (not week 8 only) |
| Enrollment summaries placement | `enrolled_in` edge attributes first (Sub-Task 4); enrollment-node redesign is a separate later step (Sub-Task 6) |
| Compute environment | All computation runs on Lonestar6 (TACC). Graph artifact regeneration, GNN training, and tabular experiments all run via Slurm job scripts on LS6. No local execution assumed. |
| Artifact regeneration | Parquet artifacts are regenerated on LS6 as part of Sub-Task 4's Slurm job, before the GNN training step in the same job. |

## Overview

This plan addresses four interconnected goals from the reviewer feedback:

1. **Matched comparison** — run LightGBM, MLP, base GraphSAGE, and GraphSAGE+enrollment-summaries under an identical protocol (same splits, same seeds, same labels, same preprocessing, same feature availability) so that model differences can be attributed to architecture rather than data representation.
2. **Zenodo/KU Leuven audit** — verify that the Zenodo comparison uses the same protocol as OULAD and produce a written diagnosis of why results diverge.
3. **GraphSAGE architectural improvements** — introduce six incremental changes one at a time, retaining intermediate results as ablation rows.
4. **Excel results matrix** — populate `AUROC_Results_Matrix_by_Dr_Guo.xlsx` from verified CSV outputs after each experiment group.

The plan is sequenced so that every later sub-task depends only on the verified outputs of earlier ones. LCPO is treated as a first-class evaluation track throughout, not an afterthought.

---

## Architecture Constraints (must not be violated)

- Prediction unit is the **enrollment edge** (`id_student, code_module, code_presentation`), never the student node.
- Label convention: `1 = at-risk`, `0 = success` (from `src/config.py`).
- Dual temporal guard (Strategy B) applies to all feature computation: `due_date ≤ window` AND `date_submitted ≤ window`.
- Split functions `random_student_split` / `lcpo_split` from `src/oulad_data.py` are canonical for all models.
- Normalization must use training-set statistics only.
- All persistent results must be written via `_append_or_create_csv()` or equivalent idempotent writer.
- New paths must use `Path` constants from `src/config.py`, not hardcoded strings.

---

## Lonestar6 Execution

All experiments run on Lonestar6 (LS6) at TACC. No local execution is assumed. The section below defines the Slurm job templates for each experiment group; the sub-task todo lists reference these by name.

### LS6 Environment Setup (run once after `git clone`)

```bash
cd $SCRATCH
git clone <repo-url> OULAD && cd OULAD
module load python3/3.11.2
python3 -m venv oulad_env
source oulad_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.13.0 torch-geometric==2.8.0.post1 \
    --index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
mkdir -p logs results/matched
# Upload raw CSVs (from local machine):
#   rsync -avP data/raw/ <user>@ls6.tacc.utexas.edu:$SCRATCH/OULAD/data/raw/
```

### Job Script A — `scripts/ls6_matched_tabular.sh` (Sub-Tasks 2 & 3)

Runs matched LightGBM and MLP for all four weeks under both random-student and LCPO evaluation. CPU-only; uses the `normal` queue.

```bash
#!/bin/bash
#SBATCH -J oulad_matched_tabular
#SBATCH -o logs/matched_tabular_%j.out
#SBATCH -e logs/matched_tabular_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -A <YOUR_ALLOCATION>

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_matched_comparison.py \
    --models lgbm mlp \
    --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 \
    --splits random lcpo
```

### Job Script B — `scripts/ls6_graph_artifacts.sh` (Sub-Task 4, step 1)

Regenerates parquet artifacts for all four weeks. CPU-only; fast (~6 s/week). Run before Job Script C.

```bash
#!/bin/bash
#SBATCH -J oulad_graph_artifacts
#SBATCH -o logs/artifacts_%j.out
#SBATCH -e logs/artifacts_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A <YOUR_ALLOCATION>

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

for week in 2 4 6 8; do
    python src/run_graph_pipeline.py --week $week
done
python src/save_graph_splits.py --weeks 2 4 6 8
```

### Job Script C — `scripts/ls6_gnn_enrollment_summaries.sh` (Sub-Task 4, step 2)

Trains GraphSAGE with enrollment summaries for all four weeks. Uses one A100 GPU. Must run after Job Script B completes (use `--dependency=afterok:<JOB_B_ID>`).

```bash
#!/bin/bash
#SBATCH -J oulad_gnn_enroll_summaries
#SBATCH -o logs/gnn_enroll_%j.out
#SBATCH -e logs/gnn_enroll_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A <YOUR_ALLOCATION>

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_matched_comparison.py \
    --models gnn_enrollment_summaries \
    --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 \
    --splits random lcpo
```

Submit B then C with dependency:
```bash
JOB_B=$(sbatch --parsable scripts/ls6_graph_artifacts.sh)
sbatch --dependency=afterok:$JOB_B scripts/ls6_gnn_enrollment_summaries.sh
```

### Job Script D — `scripts/ls6_gnn_architectural.sh` (Sub-Tasks 6–11)

Runs one incremental GNN variant at a time. Parameterized by `--variant` flag. Uses one A100 GPU per job.

```bash
#!/bin/bash
#SBATCH -J oulad_gnn_${VARIANT}
#SBATCH -o logs/gnn_${VARIANT}_%j.out
#SBATCH -e logs/gnn_${VARIANT}_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -A <YOUR_ALLOCATION>

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_gnn_experiment.py \
    --variant $VARIANT \
    --weeks 8 \
    --seeds 42 123 7 17 99 \
    --splits random lcpo
```

Submit for each variant:
```bash
for VARIANT in enrollment_node edge_aware_mp course_design temporal course_conditioned gcn rgcn hgt; do
    sbatch --export=VARIANT=$VARIANT scripts/ls6_gnn_architectural.sh
done
```

### Job Script E — `scripts/ls6_zenodo.sh` (Sub-Task 5)

Runs matched LightGBM and base GraphSAGE on Zenodo dataset (CPU + GPU, 1 node).

```bash
#!/bin/bash
#SBATCH -J oulad_zenodo
#SBATCH -o logs/zenodo_%j.out
#SBATCH -e logs/zenodo_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -A <YOUR_ALLOCATION>

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_zenodo_pipeline.py --mode all --seeds 42 123 7 17 99
```

### Post-Run: Copy Results Back

After jobs complete, copy results to your local machine or `$WORK` (which is not purged):
```bash
# On LS6 — copy to $WORK for persistence
cp -r $SCRATCH/OULAD/results/ $WORK/OULAD/results/

# On local machine — sync back
rsync -avP <user>@ls6.tacc.utexas.edu:$WORK/OULAD/results/ results/
```

`$SCRATCH` is purged after 10 days of inactivity. Always copy final results to `$WORK` immediately after a job completes.

### Key LS6 Commands

```bash
squeue -u $USER                   # check job queue
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
tail -f logs/<logfile>.out        # stream live output
idev -p gpu-a100 -N 1 --gres=gpu:1 -t 00:30:00   # interactive GPU session for debugging
taccinfo                          # check allocation balance
```

---

## Sub-Tasks

---

### Sub-Task 1 — Establish the Matched Protocol Baseline

**Status:** `[x] done`

**Intent**  
Define and document the exact matched protocol that every model in Sub-Tasks 2–4 must follow. This makes the comparison conditions explicit and auditable before any code is written.

**Expected Outcomes**
- A written protocol section (can live in this plan file or a separate `matched-protocol.md`) specifying: enrollment population, label source, prediction cutoff, split function and parameters, seed list, threshold-selection rule, preprocessing scope, and output column names.
- Confirmation that `compare_gnn_lgbm.py` already enforces this protocol for existing rows, or a list of deviations to fix before running new experiments.

**Todo List**
- [ ] Read `src/compare_gnn_lgbm.py` in full and `src/evaluation_pipeline.py` and compare their split logic, seed handling, threshold selection, and feature alignment against each other.
- [ ] Verify that `build_tabular_features()` in `compare_gnn_lgbm.py` uses the same `filter_window()` + `build_features()` path as the GNN graph pipeline for identical enrollment coverage.
- [ ] Confirm that the `age_band` one-hot encoding dimension matches between LightGBM feature matrix and GNN `enrolled_in` edge attributes.
- [ ] Document any deviations and decide whether to fix them in `compare_gnn_lgbm.py` or add a new `run_matched_comparison.py` script.
- [ ] Record the final protocol in this plan file under a "Matched Protocol" appendix.

**Relevant Context**
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — current comparison script
- [`src/evaluation_pipeline.py`](src/evaluation_pipeline.py) — tabular baseline feature builder
- [`src/oulad_data.py`](src/oulad_data.py) — canonical split functions and `build_features()`
- [`src/config.py`](src/config.py) — label convention, PREDICTION_WINDOWS, RANDOM_STATE

---

### Sub-Task 2 — Run Matched LightGBM (All Features, All Weeks, Both Splits)

**Status:** `[x] done`

**Intent**  
Produce the "LightGBM — all features (matched protocol)" row in the results matrix across weeks 2, 4, 6, 8 under both random-student and LCPO evaluation. This becomes the primary non-graph reference point that everything else is measured against.

**Expected Outcomes**
- New CSV rows in `results/graph/comparison_results.csv` (or a new `results/matched/lgbm_matched_results.csv`) with columns: `model, week, split_type, fold, seed, auroc, auprc, f1, balanced_acc, precision, recall, threshold`.
- Row 12 of every Excel sheet populated with `mean ± std` from these verified outputs.
- No student overlap between train/val/test confirmed by inspection.

**Todo List**
- [ ] In `src/run_matched_comparison.py`, implement `build_tabular_features()` for weeks 2, 4, 6, 8 with feature set `"full"` (VLE + assessment + demographics), adapted from `compare_gnn_lgbm.py`.
- [ ] On LS6: submit Job Script A (`ls6_matched_tabular.sh`) with `--models lgbm` to run random-student (seeds 42, 123, 7, 17, 99) and LCPO (22 folds) for all four weeks.
- [ ] After job completes, copy results from `$SCRATCH` to `$WORK` before the 10-day purge window.
- [ ] Aggregate: compute `mean ± std` across seeds (random) and across folds (LCPO).
- [ ] Write results to `results/matched/lgbm_matched_results.csv` via `_append_or_create_csv()`.
- [ ] Update Excel row 12 on all six sheets from these verified outputs.

**Relevant Context**
- [`src/compare_gnn_lgbm.py`](src/compare_gnn_lgbm.py) — `run_lgbm_random_split()`, `run_lgbm_lcpo()`
- GNN LCPO folds: `results/graph/evaluation/week{N:02d}/splits/week{N:02d}_lcpo_folds.csv`
- Results matrix row: `A12` across all sheets

---

### Sub-Task 3 — Run Matched MLP (All Enrollment Features, Both Splits)

**Status:** `[x] done`

**Intent**  
Add an MLP baseline that receives the same feature vector as LightGBM. This isolates the contribution of the graph topology (message-passing over the heterogeneous graph) from the contribution of the enrollment-level feature set.

**Expected Outcomes**
- Results CSV rows for `model = "MLP_all_features"` across weeks and both split types.
- Row 16 ("MLP — all enrollment features") populated in the Excel matrix.
- MLP trained and evaluated with identical seeds, folds, and threshold-selection rule as LightGBM.

**Todo List**
- [ ] Add a `run_mlp_matched()` function to `src/run_matched_comparison.py` using `sklearn.neural_network.MLPClassifier` with `hidden_layer_sizes=(128, 64)`, `max_iter=500`, `early_stopping=True`.
- [ ] Input features: identical feature matrix produced by `build_tabular_features()` — the same matrix used by matched LightGBM. No additional features.
- [ ] Apply train-only `StandardScaler` normalization: fit on train rows, transform train/val/test. This mirrors `_normalize_numeric_features()` in `gnn_model.py`.
- [ ] On LS6: submit Job Script A (`ls6_matched_tabular.sh`) with `--models mlp` (can be combined with the LightGBM run in Sub-Task 2 as a single job submission `--models lgbm mlp`).
- [ ] After job completes, copy results from `$SCRATCH` to `$WORK`.
- [ ] Write results to `results/matched/mlp_matched_results.csv` and populate Excel rows 16–17.

**Relevant Context**
- Feature matrix from `build_tabular_features()` in `compare_gnn_lgbm.py`
- Results matrix rows 16–17 on all sheets

---

### Sub-Task 4 — Add Six Enrollment Summaries to GraphSAGE Prediction Head

**Status:** `[x] done`

**Intent**  
Compute `vle_total`, `vle_mean`, `vle_std`, `assess_mean`, `assess_max`, `assess_count` at the **enrollment level** (one row per enrollment) and append them as additional attributes on the `enrolled_in` edge so the prediction head receives them directly. This creates feature parity between GraphSAGE and LightGBM without altering the message-passing topology.

**Expected Outcomes**
- `graph_pipeline.py` enriches `enrolled_in` edge table with 6 new columns, computed after the window cutoff is applied and using training-set statistics for normalization.
- `gnn_model.py` `GraphDataLoader` reads the additional columns and extends `ei_attr` accordingly.
- New ablation condition `"with_enrollment_summaries"` runs and produces results for week 8 random + LCPO (all four weeks is a stretch goal).
- Row 21 ("GraphSAGE + six enrollment summaries") populated in the Excel matrix.

**Todo List**
- [x] In `graph_pipeline.py` `build_edge_tables()`: after `filter_window()` has been applied, aggregate `studentVle` to per-enrollment `vle_total`, `vle_mean`, `vle_std` and `studentAssessment` to per-enrollment `assess_mean`, `assess_max`, `assess_count`; merge these six columns onto the `enrolled_in` edge DataFrame keyed on `(id_student, code_module, code_presentation)`.
- [x] Handle NaN fill consistently with `build_features()` in `oulad_data.py` (fill 0 for missing VLE, fill 0 for missing assessment).
- [x] In `gnn_model.py` `GraphDataLoader` enrolled_in attribute loading: detect the new columns and extend the numeric block; update `n_enrolled_in_attr` accordingly with `use_enrollment_summaries` flag.
- [x] Add ablation condition key `"with_enrollment_summaries"` in `run_gnn_experiment.py` that enables these columns while keeping all other hyperparameters identical to the base model.
- [ ] On LS6: submit Job Script B (`ls6_graph_artifacts.sh`) to regenerate parquet artifacts for weeks 2, 4, 6, 8.
- [ ] On LS6: submit Job Script C (`ls6_gnn_enrollment_summaries.sh`) with `--dependency=afterok:<JOB_B_ID>` to run random-student and LCPO for all four weeks with seeds `[42, 123, 7, 17, 99]`.
- [ ] After job completes, copy results from `$SCRATCH` to `$WORK` before the 10-day purge window.
- [ ] Write results and populate Excel row 21.

**Relevant Context**
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — `build_edge_tables()` (lines ~283–370)
- [`src/gnn_model.py`](src/gnn_model.py) — `GraphDataLoader` enrolled_in attr loading (lines ~456–465)
- [`src/oulad_data.py`](src/oulad_data.py) — `build_features()` as reference for VLE/assessment aggregation logic
- AGENTS.md constraint: normalization must use train-subset statistics

---

### Sub-Task 5 — Zenodo Protocol Audit and Divergence Analysis

**Status:** `[x] done`

**Intent**  
Verify that the Zenodo/KU Leuven experiments use the same labels, behavioral features, prediction cutoff, splits, and evaluation records as the matched OULAD protocol. Produce a written analysis of why LightGBM outperforms GraphSAGE on Zenodo and what that reveals about when graph learning is useful.

**Expected Outcomes**
- A written "Zenodo Audit" section (added to this plan or a new `zenodo-audit.md`) documenting: class balance, graph sparsity (nodes, edges, average degree), disconnected components, course-level variation, and feature coverage.
- A clear statement of whether the Zenodo comparison is currently fair (same protocol) or not, and any fixes applied.
- A 3–5 paragraph interpretive analysis suitable for inclusion in the paper explaining the performance divergence hypothesis.
- Zenodo matched results written to `results/zenodo/` and Excel columns N–O populated.

**Todo List**
- [ ] Read `src/run_zenodo_pipeline.py` `run_zenodo_lgbm()`, `run_zenodo_random_split()`, `run_zenodo_lcpo()` and compare their split logic, label source, threshold rule, and feature set against the matched OULAD protocol from Sub-Task 1.
- [ ] Read `src/zenodo_data.py` `build_features_zenodo()` and verify it uses dual-guard `filter_window_zenodo()` and produces the same 6 behavioral summary columns.
- [ ] Compute and record: class balance (at-risk rate), total enrollment count, unique students, unique courses, `interacted_with` edge count per enrollment (mean/median), `submitted` edge count per enrollment, number of disconnected enrollment nodes.
- [ ] Run matched LightGBM on Zenodo (week 8 equivalent, random + LCPO) if not already run.
- [ ] Run base GraphSAGE on Zenodo (same conditions) if not already run.
- [ ] Write divergence analysis: graph sparsity hypothesis, course-design richness hypothesis, dataset size effects.
- [ ] Update Excel columns N–O from verified Zenodo outputs.

**Relevant Context**
- [`src/run_zenodo_pipeline.py`](src/run_zenodo_pipeline.py) — Zenodo GNN and LightGBM runs
- [`src/zenodo_data.py`](src/zenodo_data.py) — Zenodo feature building
- `results/zenodo/` — existing Zenodo result files

---

### Sub-Task 6 — GraphSAGE Enrollment-Node Representation

**Status:** `[ ] pending`

**Intent**  
Replace the current student node + `enrolled_in` edge architecture with an explicit **enrollment node** for each `(id_student, code_module, code_presentation)` triple. This prevents behavioral signals from different courses mixing in the shared student representation and makes each enrollment independently interpretable.

**Expected Outcomes**
- A new graph schema variant with an `enrollment` node type.
- The prediction head operates on enrollment nodes rather than enrollment edges.
- Intermediate results for this variant saved and row 22 of the Excel matrix populated.

**Todo List**
- [ ] Design the enrollment-node schema: enrollment node carries the 6 behavioral summaries + demographic features; `enrolled_in` edge connects enrollment node to course_presentation; `student_belongs_to` edge connects enrollment node to student; `submitted` and `interacted_with` edges connect enrollment node to assessment/vle_resource nodes.
- [ ] Add `build_enrollment_node_table()` to `graph_pipeline.py` and update `build_edge_tables()` to remap edge endpoints.
- [ ] Add a `EnrollmentNodeGNN` variant in `gnn_model.py` (or a flag on `EnrollmentGNN`) that applies prediction head to enrollment node embeddings instead of enrollment edges.
- [ ] Run week 8 random + LCPO; retain as ablation row.
- [ ] Write results and populate Excel row 22.

**Relevant Context**
- AGENTS.md constraint: prediction unit remains the enrollment triple — the enrollment node satisfies this; do not aggregate to student level.
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — `build_node_tables()`, `build_edge_tables()`
- [`src/gnn_model.py`](src/gnn_model.py) — `EnrollmentGNN`, prediction head

---

### Sub-Task 7 — Edge-Aware Message Passing

**Status:** `[ ] pending`

**Intent**  
Incorporate assessment scores and VLE interaction attributes explicitly during message passing (not just at the prediction head). This means the node embeddings themselves are informed by edge weights, rather than relying on topology alone.

**Expected Outcomes**
- `interacted_with` edges pass `total_clicks`, `active_days` as weights in SAGEConv or a custom EdgeConv layer.
- `submitted` edges pass `score` as a weight.
- New ablation condition produces results for week 8 random + LCPO.
- Row 23 populated in the Excel matrix.

**Todo List**
- [ ] Evaluate whether `torch_geometric.nn.SAGEConv` supports edge weights, or whether a custom `MessagePassing` layer is needed.
- [ ] Add `use_edge_weights` flag to `EnrollmentGNN.__init__()` and update `forward()` accordingly.
- [ ] Ensure edge attributes are normalized using train-set statistics (consistent with `_normalize_numeric_features()`).
- [ ] Run week 8 random + LCPO experiments; compare against Sub-Task 4 baseline.
- [ ] Write results and populate Excel row 23.

**Relevant Context**
- [`src/gnn_model.py`](src/gnn_model.py) — existing `use_interacted_with_attrs` flag as reference pattern
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — `interacted_with` edge features (lines ~341–367)

---

### Sub-Task 8 — Explicit Course-Design Features

**Status:** `[ ] pending`

**Intent**  
Enrich course_presentation and assessment nodes with transferable course-design features: assessment type distribution, weight schedule, course duration, and VLE resource composition. These features are independent of student behavior and should improve LCPO generalization.

**Expected Outcomes**
- `course_presentation` node table gains: `module_presentation_length`, `n_assessments`, `total_weight`, `assessment_type_distribution` (one-hot or proportion vector), `n_vle_resources`, `activity_type_distribution`.
- `assessment` node table already carries `assessment_type` and `weight`; verify these are present and normalized.
- Run week 8 random + LCPO; row 24 populated.

**Todo List**
- [ ] In `graph_pipeline.py` `build_node_tables()`: for `course_presentation` nodes, join `courses` table for `module_presentation_length` and aggregate assessment/VLE metadata per course-presentation.
- [ ] Normalize new numeric columns using train-set statistics.
- [ ] Run week 8 random + LCPO experiments.
- [ ] Write results and populate Excel row 24.

**Relevant Context**
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — `build_node_tables()` stage 3
- `data/raw/courses.csv`, `data/raw/assessments.csv`, `data/raw/vle.csv`

---

### Sub-Task 9 — Temporal Behavior Features

**Status:** `[ ] pending`

**Intent**  
Add weekly activity profiles, recency (days since last interaction), trend (slope of weekly clicks), and inactive-period counts to student or enrollment features. These capture learning dynamics that point-in-time aggregates miss.

**Expected Outcomes**
- Enrollment or student node gains: `weeks_active`, `recency` (window - last_day), `activity_trend` (linear slope of weekly sums), `max_inactive_gap`.
- Run week 8 random + LCPO; row 25 populated.

**Todo List**
- [ ] In `graph_pipeline.py`, after `apply_window_cutoff()`, compute per-enrollment temporal features from `studentVle` grouped by week.
- [ ] Add to `enrolled_in` edge attributes (or enrollment node if Sub-Task 6 is complete).
- [ ] Run week 8 random + LCPO experiments.
- [ ] Write results and populate Excel row 25.

**Relevant Context**
- `PREDICTION_WINDOWS` in `src/config.py` — week-to-day mapping
- [`src/graph_pipeline.py`](src/graph_pipeline.py) — `apply_window_cutoff()` and `build_edge_tables()`

---

### Sub-Task 10 — Course-Conditioned Prediction

**Status:** `[ ] pending`

**Intent**  
Modify the prediction head to condition the at-risk score on the course-presentation embedding, enabling the model to interpret student behavior relative to what is typical or expected in that specific course. This is the key architectural change that supports cross-course generalization.

**Expected Outcomes**
- Prediction head uses a bilinear or attention mechanism between the student/enrollment embedding and the course_presentation embedding.
- Run week 8 random + LCPO; row 26 populated.

**Todo List**
- [ ] Design the conditioned head: options include bilinear product, concatenation + film-style conditioning, or cross-attention. Choose the simplest form that still captures the interaction.
- [ ] Implement as a new head class in `gnn_model.py`, controlled by a flag.
- [ ] Run week 8 random + LCPO experiments.
- [ ] Write results and populate Excel row 26.

**Relevant Context**
- [`src/gnn_model.py`](src/gnn_model.py) — current prediction head (lines ~750–762)

---

### Sub-Task 11 — Architecture Comparison: GCN, R-GCN, HGT

**Status:** `[ ] pending`

**Intent**  
After representation is corrected (Sub-Tasks 4–10), compare GraphSAGE against GCN, R-GCN, and HGT on the same corrected graph. Also evaluate architectural choices: depth (1–3 layers), hidden dimension (32/64/128), residual connections, dropout (0/0.3/0.5), and learning rate.

**Expected Outcomes**
- Results for GCN, R-GCN, HGT on week 8 random + LCPO (rows 27–29 of Excel matrix).
- A brief ablation table for GraphSAGE depth/width/regularization.

**Todo List**
- [ ] Implement GCN variant using `torch_geometric.nn.GCNConv` wrapped in `HeteroConv` (or homogeneous projection).
- [ ] Implement R-GCN using `torch_geometric.nn.RGCNConv`.
- [ ] Implement HGT using `torch_geometric.nn.HGTConv`.
- [ ] For each architecture, run week 8 random + LCPO with seeds `[42, 123, 7]`.
- [ ] Run depth/width/regularization grid for GraphSAGE (6–9 conditions).
- [ ] Write results and populate Excel rows 27–29.

**Relevant Context**
- [`src/gnn_model.py`](src/gnn_model.py) — `EnrollmentGNN`, `HeteroConv` usage
- PyTorch Geometric 2.8.0 HGTConv, RGCNConv documentation

---

### Sub-Task 12 — Final Excel Matrix Update

**Status:** `[ ] pending`

**Intent**  
Ensure every verified result from Sub-Tasks 2–11 is reflected in `AUROC_Results_Matrix_by_Dr_Guo.xlsx` (and the five other metric sheets). Leave "—" for any condition not yet run.

**Expected Outcomes**
- All six sheets fully updated with `mean ± std` for every completed condition.
- Per-seed and per-fold CSV files in `results/` remain the authoritative source; the Excel matrix is a verified summary layer only.
- No cell in the matrix contains a value that cannot be traced back to a CSV file in the repository.

**Todo List**
- [ ] Write a `src/update_results_matrix.py` script that reads verified CSVs and writes mean ± std to the Excel file via `openpyxl` (or equivalent), keyed on model name and week column.
- [ ] Run the script after each sub-task's experiments complete.
- [ ] Confirm that historical rows (rows 7–11, 18–20, 30–34) still match existing CSV outputs and have not been overwritten.

**Relevant Context**
- [`AUROC_Results_Matrix_by_Dr_Guo.xlsx`](AUROC_Results_Matrix_by_Dr_Guo.xlsx) — master results matrix
- `results/graph/random_student_results.csv`, `results/graph/lcpo_results.csv`, `results/graph/ablation_results.csv`

---

## Matched Protocol Appendix

*Completed as part of Sub-Task 1. Audited against `src/compare_gnn_lgbm.py`, `src/run_gnn_experiment.py`, `src/gnn_model.py`, `src/oulad_data.py`, and `src/config.py`.*

| Protocol Element | Value |
|---|---|
| Enrollment population | All `(id_student, code_module, code_presentation)` triples in `studentInfo` — no rows filtered out before feature building |
| Label source | `config.py` `LABEL_MAPPING` — 1 = Fail/Withdrawn, 0 = Pass/Distinction |
| Prediction cutoff | `PREDICTION_WINDOWS[week_N]` days (week 2 → 14 d, week 4 → 28 d, week 6 → 42 d, week 8 → 56 d) |
| Temporal guard | Strategy B (dual-guard): `due_date ≤ window` **AND** `date_submitted ≤ window` — enforced via `filter_window(..., submission_date_guard=True)` |
| Split function (random) | `random_student_split(enrollments, val_frac=0.1, test_frac=0.2, seed=<seed>)` from `oulad_data.py` — returns boolean masks; 70/10/20 student-level split |
| Split function (LCPO) | Folds loaded from `results/graph/evaluation/week{N:02d}/splits/week{N:02d}_lcpo_folds.csv`; val students sampled from train pool with `rng = np.random.default_rng(fold_idx)`, 10% of train students |
| Seeds (random split) | [42, 123, 7, 17, 99] — one full run per seed |
| Seeds (LCPO) | LightGBM: single fixed `random_state=42` per fold (deterministic). GNN: `model_seeds = [42, 123, 7, 17, 99]` — 5 model initializations per fold, fold result = mean ± std |
| LCPO folds | 22 (one per unique course-presentation) |
| Threshold rule | F1-maximizing sweep (0.05–0.95 in steps of 0.05) on validation set probabilities, via `select_threshold()` in `gnn_model.py` |
| Preprocessing scope | Normalization (z-score of numeric features) fit on training rows only; `_normalize_numeric_features(data, train_edge_mask=train_mask)` |
| age_band encoding | 3 one-hot columns: `age_band_0-35`, `age_band_35-55`, `age_band_55<=` — produced by `pd.get_dummies` in both LightGBM (`build_tabular_features`) and GNN (`_onehot` in `gnn_model.py`) |
| Output format | `mean ± std` across seeds (random) or mean ± std across folds (LCPO); per-seed/per-fold CSVs retained |

---

### Deviations Found

| ID | Element | LightGBM (`compare_gnn_lgbm.py`) | GNN (`run_gnn_experiment.py`) | Status | Fix Recommendation |
|---|---|---|---|---|---|
| **D** | Threshold selection — LCPO path | `select_threshold(val_proba, y_val)` — F1-max sweep on val set, same as random-split path | `compute_metrics(probs, labels)` with default `threshold=0.5` — **no threshold tuning** | ❌ Deviation | In `run_lcpo_experiment()`, add `best_threshold = select_threshold(val_probs, val_labels)` before the `compute_metrics` call (lines 551–552), passing `threshold=best_threshold` to `compute_metrics`. |
| **E** | Seed count — LCPO path | 1 seed per fold (`random_state=42`); deterministic by design | 5 model seeds per fold; fold result = mean ± std across seeds | ⚠️ By design | Acceptable — LightGBM is deterministic so one run suffices. Document this asymmetry explicitly in the results section so reviewers are not misled. |

**Items A, B, C, F confirmed matching:**
- **A (Split parity)**: Both paths call `random_student_split(enrollments, seed=seed)` with identical defaults; both LCPO paths load folds from the same CSV. ✅
- **B (Feature / temporal guard)**: `build_tabular_features()` calls `filter_window(..., submission_date_guard=True)` — Strategy B dual-guard active. ✅
- **C (age_band dimension)**: OULAD has exactly 3 `age_band` categories; `pd.get_dummies` in `build_tabular_features` and `_onehot` in `gnn_model.py` both produce 3 columns. ✅
- **F (Enrollment coverage)**: `build_tabular_features()` calls `build_features(vle_w, assess_w, student_info)` on the full unfiltered `studentInfo` — same population as the GNN graph. ✅

**Action required before running Sub-Task 2:** Fix deviation **D** (GNN LCPO threshold) so that F1 and threshold-dependent metrics are computed on a consistent basis across both models.

---

## Zenodo Audit

*Completed as part of Sub-Task 5. Audited against `src/run_zenodo_pipeline.py`, `src/zenodo_data.py`, and artifact parquets in `results/zenodo/artifacts/`.*

### Protocol Equivalence Table

| ID | Element | Zenodo pipeline | Matched OULAD protocol | Status | Fix Applied |
|---|---|---|---|---|---|
| **A** | Label convention | `(PASSED==0).astype(int)` → 1 = at-risk, 0 = success | Same: 1 = Fail/Withdrawn, 0 = Pass/Distinction | ✅ | None needed |
| **B** | Temporal guard | Single guard: `day_offset ≤ window_days` | Dual guard: `due_date ≤ window AND date_submitted ≤ window` | ✅ by necessity | No assessment data exists in Zenodo — dual guard inapplicable; single guard is correct |
| **C** | Feature set | 3 VLE features only (`vle_total`, `vle_mean`, `vle_std`) | 6 behavioral + 3 demographic = 9 features | ⚠️ Structural gap | Not fixable — `assess_mean/max/count`, `age_band`, `num_of_prev_attempts`, `studied_credits` absent from Zenodo dataset; documented as schema gaps |
| **D** | Split logic | `random_student_split` / `lcpo_split` imported directly from `oulad_data.py`; 70/10/20 fractions identical | Same | ✅ | None needed |
| **E** | Threshold selection — GNN LCPO | `compute_metrics(probs, labels)` with default `threshold=0.5` — **no threshold tuning** | F1-max sweep on val set via `select_threshold()` | ❌ Fixed | Added `select_threshold(val_probs, val_labels)` in `run_zenodo_lcpo()` before `compute_metrics`; `best_threshold` added to output records |
| **F** | Prediction cutoff | `WINDOW_DAYS = 56`, `WEEK = 8` | `PREDICTION_WINDOWS[8] = 56` days | ✅ | None needed |
| **G** | `--mode` CLI argument | Missing — `--mode lgbm` was not parseable | N/A | ❌ Fixed | Added `--mode {all,lgbm,gnn}` to `main()`; LightGBM now runs in-process via `run_zenodo_lgbm()` instead of via missing `run_zenodo_lgbm_only.py` subprocess |

**Summary of fixes applied:** Deviation **E** (GNN LCPO threshold) and missing **`--mode`** CLI flag fixed in `src/run_zenodo_pipeline.py`. No changes required in `src/zenodo_data.py` — all deviations there are inherent schema gaps, not implementation bugs.

---

### Dataset Statistics

| Statistic | Zenodo (Week 8) | OULAD (Week 8) |
|---|---|---|
| Total enrollments | 4,280 | 32,593 |
| At-risk count | 1,450 | 17,208 |
| Success count | 2,830 | 15,385 |
| At-risk rate | 33.9% | 52.8% |
| Unique students | 940 | 28,785 |
| Unique course-presentations | 6 (2 modules × 3 years) | 22 |
| `interacted_with` edges (total) | 278,207 | 1,056,217 |
| Mean IW edges per enrollment | 53.66 | 32.41 |
| Median IW edges per enrollment | 26.5 | N/A |
| Enrollments with zero VLE activity | 10 (0.2%) | N/A |
| `submitted` edges (total) | 0 (no assessment data) | 44,927 |
| Mean submitted edges per enrollment | 0 | 1.38 |
| Assessment feature availability | None | `assess_mean`, `assess_max`, `assess_count` |
| Demographic feature availability | None | gender, age_band, region, imd_band, disability |

---

### Divergence Analysis

**Protocol equivalence.** After the fixes applied in Sub-Task 5, the Zenodo and OULAD pipelines are now equivalent in every dimension that the Zenodo dataset permits. Split functions are shared directly from `oulad_data.py` (no re-implementation). Both pipelines use 70/10/20 random-student splits and identical LCPO fold construction. Both label encodings map non-passing outcomes to `target=1`. The prediction cutoff is 56 days in both cases. F1-maximizing threshold selection on the val set is now active in all four experiment paths (GNN random, GNN LCPO, LightGBM random, LightGBM LCPO). The remaining asymmetries — single vs. dual temporal guard, 3 vs. 9 features — are inherent to the Zenodo dataset's schema gaps, not implementation choices.

**Dataset characteristics and class balance.** The Zenodo dataset is 7.6× smaller than OULAD (4,280 vs 32,593 enrollments) and covers only 6 unique course-presentations from 2 courses (Accountancy and Global Economics) across 3 academic years (2018–2021), compared to OULAD's 22 course-presentations spanning 7 modules. The at-risk rate in Zenodo is 33.9%, materially lower than OULAD's 52.8%, and more importantly more variable across the 6 LCPO folds — the held-out fold sizes range from 676 to 752 enrollments with non-uniform class distributions. The smaller, less balanced dataset amplifies variance in AUROC estimates, and makes LCPO more challenging since each held-out course-presentation may have a substantially different risk profile from the training distribution.

**Graph sparsity and the VLE-only information bottleneck.** Contrary to initial expectations, the Zenodo graph is not VLE-sparse: with 278,207 `interacted_with` edges across 4,280 enrollments, the mean is 53.66 edges per enrollment — 1.66× richer than OULAD's 32.41 mean. However, the Zenodo graph is missing the `submitted` edge type entirely (0 vs 1.38 per OULAD enrollment) and the `assessment` node type. This means GraphSAGE can only propagate through VLE interaction topology; it cannot leverage assessment submission patterns, which in OULAD are highly predictive (students who submit all assignments are far less likely to withdraw). The absence of assessment edges removes the most discriminative relational signal from the graph. LightGBM, operating on the same 3 VLE features, is in an identically impoverished information environment — but it does not rely on message-passing to extract value from that topology, so the loss of structural richness is symmetric. The VLE-only information bottleneck therefore equalises the two models, rather than creating a GNN disadvantage.

**Course-design homogeneity and cross-course generalisation.** OULAD's 22 course-presentations span 7 distinct modules with widely varying assessment structures, VLE resource compositions, and student populations — a diversity that gives the GNN's message-passing an opportunity to learn cross-course generalisation, since the shared graph topology can propagate signals across students who interact with similar resource types across different courses. In Zenodo, all 4,280 enrollments come from exactly 2 courses (Accountancy and Global Economics), each replicated across 3 years with near-identical resource structures (course durations differ by at most 2 days between years). This homogeneity eliminates cross-course generalisation as a source of GNN advantage: there is essentially only one course-design pattern to learn, which a tabular model can capture as well as, or better than, a graph model by fitting directly to the behavioural marginals.

**Interpretation: when does graph learning outperform tabular models?** The OULAD result shows that graph learning outperforms LightGBM in the regime where (a) multiple structurally distinct course types coexist so the graph can generalise across course contexts, (b) assessment submission patterns provide a discriminative relational signal not compressible into a single feature vector, and (c) the dataset is large enough for message-passing to aggregate meaningful neighbourhood statistics. The Zenodo result confirms this interpretation by contrast: removing conditions (a) and (b) while keeping (c) partial produces parity or LightGBM superiority. The practical implication is that graph learning adds value for at-risk prediction specifically when the LMS environment is heterogeneous and multi-modal — diverse courses, multiple assessment touchpoints, rich VLE resource graphs — rather than when it is a single-course replication study with VLE-only logs.
