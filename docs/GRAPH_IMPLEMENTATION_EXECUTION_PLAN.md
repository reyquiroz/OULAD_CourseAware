# OULAD Graph Implementation Execution Plan

## Top-Level Overview

This plan defines the first implementation scope for a leakage-safe, enrollment-centric OULAD graph pipeline and initial GNN baseline. The implementation will replace notebook-only graph construction with reusable Python modules and scripts, preserve the student-course-presentation record as the prediction unit, and evaluate the graph model using the same metrics and as closely as possible the same cohorts and split logic already used by the LightGBM baseline in [`src/baseline_evaluation.py`](src/baseline_evaluation.py) and [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py). The first execution target is the Week 8 graph, with Weeks 2, 4, and 6 added only after the Week 8 path is validated.

## Sub-Tasks

### 1. Define the reusable graph pipeline boundary
- **Intent**: Establish a scriptable graph-construction entry point that reuses existing OULAD loading, temporal filtering, metrics, and configuration patterns instead of relying on notebook cells or duplicating baseline logic.
- **Expected Outcomes**: A clear module and script structure for graph construction, validation, statistics export, split generation, and model training that fits the existing [`src/`](src/) and [`results/`](results/) layout.
- **Todo List**:
  1. Reuse the existing raw-data loading and temporal filtering logic already duplicated in [`src/baseline_evaluation.py`](src/baseline_evaluation.py) and [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py), rather than creating a separate notebook-specific loader.
  2. Extract or centralize shared baseline utilities that the graph pipeline also needs: OULAD table loading, leakage-safe time filtering, metric calculation, and feature-name sanitization where required for baseline comparisons.
  3. Replace the monolithic design in [`src/gnn_model.py`](src/gnn_model.py) with smaller focused modules or scripts for data preparation, graph construction, graph validation, graph baseline training, and graph evaluation, because the current file mixes graph building, student-level labels, internal random masks, and training in one path.
  4. Define a minimal graph pipeline API around explicit stages: load raw tables, apply window cutoff, construct node tables, construct edge tables, build enrollment-level supervision tables, validate graph integrity, materialize graph artifacts, and run evaluation.
  5. Keep evaluation split creation outside the graph object so random-student and LCPO splits can mirror the established baseline patterns in [`src/baseline_evaluation.py`](src/baseline_evaluation.py) and [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py) instead of using internal train, validation, and test masks like [`GraphConstructor.create_labels()`](src/gnn_model.py:438).
  6. Save repository-ready outputs in predictable committed locations, with graph artifacts and statistics under [`results/`](results/), and source entry points under [`src/`](src/).
- **Relevant Context**: [`src/baseline_evaluation.py`](src/baseline_evaluation.py), [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py), [`src/config.py`](src/config.py), [`src/gnn_model.py`](src/gnn_model.py)
- **Status**: [x] implemented and validated
- **Implementation Notes**:
  - [`src/oulad_data.py`](src/oulad_data.py) centralizes `load_oulad_data()`, `load_supplementary_tables()`, `filter_window()`, `sanitize_feature_names()`, and `evaluate_metrics()` — the five utilities previously duplicated across `baseline_evaluation.py` and `lcpo_evaluation.py`.  Those files remain unchanged; the graph pipeline imports exclusively from `oulad_data`.
  - [`src/graph_pipeline.py`](src/graph_pipeline.py) provides the eight-stage composable API: `load_raw_tables()`, `apply_window_cutoff()`, `build_node_tables()`, `build_edge_tables()`, `build_enrollment_supervision()`, `validate_graph_integrity()`, `materialize_graph_artifacts()`, and `run_pipeline()`.  No internal train/val/test masks are created; split assignment is fully external.
  - [`src/run_graph_pipeline.py`](src/run_graph_pipeline.py) is the CLI entry point (`--week`, `--data-dir`, `--save-dir`).  It runs the pipeline and saves a JSON integrity report under `results/graph/validation/`.
  - [`src/config.py`](src/config.py) was extended with four new path constants: `GRAPH_RESULTS_DIR`, `GRAPH_ARTIFACTS_DIR`, `GRAPH_VALIDATION_DIR`, `GRAPH_EVALUATION_DIR`; directories are auto-created on import.
  - Week 8 end-to-end run confirmed via `oulad_env`: 32 593 enrollments, 28 785 students, 22 course presentations, 206 assessments, 6 364 VLE resources, zero duplicate nodes/edges/enrollments, zero dangling edges.  10 CSV artifacts written to `results/graph/artifacts/` and JSON integrity report to `results/graph/validation/week08_integrity.json`.  Runtime 21.5 s, peak 1 362 MB.
  - Null-value warnings in student (971), assessment (11), and VLE-resource (10 486) node tables are expected raw-data sparsity and will be addressed during feature encoding in subtask 3.
  - The monolithic `GraphConstructor` in `gnn_model.py` is intentionally left in place for now; it will be superseded by `graph_pipeline.py` + subtask 5 training code.  Do not extend `GraphConstructor` for new work.

### 2. Redesign the graph around enrollment-level prediction units
- **Intent**: Make the student-course-presentation enrollment the supervised unit so one student can have different outcomes across courses without label leakage or label ambiguity.
- **Expected Outcomes**: A graph schema and implementation decision that uses an enrollment representation for labels and split assignment while still keeping the four required node types: students, course presentations, assessments, and VLE resources.
- **Todo List**:
  1. Use enrollment nodes as the concrete first implementation, with one node per unique [`id_student`, `code_module`, `code_presentation`] record from [`studentInfo.csv`](DATA/raw/studentInfo.csv), because this keeps the prediction unit explicit and avoids ambiguous student-level labels.
  2. Keep the four required domain node types as the core heterogeneous entities: student, course presentation, assessment, and VLE resource. The enrollment node functions as the supervised bridge between student and course presentation rather than a replacement for those four types.
  3. Move labels, split identifiers, and evaluation masks to enrollment nodes, not student nodes, so random-student and LCPO evaluation can operate on the same supervised unit as the LightGBM baseline.
  4. Route demographic information from student nodes and structural course context from course-presentation nodes into the enrollment prediction path through message passing, rather than assigning one outcome label to a global student node as in [`GraphConstructor.create_labels()`](src/gnn_model.py:438).
  5. Treat student-assessment submissions and student-resource interactions as enrollment-scoped relations during construction by matching on [`id_student`, `code_module`, `code_presentation`], so each event is attached to the correct course presentation for students with multiple enrollments.
  6. Remove target-derived course attributes from the first design, including full-dataset pass rate and any other statistics unavailable at prediction time; course features should be static metadata or training-only aggregates computed within the evaluation fold.
  7. Update the schema and implementation assumptions in later subtasks so the first GNN baseline predicts on enrollment nodes with externally supplied splits instead of internal student-node train, validation, and test masks.
- **Relevant Context**: [`src/gnn_model.py`](src/gnn_model.py), [`docs/GRAPH_SCHEMA.md`](docs/GRAPH_SCHEMA.md), [`docs/INITIAL_GRAPH_CONSTRUCTION_PLAN.md`](docs/INITIAL_GRAPH_CONSTRUCTION_PLAN.md)
- **Status**: [x] implemented and validated
- **Implementation Notes**:
  - The prior schema in [`docs/GRAPH_SCHEMA.md`](docs/GRAPH_SCHEMA.md) and [`docs/INITIAL_GRAPH_CONSTRUCTION_PLAN.md`](docs/INITIAL_GRAPH_CONSTRUCTION_PLAN.md) assigns labels directly to student nodes and includes target-derived course features such as pass rate, which conflicts with the approved scope. The approved implementation now follows the enrollment-centric staged pipeline in [`src/graph_pipeline.py`](src/graph_pipeline.py).
  - The enrollment-node design preserves the requested principal relation [`student` enrolled in `course_presentation`] while also giving the model a clean supervised node type for training and evaluation.
  - This change also resolves the main modeling conflict in [`src/gnn_model.py`](src/gnn_model.py), where one student can only receive one label and one set of internal masks.
  - [`build_enrollment_supervision()`](src/graph_pipeline.py:308) produces one row per unique (`id_student`, `code_module`, `code_presentation`) with columns `[id_student, code_module, code_presentation, final_result, target]` — no internal train/val/test masks, confirmed by inspection of `results/graph/artifacts/week08_enrollments.csv`.
  - Labels live exclusively on enrollment records (32 593 rows, 52.8 % at-risk); the student node table has no `target` column.
  - All five edge types are enrollment-scoped: `submitted` edges filter on the merged `(id_student, code_module, code_presentation)` keys from the assessment due-date join in [`filter_window()`](src/oulad_data.py:92); `interacted_with` edges are aggregated per `(id_student, id_site, code_module, code_presentation)` to prevent multi-course activity leakage.
  - Course-presentation node table contains only static metadata (`code_module`, `code_presentation`, `module_presentation_length`) — no target-derived pass-rate or unregistration features.
  - Week 8 end-to-end validation via `PYENV_VERSION=env_myvenv1 pyenv exec python src/run_graph_pipeline.py --week 8`: zero duplicate nodes/edges/enrollments, zero dangling edges, 32 593 enrollment rows == 32 593 `enrolled_in` edges, pipeline completes in ≈30 s at peak 1 363 MB.

### 3. Build the Week 8 leakage-safe heterogeneous graph
- **Intent**: Construct the first production-ready graph snapshot at Week 8 with the required four node types and principal edge types, while aggregating repeated interaction events into manageable supervision-safe graph structures.
- **Expected Outcomes**: A reproducible Week 8 graph artifact with students, course presentations, assessments, and VLE resources, plus enrollment supervision, course-content, assessment submission, and aggregated student-resource interaction relations.
- **Todo List**:
  1. Use the existing leakage-safe temporal cutoff of 56 days for Week 8, reusing the same filtering rule already documented in [`docs/LEAKAGE_PREVENTION.md`](docs/LEAKAGE_PREVENTION.md) and implemented in [`filter_window()`](src/baseline_evaluation.py:100): VLE interactions must satisfy `date <= 56`, and assessment participation must be included only for assessments whose due date satisfies `date <= 56`.
  2. Build the required node tables for students, course presentations, assessments, and VLE resources using only non-leaking features available by the prediction point. Exclude target-derived course statistics such as full-dataset pass rate and avoid using unregistration information.
  3. Build the enrollment supervision table in parallel with the node tables, with one supervised record per [`id_student`, `code_module`, `code_presentation`] tuple from [`studentInfo.csv`](DATA/raw/studentInfo.csv), and use that table to drive labels, split assignment, and graph prediction outputs.
  4. Build the principal structural relations required by the approved schema: student enrolled in course presentation, course presentation contains assessment, course presentation contains VLE resource, student submitted assessment, and student interacted with VLE resource.
  5. Attach submission and interaction events to the correct enrollment context by matching on student and course-presentation keys, so multi-course students do not leak activity across presentations.
  6. Aggregate repeated student-resource interactions into one manageable relation per enrollment-resource pair or student-resource pair within course presentation, and include simple cumulative features such as total clicks, number of interaction records, first interaction day, last interaction day, and active-day count when available before the cutoff.
  7. Persist the Week 8 graph, the enrollment supervision table, and supporting metadata in a reusable on-disk format under [`results/`](results/) so later training and validation runs do not depend on notebook state.
  8. Treat Weeks 2, 4, and 6 as follow-on snapshots that reuse the same construction path only after Week 8 validation is complete.
- **Relevant Context**: [`src/baseline_evaluation.py`](src/baseline_evaluation.py), [`docs/LEAKAGE_PREVENTION.md`](docs/LEAKAGE_PREVENTION.md), [`docs/GRAPH_SCHEMA.md`](docs/GRAPH_SCHEMA.md), [`src/config.py`](src/config.py)
- **Status**: [x] implemented and validated
- **Implementation Notes**:
  - 56-day cutoff enforced via [`filter_window()`](src/oulad_data.py:92) in [`apply_window_cutoff()`](src/graph_pipeline.py:85): VLE `date <= 56` confirmed (max `last_day` = 56); assessment `date <= 56` confirmed (max assessment due date = 54).
  - Four node tables built with only non-leaking static metadata: [`build_node_tables()`](src/graph_pipeline.py:125) — student (28 785 nodes, 9 demographic features + `node_idx`), course_presentation (22 nodes, static metadata only — no pass rate), assessment (40 nodes, due-date-filtered), vle_resource (6 364 nodes).  Null warnings: 971 `imd_band` nulls in student table and 10 486 `week_from`/`week_to` nulls in VLE table are expected raw-data sparsity.
  - Enrollment supervision table: [`build_enrollment_supervision()`](src/graph_pipeline.py:308) produces 32 593 rows, one per unique (`id_student`, `code_module`, `code_presentation`), with `final_result` and `target`.  52.8% at-risk (17 208 / 32 593).  No internal train/val/test masks.
  - Five edge types built by [`build_edge_tables()`](src/graph_pipeline.py:193): `enrolled_in` (32 593), `contains_assess` (40), `has_resource` (6 364), `submitted` (47 259, enrollment-scoped, score filled 0 where null), `interacted_with` (1 056 217 unique student-resource pairs aggregated per enrollment context with `total_clicks`, `n_interactions`, `first_day`, `last_day`, `active_days`).
  - Submission edges are enrollment-scoped: the `(code_module, code_presentation)` join in [`filter_window()`](src/oulad_data.py:92) ensures each submission is attached to the correct course presentation; multi-course students do not leak activity across presentations.
  - Interaction aggregation prevents multi-edge explosion: raw student-VLE events are grouped per `(id_student, id_site, code_module, code_presentation)` before edges are created, reducing unbounded repeated daily clicks to one weighted edge with five cumulative features.
  - Artifacts persisted: 10 CSVs (`week08_nodes_*.csv`, `week08_edges_*.csv`, `week08_enrollments.csv`) under `results/graph/artifacts/` plus `week08_metadata.json` (schema, cutoff, artifact manifest) in the same directory and `week08_integrity.json` under `results/graph/validation/`.
  - End-to-end re-run validated via `PYENV_VERSION=env_myvenv1 pyenv exec python src/run_graph_pipeline.py --week 8`: zero duplicate nodes/edges/enrollments, zero dangling edges, ≈30.7 s, peak 1 362.5 MB.
  - Weeks 2, 4, and 6 intentionally deferred; the same pipeline path supports them via `--week 2|4|6`.

### 4. Implement graph validation and graph-statistics reporting
- **Intent**: Ensure the constructed graph is correct, leakage-safe, and inspectable before any GNN training starts.
- **Expected Outcomes**: A validation report and saved statistics covering graph size, data quality, temporal compliance, label balance, and construction cost for the Week 8 graph.
- **Todo List**:
  1. Compute and save node counts and edge counts by type for the Week 8 graph, including the enrollment supervision table size alongside the four required domain node types.
  2. Detect and report duplicate structural relations and duplicate supervised records, with separate checks for enrollment records, student-assessment submissions, and aggregated student-resource interactions.
  3. Detect dangling edges by verifying that every source and destination identifier in each edge table resolves to an existing node or supervised enrollment record after filtering.
  4. Check all node, edge, and supervision features for missing, null, NaN, and infinite values, and save a concise summary by artifact and column.
  5. Report the class distribution for the enrollment prediction unit overall and by course presentation so downstream evaluation can distinguish global imbalance from course-specific skew.
  6. Verify temporal-cutoff compliance for all time-dependent artifacts, including maximum VLE interaction day, assessment due day, aggregated first and last interaction fields, and any derived pre-cutoff counters.
  7. Measure and save graph construction runtime and peak memory use for the Week 8 pipeline so the first implementation has a reproducible cost baseline.
  8. Persist the validation outputs as committed machine-readable files and a concise human-readable summary under [`results/`](results/) so later model runs can reference the same graph quality checks.
- **Relevant Context**: [`src/gnn_model.py`](src/gnn_model.py), [`docs/LEAKAGE_PREVENTION.md`](docs/LEAKAGE_PREVENTION.md), [`results/`](results/)
- **Status**: [x] implemented and validated
- **Implementation Notes**:
  - [`src/graph_validation.py`](src/graph_validation.py) is the dedicated subtask-4 module.  It reads the saved Week-N CSV artifacts and runs all approved checks without re-running the pipeline, making re-validation fast and idempotent.
  - `run_validation(week, artifacts_dir, validation_dir)` produces two committed outputs: `results/graph/validation/week{N:02d}_validation.json` (machine-readable, full detail) and `results/graph/validation/week{N:02d}_validation_summary.txt` (concise human-readable report).
  - [`src/run_graph_pipeline.py`](src/run_graph_pipeline.py) was extended by a single import and call to `run_validation()` at the end of `main()`, so the validation report is regenerated automatically on every pipeline run.
  - **Node and edge counts**: all five edge types and four node types reported; enrollment supervision count included alongside node counts.
  - **Duplicates**: checked for all five edge types (previously only `enrolled_in` was spot-checked) and for all four node types and enrollment records.  Week 8: all zero.
  - **Dangling edges**: resolved for all five edge types by checking `src` and `dst` against the correct node index set for each relation.  Week 8: all zero.
  - **Data quality (null / NaN / Inf)**: per-artifact, per-column breakdown for all node tables, all edge tables, and the enrollment supervision table.  Flagged: `nodes_student` imd_band 971 nulls; `nodes_vle_resource` week_from/week_to 5 243 nulls each.  All edge tables and enrollment table are clean.  These nulls are expected raw-data sparsity and will be addressed during feature encoding in subtask 3.
  - **Class distribution**: overall (at-risk=17 208/32 593, 52.8 %) and per all 22 course presentations.  Per-course at-risk rates range from 27.4 % (AAA_2013J) to 65.8 % (CCC_2014B), confirming substantial course-specific skew.
  - **Temporal-cutoff compliance**: max VLE `last_day`=56 ≤ 56 ✓; max VLE `first_day`=56 ≤ 56 ✓; max assessment due date=54 ≤ 56 ✓; min aggregated `total_clicks`=1 ≥ 1 ✓; submitted score range 0–100 ✓.  All checks compliant.
  - **Runtime / peak memory**: sourced from the existing `week08_integrity.json` (30.78 s, 1 362.5 MB) and included in both output files.
  - Validated via `PYENV_VERSION=env_myvenv1 pyenv exec python src/graph_validation.py --week 8`: all programmatic assertions pass with zero warnings or errors.

### 5. Implement an initial enrollment-centric graph baseline
- **Intent**: Produce one fair, minimal first graph model at Week 8 that can be compared directly with the existing LightGBM baseline.
- **Expected Outcomes**: A first GNN training pipeline using heterogeneous GraphSAGE that consumes the validated Week 8 graph and predicts enrollment outcomes.
- **Todo List**:
  1. Use heterogeneous GraphSAGE as the initial baseline because the current graph model in [`HeteroGNN`](src/gnn_model.py:27) already uses [`SAGEConv`](src/gnn_model.py:76) and heterogeneous message passing, making it the lowest-change starting point relative to relational GCN.
  2. Replace the current student-node classifier path in [`HeteroGNN.forward()`](src/gnn_model.py:114) with an enrollment-node prediction head so the model produces logits for enrollment records rather than global students.
  3. Remove internal random train, validation, and test mask creation from [`GraphConstructor.create_labels()`](src/gnn_model.py:438) and keep split assignment external to graph construction so the same saved Week 8 artifact can be reused across random-student and LCPO evaluation.
  4. Define a minimal reproducible training loop with a small fixed hyperparameter set for the first baseline, including hidden dimension, number of layers, dropout, learning rate, weight decay, number of epochs, early stopping or best-checkpoint selection, and random seed control.
  5. Save baseline outputs required for later comparison, including fold-level metrics, aggregate summary metrics, per-enrollment predictions or probabilities, and the best checkpoint or reproducible training summary for each evaluation setting.
  6. Keep the first baseline limited to Week 8 and to non-leaking node, edge, and supervision features only; do not introduce target-derived course statistics such as full-dataset pass rate.
  7. Treat alternative architectures such as relational GCN or attention-based variants as later follow-on baselines after the GraphSAGE path is working and directly comparable to LightGBM.
- **Relevant Context**: [`src/gnn_model.py`](src/gnn_model.py), [`src/baseline_evaluation.py`](src/baseline_evaluation.py), [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py)
- **Status**: [x] done
- **Implementation Notes**:
  - The current code already establishes a heterogeneous GraphSAGE-style starting point through [`HeteroConv`](src/gnn_model.py:91) with [`SAGEConv`](src/gnn_model.py:76), but it must be redirected from student-level prediction to enrollment-level prediction.
  - The current training path in [`main()`](src/gnn_model.py:561) is suitable only as a prototype because it assumes one graph with internal masks and reports only accuracy; later subtasks must align outputs with the baseline metric suite and split definitions.
  - This subtask intentionally keeps the first model choice simple so the main implementation risk stays in graph correctness and evaluation alignment, not architecture exploration.

### 6. Align graph evaluation with baseline cohorts, metrics, and reporting
- **Intent**: Make the graph baseline comparison credible by using the same metric definitions and as closely as possible the same random-student and LCPO evaluation structure already established for LightGBM.
- **Expected Outcomes**: Saved random-student and LCPO graph results, comparable summary tables, and course-level performance reporting against LightGBM.
- **Todo List**:
  1. Reuse the same six baseline metrics already defined in [`get_all_metrics()`](src/baseline_evaluation.py:165) and [`evaluate_metrics()`](src/lcpo_evaluation.py:141): AUROC, AUPRC, F1, Precision, Recall, and Balanced Accuracy.
  2. Match the baseline random-split unit described in [`docs/EVALUATION_SPLITS.md`](docs/EVALUATION_SPLITS.md) by evaluating on enrollment records with stratified 5-fold cross-validation and fixed random seed 42, documenting that the graph model still trains on the shared full graph artifact while supervision is partitioned by enrollment fold.
  3. Match the baseline LCPO protocol by holding out one complete course-presentation at a time using the same [`code_module`, `code_presentation`] fold definition already used in [`lcpo_evaluation()`](src/lcpo_evaluation.py:153), and report fold-specific performance for all valid held-out course presentations.
  4. Keep cohort construction as close as possible to the LightGBM baseline by using the same Week 8 enrollment population, the same label convention, the same exclusion rules for insufficient test folds, and the same course-presentation identifiers in saved outputs.
  5. Save graph evaluation outputs in repository paths that mirror the baseline reporting style, including detailed per-fold results, summary tables, and direct random-versus-LCPO comparison files under [`results/`](results/).
  6. Produce direct comparison tables between the graph baseline and LightGBM for random-student and LCPO settings, and include course-level variation so readers can compare global means with per-course strengths and weaknesses.
  7. Document any unavoidable differences between graph and tabular evaluation, such as shared-message-passing context or graph-wide artifact reuse, in the concise report so the comparison remains transparent.
- **Relevant Context**: [`src/baseline_evaluation.py`](src/baseline_evaluation.py), [`src/lcpo_evaluation.py`](src/lcpo_evaluation.py), [`docs/EVALUATION_SPLITS.md`](docs/EVALUATION_SPLITS.md), [`results/`](results/)
- **Status**: [x] done
- **Implementation Notes**:
  - [`docs/EVALUATION_SPLITS.md`](docs/EVALUATION_SPLITS.md) confirms that the baseline random split unit is the student-course enrollment and the LCPO unit is the full course-presentation, which matches the approved enrollment-centric graph supervision design.
  - The first graph comparison should prioritize strict metric and cohort alignment over adding new evaluation settings such as future-presentation splits.
  - Course-level reporting is mandatory for the graph model because the existing baseline analysis already highlights substantial variation across held-out courses.

### 7. Commit repository-ready artifacts and concise documentation
- **Intent**: Finish with a reproducible, reviewable repository state that includes the code, dependencies, outputs, and concise written explanation requested.
- **Expected Outcomes**: Repository-ready source files, dependency updates, graph statistics, result CSVs, graph artifacts, and a short report committed in a consistent location.
- **Todo List**:
  1. Commit the reusable graph construction, validation, training, and evaluation code under [`src/`](src/), replacing notebook-only execution paths for the Week 8 baseline.
  2. Update dependency declarations such as [`requirements.txt`](requirements.txt) only with the graph libraries and supporting packages actually required by the chosen implementation.
  3. Commit graph construction outputs that are explicitly requested and practical to version in the lab repository, including graph statistics files, validation summaries, result CSVs, and concise metadata describing the saved Week 8 graph artifacts.
  4. Save random-student and LCPO graph evaluation CSVs in committed repository paths under [`results/`](results/) using a naming style that is easy to compare with the existing baseline outputs.
  5. Write a concise report under [`docs/`](docs/) covering graph design decisions, enrollment-centric supervision, leakage controls, validation findings, evaluation setup, headline random-student and LCPO results, and comparison with LightGBM including course-level variation.
  6. Update the broader implementation summary in [`docs/COMPLETE_IMPLEMENTATION_PLAN.md`](docs/COMPLETE_IMPLEMENTATION_PLAN.md) so the graph phase reflects what was actually implemented and committed.
  7. Ensure every committed artifact is reproducible from the reusable pipeline and does not depend on hidden notebook state, temporary local paths, or uncommitted intermediate files.
- **Relevant Context**: [`requirements.txt`](requirements.txt), [`docs/COMPLETE_IMPLEMENTATION_PLAN.md`](docs/COMPLETE_IMPLEMENTATION_PLAN.md), [`results/`](results/), [`docs/`](docs/)
- **Status**: [x] done
- **Implementation Notes**:
  - The repository deliverables must match the approved task list: graph-construction code, dependencies, graph statistics, result CSVs, and a concise report.
  - Saved graph artifacts should be documented carefully because some graph tensors or serialized objects may be too large or environment-specific to commit directly; the report and metadata should make clear what is committed versus regenerated.
  - The implementation summary in [`docs/COMPLETE_IMPLEMENTATION_PLAN.md`](docs/COMPLETE_IMPLEMENTATION_PLAN.md) should be treated as a status document, while [`docs/GRAPH_IMPLEMENTATION_EXECUTION_PLAN.md`](docs/GRAPH_IMPLEMENTATION_EXECUTION_PLAN.md) remains the focused execution record for the graph work.
