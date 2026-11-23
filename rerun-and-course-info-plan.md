# Plan: Post-fix reruns, course-information experiment, Zenodo, IRB and reporting

Source: reviewer feedback received after the 2026-10-16 code review. Work order follows the
reviewer's numbering (1 fixes, 2 matched rerun, 3 course-information experiment, 4 Zenodo,
5 IRB, then reporting).

**Execution venue (decided by the user).** All work, including the full GNN grid, runs
**locally on CPU first**. Results count as final unless a later server run differs. The
reviewer asked for the lab GPU server `lx138126-cse.eng.unt.edu` (no TACC) for all runs, so
the local-first choice deviates from the reviewer's instruction and must be reported as such
(see Validation Question 7). Only OULAD and Zenodo data may be placed on that server until the
IRB and data agreements are in place.

## Top-Level Overview

**Goal.** Make every result in the matrix and the report come from code that includes the
October 16 fixes (early-unregistered exclusion, `--feature-set`, `use_enrollment_nodes`,
LCPO dedup key, recency sentinel), evaluated on one shared set of splits and folds.

**Why a shared split source is the core design decision.** Today each pipeline builds its
own masks:
- `src/save_graph_splits.py` persists only a seed-42 random split and LCPO fold
  *definitions* (module, presentation, counts). No LCPO validation masks are stored.
- `src/run_matched_comparison.py` (LightGBM, MLP) calls `random_student_split()` per seed on
  its own enrollment frame and draws LCPO validation as a plain 10% of train students; it
  skips folds whose test set has a single class.
- `src/run_gnn_experiment.py` calls `random_student_split()` on the artifact enrollment
  table and draws LCPO validation with `_sample_lcpo_val()` (at least 20 positives, fallback
  sampling).
So "same splits and folds" is not guaranteed by regenerating the fold CSV alone. The plan
persists every mask (random split per seed, LCPO train/val/test per fold, later
leave-module-out per module) once, keyed on the enrollment triple, and makes all models
read them.

**Findings from research that shape the plan**
- `GRAPH_EVALUATION_DIR` is `results/results/graph/evaluation/`. Split files exist in both
  that tree and the legacy `results/graph/evaluation/`, but both carry `enrollment_count`
  32,593 and the artifact metadata for **every** week (including week 2) also reports 32,593.
  That is the unfiltered population, so the graph artifacts themselves must be regenerated
  with `run_graph_pipeline.py` before splits are regenerated. Confirm by comparing against
  `studentInfo.csv` row count.
- After regeneration the enrollment count differs per week. Hard-coded 32,593 values in
  `tests/test_gnn_data_flow.py`, `src/gnn_model.py` (self-test, docstrings) and the notes in
  `AGENTS.md` / `.bob/rules-*/AGENTS.md` become wrong and must be revisited. The
  `save_graph_splits.py` note "identical across prediction weeks" also stops being true.
- `run_mlp_matched` names feature set A `MLP_behavioral_summaries`, but A is six summaries
  plus `num_of_prev_attempts`, `studied_credits` and age-band one-hots, i.e. the same input
  as the GNN base edge attributes. Every non-A feature set is currently named
  `MLP_all_features`, so B to E would also collide under one model name.
- `results/results_bk/` already exists (made before the earlier results consolidation). It is
  not a snapshot of the current CSVs, so a fresh pre-fix archive is needed before anything
  is overwritten. `_append_or_create_csv()` merges with `keep="last"`, so rerunning into
  existing CSVs would silently mix old and new rows.
- `src/update_results_matrix.py` reads GNN rows with `condition == ""` for row 18, but
  `run_gnn_experiment.py` now writes `condition="full"`; its LCPO branch (row 18) averages
  all conditions and does not separate loss weighting. The matrix reader must be rebuilt
  around `condition` and `loss_weighting`. `GraphSAGE + six summaries` is read from
  `ablation_results.csv` today, but `run_gnn_experiment.py --condition
  with_enrollment_summaries` writes to `random_student_results.csv` / `lcpo_results.csv`.
- LCPO in LightGBM/MLP uses one fixed model seed (`seed=-1` in the CSV, `random_state=42`);
  GNN LCPO uses 5 model-init seeds per fold. See Assumptions.
- Course-information building blocks already exist: course-design columns
  (`n_assessments`, `total_assessment_weight`, `prop_cma`, `prop_tma`, `prop_exam`,
  `n_vle_resources`) are in `GraphDataLoader.load()` in `src/gnn_model.py`, but they are
  concatenated with the course ID one-hots in every condition. Tabular feature sets C/D/E
  add course identity but not design features. No leave-module-out code exists.
- Zenodo: student node features and `enrolled_in` edge attributes are constant 1.0 and the
  built `interacted_with` attributes are not passed to the model
  (`src/run_zenodo_pipeline.py` around lines 458 to 510).

**Assumptions (1 and 2 confirmed by the user; 3 still open)**
1. LightGBM and MLP get 5 model seeds per LCPO fold, mirroring GNN `--model-seeds`, so the
   "5 seeds" requirement holds for LCPO as well as random-student.
2. Reported SD: random-student = across the 5 split seeds; LCPO = across the 22 folds (after
   averaging model seeds within a fold), plus a module-level aggregate (7 modules) as the
   honest-variance companion, because folds share students and course design.
3. (Confirmed by the user) GNN LCPO is run with both weighted and unweighted loss, like the
   random-student GNN, and kept in separate matrix rows. This requires code changes, see
   Sub-Task 2A. LightGBM and MLP have no class weighting in `run_matched_comparison.py`
   (no `class_weight`; `MLPClassifier` has no such option), so they are unweighted by
   construction and have no weighted counterpart; the matrix labels this explicitly.

## Sub-Tasks (process one at a time, in order)

### Sub-Task 0 — Local environment check and run runbook
**Status:** [x] done (2026-11-22). Results:
- `pytest tests/ -q`: 108 passed, 1 skipped (threshold-column skip in `tests/test_results_matrix.py`).
  `python src/check_data.py` passed.
- Environment saved to `results/results/run_environment.json` by `src/record_environment.py`
  (Apple M1 Max, 10 cores, 32 GB, torch 2.12.1 CPU, torch_geometric 2.8.0).
- Launch wrapper: `scripts/run_local_stage.sh <stage> <python script and args>` (nohup +
  caffeinate, log in `logs/local/`, gitignored).
- Timings (one run each): GNN random w8 87 s, w2 60 s; GNN LCPO fold w8 104 s, w2 74 s
  (marginal extra model seed about 62 s at w2); LightGBM w8 5 s; MLP w8 2.5 s.
- Projection for the Sub-Task 4 GNN grid (all four weeks timed for base `full`; 1 seed, fold 0, weighted):
  LCPO fold-seed run 62 / 83 / 91 / 101 s at weeks 2/4/6/8; random split 34 / 45 / 81 / 61 s.
  The 1,760 LCPO runs already cover BOTH conditions (880 per condition), so about 21 h per condition
  and about 42 h for the full LCPO grid (upper bound; a single timed run includes about 10 s of
  graph-loading overhead that later seeds do not pay). Random split: 80 runs, about 1.2 h. Tabular about 1 h.
  Total about 40-45 h of continuous run time. An earlier note of "60-100 h for two conditions" was wrong.
  `with_enrollment_summaries` ran faster at week 2 (49 s vs 62 s), so it is not expected to be slower.

**Intent.** Runs execute on this machine (macOS, Apple Silicon, CPU only) first. Establish
that the local environment is sound, measure real run times, and set up a resumable,
detached launch method. A server runbook is deferred (see Sub-Task 0B).

**What the repo shows about the local environment**
- Raw OULAD CSVs (including `studentVle.csv`) and `data/Zenodo/dataset.zip` +
  `course_info.json` are present under `data/`.
- Virtualenv `oulad_env/` is Python 3.11.11 with `torch 2.12.1` (CPU build, `cuda` is
  `None`), `lightgbm 4.6.0` and `torch_geometric`. `requirements.txt` pins
  `torch==2.5.1` and `torch-geometric==2.6.1`, while `REPRODUCIBILITY.md` mentions 2.12.1.
  Record the installed versions of torch, torch-geometric, scikit-learn, pandas, numpy and
  pyarrow in the results metadata so the local numbers can be compared with a later server
  run. Do not silently change pinned versions.
- No source file selects a device (`.to(`, `.cuda(` and `is_available` are absent), so all
  GNN training is already CPU-based and needs no code change to run locally.
- A full pre-fix run exists (`results/results/graph/*`, training curves for all LCPO folds),
  which is the only available evidence that this machine can finish the grid; run times
  were not recorded.

**Expected Outcomes**
- `pytest tests/ -v` passes locally, and `python src/check_data.py` confirms the raw data.
- Measured wall-clock for one GNN random-split run and one GNN LCPO fold-seed run
  (week 8, base condition) and an extrapolated total for the full grid in Sub-Task 4.
- A launch wrapper in `scripts/` (no SLURM directives) that runs one stage at a time
  detached (`nohup`, with `caffeinate -i` so macOS does not sleep), writes a log file per
  stage, and can be resumed. Result paths come from `src/config.py`, never hardcoded.
- A decision on stage order and checkpoints so partial progress is never lost (see
  Sub-Task 4).

**Todo List**
1. Run `pytest tests/ -v` and `python src/check_data.py` from the project root; fix only
   failures caused by this plan's changes.
2. Record installed package versions and the machine (CPU model, cores, RAM) into a small
   metadata file saved with the results.
3. After Sub-Task 2A exists, time one GNN random run and one LCPO fold with one seed, then
   multiply out the grid in Sub-Task 4. If the projection is impractical, report it to the
   user before starting; do not shrink the grid unilaterally.
4. Write the launch wrapper and log layout; use `--start-fold` and CSV idempotency for
   resuming.

**Relevant Context**
- `scripts/ls6_*.sh` (SLURM job scripts; the commands inside are the reference),
  `scripts/reproduce_all.sh` (still checks the legacy
  `results/graph/artifacts/week08_enrollments.parquet`).
- Do not use `MPS`; the code has no device handling and enabling it would change numerics.

---

### Sub-Task 0B — Server verification (deferred)
**Status:** [ ] pending (after local results exist)

**Intent.** The reviewer asked for the lab GPU server. Local CPU results are final unless a
server run differs, so the server is used to check them, not to produce them first.

**Expected Outcomes**
- Equivalent plain-shell commands for the stages in Sub-Task 4 on the server, and a
  comparison of one representative cell (for example week 8, seed 42, random split, GraphSAGE
  base weighted and unweighted, plus one LCPO fold) between local and server.
- A written statement of any difference larger than seed-to-seed spread; if it exceeds that,
  the affected rows are rerun on the server and replace the local rows.
- Only OULAD and Zenodo data on the server.

**Todo List**
1. Obtain from the user: access method, Python/CUDA environment, GPU count, and whether
   `data/raw/` and `data/Zenodo/dataset.zip` are already on the server. Do not invent
   server details; none exist in the repo.
2. Reproduce the environment per `requirements.txt` (torch and torch-geometric installed in
   two separate `pip install` calls) and run `python src/check_data.py` and `pytest`.
3. Run the comparison cell and report the difference.

---

### Sub-Task 1 — Archive pre-fix results and regenerate corrected artifacts and shared splits
**Status:** [x] done (2026-11-22). Results:
- Pre-fix snapshot: `results/results_bk/prefix_2026-11-22/{matched,graph}` plus the matrix
  workbook, with `MANIFEST.json` (row count and sha256 per CSV); `diff -rq` against the live
  tree was identical before anything was moved. The six to-be-rerun CSVs and the pre-fix
  `training_curves_*.npz` were then removed from the live tree (copies are in the snapshot).
  Zenodo CSVs and the `*_v1_buggy` / `*_old_single_seed` files were left in place.
- Artifacts regenerated for weeks 2/4/6/8. Enrollments: 28,119 / 27,538 / 27,024 / 26,522;
  removed already-unregistered: 4,474 / 5,055 / 5,569 / 6,071. At-risk rate falls from 52.8%
  to 45.3% (w2) and 42.0% (w8), because the removed rows were almost all Withdrawn.
- `src/save_graph_splits.py` now writes, per week under `GRAPH_EVALUATION_DIR`:
  `_random_split_seed{42,123,7,17,99}.parquet` (+ legacy `_random_split.parquet` = seed 42),
  `_lcpo_folds.csv` (adds `n_val`, `n_val_pos`), `_lcpo_masks.parquet` (one train/val/test
  column per fold, using the shared sampler), `_future_split.parquet`, `_splits_config.json`
  (corrected `enrollment_count`, seeds).
- `_sample_lcpo_val` moved to `src/oulad_data.py` as `sample_lcpo_val`; `run_gnn_experiment.py`
  imports it under the old name so `tests/test_validation_fallback.py` still works. The
  validation fallback never fired (min val positives 988).
- Verified: persisted masks equal what `random_student_split` / `lcpo_split` /
  `sample_lcpo_val` compute (weeks 2 and 8, seeds 42/123/99, all 22 folds); `verify_splits.py`
  passes all checks; `pytest`: 92 passed, 17 skipped (matrix tests wait for the new CSVs).
- Stale counts: `tests/test_gnn_data_flow.py` now reads `config` paths and the count from the
  artifact; `src/gnn_model.py` self-test and docstrings no longer hard-code 32,593;
  `AGENTS.md` and `.bob/rules-*/AGENTS.md` state 26,522 at week 8.
- NOT yet done (Sub-Task 2): the tabular, GNN-random and GNN-LCPO pipelines still compute
  their own masks; they do not read these files yet. `docs/*.md`, `README.md` and older plans
  still quote 32,593 / 52.8% (historical text, left unchanged).
- Legacy paths fixed (user approved): `.gitignore` lines 41-44 and `scripts/reproduce_all.sh`
  now use `results/results/graph/artifacts/`; the script's `save_graph_splits.py --weeks ...`
  (a flag that does not exist) is now plain `save_graph_splits.py`. Parquet files already
  tracked in git stay tracked (ignore rules do not untrack them).

**Intent.** Reviewer item 1 (splits and LCPO folds). Produce one authoritative set of
artifacts and masks for the corrected population, and keep pre-fix numbers out of the new
CSVs.

**Expected Outcomes**
- A dated pre-fix snapshot in `results/results_bk/` (extended or refreshed), containing the
  current `matched/`, `graph/` CSVs and the old splits. New results never write there.
- Result CSVs that will be rerun (`matched/*.csv`, `graph/random_student_results.csv`,
  `graph/lcpo_results.csv`, `graph/lcpo_summary.csv`, `graph/ablation_results.csv`) are
  moved out of `results/results/` so the new runs start from empty files. Zenodo CSVs are
  kept (step 4 reports them).
- Graph artifacts for weeks 2, 4, 6, 8 regenerated; per-week enrollment counts recorded
  (they should now be at or below 32,593 and differ by week, because
  `exclude_early_unregistered` drops enrollments whose `date_unregistration` is on or before
  the cutoff day; verify against the logs rather than assume).
- `week*_random_split.parquet` for every split seed (42 123 7 17 99), `week*_lcpo_folds.csv`
  and **persisted LCPO train/val/test masks per fold** (including the
  `_sample_lcpo_val` fallback outcome) written under `GRAPH_EVALUATION_DIR`
  (`results/results/graph/evaluation/weekNN/splits/`). The splits config JSON records the
  corrected `enrollment_count` and seeds.
- Legacy `results/graph/...` split copies are not read or written.

**Todo List**
1. Snapshot pre-fix `results/results/{matched,graph}` into `results/results_bk/` with a dated
   subfolder; verify file counts and a checksum or row count per CSV.
2. Move (not delete) the to-be-rerun CSVs out of the live tree.
3. Run `python src/run_graph_pipeline.py --week N` for N in 2, 4, 6, 8; check the log line
   "removed N already-unregistered enrollments" and `enrollment_count` in each
   `weekNN_metadata.json`.
4. Extend `src/save_graph_splits.py`: add `--seeds` (default 42 123 7 17 99), write one
   random split per seed (keyed by seed), and persist LCPO masks using the same
   `_sample_lcpo_val` logic as the GNN (move the sampler to `src/oulad_data.py` so both
   pipelines import one copy). Keep graph construction and splits decoupled (no masks in
   the graph artifacts).
5. Run `python src/save_graph_splits.py --week N` for all weeks; verify `verify_splits.py`.
6. Resolve stale counts: update `tests/test_gnn_data_flow.py` (hard-coded 32,593 and
   `results/graph/artifacts` path), `src/gnn_model.py` self-test asserts, `AGENTS.md` and
   `.bob/rules-*/AGENTS.md` notes. Fix `.gitignore` / `scripts/reproduce_all.sh` path
   references only if it is in scope of the user's approval.

**Relevant Context**
- `src/config.py` constants (`GRAPH_ARTIFACTS_DIR`, `GRAPH_EVALUATION_DIR`).
- `src/graph_pipeline.py` lines ~95 to 135 (`apply_window_cutoff` calls
  `exclude_early_unregistered`).
- `src/oulad_data.py`: `random_student_split`, `lcpo_split`, `exclude_early_unregistered`.
- Architecture rule: prediction unit is the enrollment triple; split functions return
  boolean Series aligned to the enrollment index.

---

### Sub-Task 2 — Make every model read the shared splits
**Status:** [x] done (2026-11-22). Results:
- New `src/split_loader.py` is the only read path: `load_random_split(week, seed, frame)`,
  `load_lcpo_folds(week)`, `load_lcpo_masks(week, fold_idx, frame)`. They return boolean
  Series and align to `frame` by the enrollment triple (raise on a different population,
  duplicate triples or missing files, with the regeneration command in the message).
- `run_matched_comparison.py` (LightGBM and MLP), `run_gnn_experiment.py` (random and LCPO)
  and `gnn_model.load_split_masks` (now takes `seed` and `fold_idx`, handles the new LCPO
  format) all read through it. None of them calls `random_student_split`, `lcpo_split`,
  `sample_lcpo_val` or the old inline 10% draw any more; only `save_graph_splits.py` does.
- Tabular LCPO now loops over `--model-seeds` (default 42 123 7 17 99); random rows carry
  `model_seed` = split seed. LCPO rows keep `seed = -1` (folds are fixed) and put the init
  seed in the new `model_seed` column. Matched dedup key is now
  `model, week, split_type, seed, model_seed, fold, feature_set`.
- `PIPELINE_VERSION = "v3_post_oct16_fixes"` in `src/config.py`; written to every matched and
  GNN row (replaces the hard-coded `v2_corrected`). Archived pre-fix rows keep their old tag.
- Verified: tabular feature table has the same enrollment set, order and labels as the graph
  artifact (weeks 2 and 8). New `tests/test_shared_splits.py` (16 tests) checks that the
  tabular order and GNN order give identical train/val/test key sets for all 5 split seeds and
  all 22 LCPO folds, plus a source check that no pipeline generates masks. `pytest`: 108
  passed, 17 skipped.
- Smoke runs wrote to temp dirs only (LightGBM, MLP, GNN random 3 epochs, GNN LCPO fold 0
  3 epochs); nothing was written to `results/`.
- Behaviour notes: LightGBM LCPO gives identical results across model seeds (it is
  deterministic without subsampling, so the 5 model seeds add no variance for LightGBM; the
  MLP does vary). The GNN `lcpo_results.csv` `n_train` counts train excluding validation,
  while `_lcpo_folds.csv` `n_train` includes validation.
- Single-class test folds: both paths skip; none of the 22 folds is single-class (min test
  at-risk rate 0.24).
- Not changed (Sub-Task 2A): GNN LCPO dedup keys, loss weighting, and the training-curve
  filenames. `tests/test_lcpo_condition_key.py` still encodes the old keys.

**Intent.** Reviewer item 1: "All models must use these same splits and folds."

**Expected Outcomes**
- One loader function (in `src/oulad_data.py` or a thin helper next to
  `save_graph_splits.py`) that returns boolean masks aligned to the enrollment table for
  (week, split type, seed or fold). LightGBM, MLP, GNN random and GNN LCPO all call it.
- Tabular pipeline rows align to the graph enrollment table by the enrollment triple
  (not by position), so the same enrollment is in the same partition for every model.
- A test that, for a given week/seed/fold, asserts identical train/val/test enrollment key
  sets across the tabular and GNN code paths, and that none of the pipelines recomputes
  masks on its own.
- Folds with single-class test sets are handled identically (both skip, or both report).

**Todo List**
1. Add the shared loader; keep `random_student_split` / `lcpo_split` as the generators used
   only by `save_graph_splits.py`.
2. `run_matched_comparison.py`: replace per-seed `random_student_split` and the inline LCPO
   10% val draw with the loader; add key-based alignment between
   `build_tabular_features()` output and the artifact enrollment table.
3. `run_gnn_experiment.py`: replace the in-function `_random_student_split` call and
   `_sample_lcpo_val` use with the loader (`load_split_masks` in `gnn_model.py` currently
   handles only seed 42 and an incompatible LCPO format; update it).
4. Add model-seed loop (42 123 7 17 99) to LightGBM/MLP LCPO and write the seed to the CSV
   (currently `seed=-1`); extend dedup keys accordingly (`_append_or_create_csv` in
   `run_matched_comparison.py`).
5. Add a new `pipeline_version` tag (for example `v3_post_oct16_fixes`). The GNN runner
   hard-codes `v2_corrected` (`run_gnn_experiment.py` lines ~430 and ~616) and the matched
   runner writes no version column; add the tag to both so post-fix rows are distinguishable
   from the pre-fix rows in the archive. Add the tag as a dedup-key member where rows could
   otherwise collide.
6. Add tests (new file in `tests/`), run the full suite.

**Relevant Context**
- `src/run_matched_comparison.py` lines ~417 to 545 and ~559 to 745.
- `src/run_gnn_experiment.py` lines ~245 to 300, 340 to 365, 476 to 535.
- `src/gnn_model.py` `load_split_masks` (lines ~664 to 702).
- Existing tests to extend: `tests/test_splits.py`, `tests/test_validation_fallback.py`,
  `tests/test_label_alignment.py`, `tests/test_lcpo_condition_key.py`.

---

### Sub-Task 2A — Unweighted LCPO support in the GNN runner
**Status:** [x] done (2026-11-22). Results:
- `run_lcpo_experiment(..., weighted=True)`: `weighted=False` passes `pos_weight=False`
  (plain BCE) and writes `loss_weighting="unweighted"`. Both read the same persisted masks.
- CLI: `--lcpo-loss-weighting {weighted,unweighted,both}`, default `weighted` (unchanged
  behaviour). The rerun commands use `both`.
- Module constants `LCPO_RESULT_KEYS` (`week, fold_idx, model_seed, condition,
  loss_weighting`) and `LCPO_SUMMARY_KEYS` (`week, fold_idx, condition, loss_weighting`) are
  used at all three write sites.
- Training curves: `training_curves_lcpo_week{WW}_{condition}_{weighting}_fold{FF}_seed{S}.npz`
  (the old names had no week, condition or weighting and would have collided).
- Readers: new `src/gnn_results.py::select_reference_gnn_rows` (condition `full`, weighted)
  is used by `summarize_lcpo_modules.py`, `course_diagnostics.py` and `compare_gnn_lgbm.py`
  and prints which variant it used. `course_variation.py` only reads `n_test` and needed no
  change. `update_results_matrix.py` is handled in Sub-Task 5.
- Also fixed: the per-fold print averaged over every record in the list, now only the
  current fold.
- Tests: `tests/test_lcpo_condition_key.py` uses the runner's key constants and has 3 new
  weighting tests; new `tests/test_gnn_reference_rows.py`. `pytest`: 114 passed, 17 skipped.
- Smoke test (`--quick`, week 8, 2 folds, seeds 42 and 123, `--lcpo-loss-weighting both`):
  8 rows, weighted and unweighted coexist per fold/seed, summary has 4 rows, 8 curve files.
  Note: that run accidentally wrote into the live `results/results/graph/` (my monkeypatch
  of `RESULTS_DIR` was lost when the module was re-executed); I deleted those files and
  confirmed the live tree is empty again and the archive is untouched.

**Intent.** Add unweighted-loss LCPO runs so weighted and unweighted results are separate in
the LCPO columns as well. The current LCPO path cannot do this safely.

**What the code does today (`src/run_gnn_experiment.py`)**
- `run_lcpo_experiment()` has no `weighted` argument. It always calls
  `compute_pos_weight(None, train_y)` and passes it to `run_training_loop` (lines ~555 to
  574). The random-split path already supports an unweighted run by passing
  `pos_weight=False`, which `run_training_loop` treats as plain `BCEWithLogitsLoss`.
- The record hard-codes `"loss_weighting": "weighted"` (line ~610).
- `lcpo_results.csv` is deduplicated on `["week", "fold_idx", "model_seed", "condition"]`
  (lines ~640 and ~837), with no `loss_weighting`. An unweighted row would silently replace
  the weighted row for the same fold/seed/condition. `lcpo_summary.csv` already includes
  `loss_weighting` in its key (line ~667). `tests/test_lcpo_condition_key.py` encodes the
  old four-column key.
- Training-curve files are named `training_curves_lcpo_fold{idx}_seed{mseed}.npz` (line
  ~580) with no weighting in the name, so the unweighted run would overwrite the weighted
  curves.
- The CLI has no LCPO weighting option; only the random-student loop iterates over
  `(True, False)`.

**Expected Outcomes**
- `run_lcpo_experiment()` accepts a loss-weighting choice and writes the correct
  `loss_weighting` value in each row.
- `lcpo_results.csv` and `lcpo_summary.csv` both deduplicate on a key that includes
  `loss_weighting`; weighted and unweighted rows for the same fold/seed/condition coexist.
- Training-curve files include the weighting, and `condition` if conditions can collide.
- A CLI flag selects `weighted`, `unweighted` or `both` for LCPO. Default stays `weighted`
  (existing behavior); the rerun commands pass `both`.
- Weighted and unweighted runs read the same persisted train/val/test masks from Sub-Task
  2, so the only difference between them is the loss. The validation sampling in
  `_sample_lcpo_val` does not depend on the weighting.
- Every other consumer of `lcpo_results.csv` is checked so it does not average weighted and
  unweighted rows together.

**Todo List**
1. Add the loss-weighting parameter to `run_lcpo_experiment()`; for `unweighted` skip
   `compute_pos_weight` and pass `pos_weight=False`; set the record's `loss_weighting`
   accordingly.
2. Add the CLI flag (for example `--lcpo-loss-weighting {weighted,unweighted,both}`) and loop
   over the chosen weightings in the LCPO branch of `__main__`.
3. Add `loss_weighting` to both dedup-key lists and to the training-curve filename.
4. Update `tests/test_lcpo_condition_key.py` (new key lists plus a test that weighted and
   unweighted rows for the same fold/seed/condition both survive).
5. Audit readers of `lcpo_results.csv` and `lcpo_summary.csv` and make each filter or group by
   `loss_weighting` (and `condition`): `src/summarize_lcpo_modules.py` (reads the file with
   no filter), `src/course_diagnostics.py`, `src/course_variation.py`,
   `src/compare_gnn_lgbm.py`, `src/update_results_matrix.py` (handled in Sub-Task 5). Decide
   per reader which weighting is the reference (weighted, to match prior reports) and state
   it in the output.
6. Smoke test with `--quick` (first 2 folds, one week) for both weightings and check the CSV
   rows and the curve files.

**Relevant Context**
- `src/run_gnn_experiment.py` lines ~440 to 670, `__main__` block ~700 to 840.
- `src/gnn_model.py` `compute_pos_weight` and `run_training_loop` (`pos_weight=None` computes
  a weight; `pos_weight=False` gives plain BCE).
- Early stopping uses validation AUROC and the threshold is tuned on validation F1 in both
  modes, so no other change is needed for an unweighted run.

---

### Sub-Task 3 — Six-summaries-only MLP as its own feature set
**Status:** [x] done (2026-11-22). Results:
- New feature set `S` in `src/run_matched_comparison.py`: `build_tabular_features(...,
  behavioral_only=True)` keeps exactly `vle_total, vle_mean, vle_std, assess_mean,
  assess_max, assess_count` (`_SIX_SUMMARY_COLS`); no `num_of_prev_attempts`,
  `studied_credits` or age band. Combining it with the `add_*` options raises. Set A is
  unchanged (11 columns at week 8; S has 6), so existing A tests still pin A's flags.
- Explicit label mapping `_MLP_MODEL_NAMES`: A -> `MLP_all_features`, S ->
  `MLP_six_summaries`, B..E -> `MLP_feature_set_B..E`. The old ternary that called set A
  `MLP_behavioral_summaries` is gone. `--feature-set` accepts `S`. LightGBM keeps
  `LightGBM_matched`.
- `src/update_results_matrix.py`: row 17 now reads `MLP_six_summaries` (row 16 stays
  `MLP_all_features`); `MODEL_TO_ROW` updated. `tests/test_results_matrix.py` has no MLP
  label references, so it needed no change.
- Tests: 7 new set-S tests in `tests/test_feature_set_matched_comparison.py` (exact six
  columns, no metadata, no NaNs, aligned lengths, same rows and labels as A, rejects `add_*`
  combos, flags entry); `tests/test_loss_weighting_groupby.py` no longer re-implements the
  old ternary and now tests the real mapping. `pytest`: 122 passed, 17 skipped.
- Real-data check (week 8, temp dir only; live `matched/` still empty): both sets run end to
  end. 2-seed random AUROC A 0.784 vs S 0.773; first two LCPO folds (1 model seed) A 0.759
  vs S 0.729. Not final numbers, just a sanity check that S is a weaker, distinct control.
- Matrix handling for the old name: the pre-fix `mlp_matched_results.csv` (archived) used
  `MLP_behavioral_summaries` for set A. Nothing in the live tree uses that name now.

**Intent.** Reviewer item 1: the control must use only the six behavioral summaries
(`vle_total`, `vle_mean`, `vle_std`, `assess_mean`, `assess_max`, `assess_count`).

**Expected Outcomes**
- A new feature-set key (for example `S`) in `_FEATURE_SET_FLAGS` / `build_tabular_features`
  that excludes `num_of_prev_attempts`, `studied_credits` and age-band dummies.
- Explicit mapping from feature set to model label: A -> `MLP_all_features`,
  S -> `MLP_six_summaries`; B to E keep distinct labels (for example
  `MLP_feature_set_B`) so they never collide in the matrix.
- `--feature-set` CLI choices and docstring updated; unit tests for the new set.

**Todo List**
1. Add set S to `_FEATURE_SET_FLAGS` and a flag (or explicit column list) that drops the
   three non-behavioral base columns.
2. Replace the `mlp_model_name` ternary in `run_mlp_matched` with the mapping.
3. Extend `tests/test_feature_set_matched_comparison.py` (columns are exactly the six; no age
   band; no NaNs; aligned lengths).
4. Update `update_results_matrix.py` row labels (row 16 `MLP_all_features`, row 17
   `MLP_six_summaries`) and `tests/test_results_matrix.py`.

**Relevant Context**
- `src/run_matched_comparison.py` lines ~15 to 25, 79 to 88, 134 to 141, 581.
- Existing LightGBM model name is `LightGBM_matched`; keep unchanged.

---

### Sub-Task 4 — Rerun the matched comparison (step 2)
**Status:** [ ] pending (depends on Sub-Tasks 0 to 3)

**Intent.** Produce new results for five models on the corrected population, all weeks, 5
seeds, random-student and LCPO.

**Expected Outcomes**
- New CSVs under `results/results/` containing only post-fix rows:
  - LightGBM (set A), MLP all features (set A), MLP six summaries (set S).
  - GraphSAGE base (`--condition full`) and GraphSAGE + six summaries
    (`--condition with_enrollment_summaries`).
- Weeks 2/4/6/8, split seeds 42 123 7 17 99, model seeds 42 123 7 17 99 for LCPO, 22 folds.
- GNN rows exist for both `weighted` and `unweighted` loss in random-student **and** LCPO
  (LCPO unweighted from Sub-Task 2A). Tabular models are unweighted by construction.
- Every new row carries the new `pipeline_version` tag from Sub-Task 2, and no row with an
  older tag is present in any live result CSV.
- Expected grid (check each, per week 2/4/6/8):
  - Random-student, LightGBM / MLP (set A) / MLP (set S): 5 split seeds = 5 rows per week.
  - Random-student, GraphSAGE base and + six summaries: 5 split seeds x 2 loss weightings =
    10 rows per week per condition.
  - LCPO, LightGBM / MLP (set A) / MLP (set S): 22 folds x 5 model seeds = 110 rows per
    week.
  - LCPO, GraphSAGE base and + six summaries: 22 folds x 5 model seeds x 2 loss weightings =
    220 rows per week per condition.
  - Rows for folds skipped because the test set has a single class are listed explicitly, not
    silently missing. Confirm that the fold count is still 22 on the corrected population.
- Completeness check: row counts per (model, week, split, seed/fold) match the expected grid;
  no run with `best_epoch` below 10 is silently kept (matrix loader already filters it;
  rerun those instead).

**Todo List**
1. Tabular on server: `python src/run_matched_comparison.py --models lgbm mlp --weeks 2 4 6 8
   --seeds 42 123 7 17 99 --splits random lcpo --feature-set A`, then the same for set S
   (MLP only).
2. GNN base: `python src/run_gnn_experiment.py --condition full --weeks 2 4 6 8 --seeds 42 123
   7 17 99 --model-seeds 42 123 7 17 99 --splits random lcpo --lcpo-loss-weighting both`
   (flag name from Sub-Task 2A; the random-split loop already runs both weightings).
3. GNN + six summaries: same with `--condition with_enrollment_summaries`.
4. Write a small verification script or test that counts rows per grid cell and flags gaps.
5. Compare enrollment counts in the new CSVs/metadata with Sub-Task 1 to confirm the
   corrected population was used.

**Relevant Context**
- Results key rules: `_append_or_create_csv` dedup keys must include `condition`,
  `loss_weighting`, `feature_set`, and the model seed. For `lcpo_results.csv` this needs the
  Sub-Task 2A key change before any unweighted LCPO run is started.
- Compute: the GNN LCPO grid is 2 conditions x 4 weeks x 22 folds x 5 model seeds x 2
  weightings = 1,760 training runs (880 weighted plus 880 unweighted); the unweighted half
  is new. Add the 2 conditions x 4 weeks x 5 split seeds x 2 weightings = 80 random-split
  runs. Runs are resumable per fold with `--start-fold` and idempotent through the CSV
  keys. This is the local CPU grid; it must be timed in Sub-Task 0 before launch.
- Local execution order (each stage is launched, finished and verified before the next):
  1. tabular models (LightGBM, MLP A and S), all weeks, random and LCPO;
  2. GNN random split, base then `with_enrollment_summaries`, all weeks, both weightings;
  3. GNN LCPO per week, per condition, per weighting, in that order, so a week is complete
     before the next starts.
  After each stage run the completeness check from this sub-task; the CSV is only trusted
  once it passes.
- Local run safety: keep the machine awake (`caffeinate -i`), keep one GNN process at a
  time (they share all CPU cores), and keep per-stage logs. Results written to
  `results/results/` are final candidates; do not edit them by hand.
- LCPO validation: at least 20 positives, patience 50 (`--lcpo-patience`).

---

### Sub-Task 5 — Update the matrix with new numbers only (step 2, continued)
**Status:** [ ] pending

**Intent.** The matrix (`AUROC_Results_Matrix_by_Dr_Guo.xlsx`) must reflect only post-fix
CSVs, with mean and SD per model and weighted/unweighted kept separate.

**Expected Outcomes**
- `src/update_results_matrix.py` reads GNN rows by `condition` and `loss_weighting`
  (base weighted row 18, base unweighted row 19, `with_enrollment_summaries` row 21) and
  matched rows by `feature_set`/model label (rows 12, 16, 17).
- Random-student cell: mean ± SD across the 5 split seeds. LCPO cell: mean ± SD across the 22
  folds after averaging model seeds. A companion module-level table (7 modules) is
  produced for honest variance.
- Old values in the rows being rewritten are fully replaced (including cells for which no
  new data exists yet are cleared or marked pending, not left stale). Protected historical
  rows 1 to 11 remain untouched.
- "New numbers only" is enforced in code: the loader keeps only rows with the new
  `pipeline_version` tag and refuses to write if any other tag is present in the source
  CSV. The five rerun models (rows 12, 16, 17, 18, 19, 21) contain new numbers only.
- Rows not part of the rerun (20 and 22 to 34: interaction attrs, enrollment node,
  edge-aware, course design, temporal, course-conditioned, GCN, RGCN, HGT, ablations) keep
  their existing values (decided by the user), but each is visibly labeled "pre-fix, not
  rerun" (row label in column A plus a distinct cell style or note), and
  `update_results_matrix.py` no longer rewrites them from the pre-fix `ablation_results.csv`
  (`CONDITION_TO_MODEL`). The report states that these rows are not comparable to the new rows.
- Weighted and unweighted GNN loss runs have separate rows (18 and 19) and are never
  averaged together, in both the random-student and the LCPO columns. The LCPO branch is
  changed so it no longer averages all conditions into row 18; it filters by `condition` and
  `loss_weighting` (base weighted row 18, base unweighted row 19).
- Rows for LightGBM and MLP carry a note that they are unweighted by construction and have no
  weighted counterpart, so the weighted/unweighted split applies to the GNN rows only.
- Each cell reports mean ± SD with the aggregation unit stated: random-student across the 5
  split seeds; LCPO across 22 folds (model seeds averaged within a fold). The module-level
  (7 modules) companion table is written to a separate sheet or CSV, not into the main
  cells.
- `tests/test_results_matrix.py` updated for the new filters, including a test that an
  old-tag row in the input fails the run.

**Todo List**
1. Run `--dry-run` and review the planned cell updates against the new CSVs.
2. Rework the GNN branches (random and LCPO) to filter on `condition`/`loss_weighting`, using
   the fold-then-seed averaging for LCPO per weighting. Add `with_enrollment_summaries`
   (row 21) for both splits; add a test that weighted and unweighted LCPO rows land in
   different rows.
3. Add stale-cell handling for rows 12 to 19 and 21 and the version-tag guard. Stop reading
   `ablation_results.csv` for any row (row 21 now has a new-run source; the other
   ablation rows are frozen) and add the "pre-fix, not rerun" label to rows 20 and 22 to
   34 in all six sheets.
4. Write the matrix; back up the workbook first (`office_edit` creates a backup).
5. Spot-check three cells against a manual `groupby` of the CSVs.

**Relevant Context**
- `src/update_results_matrix.py` (`MODEL_TO_ROW`, `build_update_plan`).
- Workbook has six sheets (Results, AUPRC, F1, Balanced Accuracy, Precision, Recall).

---

### Sub-Task 6 — Course-information experiment (step 3; plan Sub-Tasks 7 to 10 of the Oct 16 review)
**Status:** [ ] pending (starts after Sub-Task 5 is accepted)

**Intent.** Test whether course-design features generalize to unseen courses where a course
ID cannot. Change one thing at a time and keep intermediate results.

| Course information | Random-student | LCPO | Leave-module-out |
|---|---|---|---|
| None | baseline | baseline | baseline |
| Course ID | | | |
| Course design | | | |

Models: LightGBM and GraphSAGE. Nine cells per model.

**Expected Outcomes**
- Leave-module-out masks persisted alongside the other shared splits (same loader).
- Three conditions implemented identically for tabular and GNN:
  `no_course` (no identity, no design), `course_id` (code_module and code_presentation
  one-hots only), `course_design` (design columns plus presentation length, no identity).
- Results keyed by `eval_setting` and `course_info` so no row overwrites another; each
  stage written to its own CSV and retained.
- Summary table AUROC ± SD per cell, with Wilcoxon paired tests for `course_design` vs
  `course_id` under LCPO and leave-module-out; module-level aggregation for LCPO and LMO
  variance.

**Todo List**
1. 6a. Add `leave_module_out_split()` to `src/oulad_data.py` and persist masks in
   `save_graph_splits.py`. Keep `enrolled_in`/`rev_enrolled_in` unfiltered and the same
   destination-node masking as LCPO. Test it.
2. 6b. `no_course` condition only: run LightGBM and GraphSAGE under the three evaluation
   settings. Check this baseline against Sub-Task 4 numbers (it should differ from set A
   only through the removed course columns; note GNN "full" currently contains both
   identity and design inputs).
3. 6c. Add `course_id` alone, rerun; then `course_design` alone, rerun. One change per
   stage; keep every intermediate CSV.
4. 6d. Tabular feature plumbing: LightGBM reads design columns from the
   `nodes_course_presentation` artifact; GNN adds a condition switch in
   `GraphDataLoader.load()` for the course-presentation node.
5. 6e. Summary and statistical tests; decide week scope (week 8 first, other weeks after
   review).
6. Add `--eval-setting {random,lcpo,lmo}` and `--course-info {none,id,design}` flags to
   `run_matched_comparison.py` and `run_gnn_experiment.py`; fold new keys into dedup keys.

**Relevant Context**
- `courseaware-review-plan.md` lines ~335 to 385 (Sub-Task 9 text and the 3x3 matrix).
- `src/gnn_model.py` lines ~460 to 470; `src/oulad_data.py` `lcpo_split`.
- Architecture rule: normalization from training rows only; course-presentation nodes stay
  visible when a course is held out.
- Open item: whether `module_presentation_length` counts as design (included in the Oct 16
  plan) and how presentation-year information is treated under `course_design`.

---

### Sub-Task 7 — Zenodo (step 4)
**Status:** [ ] pending (can be prepared in parallel with Sub-Task 6 code, run after it)

**Intent.** Report existing LightGBM results; give the GNN usable inputs before any tuning;
document class balance and graph statistics.

**Expected Outcomes**
- Existing Zenodo LightGBM results (`results/results/zenodo/comparison_results.csv`) are
  reported unchanged and added to the matrix (columns N, O).
- Zenodo GNN inputs are no longer constants: `enrolled_in` edge attributes carry the same
  VLE summaries LightGBM uses (`vle_total`, `vle_mean`, `vle_std` from
  `build_features_zenodo`), and the already-built `interacted_with` attributes are passed
  to the model. Student node features remain featureless by dataset limitation, documented.
- Class balance (fraction at-risk per split) and graph statistics (node and edge counts per
  type) written to a Zenodo metadata/summary file and a markdown table.
- No hyperparameter tuning in this step.

**Todo List**
1. Report existing LightGBM numbers (mean ± SD, random and LCPO) from the Zenodo CSVs.
2. Replace the constant-1.0 `enrolled_in` edge attributes with derived VLE summaries;
   normalize with training rows only.
3. Enable `interacted_with` edge attributes in the Zenodo GNN (expose as a flag, default
   on) and record which attributes the model receives.
4. Add class-balance and graph-statistics reporting after graph construction; save to
   `ZENODO_RESULTS_DIR` (never pool with OULAD results; follow `src/reporting_schema.py`).
5. Rerun Zenodo GNN with 5 split seeds and 5 model seeds on the lab server; separate CSV
   key or condition from the earlier constant-input run so old rows are not overwritten.
6. Extend the matrix update for Zenodo columns.

**Relevant Context**
- `src/run_zenodo_pipeline.py` lines ~440 to 520 (graph construction), ~526 to 740 (GNN
  runs), ~740 to 880 (LightGBM).
- `src/zenodo_data.py` header lists the schema gaps (no assessments, no demographics).

---

### Sub-Task 8 — IRB-26-319 revision (step 5)
**Status:** [ ] pending (blocked on user inputs)

**Intent.** Revise the IRB-26-319 draft per the reviewer's October 16 comments and send the
revised PDF for approval **before** submitting in Cayuse.

**Expected Outcomes**
- A revised draft and a PDF delivered to the reviewer. No Cayuse submission by the agent.
- Data-handling language states that only OULAD and Zenodo data reside on the GPU server
  until IRB and data agreements are in place.

**Todo List**
1. Obtain the current draft file and the written October 16 comments (neither is in the
   repository; a search of docs/, paper/ and cv/ found no IRB-26-319 file).
2. Map each comment to a revision and keep a comment-to-change list.
3. Produce the revised document and PDF; share for review.

**Relevant Context**
- Not a code task. Do not guess at the protocol content.

---

### Sub-Task 9 — Report from new result CSVs
**Status:** [ ] pending

**Intent.** The next report must be generated from the new result CSVs, not from the
earlier QA plan.

**Expected Outcomes**
- A report whose top section lists steps 1 to 5 as done / in progress / not started.
- Tables built only from post-fix CSVs: matched comparison (mean ± SD per model, weighted
  and unweighted separate), LCPO at fold and module level, per-week enrollment counts,
  course-information results (when available), Zenodo class balance and graph statistics.
- A reproducible script (not manual copy) that reads the CSVs through `src/config.py`
  constants.

**Todo List**
1. Build on `summarize_lcpo_modules.py` (module aggregation) and
   `summarize_graph_weeks.py` (graph and class statistics); check what
   `generate_report_figures.py` reads (`comparison_results.csv` and others) and whether
   those inputs are regenerated by the new runs, or replace them with direct CSV reads.
2. Add the status block (generated from a small status file or from the existence and
   completeness checks of Sub-Task 4).
3. Render to the same format as prior reports (HTML in repo root) after confirming with the
   user.

**Relevant Context**
- Prior reports: `graphsage-at-risk-prediction-progress-report.html`,
  `weekly-progress-report-graphsage-for-at-risk-prediction.html` (not to be used as sources).

---

## Dependency Order

Sub-Task 0 -> 1 -> 2 -> 2A -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 (parallel, blocked on inputs) -> 9,
with Sub-Task 0B (server verification) after Sub-Task 5 or any later stage whose results are
to be confirmed.
Sub-Task 3 can be done before 2 or 2A if desired, but all of them must precede Sub-Task 4.

## Validation Questions

Confirmed by the user:
1. LightGBM and MLP run 5 model seeds per LCPO fold.
2. LCPO SD is across 22 folds, with a module-level companion table.

Still open:
3. Provide the IRB-26-319 draft with the October 16 comments. Server access details are only
   needed later, for Sub-Task 0B.
4. Should the legacy `results/graph/` path references in `.gitignore`,
   `scripts/reproduce_all.sh` and `tests/test_gnn_data_flow.py` be fixed in Sub-Task 1, or
   left for a separate change?
5. (Answered) Unweighted GNN LCPO runs are added so weighted and unweighted are separate in
   the LCPO columns too. See Sub-Task 2A.
6. (Answered) Matrix rows 20 and 22 to 34 keep their old values, visibly marked
   "pre-fix, not rerun". See Sub-Task 5.
7. (Answered, with a flag) Execution is local CPU first, including the full GNN grid; results
   are final unless a server run differs. The reviewer asked for the lab GPU server for all
   runs. The status block of the next report should state that results were produced locally
   and that server verification (Sub-Task 0B) is pending or done. Decide whether to tell the
   reviewer before the runs start.
