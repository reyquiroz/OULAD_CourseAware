# Graph Pipeline Feedback Resolution Plan

## Overview

This plan addresses four issues raised in the supervisor feedback before the
graph pipeline can be considered finalized. The issues are independent of each
other and are ordered by risk: data privacy first, schema correctness second,
documentation accuracy third, and report completeness fourth.

**Scope**: correctness and documentation only — no new features, no architecture
changes, no model training.

---

## Sub-Task 1 — Remove unrelated data file and audit repository for PII

**Status**: `[ ] pending`

### Intent

`data/CompletionReport-2023.csv` is an IBM employee learning-platform export
that contains personally identifiable information (names, email addresses,
training records). It is unrelated to the OULAD project and must be purged from
the working tree and from all Git history. The repository must also be audited
for any other files that should not be tracked, and `.gitignore` must be
updated to prevent recurrence.

### Expected Outcomes

- `data/CompletionReport-2023.csv` is no longer present in the working tree.
- The file does not appear in any commit in `git log --all --full-history`.
- `.gitignore` contains an explicit rule blocking `data/CompletionReport*.csv`
  (or a broader pattern such as `data/*.csv` with exceptions for OULAD files).
- A brief audit note is added to `CONTRIBUTING.md` or `.gitignore` explaining
  what files are and are not allowed in `data/`.
- Any other non-OULAD, identifiable, or restricted files found during the audit
  are removed and blocked.

### Todo List

1. **Identify the file's full Git history** — run `git log --all --full-history -- data/CompletionReport-2023.csv` to confirm which commits contain it.
2. **Rewrite Git history** — use `git filter-repo --path data/CompletionReport-2023.csv --invert-paths` (preferred over BFG) to remove the file from all commits. If `git-filter-repo` is not installed, use the BFG Repo Cleaner as an alternative.
3. **Remove from working tree** — delete the file if `filter-repo` left it staged or unstaged.
4. **Audit the repository** — scan for any other non-OULAD CSV/Excel/text files in `data/`, `src/`, `notebooks/`, and the root directory; check `img_1782516990144.png` and any other image files for PII.
5. **Update `.gitignore`** — add a rule (e.g., `data/Completion*.csv` or `data/*Report*.csv`) to block similar files in the future. Add a comment explaining the OULAD-only policy for the `data/` directory.
6. **Force-push** the rewritten history to the remote (after confirming with Reynaldo that a force-push is acceptable and that collaborators have been notified).

### Relevant Context

- File confirmed at: `data/CompletionReport-2023.csv`
- Current `.gitignore` excludes `data/studentVle.csv` and large artifacts but
  does not mention CompletionReport files.
- Tool for history rewrite: `git filter-repo` (https://github.com/newren/git-filter-repo)

---

## Sub-Task 2 — Correct enrollment-feature representation on the student node

**Status**: `[ ] pending`

### Intent

In `src/graph_pipeline.py` lines 144–155, `student_cols` includes `age_band`,
`num_of_prev_attempts`, and `studied_credits`. When `drop_duplicates("id_student")`
is called at line 155, a student enrolled in multiple courses has only one
(arbitrarily chosen) enrollment's values stored on the student node — silently
discarding variation. `num_of_prev_attempts` and `studied_credits` are
correctly duplicated on the `enrolled_in` edge, but they should not appear on
the student node at all. `age_band` also appears on the student node but the
OULAD data dictionary notes it can differ across a student's course presentations
(a student may enroll in courses across multiple years); its placement must be
audited.

The fix is to:
1. Confirm empirically whether `age_band` varies per student in `studentInfo.csv`.
2. If it varies: move it to the `enrolled_in` edge as an enrollment-scoped attribute.
3. Remove `num_of_prev_attempts` and `studied_credits` from `student_cols` (they
   are already correctly on the edge and should not be on the node).
4. After correcting the schema, regenerate all four prediction-week artifacts and
   validation outputs.

### Expected Outcomes

- `student_cols` in `build_node_tables()` contains only stable student-level
  attributes: `id_student`, `gender`, `region`, `highest_education`, `imd_band`,
  `disability` (and `age_band` only if confirmed invariant per student).
- `num_of_prev_attempts` and `studied_credits` are present only on the
  `enrolled_in` edge, not on the student node.
- If `age_band` is enrollment-scoped, it appears on the `enrolled_in` edge and
  not on the student node.
- `docs/GRAPH_SCHEMA.md` student-node table and any enrolled_in table are
  updated to reflect the final column set.
- All four week artifacts (`week02`, `week04`, `week06`, `week08`) and
  corresponding validation outputs are regenerated and committed (metadata JSON)
  or re-saved (Parquet — gitignored).

### Todo List

1. **Audit `age_band` variation** — write a short script (or add a cell to the
   analysis notebook) that loads `studentInfo.csv` and counts how many
   `id_student` values have more than one distinct `age_band` across their rows.
   Record the result as a code comment in `graph_pipeline.py`.
2. **Remove `num_of_prev_attempts` and `studied_credits` from `student_cols`**
   in `build_node_tables()` (`src/graph_pipeline.py` lines 151–152). These
   values are already stored on the `enrolled_in` edge; having them on the
   student node is both redundant and incorrect for multi-course students.
3. **Decide `age_band` placement** based on the audit result:
   - If invariant per student: keep on student node, add a code comment
     confirming the audit result.
   - If it varies: remove from `student_cols` and add to the `enrolled_in` edge
     construction block (lines ~277–291) alongside `num_of_prev_attempts` and
     `studied_credits`.
4. **Update `docs/GRAPH_SCHEMA.md`** — student-node table and enrolled_in table
   to reflect the corrected column set (remove or move attributes as
   appropriate). Add a note documenting the audit result for `age_band`.
5. **Regenerate artifacts for all four weeks** — run `python src/run_graph_pipeline.py --week 2`, `--week 4`, `--week 6`, `--week 8`. Confirm the pipeline runs without errors.
6. **Commit updated metadata JSON files** for all four weeks
   (`results/graph/artifacts/week02_metadata.json`, etc.) and updated
   validation files under `results/graph/validation/`.

### Relevant Context

- `src/graph_pipeline.py` lines 144–158 (student node construction)
- `src/graph_pipeline.py` lines 277–291 (enrolled_in edge construction)
- `docs/GRAPH_SCHEMA.md` student-node and enrolled_in tables
- OULAD data dictionary: `age_band` is described as a categorization of the
  student's age at the time of enrollment — implying it could vary per
  enrollment if presentations span years.

---

## Sub-Task 3 — Reconcile documentation with current implementation and outputs

**Status**: `[ ] pending`

### Intent

Several documentation files contain stale figures that describe an older version
of the pipeline (before the dual-guard assessment filter was applied). The most
visible discrepancy is the Week 8 `submitted`-edge count: **47,259** appears in
`README.md` (line 169) and `docs/validation_report_week8.md` (line 32), while
the current pipeline produces **44,927** (confirmed in `docs/GRAPH_SCHEMA.md`
and `docs/graph_validation_summary.md`). All graph counts, runtime/memory
figures, feature definitions, and course-difficulty result descriptions must be
verified against the saved artifacts and corrected.

### Expected Outcomes

- `README.md` edge-count table shows `submitted | 44,927`.
- `docs/validation_report_week8.md` shows `submitted | 44,927` in the edge
  counts table.
- All per-week edge/node counts in `docs/graph_validation_summary.md` match
  the current metadata JSON files for weeks 2, 4, 6, and 8.
- Runtime and memory figures in `docs/validation_report_week8.md` and
  `docs/graph_validation_summary.md` match the current pipeline output.
- `README.md` student-node feature list reflects the corrected schema from
  Sub-Task 2 (e.g., removing `num_of_prev_attempts` if it was moved).
- Any course-difficulty result descriptions (e.g., in
  `docs/CROSS_COURSE_EVALUATION_REPORT.md`) are verified against saved result
  files and updated if stale.
- A note is added to `docs/GRAPH_SCHEMA.md` (or a comment in
  `src/summarize_graph_weeks.py`) explaining that the multi-week summary
  document is generated from metadata JSON files, so it should be regenerated
  after any pipeline run rather than edited by hand.

### Todo List

1. **Fix `README.md` line 169** — change `submitted | 47,259` to
   `submitted | 44,927`.
2. **Fix `docs/validation_report_week8.md` line 32** — change
   `submitted | 47,259` to `submitted | 44,927`.
3. **Verify all other figures in `docs/validation_report_week8.md`** against
   `results/graph/validation/week08_validation.json` and
   `results/graph/artifacts/week08_metadata.json`; update runtime and memory
   if they differ.
4. **Verify `docs/graph_validation_summary.md`** against the four metadata
   JSON files for weeks 2, 4, 6, and 8; update any stale rows.
5. **Check `README.md` node feature list** (line 157) — ensure it reflects the
   schema after Sub-Task 2 changes (e.g., if `age_band` moved to the edge).
6. **Audit `docs/CROSS_COURSE_EVALUATION_REPORT.md`** — check at-risk rates and
   per-course difficulty figures against saved result files; update any
   descriptions that reference older numbers.
7. **Add a regeneration reminder** to `docs/graph_validation_summary.md`
   (or its source script `src/summarize_graph_weeks.py`) so that future pipeline
   runs prompt an update of the summary document.

### Relevant Context

- Stale value confirmed: `README.md:169` and `docs/validation_report_week8.md:32`
  both show `47,259`
- Correct value confirmed: `docs/GRAPH_SCHEMA.md` (line 299) and
  `docs/graph_validation_summary.md` (line 34) both show `44,927`
- Runtime/memory reference values: `docs/graph_validation_summary.md` shows
  Week 8 at 5.2 s / 1,049.5 MB; `docs/validation_report_week8.md` shows 6.6 s
  / 1,048.7 MB — one of these needs to be verified against a current run

---

## Sub-Task 4 — Add quantitative evidence to the progress report

**Status**: `[ ] pending`

### Intent

The written progress report (Word document, external to the repository)
understates the work already done in the codebase. The supervisor expects
concise quantitative tables and measurement-backed statements. This sub-task
is about preparing the content (tables and specific figures drawn from saved
artifacts and result files in the repository) so they can be pasted or
referenced in the report. No speculation — every number must come from a saved
file.

### Expected Outcomes

- A new markdown file `docs/progress_report_tables.md` contains ready-to-use
  tables covering:
  1. **Strategy A vs. B comparison** — submitted-edge counts before and after
     the dual-guard filter, by week (drawn from metadata JSON files or prior
     pipeline runs).
  2. **Graph statistics by prediction week** — node counts, edge counts,
     enrollment count, at-risk rate (drawn from `docs/graph_validation_summary.md`
     and metadata JSON files).
  3. **Missing values before and after imputation** — pre-imputation null counts
     by column and week (drawn from `pre_imputation_nulls` in metadata JSON files).
  4. **Temporal-validation results** — AUROC by week (Week 2: 0.714, Week 4:
     0.781, Week 6: 0.812, Week 8: 0.835) with source file references.
  5. **Test results** — count and names of passing tests from `tests/`; the
     statement "stable across folds" replaced by the actual fold-level AUROC
     values from the LCPO evaluation (AUROC 0.804 ± 0.087) with the full
     per-fold breakdown if available in saved result files.
- All statements of the form "stable across folds" or similar vague phrases are
  replaced by specific measurements with the result-file path cited.

### Todo List

1. **Extract Strategy A vs. B submitted-edge counts** — if a Strategy A result
   file exists (e.g., from an earlier pipeline run before the dual-guard fix),
   record the counts; if not, note the pre-filter count from `validation_report_week8.md`
   (47,259) and the post-filter count (44,927) with an explanation.
2. **Build graph-statistics table** — read `docs/graph_validation_summary.md`
   and the four `week{N}_metadata.json` files; produce a clean Markdown table
   with one row per prediction week.
3. **Build missing-value table** — read `pre_imputation_nulls` from each
   `week{N}_metadata.json` and produce a table showing columns with nulls, their
   counts, and imputed values.
4. **Build temporal-validation table** — read the saved evaluation result files
   under `results/` to confirm the Week 2/4/6/8 AUROC figures; cite the source
   file path next to each number.
5. **Extract fold-level LCPO results** — find the per-fold LCPO AUROC values in
   saved result files under `results/`; replace the vague "stable across folds"
   description with the actual fold-level measurements (e.g., a table of
   per-presentation AUROC values contributing to the 0.804 ± 0.087 aggregate).
6. **Write `docs/progress_report_tables.md`** — compile all the above into a
   single, clearly labelled Markdown file that can be used as a source for the
   Word report.

### Relevant Context

- LCPO aggregate result already in `README.md`: AUROC 0.804 ± 0.087
- Temporal progression already in `README.md`: Week 2–8 AUROC values
- LCPO per-fold results may be in `results/evaluation/` or similar — location
  to be confirmed during implementation
- Metadata JSON files: `results/graph/artifacts/week{02,04,06,08}_metadata.json`
  (week08 is committed; others may need the pipeline to run first per Sub-Task 2)
