# OULAD Heterogeneous Graph Schema

## Overview

This document is the authoritative reference for the enrollment-centric
heterogeneous graph built by `src/graph_pipeline.py`. Column names and types
match the saved Parquet artifacts exactly. Aspirational or derived features
that are not yet implemented are listed separately in the **Planned Extensions**
section at the end.

## Graph Type

**Heterogeneous directed multigraph** with 4 node types and 5 edge types.

The **supervised unit** is the *enrollment* `(id_student, code_module,
code_presentation)` — not the student node. This allows one student to have
different outcomes across courses without label ambiguity. Labels live in the
enrollment supervision table (`week{N}_enrollments.parquet`), not in any node
table.

## Temporal Filtering

Each graph is built for one of four prediction windows:

| Week | Window (days) | VLE cutoff | Assessment cutoff |
|------|--------------|------------|-------------------|
| 2 | 14 | `date ≤ 14` | `due_date ≤ 14` AND `date_submitted ≤ 14` |
| 4 | 28 | `date ≤ 28` | `due_date ≤ 28` AND `date_submitted ≤ 28` |
| 6 | 42 | `date ≤ 42` | `due_date ≤ 42` AND `date_submitted ≤ 42` |
| 8 | 56 | `date ≤ 56` | `due_date ≤ 56` AND `date_submitted ≤ 56` |

Assessment submissions use a **dual guard (Strategy B)**:
1. **Due-date guard** — assessment must have been due by the cutoff.
2. **Submission-date guard** — the score must have been submitted by the cutoff
   (28.8 % of OULAD submissions have `date_submitted > due_date`; Guard 2
   removes these late-submitted scores from earlier windows).

VLE interactions are filtered on the interaction date only.

## Node Types

Column lists sourced from `week08_metadata.json` → `node_schema`.

---

### 1. `student`

**Source**: `studentInfo.csv`  
**Count** (Week 8): 28,785 unique students  
**Primary key**: `id_student`  
**File**: `week{N}_nodes_student.parquet`

| Column | Type | Source | Null handling |
|--------|------|--------|---------------|
| `id_student` | int | studentInfo | never null |
| `gender` | str | studentInfo | imputed → `"Unknown"` |
| `region` | str | studentInfo | imputed → `"Unknown"` |
| `highest_education` | str | studentInfo | imputed → `"Unknown"` |
| `imd_band` | str | studentInfo | **~971 nulls** (geocoding failures) → imputed to `"Unknown"` |
| `age_band` | str | studentInfo | imputed → `"Unknown"` |
| `disability` | str | studentInfo | imputed → `"Unknown"` |
| `node_idx` | int | pipeline | 0-based integer index used as src/dst in edge tables |

> **Note**: `num_of_prev_attempts` and `studied_credits` are enrollment-scoped
> (a student can have different values across courses) and are stored as attributes
> on the `enrolled_in` edge, not here.

**Target label**: stored in `week{N}_enrollments.parquet`, not in this table.

---

### 2. `course_presentation`

**Source**: `courses.csv`  
**Count**: 22 unique course-presentation combinations  
**Primary key**: composite `(code_module, code_presentation)`  
**File**: `week{N}_nodes_course_presentation.parquet`

| Column | Type | Source | Null handling |
|--------|------|--------|---------------|
| `code_module` | str | courses | never null |
| `code_presentation` | str | courses | never null |
| `module_presentation_length` | int | courses | never null |
| `node_idx` | int | pipeline | 0-based integer index |

---

### 3. `assessment`

**Source**: `assessments.csv` (filtered to `date ≤ window_days`)  
**Count** (Week 8): 40 assessments with due date ≤ 56 days  
**Primary key**: `id_assessment`  
**File**: `week{N}_nodes_assessment.parquet`

| Column | Type | Source | Null handling |
|--------|------|--------|---------------|
| `id_assessment` | int | assessments | never null |
| `code_module` | str | assessments | never null |
| `code_presentation` | str | assessments | never null |
| `assessment_type` | str | assessments | imputed → `"Unknown"` |
| `weight` | float | assessments | imputed → `0` |
| `date` | int | assessments | due date in days from course start; never null after filter |
| `node_idx` | int | pipeline | 0-based integer index |

> Assessment count varies by week: only assessments due **on or before** the
> prediction cutoff are included. Week 2 typically has 0–few assessments;
> Week 8 has 40.

---

### 4. `vle_resource`

**Source**: `vle.csv`  
**Count** (Week 8): 6,364 unique VLE resources  
**Primary key**: `id_site`  
**File**: `week{N}_nodes_vle_resource.parquet`

| Column | Type | Source | Null handling |
|--------|------|--------|---------------|
| `id_site` | int | vle | never null |
| `code_module` | str | vle | never null |
| `code_presentation` | str | vle | never null |
| `activity_type` | str | vle | imputed → `"Unknown"` |
| `week_from` | int | vle | **~5,243 nulls** (resources with no scheduled week) → imputed to `0` |
| `week_to` | int | vle | **~5,243 nulls** (same as above) → imputed to `0` |
| `node_idx` | int | pipeline | 0-based integer index |

> VLE resource nodes are not time-filtered — all resources in `vle.csv` are
> included regardless of week. Only *interactions* (the `interacted_with` edge)
> are filtered by the prediction cutoff.

---

## Edge Types

All edges use integer `src` and `dst` columns referencing `node_idx` values
from the corresponding node tables.

---

### 1. `enrolled_in` — student → course_presentation

**Source**: `studentInfo.csv`  
**Count**: 32,593 (one per unique enrollment)  
**File**: `week{N}_edges_enrolled_in.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `src` | int | Student `node_idx` |
| `dst` | int | Course-presentation `node_idx` |
| `num_of_prev_attempts` | int | Number of times this student previously attempted this course |
| `studied_credits` | int | Total credits the student was studying at enrollment time |

> `num_of_prev_attempts` and `studied_credits` are enrollment-scoped attributes.
> They vary per (student, course, presentation) and are therefore attached to
> this edge rather than the student node to preserve per-enrollment accuracy for
> multi-course students.

---

### 2. `contains_assess` — course_presentation → assessment

**Source**: `assessments.csv` (filtered)  
**Count**: 40 (Week 8) — matches the number of assessment nodes  
**File**: `week{N}_edges_contains_assess.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `src` | int | Course-presentation `node_idx` |
| `dst` | int | Assessment `node_idx` |

Structural edge only; no attributes.

---

### 3. `has_resource` — course_presentation → vle_resource

**Source**: `vle.csv`  
**Count**: 6,364 (Week 8) — matches the number of VLE resource nodes  
**File**: `week{N}_edges_has_resource.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `src` | int | Course-presentation `node_idx` |
| `dst` | int | VLE resource `node_idx` |

Structural edge only; no attributes.

---

### 4. `submitted` — student → assessment

**Source**: `studentAssessment.csv` (filtered: Guard 1 + Guard 2)  
**Count**: 44,927 (Week 8)  
**File**: `week{N}_edges_submitted.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `src` | int | Student `node_idx` |
| `dst` | int | Assessment `node_idx` |
| `score` | float | Assessment score 0–100; null raw scores filled to `0.0` |

**Enrollment-scoped**: edges are matched on `(id_student, code_module,
code_presentation)` to prevent activity from one course leaking into another.

**Temporal filtering**: both due-date guard (`date ≤ window_days`) and
submission-date guard (`date_submitted ≤ window_days`) are applied before
this edge is built. The `date_submitted` column is dropped from the saved
artifact; the max value is recorded in `week{N}_metadata.json` →
`max_date_submitted` for validation.

---

### 5. `interacted_with` — student → vle_resource

**Source**: `studentVle.csv` (filtered: `date ≤ window_days`)  
**Count**: 1,056,217 (Week 8, after aggregation)  
**File**: `week{N}_edges_interacted_with.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `src` | int | Student `node_idx` |
| `dst` | int | VLE resource `node_idx` |
| `total_clicks` | int | Sum of all clicks in this enrollment-resource pair |
| `n_interactions` | int | Number of interaction records aggregated |
| `first_day` | int | Earliest interaction day (days from course start) |
| `last_day` | int | Latest interaction day (≤ window_days) |
| `active_days` | int | Number of distinct days with at least one click |

**Aggregation**: raw `studentVle` rows are grouped by
`(id_student, id_site, code_module, code_presentation)` to produce one edge
per student-resource pair per enrollment. This prevents multi-edge explosion
(a student may interact with the same resource on many days).

**Enrollment-scoped**: the grouping key includes `code_module` and
`code_presentation` to prevent cross-course activity leakage.

---

## Enrollment Supervision Table

**File**: `week{N}_enrollments.parquet`  
**Count**: 32,593 (one row per enrollment, all weeks)  
**Note**: This table's content is identical across all 4 prediction weeks
because it derives from `studentInfo.csv` — labels do not depend on the
time cutoff.

| Column | Type | Description |
|--------|------|-------------|
| `id_student` | int | Student identifier |
| `code_module` | str | Course code |
| `code_presentation` | str | Presentation code |
| `final_result` | str | Raw outcome: Pass / Distinction / Fail / Withdrawn |
| `target` | int | Binary label: `1` = at-risk (Fail/Withdrawn), `0` = success (Pass/Distinction) |

**Label distribution** (Week 8): 17,208 at-risk (52.8%), 15,385 success (47.2%).  
Per-course at-risk rate ranges from 27.4 % (AAA/2013J) to 65.8 % (CCC/2014B).

---

## Null Imputation Strategy

Consistent with `oulad_data.build_features()`:

| Type | Rule |
|------|------|
| Numeric columns | Fill `NaN` → `0` |
| Categorical columns | Fill `NaN` → `"Unknown"` |

Applied in `build_node_tables()` after building each node DataFrame.
The pre-imputation null counts are recorded per column in
`week{N}_metadata.json` → `pre_imputation_nulls` and displayed in the
validation summary under "pre-imputation audit".

**Known expected nulls from raw OULAD source data** (not bugs):

| Column | Approx. count | Reason | Imputed to |
|--------|--------------|--------|------------|
| `student.imd_band` | 971 | Postcode could not be geocoded | `"Unknown"` |
| `vle_resource.week_from` | 5,243 | Resource has no scheduled week range in `vle.csv` | `0` |
| `vle_resource.week_to` | 5,243 | Same as above | `0` |

All other node and edge tables have zero pre-imputation nulls.

---

## Week 8 Graph Statistics (Reference)

| Metric | Value |
|--------|-------|
| Node types | 4 |
| Edge types | 5 |
| Students | 28,785 |
| Course-presentations | 22 |
| Assessments | 40 |
| VLE resources | 6,364 |
| enrolled_in edges | 32,593 |
| contains_assess edges | 40 |
| has_resource edges | 6,364 |
| submitted edges | 44,927 |
| interacted_with edges | 1,056,217 |
| Enrollments | 32,593 |
| At-risk rate | 52.8 % |
| Construction runtime | ~5 s |
| Peak memory | ~1,050 MB |

For per-week statistics across all 4 prediction windows, see
`docs/graph_validation_summary.md` and
`results/graph/validation/all_weeks_summary.csv`.

---

## Artifact File Naming Convention

All artifacts are prefixed `week{N:02d}_`, e.g. `week02_`, `week04_`, `week06_`,
`week08_`.

| Logical name | File |
|-------------|------|
| Student nodes | `week{N}_nodes_student.parquet` |
| Course-presentation nodes | `week{N}_nodes_course_presentation.parquet` |
| Assessment nodes | `week{N}_nodes_assessment.parquet` |
| VLE resource nodes | `week{N}_nodes_vle_resource.parquet` |
| enrolled_in edges | `week{N}_edges_enrolled_in.parquet` |
| contains_assess edges | `week{N}_edges_contains_assess.parquet` |
| has_resource edges | `week{N}_edges_has_resource.parquet` |
| submitted edges | `week{N}_edges_submitted.parquet` |
| interacted_with edges | `week{N}_edges_interacted_with.parquet` |
| Enrollment supervision | `week{N}_enrollments.parquet` |
| Metadata JSON | `week{N}_metadata.json` |

All Parquet files are gitignored and must be regenerated locally with:

```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
```

---

## Planned Extensions (Not Yet Implemented)

The following features are in scope for later iterations but are **not** present
in the current parquet files:

**Student node**: none planned — student features are intentionally minimal
(no aggregate statistics computed from activity data, to prevent leakage at
node construction time).

**Course-presentation node**: aggregate statistics such as historical pass rate,
average VLE clicks, number of enrolled students. These are computable from
`studentInfo.csv` and `studentVle.csv` without leakage, but are not yet added.

**Assessment node**: derived temporal features such as `week_due` (week number
of due date), `is_exam` (boolean), `date_normalized` (scaled to [0, 1]).

**VLE resource node**: aggregate statistics such as `total_clicks_all_students`,
`avg_clicks_per_student`. These would need to be computed using only data
within the prediction window to remain leakage-safe.

**enrolled_in edge**: registration dates (`date_registration`,
`date_unregistration`) from `studentRegistration.csv`. Note that
`date_unregistration` reveals withdrawal (part of the target) and must be
excluded or handled with care.
