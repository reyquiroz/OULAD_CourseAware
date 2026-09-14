# External Dataset Feasibility Report — Zenodo Record 17087849

**Prepared for**: OULAD GraphSAGE Dissertation Chapter  
**Date**: September 2025  
**Status**: Complete — based on direct inspection of data files in `data/Zenodo/`  
**Related**: Sub-Task 6 of `pipeline-correctness-and-external-validation-plan.md`  
**Zenodo URL**: https://zenodo.org/records/17087849  
**DOI**: 10.5281/zenodo.17087849

---

## 1. Dataset Identification

| Field | Value |
|-------|-------|
| Institution | KU Leuven (Tiukhova et al., 2024) |
| Platform | Blackboard LMS |
| Courses | Accountancy, Global economics |
| Academic years | 2018–19, 2019–20, 2020–21 |
| License | Pending confirmation from record page (CC BY assumed from Zenodo convention) |
| Re-identification risk | Low — all student identifiers are anonymised integer USER_IDs; no names, emails, or postcodes are present |
| Files available locally | `dataset.zip` (5 data files × 3 years = 15 files), `course_info.json`, `final_column_names.json`, `Feature engineering.ipynb` |

### File Listing (from `dataset.zip`)

| File | Description | Rows (all years combined) |
|------|-------------|--------------------------|
| `{yr}_course_participation.xlsx` | Enrollment outcomes — USER_ID, COURSE_ID, exam-session score categories, PASSED, PASSED_FIRST_ATTEMPT | 4,292 total (1,499 + 1,392 + 1,401) |
| `{yr}_log_activity.csv` | Clickstream — ACTION_ID, COURSE_ID, USER_ID, CONTENT_ID, SESSION_ID, TIMESTAMP | ~1.5M (1819), ~2.8M est. (1920), ~3.5M est. (2021) |
| `{yr}_course_content.xlsx` | Content item metadata — CONTENT_ID, PARENT_ID, CONTENT_TYPE, timestamps | 624 / 601 / 947 items per year |
| `{yr}_df_consumption.xlsx` | Discussion post reads — USER_ID, COURSE_ID, CONTEXT_ID, NUM_READ_POSTS | 4,383 / 3,664 / 4,030 per year |
| `{yr}_df_contribution.xlsx` | Discussion posts created — POST_ID, USER_ID, COURSE_ID, CONTEXT_ID, message length, reply counts | 1,278 / 871 / 1,586 per year |

### Companion Resources

- `course_info.json` — Per-(year, course) start date, finish date, exam weeks, and semester weeks. Provides the course-start reference date needed for day-offset computation.
- `final_column_names.json` — Mapping from internal feature engineering column names to human-readable labels (Tiukhova et al. 2024 naming convention).
- `Feature engineering.ipynb` — Reference notebook demonstrating how the authors compute engagement features (entropy/constancy of clicks, session regularity, active-day proportions, discussion metrics) from the raw files.

---

## 2. Criteria Assessment

### C1 — Early Prediction Windows

**Status: ✅ Satisfied**

`log_activity.csv` contains a `TIMESTAMP` column with full datetime precision (e.g., `2018-09-24 13:55:42`). `course_info.json` provides exact course-start dates per (year, course). Day offsets from course start are directly computable:

```
day_offset = (TIMESTAMP.date() − course_start).days
```

Verified empirically for Accountancy 1819 (start = 2018-09-24):
- Week 2 cutoff (day ≤ 14): 7,061 log rows
- Week 4 cutoff (day ≤ 28): 19,054 log rows  
- Week 8 cutoff (day ≤ 56): 50,585 log rows

All 6 course presentations have durations of 107–139 days (15–19 weeks), so all four OULAD prediction windows (14, 28, 42, 56 days) fall within the active semester period.

---

### C2 — Risk Label

**Status: ⚠️ Partially satisfied — proxy required; semantics differ from OULAD**

`course_participation.xlsx` contains two binary outcome fields:
- `PASSED` (1 = passed at any exam session, 0 = never passed) — **most analogous to OULAD's Fail/Withdrawn = at-risk**
- `PASSED_FIRST_ATTEMPT` (1 = passed in the first scheduled exam session)

`SCORE_CATEGORY_FINAL` values: `Fail`, `Pass`, `Excellent`, `Borderline`, `Able to Push` — no direct "Withdrawn" category.

**At-risk rate** using `PASSED == 0`:
- Overall: 33.9% (1,457 / 4,292 enrollments) — ✅ above 20% threshold
- Per course-presentation range: 24.5% (Global Econ 1920) to 48.7% (Global Econ 2021 part 2)

**Semantic gap**: OULAD distinguishes `Withdrawn` (disengagement before completion) from `Fail` (completed but failed). This dataset has no withdrawal label — students who stopped engaging but never formally withdrew appear as `PASSED == 0` at the final exam session. The at-risk definition is therefore "did not achieve a passing grade at any exam session attempt," which is a post-hoc outcome rather than a mid-course withdrawal signal. This semantic difference must be stated as a limitation in the dissertation.

**Recommendation**: Use `PASSED == 0` as the at-risk label (target = 1). This is the closest available analogue to OULAD's Fail+Withdrawn = at-risk.

---

### C3 — Course/Year Transfer (LCPO)

**Status: ⚠️ Partially satisfied — 7 course-presentation instances; fewer than OULAD's 22**

Course-presentation instances confirmed from direct inspection:

| Year | Course | N students | At-risk rate |
|------|--------|-----------|-------------|
| 1819 | Accountancy | 756 | 29.8% |
| 1819 | Global economics | 743 | 43.7% |
| 1920 | Accountancy | 711 | 32.2% |
| 1920 | Global economics | 681 | 24.5% |
| 2021 | Accountancy | 725 | 29.9% |
| 2021 | Global economics 1 | 341 | 38.4% |
| 2021 | Global economics 2 | 335 | 48.7% |
| **Total** | | **4,292** | **33.9%** |

**Assessment**: 7 instances supports LCPO evaluation (minimum 4 met), but with only 7 folds the LCPO variance estimate is less stable than OULAD's 22 folds. The two courses are from the same institution and likely have correlated VLE usage patterns, reducing cross-course transfer signal compared to OULAD's 7 structurally distinct modules. This must be acknowledged as a scope limitation.

---

### C4 — Graph Schema Compatibility

**Status: ⚠️ Partially satisfied — 3 of 5 OULAD edge types constructible; assessment submission edges unavailable**

#### Node Mapping

| OULAD node type | Dataset equivalent | Confidence | Gap |
|-----------------|-------------------|-----------|-----|
| `student` | `USER_ID` (anonymised integer) | ✅ High | No demographics (gender, age, IMD, disability, region) |
| `course_presentation` | `(year, COURSE_ID)` pair | ✅ High | No module-presentation-length field; derivable from `course_info.json` |
| `assessment` | **No direct equivalent** | ❌ None | No assessment submission table with per-student scores; `course_content` has `CONTENT_TYPE == 'Assessments'` items, but no score data is present |
| `vle_resource` | `CONTENT_ID` from `course_content.xlsx` | ✅ High | `CONTENT_TYPE` maps to OULAD `activity_type`; hierarchy via `PARENT_ID` |

#### Edge Mapping

| OULAD edge type | Dataset equivalent | Confidence | Gap |
|-----------------|-------------------|-----------|-----|
| `enrolled_in` (student → course_presentation) | Rows in `course_participation.xlsx` | ✅ High | No age_band, studied_credits, or num_of_prev_attempts |
| `interacted_with` (student → vle_resource) | `log_activity.csv` rows aggregated by (USER_ID, CONTENT_ID) per course-year | ✅ High | `total_clicks` = event count (no `sum_click` field, but action count is a direct proxy) |
| `has_resource` (course_presentation → vle_resource) | `course_content.xlsx` rows (one content item per course) | ✅ High | Direct structural edge; hierarchy via PARENT_ID |
| `submitted` (student → assessment) | **Not available** | ❌ None | No per-student assessment score table; `SCORE_CATEGORY_*` columns in participation are exam-session level aggregates (3–5 categories), not per-assessment submission records |
| `contains_assess` (course_presentation → assessment) | **Not available** | ❌ None | Cannot be built without assessment node table |

**Discussion forum edges** (additional, not in OULAD schema):
- `created_post` (student → discussion context): derivable from `df_contribution.xlsx`
- `read_posts` (student → discussion context): derivable from `df_consumption.xlsx`
These are novel edges not present in OULAD; they add signal but require new edge type handling in `gnn_model.py`.

---

### C5 — Tabular Schema Compatibility (LightGBM Features)

**Status: ⚠️ Partially satisfied — 3 of 9 OULAD features directly available; 3 constructible as proxies; 3 unavailable**

| OULAD feature | Dataset reconstruction | Feasibility |
|--------------|------------------------|------------|
| `vle_total` | Sum of log_activity rows per (USER_ID, year, COURSE_ID) within day window | ✅ Direct proxy (action count ≈ click count) |
| `vle_mean` | Mean action count per content item within window | ✅ Direct proxy |
| `vle_std` | Std of action counts per content item | ✅ Direct proxy |
| `assess_mean` | **Not available** — no per-student score table | ❌ Unavailable |
| `assess_max` | **Not available** | ❌ Unavailable |
| `assess_count` | **Not available** | ❌ Unavailable |
| `num_of_prev_attempts` | **Not available** — no historical enrollment data | ❌ Unavailable |
| `studied_credits` | **Not available** — no registration system data | ❌ Unavailable |
| `age_band` | **Not available** — USER_IDs are anonymised integers | ❌ Unavailable |

**Additional features unique to this dataset** (as per Tiukhova et al. 2024 and `final_column_names.json`):
- Constancy/entropy of clicks (daily/weekly)
- Session regularity, active-day proportion
- Discussion forum participation (posts created/read)

These can supplement the reduced OULAD feature set if used.

---

### C6 — Sample Size and Class Balance

**Status: ✅ Satisfied**

- Total enrollments: 4,292 across 7 course-presentations ✅ (> 1,000)
- At-risk rate (PASSED == 0): 33.9% ✅ (> 20%)
- Smallest single course-presentation: 335 students (Global Econ 2021 part 2) — borderline for per-fold LCPO stability but workable
- Total log events: ~7.8M rows across all three years (84 MB + 258 MB + 333 MB uncompressed)

---

## 3. Overall Assessment Summary

| Criterion | Status | Key finding |
|-----------|--------|------------|
| C1 — Early prediction windows | ✅ Satisfied | Second-level timestamps + `course_info.json` start dates enable exact day-offset computation |
| C2 — Risk label | ⚠️ Partially | `PASSED == 0` is the best proxy; no withdrawal category; post-hoc outcome rather than mid-course |
| C3 — Course/year transfer | ⚠️ Partially | 7 instances (2 courses × 3 years, one course split in 2021); fewer than OULAD's 22 |
| C4 — Graph schema | ⚠️ Partially | `submitted` and `contains_assess` edges cannot be built (no per-assessment score data) |
| C5 — Tabular schema | ⚠️ Partially | 3/9 OULAD features available; assessment and demographic features absent |
| C6 — Sample size & balance | ✅ Satisfied | 4,292 enrollments; 33.9% at-risk rate |

**Score**: 2 criteria fully satisfied, 4 partially satisfied, 0 not satisfied, 0 unknown.

---

## 4. Go / No-Go Verdict

**Verdict: ⚠️ CONDITIONAL GO — Feasible with scope reduction; requires supervisor approval**

The dataset is publicly archived with full day-level clickstream data, multiple course-year presentations, and a constructible at-risk label. Implementation is feasible but requires an explicitly reduced scope:

### Required Scope Reductions

1. **Reduced graph schema (3 edge types instead of 5)**  
   Build: `enrolled_in`, `interacted_with`, `has_resource`  
   Omit: `submitted`, `contains_assess` (no per-assessment score data available)  
   Document as `# GAP: no assessment submission data` in `src/zenodo_data.py`

2. **Reduced LightGBM feature set (3 features instead of 9)**  
   Use: `vle_total`, `vle_mean`, `vle_std`  
   Omit: `assess_mean`, `assess_max`, `assess_count`, `num_of_prev_attempts`, `studied_credits`, `age_band`  
   The OULAD LightGBM baseline must be re-run on the same 3-feature set for a fair cross-dataset comparison.

3. **Smaller LCPO pool (7 folds instead of 22)**  
   LCPO variance estimates will be less stable. Report mean ± std as before but note the smaller fold count explicitly.

4. **Outcome label semantic difference**  
   `PASSED == 0` (never passed any exam session) vs. OULAD's `Fail + Withdrawn`.  
   Frame as "at-risk prediction under matched temporal constraints" rather than "identical outcome definition." State as a limitation.

5. **No student demographic features**  
   The graph `student` node will carry no feature attributes (or only a constant embedding). The node-feature ablation (`no_course_features`) from OULAD is effectively the baseline here.

### Pre-Conditions Before Sub-Task 7

- [ ] **License confirmed** — Verify CC BY or equivalent on the Zenodo record page before any analysis results are published.
- [ ] **Supervisor approval** — Explicit approval of corrected OULAD results (Sub-Task 5) AND this verdict is required before implementation begins.
- [ ] **OULAD re-run** — Re-run OULAD LightGBM baseline with 3-feature VLE-only set and record as a separate row in `results/graph/comparison_results.csv` with `feature_set = "vle_only"`.
- [ ] **Outcome label decision confirmed** — Supervisor to confirm `PASSED == 0` as the at-risk proxy before pipeline code is written.

---

## 5. Implementation Notes for Sub-Task 7

### New Files Required

| File | Purpose |
|------|---------|
| `src/zenodo_data.py` | Data loader exposing same public API as `src/oulad_data.py`; maps Zenodo files to OULAD-equivalent tables |
| `src/run_zenodo_pipeline.py` | CLI to build Zenodo graph artifacts → `results/zenodo/artifacts/` |
| `results/zenodo/` | Mirrors `results/graph/` directory structure |

### Key Structural Differences from OULAD Pipeline

| OULAD | Zenodo adaptation |
|-------|-----------------|
| Day offsets are integers in source CSVs | Must compute `(TIMESTAMP.date() − course_start).days` from calendar timestamps using `course_info.json` |
| 7 modules × multiple presentations | 2 courses × 3 academic years = 7 presentations; `(year, COURSE_ID)` is the presentation key |
| `studentVle.csv` has `sum_click` | `log_activity.csv` has individual action rows; aggregate by `(USER_ID, CONTENT_ID)` to get click count |
| `assessment` node and `submitted` edge | **Not available** — omit entirely; document as `# GAP` |
| Student demographics from `studentInfo.csv` | Not available — student nodes have no feature attributes |
| `code_module` + `code_presentation` as composite key | `year` + `COURSE_ID` as composite key |

### Discussion Forum Edges (Novel — Not in OULAD)

`df_contribution` and `df_consumption` have no OULAD equivalent. They can optionally be included as additional edge types (`created_post`, `read_posts`) in the Zenodo graph, providing richer social-interaction signal absent from OULAD. If included, they must be treated as an additive extension (not a replacement) and ablated separately to isolate their contribution.

---

*End of report.*
