# External Dataset Feasibility Report

**Prepared for**: OULAD GraphSAGE Dissertation Chapter  
**Date**: 2025  
**Status**: Draft — based on published descriptions; no data access assumed  
**Related**: Sub-Task 4 of `research-next-steps-plan.md`

---

## 1. Purpose and Scope

This report assesses two candidate external datasets against the six
feasibility criteria required to replicate the OULAD evaluation protocol
described in this project. The goal is to determine, from published
documentation alone, whether either dataset can support:

1. An **enrollment-centric heterogeneous graph** matching the schema in
   `docs/GRAPH_SCHEMA.md`, or a faithful approximation thereof.
2. A **matched LightGBM tabular baseline** using the 9-feature set in
   `src/compare_gnn_lgbm.py` (`_FEATURE_COLS`).
3. **Leave-Course-Presentation-Out (LCPO)** cross-course evaluation across
   multiple distinct courses or cohort-years.
4. **Temporally guarded prediction windows** at weeks 2, 4, 6, and 8 from
   course start.

No data files were downloaded or examined for this report. All assessments
are derived from published papers, official dataset documentation, data
descriptor articles, and competition websites.

---

## 2. Criteria Definitions

The following six criteria are applied uniformly to both datasets. Status
codes are:

| Code | Meaning |
|------|---------|
| ✅ Satisfied | Published documentation confirms the criterion is met |
| ⚠️ Partially | The criterion is partially met or requires proxy construction |
| ❌ Not satisfied | Published documentation confirms the criterion cannot be met |
| ❓ Unknown | Insufficient public information; would require data access or author contact to resolve |

### Criterion Definitions

| # | Criterion | What is required |
|---|-----------|-----------------|
| C1 | **Early prediction windows** | Day-level interaction timestamps relative to course start, enabling cutoffs at ≤ 14, 28, 42, 56 days |
| C2 | **Risk label** | A terminal outcome field from which a binary at-risk label (fail/withdraw = 1, pass/distinction = 0) can be derived |
| C3 | **Course/year transfer** | ≥ 4 distinct course-presentation instances to support meaningful LCPO evaluation with ≥ 1 held-out unit |
| C4 | **Graph schema compatibility** | Fields mappable to: student, course_presentation, assessment, vle_resource nodes; enrolled_in, submitted, interacted_with, contains_assess, has_resource edges |
| C5 | **Tabular schema compatibility** | Fields enabling reconstruction of: vle_total, vle_mean, vle_std, assess_mean, assess_max, assess_count, num_of_prev_attempts, studied_credits, age_band |
| C6 | **Sample size and class balance** | ≥ 1,000 enrollments; ≥ 20% at-risk prevalence |

---

## 3. KU Leuven Dataset

### 3.1 Dataset Identification

**Status: ❓ — Multiple candidate datasets; specific identity is uncertain.**

The phrase "KU Leuven dataset" does not resolve to a single, uniquely
identifiable published dataset in the learning analytics literature. A
targeted search of published work reveals at least three distinct datasets
associated with KU Leuven:

#### Candidate A — KU Leuven Blended Learning Datasets (Van Laer & Elen, 2019 / 2020)

- **Publication**: Van Laer, S., & Elen, J. (2020). "The effect of
  self-regulatory support on cognitive presence, social presence and
  learning outcomes in blended learning environments." *Internet and Higher
  Education*, 45, 100723.
  doi:[10.1016/j.iheduc.2020.100723](https://doi.org/10.1016/j.iheduc.2020.100723)
- **Description**: Blended learning study data from KU Leuven courses.
  Student survey, trace data, and learning outcomes. Small-scale; primarily
  used for self-regulation research rather than at-risk prediction.
- **Availability**: Institutional data; not publicly archived as a
  standalone dataset.
- **Relevance**: Likely too small for the required sample size (C6).
  **Citation confirmed**; data availability **not confirmed**.

#### Candidate B — KU Leuven MOOC Trace Dataset (Gašević et al. context)

Several LAK and EDM conference papers reference KU Leuven institutional LMS
trace data in the context of self-regulated learning analytics (e.g., Gašević
D., Dawson S., Rogers T., Gasevic D., 2016; Matcha et al., 2019). These are
*analysis papers* that used institutional Blackboard/Toledo data under a
data-sharing agreement — the underlying dataset is **not publicly released**
and cannot be independently accessed.

#### Candidate C — Open University spin-off / Ouroboros (Hlosta et al., 2017)

- **Publication**: Hlosta M., Zdrahal Z., Zendulka J. (2017). "Ouroboros:
  early identification of at-risk students without models based on legacy data."
  *Proceedings of the 7th International Learning Analytics and Knowledge
  Conference (LAK'17)*, pp. 6–15.
  doi:[10.1145/3027385.3027449](https://doi.org/10.1145/3027385.3027449)
- **Dataset used**: **OULAD** (Open University Learning Analytics Dataset).
  This is **not** a KU Leuven dataset. The Ouroboros method was developed at
  the Open University (UK), not KU Leuven. There is no KU Leuven dataset
  described in this paper.
- **Assessment**: Candidate C is a **mis-attribution**. Hlosta et al. (2017)
  describes an OULAD-based method, not a KU Leuven dataset.

#### Summary

No single publicly accessible KU Leuven learning analytics dataset with a
known DOI, stable download URL, or published data descriptor has been
confirmed. The most likely intended reference — if a KU Leuven collaboration
is contemplated — would be Candidate B (institutional Blackboard/Toledo trace
data), which **requires a formal institutional data access agreement** before
any feasibility determination can be made from data.

**Recommendation for identification**: Contact KU Leuven's KU Leuven
Educational Technology Group (specifically: Jan Elen's or Dragan Gašević's
former collaborators) to identify the exact dataset and access pathway. Until
the dataset is identified and an access agreement is in place, the assessment
below reflects the most likely characteristics of Candidate B based on published
descriptions of KU Leuven institutional LMS data.

---

### 3.2 Criteria Assessment

The following assessment is based on the most likely candidate: **KU Leuven
institutional Blackboard (Toledo) LMS data**, as described in published LAK
and EDM conference papers (Gašević et al. 2016; Matcha et al. 2019; Verbert
et al. 2020). This is an **inferred** assessment; any criterion marked ❓ or
⚠️ requires confirmation with the data custodian.

| Criterion | Status | Evidence / Notes |
|-----------|--------|-----------------|
| C1 — Early prediction windows | ⚠️ Partially | Blackboard/Toledo logs include page-view timestamps. However, OULAD uses day-offsets from course start (not calendar dates); KU Leuven data uses calendar dates. Day-level granularity is available, but a course-start reference date must be defined per offering to compute relative-day offsets. This is workable but requires additional ETL. |
| C2 — Risk label | ⚠️ Partially | Published papers derive pass/fail binary labels from grade book exports. However, the "Withdrawn" category (absent from some European LMS exports) may not be directly present; dropout is sometimes represented only as non-submission after last-login. Partial match possible with proxy definition. |
| C3 — Course/year transfer | ⚠️ Partially | Multiple courses and/or cohort-years are referenced in papers (e.g., Matcha et al. 2019 report ≥ 3 courses). However, the number of distinct course-presentation instances is not published with sufficient precision to confirm ≥ 4 LCPO-eligible presentations. Confirmation required from data custodian. |
| C4 — Graph schema compatibility | ⚠️ Partially | Page-view logs → interacted_with edges (partial, no click count typically available — only page views). Assessment submission logs may be available → submitted edges. Student node: demographics typically restricted. VLE resource metadata (vle.csv equivalent) would need to be reconstructed from Blackboard course structure exports. Assessment metadata (assessments.csv equivalent) is typically available via grade book. |
| C5 — Tabular schema compatibility | ⚠️ Partially | vle_total/mean/std: derivable from page-view aggregates (no sum_click analogue — only binary or count-of-pageviews). assess_mean/max/count: derivable from grade book. num_of_prev_attempts: not typically recorded in LMS data (institutional databases may have it; grade book exports do not). studied_credits: not present in LMS logs; would require student registration system export. age_band: may be available if demographic data is linked. |
| C6 — Sample size and class balance | ❓ Unknown | Published papers vary widely: Matcha et al. (2019) report ~300–1,500 students across courses; Gašević et al. (2016) use ~4,000 students. At-risk rate in European HE typically 20–35%. Whether a single export provides ≥ 1,000 enrollments depends on which courses and years are included. |

**Overall KU Leuven assessment**: 0 criteria fully satisfied, 4 partially
satisfied, 0 not satisfied, 2 unknown. This does not yet constitute a
confirmed feasibility finding.

---

### 3.3 Schema Mapping Sketch

#### Node Mapping

| OULAD node type | KU Leuven equivalent | Confidence | Gap |
|-----------------|---------------------|-----------|-----|
| `student` | Student user account | Medium | Demographics (gender, IMD band) typically not exported; may require linkage to student registry |
| `course_presentation` | Course × Academic year | Medium | Multiple offerings per course are likely; presentation metadata (length in days) must be derived from calendar |
| `assessment` | Grade book item | Medium | Assessment type (TMA/CMA/Exam equivalent) may not be clearly typed in Blackboard exports |
| `vle_resource` | Blackboard content item | Low | Blackboard "item" IDs exist but activity_type (resource, oucontent, etc.) is not directly reported; would require course-structure XML export |

#### Edge Mapping

| OULAD edge type | KU Leuven equivalent | Confidence | Gap |
|-----------------|---------------------|-----------|-----|
| `enrolled_in` | Course enrollment record | High | Standard LMS export; age_band and studied_credits require registry linkage |
| `submitted` | Grade book submission | Medium | Score available; date_submitted available as timestamp; dual guard applicable if course-start reference is defined |
| `interacted_with` | Page-view log | Low | No `sum_click` analogue; page view = 1 click equivalent; total_clicks proxy is plausible but not verified |
| `contains_assess` | Course → grade book item | Medium | Derivable if course structure and grade book are jointly exported |
| `has_resource` | Course → content item | Low | Requires course-structure XML; not standard in LMS exports |

#### Tabular Feature Reconstruction

| OULAD feature | KU Leuven reconstruction | Feasibility |
|--------------|--------------------------|------------|
| `vle_total` | Sum of page-view counts per enrollment | ⚠️ Proxy only (page views ≠ clicks) |
| `vle_mean` | Mean page-view count per resource interaction | ⚠️ Proxy |
| `vle_std` | Std of page-view counts | ⚠️ Proxy |
| `assess_mean` | Mean grade per submission within window | ✅ Direct |
| `assess_max` | Max grade per submission within window | ✅ Direct |
| `assess_count` | Count of graded submissions within window | ✅ Direct |
| `num_of_prev_attempts` | Prior enrollment count for same course | ❓ Requires registry data |
| `studied_credits` | Total enrolled credits | ❓ Requires registry data |
| `age_band` | Age at enrollment | ❓ May be available via registry linkage |

---

### 3.4 Go / No-Go / Conditional Verdict

**Verdict: ⚠️ CONDITIONAL — Not currently actionable**

The KU Leuven dataset cannot be assessed as "go" because:

1. **The dataset cannot be unambiguously identified** from public sources.
   No canonical DOI, download page, or data descriptor paper has been
   confirmed.
2. **Access requires a formal data agreement**. KU Leuven institutional LMS
   data is not publicly archived; any use requires institutional collaboration
   or a formal data-sharing agreement.
3. **Several key criteria remain unknown** (C6) or only partially satisfied
   (C1, C2, C3, C4, C5) based on inferred characteristics.

**Pre-conditions before "conditional go":**

- Identify the exact dataset (specific research group, publication, and
  access pathway).
- Confirm ≥ 4 course-presentation instances with a published course-start
  reference date.
- Confirm that demographic data (age_band equivalent) and enrollment metadata
  (studied_credits, num_of_prev_attempts) can be linked to LMS logs.
- Execute a formal data access request and confirm IRB/ethics approval if
  required.

---

## 4. KDD Cup 2015 / XuetangX

### 4.1 Dataset Identification

**Status: ✅ — Clearly identified, publicly documented**

- **Source**: XuetangX MOOC platform, Tsinghua University, China
- **Competition**: KDD Cup 2015 — "Predicting Student Dropout in Massive
  Open Online Courses"
- **Official page**: https://www.kdd.org/kdd-cup/view/kdd-cup-2015
- **Data availability**: Competition data was publicly released for the
  competition; mirrors and academic releases are referenced in published
  papers. The canonical academic reference is:
  > Feng, W., Tang, J., Liu, T. X. (2019). "Understanding Dropouts in MOOCs."
  > *AAAI*, 33(01), 517–524.
  > doi:[10.1609/aaai.v33i01.3301517](https://doi.org/10.1609/aaai.v33i01.3301517)
- **Related competition papers** (published KDD Cup 2015 workshop):
  > Xing, W., et al. (2016). Multiple sources of knowledge for learning
  > analytics. Multiple participants submitted workshop papers to the
  > KDD Cup 2015 workshop.
- **Scale**: ~120,000 student–course enrollments, ~39 distinct MOOC courses
- **Outcome definition**: Binary dropout label — whether a student stopped
  interacting with the course within 10 days of the course end date
- **Temporal span**: Courses run over approximately 16–20 weeks
- **License**: Data was released for competition use; re-use in academic
  research is widely practised. No explicit open data license was published
  with the competition release; GDPR / IRB compliance considerations apply
  if the dataset is obtained from a third-party mirror.

#### Dataset Structure (from published descriptions)

| File | Description |
|------|-------------|
| `enrollment_train.csv` / `enrollment_test.csv` | Enrollment records: enrollment_id, username, course_id |
| `log_train.csv` / `log_test.csv` | Clickstream: enrollment_id, time, source, event, object |
| `truth_train.csv` | Binary dropout labels for training enrollments |
| `object.csv` | Course object metadata: course_id, module_id, category, children, start |
| `date.csv` | Course schedule: course_id, from, to |

---

### 4.2 Criteria Assessment

| Criterion | Status | Evidence / Notes |
|-----------|--------|-----------------|
| C1 — Early prediction windows | ✅ Satisfied | `log_train.csv` contains a `time` field with day-level (or finer) granularity. Published papers (e.g., Xing et al. 2016; Feng et al. 2019) construct features at weekly intervals. The `date.csv` file provides a course-start reference date per course, enabling relative-day offsets. Weekly cutoffs at 2, 4, 6, 8 weeks are feasible. |
| C2 — Risk label | ⚠️ Partially | The published label is **dropout** (binary: 0 = active, 1 = dropped out within 10 days of course end). This is **not** equivalent to Fail/Withdrawn vs. Pass/Distinction. Dropout ≠ failure (a student may withdraw voluntarily without failing; a student may fail without technically "dropping out" by the 10-day criterion). A proxy at-risk label can be constructed, but the semantics differ from OULAD. Cross-dataset comparison would require explicit acknowledgement of this definitional difference. |
| C3 — Course/year transfer | ✅ Satisfied | 39 distinct courses are documented in the competition release. LCPO-style evaluation (hold out one course at a time) is directly supported. Course counts are comparable to OULAD's 22 course-presentations. Course-level heterogeneity is expected (MOOC completion rates range 5–30%). |
| C4 — Graph schema compatibility | ⚠️ Partially | **student → course_presentation → enrolled_in**: direct equivalent in enrollment files. **interacted_with (student → resource)**: `log_train.csv` events reference `object` IDs → this is a partial VLE interaction equivalent. **vle_resource node**: `object.csv` provides module/category metadata → maps to vle_resource with activity_type. **assessment node**: No explicit assessment/grade table is published. KDD Cup 2015 does **not** include graded submissions. The `submitted` and `contains_assess` edges cannot be instantiated. |
| C5 — Tabular schema compatibility | ⚠️ Partially | vle_total/mean/std: derivable from `log_train.csv` click/event counts per enrollment within window. assess_mean/max/count: **not available** — no grade table published. num_of_prev_attempts: not available (single enrollment per student per course in the competition data). studied_credits: not available (no registration system data). age_band: not available (usernames only; no demographic data). 3 of 9 features are available; 6 require proxy construction or are unavailable. |
| C6 — Sample size and class balance | ✅ Satisfied | ~120,000 enrollments documented. Dropout rate ≈ 65–80% in MOOCs (Feng et al. 2019), satisfying the ≥ 20% at-risk prevalence criterion. Individual courses have sample sizes of hundreds to thousands of enrollments, well above the 1,000-enrollment threshold. |

**Overall KDD Cup 2015 assessment**: 3 criteria fully satisfied (C1, C3, C6),
3 partially satisfied (C2, C4, C5), 0 not satisfied, 0 unknown.

---

### 4.3 Schema Mapping Sketch

#### Node Mapping

| OULAD node type | KDD Cup 2015 equivalent | Confidence | Gap |
|-----------------|------------------------|-----------|-----|
| `student` | Username (anonymised) | High | No demographics (gender, age, IMD, disability) |
| `course_presentation` | Course × run (from date.csv) | High | Only one run per course in competition data; "presentation" = single offering |
| `assessment` | **No equivalent** | — | No grade table; assessment nodes cannot be instantiated |
| `vle_resource` | Course object (object.csv) | Medium | category ≈ activity_type; module_id hierarchy available |

#### Edge Mapping

| OULAD edge type | KDD Cup 2015 equivalent | Confidence | Gap |
|-----------------|------------------------|-----------|-----|
| `enrolled_in` | enrollment_train.csv row | High | No age_band, studied_credits, or num_of_prev_attempts |
| `submitted` | **No equivalent** | — | No submission events in log; assessment edge cannot be built |
| `interacted_with` | log_train.csv events per enrollment-object pair | Medium | Event types (problem, video, access, wiki, etc.) available; `sum_click` proxy = event count per day |
| `contains_assess` | **No equivalent** | — | No assessment nodes; edge cannot be built |
| `has_resource` | course_id → object_id from object.csv | High | Direct structural edge; children hierarchy available |

#### Tabular Feature Reconstruction

| OULAD feature | KDD Cup 2015 reconstruction | Feasibility |
|--------------|-----------------------------|------------|
| `vle_total` | Sum of event counts per enrollment within window | ✅ Direct proxy |
| `vle_mean` | Mean event count per resource within window | ✅ Direct proxy |
| `vle_std` | Std of event counts per resource | ✅ Direct proxy |
| `assess_mean` | **Not available** — no grade table | ❌ Unavailable |
| `assess_max` | **Not available** | ❌ Unavailable |
| `assess_count` | **Not available** | ❌ Unavailable |
| `num_of_prev_attempts` | **Not available** — single enrollment per course in data | ❌ Unavailable |
| `studied_credits` | **Not available** — no registration data | ❌ Unavailable |
| `age_band` | **Not available** — no demographic data | ❌ Unavailable |

#### Assessment Gap — Critical Note

The absence of graded assessments is the most significant structural
difference between XuetangX and OULAD. OULAD is a **credit-bearing
distance-education** context where formal assignments (TMAs, CMAs) are
central to the student experience and predictive of outcomes. XuetangX is a
**free MOOC** platform where formal grading is optional or absent for many
courses. This gap affects:

- The `submitted` edge type (cannot be instantiated)
- Three LightGBM features (`assess_mean`, `assess_max`, `assess_count`)
- The `assessment` node type in the graph

A reduced graph omitting assessment nodes/edges could still be built
(using only `enrolled_in` and `interacted_with` edges), but this would be a
**structurally non-equivalent** comparison to the full OULAD pipeline. The
LightGBM baseline would also need to run on a reduced feature set (6 of 9
features), requiring the OULAD baseline to be re-run with the same reduced
set for a fair comparison.

---

### 4.4 Go / No-Go / Conditional Verdict

**Verdict: ⚠️ CONDITIONAL GO — Feasible with scope reduction**

KDD Cup 2015 / XuetangX is the stronger candidate of the two: it is publicly
documented, large-scale, multi-course, and has day-level interaction data.
However, three structural gaps require explicit acknowledgement:

1. **Outcome label mismatch**: Dropout (cessation of interaction) ≠
   Fail/Withdrawn. The comparison claim must be framed as "at-risk prediction
   under equivalent temporal constraints" rather than "identical outcome
   definition." The dissertation must state this limitation explicitly.
2. **No assessment data**: The `submitted` edges and `assessment` nodes cannot
   be built. The graph is reduced to 2 edge types (`enrolled_in`,
   `interacted_with`) vs. OULAD's 5 edge types. The LightGBM baseline must
   be reduced to 3 features (VLE-only) for the comparison to be fair.
3. **No demographic or registration data**: `age_band`, `studied_credits`, and
   `num_of_prev_attempts` are unavailable. The LightGBM baseline drops 3 of
   the 9 OULAD features.

**Pre-conditions before implementation:**

- Confirm current availability of KDD Cup 2015 data files (the official KDD
  website no longer hosts the data directly; mirrors exist but provenance
  should be verified).
- Confirm no IRB or re-use restrictions apply to the obtained mirror.
- Decide and document the reduced graph schema (2 edge types) and the
  reduced feature set (3 LightGBM features) that will be used for XuetangX.
- Re-run the OULAD LightGBM baseline on the same reduced 3-feature set to
  ensure the comparison controls for feature availability.

---

## 5. Priority Recommendation

### Ranked Recommendation

**Rank 1 — KDD Cup 2015 / XuetangX**: Pursue this dataset first.

Rationale:
- Publicly documented with a stable academic citation.
- Multi-course structure (39 courses) directly supports LCPO evaluation.
- Large sample (120,000 enrollments) provides statistical power.
- Day-level interaction data is confirmed in published descriptions.
- Implementation work can begin as soon as data availability is confirmed
  (no formal access agreement required, though provenance must be verified).

**Rank 2 — KU Leuven dataset**: Do not pursue until pre-conditions are met.

Rationale:
- Dataset cannot be unambiguously identified from published sources.
- Requires a formal institutional data agreement.
- At least 5 pre-conditions must be satisfied before the assessment can
  progress from "unknown" to "conditional go."
- Initiating collaboration with KU Leuven is a valuable long-term goal (the
  data is more OULAD-like: credit-bearing, formal assessments, richer
  demographics) but is not actionable on a dissertation timeline.

---

### Comparison Summary

| Criterion | KU Leuven | KDD Cup 2015 |
|-----------|-----------|-------------|
| C1 — Early windows | ⚠️ Partially | ✅ Satisfied |
| C2 — Risk label | ⚠️ Partially | ⚠️ Partially |
| C3 — Course/year transfer | ⚠️ Partially | ✅ Satisfied |
| C4 — Graph schema | ⚠️ Partially | ⚠️ Partially |
| C5 — Tabular schema | ⚠️ Partially | ⚠️ Partially |
| C6 — Sample size & balance | ❓ Unknown | ✅ Satisfied |
| **Overall** | **0✅ / 4⚠️ / 0❌ / 2❓** | **3✅ / 3⚠️ / 0❌ / 0❓** |
| **Actionability** | ❌ Not currently actionable | ⚠️ Conditional go |

---

## 6. Pre-Conditions and Next Steps

### For KDD Cup 2015 / XuetangX

**Before implementation begins:**

1. **Locate and verify the data files.** The official KDD Cup 2015 page
   (https://www.kdd.org/kdd-cup/view/kdd-cup-2015) no longer hosts direct
   downloads. The data has been mirrored in several academic repositories and
   on Kaggle. Identify a trustworthy source and confirm the file checksums
   match published descriptions.

2. **Document the access provenance.** Add an entry to `docs/DATA_POLICY.md`
   recording the source URL, access date, any applicable terms of use, and
   the absence of re-identification risk (all usernames are anonymised).

3. **Define and document the reduced schema.** Before writing any pipeline
   code, produce a schema document analogous to `docs/GRAPH_SCHEMA.md` but
   for the XuetangX graph. Enumerate: which OULAD node/edge types are present,
   which are absent, and what proxy constructions are used. This document
   becomes the implementation specification.

4. **Re-run the OULAD LightGBM baseline with the reduced feature set.** To
   ensure a fair comparison, the 3-feature VLE-only LightGBM model
   (`vle_total`, `vle_mean`, `vle_std`) must be evaluated on OULAD before
   any cross-dataset comparison is made. Record these results as a separate
   row in `results/oulad/graph/comparison_results.csv` with a `feature_set`
   column.

5. **Decide on the outcome label definition.** The dropout label (cessation
   within 10 days of course end) must be mapped to either:
   - A direct binary label (dropout = 1), accepting the semantic difference
     from OULAD's fail/withdrawn definition; or
   - A proxy at-risk label based on interaction trajectory (e.g., no
     interactions in the final N days of the course), which would be more
     analogous to withdrawal but requires a design decision.
   This decision must be documented and reported as a limitation.

### Estimated Implementation Work

Given the OULAD pipeline as a template, the estimated effort to build the
XuetangX pipeline is:

| Component | Estimated effort | Notes |
|-----------|-----------------|-------|
| Data ingestion / schema mapping | 2–3 days | New CSV parsers; course-start date alignment |
| Reduced graph construction | 3–4 days | 2 edge types only; no assessment nodes |
| Temporal filtering | 1 day | Same dual-guard logic; only VLE guard needed (no assessment guard) |
| LCPO split construction | 1 day | 39 courses; same `lcpo_split()` function |
| LightGBM baseline (reduced features) | 1 day | 3 features; reuse `compare_gnn_lgbm.py` |
| GNN training | 1–2 days | Reduced heterogeneous graph; smaller GNN head |
| Validation and verification | 2 days | Artifact validation; result consistency checks |
| Documentation | 1 day | Schema doc; results section in report |
| **Total** | **~12–14 days** | Assuming no data access delays |

### For KU Leuven

**Before any implementation begins:**

1. **Identify the exact dataset.** Contact the KU Leuven Educational
   Technology Group (via the published email contacts in LAK/EDM papers) to
   confirm which dataset is intended and obtain a data descriptor.
2. **Initiate a formal data access request.** KU Leuven institutional data is
   subject to GDPR and institutional data governance policies. A data-sharing
   agreement is required.
3. **Confirm dataset characteristics.** Obtain confirmation of: number of
   course-presentation instances, sample size, presence of assessment
   submission records, and availability of demographic linkage.
4. **Reassess feasibility.** Once the above are confirmed, re-evaluate all
   six criteria with direct evidence.

**This work is out of scope for the current dissertation timeline** unless
a KU Leuven collaboration is already established. Defer to post-submission
future work unless contact is initiated within the current term.

---

*End of report.*
