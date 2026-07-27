# Graph Pipeline — Consolidated Statistics

All figures in this document are sourced directly from pipeline outputs.
To regenerate after a pipeline run:

```bash
python src/summarize_graph_weeks.py          # updates all_weeks_summary.csv
```

To regenerate the graph artifacts themselves:

```bash
python src/run_graph_pipeline.py --week 2
python src/run_graph_pipeline.py --week 4
python src/run_graph_pipeline.py --week 6
python src/run_graph_pipeline.py --week 8
```

See `results/graph/validation/last_regenerated.txt` for the timestamp and git
commit hash of the most recent run.

---

## Week 2 — Prediction Window: 14 days

**Source**: `results/graph/artifacts/week02_metadata.json`,
`results/graph/validation/week02_validation_summary.txt`

| Category | Item | Value |
|----------|------|-------|
| **Window** | Days from course start | 14 |
| **Node counts** | student | 28,785 |
| | course_presentation | 22 |
| | assessment | 1 |
| | vle_resource | 6,364 |
| **Edge counts** | enrolled_in | 32,593 |
| | contains_assess | 1 |
| | has_resource | 6,364 |
| | submitted | 1,089 |
| | interacted_with | 634,723 |
| **Labels** | Total enrollments | 32,593 |
| | At-risk (target=1) | 17,208 (52.8%) |
| | Success (target=0) | 15,385 (47.2%) |
| **Missingness** | student.imd_band (pre-imputation) | 971 → "Unknown" |
| | vle_resource.week_from (pre-imputation) | 5,243 → 0 |
| | vle_resource.week_to (pre-imputation) | 5,243 → 0 |
| | All other columns (post-imputation) | 0 nulls ✓ |
| **Temporal checks** | Max VLE interaction date | ≤ 14 ✓ |
| | Max assessment due date | ≤ 14 ✓ |
| | Max date_submitted | 14 ✓ |
| | All checks compliant | ✓ |
| **Construction** | Runtime | 5.2 s |
| | Peak memory | 915.5 MB |

---

## Week 4 — Prediction Window: 28 days

**Source**: `results/graph/artifacts/week04_metadata.json`,
`results/graph/validation/week04_validation_summary.txt`

| Category | Item | Value |
|----------|------|-------|
| **Window** | Days from course start | 28 |
| **Node counts** | student | 28,785 |
| | course_presentation | 22 |
| | assessment | 17 |
| | vle_resource | 6,364 |
| **Edge counts** | enrolled_in | 32,593 |
| | contains_assess | 17 |
| | has_resource | 6,364 |
| | submitted | 21,393 |
| | interacted_with | 835,935 |
| **Labels** | Total enrollments | 32,593 |
| | At-risk (target=1) | 17,208 (52.8%) |
| | Success (target=0) | 15,385 (47.2%) |
| **Missingness** | student.imd_band (pre-imputation) | 971 → "Unknown" |
| | vle_resource.week_from (pre-imputation) | 5,243 → 0 |
| | vle_resource.week_to (pre-imputation) | 5,243 → 0 |
| | All other columns (post-imputation) | 0 nulls ✓ |
| **Temporal checks** | Max VLE interaction date | ≤ 28 ✓ |
| | Max assessment due date | ≤ 28 ✓ |
| | Max date_submitted | 28 ✓ |
| | All checks compliant | ✓ |
| **Construction** | Runtime | 5.2 s |
| | Peak memory | 915.5 MB |

---

## Week 6 — Prediction Window: 42 days

**Source**: `results/graph/artifacts/week06_metadata.json`,
`results/graph/validation/week06_validation_summary.txt`

| Category | Item | Value |
|----------|------|-------|
| **Window** | Days from course start | 42 |
| **Node counts** | student | 28,785 |
| | course_presentation | 22 |
| | assessment | 24 |
| | vle_resource | 6,364 |
| **Edge counts** | enrolled_in | 32,593 |
| | contains_assess | 24 |
| | has_resource | 6,364 |
| | submitted | 28,569 |
| | interacted_with | 952,241 |
| **Labels** | Total enrollments | 32,593 |
| | At-risk (target=1) | 17,208 (52.8%) |
| | Success (target=0) | 15,385 (47.2%) |
| **Missingness** | student.imd_band (pre-imputation) | 971 → "Unknown" |
| | vle_resource.week_from (pre-imputation) | 5,243 → 0 |
| | vle_resource.week_to (pre-imputation) | 5,243 → 0 |
| | All other columns (post-imputation) | 0 nulls ✓ |
| **Temporal checks** | Max VLE interaction date | ≤ 42 ✓ |
| | Max assessment due date | ≤ 42 ✓ |
| | Max date_submitted | 42 ✓ |
| | All checks compliant | ✓ |
| **Construction** | Runtime | 5.7 s |
| | Peak memory | 915.5 MB |

---

## Week 8 — Prediction Window: 56 days

**Source**: `results/graph/artifacts/week08_metadata.json`,
`results/graph/validation/week08_validation_summary.txt`

| Category | Item | Value |
|----------|------|-------|
| **Window** | Days from course start | 56 |
| **Node counts** | student | 28,785 |
| | course_presentation | 22 |
| | assessment | 40 |
| | vle_resource | 6,364 |
| **Edge counts** | enrolled_in | 32,593 |
| | contains_assess | 40 |
| | has_resource | 6,364 |
| | submitted | 44,927 |
| | interacted_with | 1,056,217 |
| **Labels** | Total enrollments | 32,593 |
| | At-risk (target=1) | 17,208 (52.8%) |
| | Success (target=0) | 15,385 (47.2%) |
| **Missingness** | student.imd_band (pre-imputation) | 971 → "Unknown" |
| | vle_resource.week_from (pre-imputation) | 5,243 → 0 |
| | vle_resource.week_to (pre-imputation) | 5,243 → 0 |
| | All other columns (post-imputation) | 0 nulls ✓ |
| **Temporal checks** | Max VLE interaction date | ≤ 56 ✓ |
| | Max assessment due date | ≤ 54 ✓ |
| | Max date_submitted | 56 ✓ |
| | All checks compliant | ✓ |
| **Construction** | Runtime | 5.6 s |
| | Peak memory | 1,049.0 MB |

---

## Cross-Week Summary

| Week | Window | Assessments | submitted | interacted_with | At-risk% | Compliant |
|------|--------|-------------|-----------|-----------------|----------|-----------|
| 2 | 14 d | 1 | 1,089 | 634,723 | 52.8% | ✓ |
| 4 | 28 d | 17 | 21,393 | 835,935 | 52.8% | ✓ |
| 6 | 42 d | 24 | 28,569 | 952,241 | 52.8% | ✓ |
| 8 | 56 d | 40 | 44,927 | 1,056,217 | 52.8% | ✓ |

> Node counts for student (28,785), course_presentation (22), and vle_resource
> (6,364) are identical across all weeks — these tables are built from static
> source files and are not time-filtered. enrolled_in (32,593) and has_resource
> (6,364) are also constant. Label distribution (52.8% at-risk) is constant
> because `final_result` is a terminal outcome independent of the prediction
> cutoff.

For the machine-readable version of this table see
`results/graph/validation/all_weeks_summary.csv`.
