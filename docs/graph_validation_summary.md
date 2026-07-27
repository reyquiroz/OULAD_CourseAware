# OULAD Graph Pipeline — Multi-Week Validation Summary

> For the full per-week consolidated statistics table (node counts, edge counts,
> label distribution, missingness, temporal checks, runtime), see
> **[`docs/graph_consolidated_stats.md`](graph_consolidated_stats.md)**.

Produced by `src/summarize_graph_weeks.py`.
Source: `week{N}_metadata.json` and `week{N}_validation.json` in
`results/graph/`.

**Temporal filtering**: dual guard (Strategy B) —
`assessments.date ≤ window` AND `date_submitted ≤ window`.
VLE interactions: `date ≤ window`.

**Supervised unit**: enrollment `(id_student, code_module, code_presentation)`.

**Label convention**: `target=1` → at-risk (Fail/Withdrawn);
`target=0` → success (Pass/Distinction).

---

## Node Counts by Week

| Week | Window (days) | Students | Course-Pres. | Assessments | VLE Resources |
| --- | --- | --- | --- | --- | --- |
| 2 | 14 | 28,785 | 22 | 1 | 6,364 |
| 4 | 28 | 28,785 | 22 | 17 | 6,364 |
| 6 | 42 | 28,785 | 22 | 24 | 6,364 |
| 8 | 56 | 28,785 | 22 | 40 | 6,364 |

## Edge Counts by Week

| Week | enrolled_in | contains_assess | has_resource | submitted | interacted_with |
| --- | --- | --- | --- | --- | --- |
| 2 | 32,593 | 1 | 6,364 | 1,089 | 634,723 |
| 4 | 32,593 | 17 | 6,364 | 21,393 | 835,935 |
| 6 | 32,593 | 24 | 6,364 | 28,569 | 952,241 |
| 8 | 32,593 | 40 | 6,364 | 44,927 | 1,056,217 |

## Label Distribution

| Week | Total Enrollments | At-risk | At-risk Rate |
| --- | --- | --- | --- |
| 2 | 32,593 | 17,208 | 52.8% |
| 4 | 32,593 | 17,208 | 52.8% |
| 6 | 32,593 | 17,208 | 52.8% |
| 8 | 32,593 | 17,208 | 52.8% |

## Temporal Compliance

| Week | Window (days) | Max VLE last_day | Max assess due_date | Max date_submitted | All compliant |
| --- | --- | --- | --- | --- | --- |
| 2 | 14 | ≤14 | ≤14 | 14 | ✓ |
| 4 | 28 | ≤28 | ≤28 | 28 | ✓ |
| 6 | 42 | ≤42 | ≤42 | 42 | ✓ |
| 8 | 56 | ≤56 | ≤56 | 56 | ✓ |

## Pre-Imputation Null Counts

Expected nulls from raw OULAD source data (resolved by imputation):

| Week | student.imd_band | vle_resource.week_from | vle_resource.week_to |
| --- | --- | --- | --- |
| 2 | 971 | 5243 | 5243 |
| 4 | 971 | 5243 | 5243 |
| 6 | 971 | 5243 | 5243 |
| 8 | 971 | 5243 | 5243 |

## Runtime and Memory

| Week | Runtime (s) | Peak memory (MB) |
| --- | --- | --- |
| 2 | 7.0 | 915.5 |
| 4 | 5.3 | 915.5 |
| 6 | 6.3 | 915.5 |
| 8 | 6.1 | 1049.0 |

---

For the full integrity report for any week, see
`results/graph/validation/week{N:02d}_validation_summary.txt`.

For Week 8 deep-dive, see `docs/validation_report_week8.md`.

For the graph schema, see `docs/GRAPH_SCHEMA.md`.
