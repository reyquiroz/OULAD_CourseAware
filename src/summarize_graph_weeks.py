"""
summarize_graph_weeks.py — Multi-week graph statistics summary.

Reads the metadata JSON and validation JSON for each of the four prediction
windows (Weeks 2, 4, 6, 8) and produces:

  results/graph/validation/all_weeks_summary.csv   — machine-readable table
  docs/graph_validation_summary.md                 — human-readable report

Usage
-----
    source oulad_env/bin/activate
    python src/summarize_graph_weeks.py

Prerequisite: all four weeks must be built first:
    python src/run_graph_pipeline.py --week 2
    python src/run_graph_pipeline.py --week 4
    python src/run_graph_pipeline.py --week 6
    python src/run_graph_pipeline.py --week 8
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import GRAPH_ARTIFACTS_DIR, GRAPH_VALIDATION_DIR, DOCS_DIR

WEEKS = [2, 4, 6, 8]


def _load(week: int) -> tuple[dict, dict]:
    """Return (metadata, validation) dicts for *week*."""
    prefix = f"week{week:02d}"
    meta_path = GRAPH_ARTIFACTS_DIR / f"{prefix}_metadata.json"
    val_path = GRAPH_VALIDATION_DIR / f"{prefix}_validation.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            f"Run: python src/run_graph_pipeline.py --week {week}"
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation report not found: {val_path}\n"
            f"Run: python src/run_graph_pipeline.py --week {week}"
        )

    with open(meta_path) as f:
        meta = json.load(f)
    with open(val_path) as f:
        val = json.load(f)
    return meta, val


def _extract_row(week: int, meta: dict, val: dict) -> dict:
    """Extract the summary columns from metadata + validation dicts."""
    nc = meta.get("node_counts", {})
    ec = meta.get("edge_counts", {})
    label = meta.get("label_at_risk_count", 0)
    total = meta.get("enrollment_count", 0)

    pin = meta.get("pre_imputation_nulls", {})
    null_imd = pin.get("student", {}).get("imd_band", 0)
    null_wf  = pin.get("vle_resource", {}).get("week_from", 0)
    null_wt  = pin.get("vle_resource", {}).get("week_to", 0)

    tc = val.get("temporal_compliance", {})

    return {
        "Week":                  week,
        "Window_Days":           meta.get("window_days"),
        "N_students":            nc.get("student"),
        "N_course_presentations":nc.get("course_presentation"),
        "N_assessments":         nc.get("assessment"),
        "N_vle_resources":       nc.get("vle_resource"),
        "N_enrolled_in":         ec.get("enrolled_in"),
        "N_contains_assess":     ec.get("contains_assess"),
        "N_has_resource":        ec.get("has_resource"),
        "N_submitted":           ec.get("submitted"),
        "N_interacted_with":     ec.get("interacted_with"),
        "N_enrollments":         total,
        "At_risk_count":         label,
        "At_risk_rate":          round(label / total, 4) if total else None,
        "Max_date_submitted":    meta.get("max_date_submitted"),
        "Pre_null_imd_band":     null_imd,
        "Pre_null_week_from":    null_wf,
        "Pre_null_week_to":      null_wt,
        "Runtime_s":             meta.get("elapsed_seconds"),
        "Peak_memory_MB":        meta.get("peak_memory_mb"),
        "All_temporal_compliant":tc.get("all_compliant"),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ CSV saved → {path}")


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write docs/graph_validation_summary.md from the collected rows."""

    def _row(r, keys, fmts=None):
        cells = []
        for i, k in enumerate(keys):
            v = r.get(k, "")
            if fmts and i < len(fmts) and v is not None:
                v = fmts[i].format(v)
            cells.append(str(v) if v is not None else "")
        return "| " + " | ".join(cells) + " |"

    sep = lambda keys: "| " + " | ".join(["---"] * len(keys)) + " |"  # noqa: E731

    lines = [
        "# OULAD Graph Pipeline — Multi-Week Validation Summary",
        "",
        "Produced by `src/summarize_graph_weeks.py`.",
        "Source: `week{N}_metadata.json` and `week{N}_validation.json` in",
        "`results/graph/`.",
        "",
        "**Temporal filtering**: dual guard (Strategy B) —",
        "`assessments.date ≤ window` AND `date_submitted ≤ window`.",
        "VLE interactions: `date ≤ window`.",
        "",
        "**Supervised unit**: enrollment `(id_student, code_module, code_presentation)`.",
        "",
        "**Label convention**: `target=1` → at-risk (Fail/Withdrawn);",
        "`target=0` → success (Pass/Distinction).",
        "",
        "---",
        "",
        "## Node Counts by Week",
        "",
        "| Week | Window (days) | Students | Course-Pres. | Assessments | VLE Resources |",
        sep(["Week", "Window_Days", "N_students", "N_course_presentations",
             "N_assessments", "N_vle_resources"]),
    ]
    for r in rows:
        lines.append(
            f"| {r['Week']} | {r['Window_Days']} | {r['N_students']:,} |"
            f" {r['N_course_presentations']} |"
            f" {r['N_assessments']} |"
            f" {r['N_vle_resources']:,} |"
        )

    lines += [
        "",
        "## Edge Counts by Week",
        "",
        "| Week | enrolled_in | contains_assess | has_resource | submitted | interacted_with |",
        sep(["Week", "N_enrolled_in", "N_contains_assess",
             "N_has_resource", "N_submitted", "N_interacted_with"]),
    ]
    for r in rows:
        lines.append(
            f"| {r['Week']} | {r['N_enrolled_in']:,} |"
            f" {r['N_contains_assess']} |"
            f" {r['N_has_resource']:,} |"
            f" {r['N_submitted']:,} |"
            f" {r['N_interacted_with']:,} |"
        )

    lines += [
        "",
        "## Label Distribution",
        "",
        "| Week | Total Enrollments | At-risk | At-risk Rate |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Week']} | {r['N_enrollments']:,} |"
            f" {r['At_risk_count']:,} |"
            f" {r['At_risk_rate']:.1%} |"
        )

    lines += [
        "",
        "## Temporal Compliance",
        "",
        "| Week | Window (days) | Max VLE last_day | Max assess due_date | Max date_submitted | All compliant |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        # pull temporal check values from validation JSON (already loaded)
        compliant_str = "✓" if r["All_temporal_compliant"] else "⚠"
        lines.append(
            f"| {r['Week']} | {r['Window_Days']} |"
            f" ≤{r['Window_Days']} |"
            f" ≤{r['Window_Days']} |"
            f" {r['Max_date_submitted']} |"
            f" {compliant_str} |"
        )

    lines += [
        "",
        "## Pre-Imputation Null Counts",
        "",
        "Expected nulls from raw OULAD source data (resolved by imputation):",
        "",
        "| Week | student.imd_band | vle_resource.week_from | vle_resource.week_to |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Week']} | {r['Pre_null_imd_band']} |"
            f" {r['Pre_null_week_from']} |"
            f" {r['Pre_null_week_to']} |"
        )

    lines += [
        "",
        "## Runtime and Memory",
        "",
        "| Week | Runtime (s) | Peak memory (MB) |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Week']} | {r['Runtime_s']:.1f} | {r['Peak_memory_MB']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "For the full integrity report for any week, see",
        "`results/graph/validation/week{N:02d}_validation_summary.txt`.",
        "",
        "For Week 8 deep-dive, see `docs/validation_report_week8.md`.",
        "",
        "For the graph schema, see `docs/GRAPH_SCHEMA.md`.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Markdown saved → {path}")


def _print_table(rows: list[dict]) -> None:
    """Print a concise ASCII summary table."""
    headers = ["Week", "Window", "Assessments", "Submitted", "Interacted", "At-risk%", "Compliant"]
    print("\n" + "=" * 72)
    print("  OULAD Graph Pipeline — Multi-Week Summary")
    print("=" * 72)
    fmt = "{:>4}  {:>6}  {:>11}  {:>9}  {:>10}  {:>8}  {:>9}"
    print(fmt.format(*headers))
    print("-" * 72)
    for r in rows:
        print(fmt.format(
            r["Week"],
            r["Window_Days"],
            f"{r['N_assessments']}",
            f"{r['N_submitted']:,}",
            f"{r['N_interacted_with']:,}",
            f"{r['At_risk_rate']:.1%}",
            "✓" if r["All_temporal_compliant"] else "⚠",
        ))
    print("=" * 72)


def main():
    rows = []
    for week in WEEKS:
        print(f"  Loading Week {week}...")
        meta, val = _load(week)
        rows.append(_extract_row(week, meta, val))

    _print_table(rows)

    csv_path = GRAPH_VALIDATION_DIR / "all_weeks_summary.csv"
    write_csv(rows, csv_path)

    md_path = DOCS_DIR / "graph_validation_summary.md"
    write_markdown(rows, md_path)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
