"""
graph_validation.py — Graph validation and statistics reporting (subtask 4)
============================================================================
Reads the saved Week-N graph CSV artifacts produced by graph_pipeline.py and
produces two outputs:

  results/graph/validation/week{N:02d}_validation.json  — machine-readable
  results/graph/validation/week{N:02d}_validation_summary.txt — human-readable

Checks implemented (matching the approved subtask-4 scope):
  1. Node and edge counts by type (including enrollment supervision count)
  2. Duplicate nodes and edges per type (all five edge types)
  3. Dangling edges for all five edge relations
  4. Missing / null / NaN / infinite values per artifact and column
  5. Class distribution overall and by course presentation
  6. Temporal-cutoff compliance for VLE interaction days, assessment due dates,
     and aggregated first_day / last_day in interacted_with edges
  7. Graph construction runtime and peak memory (from existing integrity JSON)

Usage:
    python src/graph_validation.py --week 8
    python src/run_graph_validation.py --week 8   # CLI wrapper
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import GRAPH_ARTIFACTS_DIR, GRAPH_VALIDATION_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_inf_summary(df: pd.DataFrame) -> dict:
    """Return per-column counts of null/NaN and +/-Inf for numeric columns."""
    summary = {}
    for col in df.columns:
        null_n = int(df[col].isna().sum())
        inf_n = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_n = int(np.isinf(df[col].replace({None: np.nan})).sum())
        if null_n > 0 or inf_n > 0:
            summary[col] = {"null": null_n, "inf": inf_n}
    return summary


def _total_issues(summary: dict) -> int:
    """Sum null+inf counts across all columns in a _null_inf_summary result."""
    return sum(v["null"] + v["inf"] for v in summary.values())


def _dangling_check(edf: pd.DataFrame, src_idx: "set", dst_idx: "set") -> dict:
    """Count src/dst ids in edf that are not present in the valid index sets."""
    dangling_src = int((~edf["src"].isin(src_idx)).sum())
    dangling_dst = int((~edf["dst"].isin(dst_idx)).sum())
    return {"dangling_src": dangling_src, "dangling_dst": dangling_dst}


# ---------------------------------------------------------------------------
# Core validation function
# ---------------------------------------------------------------------------

def run_validation(week: int, artifacts_dir: Path, validation_dir: Path) -> dict:
    """
    Load saved graph CSV artifacts for *week* and run all subtask-4 checks.

    Returns a dict suitable for JSON serialisation.  Also prints a concise
    summary to stdout and writes the two output files.
    """
    prefix = f"week{week:02d}"
    ad = artifacts_dir   # shorthand

    # ── 1. Load artifacts ────────────────────────────────────────────────────
    nodes = {
        "student":             pd.read_csv(ad / f"{prefix}_nodes_student.csv"),
        "course_presentation": pd.read_csv(ad / f"{prefix}_nodes_course_presentation.csv"),
        "assessment":          pd.read_csv(ad / f"{prefix}_nodes_assessment.csv"),
        "vle_resource":        pd.read_csv(ad / f"{prefix}_nodes_vle_resource.csv"),
    }
    edges = {
        "enrolled_in":     pd.read_csv(ad / f"{prefix}_edges_enrolled_in.csv"),
        "contains_assess": pd.read_csv(ad / f"{prefix}_edges_contains_assess.csv"),
        "has_resource":    pd.read_csv(ad / f"{prefix}_edges_has_resource.csv"),
        "submitted":       pd.read_csv(ad / f"{prefix}_edges_submitted.csv"),
        "interacted_with": pd.read_csv(ad / f"{prefix}_edges_interacted_with.csv"),
    }
    enrollments = pd.read_csv(ad / f"{prefix}_enrollments.csv")

    # ── 2. Load existing integrity JSON for runtime / memory ─────────────────
    integrity_path = validation_dir / f"{prefix}_integrity.json"
    runtime_stats: dict = {}
    if integrity_path.exists():
        with open(integrity_path) as f:
            integrity_data = json.load(f)
        runtime_stats = {
            "elapsed_seconds": integrity_data.get("elapsed_seconds"),
            "peak_memory_mb":  integrity_data.get("peak_memory_mb"),
        }

    # ── 3. Node and edge counts ──────────────────────────────────────────────
    node_counts = {ntype: int(len(df)) for ntype, df in nodes.items()}
    edge_counts = {etype: int(len(df)) for etype, df in edges.items()}
    enrollment_count = int(len(enrollments))

    # ── 4. Duplicate nodes ───────────────────────────────────────────────────
    id_cols = {
        "student":             "id_student",
        "course_presentation": ["code_module", "code_presentation"],
        "assessment":          "id_assessment",
        "vle_resource":        "id_site",
    }
    dup_nodes = {
        ntype: int(nodes[ntype].duplicated(subset=col).sum())
        for ntype, col in id_cols.items()
    }

    # ── 5. Duplicate edges (all five types) ──────────────────────────────────
    dup_edges = {
        etype: int(edf.duplicated(subset=["src", "dst"]).sum())
        for etype, edf in edges.items()
    }

    # ── 6. Dangling edges (all five types) ───────────────────────────────────
    stu_idx  = set(nodes["student"]["node_idx"].values)
    cp_idx   = set(nodes["course_presentation"]["node_idx"].values)
    ass_idx  = set(nodes["assessment"]["node_idx"].values)
    vle_idx  = set(nodes["vle_resource"]["node_idx"].values)

    # Edge (src_type, dst_type) pairs
    edge_node_map = {
        "enrolled_in":     (stu_idx, cp_idx),
        "contains_assess": (cp_idx,  ass_idx),
        "has_resource":    (cp_idx,  vle_idx),
        "submitted":       (stu_idx, ass_idx),
        "interacted_with": (stu_idx, vle_idx),
    }
    dangling_edges = {
        etype: _dangling_check(edges[etype], *node_sets)
        for etype, node_sets in edge_node_map.items()
    }

    # ── 7. Missing / null / NaN / Inf per artifact and column ────────────────
    data_quality: dict = {}

    for ntype, ndf in nodes.items():
        summary = _null_inf_summary(ndf)
        data_quality[f"nodes_{ntype}"] = {
            "total_null_inf": _total_issues(summary),
            "by_column": summary,
        }

    for etype, edf in edges.items():
        summary = _null_inf_summary(edf)
        data_quality[f"edges_{etype}"] = {
            "total_null_inf": _total_issues(summary),
            "by_column": summary,
        }

    enroll_summary = _null_inf_summary(enrollments)
    data_quality["enrollments"] = {
        "total_null_inf": _total_issues(enroll_summary),
        "by_column": enroll_summary,
    }

    # ── 8. Class distribution overall and by course presentation ─────────────
    label_overall = {
        "at_risk":       int((enrollments["target"] == 1).sum()),
        "success":       int((enrollments["target"] == 0).sum()),
        "total":         enrollment_count,
        "at_risk_rate":  round(float((enrollments["target"] == 1).mean()), 4),
    }

    label_by_cp: dict = {}
    for (mod, pres), grp in enrollments.groupby(["code_module", "code_presentation"]):
        key = f"{mod}_{pres}"
        n_total  = int(len(grp))
        n_atrisk = int((grp["target"] == 1).sum())
        label_by_cp[key] = {
            "at_risk":      n_atrisk,
            "success":      n_total - n_atrisk,
            "total":        n_total,
            "at_risk_rate": round(n_atrisk / n_total, 4) if n_total > 0 else None,
        }

    # ── 9. Temporal-cutoff compliance ─────────────────────────────────────────
    # Derive the expected cutoff from the artifact prefix / metadata
    meta_path = ad / f"{prefix}_metadata.json"
    window_days: int = week * 7  # fallback
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        window_days = int(meta.get("window_days", window_days))

    temporal: dict = {"window_days": window_days, "checks": {}}

    # VLE interaction day (interacted_with: first_day, last_day)
    iw = edges["interacted_with"]
    if "last_day" in iw.columns:
        max_last = int(iw["last_day"].max())
        temporal["checks"]["interacted_with_max_last_day"] = {
            "value": max_last,
            "threshold": window_days,
            "compliant": bool(max_last <= window_days),
        }
    if "first_day" in iw.columns:
        max_first = int(iw["first_day"].max())
        temporal["checks"]["interacted_with_max_first_day"] = {
            "value": max_first,
            "threshold": window_days,
            "compliant": bool(max_first <= window_days),
        }

    # Assessment due date (assessment node: date column)
    ass_df = nodes["assessment"]
    if "date" in ass_df.columns:
        max_due = int(ass_df["date"].dropna().max())
        temporal["checks"]["assessment_max_due_date"] = {
            "value": max_due,
            "threshold": window_days,
            "compliant": bool(max_due <= window_days),
        }

    # Aggregated click counters must be > 0 (sanity check)
    if "total_clicks" in iw.columns:
        min_clicks = int(iw["total_clicks"].min())
        temporal["checks"]["interacted_with_min_total_clicks"] = {
            "value": min_clicks,
            "compliant": bool(min_clicks >= 1),
            "note": "Every aggregated interaction record must have >=1 click.",
        }

    # Submitted score: should be >= 0 (nulls are filled to 0 by the pipeline)
    sub_df = edges["submitted"]
    if "score" in sub_df.columns:
        min_score = float(sub_df["score"].min())
        max_score = float(sub_df["score"].max())
        temporal["checks"]["submitted_score_range"] = {
            "min": round(min_score, 2),
            "max": round(max_score, 2),
            "compliant": bool(min_score >= 0.0),
            "note": "Null scores are filled to 0 by the pipeline.",
        }

    all_temporal_compliant = all(
        v.get("compliant", True)
        for v in temporal["checks"].values()
        if isinstance(v, dict)
    )
    temporal["all_compliant"] = all_temporal_compliant

    # ── 10. Duplicate enrollments ─────────────────────────────────────────────
    dup_enrollments = int(
        enrollments.duplicated(["id_student", "code_module", "code_presentation"]).sum()
    )

    # ── Assemble full report ──────────────────────────────────────────────────
    report = {
        "week":               week,
        "runtime":            runtime_stats,
        "node_counts":        node_counts,
        "edge_counts":        edge_counts,
        "enrollment_count":   enrollment_count,
        "duplicate_nodes":    dup_nodes,
        "duplicate_edges":    dup_edges,
        "duplicate_enrollments": dup_enrollments,
        "dangling_edges":     dangling_edges,
        "data_quality":       data_quality,
        "label_distribution": {
            "overall":            label_overall,
            "by_course_presentation": label_by_cp,
        },
        "temporal_compliance": temporal,
    }

    # ── Write JSON output ────────────────────────────────────────────────────
    validation_dir.mkdir(parents=True, exist_ok=True)
    json_path = validation_dir / f"{prefix}_validation.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation JSON saved to {json_path}")

    # ── Write human-readable summary ─────────────────────────────────────────
    summary_lines = _build_summary(report)
    summary_path = validation_dir / f"{prefix}_validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Validation summary saved to {summary_path}")

    # Print to stdout as well
    print()
    print("\n".join(summary_lines))

    return report


# ---------------------------------------------------------------------------
# Human-readable summary builder
# ---------------------------------------------------------------------------

def _build_summary(r: dict) -> list:
    """Build a concise, human-readable validation summary."""
    lines = []
    w = r["week"]
    lines += [
        "=" * 68,
        f"  OULAD Graph Validation Summary — Week {w}",
        "=" * 68,
        "",
    ]

    # Runtime
    rt = r.get("runtime", {})
    if rt:
        lines.append(
            f"Construction runtime : {rt.get('elapsed_seconds', 'n/a')} s"
            f"  |  Peak memory : {rt.get('peak_memory_mb', 'n/a')} MB"
        )
        lines.append("")

    # Counts
    lines.append("── Node counts ──────────────────────────────────────────────────")
    for ntype, cnt in r["node_counts"].items():
        lines.append(f"  {ntype:<25} {cnt:>10,}")
    lines.append("")
    lines.append("── Edge counts ──────────────────────────────────────────────────")
    for etype, cnt in r["edge_counts"].items():
        lines.append(f"  {etype:<25} {cnt:>10,}")
    lines.append(f"  {'enrollments':<25} {r['enrollment_count']:>10,}")
    lines.append("")

    # Duplicates
    dup_node_total = sum(r["duplicate_nodes"].values())
    dup_edge_total = sum(r["duplicate_edges"].values())
    lines.append("── Duplicates ───────────────────────────────────────────────────")
    lines.append(f"  Duplicate nodes (all types) : {dup_node_total}")
    lines.append(f"  Duplicate edges (all types) : {dup_edge_total}")
    lines.append(f"  Duplicate enrollments       : {r['duplicate_enrollments']}")
    lines.append("")

    # Dangling edges
    lines.append("── Dangling edges ───────────────────────────────────────────────")
    all_clear = True
    for etype, d in r["dangling_edges"].items():
        src_ok = d["dangling_src"] == 0
        dst_ok = d["dangling_dst"] == 0
        flag = "" if (src_ok and dst_ok) else " ⚠"
        lines.append(
            f"  {etype:<25} src={d['dangling_src']}  dst={d['dangling_dst']}{flag}"
        )
        if not (src_ok and dst_ok):
            all_clear = False
    if all_clear:
        lines.append("  All edge endpoints resolve to known nodes. ✓")
    lines.append("")

    # Data quality
    lines.append("── Data quality (null / NaN / Inf) ──────────────────────────────")
    any_issues = False
    for artifact, dq in r["data_quality"].items():
        total = dq["total_null_inf"]
        flag = ""
        if total > 0:
            any_issues = True
            by_col = dq["by_column"]
            detail = ", ".join(f"{c}: null={v['null']} inf={v['inf']}"
                               for c, v in by_col.items())
            flag = f" ⚠  [{detail}]"
        lines.append(f"  {artifact:<30} total={total:>6}{flag}")
    if not any_issues:
        lines.append("  No unexpected null/NaN/Inf values found. ✓")
    lines.append("")

    # Label distribution
    ov = r["label_distribution"]["overall"]
    lines.append("── Label distribution ───────────────────────────────────────────")
    lines.append(
        f"  Overall : at-risk={ov['at_risk']:,}  success={ov['success']:,}"
        f"  total={ov['total']:,}  at-risk-rate={ov['at_risk_rate']:.1%}"
    )
    lines.append("  By course presentation:")
    bcp = r["label_distribution"]["by_course_presentation"]
    for cp_key, cv in sorted(bcp.items()):
        lines.append(
            f"    {cp_key:<18} at-risk={cv['at_risk']:>5}  "
            f"success={cv['success']:>5}  "
            f"rate={cv['at_risk_rate']:.1%}"
        )
    lines.append("")

    # Temporal compliance
    tc = r["temporal_compliance"]
    lines.append("── Temporal-cutoff compliance ───────────────────────────────────")
    lines.append(f"  Window (days): {tc['window_days']}")
    for check_name, cv in tc["checks"].items():
        ok = cv.get("compliant", True)
        flag = "✓" if ok else "⚠"
        if "value" in cv:
            lines.append(
                f"  [{flag}] {check_name:<45} value={cv['value']} "
                f"(threshold={cv.get('threshold', 'n/a')})"
            )
        elif "min" in cv:
            lines.append(
                f"  [{flag}] {check_name:<45} min={cv['min']} max={cv['max']}"
            )
    overall_flag = "✓" if tc["all_compliant"] else "⚠ VIOLATION DETECTED"
    lines.append(f"  Overall temporal compliance: {overall_flag}")
    lines.append("")

    lines.append("=" * 68)
    return lines


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Run graph validation and statistics reporting for a saved week."
    )
    p.add_argument("--week", type=int, default=8, choices=[2, 4, 6, 8])
    p.add_argument("--artifacts-dir", type=str, default=None)
    p.add_argument("--validation-dir", type=str, default=None)
    args = p.parse_args()

    ad = Path(args.artifacts_dir) if args.artifacts_dir else GRAPH_ARTIFACTS_DIR
    vd = Path(args.validation_dir) if args.validation_dir else GRAPH_VALIDATION_DIR

    run_validation(week=args.week, artifacts_dir=ad, validation_dir=vd)


if __name__ == "__main__":
    main()
