"""
OULAD Graph Pipeline — Reusable staged API
==========================================
Provides explicit, composable stages for building a leakage-safe
enrollment-centric heterogeneous graph from raw OULAD tables.

Design principles (from docs/GRAPH_IMPLEMENTATION_EXECUTION_PLAN.md):

* Each stage is a plain function — no hidden state coupling data loading,
  graph construction, labels, and training the way the original
  GraphConstructor did.
* The supervised unit is the *enrollment* (id_student, code_module,
  code_presentation), not the student node, so one student can have
  different outcomes across courses without label ambiguity.
* Evaluation split creation is fully *external* to this module.  Callers
  supply or request splits after the enrollment supervision table is built;
  the graph artifact itself carries no train/val/test masks.
* Temporal filtering reuses oulad_data.filter_window(), which gates
  VLE interactions on interaction date and assessment submissions on
  assessment *due date* (not submission date) — consistent with
  docs/LEAKAGE_PREVENTION.md and the tabular baseline.
* All on-disk outputs land under results/graph/ sub-directories that are
  declared in config.py.

Typical call sequence for a Week 8 pipeline run:

    raw      = load_raw_tables()
    filtered = apply_window_cutoff(raw, window_days=56)
    nodes    = build_node_tables(filtered)
    edges    = build_edge_tables(filtered, nodes)
    enrolls  = build_enrollment_supervision(filtered)
    issues   = validate_graph_integrity(nodes, edges, enrolls)
    artifacts = materialize_graph_artifacts(nodes, edges, enrolls,
                                            week=8, save_dir=GRAPH_ARTIFACTS_DIR)
"""

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    GRAPH_ARTIFACTS_DIR,
    GRAPH_VALIDATION_DIR,
    PREDICTION_WINDOWS,
    LABEL_MAPPING,
)
from oulad_data import load_oulad_data, load_supplementary_tables, filter_window


# ---------------------------------------------------------------------------
# Stage 1 — Load raw tables
# ---------------------------------------------------------------------------

def load_raw_tables(data_dir=None) -> Dict[str, pd.DataFrame]:
    """
    Load all raw OULAD tables required by the graph pipeline.

    Returns a dict with keys:
        student_info, student_vle, student_assess, assessments,
        vle, courses, student_registration
    """
    student_info, student_vle, student_assess, assessments = load_oulad_data(data_dir)
    vle, courses, student_registration = load_supplementary_tables(data_dir)

    return {
        "student_info": student_info,
        "student_vle": student_vle,
        "student_assess": student_assess,
        "assessments": assessments,
        "vle": vle,
        "courses": courses,
        "student_registration": student_registration,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Apply window cutoff
# ---------------------------------------------------------------------------

def apply_window_cutoff(
    raw: Dict[str, pd.DataFrame], window_days: int
) -> Dict[str, pd.DataFrame]:
    """
    Return a shallow-copy of *raw* with time-dependent tables filtered to
    *window_days* (inclusive) from the start of each course presentation.

    VLE interactions: date <= window_days
    Assessment submissions: assessment due date <= window_days
        (availability is driven by due date, not submission date)

    Args:
        raw:         Dict produced by load_raw_tables().
        window_days: Temporal cutoff in days (e.g. 56 for Week 8).

    Returns:
        A new dict with the same keys as *raw*; static tables are shared
        references, filtered tables are new DataFrames.
    """
    filtered = dict(raw)  # shallow copy; static tables shared

    vle_w, assess_w = filter_window(
        raw["student_vle"],
        raw["student_assess"],
        raw["assessments"],
        window_days,
    )
    filtered["student_vle"] = vle_w
    filtered["student_assess"] = assess_w
    filtered["assessments"] = raw["assessments"][
        raw["assessments"]["date"] <= window_days
    ].copy()

    return filtered


# ---------------------------------------------------------------------------
# Stage 3 — Build node tables
# ---------------------------------------------------------------------------

def build_node_tables(filtered: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Construct one DataFrame per node type using only features available by
    the prediction point (no target-derived statistics).

    Node types:
        student          — demographic attributes from studentInfo
        course_presentation — static metadata from courses
        assessment       — structural metadata from assessments (filtered)
        vle_resource     — VLE activity metadata

    Returns:
        Dict with keys: 'student', 'course_presentation', 'assessment',
                        'vle_resource'
        Each DataFrame has an integer 'node_idx' column as the graph index.
    """
    nodes: Dict[str, pd.DataFrame] = {}

    # --- student nodes ---
    student_cols = [
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
    ]
    s = filtered["student_info"][student_cols].drop_duplicates("id_student").copy()
    s = s.reset_index(drop=True)
    s["node_idx"] = s.index
    nodes["student"] = s

    # --- course_presentation nodes ---
    cp_cols = ["code_module", "code_presentation", "module_presentation_length"]
    cp = filtered["courses"][cp_cols].drop_duplicates(
        ["code_module", "code_presentation"]
    ).copy()
    cp = cp.reset_index(drop=True)
    cp["node_idx"] = cp.index
    nodes["course_presentation"] = cp

    # --- assessment nodes (only those with due date <= window) ---
    assess_meta = filtered["assessments"][
        ["id_assessment", "code_module", "code_presentation", "assessment_type",
         "weight", "date"]
    ].drop_duplicates("id_assessment").copy()
    assess_meta = assess_meta.reset_index(drop=True)
    assess_meta["node_idx"] = assess_meta.index
    nodes["assessment"] = assess_meta

    # --- vle_resource nodes ---
    vle_cols = ["id_site", "code_module", "code_presentation", "activity_type",
                "week_from", "week_to"]
    v = filtered["vle"][vle_cols].drop_duplicates("id_site").copy()
    v = v.reset_index(drop=True)
    v["node_idx"] = v.index
    nodes["vle_resource"] = v

    return nodes


# ---------------------------------------------------------------------------
# Stage 4 — Build edge tables
# ---------------------------------------------------------------------------

def build_edge_tables(
    filtered: Dict[str, pd.DataFrame],
    nodes: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Construct one DataFrame per edge type as (src_node_idx, dst_node_idx)
    pairs, with optional edge attributes.

    Structural relations:
        enrolled_in        student -> course_presentation
        contains_assess    course_presentation -> assessment
        has_resource       course_presentation -> vle_resource
        submitted          student -> assessment   (enrollment-scoped)
        interacted_with    student -> vle_resource (aggregated per
                           enrollment-resource pair; includes click stats)

    All submission and interaction edges are matched on
    (id_student, code_module, code_presentation) so multi-course students do
    not leak activity across presentations.

    Returns:
        Dict mapping edge-type name -> DataFrame with at minimum
        columns [src, dst].
    """
    edges: Dict[str, pd.DataFrame] = {}

    # ── index lookups ──────────────────────────────────────────────────────
    stu_idx = nodes["student"].set_index("id_student")["node_idx"]
    cp_idx = (
        nodes["course_presentation"]
        .assign(cp_key=lambda d: d["code_module"] + "_" + d["code_presentation"])
        .set_index("cp_key")["node_idx"]
    )
    assess_idx = nodes["assessment"].set_index("id_assessment")["node_idx"]
    vle_idx = nodes["vle_resource"].set_index("id_site")["node_idx"]

    # ── enrolled_in: student -> course_presentation ────────────────────────
    ei = filtered["student_info"][
        ["id_student", "code_module", "code_presentation"]
    ].copy()
    ei["cp_key"] = ei["code_module"] + "_" + ei["code_presentation"]
    ei["src"] = ei["id_student"].map(stu_idx)
    ei["dst"] = ei["cp_key"].map(cp_idx)
    ei = ei.dropna(subset=["src", "dst"])
    ei[["src", "dst"]] = ei[["src", "dst"]].astype(int)
    edges["enrolled_in"] = ei[["src", "dst"]].copy()

    # ── contains_assess: course_presentation -> assessment ─────────────────
    ca = nodes["assessment"][
        ["id_assessment", "code_module", "code_presentation"]
    ].copy()
    ca["cp_key"] = ca["code_module"] + "_" + ca["code_presentation"]
    ca["src"] = ca["cp_key"].map(cp_idx)
    ca["dst"] = ca["id_assessment"].map(assess_idx)
    ca = ca.dropna(subset=["src", "dst"])
    ca[["src", "dst"]] = ca[["src", "dst"]].astype(int)
    edges["contains_assess"] = ca[["src", "dst"]].copy()

    # ── has_resource: course_presentation -> vle_resource ─────────────────
    hr = filtered["vle"][
        ["id_site", "code_module", "code_presentation"]
    ].drop_duplicates().copy()
    hr["cp_key"] = hr["code_module"] + "_" + hr["code_presentation"]
    hr["src"] = hr["cp_key"].map(cp_idx)
    hr["dst"] = hr["id_site"].map(vle_idx)
    hr = hr.dropna(subset=["src", "dst"])
    hr[["src", "dst"]] = hr[["src", "dst"]].astype(int)
    edges["has_resource"] = hr[["src", "dst"]].copy()

    # ── submitted: student -> assessment (enrollment-scoped) ───────────────
    sub = filtered["student_assess"][
        ["id_student", "id_assessment", "score", "code_module", "code_presentation"]
    ].copy()
    sub["src"] = sub["id_student"].map(stu_idx)
    sub["dst"] = sub["id_assessment"].map(assess_idx)
    sub = sub.dropna(subset=["src", "dst"])
    sub[["src", "dst"]] = sub[["src", "dst"]].astype(int)
    sub["score"] = sub["score"].fillna(0.0).astype(float)
    edges["submitted"] = sub[["src", "dst", "score"]].copy()

    # ── interacted_with: student -> vle_resource (aggregated) ─────────────
    # Aggregate per (id_student, id_site, code_module, code_presentation)
    # to avoid multi-edge explosion and attach lightweight click statistics.
    sv = filtered["student_vle"][
        ["id_student", "id_site", "code_module", "code_presentation", "date",
         "sum_click"]
    ].copy()
    agg = (
        sv.groupby(
            ["id_student", "id_site", "code_module", "code_presentation"],
            as_index=False,
        )
        .agg(
            total_clicks=("sum_click", "sum"),
            n_interactions=("sum_click", "count"),
            first_day=("date", "min"),
            last_day=("date", "max"),
            active_days=("date", "nunique"),
        )
    )
    agg["src"] = agg["id_student"].map(stu_idx)
    agg["dst"] = agg["id_site"].map(vle_idx)
    agg = agg.dropna(subset=["src", "dst"])
    agg[["src", "dst"]] = agg[["src", "dst"]].astype(int)
    edge_attr_cols = ["src", "dst", "total_clicks", "n_interactions",
                      "first_day", "last_day", "active_days"]
    edges["interacted_with"] = agg[edge_attr_cols].copy()

    return edges


# ---------------------------------------------------------------------------
# Stage 5 — Build enrollment supervision table
# ---------------------------------------------------------------------------

def build_enrollment_supervision(
    filtered: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the enrollment-level supervised artifact: one row per unique
    (id_student, code_module, code_presentation) from studentInfo.

    Columns in the returned DataFrame:
        id_student, code_module, code_presentation,
        final_result, target  (binary: 1=at-risk, 0=success)

    Labels and split assignment remain here — this table is the authoritative
    supervised unit for both random-student and LCPO evaluation.  No internal
    train/val/test masks are created here; callers create splits externally.

    Returns:
        DataFrame with one row per enrollment.
    """
    enroll_cols = [
        "id_student", "code_module", "code_presentation", "final_result", "target"
    ]
    enrollments = (
        filtered["student_info"][enroll_cols]
        .drop_duplicates(["id_student", "code_module", "code_presentation"])
        .copy()
        .reset_index(drop=True)
    )
    return enrollments


# ---------------------------------------------------------------------------
# Stage 6 — Validate graph integrity
# ---------------------------------------------------------------------------

def validate_graph_integrity(
    nodes: Dict[str, pd.DataFrame],
    edges: Dict[str, pd.DataFrame],
    enrollments: pd.DataFrame,
) -> Dict[str, object]:
    """
    Run lightweight pre-training integrity checks.  Returns a dict of
    findings; a summary is printed to stdout.  Full validation with temporal
    compliance reporting is implemented in graph_validation.py (subtask 4).

    Checks performed here:
        * Duplicate node rows per type
        * Duplicate edge (src, dst) pairs per type
        * Dangling edges (src or dst not in the corresponding node table)
        * Duplicate enrollment supervision records
        * Missing / null values in node tables
        * Enrollment label distribution

    Returns:
        Dict with keys matching check names; values are counts or DataFrames
        suitable for downstream reporting.
    """
    issues: Dict[str, object] = {}

    # ── duplicate nodes ────────────────────────────────────────────────────
    id_cols = {
        "student": "id_student",
        "course_presentation": ["code_module", "code_presentation"],
        "assessment": "id_assessment",
        "vle_resource": "id_site",
    }
    for ntype, key in id_cols.items():
        dup = nodes[ntype].duplicated(subset=key).sum()
        issues[f"dup_nodes_{ntype}"] = int(dup)

    # ── duplicate edges ────────────────────────────────────────────────────
    for etype, edf in edges.items():
        dup = edf.duplicated(subset=["src", "dst"]).sum()
        issues[f"dup_edges_{etype}"] = int(dup)

    # ── dangling edges (spot-check enrolled_in) ────────────────────────────
    valid_stu = set(nodes["student"]["node_idx"].values)
    valid_cp = set(nodes["course_presentation"]["node_idx"].values)
    ei = edges.get("enrolled_in", pd.DataFrame(columns=["src", "dst"]))
    dangling_src = (~ei["src"].isin(valid_stu)).sum()
    dangling_dst = (~ei["dst"].isin(valid_cp)).sum()
    issues["dangling_enrolled_in_src"] = int(dangling_src)
    issues["dangling_enrolled_in_dst"] = int(dangling_dst)

    # ── duplicate enrollments ─────────────────────────────────────────────
    dup_enroll = enrollments.duplicated(
        ["id_student", "code_module", "code_presentation"]
    ).sum()
    issues["dup_enrollments"] = int(dup_enroll)

    # ── missing values in node feature columns ────────────────────────────
    for ntype, ndf in nodes.items():
        null_count = ndf.isnull().sum().sum()
        issues[f"null_values_{ntype}"] = int(null_count)

    # ── label distribution ────────────────────────────────────────────────
    label_counts = enrollments["target"].value_counts().to_dict()
    issues["label_at_risk"] = int(label_counts.get(1, 0))
    issues["label_success"] = int(label_counts.get(0, 0))
    issues["label_at_risk_rate"] = round(
        label_counts.get(1, 0) / max(len(enrollments), 1), 4
    )

    # ── print summary ─────────────────────────────────────────────────────
    print("Graph integrity check results:")
    for k, v in issues.items():
        flag = " ⚠" if isinstance(v, int) and k.startswith(
            ("dup_", "dangling_", "null_")
        ) and v > 0 else ""
        print(f"  {k}: {v}{flag}")

    return issues


# ---------------------------------------------------------------------------
# Stage 7 — Materialize graph artifacts to disk
# ---------------------------------------------------------------------------

def materialize_graph_artifacts(
    nodes: Dict[str, pd.DataFrame],
    edges: Dict[str, pd.DataFrame],
    enrollments: pd.DataFrame,
    week: int,
    save_dir: Path = None,
) -> Dict[str, Path]:
    """
    Persist graph node tables, edge tables, and the enrollment supervision
    table as Parquet files under *save_dir*.

    All artifacts are named deterministically so later training and
    validation runs do not depend on notebook state.

    Args:
        nodes:       Dict from build_node_tables().
        edges:       Dict from build_edge_tables().
        enrollments: DataFrame from build_enrollment_supervision().
        week:        Prediction week (e.g. 8); used in file names.
        save_dir:    Destination directory.  Defaults to
                     config.GRAPH_ARTIFACTS_DIR.

    Returns:
        Dict mapping artifact name -> Path of the saved file.
    """
    if save_dir is None:
        save_dir = GRAPH_ARTIFACTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    prefix = f"week{week:02d}"

    # Try parquet (pyarrow / fastparquet); fall back to CSV so that the
    # pipeline works without optional parquet dependencies.
    try:
        import pyarrow  # noqa: F401
        _ext, _write = ".parquet", lambda df, p: df.to_parquet(p, index=False)
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            _ext, _write = ".parquet", lambda df, p: df.to_parquet(p, index=False)
        except ImportError:
            _ext, _write = ".csv", lambda df, p: df.to_csv(p, index=False)

    for ntype, ndf in nodes.items():
        path = save_dir / f"{prefix}_nodes_{ntype}{_ext}"
        _write(ndf, path)
        saved[f"nodes_{ntype}"] = path

    for etype, edf in edges.items():
        path = save_dir / f"{prefix}_edges_{etype}{_ext}"
        _write(edf, path)
        saved[f"edges_{etype}"] = path

    enroll_path = save_dir / f"{prefix}_enrollments{_ext}"
    _write(enrollments, enroll_path)
    saved["enrollments"] = enroll_path

    print(f"Saved {len(saved)} artifacts to {save_dir}/")
    for name, path in saved.items():
        print(f"  {name}: {path.name}")

    return saved


# ---------------------------------------------------------------------------
# Convenience: run full pipeline for a single week
# ---------------------------------------------------------------------------

def run_pipeline(week: int = 8, data_dir=None, save_dir=None) -> Dict[str, object]:
    """
    Execute all pipeline stages for a given prediction week.

    Args:
        week:     Prediction week (2, 4, 6, or 8).
        data_dir: Raw-data directory; defaults to config.DATA_DIR.
        save_dir: Output directory; defaults to config.GRAPH_ARTIFACTS_DIR.

    Returns:
        Dict with keys: raw, filtered, nodes, edges, enrollments,
                        integrity, artifacts, elapsed_seconds, peak_memory_mb.
    """
    window_days = PREDICTION_WINDOWS.get(f"week_{week}", week * 7)

    tracemalloc.start()
    t0 = time.perf_counter()

    raw = load_raw_tables(data_dir)
    filtered = apply_window_cutoff(raw, window_days)
    nodes = build_node_tables(filtered)
    edges = build_edge_tables(filtered, nodes)
    enrollments = build_enrollment_supervision(filtered)
    integrity = validate_graph_integrity(nodes, edges, enrollments)
    artifacts = materialize_graph_artifacts(nodes, edges, enrollments,
                                            week=week, save_dir=save_dir)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = round(peak_bytes / 1024 / 1024, 1)

    print(f"\nPipeline complete — {elapsed:.1f}s, peak memory {peak_mb} MB")

    return {
        "raw": raw,
        "filtered": filtered,
        "nodes": nodes,
        "edges": edges,
        "enrollments": enrollments,
        "integrity": integrity,
        "artifacts": artifacts,
        "elapsed_seconds": round(elapsed, 2),
        "peak_memory_mb": peak_mb,
    }
