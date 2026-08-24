"""Course-level diagnostic analysis for LCPO folds.

Computes six explanatory factors per fold and correlates each with GNN AUROC,
so reviewers can distinguish model instability from genuine course difficulty.

Usage
-----
    python src/course_diagnostics.py --week 8

Output
------
    results/graph/course_diagnostics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "graph" / "artifacts"
GRAPH_DIR = PROJECT_ROOT / "results" / "graph"
EVAL_DIR = PROJECT_ROOT / "results" / "graph" / "evaluation"

WEEK_TAG = {2: "week02", 4: "week04", 6: "week06", 8: "week08"}


def _parquet(week_tag: str, name: str) -> Path:
    return ARTIFACTS_DIR / f"{week_tag}_{name}.parquet"


def _build_sample_size(folds: pd.DataFrame) -> pd.DataFrame:
    """Factor 1 — n_test from LCPO fold definitions."""
    return folds[
        ["fold_idx", "held_out_module", "held_out_presentation", "n_test"]
    ].copy()


def _build_at_risk_rate(enrollments: pd.DataFrame) -> pd.Series:
    """Factor 2 — at-risk rate (target mean) per (code_module, code_presentation)."""
    return (
        enrollments.groupby(["code_module", "code_presentation"])["target"]
        .mean()
        .rename("at_risk_rate")
    )


def _build_assessment_count(week_tag: str) -> pd.Series:
    """Factor 3 — number of assessments per (code_module, code_presentation)."""
    edges = pd.read_parquet(_parquet(week_tag, "edges_contains_assess"))
    cp_nodes = pd.read_parquet(_parquet(week_tag, "nodes_course_presentation"))

    # edges.src = course_presentation node_idx, edges.dst = assessment node_idx
    assess_counts = (
        edges.groupby("src")["dst"]
        .count()
        .rename("n_assessments")
        .reset_index()
        .rename(columns={"src": "node_idx"})
    )
    merged = cp_nodes[["code_module", "code_presentation", "node_idx"]].merge(
        assess_counts, on="node_idx", how="left"
    )
    merged["n_assessments"] = merged["n_assessments"].fillna(0).astype(int)
    return merged.set_index(["code_module", "code_presentation"])["n_assessments"]


def _build_vle_density(week_tag: str, enrollments: pd.DataFrame) -> pd.Series:
    """Factor 4 — mean total_clicks per enrolled student per (code_module, code_presentation).

    Join path:
      enrollments (id_student, code_module, code_presentation)
      nodes_student (id_student, node_idx)          → student_node_idx
      edges_interacted_with (src=student_node_idx, total_clicks)
    Sum clicks per student; average over students in each CP.
    """
    student_nodes = pd.read_parquet(_parquet(week_tag, "nodes_student"))[
        ["id_student", "node_idx"]
    ].rename(columns={"node_idx": "student_node_idx"})

    iw = pd.read_parquet(_parquet(week_tag, "edges_interacted_with"))[
        ["src", "total_clicks"]
    ].rename(columns={"src": "student_node_idx"})

    # Sum clicks across all VLE resources per student
    clicks_per_student = (
        iw.groupby("student_node_idx")["total_clicks"].sum().rename("total_clicks")
    )

    # Map student id → node_idx, then attach click totals (NaN if no VLE activity)
    enrol_with_idx = enrollments[
        ["id_student", "code_module", "code_presentation"]
    ].merge(student_nodes, on="id_student", how="left")

    enrol_with_clicks = enrol_with_idx.merge(
        clicks_per_student.reset_index(),
        on="student_node_idx",
        how="left",
    )
    enrol_with_clicks["total_clicks"] = enrol_with_clicks["total_clicks"].fillna(0)

    density = (
        enrol_with_clicks.groupby(["code_module", "code_presentation"])[
            "total_clicks"
        ]
        .mean()
        .rename("mean_clicks_per_student")
    )
    return density


def _build_student_overlap(enrollments: pd.DataFrame) -> pd.Series:
    """Factor 5 — fraction of held-out CP students that also appear in another
    presentation of the same module.

    Returns a Series indexed by (code_module, code_presentation).
    """
    # Build per-CP student sets
    cp_students: dict[tuple[str, str], set[int]] = {}
    for (mod, pres), grp in enrollments.groupby(
        ["code_module", "code_presentation"]
    ):
        cp_students[(mod, pres)] = set(grp["id_student"])

    # For each CP, compute overlap with union of same-module other CPs
    records = []
    for (mod, pres), students in cp_students.items():
        same_module_others = set()
        for (m, p), s in cp_students.items():
            if m == mod and p != pres:
                same_module_others |= s
        if len(students) == 0:
            overlap = float("nan")
        else:
            overlap = len(students & same_module_others) / len(students)
        records.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "student_overlap_rate": overlap,
            }
        )

    result = pd.DataFrame(records).set_index(["code_module", "code_presentation"])[
        "student_overlap_rate"
    ]
    return result


def _build_gnn_auroc(week: int) -> pd.DataFrame:
    """Factor 6a — GNN AUROC mean and std per fold from lcpo_results.csv.

    Groups by (week, fold_idx, held_out_module, held_out_presentation) across
    model_seed replicates.  Falls back gracefully to whatever rows are present.
    """
    lcpo_path = GRAPH_DIR / "lcpo_results.csv"
    if not lcpo_path.exists():
        return pd.DataFrame(
            columns=[
                "fold_idx",
                "held_out_module",
                "held_out_presentation",
                "gnn_auroc_mean",
                "gnn_auroc_std",
            ]
        )

    results = pd.read_csv(lcpo_path)
    results = results[results["week"] == week]
    if results.empty:
        return pd.DataFrame(
            columns=[
                "fold_idx",
                "held_out_module",
                "held_out_presentation",
                "gnn_auroc_mean",
                "gnn_auroc_std",
            ]
        )

    grp = (
        results.groupby(
            ["fold_idx", "held_out_module", "held_out_presentation"]
        )["auroc"]
        .agg(gnn_auroc_mean="mean", gnn_auroc_std="std")
        .reset_index()
    )
    # std is NaN when only one seed is present — fill with 0 to signal no variance
    grp["gnn_auroc_std"] = grp["gnn_auroc_std"].fillna(0.0)
    return grp


def _build_lgbm_auroc() -> pd.DataFrame:
    """Factor 6b — LightGBM AUROC per fold from comparison_results.csv."""
    comp_path = GRAPH_DIR / "comparison_results.csv"
    if not comp_path.exists():
        return pd.DataFrame(
            columns=[
                "held_out_module",
                "held_out_presentation",
                "lgbm_auroc",
            ]
        )

    comp = pd.read_csv(comp_path)
    lcpo_lgbm = comp[
        (comp["split_type"] == "lcpo") & (comp["model"] == "LightGBM")
    ].copy()

    if lcpo_lgbm.empty:
        return pd.DataFrame(
            columns=[
                "held_out_module",
                "held_out_presentation",
                "lgbm_auroc",
            ]
        )

    # Average over any replicate seeds that may be present
    lgbm_auroc = (
        lcpo_lgbm.groupby(["held_out_module", "held_out_presentation"])["auroc"]
        .mean()
        .rename("lgbm_auroc")
        .reset_index()
    )
    return lgbm_auroc


def _build_distribution_shift(
    week_tag: str,
    folds: pd.DataFrame,
    enrollments: pd.DataFrame,
) -> Optional[pd.Series]:
    """Optional factor — mean cosine distance of held-out vs training student
    feature vectors.

    Returns None if torch is unavailable or node feature parquets are missing.
    Requires the student node feature matrix to be reconstructable from the
    parquet (one-hot encode categorical columns).
    """
    try:
        import torch  # noqa: F401 — availability check only
    except ImportError:
        return None

    student_path = _parquet(week_tag, "nodes_student")
    if not student_path.exists():
        return None

    student_nodes = pd.read_parquet(student_path)

    # One-hot encode categorical columns to build feature matrix
    cat_cols = [c for c in student_nodes.columns if c not in ("id_student", "node_idx")]
    features = pd.get_dummies(student_nodes[cat_cols]).values.astype(np.float32)

    # Normalise rows to unit length for cosine distance (dist = 1 - cosine_sim)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features_normed = features / norms  # shape: (n_students, n_features)

    # Map student id_student → row index in features matrix
    id_to_row = dict(zip(student_nodes["id_student"], student_nodes.index))

    # For each fold build the held-out student set and training student set
    records = []
    for _, fold in folds.iterrows():
        mod, pres = fold["held_out_module"], fold["held_out_presentation"]

        held_ids = set(
            enrollments.loc[
                (enrollments["code_module"] == mod)
                & (enrollments["code_presentation"] == pres),
                "id_student",
            ]
        )
        train_ids = set(enrollments["id_student"]) - held_ids

        held_rows = [id_to_row[i] for i in held_ids if i in id_to_row]
        train_rows = [id_to_row[i] for i in train_ids if i in id_to_row]

        if not held_rows or not train_rows:
            dist_shift = float("nan")
        else:
            held_mat = features_normed[held_rows]  # (h, d)
            train_mat = features_normed[train_rows]  # (t, d)
            # Mean held-out vector vs mean training vector
            mean_held = held_mat.mean(axis=0)
            mean_train = train_mat.mean(axis=0)
            cosine_sim = float(np.dot(mean_held, mean_train))
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            dist_shift = 1.0 - cosine_sim

        records.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "dist_shift": dist_shift,
            }
        )

    result = pd.DataFrame(records).set_index(["code_module", "code_presentation"])[
        "dist_shift"
    ]
    return result


def build_course_diagnostics(week: int = 8) -> pd.DataFrame:
    """Compute six explanatory factors per LCPO fold for the given prediction window.

    Parameters
    ----------
    week : int
        Prediction window week (2, 4, 6, or 8).

    Returns
    -------
    pd.DataFrame
        One row per LCPO fold.  Columns:
        held_out_module, held_out_presentation,
        n_test, at_risk_rate, n_assessments, mean_clicks_per_student,
        student_overlap_rate, gnn_auroc_mean, gnn_auroc_std, lgbm_auroc,
        auroc_delta, dist_shift (optional — NaN if torch unavailable).
    """
    tag = WEEK_TAG[week]
    folds_path = (
        EVAL_DIR / f"week{week:02d}" / "splits" / f"week{week:02d}_lcpo_folds.csv"
    )

    # ── Base: fold definitions ────────────────────────────────────────────────
    folds = pd.read_csv(folds_path)
    diag = _build_sample_size(folds)  # fold_idx, held_out_module, held_out_presentation, n_test

    # ── Load enrollments (shared across multiple factors) ────────────────────
    enrollments = pd.read_parquet(_parquet(tag, "enrollments"))

    # ── Factor 2: at-risk rate ────────────────────────────────────────────────
    at_risk = _build_at_risk_rate(enrollments)
    diag = diag.merge(
        at_risk.reset_index(),
        left_on=["held_out_module", "held_out_presentation"],
        right_on=["code_module", "code_presentation"],
        how="left",
    ).drop(columns=["code_module", "code_presentation"])

    # ── Factor 3: assessment count ────────────────────────────────────────────
    n_assess = _build_assessment_count(tag)
    diag = diag.merge(
        n_assess.reset_index(),
        left_on=["held_out_module", "held_out_presentation"],
        right_on=["code_module", "code_presentation"],
        how="left",
    ).drop(columns=["code_module", "code_presentation"])

    # ── Factor 4: VLE activity density ───────────────────────────────────────
    vle_density = _build_vle_density(tag, enrollments)
    diag = diag.merge(
        vle_density.reset_index(),
        left_on=["held_out_module", "held_out_presentation"],
        right_on=["code_module", "code_presentation"],
        how="left",
    ).drop(columns=["code_module", "code_presentation"])

    # ── Factor 5: student overlap ─────────────────────────────────────────────
    overlap = _build_student_overlap(enrollments)
    diag = diag.merge(
        overlap.reset_index(),
        left_on=["held_out_module", "held_out_presentation"],
        right_on=["code_module", "code_presentation"],
        how="left",
    ).drop(columns=["code_module", "code_presentation"])

    # ── Factor 6a: GNN AUROC mean/std ─────────────────────────────────────────
    gnn_auroc = _build_gnn_auroc(week)
    diag = diag.merge(
        gnn_auroc,
        on=["fold_idx", "held_out_module", "held_out_presentation"],
        how="left",
    )

    # ── Factor 6b: LightGBM AUROC ─────────────────────────────────────────────
    lgbm_auroc = _build_lgbm_auroc()
    diag = diag.merge(
        lgbm_auroc,
        on=["held_out_module", "held_out_presentation"],
        how="left",
    )

    # ── AUROC delta ───────────────────────────────────────────────────────────
    diag["auroc_delta"] = diag["gnn_auroc_mean"] - diag["lgbm_auroc"]

    # ── Optional: distribution shift ─────────────────────────────────────────
    dist_shift = _build_distribution_shift(tag, folds, enrollments)
    if dist_shift is not None:
        diag = diag.merge(
            dist_shift.reset_index(),
            left_on=["held_out_module", "held_out_presentation"],
            right_on=["code_module", "code_presentation"],
            how="left",
        ).drop(columns=["code_module", "code_presentation"])
    else:
        diag["dist_shift"] = float("nan")

    # ── Canonical column order ────────────────────────────────────────────────
    ordered = [
        "held_out_module",
        "held_out_presentation",
        "n_test",
        "at_risk_rate",
        "n_assessments",
        "mean_clicks_per_student",
        "student_overlap_rate",
        "gnn_auroc_mean",
        "gnn_auroc_std",
        "lgbm_auroc",
        "auroc_delta",
        "dist_shift",
    ]
    # Keep only columns that exist (guards against missing upstream data)
    diag = diag[[c for c in ordered if c in diag.columns]]

    return diag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build course-level diagnostic CSV for LCPO folds."
    )
    parser.add_argument(
        "--week",
        type=int,
        default=8,
        choices=[2, 4, 6, 8],
        help="Prediction window in weeks (default: 8).",
    )
    args = parser.parse_args()

    out_path = GRAPH_DIR / "course_diagnostics.csv"

    print(f"Building course diagnostics for week {args.week}…")
    diag = build_course_diagnostics(week=args.week)

    diag.to_csv(out_path, index=False)
    print(f"Saved {len(diag)} rows → {out_path}")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
