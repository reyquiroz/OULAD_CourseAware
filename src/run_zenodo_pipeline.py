"""
run_zenodo_pipeline.py — End-to-end pipeline for the KU Leuven Zenodo dataset.

Mirrors the OULAD pipeline structure for Week 8 only (the single most
informative prediction window; extends to earlier weeks if needed).

Usage
-----
    python src/run_zenodo_pipeline.py               # build artifacts + save splits
    python src/run_zenodo_pipeline.py --quick       # 5-epoch GNN smoke test, 2 LCPO folds
    python src/run_zenodo_pipeline.py --skip-gnn    # artifacts + LightGBM only (faster)
    python src/run_zenodo_pipeline.py --week 8      # explicit week (only 8 supported)

Outputs
-------
    results/zenodo/artifacts/
        week08_nodes_student.parquet
        week08_nodes_course_presentation.parquet
        week08_nodes_vle_resource.parquet
        week08_edges_enrolled_in.parquet
        week08_edges_interacted_with.parquet
        week08_edges_has_resource.parquet
        week08_enrollments.parquet
        week08_metadata.json
    results/zenodo/evaluation/week08/splits/
        week08_random_split.parquet
        week08_lcpo_folds.csv
        week08_splits_config.json
    results/zenodo/
        random_student_results.csv
        lcpo_results.csv
        lcpo_summary.csv
        comparison_results.csv

Schema gaps vs. OULAD (see docs/zenodo_dataset_feasibility.md)
--------------------------------------------------------------
# GAP: no assessment nodes / submitted / contains_assess edges
# GAP: no student demographic features (no gender, age_band, imd_band, etc.)
# GAP: timestamps are calendar datetimes — day offsets computed from course_info.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from config import PROJECT_ROOT
from zenodo_data import (
    build_enrollment_supervision_zenodo,
    build_features_zenodo,
    build_oulad_compatible_tables,
    filter_window_zenodo,
    iter_log_activity,
    load_zenodo_tables,
    lcpo_split,
    random_student_split,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ZENODO_RESULTS_DIR = PROJECT_ROOT / "results" / "zenodo"
_ARTIFACTS_DIR = _ZENODO_RESULTS_DIR / "artifacts"
_EVAL_DIR = _ZENODO_RESULTS_DIR / "evaluation"

# Prediction window — 8 weeks = 56 days (same as OULAD Week 8)
WINDOW_DAYS = 56
WEEK = 8


# ---------------------------------------------------------------------------
# Stage A — Build node and edge tables for the reduced Zenodo graph
# ---------------------------------------------------------------------------

def build_zenodo_graph(window_days: int = WINDOW_DAYS) -> Dict:
    """Build the Zenodo heterogeneous graph for a given window cutoff.

    Node types (3 of OULAD's 4):
        student           — USER_ID only; no demographic features
        course_presentation — (year, COURSE_ID); duration from course_info.json
        vle_resource      — CONTENT_ID with CONTENT_TYPE as activity_type

    Edge types (3 of OULAD's 5):
        enrolled_in       — student → course_presentation (no age/credit attrs)
        interacted_with   — student → vle_resource (action count ≈ sum_click)
        has_resource      — course_presentation → vle_resource

    # GAP: assessment nodes, submitted edges, contains_assess edges all absent.
    """
    print(f"Building Zenodo graph (window={window_days} days)...")
    tracemalloc.start()
    t0 = time.time()

    raw = load_zenodo_tables()
    compat = build_oulad_compatible_tables(raw)
    start_dates = raw["start_dates"]

    student_info = compat["student_info"]
    vle = compat["vle"]
    courses = compat["courses"]

    # ── Node tables ────────────────────────────────────────────────────────

    # student nodes — no demographic features available
    # # GAP: gender, region, highest_education, imd_band, disability all absent
    stu = (
        student_info[["id_student"]]
        .drop_duplicates("id_student")
        .reset_index(drop=True)
        .copy()
    )
    stu["node_idx"] = stu.index

    # course_presentation nodes
    cp = courses.drop_duplicates(["code_module", "code_presentation"]).reset_index(drop=True).copy()
    cp["node_idx"] = cp.index

    # vle_resource nodes — one per (CONTENT_ID, year) since content items are
    # course-specific (different year → different teaching materials)
    vle_nodes = (
        vle[["id_site", "code_module", "code_presentation", "activity_type"]]
        .drop_duplicates("id_site")
        .reset_index(drop=True)
        .copy()
    )
    vle_nodes["node_idx"] = vle_nodes.index

    nodes = {
        "student": stu,
        "course_presentation": cp,
        "vle_resource": vle_nodes,
    }

    # ── Index lookups ──────────────────────────────────────────────────────
    stu_idx = stu.set_index("id_student")["node_idx"]
    cp_idx = (
        cp.assign(cp_key=lambda d: d["code_module"] + "_" + d["code_presentation"])
        .set_index("cp_key")["node_idx"]
    )
    vle_idx = vle_nodes.set_index("id_site")["node_idx"]

    # ── enrolled_in edges ─────────────────────────────────────────────────
    # # GAP: no age_band, num_of_prev_attempts, studied_credits attributes
    # Deduplicate enrollment triples first (12 students appear in both GE 1 and GE 2
    # in 2021 after course-name normalisation; keep the first occurrence).
    ei = (
        student_info[["id_student", "code_module", "code_presentation"]]
        .drop_duplicates(["id_student", "code_module", "code_presentation"])
        .copy()
    )
    ei["cp_key"] = ei["code_module"] + "_" + ei["code_presentation"]
    ei["src"] = ei["id_student"].map(stu_idx)
    ei["dst"] = ei["cp_key"].map(cp_idx)
    ei = ei.dropna(subset=["src", "dst"])
    ei[["src", "dst"]] = ei[["src", "dst"]].astype(int)
    edges_enrolled_in = ei[["src", "dst"]].copy()

    # ── has_resource edges ────────────────────────────────────────────────
    hr = vle[["id_site", "code_module", "code_presentation"]].drop_duplicates().copy()
    hr["cp_key"] = hr["code_module"] + "_" + hr["code_presentation"]
    hr["src"] = hr["cp_key"].map(cp_idx)
    hr["dst"] = hr["id_site"].map(vle_idx)
    hr = hr.dropna(subset=["src", "dst"])
    hr[["src", "dst"]] = hr[["src", "dst"]].astype(int)
    edges_has_resource = hr[["src", "dst"]].copy()

    # ── interacted_with edges (built from log_activity) ───────────────────
    # Process one year at a time and aggregate immediately to avoid holding
    # all three log CSVs (~675 MB total) in memory simultaneously.
    print("  Loading log_activity and applying window cutoff (one year at a time)...")
    agg_parts = []
    for yr, log_df in iter_log_activity():
        year_courses = start_dates[start_dates["year"] == yr]["course_id"].unique()
        yr_parts = []
        for course_id in year_courses:
            base_id = course_id.rstrip(" 12").strip()
            try:
                filtered = filter_window_zenodo(
                    log_df, yr, base_id, window_days, start_dates
                )
                yr_parts.append(filtered[
                    ["id_student", "id_site", "code_module",
                     "code_presentation", "sum_click", "day_offset"]
                ])
            except ValueError:
                pass
        del log_df  # free memory before next year

        if yr_parts:
            yr_log = pd.concat(yr_parts, ignore_index=True)
            yr_agg = (
                yr_log.groupby(
                    ["id_student", "id_site", "code_module", "code_presentation"],
                    as_index=False,
                )
                .agg(
                    total_clicks=("sum_click", "sum"),
                    n_interactions=("sum_click", "count"),
                    first_day=("day_offset", "min"),
                    last_day=("day_offset", "max"),
                    active_days=("day_offset", "nunique"),
                )
            )
            agg_parts.append(yr_agg)
            del yr_log
        print(f"    {yr}: done")

    if agg_parts:
        agg = pd.concat(agg_parts, ignore_index=True)
        # Re-aggregate across years (same student may appear in multiple years)
        agg = (
            agg.groupby(
                ["id_student", "id_site", "code_module", "code_presentation"],
                as_index=False,
            )
            .agg(
                total_clicks=("total_clicks", "sum"),
                n_interactions=("n_interactions", "sum"),
                first_day=("first_day", "min"),
                last_day=("last_day", "max"),
                active_days=("active_days", "sum"),
            )
        )
        agg["src"] = agg["id_student"].map(stu_idx)
        agg["dst"] = agg["id_site"].map(vle_idx)
        agg = agg.dropna(subset=["src", "dst"])
        agg[["src", "dst"]] = agg[["src", "dst"]].astype(int)
        edges_interacted_with = agg[
            ["src", "dst", "total_clicks", "n_interactions",
             "first_day", "last_day", "active_days"]
        ].copy()
    else:
        edges_interacted_with = pd.DataFrame(
            columns=["src", "dst", "total_clicks", "n_interactions",
                     "first_day", "last_day", "active_days"]
        )

    edges = {
        "enrolled_in": edges_enrolled_in,
        "has_resource": edges_has_resource,
        "interacted_with": edges_interacted_with,
    }

    # ── Enrollment supervision table ──────────────────────────────────────
    enrollments = build_enrollment_supervision_zenodo(student_info)

    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Done in {elapsed:.1f}s  peak RAM {peak/1e6:.0f} MB")
    _print_graph_stats(nodes, edges, enrollments)

    return {
        "nodes": nodes,
        "edges": edges,
        "enrollments": enrollments,
        "elapsed_seconds": round(elapsed, 2),
        "peak_memory_mb": round(peak / 1e6, 1),
    }


def _print_graph_stats(nodes, edges, enrollments):
    print("  Graph statistics:")
    for ntype, df in nodes.items():
        print(f"    {ntype}: {len(df):,} nodes")
    for etype, df in edges.items():
        print(f"    {etype}: {len(df):,} edges")
    print(f"    enrollments: {len(enrollments):,}")
    at_risk = (enrollments["target"] == 1).mean()
    print(f"    at-risk rate: {at_risk:.1%}")


# ---------------------------------------------------------------------------
# Stage B — Materialize artifacts to results/zenodo/artifacts/
# ---------------------------------------------------------------------------

def materialize_zenodo_artifacts(result: Dict, week: int = WEEK) -> Dict[str, Path]:
    """Write parquet artifacts and metadata JSON to results/zenodo/artifacts/."""
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"week{week:02d}"
    saved = {}

    for ntype, df in result["nodes"].items():
        p = _ARTIFACTS_DIR / f"{prefix}_nodes_{ntype}.parquet"
        df.to_parquet(p, index=False)
        saved[f"nodes_{ntype}"] = p

    for etype, df in result["edges"].items():
        p = _ARTIFACTS_DIR / f"{prefix}_edges_{etype}.parquet"
        df.to_parquet(p, index=False)
        saved[f"edges_{etype}"] = p

    enroll_path = _ARTIFACTS_DIR / f"{prefix}_enrollments.parquet"
    result["enrollments"].to_parquet(enroll_path, index=False)
    saved["enrollments"] = enroll_path

    metadata = {
        "dataset": "zenodo_17087849",
        "institution": "KU Leuven (Tiukhova et al., 2024)",
        "week": week,
        "window_days": week * 7,
        "supervised_unit": "enrollment (id_student, code_module, code_presentation)",
        "label_definition": "PASSED==0 → at-risk (target=1); PASSED==1 → success (target=0)",
        "gaps": [
            "No assessment nodes or submitted/contains_assess edges",
            "No student demographic features",
            "Timestamps are calendar datetimes (not pre-computed day offsets)",
        ],
        "node_counts": {ntype: int(len(df)) for ntype, df in result["nodes"].items()},
        "edge_counts": {etype: int(len(df)) for etype, df in result["edges"].items()},
        "enrollment_count": int(len(result["enrollments"])),
        "label_at_risk_count": int((result["enrollments"]["target"] == 1).sum()),
        "label_success_count": int((result["enrollments"]["target"] == 0).sum()),
        "label_at_risk_rate": round(float((result["enrollments"]["target"] == 1).mean()), 4),
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_memory_mb": result["peak_memory_mb"],
    }
    meta_path = _ARTIFACTS_DIR / f"{prefix}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    saved["metadata"] = meta_path

    print(f"\nArtifacts written to {_ARTIFACTS_DIR}:")
    for name, p in saved.items():
        print(f"  {p.name}")

    return saved


# ---------------------------------------------------------------------------
# Stage C — Save split definitions
# ---------------------------------------------------------------------------

def save_zenodo_splits(week: int = WEEK) -> None:
    """Save random-student and LCPO split definitions for the Zenodo enrollment table."""
    prefix = f"week{week:02d}"
    enroll_path = _ARTIFACTS_DIR / f"{prefix}_enrollments.parquet"
    if not enroll_path.exists():
        raise FileNotFoundError(
            f"Enrollment artifact not found: {enroll_path}\n"
            f"Run build_zenodo_graph() first."
        )

    enrollments = pd.read_parquet(enroll_path)
    split_dir = _EVAL_DIR / f"week{week:02d}" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Random student split (70 / 10 / 20)
    train_mask, val_mask, test_mask = random_student_split(
        enrollments, val_frac=0.1, test_frac=0.2, seed=42
    )
    rs = enrollments.copy()
    rs["is_train"] = train_mask
    rs["is_val"] = val_mask
    rs["is_test"] = test_mask
    rs_path = split_dir / f"{prefix}_random_split.parquet"
    rs.to_parquet(rs_path, index=False)

    # LCPO folds — one per (code_module, code_presentation)
    presentations = (
        enrollments[["code_module", "code_presentation"]]
        .drop_duplicates()
        .sort_values(["code_module", "code_presentation"])
        .itertuples(index=False)
    )
    lcpo_rows = []
    for fold_idx, row in enumerate(presentations):
        tr_mask, te_mask = lcpo_split(enrollments, row.code_module, row.code_presentation)
        lcpo_rows.append({
            "fold_idx": fold_idx,
            "held_out_module": row.code_module,
            "held_out_presentation": row.code_presentation,
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
        })
    lcpo_df = pd.DataFrame(lcpo_rows)
    lcpo_path = split_dir / f"{prefix}_lcpo_folds.csv"
    lcpo_df.to_csv(lcpo_path, index=False)

    config = {
        "dataset": "zenodo_17087849",
        "week": week,
        "enrollment_count": len(enrollments),
        "random_student_split": {
            "seed": 42, "val_frac": 0.1, "test_frac": 0.2,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        },
        "lcpo": {"n_folds": len(lcpo_rows), "folds_file": f"{prefix}_lcpo_folds.csv"},
    }
    cfg_path = split_dir / f"{prefix}_splits_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSplits saved to {split_dir}:")
    print(f"  random: {int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())} "
          f"(train/val/test)")
    print(f"  LCPO: {len(lcpo_rows)} folds")
    for fold_row in lcpo_rows:
        print(f"    fold {fold_row['fold_idx']:02d}: "
              f"{fold_row['held_out_module']} / {fold_row['held_out_presentation']}  "
              f"n_test={fold_row['n_test']}")


# ---------------------------------------------------------------------------
# Stage D — Build GNN-ready HeteroData from Zenodo artifacts
# ---------------------------------------------------------------------------

def _build_zenodo_hetero_data(week: int = WEEK, skip_normalize: bool = False):
    """Load Zenodo artifacts into a PyG HeteroData object.

    Mirrors GraphDataLoader from gnn_model.py but for the reduced Zenodo schema:
        3 node types (student, course_presentation, vle_resource)
        3 edge types + their reverses (enrolled_in, interacted_with, has_resource)

    # GAP: student nodes have no feature attributes — a single constant (1.0)
    #      is used as a placeholder so the GNN can still compute embeddings.
    # GAP: enrolled_in edges have no attribute features (no age_band etc).
    """
    from torch_geometric.data import HeteroData
    from gnn_model import _normalize_numeric_features

    prefix = f"week{week:02d}"

    # Load node tables
    stu = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_nodes_student.parquet")
    cp = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_nodes_course_presentation.parquet")
    vle = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_nodes_vle_resource.parquet")

    # Load edge tables
    ei = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_edges_enrolled_in.parquet")
    iw = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_edges_interacted_with.parquet")
    hr = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_edges_has_resource.parquet")

    # Load labels
    enroll = pd.read_parquet(_ARTIFACTS_DIR / f"{prefix}_enrollments.parquet")

    data = HeteroData()

    # --- Node features ---
    # student: no demographic features available — use constant 1.0 placeholder
    n_stu = len(stu)
    data["student"].x = torch.ones(n_stu, 1, dtype=torch.float32)

    # course_presentation: module_presentation_length (numeric, normalised below)
    cp_len = torch.tensor(
        cp["module_presentation_length"].fillna(0).values, dtype=torch.float32
    ).unsqueeze(1)
    data["course_presentation"].x = cp_len

    # vle_resource: one-hot activity_type
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    act_codes = le.fit_transform(vle["activity_type"].fillna("Unknown").astype(str))
    n_classes = len(le.classes_)
    act_onehot = torch.zeros(len(vle), n_classes, dtype=torch.float32)
    act_onehot[torch.arange(len(vle)), torch.tensor(act_codes)] = 1.0
    data["vle_resource"].x = act_onehot

    # --- enrolled_in edges ---
    ei_src = torch.tensor(ei["src"].values, dtype=torch.long)
    ei_dst = torch.tensor(ei["dst"].values, dtype=torch.long)
    data["student", "enrolled_in", "course_presentation"].edge_index = torch.stack([ei_src, ei_dst])
    # No enrollment-scoped attributes (GAP) — use a constant 1.0 placeholder
    # so the edge attribute projection layer has something to project
    data["student", "enrolled_in", "course_presentation"].edge_attr = torch.ones(
        len(ei), 1, dtype=torch.float32
    )
    # enrollment_idx: maps each enrolled_in edge back to its row in the enrollment table.
    # For Zenodo, enrolled_in edges are built from the deduplicated enrollment table in the
    # same order, so the mapping is 1-to-1 with arange.
    data["student", "enrolled_in", "course_presentation"].enrollment_idx = torch.arange(
        len(ei), dtype=torch.long
    )
    # Attach labels (aligned to enrolled_in edge order = enrollment order)
    data["student", "enrolled_in", "course_presentation"].y = torch.tensor(
        enroll["target"].values, dtype=torch.float32
    )

    # Reverse enrolled_in
    data["course_presentation", "rev_enrolled_in", "student"].edge_index = torch.stack([ei_dst, ei_src])

    # --- interacted_with edges ---
    if len(iw) > 0:
        iw_src = torch.tensor(iw["src"].values, dtype=torch.long)
        iw_dst = torch.tensor(iw["dst"].values, dtype=torch.long)
        data["student", "interacted_with", "vle_resource"].edge_index = torch.stack([iw_src, iw_dst])
        iw_attr_cols = ["total_clicks", "n_interactions", "first_day", "last_day", "active_days"]
        iw_attr = torch.tensor(
            iw[iw_attr_cols].fillna(0).values.astype(float), dtype=torch.float32
        )
        data["student", "interacted_with", "vle_resource"].edge_attr = iw_attr
        data["vle_resource", "rev_interacted_with", "student"].edge_index = torch.stack([iw_dst, iw_src])

    # --- has_resource edges ---
    if len(hr) > 0:
        hr_src = torch.tensor(hr["src"].values, dtype=torch.long)
        hr_dst = torch.tensor(hr["dst"].values, dtype=torch.long)
        data["course_presentation", "has_resource", "vle_resource"].edge_index = torch.stack([hr_src, hr_dst])
        data["vle_resource", "rev_has_resource", "course_presentation"].edge_index = torch.stack([hr_dst, hr_src])

    return data, enroll


# ---------------------------------------------------------------------------
# Stage E — GNN experiments on Zenodo
# ---------------------------------------------------------------------------

def run_zenodo_random_split(
    week: int = WEEK,
    max_epochs: int = 200,
    patience: int = 20,
    seeds: list = None,
) -> list:
    """Run GNN random-student split experiments on Zenodo data."""
    from gnn_model import (
        EnrollmentGNN,
        _normalize_numeric_features,
        build_train_subgraph,
        compute_metrics,
        compute_pos_weight,
        run_training_loop,
        select_threshold,
    )

    if seeds is None:
        seeds = [42]

    data, enroll = _build_zenodo_hetero_data(week, skip_normalize=True)
    rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_s, val_s, test_s = random_student_split(enroll, seed=seed)
        train_mask = torch.tensor(train_s.values, dtype=torch.bool)
        val_mask = torch.tensor(val_s.values, dtype=torch.bool)
        test_mask = torch.tensor(test_s.values, dtype=torch.bool)

        data_norm = _normalize_numeric_features(
            copy.deepcopy(data), train_edge_mask=train_mask
        )
        train_subgraph = build_train_subgraph(data_norm, train_mask)

        train_y = train_subgraph[("student", "enrolled_in", "course_presentation")].y
        pos_weight = compute_pos_weight(None, train_y)

        in_channels_dict = {ntype: data_norm[ntype].x.shape[1]
                            for ntype in data_norm.node_types}
        ei_key = ("student", "enrolled_in", "course_presentation")
        n_ei_attr = data_norm[ei_key].edge_attr.shape[1]
        model = EnrollmentGNN(
            in_channels_dict=in_channels_dict,
            hidden_dim=64,
            n_enrolled_in_attr=n_ei_attr,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_auroc, best_epoch, train_losses, val_aurocs = run_training_loop(
            model, train_subgraph, data_norm, val_mask, optimizer,
            max_epochs=max_epochs, patience=patience, pos_weight=pos_weight,
        )
        print(f"  seed={seed}  best_val_auroc={best_val_auroc:.4f}  epoch={best_epoch}")

        model.eval()
        with torch.no_grad():
            logits = model(data_norm)
        y = data_norm[ei_key].y
        val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
        val_labels = y[val_mask].cpu().numpy()
        threshold = select_threshold(val_probs, val_labels) if (
            val_labels.sum() > 0 and (1 - val_labels).sum() > 0
        ) else 0.5

        test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()
        metrics = compute_metrics(test_probs, test_labels, threshold=threshold)

        rows.append({
            "dataset": "zenodo",
            "week": week,
            "seed": seed,
            "model": "EnrollmentGNN",
            "split": "random_student",
            "loss_weighting": "weighted",
            "pipeline_version": "v2_corrected",
            **metrics,
            "best_val_auroc": best_val_auroc,
            "best_epoch": best_epoch,
            "best_threshold": threshold,
        })

    return rows


def run_zenodo_lcpo(
    week: int = WEEK,
    max_epochs: int = 200,
    patience: int = 50,
    model_seeds: list = None,
    max_folds: int = None,
) -> pd.DataFrame:
    """Run GNN LCPO experiments on Zenodo data."""
    from gnn_model import (
        EnrollmentGNN,
        _normalize_numeric_features,
        build_train_subgraph,
        compute_metrics,
        compute_pos_weight,
        run_training_loop,
    )

    if model_seeds is None:
        model_seeds = [42, 123, 7, 17, 99]

    folds_path = _EVAL_DIR / f"week{week:02d}" / "splits" / f"week{week:02d}_lcpo_folds.csv"
    folds_df = pd.read_csv(folds_path)
    if max_folds is not None:
        folds_df = folds_df.iloc[:max_folds].copy()

    data_base, enroll = _build_zenodo_hetero_data(week, skip_normalize=True)
    records = []

    for _, fold_row in folds_df.iterrows():
        fold_idx = int(fold_row["fold_idx"])
        ho_mod = fold_row["held_out_module"]
        ho_pres = fold_row["held_out_presentation"]

        print(f"  Fold {fold_idx:02d}: {ho_mod}/{ho_pres}", end="  ", flush=True)

        # enroll["code_presentation"] is a string (e.g. "1819"); folds CSV stores it as int
        ho_rows = (
            (enroll["code_module"] == ho_mod)
            & (enroll["code_presentation"] == str(ho_pres))
        )
        test_mask_np = ho_rows.values
        train_all_np = ~test_mask_np

        y_np = enroll["target"].to_numpy().astype(np.int32)
        train_student_ids = enroll.loc[train_all_np, "id_student"].unique()
        rng = np.random.default_rng(fold_idx)
        val_size = max(1, int(0.10 * len(train_student_ids)))
        val_student_ids = rng.choice(train_student_ids, size=val_size, replace=False)
        val_mask_np = train_all_np & enroll["id_student"].isin(val_student_ids).to_numpy()
        train_mask_np = train_all_np & ~val_mask_np

        train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
        test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

        data = _normalize_numeric_features(
            copy.deepcopy(data_base), train_edge_mask=train_mask
        )
        train_subgraph = build_train_subgraph(data, train_mask)
        train_y = train_subgraph[("student", "enrolled_in", "course_presentation")].y
        pos_weight = compute_pos_weight(None, train_y)

        for mseed in model_seeds:
            torch.manual_seed(mseed)
            np.random.seed(mseed)

            in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
            ei_key = ("student", "enrolled_in", "course_presentation")
            n_ei_attr = data[ei_key].edge_attr.shape[1]
            model = EnrollmentGNN(
                in_channels_dict=in_channels_dict,
                hidden_dim=64,
                n_enrolled_in_attr=n_ei_attr,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            best_val_auroc, best_epoch, _, _ = run_training_loop(
                model, train_subgraph, data, val_mask, optimizer,
                max_epochs=max_epochs, patience=patience, pos_weight=pos_weight,
            )

            model.eval()
            with torch.no_grad():
                logits = model(data)
            y = data[ei_key].y
            test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
            test_labels = y[test_mask].cpu().numpy()

            if test_labels.sum() == 0 or (1 - test_labels).sum() == 0:
                print("SKIP (single class in test)")
                break

            metrics = compute_metrics(test_probs, test_labels)
            records.append({
                "dataset": "zenodo",
                "week": week,
                "fold_idx": fold_idx,
                "held_out_module": ho_mod,
                "held_out_presentation": ho_pres,
                "model_seed": mseed,
                "n_train": int(train_mask_np.sum()),
                "n_test": int(test_mask_np.sum()),
                "best_val_auroc": best_val_auroc,
                "best_epoch": best_epoch,
                "pipeline_version": "v2_corrected",
                **metrics,
            })

        if records:
            fold_aurocs = [r["auroc"] for r in records
                           if r["fold_idx"] == fold_idx]
            if fold_aurocs:
                print(f"auroc={np.mean(fold_aurocs):.4f}±{np.std(fold_aurocs):.4f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stage F — LightGBM on Zenodo (VLE-only features)
# ---------------------------------------------------------------------------

def run_zenodo_lgbm(week: int = WEEK, seeds: list = None) -> dict:
    """Run LightGBM random-split and LCPO on Zenodo using VLE-only features."""
    from lightgbm import LGBMClassifier
    from gnn_model import select_threshold
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, balanced_accuracy_score,
    )

    if seeds is None:
        seeds = [42]

    raw = load_zenodo_tables()
    compat = build_oulad_compatible_tables(raw)
    student_info = compat["student_info"]
    start_dates = raw["start_dates"]

    print("  Building Zenodo VLE features (window filter)...")
    chunks = []
    for yr, log_df in iter_log_activity():
        year_courses = start_dates[start_dates["year"] == yr]["course_id"].unique()
        for course_id in year_courses:
            base_id = course_id.rstrip(" 12").strip()
            try:
                filtered = filter_window_zenodo(
                    log_df, yr, base_id, week * 7, start_dates
                )
                chunks.append(filtered)
            except ValueError:
                pass
        del log_df  # free memory before loading next year
    log_w = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    del chunks

    df_feats = build_features_zenodo(log_w, student_info)
    del log_w
    enroll_path = _ARTIFACTS_DIR / f"week{week:02d}_enrollments.parquet"
    enrollments = pd.read_parquet(enroll_path)

    # Align feature rows to enrollment ordering (deduplicate first — build_features_zenodo
    # may return duplicate enrollment keys)
    df_feats_dedup = df_feats.drop_duplicates(
        ["id_student", "code_module", "code_presentation"]
    ).reset_index(drop=True)
    enroll_reset = enrollments.reset_index(drop=True)
    merged = enroll_reset[["id_student", "code_module", "code_presentation"]].merge(
        df_feats_dedup[["id_student", "code_module", "code_presentation",
                        "vle_total", "vle_mean", "vle_std"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    X_full = merged[["vle_total", "vle_mean", "vle_std"]].fillna(0).values
    y_full = enrollments["target"].values

    def _metrics(y_true, y_pred, y_proba, threshold):
        return {
            "auroc": float(roc_auc_score(y_true, y_proba)),
            "auprc": float(average_precision_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }

    # --- Random split ---
    random_rows = []
    for seed in seeds:
        train_s, val_s, test_s = random_student_split(enrollments, seed=seed)
        X_tr, y_tr = X_full[train_s.values], y_full[train_s.values]
        X_val, y_val = X_full[val_s.values], y_full[val_s.values]
        X_te, y_te = X_full[test_s.values], y_full[test_s.values]

        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr)
        val_proba = clf.predict_proba(X_val)[:, 1]
        thr = select_threshold(val_proba, y_val) if (
            y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0
        ) else 0.5
        test_proba = clf.predict_proba(X_te)[:, 1]
        test_pred = (test_proba >= thr).astype(int)
        m = _metrics(y_te, test_pred, test_proba, thr)
        random_rows.append({
            "dataset": "zenodo", "week": week, "model": "LightGBM",
            "split_type": "random_student", "fold_or_seed": seed,
            "feature_set": "vle_only", **m,
        })
        print(f"    LGBM random seed={seed}  AUROC={m['auroc']:.4f}")

    # --- LCPO ---
    folds_path = _EVAL_DIR / f"week{week:02d}" / "splits" / f"week{week:02d}_lcpo_folds.csv"
    folds_df = pd.read_csv(folds_path)
    lcpo_rows = []
    for _, fold_row in folds_df.iterrows():
        fold_idx = int(fold_row["fold_idx"])
        ho_mod = fold_row["held_out_module"]
        ho_pres = fold_row["held_out_presentation"]

        is_ho = (
            (enroll_reset["code_module"] == ho_mod) &
            (enroll_reset["code_presentation"] == ho_pres)
        )
        train_all = (~is_ho).values
        test_mask = is_ho.values

        train_stu = enroll_reset.loc[train_all, "id_student"].unique()
        rng = np.random.default_rng(fold_idx)
        val_size = max(1, int(0.10 * len(train_stu)))
        val_stu_ids = set(rng.choice(train_stu, size=val_size, replace=False))
        val_mask = train_all & enroll_reset["id_student"].isin(val_stu_ids).to_numpy()
        train_mask = train_all & ~val_mask

        X_tr, y_tr = X_full[train_mask], y_full[train_mask]
        X_val, y_val = X_full[val_mask], y_full[val_mask]
        X_te, y_te = X_full[test_mask], y_full[test_mask]

        if y_te.sum() == 0 or (len(y_te) - y_te.sum()) == 0:
            continue

        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr)
        val_proba = clf.predict_proba(X_val)[:, 1]
        thr = select_threshold(val_proba, y_val) if (
            y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0
        ) else 0.5
        test_proba = clf.predict_proba(X_te)[:, 1]
        test_pred = (test_proba >= thr).astype(int)
        m = _metrics(y_te, test_pred, test_proba, thr)
        lcpo_rows.append({
            "dataset": "zenodo", "week": week, "model": "LightGBM",
            "split_type": "lcpo", "fold_or_seed": fold_idx,
            "held_out_module": ho_mod, "held_out_presentation": ho_pres,
            "feature_set": "vle_only", **m,
        })

    if lcpo_rows:
        lcpo_aurocs = [r["auroc"] for r in lcpo_rows]
        print(f"    LGBM LCPO ({len(lcpo_rows)} folds)  AUROC={np.mean(lcpo_aurocs):.4f}"
              f"±{np.std(lcpo_aurocs):.4f}")

    return {"random": random_rows, "lcpo": lcpo_rows}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Zenodo KU Leuven pipeline runner")
    p.add_argument("--week", type=int, default=WEEK,
                   help=f"Prediction week in days/7 (default: {WEEK})")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 5 GNN epochs, 2 LCPO folds")
    p.add_argument("--skip-gnn", action="store_true",
                   help="Skip GNN experiments; run LightGBM only")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 17, 99],
                   help="Random seeds for random-split experiments")
    p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 123, 7, 17, 99],
                   help="Model init seeds for LCPO GNN experiments")
    args = p.parse_args()

    week = args.week
    max_epochs = 5 if args.quick else 200
    patience = 3 if args.quick else 20
    lcpo_patience = 3 if args.quick else 50
    max_folds = 2 if args.quick else None

    print("=" * 60)
    print(f"Zenodo pipeline — Week {week}")
    print("=" * 60)

    # A. Build graph
    result = build_zenodo_graph(window_days=week * 7)

    # B. Materialize artifacts
    materialize_zenodo_artifacts(result, week=week)

    # C. Save splits
    save_zenodo_splits(week=week)

    _ZENODO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # D. GNN experiments
    if not args.skip_gnn:
        print(f"\n=== GNN Random-split (seeds={args.seeds}) ===")
        random_rows = run_zenodo_random_split(
            week=week, max_epochs=max_epochs, patience=patience, seeds=args.seeds
        )
        random_df = pd.DataFrame(random_rows)
        rand_path = _ZENODO_RESULTS_DIR / "random_student_results.csv"
        random_df.to_csv(rand_path, index=False)
        print(f"\nGNN random results → {rand_path}")

        print(f"\n=== GNN LCPO (model_seeds={args.model_seeds}) ===")
        lcpo_df = run_zenodo_lcpo(
            week=week, max_epochs=max_epochs, patience=lcpo_patience,
            model_seeds=args.model_seeds, max_folds=max_folds,
        )
        if not lcpo_df.empty:
            lcpo_path = _ZENODO_RESULTS_DIR / "lcpo_results.csv"
            lcpo_df.to_csv(lcpo_path, index=False)
            print(f"\nGNN LCPO results → {lcpo_path}")

            # LCPO summary
            metric_cols = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
            summary_rows = []
            for (fidx, ho_mod, ho_pres), grp in lcpo_df.groupby(
                ["fold_idx", "held_out_module", "held_out_presentation"], sort=False
            ):
                row = {"fold_idx": fidx, "held_out_module": ho_mod,
                       "held_out_presentation": ho_pres}
                for col in metric_cols:
                    row[f"{col}_mean"] = grp[col].mean()
                    row[f"{col}_std"] = grp[col].std()
                summary_rows.append(row)
            summary_df = pd.DataFrame(summary_rows)
            summary_path = _ZENODO_RESULTS_DIR / "lcpo_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"LCPO summary     → {summary_path}")

    # E. LightGBM comparison — run in a separate subprocess to avoid OOM.
    # The graph build (stage A) peaks at ~1.3 GB RAM; PyTorch + LightGBM + log
    # data together exceed available memory in a single process on this machine.
    # run_zenodo_lgbm_only.py imports NO PyTorch and is safe to run immediately after.
    import subprocess
    seeds_str = " ".join(str(s) for s in args.seeds)
    script = Path(__file__).parent / "run_zenodo_lgbm_only.py"
    print(f"\n=== LightGBM VLE-only (delegated to subprocess, seeds={args.seeds}) ===")
    ret = subprocess.run(
        [sys.executable, str(script), "--week", str(week), "--seeds"] + [str(s) for s in args.seeds],
        check=False,
    )
    if ret.returncode != 0:
        print(f"WARNING: LightGBM subprocess exited with code {ret.returncode}")

    # Summary
    comp_path = _ZENODO_RESULTS_DIR / "comparison_results.csv"
    print("\n" + "=" * 60)
    print("ZENODO RESULTS SUMMARY")
    print("=" * 60)
    if comp_path.exists():
        lgbm_df = pd.read_csv(comp_path)
        lgbm_rand = lgbm_df[lgbm_df["split_type"] == "random_student"]
        lgbm_lcpo = lgbm_df[lgbm_df["split_type"] == "lcpo"]
        print(f"LightGBM random AUROC: {lgbm_rand['auroc'].mean():.4f} ± {lgbm_rand['auroc'].std():.4f}")
        if not lgbm_lcpo.empty:
            print(f"LightGBM LCPO   AUROC: {lgbm_lcpo['auroc'].mean():.4f} ± {lgbm_lcpo['auroc'].std():.4f}")
    if not args.skip_gnn and "random_rows" in dir() and random_rows:
        gnn_rand_aurocs = [r["auroc"] for r in random_rows]
        print(f"GNN random      AUROC: {np.mean(gnn_rand_aurocs):.4f} ± {np.std(gnn_rand_aurocs):.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
