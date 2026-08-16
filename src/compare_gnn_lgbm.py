"""
compare_gnn_lgbm.py — Side-by-side GNN vs. LightGBM comparison.

Runs LightGBM under the same split definitions (70/10/20 random-student split
and the 22-fold LCPO protocol) that were used for the GNN experiment, then
prints and saves a Markdown comparison table.

Usage
-----
    python src/compare_gnn_lgbm.py                    # full run (all 22 LCPO folds)
    python src/compare_gnn_lgbm.py --quick            # 2 LCPO folds, quick sanity check
    python src/compare_gnn_lgbm.py --week 8           # explicit week (default: 8)
    python src/compare_gnn_lgbm.py --seeds 42 123 7   # multi-seed random split
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path helpers — resolve project root relative to this file's location
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
_GRAPH_DIR = _PROJECT_ROOT / "results" / "graph"
_SPLITS_DIR = _GRAPH_DIR / "evaluation"
_ARTIFACTS_DIR = _GRAPH_DIR / "artifacts"

# Add src/ to the path so local modules can be imported
sys.path.insert(0, str(_SRC_DIR))

from oulad_data import (
    build_features,
    filter_window,
    load_oulad_data,
    sanitize_feature_names,
)


# ---------------------------------------------------------------------------
# 1. LightGBM configuration (mirrored from src/config.py MODEL_PARAMS)
# ---------------------------------------------------------------------------

def get_lgbm_config() -> dict:
    """Return the canonical LightGBM hyperparameter dict from config.py.

    Mirrors ``config.MODEL_PARAMS["lightgbm"]`` exactly so results are
    directly comparable with the existing pipeline.
    """
    return {"n_estimators": 100, "random_state": 42, "verbose": -1}


# ---------------------------------------------------------------------------
# 2. Tabular feature building
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "vle_total", "vle_mean", "vle_std",
    "assess_mean", "assess_max", "assess_count",
    "num_of_prev_attempts",
]

# Cache so we don't reload CSV files multiple times within one run
_DATA_CACHE: dict = {}


def _load_raw_data():
    """Load and cache core OULAD tables."""
    if "loaded" not in _DATA_CACHE:
        student_info, student_vle, student_assess, assessments = load_oulad_data()
        _DATA_CACHE["student_info"] = student_info
        _DATA_CACHE["student_vle"] = student_vle
        _DATA_CACHE["student_assess"] = student_assess
        _DATA_CACHE["assessments"] = assessments
        _DATA_CACHE["loaded"] = True
    return (
        _DATA_CACHE["student_info"],
        _DATA_CACHE["student_vle"],
        _DATA_CACHE["student_assess"],
        _DATA_CACHE["assessments"],
    )


def build_tabular_features(week: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and labels for *week*.

    Replicates exactly what ``create_datasets()`` does in ``oulad_data.py``:
    apply the prediction-window filter (Strategy B — due-date + submission-date
    guards), then call ``build_features()`` to get one row per enrollment.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric columns only; NaNs filled to 0).
    y : pd.Series
        Binary at-risk labels aligned to X.
    df : pd.DataFrame
        Full frame including ``id_student``, ``code_module``,
        ``code_presentation`` for alignment.
    """
    student_info, student_vle, student_assess, assessments = _load_raw_data()
    window = week * 7  # days from course start

    vle_w, assess_w = filter_window(
        student_vle, student_assess, assessments, window,
        submission_date_guard=True,
    )
    df = build_features(vle_w, assess_w, student_info)
    df = sanitize_feature_names(df)

    # Select numeric feature columns that are available (guard against missing ones)
    available_feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[available_feature_cols].copy()
    y = df["target"].copy()
    return X, y, df


def _align_to_enrollments(df_features: pd.DataFrame, enrollments: pd.DataFrame) -> pd.Index:
    """Return the integer positions in *df_features* that match *enrollments*.

    Both DataFrames use the same key triple: ``id_student``, ``code_module``,
    ``code_presentation``.  We merge on those keys to get a positional index.
    """
    key_cols = ["id_student", "code_module", "code_presentation"]
    df_features = df_features.reset_index(drop=True)
    enrollments = enrollments.reset_index(drop=True)

    merged = enrollments[key_cols].merge(
        df_features[key_cols].reset_index().rename(columns={"index": "_feat_pos"}),
        on=key_cols,
        how="left",
    )
    if merged["_feat_pos"].isna().any():
        raise ValueError("Some enrollment rows have no matching feature row.")
    return merged["_feat_pos"].astype(int).values


# ---------------------------------------------------------------------------
# 3. LightGBM — random split
# ---------------------------------------------------------------------------

def run_lgbm_random_split(week: int = 8, seed: int = 42) -> dict:
    """Train LightGBM on a 70/10/20 random-student split.

    Parameters
    ----------
    week : int
        Prediction week.
    seed : int
        Random seed used to draw the student split via ``random_student_split()``.
        Mirrors the GNN's multi-seed protocol.

    Returns
    -------
    dict
        Metrics dict with lowercase keys: auroc, auprc, f1, precision,
        recall, balanced_acc.
    """
    from oulad_data import random_student_split as _random_student_split

    # Load enrollment table to derive split masks at this seed
    enrollments_path = _ARTIFACTS_DIR / f"week{week:02d}_enrollments.parquet"
    enrollments = pd.read_parquet(enrollments_path)

    X, y, df_full = build_tabular_features(week)

    # Align the feature rows to the canonical enrollment ordering
    feat_pos = _align_to_enrollments(df_full, enrollments)
    X_aligned = X.iloc[feat_pos].reset_index(drop=True)
    y_aligned = y.iloc[feat_pos].reset_index(drop=True)

    # Derive per-seed masks using the same utility as the GNN
    train_s, val_s, test_s = _random_student_split(enrollments, seed=seed)
    # Match the GNN protocol: LightGBM trains on train+val, tests on test.
    train_mask = (train_s | val_s).values
    test_mask = test_s.values

    X_train, y_train = X_aligned[train_mask], y_aligned[train_mask]
    X_test, y_test = X_aligned[test_mask], y_aligned[test_mask]

    model = LGBMClassifier(**get_lgbm_config())
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return _compute_metrics(y_test.values, y_pred, y_proba)


# ---------------------------------------------------------------------------
# 4. LightGBM — LCPO
# ---------------------------------------------------------------------------

def run_lgbm_lcpo(week: int = 8, max_folds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LightGBM on every LCPO fold.

    Parameters
    ----------
    week : int
        Prediction week.
    max_folds : int or None
        If not None, stop after this many folds (for ``--quick`` mode).

    Returns
    -------
    fold_df : pd.DataFrame
        Per-fold metrics (one row per fold).
    summary_df : pd.DataFrame
        Mean ± std across folds (one row per metric).
    """
    folds_path = _SPLITS_DIR / f"week{week:02d}" / "splits" / f"week{week:02d}_lcpo_folds.csv"
    folds_df = pd.read_csv(folds_path)
    # Columns: fold_idx, held_out_module, held_out_presentation, n_train, n_test

    enrollments_path = _ARTIFACTS_DIR / f"week{week:02d}_enrollments.parquet"
    enrollments = pd.read_parquet(enrollments_path)
    # Columns: id_student, code_module, code_presentation, final_result, target

    X, y, df_full = build_tabular_features(week)
    # Align feature rows to the canonical enrollment ordering
    feat_pos = _align_to_enrollments(df_full, enrollments)
    X_aligned = X.iloc[feat_pos].reset_index(drop=True)
    y_aligned = y.iloc[feat_pos].reset_index(drop=True)
    enroll_aligned = enrollments.reset_index(drop=True)

    records = []
    n_folds = len(folds_df) if max_folds is None else min(max_folds, len(folds_df))

    for _, fold_row in folds_df.iloc[:n_folds].iterrows():
        held_mod = fold_row["held_out_module"]
        held_pres = fold_row["held_out_presentation"]
        fold_idx = int(fold_row["fold_idx"])

        is_held_out = (
            (enroll_aligned["code_module"] == held_mod) &
            (enroll_aligned["code_presentation"] == held_pres)
        )
        train_mask = ~is_held_out
        test_mask = is_held_out

        X_train = X_aligned[train_mask]
        y_train = y_aligned[train_mask]
        X_test = X_aligned[test_mask]
        y_test = y_aligned[test_mask]

        model = LGBMClassifier(**get_lgbm_config())
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = _compute_metrics(y_test.values, y_pred, y_proba)
        records.append({
            "fold_idx": fold_idx,
            "held_out_module": held_mod,
            "held_out_presentation": held_pres,
            **metrics,
        })

    fold_df = pd.DataFrame(records)

    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    summary_rows = []
    for m in metric_keys:
        vals = fold_df[m]
        summary_rows.append({
            "metric": m,
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if len(vals) > 1 else float("nan"),
            "min": vals.min(),
            "max": vals.max(),
        })
    summary_df = pd.DataFrame(summary_rows)

    return fold_df, summary_df


# ---------------------------------------------------------------------------
# 5. Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison_table(
    gnn_random: dict,
    lgbm_random: dict,
    gnn_lcpo_summary: pd.DataFrame,
    lgbm_lcpo_summary: pd.DataFrame,
) -> str:
    """Build the legacy single-row Markdown comparison table.

    Parameters
    ----------
    gnn_random / lgbm_random
        Metrics dicts with keys ``auroc``, ``auprc``, ``f1``, ``precision``,
        ``recall``, ``balanced_acc``.
    gnn_lcpo_summary / lgbm_lcpo_summary
        DataFrames with columns ``metric``, ``mean``, ``std``
        (one row per metric).
    """
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    header_labels = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced Acc"]

    def _fmt_scalar(d: dict, key: str) -> str:
        v = d.get(key, float("nan"))
        return f"{v:.4f}" if not np.isnan(v) else "—"

    def _fmt_mean_std(summary_df: pd.DataFrame, key: str) -> str:
        row = summary_df[summary_df["metric"] == key]
        if row.empty:
            return "—"
        mean = row.iloc[0]["mean"]
        std = row.iloc[0]["std"]
        if np.isnan(mean):
            return "—"
        if np.isnan(std):
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    rows = [
        ("GNN (EnrollmentGNN)", "Random split", gnn_random, None),
        ("LightGBM", "Random split", lgbm_random, None),
        ("GNN (EnrollmentGNN)", "LCPO mean ± std", None, gnn_lcpo_summary),
        ("LightGBM", "LCPO mean ± std", None, lgbm_lcpo_summary),
    ]

    col_header = "| Model | Split | " + " | ".join(header_labels) + " |"
    separator = "|---|---|" + "|".join(["---"] * len(header_labels)) + "|"

    lines = [col_header, separator]
    for model_name, split_name, scalar_d, summary_df in rows:
        if scalar_d is not None:
            cells = [_fmt_scalar(scalar_d, k) for k in metric_keys]
        else:
            cells = [_fmt_mean_std(summary_df, k) for k in metric_keys]
        lines.append(f"| {model_name} | {split_name} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_combined_csv(
    gnn_random_df: pd.DataFrame,
    lgbm_random_rows: list[dict],
    gnn_lcpo_df: pd.DataFrame,
    lgbm_lcpo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a combined per-fold/per-seed results DataFrame.

    Columns: model, split_type, fold_or_seed, held_out_module,
             held_out_presentation, auroc, auprc, f1, precision, recall,
             balanced_acc.
    """
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    records = []

    # ---- GNN random rows (one per seed × loss_weighting) ----
    for _, r in gnn_random_df.iterrows():
        records.append({
            "model": f"GNN ({r.get('loss_weighting', 'weighted')})",
            "split_type": "random_student",
            "fold_or_seed": r.get("seed", ""),
            "held_out_module": "",
            "held_out_presentation": "",
            **{k: r.get(k, float("nan")) for k in metric_keys},
        })

    # ---- LightGBM random rows (one per seed) ----
    for rec in lgbm_random_rows:
        records.append({
            "model": "LightGBM",
            "split_type": "random_student",
            "fold_or_seed": rec.get("seed", ""),
            "held_out_module": "",
            "held_out_presentation": "",
            **{k: rec.get(k, float("nan")) for k in metric_keys},
        })

    # ---- GNN LCPO rows (one per fold) ----
    for _, r in gnn_lcpo_df.iterrows():
        records.append({
            "model": "GNN",
            "split_type": "lcpo",
            "fold_or_seed": r.get("fold_idx", ""),
            "held_out_module": r.get("held_out_module", ""),
            "held_out_presentation": r.get("held_out_presentation", ""),
            **{k: r.get(k, float("nan")) for k in metric_keys},
        })

    # ---- LightGBM LCPO rows (one per fold) ----
    for _, r in lgbm_lcpo_df.iterrows():
        records.append({
            "model": "LightGBM",
            "split_type": "lcpo",
            "fold_or_seed": r.get("fold_idx", ""),
            "held_out_module": r.get("held_out_module", ""),
            "held_out_presentation": r.get("held_out_presentation", ""),
            **{k: r.get(k, float("nan")) for k in metric_keys},
        })

    return pd.DataFrame(records)


def build_summary_markdown(combined_df: pd.DataFrame) -> str:
    """Build a Markdown summary with mean ± std per (model, split_type).

    The table format mirrors the spec in Sub-task 5 of the improvement plan.
    """
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    header_labels = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced Acc"]

    def _fmt(vals: pd.Series) -> str:
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else float("nan")
        if np.isnan(mean):
            return "—"
        if np.isnan(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    def _section(title: str, sub_df: pd.DataFrame) -> list[str]:
        lines = [f"\n## {title}\n"]
        col_header = "| Model | " + " | ".join(header_labels) + " |"
        separator = "|-------|" + "|".join(["-------"] * len(header_labels)) + "|"
        lines += [col_header, separator]
        for model_name, grp in sub_df.groupby("model", sort=False):
            cells = [_fmt(grp[k]) for k in metric_keys]
            lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
        return lines

    parts = ["# GNN vs LightGBM Comparison Summary\n"]

    random_df = combined_df[combined_df["split_type"] == "random_student"]
    lcpo_df = combined_df[combined_df["split_type"] == "lcpo"]

    if not random_df.empty:
        parts += _section("Random-student split", random_df)
    if not lcpo_df.empty:
        parts += _section("LCPO (Leave-Course-Presentation-Out)", lcpo_df)

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return a dict with lowercase metric keys."""
    return {
        "auroc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 6. main()
# ---------------------------------------------------------------------------

def main(week: int = 8, quick: bool = False, seeds: list[int] | None = None):
    if seeds is None:
        seeds = [42]
    max_folds = 2 if quick else None

    print(f"[compare_gnn_lgbm] week={week}, quick={quick}, seeds={seeds}")

    # --- Load GNN results (written by run_gnn_experiment.py) ---
    gnn_random_path = _GRAPH_DIR / "random_student_results.csv"
    gnn_lcpo_path = _GRAPH_DIR / "lcpo_results.csv"
    gnn_lcpo_summary_path = _GRAPH_DIR / "lcpo_summary.csv"

    gnn_random_df = pd.read_csv(gnn_random_path)

    # Legacy single-row summary for the old comparison table
    gnn_random: dict = {}
    if not gnn_random_df.empty:
        # Use the first weighted row from the first seed for the legacy table
        if "loss_weighting" in gnn_random_df.columns:
            weighted_rows = gnn_random_df[gnn_random_df["loss_weighting"] == "weighted"]
            ref_row = weighted_rows.iloc[0] if not weighted_rows.empty else gnn_random_df.iloc[0]
        else:
            ref_row = gnn_random_df.iloc[0]
        gnn_random = {k: ref_row.get(k, float("nan"))
                      for k in ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]}

    gnn_lcpo_df = pd.read_csv(gnn_lcpo_path) if gnn_lcpo_path.exists() else pd.DataFrame()
    gnn_lcpo_summary = pd.read_csv(gnn_lcpo_summary_path)
    # Columns: metric, mean, std, min, max

    # --- Run LightGBM random split (one run per seed) ---
    lgbm_random_rows: list[dict] = []
    for seed_val in seeds:
        print(f"[compare_gnn_lgbm] Running LightGBM random split (seed={seed_val}) …")
        metrics = run_lgbm_random_split(week=week, seed=seed_val)
        lgbm_random_rows.append({"seed": seed_val, **metrics})
        print(f"  LightGBM random split AUROC (seed {seed_val}): {metrics['auroc']:.4f}")

    # For legacy table: use mean across seeds
    lgbm_random: dict = {}
    if lgbm_random_rows:
        metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
        lgbm_random = {k: float(np.mean([r[k] for r in lgbm_random_rows])) for k in metric_keys}

    # --- Run LightGBM LCPO ---
    print(f"[compare_gnn_lgbm] Running LightGBM LCPO ({max_folds or 22} folds) …")
    lgbm_fold_df, lgbm_lcpo_summary = run_lgbm_lcpo(week=week, max_folds=max_folds)
    print(f"  LightGBM LCPO mean AUROC: {lgbm_lcpo_summary[lgbm_lcpo_summary['metric']=='auroc']['mean'].values[0]:.4f}")

    # --- Build legacy comparison table (gnn_vs_lgbm_comparison.md) ---
    table_md = build_comparison_table(
        gnn_random=gnn_random,
        lgbm_random=lgbm_random,
        gnn_lcpo_summary=gnn_lcpo_summary,
        lgbm_lcpo_summary=lgbm_lcpo_summary,
    )
    out_path = _GRAPH_DIR / "gnn_vs_lgbm_comparison.md"
    out_path.write_text(table_md + "\n")
    print(f"\n[compare_gnn_lgbm] Comparison table saved to {out_path}")
    print(table_md)

    # --- Build combined per-fold/per-seed CSV ---
    combined_df = build_combined_csv(
        gnn_random_df=gnn_random_df,
        lgbm_random_rows=lgbm_random_rows,
        gnn_lcpo_df=gnn_lcpo_df,
        lgbm_lcpo_df=lgbm_fold_df,
    )
    combined_path = _GRAPH_DIR / "comparison_results.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"[compare_gnn_lgbm] Combined results CSV saved to {combined_path}")

    # --- Build Markdown summary (comparison_summary.md) ---
    summary_md = build_summary_markdown(combined_df)
    summary_path = _GRAPH_DIR / "comparison_summary.md"
    summary_path.write_text(summary_md)
    print(f"[compare_gnn_lgbm] Summary Markdown saved to {summary_path}")
    print(summary_md)


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare GNN vs. LightGBM under matched split definitions."
    )
    parser.add_argument(
        "--week", type=int, default=8,
        help="Prediction week (default: 8).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only 2 LCPO folds for a fast smoke test.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="One or more random seeds for the LightGBM random-student split "
             "(default: 42).  Mirrors the GNN's --seeds argument so both models "
             "are evaluated on the same set of partitions.",
    )
    args = parser.parse_args()
    main(week=args.week, quick=args.quick, seeds=args.seeds)
