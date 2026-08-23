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

from gnn_model import SEED, select_threshold
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
    # age_band one-hot columns are added dynamically after pd.get_dummies expansion
    "studied_credits",
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


def build_enrolled_in_features(week: int) -> pd.DataFrame:
    """Read enrolled_in edge parquet and return enrollment-scoped features.

    Returns a DataFrame with columns:
        id_student, code_module, code_presentation,
        age_band, studied_credits
    aligned to the canonical enrollment ordering in the parquet.

    The enrolled_in parquet (produced by GraphDataLoader) stores enrollment
    metadata columns alongside the edge src/dst indices.  We join them back
    to the enrollment key triple using the enrollments parquet (same row
    order — both are derived from studentInfo).

    Note: age_band is returned as a raw string category here; callers are
    responsible for one-hot encoding it via ``pd.get_dummies`` to match the
    GNN's encoding of enrolled_in edge attributes.
    """
    ei_path = _ARTIFACTS_DIR / f"week{week:02d}_edges_enrolled_in.parquet"
    ei = pd.read_parquet(ei_path)
    enroll = pd.read_parquet(_ARTIFACTS_DIR / f"week{week:02d}_enrollments.parquet")
    # Both files are derived in the same row order (studentInfo), so we can
    # assign the attribute columns directly.
    result = enroll[["id_student", "code_module", "code_presentation"]].copy()
    result["age_band"] = ei["age_band"].values
    result["studied_credits"] = ei["studied_credits"].values
    return result


def build_tabular_features(week: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and labels for *week*.

    Replicates exactly what ``create_datasets()`` does in ``oulad_data.py``:
    apply the prediction-window filter (Strategy B — due-date + submission-date
    guards), then call ``build_features()`` to get one row per enrollment.

    Also merges in enrollment-scoped features from the enrolled_in edge
    artifact (``age_band`` one-hot columns and ``studied_credits``) so that
    LightGBM receives the same features as the GNN edge prediction head.

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

    # Merge in enrollment-scoped features from the enrolled_in edge artifact
    ei_feats = build_enrolled_in_features(week)
    df = df.merge(
        ei_feats[["id_student", "code_module", "code_presentation",
                  "age_band", "studied_credits"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )

    # One-hot encode age_band to match GNN encoding; drop original string col
    age_dummies = pd.get_dummies(df["age_band"], prefix="age_band")
    df = pd.concat([df.drop(columns=["age_band"]), age_dummies], axis=1)
    age_dummy_cols = list(age_dummies.columns)

    # Build the full feature column list dynamically (base cols + age dummies)
    base_cols = [c for c in _FEATURE_COLS if c in df.columns]
    available_feature_cols = base_cols + age_dummy_cols

    X = df[available_feature_cols].fillna(0).copy()
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
    # Train on train only (not train+val) to match GNN protocol
    train_mask = train_s.values
    val_mask = val_s.values
    test_mask = test_s.values

    X_train, y_train = X_aligned[train_mask], y_aligned[train_mask]
    X_val, y_val = X_aligned[val_mask], y_aligned[val_mask]
    X_test, y_test = X_aligned[test_mask], y_aligned[test_mask]

    model = LGBMClassifier(**get_lgbm_config())
    model.fit(X_train, y_train)

    # Select F1-maximising threshold on the val set, same as GNN protocol
    val_proba = model.predict_proba(X_val)[:, 1]
    if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
        threshold = select_threshold(val_proba, y_val.values)
    else:
        threshold = 0.5

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    return _compute_metrics(y_test.values, test_pred, test_proba, threshold=threshold)


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

    Notes
    -----
    LightGBM is deterministic given ``random_state``, so one run per fold
    suffices (no model-seed loop needed, unlike the GNN's 5-seed protocol).
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
        train_all_mask = (~is_held_out).values
        test_mask = is_held_out.values

        # Draw 10% val students from the train pool, matching the GNN LCPO
        # protocol (rng keyed on fold_idx only, matching the decoupled RNG in
        # run_lcpo_experiment() — Sub-Task 5).
        train_student_ids = enroll_aligned.loc[train_all_mask, "id_student"].unique()
        rng = np.random.default_rng(fold_idx)
        val_size = max(1, int(0.10 * len(train_student_ids)))
        val_student_ids = rng.choice(train_student_ids, size=val_size, replace=False)
        val_student_set = set(val_student_ids)

        val_mask = train_all_mask & enroll_aligned["id_student"].isin(val_student_set).to_numpy()
        train_mask = train_all_mask & ~val_mask

        X_train = X_aligned[train_mask]
        y_train = y_aligned[train_mask]
        X_val = X_aligned[val_mask]
        y_val = y_aligned[val_mask]
        X_test = X_aligned[test_mask]
        y_test = y_aligned[test_mask]

        model = LGBMClassifier(**get_lgbm_config())
        model.fit(X_train, y_train)

        # Tune threshold on val set using F1-max, matching GNN protocol
        val_proba = model.predict_proba(X_val)[:, 1]
        if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
            threshold = select_threshold(val_proba, y_val.values)
        else:
            threshold = 0.5

        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)

        metrics = _compute_metrics(y_test.values, test_pred, test_proba, threshold=threshold)
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

    Columns: week, model, split_type, fold_or_seed, held_out_module,
             held_out_presentation, auroc, auprc, f1, precision, recall,
             balanced_acc.
    """
    metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    records = []

    # ---- GNN random rows (one per seed × loss_weighting) ----
    for _, r in gnn_random_df.iterrows():
        records.append({
            "week": r.get("week", ""),
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
            "week": rec.get("week", ""),
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
            "week": r.get("week", ""),
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
            "week": r.get("week", ""),
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

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return a dict with lowercase metric keys.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels (should already be thresholded at *threshold*).
    y_proba : np.ndarray
        Predicted probabilities (used for ranking metrics AUROC / AUPRC).
    threshold : float, optional
        Classification threshold used to produce *y_pred* (default 0.5).
        Stored in the returned dict for traceability.
    """
    return {
        "auroc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 6. main()
# ---------------------------------------------------------------------------

def _load_gnn_results_from_csv(
    weeks: list[int],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    """Load existing GNN result CSVs and run LightGBM fresh (fast).

    Used by the --from-csv path so experiments don't need to be re-run.
    Returns the same tuple of accumulators as the normal path.
    """
    gnn_random_path = _GRAPH_DIR / "random_student_results.csv"
    gnn_lcpo_path = _GRAPH_DIR / "lcpo_results.csv"
    gnn_lcpo_summary_path = _GRAPH_DIR / "lcpo_summary.csv"

    if not gnn_random_path.exists():
        raise FileNotFoundError(
            f"GNN random results not found: {gnn_random_path}. "
            "Run run_gnn_experiment.py first."
        )

    gnn_random_df = pd.read_csv(gnn_random_path)
    gnn_lcpo_df = pd.read_csv(gnn_lcpo_path) if gnn_lcpo_path.exists() else pd.DataFrame()
    gnn_lcpo_summary = pd.read_csv(gnn_lcpo_summary_path) if gnn_lcpo_summary_path.exists() else pd.DataFrame()

    all_lgbm_random_rows: list[dict] = []
    all_lgbm_fold_dfs: list[pd.DataFrame] = []
    all_lgbm_lcpo_summaries: list[pd.DataFrame] = []

    for wk in weeks:
        print(f"\n[compare_gnn_lgbm] === Week {wk} (--from-csv: running LightGBM fresh) ===")

        for seed_val in seeds:
            print(f"[compare_gnn_lgbm] Running LightGBM random split (week={wk}, seed={seed_val}) …")
            metrics = run_lgbm_random_split(week=wk, seed=seed_val)
            all_lgbm_random_rows.append({"week": wk, "seed": seed_val, **metrics})
            print(f"  LightGBM random split AUROC (seed {seed_val}): {metrics['auroc']:.4f}")

        print(f"[compare_gnn_lgbm] Running LightGBM LCPO week={wk} (all folds) …")
        lgbm_fold_df, lgbm_lcpo_summary_wk = run_lgbm_lcpo(week=wk, max_folds=None)
        lgbm_fold_df = lgbm_fold_df.copy()
        lgbm_fold_df["week"] = wk
        all_lgbm_fold_dfs.append(lgbm_fold_df)
        all_lgbm_lcpo_summaries.append(lgbm_lcpo_summary_wk)
        mean_auroc = lgbm_lcpo_summary_wk[lgbm_lcpo_summary_wk["metric"] == "auroc"]["mean"].values[0]
        print(f"  LightGBM LCPO mean AUROC (week {wk}): {mean_auroc:.4f}")

    return (
        gnn_random_df,
        gnn_lcpo_df,
        gnn_lcpo_summary,
        all_lgbm_random_rows,
        all_lgbm_fold_dfs,
        all_lgbm_lcpo_summaries,
    )


def main(
    week: int = 8,
    weeks: list[int] | None = None,
    quick: bool = False,
    seeds: list[int] | None = None,
    from_csv: bool = False,
):
    if seeds is None:
        seeds = [42]
    # Resolve week list: explicit `weeks` wins; fall back to scalar `week`
    if weeks is None:
        weeks = [week]
    max_folds = 2 if quick else None

    print(f"[compare_gnn_lgbm] weeks={weeks}, quick={quick}, seeds={seeds}, from_csv={from_csv}")

    if from_csv:
        # --from-csv: load existing GNN CSVs; run LightGBM fresh; build comparison_results.csv
        (
            gnn_random_df,
            gnn_lcpo_df,
            gnn_lcpo_summary,
            all_lgbm_random_rows,
            all_lgbm_fold_dfs,
            all_lgbm_lcpo_summaries,
        ) = _load_gnn_results_from_csv(weeks=weeks, seeds=seeds)
    else:
        # Normal path: load GNN results written by run_gnn_experiment.py, run LightGBM
        gnn_random_path = _GRAPH_DIR / "random_student_results.csv"
        gnn_lcpo_path = _GRAPH_DIR / "lcpo_results.csv"
        gnn_lcpo_summary_path = _GRAPH_DIR / "lcpo_summary.csv"

        gnn_random_df = pd.read_csv(gnn_random_path)
        gnn_lcpo_df = pd.read_csv(gnn_lcpo_path) if gnn_lcpo_path.exists() else pd.DataFrame()
        gnn_lcpo_summary = pd.read_csv(gnn_lcpo_summary_path) if gnn_lcpo_summary_path.exists() else pd.DataFrame()

        # Accumulate LightGBM results across all weeks
        all_lgbm_random_rows: list[dict] = []
        all_lgbm_fold_dfs: list[pd.DataFrame] = []
        all_lgbm_lcpo_summaries: list[pd.DataFrame] = []

        for wk in weeks:
            print(f"\n[compare_gnn_lgbm] === Week {wk} ===")

            # --- Run LightGBM random split (one run per seed) ---
            for seed_val in seeds:
                print(f"[compare_gnn_lgbm] Running LightGBM random split (week={wk}, seed={seed_val}) …")
                metrics = run_lgbm_random_split(week=wk, seed=seed_val)
                all_lgbm_random_rows.append({"week": wk, "seed": seed_val, **metrics})
                print(f"  LightGBM random split AUROC (seed {seed_val}): {metrics['auroc']:.4f}")

            # --- Run LightGBM LCPO ---
            print(f"[compare_gnn_lgbm] Running LightGBM LCPO week={wk} ({max_folds or 22} folds) …")
            lgbm_fold_df, lgbm_lcpo_summary = run_lgbm_lcpo(week=wk, max_folds=max_folds)
            lgbm_fold_df = lgbm_fold_df.copy()
            lgbm_fold_df["week"] = wk
            all_lgbm_fold_dfs.append(lgbm_fold_df)
            all_lgbm_lcpo_summaries.append(lgbm_lcpo_summary)
            mean_auroc = lgbm_lcpo_summary[lgbm_lcpo_summary["metric"] == "auroc"]["mean"].values[0]
            print(f"  LightGBM LCPO mean AUROC (week {wk}): {mean_auroc:.4f}")

    # Combine all weeks
    lgbm_all_fold_df = pd.concat(all_lgbm_fold_dfs, ignore_index=True) if all_lgbm_fold_dfs else pd.DataFrame()
    # Use the last week's summary for the legacy table (or the overall one if single week)
    lgbm_lcpo_summary_for_table = all_lgbm_lcpo_summaries[-1] if all_lgbm_lcpo_summaries else pd.DataFrame()

    # Legacy single-row summary for the old comparison table (use first week's data)
    ref_week = weeks[0]
    gnn_random: dict = {}
    gnn_random_week_df = gnn_random_df[gnn_random_df["week"] == ref_week] if "week" in gnn_random_df.columns else gnn_random_df
    if not gnn_random_week_df.empty:
        if "loss_weighting" in gnn_random_week_df.columns:
            weighted_rows = gnn_random_week_df[gnn_random_week_df["loss_weighting"] == "weighted"]
            ref_row = weighted_rows.iloc[0] if not weighted_rows.empty else gnn_random_week_df.iloc[0]
        else:
            ref_row = gnn_random_week_df.iloc[0]
        gnn_random = {k: ref_row.get(k, float("nan"))
                      for k in ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]}

    lgbm_random_ref = [r for r in all_lgbm_random_rows if r.get("week") == ref_week]
    lgbm_random: dict = {}
    if lgbm_random_ref:
        metric_keys = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
        lgbm_random = {k: float(np.mean([r[k] for r in lgbm_random_ref])) for k in metric_keys}

    gnn_lcpo_summary_for_table = gnn_lcpo_summary if not gnn_lcpo_summary.empty else pd.DataFrame(
        [{"metric": m, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
         for m in ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]]
    )

    # --- Build legacy comparison table (gnn_vs_lgbm_comparison.md) ---
    table_md = build_comparison_table(
        gnn_random=gnn_random,
        lgbm_random=lgbm_random,
        gnn_lcpo_summary=gnn_lcpo_summary_for_table,
        lgbm_lcpo_summary=lgbm_lcpo_summary_for_table,
    )
    out_path = _GRAPH_DIR / "gnn_vs_lgbm_comparison.md"
    out_path.write_text(table_md + "\n")
    print(f"\n[compare_gnn_lgbm] Comparison table saved to {out_path}")
    print(table_md)

    # --- Build combined per-fold/per-seed CSV ---
    combined_df = build_combined_csv(
        gnn_random_df=gnn_random_df,
        lgbm_random_rows=all_lgbm_random_rows,
        gnn_lcpo_df=gnn_lcpo_df,
        lgbm_lcpo_df=lgbm_all_fold_df,
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
        "--week", type=int, default=None,
        help="Single prediction week (default: 8). Superseded by --weeks.",
    )
    parser.add_argument(
        "--weeks", nargs="+", type=int, default=None,
        help="One or more prediction weeks (e.g. --weeks 2 4 6 8). Overrides --week.",
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
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Build comparison_results.csv from existing GNN CSVs without re-running "
             "GNN experiments. LightGBM is re-run fresh (it is fast).",
    )
    args = parser.parse_args()

    # Resolve week list
    if args.weeks is not None:
        resolved_weeks = args.weeks
    elif args.week is not None:
        resolved_weeks = [args.week]
    else:
        resolved_weeks = [8]

    main(weeks=resolved_weeks, quick=args.quick, seeds=args.seeds, from_csv=args.from_csv)
