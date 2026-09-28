"""
run_matched_comparison.py
-------------------------
Matched LightGBM and MLP comparison for the OULAD enrollment-risk task.

Implements the "matched protocol" from matched-comparison-plan.md:
  - Same enrollment population, labels, splits, seeds, and threshold-selection
    rule as the GNN pipeline (src/run_gnn_experiment.py).
  - Feature matrix: VLE + assessment + demographics (age_band 3-column one-hot,
    studied_credits, num_of_prev_attempts).
  - Dual temporal guard (Strategy B): due_date <= window AND date_submitted <= window.
  - Threshold: F1-max sweep on validation set.
  - Normalization (MLP only): StandardScaler fit on training rows only.

Usage
-----
  python src/run_matched_comparison.py --models lgbm mlp --weeks 2 4 6 8 \\
      --seeds 42 123 7 17 99 --splits random lcpo

  # Quick smoke test (one week, one seed, random only):
  PYTHONPATH=src python src/run_matched_comparison.py \\
      --models lgbm --weeks 8 --seeds 42 --splits random
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — flat imports from src/
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

from config import (
    DATA_DIR,
    GRAPH_ARTIFACTS_DIR,
    GRAPH_EVALUATION_DIR,
    PREDICTION_WINDOWS,
    RESULTS_DIR,
)
from oulad_data import (
    build_features,
    filter_window,
    load_oulad_data,
    random_student_split,
    sanitize_feature_names,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_MATCHED_DIR = RESULTS_DIR / "matched"

# ---------------------------------------------------------------------------
# Feature columns (base set — age_band dummies added dynamically)
# ---------------------------------------------------------------------------

_BASE_FEATURE_COLS = [
    "vle_total",
    "vle_mean",
    "vle_std",
    "assess_mean",
    "assess_max",
    "assess_count",
    "num_of_prev_attempts",
    "studied_credits",
]

# ---------------------------------------------------------------------------
# Raw-data cache — avoid reloading CSV files for every week/seed
# ---------------------------------------------------------------------------

_DATA_CACHE: dict = {}


def _load_raw_data():
    """Load and cache core OULAD tables (loaded once per process)."""
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


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_tabular_features(week: int, data=None):
    """Build enrollment-level feature matrix for *week*.

    Applies the dual temporal guard (Strategy B):
        due_date <= window AND date_submitted <= window

    Parameters
    ----------
    week : int
        Prediction week (2, 4, 6, or 8).
    data : tuple or None
        Optional pre-loaded (student_info, student_vle, student_assess,
        assessments) to avoid reloading from disk per week.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric only; NaNs filled to 0); feature names sanitized
        for XGBoost / LightGBM compatibility.
    y : pd.Series
        Binary at-risk labels aligned to X.
    enrollment_df : pd.DataFrame
        Full DataFrame including id_student, code_module, code_presentation and
        all other columns produced by build_features().  Needed to derive split
        masks via random_student_split() or to cross-reference LCPO folds.
    """
    if data is None:
        student_info, student_vle, student_assess, assessments = _load_raw_data()
    else:
        student_info, student_vle, student_assess, assessments = data

    # week * 7 converts week number to day-from-course-start (see PREDICTION_WINDOWS)
    window = PREDICTION_WINDOWS[f"week_{week}"]

    vle_w, assess_w = filter_window(
        student_vle, student_assess, assessments,
        window=window,
        submission_date_guard=True,
    )

    df = build_features(vle_w, assess_w, student_info)
    df = sanitize_feature_names(df)

    # One-hot encode age_band (3 categories in OULAD: 0-35, 35-55, 55<=)
    # drop_first=False preserves all 3 columns to match GNN enrolled_in edge attrs.
    age_dummies = pd.get_dummies(df["age_band"], prefix="age_band", drop_first=False)
    df = pd.concat([df.drop(columns=["age_band"]), age_dummies], axis=1)
    age_dummy_cols = list(age_dummies.columns)

    # Build feature column list dynamically (base cols present in df + age dummies)
    base_cols = [c for c in _BASE_FEATURE_COLS if c in df.columns]
    feature_cols = base_cols + age_dummy_cols

    X = df[feature_cols].fillna(0).copy()
    X = sanitize_feature_names(X)  # sanitize again after concat (dummies may have <>)
    y = df["target"].copy()

    return X, y, df


# ---------------------------------------------------------------------------
# Threshold selection (F1-maximising sweep)
# ---------------------------------------------------------------------------

def _select_threshold(proba: np.ndarray, labels: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given set.

    Sweeps thresholds from 0.05 to 0.95 in steps of 0.05, matching the GNN
    protocol in gnn_model.select_threshold().
    """
    from sklearn.metrics import f1_score as _f1

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 1.0, 0.05):
        preds = (proba >= thr).astype(int)
        f1 = _f1(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Return a metrics dict with lowercase keys."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

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
# CSV writer (idempotent)
# ---------------------------------------------------------------------------

def _append_or_create_csv(
    df: pd.DataFrame, path: Path, dedup_keys: list
) -> None:
    """Append *df* to *path*, deduplicating on *dedup_keys* (keep last).

    If *path* does not exist, writes *df* directly.  This matches the pattern
    in run_gnn_experiment.py so re-running experiments updates existing rows
    rather than creating duplicates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    existing = pd.read_csv(path)
    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    combined.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Experiment 1: Matched LightGBM
# ---------------------------------------------------------------------------

def run_lgbm_matched(
    weeks: list,
    seeds: list,
    splits: list,
) -> None:
    """Run matched LightGBM for all weeks and split types.

    Parameters
    ----------
    weeks  : list of int — prediction weeks to evaluate.
    seeds  : list of int — random seeds for the random-student split.
    splits : list of str — any subset of ["random", "lcpo"].
    """
    out_path = _MATCHED_DIR / "lgbm_matched_results.csv"
    all_rows = []

    for week in weeks:
        print(f"\n=== LightGBM matched — week {week:02d} ===")
        X, y, enrollment_df = build_tabular_features(week)

        # ------------------------------------------------------------------
        # Random-student split
        # ------------------------------------------------------------------
        if "random" in splits:
            print(f"  [random split] seeds: {seeds}")
            for seed in seeds:
                train_mask, val_mask, test_mask = random_student_split(
                    enrollment_df, val_frac=0.1, test_frac=0.2, seed=seed
                )

                X_train = X[train_mask.values]
                y_train = y[train_mask.values]
                X_val = X[val_mask.values]
                y_val = y[val_mask.values]
                X_test = X[test_mask.values]
                y_test = y[test_mask.values]

                model = LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1)
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
                    threshold = _select_threshold(val_proba, y_val.values)
                else:
                    threshold = 0.5

                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)
                metrics = _compute_metrics(y_test.values, test_pred, test_proba, threshold)

                row = {
                    "model": "LightGBM_matched",
                    "week": week,
                    "split_type": "random",
                    "seed": seed,
                    "fold": -1,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"    seed={seed}  auroc={metrics['auroc']:.4f}"
                    f"  f1={metrics['f1']:.4f}  thr={threshold:.2f}"
                )

        # ------------------------------------------------------------------
        # LCPO split
        # ------------------------------------------------------------------
        if "lcpo" in splits:
            folds_path = (
                GRAPH_EVALUATION_DIR
                / f"week{week:02d}"
                / "splits"
                / f"week{week:02d}_lcpo_folds.csv"
            )
            folds_df = pd.read_csv(folds_path)
            print(f"  [LCPO] {len(folds_df)} folds from {folds_path}")

            for _, fold_row in folds_df.iterrows():
                fold_idx = int(fold_row["fold_idx"])
                held_mod = fold_row["held_out_module"]
                held_pres = fold_row["held_out_presentation"]

                is_held_out = (
                    (enrollment_df["code_module"] == held_mod)
                    & (enrollment_df["code_presentation"] == held_pres)
                )
                train_all_mask = (~is_held_out).values
                test_mask_lc = is_held_out.values

                # Val: 10% of train students, seeded by fold_idx (matches GNN LCPO)
                train_student_ids = enrollment_df.loc[train_all_mask, "id_student"].unique()
                rng = np.random.default_rng(fold_idx)
                val_size = max(1, int(0.10 * len(train_student_ids)))
                val_student_ids = rng.choice(train_student_ids, size=val_size, replace=False)
                val_student_set = set(val_student_ids)

                val_mask_lc = (
                    train_all_mask
                    & enrollment_df["id_student"].isin(val_student_set).to_numpy()
                )
                train_mask_lc = train_all_mask & ~val_mask_lc

                X_train = X[train_mask_lc]
                y_train = y[train_mask_lc]
                X_val = X[val_mask_lc]
                y_val = y[val_mask_lc]
                X_test = X[test_mask_lc]
                y_test = y[test_mask_lc]

                if y_test.sum() == 0 or (len(y_test) - y_test.sum()) == 0:
                    print(f"    fold {fold_idx:02d}: SKIP (single class in test)")
                    continue

                model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
                    threshold = _select_threshold(val_proba, y_val.values)
                else:
                    threshold = 0.5

                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)
                metrics = _compute_metrics(y_test.values, test_pred, test_proba, threshold)

                row = {
                    "model": "LightGBM_matched",
                    "week": week,
                    "split_type": "lcpo",
                    "seed": -1,
                    "fold": fold_idx,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"    fold {fold_idx:02d} ({held_mod}/{held_pres})"
                    f"  auroc={metrics['auroc']:.4f}"
                    f"  f1={metrics['f1']:.4f}"
                )

    if not all_rows:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_rows)
    _append_or_create_csv(
        results_df, out_path,
        dedup_keys=["model", "week", "split_type", "seed", "fold"],
    )
    print(f"\nResults written to {out_path}")
    _print_summary(results_df, "LightGBM_matched")


# ---------------------------------------------------------------------------
# Experiment 2: Matched MLP
# ---------------------------------------------------------------------------

def run_mlp_matched(
    weeks: list,
    seeds: list,
    splits: list,
) -> None:
    """Run matched MLP for all weeks and split types.

    Architecture: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
    early_stopping=True, validation_fraction=0.1).
    Normalization: StandardScaler fit on training rows only, applied to all splits.
    """
    out_path = _MATCHED_DIR / "mlp_matched_results.csv"
    all_rows = []

    for week in weeks:
        print(f"\n=== MLP matched — week {week:02d} ===")
        X, y, enrollment_df = build_tabular_features(week)

        # ------------------------------------------------------------------
        # Random-student split
        # ------------------------------------------------------------------
        if "random" in splits:
            print(f"  [random split] seeds: {seeds}")
            for seed in seeds:
                train_mask, val_mask, test_mask = random_student_split(
                    enrollment_df, val_frac=0.1, test_frac=0.2, seed=seed
                )

                X_train_raw = X[train_mask.values].values
                y_train = y[train_mask.values].values
                X_val_raw = X[val_mask.values].values
                y_val = y[val_mask.values].values
                X_test_raw = X[test_mask.values].values
                y_test = y[test_mask.values].values

                # Normalization fit on training rows only
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val = scaler.transform(X_val_raw)
                X_test = scaler.transform(X_test_raw)

                model = MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed,
                )
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
                    threshold = _select_threshold(val_proba, y_val)
                else:
                    threshold = 0.5

                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)
                metrics = _compute_metrics(y_test, test_pred, test_proba, threshold)

                row = {
                    "model": "MLP_all_features",
                    "week": week,
                    "split_type": "random",
                    "seed": seed,
                    "fold": -1,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"    seed={seed}  auroc={metrics['auroc']:.4f}"
                    f"  f1={metrics['f1']:.4f}  thr={threshold:.2f}"
                )

        # ------------------------------------------------------------------
        # LCPO split
        # ------------------------------------------------------------------
        if "lcpo" in splits:
            folds_path = (
                GRAPH_EVALUATION_DIR
                / f"week{week:02d}"
                / "splits"
                / f"week{week:02d}_lcpo_folds.csv"
            )
            folds_df = pd.read_csv(folds_path)
            print(f"  [LCPO] {len(folds_df)} folds from {folds_path}")

            for _, fold_row in folds_df.iterrows():
                fold_idx = int(fold_row["fold_idx"])
                held_mod = fold_row["held_out_module"]
                held_pres = fold_row["held_out_presentation"]

                is_held_out = (
                    (enrollment_df["code_module"] == held_mod)
                    & (enrollment_df["code_presentation"] == held_pres)
                )
                train_all_mask = (~is_held_out).values
                test_mask_lc = is_held_out.values

                # Val: 10% of train students, seeded by fold_idx (matches GNN LCPO)
                train_student_ids = enrollment_df.loc[train_all_mask, "id_student"].unique()
                rng = np.random.default_rng(fold_idx)
                val_size = max(1, int(0.10 * len(train_student_ids)))
                val_student_ids = rng.choice(train_student_ids, size=val_size, replace=False)
                val_student_set = set(val_student_ids)

                val_mask_lc = (
                    train_all_mask
                    & enrollment_df["id_student"].isin(val_student_set).to_numpy()
                )
                train_mask_lc = train_all_mask & ~val_mask_lc

                X_train_raw = X[train_mask_lc].values
                y_train = y[train_mask_lc].values
                X_val_raw = X[val_mask_lc].values
                y_val = y[val_mask_lc].values
                X_test_raw = X[test_mask_lc].values
                y_test = y[test_mask_lc].values

                if y_test.sum() == 0 or (len(y_test) - y_test.sum()) == 0:
                    print(f"    fold {fold_idx:02d}: SKIP (single class in test)")
                    continue

                # Normalization fit on training rows only
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_val = scaler.transform(X_val_raw)
                X_test = scaler.transform(X_test_raw)

                model = MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                )
                model.fit(X_train, y_train)

                val_proba = model.predict_proba(X_val)[:, 1]
                if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
                    threshold = _select_threshold(val_proba, y_val)
                else:
                    threshold = 0.5

                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)
                metrics = _compute_metrics(y_test, test_pred, test_proba, threshold)

                row = {
                    "model": "MLP_all_features",
                    "week": week,
                    "split_type": "lcpo",
                    "seed": -1,
                    "fold": fold_idx,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"    fold {fold_idx:02d} ({held_mod}/{held_pres})"
                    f"  auroc={metrics['auroc']:.4f}"
                    f"  f1={metrics['f1']:.4f}"
                )

    if not all_rows:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_rows)
    _append_or_create_csv(
        results_df, out_path,
        dedup_keys=["model", "week", "split_type", "seed", "fold"],
    )
    print(f"\nResults written to {out_path}")
    _print_summary(results_df, "MLP_all_features")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, model_name: str) -> None:
    """Print mean ± std of AUROC and F1 grouped by week and split_type."""
    print(f"\n{'='*60}")
    print(f"Summary: {model_name}")
    print(f"{'='*60}")
    for (week, split_type), grp in df.groupby(["week", "split_type"]):
        auroc_mean = grp["auroc"].mean()
        auroc_std = grp["auroc"].std(ddof=1) if len(grp) > 1 else float("nan")
        f1_mean = grp["f1"].mean()
        f1_std = grp["f1"].std(ddof=1) if len(grp) > 1 else float("nan")
        print(
            f"  week={week}  split={split_type:<8}"
            f"  AUROC={auroc_mean:.4f}±{auroc_std:.4f}"
            f"  F1={f1_mean:.4f}±{f1_std:.4f}"
            f"  (n={len(grp)})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Matched LightGBM / MLP comparison for OULAD enrollment risk."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "mlp"],
        required=True,
        help="Which models to run.",
    )
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8],
        help="Prediction weeks (default: 2 4 6 8).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 7, 17, 99],
        help="Random seeds for random-student split (default: 42 123 7 17 99).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["random", "lcpo"],
        default=["random", "lcpo"],
        help="Which split types to run (default: random lcpo).",
    )

    args = parser.parse_args()
    weeks = args.weeks
    seeds = args.seeds
    splits = args.splits

    print(f"Models : {args.models}")
    print(f"Weeks  : {weeks}")
    print(f"Seeds  : {seeds}")
    print(f"Splits : {splits}")

    if "lgbm" in args.models:
        run_lgbm_matched(weeks=weeks, seeds=seeds, splits=splits)

    if "mlp" in args.models:
        run_mlp_matched(weeks=weeks, seeds=seeds, splits=splits)


if __name__ == "__main__":
    main()
