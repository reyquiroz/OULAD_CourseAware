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

Feature sets (--feature-set {A,B,C,D,E,S})
-------------------------------------------
  A : base — six behavioral summaries + num_of_prev_attempts, studied_credits,
      age_band one-hot  (original matched set; default)
  S : the six behavioral summaries ONLY (vle_total, vle_mean, vle_std, assess_mean,
      assess_max, assess_count). No previous attempts, credits or age band. This is
      the control that matches the GNN "+ six summaries" input.
  B : A + demographics one-hots  (gender, region, highest_education,
      imd_band, disability)
  C : A + course identity one-hots  (code_module, code_presentation)
  D : A + demographics + course identity
  E : D + four temporal features  (weeks_active, recency, activity_trend,
      max_inactive_gap) pulled from the enrollment parquet artifacts

Usage
-----
  python src/run_matched_comparison.py --models lgbm mlp --weeks 2 4 6 8 \\
      --seeds 42 123 7 17 99 --splits random lcpo --feature-set A

  # Quick smoke test (one week, one seed, random only):
  PYTHONPATH=src python src/run_matched_comparison.py \\
      --models lgbm --weeks 8 --seeds 42 --splits random --feature-set B
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
    PIPELINE_VERSION,
    PREDICTION_WINDOWS,
    RESULTS_DIR,
)
from oulad_data import (
    build_features,
    exclude_early_unregistered,
    filter_window,
    load_oulad_data,
    sanitize_feature_names,
)
from split_loader import load_lcpo_folds, load_lcpo_masks, load_random_split

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_MATCHED_DIR = RESULTS_DIR / "matched"

DEFAULT_MODEL_SEEDS = [42, 123, 7, 17, 99]
_DEDUP_KEYS = ["model", "week", "split_type", "seed", "model_seed", "fold", "feature_set"]

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

# The six behavioral summaries (feature set S). Everything else in the base set
# (num_of_prev_attempts, studied_credits, age-band dummies) is enrollment metadata.
_SIX_SUMMARY_COLS = [
    "vle_total",
    "vle_mean",
    "vle_std",
    "assess_mean",
    "assess_max",
    "assess_count",
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


def _load_registration():
    """Load and cache studentRegistration.csv from DATA_DIR."""
    if "registration" not in _DATA_CACHE:
        _DATA_CACHE["registration"] = pd.read_csv(DATA_DIR / "studentRegistration.csv")
    return _DATA_CACHE["registration"]


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

# Demographic columns in studentInfo that are one-hot encoded for set B/D/E.
_DEMOGRAPHIC_COLS = ["gender", "region", "highest_education", "imd_band", "disability"]

# Course identity columns one-hot encoded for set C/D/E.
_COURSE_IDENTITY_COLS = ["code_module", "code_presentation"]

# Temporal feature columns pulled from graph artifacts for set E.
_TEMPORAL_COLS = ["weeks_active", "recency", "activity_trend", "max_inactive_gap"]

# Mapping from feature-set letter to the three add_* flags.
_FEATURE_SET_FLAGS = {
    "A": dict(add_demographics=False, add_course_identity=False, add_temporal=False),
    "B": dict(add_demographics=True,  add_course_identity=False, add_temporal=False),
    "C": dict(add_demographics=False, add_course_identity=True,  add_temporal=False),
    "D": dict(add_demographics=True,  add_course_identity=True,  add_temporal=False),
    "E": dict(add_demographics=True,  add_course_identity=True,  add_temporal=True),
    "S": dict(add_demographics=False, add_course_identity=False, add_temporal=False,
              behavioral_only=True),
}

# Explicit feature set -> MLP model label (results CSV `model` column). Every set has its
# own label, so no two feature sets can land in the same matrix row.
_MLP_MODEL_NAMES = {
    "A": "MLP_all_features",
    "B": "MLP_feature_set_B",
    "C": "MLP_feature_set_C",
    "D": "MLP_feature_set_D",
    "E": "MLP_feature_set_E",
    "S": "MLP_six_summaries",
}


def build_tabular_features(
    week: int,
    data=None,
    add_demographics: bool = False,
    add_course_identity: bool = False,
    add_temporal: bool = False,
    behavioral_only: bool = False,
):
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
    add_demographics : bool
        When True, one-hot encode gender, region, highest_education, imd_band,
        and disability and append to the feature matrix (feature sets B/D/E).
    add_course_identity : bool
        When True, one-hot encode code_module and code_presentation and append
        (feature sets C/D/E).
    add_temporal : bool
        When True, merge weeks_active, recency, activity_trend, and
        max_inactive_gap from the graph enrollment artifacts and append
        (feature set E).  Requires the parquet/CSV artifacts for *week* to
        exist under GRAPH_ARTIFACTS_DIR.  After Sub-Task 6, inactive students
        already carry the correct recency sentinel — no additional fillna needed.
    behavioral_only : bool
        When True (feature set S), keep only the six behavioral summaries
        (``_SIX_SUMMARY_COLS``): no num_of_prev_attempts, studied_credits or age
        band. Cannot be combined with the add_* options.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric only; NaNs filled to 0); feature names sanitized
        for XGBoost / LightGBM compatibility.
    y : pd.Series
        Binary at-risk labels aligned to X.
    enrollment_df : pd.DataFrame
        Full DataFrame including id_student, code_module, code_presentation and
        all other columns produced by build_features().  Passed to split_loader so the
        persisted masks are aligned by enrollment triple.
    """
    if behavioral_only and (add_demographics or add_course_identity or add_temporal):
        raise ValueError("behavioral_only (feature set S) cannot be combined with add_* options")

    if data is None:
        student_info, student_vle, student_assess, assessments = _load_raw_data()
    else:
        student_info, student_vle, student_assess, assessments = data

    # week * 7 converts week number to day-from-course-start (see PREDICTION_WINDOWS)
    window = PREDICTION_WINDOWS[f"week_{week}"]

    # Drop enrollments already unregistered at the cutoff so the population
    # matches the GNN pipeline (see graph_pipeline.apply_window_cutoff).
    student_info, n_removed = exclude_early_unregistered(
        student_info, _load_registration(), window
    )
    print(f"Week {week}: removed {n_removed} already-unregistered enrollments")

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
    if behavioral_only:
        missing = [c for c in _SIX_SUMMARY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Feature set S needs {missing}, absent from build_features() output")
        feature_cols = list(_SIX_SUMMARY_COLS)
    else:
        base_cols = [c for c in _BASE_FEATURE_COLS if c in df.columns]
        feature_cols = base_cols + age_dummy_cols

    # ------------------------------------------------------------------
    # Optional: demographics one-hots (feature sets B / D / E)
    # ------------------------------------------------------------------
    if add_demographics:
        demo_src_cols = [c for c in _DEMOGRAPHIC_COLS if c in df.columns]
        if demo_src_cols:
            demo_dummies = pd.get_dummies(
                df[demo_src_cols], prefix=demo_src_cols, drop_first=False
            )
            demo_dummies = sanitize_feature_names(demo_dummies)
            df = pd.concat([df, demo_dummies], axis=1)
            feature_cols = feature_cols + list(demo_dummies.columns)

    # ------------------------------------------------------------------
    # Optional: course identity one-hots (feature sets C / D / E)
    # ------------------------------------------------------------------
    if add_course_identity:
        ci_src_cols = [c for c in _COURSE_IDENTITY_COLS if c in df.columns]
        if ci_src_cols:
            ci_dummies = pd.get_dummies(
                df[ci_src_cols], prefix=ci_src_cols, drop_first=False
            )
            ci_dummies = sanitize_feature_names(ci_dummies)
            df = pd.concat([df, ci_dummies], axis=1)
            feature_cols = feature_cols + list(ci_dummies.columns)

    # ------------------------------------------------------------------
    # Optional: temporal features from graph artifacts (feature set E)
    # ------------------------------------------------------------------
    if add_temporal:
        enroll_key = f"week{week:02d}_edges_enrolled_in"
        artifact_base = GRAPH_ARTIFACTS_DIR / enroll_key
        artifact_path = None
        for ext in (".parquet", ".csv"):
            candidate = artifact_base.with_suffix(ext)
            if candidate.exists():
                artifact_path = candidate
                break
        if artifact_path is None:
            raise FileNotFoundError(
                f"Temporal features requested (feature set E) but no artifact found at "
                f"{artifact_base}.parquet / .csv — run graph_pipeline first."
            )
        if artifact_path.suffix == ".parquet":
            temporal_df = pd.read_parquet(artifact_path)
        else:
            temporal_df = pd.read_csv(artifact_path)

        # The artifact uses integer node indices (src/dst), not the original keys.
        # Load the student / course-presentation node tables to reconstruct the join.
        node_stu_path = None
        for ext in (".parquet", ".csv"):
            candidate = GRAPH_ARTIFACTS_DIR / f"week{week:02d}_nodes_student{ext}"
            if candidate.exists():
                node_stu_path = candidate
                break
        node_cp_path = None
        for ext in (".parquet", ".csv"):
            candidate = GRAPH_ARTIFACTS_DIR / f"week{week:02d}_nodes_course_presentation{ext}"
            if candidate.exists():
                node_cp_path = candidate
                break

        if node_stu_path is not None and node_cp_path is not None:
            # Re-attach id_student / code_module / code_presentation from node tables
            read = lambda p: pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            stu_nodes = read(node_stu_path)[["id_student"]].reset_index().rename(columns={"index": "src"})
            cp_nodes = read(node_cp_path)[["code_module", "code_presentation"]].reset_index().rename(columns={"index": "dst"})
            temporal_df = temporal_df.merge(stu_nodes, on="src", how="left")
            temporal_df = temporal_df.merge(cp_nodes, on="dst", how="left")

        # Select only the four temporal columns plus the join keys, then merge.
        join_keys = ["id_student", "code_module", "code_presentation"]
        temporal_cols_present = [c for c in _TEMPORAL_COLS if c in temporal_df.columns]
        if temporal_cols_present and all(k in temporal_df.columns for k in join_keys):
            temporal_subset = temporal_df[join_keys + temporal_cols_present].drop_duplicates(join_keys)
            # Ensure join keys exist in df (they should — code_module / code_presentation
            # come from student_info; id_student is always present).
            df = df.merge(temporal_subset, how="left", on=join_keys)
            feature_cols = feature_cols + temporal_cols_present
        else:
            print(
                "Warning: temporal artifact found but expected columns missing; "
                "temporal features skipped."
            )

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
    feature_set: str = "A",
    model_seeds: list = None,
) -> None:
    """Run matched LightGBM for all weeks and split types.

    Parameters
    ----------
    weeks       : list of int — prediction weeks to evaluate.
    seeds       : list of int — random seeds for the random-student split.
    splits      : list of str — any subset of ["random", "lcpo"].
    feature_set : str — one of "A"–"E" or "S" controlling which features are used.
    model_seeds : list of int — model-initialisation seeds for LCPO folds (one fit per
                  seed per fold; default 42 123 7 17 99). Not the split seed: the LCPO
                  folds are fixed, so ``seed`` is -1 in LCPO rows and the init seed is
                  recorded in ``model_seed``. Random-split rows use ``seed`` for both.
    """
    if model_seeds is None:
        model_seeds = DEFAULT_MODEL_SEEDS
    flags = _FEATURE_SET_FLAGS[feature_set]
    out_path = _MATCHED_DIR / "lgbm_matched_results.csv"
    all_rows = []

    for week in weeks:
        print(f"\n=== LightGBM matched — week {week:02d}  feature_set={feature_set} ===")
        X, y, enrollment_df = build_tabular_features(week, **flags)

        # ------------------------------------------------------------------
        # Random-student split
        # ------------------------------------------------------------------
        if "random" in splits:
            print(f"  [random split] seeds: {seeds}")
            for seed in seeds:
                train_mask, val_mask, test_mask = load_random_split(
                    week, seed, frame=enrollment_df
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
                    "model_seed": seed,
                    "fold": -1,
                    "feature_set": feature_set,
                    **metrics,
                    "pipeline_version": PIPELINE_VERSION,
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
            folds_df = load_lcpo_folds(week)
            print(f"  [LCPO] {len(folds_df)} folds, model seeds: {model_seeds}")

            for _, fold_row in folds_df.iterrows():
                fold_idx = int(fold_row["fold_idx"])
                held_mod = fold_row["held_out_module"]
                held_pres = fold_row["held_out_presentation"]

                # Persisted train/val/test masks shared with the GNN (save_graph_splits.py)
                tr_s, va_s, te_s = load_lcpo_masks(week, fold_idx, frame=enrollment_df)
                train_mask_lc = tr_s.to_numpy()
                val_mask_lc = va_s.to_numpy()
                test_mask_lc = te_s.to_numpy()
                held = enrollment_df.loc[test_mask_lc]
                assert (held["code_module"] == held_mod).all()
                assert (held["code_presentation"] == held_pres).all()

                X_train = X[train_mask_lc]
                y_train = y[train_mask_lc]
                X_val = X[val_mask_lc]
                y_val = y[val_mask_lc]
                X_test = X[test_mask_lc]
                y_test = y[test_mask_lc]

                if y_test.sum() == 0 or (len(y_test) - y_test.sum()) == 0:
                    print(f"    fold {fold_idx:02d}: SKIP (single class in test)")
                    continue

                for mseed in model_seeds:
                    model = LGBMClassifier(n_estimators=200, random_state=mseed, verbose=-1)
                    model.fit(X_train, y_train)

                    val_proba = model.predict_proba(X_val)[:, 1]
                    if y_val.sum() > 0 and (len(y_val) - y_val.sum()) > 0:
                        threshold = _select_threshold(val_proba, y_val.values)
                    else:
                        threshold = 0.5

                    test_proba = model.predict_proba(X_test)[:, 1]
                    test_pred = (test_proba >= threshold).astype(int)
                    metrics = _compute_metrics(y_test.values, test_pred, test_proba, threshold)

                    all_rows.append({
                        "model": "LightGBM_matched",
                        "week": week,
                        "split_type": "lcpo",
                        "seed": -1,
                        "model_seed": mseed,
                        "fold": fold_idx,
                        "feature_set": feature_set,
                        **metrics,
                        "pipeline_version": PIPELINE_VERSION,
                    })
                    print(
                        f"    fold {fold_idx:02d} ({held_mod}/{held_pres}) mseed={mseed}"
                        f"  auroc={metrics['auroc']:.4f}"
                        f"  f1={metrics['f1']:.4f}"
                    )

    if not all_rows:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_rows)
    _append_or_create_csv(
        results_df, out_path,
        dedup_keys=_DEDUP_KEYS,
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
    feature_set: str = "A",
    model_seeds: list = None,
) -> None:
    """Run matched MLP for all weeks and split types.

    Architecture: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
    early_stopping=True, validation_fraction=0.1).
    Normalization: StandardScaler fit on training rows only, applied to all splits.

    Parameters
    ----------
    weeks       : list of int — prediction weeks to evaluate.
    seeds       : list of int — random seeds for the random-student split.
    splits      : list of str — any subset of ["random", "lcpo"].
    feature_set : str — one of "A"–"E" or "S" controlling which features are used.
    model_seeds : list of int — model-initialisation seeds for LCPO folds (one fit per
                  seed per fold; default 42 123 7 17 99). Not the split seed: the LCPO
                  folds are fixed, so ``seed`` is -1 in LCPO rows and the init seed is
                  recorded in ``model_seed``. Random-split rows use ``seed`` for both.
    """
    if model_seeds is None:
        model_seeds = DEFAULT_MODEL_SEEDS
    flags = _FEATURE_SET_FLAGS[feature_set]
    # One label per feature set (see _MLP_MODEL_NAMES) so update_results_matrix.py routes
    # A -> row 16 (all features) and S -> row 17 (six summaries only) without collisions.
    mlp_model_name = _MLP_MODEL_NAMES[feature_set]
    out_path = _MATCHED_DIR / "mlp_matched_results.csv"
    all_rows = []

    for week in weeks:
        print(f"\n=== MLP matched — week {week:02d}  feature_set={feature_set} ===")
        X, y, enrollment_df = build_tabular_features(week, **flags)

        # ------------------------------------------------------------------
        # Random-student split
        # ------------------------------------------------------------------
        if "random" in splits:
            print(f"  [random split] seeds: {seeds}")
            for seed in seeds:
                train_mask, val_mask, test_mask = load_random_split(
                    week, seed, frame=enrollment_df
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
                    "model": mlp_model_name,
                    "week": week,
                    "split_type": "random",
                    "seed": seed,
                    "model_seed": seed,
                    "fold": -1,
                    "feature_set": feature_set,
                    **metrics,
                    "pipeline_version": PIPELINE_VERSION,
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
            folds_df = load_lcpo_folds(week)
            print(f"  [LCPO] {len(folds_df)} folds, model seeds: {model_seeds}")

            for _, fold_row in folds_df.iterrows():
                fold_idx = int(fold_row["fold_idx"])
                held_mod = fold_row["held_out_module"]
                held_pres = fold_row["held_out_presentation"]

                # Persisted train/val/test masks shared with the GNN (save_graph_splits.py)
                tr_s, va_s, te_s = load_lcpo_masks(week, fold_idx, frame=enrollment_df)
                train_mask_lc = tr_s.to_numpy()
                val_mask_lc = va_s.to_numpy()
                test_mask_lc = te_s.to_numpy()
                held = enrollment_df.loc[test_mask_lc]
                assert (held["code_module"] == held_mod).all()
                assert (held["code_presentation"] == held_pres).all()

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

                for mseed in model_seeds:
                    model = MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=mseed,
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

                    all_rows.append({
                        "model": mlp_model_name,
                        "week": week,
                        "split_type": "lcpo",
                        "seed": -1,
                        "model_seed": mseed,
                        "fold": fold_idx,
                        "feature_set": feature_set,
                        **metrics,
                        "pipeline_version": PIPELINE_VERSION,
                    })
                    print(
                        f"    fold {fold_idx:02d} ({held_mod}/{held_pres}) mseed={mseed}"
                        f"  auroc={metrics['auroc']:.4f}"
                        f"  f1={metrics['f1']:.4f}"
                    )

    if not all_rows:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_rows)
    _append_or_create_csv(
        results_df, out_path,
        dedup_keys=_DEDUP_KEYS,
    )
    print(f"\nResults written to {out_path}")
    _print_summary(results_df, mlp_model_name)


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
        "--model-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_MODEL_SEEDS,
        dest="model_seeds",
        help="Model-init seeds for LCPO folds (default: 42 123 7 17 99). Not the split seed.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["random", "lcpo"],
        default=["random", "lcpo"],
        help="Which split types to run (default: random lcpo).",
    )
    parser.add_argument(
        "--feature-set",
        choices=["A", "B", "C", "D", "E", "S"],
        default="A",
        dest="feature_set",
        help=(
            "Feature configuration to use (default: A — original matched set). "
            "A=base, B=A+demographics, C=A+course_identity, D=A+demo+course_id, "
            "E=D+temporal, S=six behavioral summaries only."
        ),
    )

    args = parser.parse_args()
    weeks = args.weeks
    seeds = args.seeds
    splits = args.splits
    feature_set = args.feature_set

    print(f"Models      : {args.models}")
    print(f"Weeks       : {weeks}")
    print(f"Seeds       : {seeds}  (split seeds)")
    print(f"Model seeds : {args.model_seeds}  (LCPO init seeds)")
    print(f"Splits      : {splits}")
    print(f"Feature set : {feature_set}")

    if "lgbm" in args.models:
        run_lgbm_matched(
            weeks=weeks, seeds=seeds, splits=splits, feature_set=feature_set,
            model_seeds=args.model_seeds,
        )

    if "mlp" in args.models:
        run_mlp_matched(
            weeks=weeks, seeds=seeds, splits=splits, feature_set=feature_set,
            model_seeds=args.model_seeds,
        )


if __name__ == "__main__":
    main()
