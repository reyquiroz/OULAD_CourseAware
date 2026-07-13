"""
OULAD Evaluation Pipeline — shared module for all split strategies.

Label convention (uniform across all evaluations):
  1 = at-risk (Fail / Withdrawn)  — positive class for early-intervention
  0 = success (Pass / Distinction) — negative class

Public API
----------
get_models()
    Return dict of model-name → fitted-able sklearn estimator.

get_feature_subsets(df)
    Return dict of subset-name → list of column names.

evaluate_split(model, X_train, y_train, X_test, y_test)
    Train on train, predict on test, return metrics dict.

run_random_student_evaluation(datasets)
    5-fold GroupKFold CV on id_student across all weeks/models/feature-subsets.
    Returns a detailed DataFrame (one row per week × model × feature-subset × fold).

run_lcpo_evaluation(datasets)
    Leave-Course-Presentation-Out across all 22 course-presentations × weeks × models.
    Returns a detailed DataFrame (one row per week × model × course-presentation).

run_future_presentation_evaluation(datasets)
    Temporal split: train on 2013B/2013J/2014B, test on 2014J.
    Returns a detailed DataFrame (one row per week × model).

build_unified_comparison_table(random_df, lcpo_df, future_df)
    Merge best-per-week results from all three strategies into a single
    DataFrame keyed by (Week, Model, Split).

analyze_course_difficulty(lcpo_df, output_dir)
    Aggregate AUROC per course-presentation, sort hardest → easiest,
    save CSV and a matplotlib boxplot PNG.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    LCPO_RESULTS_DIR,
    MODEL_PARAMS,
    PREDICTION_WINDOWS,
    RANDOM_STATE,
)
from oulad_data import (
    evaluate_metrics,
    lcpo_split,
    sanitize_feature_names,
)

warnings.filterwarnings("ignore")

# Presentation codes for the temporal (future-presentation) split
_TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
_TEST_PRESENTATIONS = ["2014J"]


# ---------------------------------------------------------------------------
# Model and feature-subset helpers
# ---------------------------------------------------------------------------

def get_models():
    """Return a fresh dict of model-name → sklearn estimator.

    Uses canonical hyperparameters from config.MODEL_PARAMS.  A new dict is
    returned on each call so callers always get unfitted estimator instances.
    """
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            **MODEL_PARAMS["logistic_regression"]
        ),
        "RandomForest": RandomForestClassifier(
            **MODEL_PARAMS["random_forest"]
        ),
        "XGBoost": XGBClassifier(
            **MODEL_PARAMS["xgboost"]
        ),
        "LightGBM": LGBMClassifier(
            **MODEL_PARAMS["lightgbm"]
        ),
    }


def get_feature_subsets(df):
    """Return a dict of feature-subset name → list of column names.

    The four subsets are derived dynamically from the column names of *df*:
    - ``VLE_only``       columns containing ``"vle_"``
    - ``Assessment_only``  columns containing ``"assess_"``
    - ``VLE+Assessment``   union of above
    - ``All_features``     all non-metadata columns
    """
    vle_cols = [c for c in df.columns if "vle_" in c]
    assess_cols = [c for c in df.columns if "assess_" in c]
    # demographics = everything that isn't VLE, assessment, or metadata
    meta_cols = {"target", "id_student", "final_result", "code_module", "code_presentation"}
    demo_cols = [c for c in df.columns if c not in vle_cols + assess_cols and c not in meta_cols]

    return {
        "VLE_only": vle_cols,
        "Assessment_only": assess_cols,
        "VLE+Assessment": vle_cols + assess_cols,
        "All_features": vle_cols + assess_cols + demo_cols,
    }


# ---------------------------------------------------------------------------
# Generic single-split evaluator
# ---------------------------------------------------------------------------

def evaluate_split(model, X_train, y_train, X_test, y_test):
    """Train *model* on train data, evaluate on test data.

    Args:
        model:    An unfitted sklearn-compatible estimator.
        X_train:  Feature matrix for training.
        y_train:  Binary labels for training.
        X_test:   Feature matrix for evaluation.
        y_test:   Binary labels for evaluation.

    Returns:
        dict with keys AUROC, AUPRC, F1, Precision, Recall, Balanced_Acc.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # predict_proba: column 1 = probability of positive class (at-risk)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    return evaluate_metrics(y_test, y_pred, y_proba)


# ---------------------------------------------------------------------------
# Random student evaluation (5-fold GroupKFold CV)
# ---------------------------------------------------------------------------

def run_random_student_evaluation(datasets, weeks=(2, 4, 6, 8)):
    """5-fold GroupKFold CV on id_student for all weeks, models, and feature subsets.

    The same student never appears in both train and test within a fold.
    This guarantees true student-level separation.

    Args:
        datasets: dict of week (int) → DataFrame as returned by
                  ``oulad_data.create_datasets()``.
        weeks:    Iterable of week numbers to evaluate (default: 2, 4, 6, 8).

    Returns:
        DataFrame with columns:
        Week, Model, Features, Fold, N_train, N_test, N_features,
        AUROC, AUPRC, F1, Precision, Recall, Balanced_Acc.
    """
    rows = []

    for week in weeks:
        print(f"\n{'='*60}")
        print(f"Random-Student CV — Week {week}")
        print(f"{'='*60}")

        df = datasets[week]
        subsets = get_feature_subsets(df)
        models = get_models()

        y = df["target"].values
        student_ids = df["id_student"].values

        # Build GroupKFold group labels — shuffle students first for balanced folds
        rng = np.random.default_rng(RANDOM_STATE)
        unique_students = np.unique(student_ids)
        shuffled = rng.permutation(unique_students)
        n_folds = 5
        fold_label = {sid: int(i % n_folds) for i, sid in enumerate(shuffled)}
        groups = np.array([fold_label[sid] for sid in student_ids])

        gkf = GroupKFold(n_splits=n_folds)

        # Full encoded matrix (needed to get consistent column space per subset)
        X_full_raw = df.drop(
            columns=["target", "id_student", "final_result", "code_module", "code_presentation"],
            errors="ignore",
        )
        X_full_enc = sanitize_feature_names(pd.get_dummies(X_full_raw))

        for model_name, model in models.items():
            print(f"  {model_name}:")

            for subset_name, feat_cols in subsets.items():
                # Majority baseline: run only once (All_features)
                if model_name == "Majority" and subset_name != "All_features":
                    continue

                available = [c for c in feat_cols if c in X_full_raw.columns]
                if not available:
                    continue

                X_sub_raw = X_full_raw[available]
                X_sub = sanitize_feature_names(pd.get_dummies(X_sub_raw))

                fold_aucs = []
                for fold_idx, (train_idx, test_idx) in enumerate(
                    gkf.split(X_sub, y, groups=groups)
                ):
                    X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]

                    try:
                        from sklearn.base import clone
                        m = clone(model)
                        metrics = evaluate_split(m, X_tr, y_tr, X_te, y_te)
                    except Exception as exc:
                        print(f"    fold {fold_idx} error: {exc}")
                        continue

                    row = {
                        "Week": week,
                        "Model": model_name,
                        "Features": subset_name,
                        "Fold": fold_idx,
                        "N_train": len(train_idx),
                        "N_test": len(test_idx),
                        "N_features": X_sub.shape[1],
                    }
                    row.update(metrics)
                    rows.append(row)
                    fold_aucs.append(metrics["AUROC"])

                if fold_aucs:
                    print(
                        f"    {subset_name}: AUROC={np.mean(fold_aucs):.3f}"
                        f"±{np.std(fold_aucs):.3f}"
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LCPO evaluation
# ---------------------------------------------------------------------------

def run_lcpo_evaluation(datasets, weeks=(2, 4, 6, 8)):
    """Leave-Course-Presentation-Out evaluation across all course-presentations.

    For each unique (code_module, code_presentation) pair, holds it out as the
    test set and trains on all remaining enrollments.  Uses the canonical
    ``lcpo_split()`` from ``oulad_data`` to build the masks.

    Args:
        datasets: dict of week (int) → DataFrame.
        weeks:    Iterable of week numbers to evaluate.

    Returns:
        DataFrame with columns:
        Week, Model, Test_Module, Test_Presentation, N_train, N_test,
        AUROC, AUPRC, F1, Precision, Recall, Balanced_Acc.
    """
    rows = []

    for week in weeks:
        print(f"\n{'='*60}")
        print(f"LCPO Evaluation — Week {week}")
        print(f"{'='*60}")

        df = datasets[week]
        models = get_models()
        # LCPO does not use Majority baseline
        models.pop("Majority", None)

        course_presentations = (
            df[["code_module", "code_presentation"]]
            .drop_duplicates()
            .sort_values(["code_module", "code_presentation"])
        )

        for _, cp_row in course_presentations.iterrows():
            module = cp_row["code_module"]
            presentation = cp_row["code_presentation"]

            train_mask, test_mask = lcpo_split(df, module, presentation)
            df_train = df[train_mask]
            df_test = df[test_mask]

            # Build feature matrices — All_features only for LCPO
            meta = ["target", "id_student", "final_result", "code_module", "code_presentation"]
            X_tr_raw = df_train.drop(columns=meta, errors="ignore")
            X_te_raw = df_test.drop(columns=meta, errors="ignore")

            # Align columns after one-hot encoding
            X_tr_enc = sanitize_feature_names(pd.get_dummies(X_tr_raw))
            X_te_enc = sanitize_feature_names(pd.get_dummies(X_te_raw))
            X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join="left", axis=1, fill_value=0)

            y_train = df_train["target"].values
            y_test = df_test["target"].values

            # Skip if test set has only one class
            if len(np.unique(y_test)) < 2:
                print(f"  Skipping {module}/{presentation}: single class in test set")
                continue

            for model_name, model in models.items():
                try:
                    from sklearn.base import clone
                    m = clone(model)
                    metrics = evaluate_split(m, X_tr_enc, y_train, X_te_enc, y_test)
                except Exception as exc:
                    print(f"  {module}/{presentation} {model_name} error: {exc}")
                    continue

                row = {
                    "Week": week,
                    "Model": model_name,
                    "Test_Module": module,
                    "Test_Presentation": presentation,
                    "N_train": len(df_train),
                    "N_test": len(df_test),
                }
                row.update(metrics)
                rows.append(row)

            print(f"  {module}/{presentation}: {len(df_test)} test enrollments")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Future-presentation evaluation
# ---------------------------------------------------------------------------

def run_future_presentation_evaluation(datasets, weeks=(2, 4, 6, 8)):
    """Temporal split: train on earlier presentations, test on 2014J.

    Train presentations: 2013B, 2013J, 2014B
    Test  presentation:  2014J

    Args:
        datasets: dict of week (int) → DataFrame.
        weeks:    Iterable of week numbers to evaluate.

    Returns:
        DataFrame with columns:
        Week, Model, Split, Train_Presentations, Test_Presentations,
        N_train, N_test, AUROC, AUPRC, F1, Precision, Recall, Balanced_Acc.
    """
    rows = []

    for week in weeks:
        print(f"\n{'='*60}")
        print(f"Future-Presentation Evaluation — Week {week}")
        print(f"{'='*60}")

        df = datasets[week]
        models = get_models()
        # Future-presentation does not use Majority baseline
        models.pop("Majority", None)

        train_mask = df["code_presentation"].isin(_TRAIN_PRESENTATIONS)
        test_mask = df["code_presentation"].isin(_TEST_PRESENTATIONS)

        df_train = df[train_mask]
        df_test = df[test_mask]

        if len(df_test) < 50:
            print(f"  Skipping Week {week}: test set too small ({len(df_test)} rows)")
            continue

        y_train = df_train["target"].values
        y_test = df_test["target"].values

        if len(np.unique(y_test)) < 2:
            print(f"  Skipping Week {week}: single class in test set")
            continue

        meta = ["target", "id_student", "final_result", "code_module", "code_presentation"]
        X_tr_raw = df_train.drop(columns=meta, errors="ignore")
        X_te_raw = df_test.drop(columns=meta, errors="ignore")

        X_tr_enc = sanitize_feature_names(pd.get_dummies(X_tr_raw))
        X_te_enc = sanitize_feature_names(pd.get_dummies(X_te_raw))
        X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join="left", axis=1, fill_value=0)

        print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")

        for model_name, model in models.items():
            try:
                from sklearn.base import clone
                m = clone(model)
                metrics = evaluate_split(m, X_tr_enc, y_train, X_te_enc, y_test)
            except Exception as exc:
                print(f"  {model_name} error: {exc}")
                continue

            row = {
                "Week": week,
                "Model": model_name,
                "Split": "Future-Presentation",
                "Train_Presentations": ", ".join(_TRAIN_PRESENTATIONS),
                "Test_Presentations": ", ".join(_TEST_PRESENTATIONS),
                "N_train": len(df_train),
                "N_test": len(df_test),
            }
            row.update(metrics)
            rows.append(row)
            print(f"  {model_name}: AUROC={metrics['AUROC']:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unified comparison table
# ---------------------------------------------------------------------------

def build_unified_comparison_table(random_df, lcpo_df, future_df):
    """Merge per-week best results from all three splits into one DataFrame.

    For the random split the mean across folds and the All_features subset is
    used.  For LCPO the mean across course-presentations is used.
    For future-presentation, no aggregation is needed (single train/test pair).

    Returns:
        DataFrame with columns:
        Week, Model, Split, AUROC_mean, AUROC_std, F1_mean, F1_std,
        Precision_mean, Precision_std, Recall_mean, Recall_std,
        Balanced_Acc_mean, Balanced_Acc_std.
    """
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]

    # --- Random split: aggregate All_features rows across folds ---
    rand_all = random_df[random_df["Features"] == "All_features"].copy()
    rand_agg = (
        rand_all.groupby(["Week", "Model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    rand_agg.columns = ["Week", "Model"] + [
        f"{m}_{s}" for m, s in rand_agg.columns[2:]
    ]
    rand_agg["Split"] = "Random-Student"

    # --- LCPO: aggregate across course-presentations ---
    lcpo_agg = (
        lcpo_df.groupby(["Week", "Model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    lcpo_agg.columns = ["Week", "Model"] + [
        f"{m}_{s}" for m, s in lcpo_agg.columns[2:]
    ]
    lcpo_agg["Split"] = "LCPO"

    # --- Future-presentation: std=0 (single evaluation per week/model) ---
    fp = future_df.copy()
    for m in metrics:
        fp[f"{m}_mean"] = fp[m]
        fp[f"{m}_std"] = 0.0
    fp["Split"] = "Future-Presentation"
    fp_agg = fp[["Week", "Model", "Split"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]]

    unified = pd.concat([rand_agg, lcpo_agg, fp_agg], ignore_index=True)
    unified = unified.sort_values(["Week", "Split", "Model"]).reset_index(drop=True)
    return unified


# ---------------------------------------------------------------------------
# Course difficulty analysis
# ---------------------------------------------------------------------------

def analyze_course_difficulty(lcpo_df, output_dir=None):
    """Aggregate per-course AUROC from LCPO results and rank by difficulty.

    Aggregates AUROC across all models and weeks for each course-presentation,
    computes mean ± std, sorts ascending (hardest first), saves a CSV and a
    matplotlib boxplot.

    Args:
        lcpo_df:    DataFrame returned by ``run_lcpo_evaluation()``.
        output_dir: Path-like where CSV and PNG are saved.
                    Defaults to ``config.LCPO_RESULTS_DIR``.

    Returns:
        DataFrame with columns:
        Course_Presentation, AUROC_mean, AUROC_std, N_folds,
        sorted ascending by AUROC_mean.
    """
    if output_dir is None:
        output_dir = LCPO_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lcpo_df = lcpo_df.copy()
    lcpo_df["Course_Presentation"] = (
        lcpo_df["Test_Module"] + "/" + lcpo_df["Test_Presentation"]
    )

    agg = (
        lcpo_df.groupby("Course_Presentation")["AUROC"]
        .agg(AUROC_mean="mean", AUROC_std="std", N_folds="count")
        .reset_index()
        .sort_values("AUROC_mean", ascending=True)
        .reset_index(drop=True)
    )

    # Save CSV
    csv_path = output_dir / "course_presentation_difficulty.csv"
    agg.to_csv(csv_path, index=False)
    print(f"  ✓ Saved course difficulty CSV → {csv_path}")

    # --- Boxplot ---
    # Collect per-course AUROC values (across models and weeks)
    cp_auroc = (
        lcpo_df.groupby("Course_Presentation")["AUROC"]
        .apply(list)
        .reindex(agg["Course_Presentation"])  # preserve difficulty order
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(
        cp_auroc.values,
        tick_labels=cp_auroc.index,
        vert=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    ax.set_xlabel("Course Presentation", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(
        "Per-Course-Presentation AUROC Distribution (LCPO, all models & weeks)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axhline(0.5, linestyle="--", color="red", linewidth=0.8, label="Random baseline")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / "course_difficulty_chart.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved course difficulty chart → {png_path}")

    return agg
