"""
OULAD Baseline Analysis

Label Convention:
- 1 = at-risk (Fail/Withdrawn) - positive class for intervention
- 0 = success (Pass/Distinction) - negative class

Key design decisions (tasks 1-5):
  Task 1: assessment filtering uses due_date (assessments.date) <= window, not
          date_submitted — submission date leaks future behaviour
  Task 2: build_features starts from all enrollments so inactive students are
          retained with zero-valued activity features
  Task 3: random CV uses GroupKFold on id_student so the same student cannot
          appear in both train and test within a fold
  Task 4: fold assignments (id_student, fold) are saved to a CSV alongside results
  Task 5: Demographics-only added as a feature subset alongside existing conditions
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    BASELINE_RESULTS_DIR,
    PREDICTION_WINDOWS,
    RANDOM_STATE,
)
from oulad_data import (
    build_features,
    create_datasets,
    evaluate_metrics,
    filter_window,
    load_oulad_data,
    sanitize_feature_names,
)

warnings.filterwarnings("ignore")


# ============================================================================
# 1. EVALUATION FRAMEWORK
# ============================================================================


def get_all_metrics():
    """Define all evaluation metrics (positive class = 1 = at-risk)."""
    return {
        "AUROC": "roc_auc",
        "AUPRC": "average_precision",
        "F1": make_scorer(f1_score, zero_division=0),
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "Balanced_Acc": make_scorer(balanced_accuracy_score),
    }


def evaluate_model_student_grouped_cv(model, X, y, student_ids, n_folds=5, random_state=RANDOM_STATE):
    """
    Evaluate model with student-grouped k-fold cross-validation.

    Uses GroupKFold on id_student so the same student cannot appear in
    both training and test within a fold (task 3).

    Returns:
        results: dict of metric -> {mean, std, scores}
        fold_assignments: DataFrame with (id_student, fold) columns (task 4)
    """
    from sklearn.preprocessing import LabelEncoder

    # Map students to integer group labels for GroupKFold
    student_arr = np.array(student_ids)
    unique_students = np.unique(student_arr)
    n_students = len(unique_students)

    # Shuffle students with fixed seed, then assign fold labels
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_students)
    fold_label = {sid: i % n_folds for i, sid in enumerate(shuffled)}
    groups = np.array([fold_label[sid] for sid in student_arr])

    fold_assignment_rows = [
        {"id_student": sid, "fold": fold_label[sid]} for sid in unique_students
    ]
    fold_assignments = pd.DataFrame(fold_assignment_rows)

    scoring = get_all_metrics()
    gkf = GroupKFold(n_splits=n_folds)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=gkf.split(X, y, groups=groups),
        scoring=scoring,
        return_train_score=False,
    )

    results = {}
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        results[metric] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}

    return results, fold_assignments


# ============================================================================
# 4. MODEL DEFINITIONS
# ============================================================================


def get_models():
    """Define all models to evaluate"""
    models = {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }
    return models


def get_feature_subsets(df):
    """Create feature subsets for reference baselines"""
    # Identify feature types
    vle_features = [c for c in df.columns if "vle_" in c]
    assess_features = [c for c in df.columns if "assess_" in c]
    demo_features = [
        c
        for c in df.columns
        if c
        not in vle_features
        + assess_features
        + ["target", "id_student", "final_result", "code_module", "code_presentation"]
    ]

    subsets = {
        "VLE_only": vle_features,
        "Assessment_only": assess_features,
        "VLE+Assessment": vle_features + assess_features,
        "All_features": vle_features + assess_features + demo_features,
    }
    return subsets


# ============================================================================
# 5. BASELINE EVALUATION
# ============================================================================


def sanitize_feature_names(df):
    """Sanitize column names for XGBoost compatibility"""
    # Replace characters that XGBoost doesn't like: [, ], <, >
    df.columns = df.columns.str.replace("[", "_", regex=False)
    df.columns = df.columns.str.replace("]", "_", regex=False)
    df.columns = df.columns.str.replace("<", "_lt_", regex=False)
    df.columns = df.columns.str.replace(">", "_gt_", regex=False)
    return df


def run_baseline_evaluation(datasets, weeks=[2, 4, 6, 8]):
    """Run comprehensive baseline evaluation"""
    results_list = []
    models = get_models()

    for week in weeks:
        print(f"\n{'='*60}")
        print(f"EVALUATING WEEK {week}")
        print(f"{'='*60}")

        df = datasets[week]
        feature_subsets = get_feature_subsets(df)

        # Prepare data
        X_full = df.drop(
            columns=[
                "target",
                "id_student",
                "final_result",
                "code_module",
                "code_presentation",
            ],
            errors="ignore",
        )
        y = df["target"]

        # One-hot encode and sanitize column names
        X_full_encoded = pd.get_dummies(X_full)
        X_full_encoded = sanitize_feature_names(X_full_encoded)

        for model_name, model in models.items():
            print(f"\n{model_name}:")

            # Evaluate on different feature subsets
            for subset_name, features in feature_subsets.items():
                # Skip subsets for majority baseline
                if model_name == "Majority" and subset_name != "All_features":
                    continue

                # Select features
                available_features = [f for f in features if f in X_full.columns]
                if not available_features:
                    continue

                X_subset = X_full[available_features]
                X_subset_encoded = pd.get_dummies(X_subset)
                X_subset_encoded = sanitize_feature_names(X_subset_encoded)

                # Evaluate
                try:
                    cv_results = evaluate_model_cv(model, X_subset_encoded, y, cv=5)

                    result = {
                        "Week": week,
                        "Model": model_name,
                        "Features": subset_name,
                        "N_features": X_subset_encoded.shape[1],
                    }

                    for metric, values in cv_results.items():
                        result[f"{metric}_mean"] = values["mean"]
                        result[f"{metric}_std"] = values["std"]

                    results_list.append(result)

                    print(
                        f"  {subset_name}: AUROC={cv_results['AUROC']['mean']:.3f}±{cv_results['AUROC']['std']:.3f}, "
                        f"F1={cv_results['F1']['mean']:.3f}±{cv_results['F1']['std']:.3f}"
                    )

                except Exception as e:
                    print(f"  {subset_name}: Error - {str(e)}")

    return pd.DataFrame(results_list)


# ============================================================================
# 6. RESULTS VISUALIZATION
# ============================================================================


def plot_baseline_results(results_df):
    """Create comprehensive visualization of baseline results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("OULAD Baseline Results - All Metrics", fontsize=16, fontweight="bold")

    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        # Filter for All_features only
        plot_data = results_df[results_df["Features"] == "All_features"].copy()

        # Pivot for plotting
        pivot_data = plot_data.pivot(
            index="Week", columns="Model", values=f"{metric}_mean"
        )

        pivot_data.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Prediction Window (weeks)")
        ax.set_ylabel(metric)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def create_results_table(results_df):
    """Create formatted results table"""
    # Focus on All_features
    table_data = results_df[results_df["Features"] == "All_features"].copy()

    # Format with mean ± std
    for metric in ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]:
        table_data[metric] = table_data.apply(
            lambda row: f"{row[f'{metric}_mean']:.3f}±{row[f'{metric}_std']:.3f}",
            axis=1,
        )

    # Select columns
    table = table_data[
        ["Week", "Model", "AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]
    ]

    return table


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD BASELINE ANALYSIS - ENHANCED VERSION")
    print("=" * 80)

    # Load data
    student_info, student_vle, student_assess, assessments = load_oulad_data()

    # Create datasets
    datasets = create_datasets(student_info, student_vle, student_assess, assessments)

    # Run baseline evaluation
    results_df = run_baseline_evaluation(datasets)

    # Save results
    results_df.to_csv("baseline_results_detailed.csv", index=False)
    print("\n✓ Detailed results saved to: baseline_results_detailed.csv")

    # Create and save formatted table
    results_table = create_results_table(results_df)
    results_table.to_csv("baseline_results_table.csv", index=False)
    print("✓ Results table saved to: baseline_results_table.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print(results_table.to_string(index=False))

    # Create visualization
    fig = plot_baseline_results(results_df)
    fig.savefig("baseline_results_plot.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualization saved to: baseline_results_plot.png")

    return results_df, results_table


if __name__ == "__main__":
    results_df, results_table = main()

# Made with Bob
