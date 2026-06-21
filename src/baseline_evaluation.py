"""
OULAD Baseline Analysis - Enhanced Version
Complete implementation with all required improvements

Label Convention (CORRECTED):
- 1 = at-risk (Fail/Withdrawn) - positive class for intervention
- 0 = success (Pass/Distinction) - negative class
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# Import configuration
from config import (
    DATA_DIR,
    BASELINE_RESULTS_DIR,
    MODEL_PARAMS,
    PREDICTION_WINDOWS,
    LABEL_MAPPING,
    METRICS,
    MODELS,
)

warnings.filterwarnings("ignore")

# ============================================================================
# 1. DATA LOADING
# ============================================================================


def load_oulad_data(data_dir=None):
    """
    Load OULAD datasets

    Args:
        data_dir: Path to data directory (uses config.DATA_DIR if None)

    Returns:
        Tuple of (student_info, student_vle, student_assess, assessments)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    print("Loading OULAD data...")
    student_info = pd.read_csv(data_dir / "studentInfo.csv")
    student_vle = pd.read_csv(data_dir / "studentVle.csv")
    student_assess = pd.read_csv(data_dir / "studentAssessment.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")

    # Define risk label (CORRECTED):
    # 1 = at-risk (Fail/Withdrawn) - students who need intervention
    # 0 = success (Pass/Distinction) - students on track
    student_info["target"] = student_info["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    print(f"Loaded {len(student_info)} students")
    print(f"Target distribution:")
    print(f"  At-risk (1): {(student_info['target'] == 1).sum()} students")
    print(f"  Success (0): {(student_info['target'] == 0).sum()} students")

    return student_info, student_vle, student_assess, assessments


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================


def filter_window(vle, assess, assessments, window):
    """Filter data up to specified day (leakage-safe)"""
    vle_w = vle[vle["date"] <= window]
    assess = assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_w = assess[assess["date"] <= window]
    return vle_w, assess_w


def build_features(vle_w, assess_w, student_info):
    """Build feature set from VLE and assessment data"""
    # VLE features
    vle = vle_w.groupby(["id_student", "code_module", "code_presentation"]).agg(
        {"sum_click": ["sum", "mean", "std"]}
    )
    vle.columns = ["vle_total", "vle_mean", "vle_std"]
    vle = vle.reset_index()

    # Assessment features
    assess = assess_w.groupby(["id_student", "code_module", "code_presentation"]).agg(
        {"score": ["mean", "max"], "date": "count"}
    )
    assess.columns = ["assess_mean", "assess_max", "assess_count"]
    assess = assess.reset_index()

    # Merge all
    df = vle.merge(
        assess, how="left", on=["id_student", "code_module", "code_presentation"]
    )
    df = df.merge(
        student_info, on=["id_student", "code_module", "code_presentation"], how="left"
    )

    # Handle missing values by type
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.columns.difference(num_cols)

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def create_datasets(
    student_info, student_vle, student_assess, assessments, weeks=[2, 4, 6, 8]
):
    """Create datasets for multiple prediction windows"""
    datasets = {}
    for w in weeks:
        print(f"Building features for week {w}...")
        vle_w, assess_w = filter_window(student_vle, student_assess, assessments, w * 7)
        df = build_features(vle_w, assess_w, student_info)
        datasets[w] = df
        print(f"  Week {w}: {df.shape[0]} samples, {df.shape[1]} features")
    return datasets


# ============================================================================
# 3. EVALUATION FRAMEWORK
# ============================================================================


def get_all_metrics():
    """Define all evaluation metrics"""
    return {
        "AUROC": "roc_auc",
        "AUPRC": "average_precision",
        "F1": make_scorer(f1_score),
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "Balanced_Acc": make_scorer(balanced_accuracy_score),
    }


def evaluate_model_cv(model, X, y, cv=5):
    """Evaluate model with cross-validation"""
    scoring = get_all_metrics()
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=False,
    )

    results = {}
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        results[metric] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    return results


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
