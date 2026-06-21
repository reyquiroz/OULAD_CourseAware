"""
OULAD LCPO (Leave-Course-Presentation-Out) Evaluation
Tests model generalization across different course presentations

Label Convention (CORRECTED):
- 1 = at-risk (Fail/Withdrawn) - positive class for intervention
- 0 = success (Pass/Distinction) - negative class
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# Import configuration
from config import (
    DATA_DIR,
    LCPO_RESULTS_DIR,
    MODEL_PARAMS,
    LABEL_MAPPING,
    METRICS,
    MODELS,
)

warnings.filterwarnings("ignore")


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
    print(f"  At-risk (1): {(student_info['target'] == 1).sum()} students")
    print(f"  Success (0): {(student_info['target'] == 0).sum()} students")

    return student_info, student_vle, student_assess, assessments


def filter_window(vle, assess, assessments, window):
    """Filter data up to specified day"""
    vle_w = vle[vle["date"] <= window]
    assess = assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_w = assess[assess["date"] <= window]
    return vle_w, assess_w


def build_features(vle_w, assess_w, student_info):
    """Build feature set"""
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

    # Merge
    df = vle.merge(
        assess, how="left", on=["id_student", "code_module", "code_presentation"]
    )
    df = df.merge(
        student_info, on=["id_student", "code_module", "code_presentation"], how="left"
    )

    # Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.columns.difference(num_cols)
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def sanitize_feature_names(df):
    """Sanitize column names for XGBoost"""
    df.columns = df.columns.str.replace("[", "_", regex=False)
    df.columns = df.columns.str.replace("]", "_", regex=False)
    df.columns = df.columns.str.replace("<", "_lt_", regex=False)
    df.columns = df.columns.str.replace(">", "_gt_", regex=False)
    return df


def get_models():
    """Define models"""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }


def evaluate_metrics(y_true, y_pred, y_proba):
    """Calculate all metrics"""
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
    }


def lcpo_evaluation(datasets, week=8):
    """
    Leave-Course-Presentation-Out evaluation
    Train on all course presentations except one, test on the held-out one
    """
    print(f"\n{'='*80}")
    print(f"LCPO EVALUATION - WEEK {week}")
    print(f"{'='*80}")

    df = datasets[week]

    # Get unique course-presentation combinations
    course_presentations = df[["code_module", "code_presentation"]].drop_duplicates()
    print(f"\nFound {len(course_presentations)} unique course presentations")

    results_list = []
    models = get_models()

    # Prepare features
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
    X_full = pd.get_dummies(X_full)
    X_full = sanitize_feature_names(X_full)
    y = df["target"]

    for idx, (_, cp_row) in enumerate(course_presentations.iterrows(), 1):
        module = cp_row["code_module"]
        presentation = cp_row["code_presentation"]

        print(
            f"\n[{idx}/{len(course_presentations)}] Testing on: {module}-{presentation}"
        )

        # Split data
        test_mask = (df["code_module"] == module) & (
            df["code_presentation"] == presentation
        )
        train_mask = ~test_mask

        X_train, X_test = X_full[train_mask], X_full[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Skip if test set is too small
        if len(X_test) < 50 or y_test.nunique() < 2:
            print(f"  Skipping (insufficient test samples or single class)")
            continue

        # Evaluate each model
        for model_name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # Evaluate
                metrics = evaluate_metrics(y_test, y_pred, y_proba)

                result = {
                    "Week": week,
                    "Model": model_name,
                    "Test_Module": module,
                    "Test_Presentation": presentation,
                    "N_train": len(X_train),
                    "N_test": len(X_test),
                    **metrics,
                }
                results_list.append(result)

                print(
                    f"  {model_name}: AUROC={metrics['AUROC']:.3f}, F1={metrics['F1']:.3f}"
                )

            except Exception as e:
                print(f"  {model_name}: Error - {str(e)}")

    return pd.DataFrame(results_list)


def compare_random_vs_lcpo(random_results, lcpo_results, week=8):
    """Compare random split vs LCPO performance"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: RANDOM SPLIT vs LCPO - WEEK {week}")
    print(f"{'='*80}\n")

    # Filter for the specific week and All_features
    random_week = random_results[
        (random_results["Week"] == week)
        & (random_results["Features"] == "All_features")
    ].copy()

    # Aggregate LCPO results by model
    lcpo_agg = (
        lcpo_results[lcpo_results["Week"] == week]
        .groupby("Model")
        .agg(
            {
                "AUROC": ["mean", "std"],
                "AUPRC": ["mean", "std"],
                "F1": ["mean", "std"],
                "Precision": ["mean", "std"],
                "Recall": ["mean", "std"],
                "Balanced_Acc": ["mean", "std"],
            }
        )
    )

    # Create comparison table
    comparison = []
    for model in random_week["Model"].unique():
        if model == "Majority":
            continue

        random_row = random_week[random_week["Model"] == model].iloc[0]

        if model in lcpo_agg.index:
            lcpo_row = lcpo_agg.loc[model]

            comparison.append(
                {
                    "Model": model,
                    "Split": "Random",
                    "AUROC": f"{random_row['AUROC_mean']:.3f}±{random_row['AUROC_std']:.3f}",
                    "F1": f"{random_row['F1_mean']:.3f}±{random_row['F1_std']:.3f}",
                    "Balanced_Acc": f"{random_row['Balanced_Acc_mean']:.3f}±{random_row['Balanced_Acc_std']:.3f}",
                }
            )

            comparison.append(
                {
                    "Model": model,
                    "Split": "LCPO",
                    "AUROC": f"{lcpo_row['AUROC']['mean']:.3f}±{lcpo_row['AUROC']['std']:.3f}",
                    "F1": f"{lcpo_row['F1']['mean']:.3f}±{lcpo_row['F1']['std']:.3f}",
                    "Balanced_Acc": f"{lcpo_row['Balanced_Acc']['mean']:.3f}±{lcpo_row['Balanced_Acc']['std']:.3f}",
                }
            )

    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))

    return comparison_df


def main():
    print("=" * 80)
    print("OULAD LCPO EVALUATION")
    print("=" * 80)

    # Load data
    student_info, student_vle, student_assess, assessments = load_oulad_data()

    # Build dataset for week 8 (most data available)
    print("\nBuilding features for week 8...")
    vle_w, assess_w = filter_window(student_vle, student_assess, assessments, 8 * 7)
    df = build_features(vle_w, assess_w, student_info)
    datasets = {8: df}
    print(f"  Week 8: {df.shape[0]} samples, {df.shape[1]} features")

    # Run LCPO evaluation
    lcpo_results = lcpo_evaluation(datasets, week=8)

    # Save results
    lcpo_results.to_csv("lcpo_results_detailed.csv", index=False)
    print(f"\n✓ LCPO results saved to: lcpo_results_detailed.csv")

    # Create summary
    summary = lcpo_results.groupby("Model").agg(
        {
            "AUROC": ["mean", "std", "min", "max"],
            "F1": ["mean", "std", "min", "max"],
            "Balanced_Acc": ["mean", "std", "min", "max"],
        }
    )

    print(f"\n{'='*80}")
    print("LCPO RESULTS SUMMARY")
    print(f"{'='*80}")
    print(summary)

    # Load random split results for comparison
    try:
        random_results = pd.read_csv("baseline_results_detailed.csv")
        comparison = compare_random_vs_lcpo(random_results, lcpo_results, week=8)
        comparison.to_csv("random_vs_lcpo_comparison.csv", index=False)
        print(f"\n✓ Comparison saved to: random_vs_lcpo_comparison.csv")
    except FileNotFoundError:
        print(
            "\nWarning: baseline_results_detailed.csv not found. Skipping comparison."
        )


if __name__ == "__main__":
    main()

# Made with Bob
