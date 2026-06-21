"""
OULAD Future-Presentation Split Evaluation
Tests model performance on future course presentations (temporal generalization)

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
    CROSS_COURSE_RESULTS_DIR,
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


def future_presentation_evaluation(datasets, week=8):
    """
    Future-Presentation Split Evaluation
    Train on earlier presentations, test on later presentations

    Train: 2013B, 2013J, 2014B
    Test: 2014J (future presentations)
    """
    print(f"\n{'='*80}")
    print(f"FUTURE-PRESENTATION EVALUATION - WEEK {week}")
    print(f"{'='*80}")

    df = datasets[week]

    # Define temporal split
    train_presentations = ["2013B", "2013J", "2014B"]
    test_presentations = ["2014J"]

    print(f"\nTrain presentations: {train_presentations}")
    print(f"Test presentations: {test_presentations}")

    # Check data availability
    presentation_counts = df.groupby("code_presentation").size()
    print(f"\nPresentation counts:")
    for pres in train_presentations + test_presentations:
        count = presentation_counts.get(pres, 0)
        print(f"  {pres}: {count} students")

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

    # Split data by presentation time
    train_mask = df["code_presentation"].isin(train_presentations)
    test_mask = df["code_presentation"].isin(test_presentations)

    X_train, X_test = X_full[train_mask], X_full[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTrain set: {len(X_train)} samples")
    print(
        f"  At-risk: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)"
    )
    print(
        f"  Success: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)"
    )

    print(f"\nTest set: {len(X_test)} samples")
    print(
        f"  At-risk: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)"
    )
    print(
        f"  Success: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)"
    )

    # Check if test set is valid
    if len(X_test) < 50:
        print("\nError: Test set too small (< 50 samples)")
        return pd.DataFrame()

    if y_test.nunique() < 2:
        print("\nError: Test set has only one class")
        return pd.DataFrame()

    # Evaluate each model
    print(f"\n{'='*80}")
    print("TRAINING AND EVALUATION")
    print(f"{'='*80}\n")

    for model_name, model in models.items():
        try:
            print(f"Training {model_name}...")

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
                "Split": "Future-Presentation",
                "Train_Presentations": ", ".join(train_presentations),
                "Test_Presentations": ", ".join(test_presentations),
                "N_train": len(X_train),
                "N_test": len(X_test),
                **metrics,
            }
            results_list.append(result)

            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  AUPRC: {metrics['AUPRC']:.4f}")
            print(f"  F1: {metrics['F1']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  Balanced Acc: {metrics['Balanced_Acc']:.4f}\n")

        except Exception as e:
            print(f"  Error: {str(e)}\n")

    return pd.DataFrame(results_list)


def compare_all_splits(random_results, lcpo_results, future_results, week=8):
    """Compare all three evaluation strategies"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: ALL EVALUATION STRATEGIES - WEEK {week}")
    print(f"{'='*80}\n")

    comparison_data = []

    # Get results for each model
    models = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]

    for model in models:
        # Random split (from baseline results)
        if random_results is not None and len(random_results) > 0:
            random_model = random_results[
                (random_results["Model"] == model)
                & (random_results["Week"] == week)
                & (random_results["Features"] == "All_features")
            ]
            if len(random_model) > 0:
                comparison_data.append(
                    {
                        "Model": model,
                        "Split": "Random",
                        "AUROC": random_model["AUROC_mean"].values[0],
                        "F1": random_model["F1_mean"].values[0],
                        "Balanced_Acc": random_model["Balanced_Acc_mean"].values[0],
                    }
                )

        # LCPO (aggregate across course presentations)
        if lcpo_results is not None and len(lcpo_results) > 0:
            lcpo_model = lcpo_results[
                (lcpo_results["Model"] == model) & (lcpo_results["Week"] == week)
            ]
            if len(lcpo_model) > 0:
                comparison_data.append(
                    {
                        "Model": model,
                        "Split": "LCPO",
                        "AUROC": lcpo_model["AUROC"].mean(),
                        "F1": lcpo_model["F1"].mean(),
                        "Balanced_Acc": lcpo_model["Balanced_Acc"].mean(),
                    }
                )

        # Future-presentation
        if future_results is not None and len(future_results) > 0:
            future_model = future_results[
                (future_results["Model"] == model) & (future_results["Week"] == week)
            ]
            if len(future_model) > 0:
                comparison_data.append(
                    {
                        "Model": model,
                        "Split": "Future-Presentation",
                        "AUROC": future_model["AUROC"].values[0],
                        "F1": future_model["F1"].values[0],
                        "Balanced_Acc": future_model["Balanced_Acc"].values[0],
                    }
                )

    comparison_df = pd.DataFrame(comparison_data)

    if len(comparison_df) > 0:
        # Pivot for better display
        for metric in ["AUROC", "F1", "Balanced_Acc"]:
            print(f"\n{metric} Comparison:")
            pivot = comparison_df.pivot(index="Model", columns="Split", values=metric)
            print(pivot.to_string())

            # Calculate generalization gaps
            if "Random" in pivot.columns:
                if "LCPO" in pivot.columns:
                    pivot["LCPO_Gap"] = pivot["Random"] - pivot["LCPO"]
                if "Future-Presentation" in pivot.columns:
                    pivot["Future_Gap"] = pivot["Random"] - pivot["Future-Presentation"]

                if "LCPO_Gap" in pivot.columns or "Future_Gap" in pivot.columns:
                    print(f"\nGeneralization Gaps (Random - Other):")
                    gap_cols = [
                        c for c in ["LCPO_Gap", "Future_Gap"] if c in pivot.columns
                    ]
                    print(pivot[gap_cols].to_string())

    return comparison_df


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD FUTURE-PRESENTATION SPLIT EVALUATION")
    print("=" * 80)

    # Load data
    student_info, student_vle, student_assess, assessments = load_oulad_data()

    # Build features for all weeks
    print("\nBuilding features for prediction windows...")
    datasets = {}
    for week in [2, 4, 6, 8]:
        window_days = week * 7
        print(f"  Week {week} (day {window_days})...")
        vle_w, assess_w = filter_window(
            student_vle, student_assess, assessments, window_days
        )
        df = build_features(vle_w, assess_w, student_info)
        datasets[week] = df
        print(f"    {len(df)} samples, {len(df.columns)} features")

    # Run future-presentation evaluation for all weeks
    all_results = []
    for week in [2, 4, 6, 8]:
        print(f"\n{'='*80}")
        print(f"EVALUATING WEEK {week}")
        print(f"{'='*80}")

        results = future_presentation_evaluation(datasets, week=week)
        if len(results) > 0:
            all_results.append(results)

    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # Save results
        output_file = CROSS_COURSE_RESULTS_DIR / "future_presentation_results.csv"
        final_results.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")

        # Display summary
        print("\nSummary by Week and Model:")
        summary = final_results.groupby(["Week", "Model"])[
            ["AUROC", "F1", "Balanced_Acc"]
        ].mean()
        print(summary.to_string())

        # Try to load and compare with other splits
        try:
            from pathlib import Path

            baseline_file = Path("results/baseline/baseline_results_detailed.csv")
            lcpo_file = Path("results/lcpo/lcpo_results_detailed.csv")

            random_results = None
            lcpo_results = None

            if baseline_file.exists():
                random_results = pd.read_csv(baseline_file)
                print("\nLoaded baseline (random split) results for comparison")

            if lcpo_file.exists():
                lcpo_results = pd.read_csv(lcpo_file)
                print("Loaded LCPO results for comparison")

            if random_results is not None or lcpo_results is not None:
                comparison = compare_all_splits(
                    random_results, lcpo_results, final_results, week=8
                )

                # Save comparison
                comparison_file = CROSS_COURSE_RESULTS_DIR / "all_splits_comparison.csv"
                comparison.to_csv(comparison_file, index=False)
                print(f"\nComparison saved to: {comparison_file}")

        except Exception as e:
            print(f"\nCould not load comparison data: {e}")

    else:
        print("\nNo results generated (insufficient data)")


if __name__ == "__main__":
    main()

# Made with Bob
