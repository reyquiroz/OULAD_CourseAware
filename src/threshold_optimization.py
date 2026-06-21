"""
OULAD Threshold Optimization for Deployment
Optimize classification threshold for at-risk student prediction

This script analyzes precision-recall tradeoffs and determines optimal
classification thresholds for different deployment scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def load_and_prepare_data(prediction_week=8):
    """Load OULAD data and create features"""
    print(f"Loading data for week {prediction_week}...")

    # Load datasets
    student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
    student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
    student_assessment = pd.read_csv(DATA_DIR / "studentAssessment.csv")
    assessments = pd.read_csv(DATA_DIR / "assessments.csv")
    vle = pd.read_csv(DATA_DIR / "vle.csv")

    # Apply label mapping
    student_info["target"] = student_info["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    # Filter by prediction window
    window_days = prediction_week * 7
    vle_filtered = student_vle[student_vle["date"] <= window_days]

    assess_merged = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_filtered = assess_merged[assess_merged["date"] <= window_days]

    # Create features (simplified for speed)
    vle_features = (
        vle_filtered.groupby(["code_module", "code_presentation", "id_student"])
        .agg({"sum_click": ["sum", "mean", "std", "max"]})
        .reset_index()
    )
    vle_features.columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "vle_total",
        "vle_mean",
        "vle_std",
        "vle_max",
    ]

    assess_features = (
        assess_filtered.groupby(["code_module", "code_presentation", "id_student"])
        .agg({"score": ["mean", "std", "count"]})
        .reset_index()
    )
    assess_features.columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "assess_mean",
        "assess_std",
        "assess_count",
    ]

    # Merge
    combined = student_info.merge(
        vle_features, on=["code_module", "code_presentation", "id_student"], how="left"
    ).merge(
        assess_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    combined = combined.fillna(0)

    # One-hot encode
    categorical_cols = [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    # Clean feature names
    combined.columns = (
        combined.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )

    print(f"✓ Created features for {len(combined):,} students")

    return combined


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and get probability predictions"""
    print("\nTraining models...")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    results = {}

    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)

        # Get probability predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results[model_name] = {"model": model, "y_pred_proba": y_pred_proba}

        print(f"    ✓ Trained")

    return results


def analyze_threshold_impact(y_true, y_pred_proba, thresholds=None):
    """Analyze impact of different thresholds"""

    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Calculate intervention metrics
        total_flagged = tp + fp
        flagged_rate = total_flagged / len(y_true)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_flagged": total_flagged,
                "flagged_rate": flagged_rate,
            }
        )

    return pd.DataFrame(results)


def find_optimal_thresholds(threshold_analysis, y_true):
    """Find optimal thresholds for different scenarios"""

    optimal_thresholds = {}

    # 1. Maximum F1 Score
    max_f1_idx = threshold_analysis["f1"].idxmax()
    optimal_thresholds["max_f1"] = {
        "threshold": threshold_analysis.loc[max_f1_idx, "threshold"],
        "f1": threshold_analysis.loc[max_f1_idx, "f1"],
        "precision": threshold_analysis.loc[max_f1_idx, "precision"],
        "recall": threshold_analysis.loc[max_f1_idx, "recall"],
        "description": "Balanced precision and recall",
    }

    # 2. High Precision (minimize false positives)
    high_precision = threshold_analysis[threshold_analysis["precision"] >= 0.7]
    if len(high_precision) > 0:
        max_recall_idx = high_precision["recall"].idxmax()
        optimal_thresholds["high_precision"] = {
            "threshold": high_precision.loc[max_recall_idx, "threshold"],
            "f1": high_precision.loc[max_recall_idx, "f1"],
            "precision": high_precision.loc[max_recall_idx, "precision"],
            "recall": high_precision.loc[max_recall_idx, "recall"],
            "description": "Minimize false alarms (precision ≥ 0.7)",
        }

    # 3. High Recall (minimize false negatives)
    high_recall = threshold_analysis[threshold_analysis["recall"] >= 0.8]
    if len(high_recall) > 0:
        max_precision_idx = high_recall["precision"].idxmax()
        optimal_thresholds["high_recall"] = {
            "threshold": high_recall.loc[max_precision_idx, "threshold"],
            "f1": high_recall.loc[max_precision_idx, "f1"],
            "precision": high_recall.loc[max_precision_idx, "precision"],
            "recall": high_recall.loc[max_precision_idx, "recall"],
            "description": "Catch most at-risk students (recall ≥ 0.8)",
        }

    # 4. Resource-Constrained (limit interventions to 20% of students)
    at_risk_rate = y_true.mean()
    target_flagged_rate = min(
        0.20, at_risk_rate * 1.5
    )  # Flag at most 20% or 1.5x at-risk rate

    resource_constrained = threshold_analysis[
        threshold_analysis["flagged_rate"] <= target_flagged_rate
    ]
    if len(resource_constrained) > 0:
        max_f1_idx = resource_constrained["f1"].idxmax()
        optimal_thresholds["resource_constrained"] = {
            "threshold": resource_constrained.loc[max_f1_idx, "threshold"],
            "f1": resource_constrained.loc[max_f1_idx, "f1"],
            "precision": resource_constrained.loc[max_f1_idx, "precision"],
            "recall": resource_constrained.loc[max_f1_idx, "recall"],
            "flagged_rate": resource_constrained.loc[max_f1_idx, "flagged_rate"],
            "description": f"Limit interventions to {target_flagged_rate*100:.1f}% of students",
        }

    # 5. Cost-Sensitive (custom cost function)
    # Assume: Cost of false negative = 3x cost of false positive
    fn_cost = 3
    fp_cost = 1

    threshold_analysis["total_cost"] = (
        threshold_analysis["fn"] * fn_cost + threshold_analysis["fp"] * fp_cost
    )

    min_cost_idx = threshold_analysis["total_cost"].idxmin()
    optimal_thresholds["cost_sensitive"] = {
        "threshold": threshold_analysis.loc[min_cost_idx, "threshold"],
        "f1": threshold_analysis.loc[min_cost_idx, "f1"],
        "precision": threshold_analysis.loc[min_cost_idx, "precision"],
        "recall": threshold_analysis.loc[min_cost_idx, "recall"],
        "total_cost": threshold_analysis.loc[min_cost_idx, "total_cost"],
        "description": f"Minimize cost (FN cost = {fn_cost}x FP cost)",
    }

    return optimal_thresholds


def create_visualizations(model_results, y_test, output_dir):
    """Create comprehensive threshold optimization visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Precision-Recall Curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (model_name, results) in enumerate(model_results.items()):
        ax = axes[idx // 2, idx % 2]

        y_pred_proba = results["y_pred_proba"]

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Plot
        ax.plot(recall, precision, linewidth=2, label=f"{model_name}")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"Precision-Recall Curve - {model_name}", fontsize=13, fontweight="bold"
        )
        ax.grid(alpha=0.3)
        ax.legend()

        # Add optimal F1 point
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        max_f1_idx = np.argmax(f1_scores)
        ax.plot(
            recall[max_f1_idx],
            precision[max_f1_idx],
            "ro",
            markersize=10,
            label=f"Max F1={f1_scores[max_f1_idx]:.3f}",
        )
        ax.legend()

    # Fourth subplot: Comparison
    ax = axes[1, 1]
    for model_name, results in model_results.items():
        y_pred_proba = results["y_pred_proba"]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax.plot(recall, precision, linewidth=2, label=model_name)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Comparison", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        output_dir / "precision_recall_curves.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: {output_dir / 'precision_recall_curves.png'}")
    plt.close()

    # Figure 2: Threshold Impact Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Use best model for detailed analysis
    best_model_name = list(model_results.keys())[0]
    y_pred_proba = model_results[best_model_name]["y_pred_proba"]

    threshold_analysis = analyze_threshold_impact(y_test, y_pred_proba)

    # Plot 1: Precision, Recall, F1 vs Threshold
    ax = axes[0, 0]
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["precision"],
        label="Precision",
        linewidth=2,
    )
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["recall"],
        label="Recall",
        linewidth=2,
    )
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["f1"],
        label="F1 Score",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Metrics vs Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: False Positive Rate and False Negative Rate
    ax = axes[0, 1]
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["fpr"],
        label="False Positive Rate",
        linewidth=2,
        color="red",
    )
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["fnr"],
        label="False Negative Rate",
        linewidth=2,
        color="orange",
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Error Rates vs Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Number of Students Flagged
    ax = axes[1, 0]
    ax.plot(
        threshold_analysis["threshold"],
        threshold_analysis["total_flagged"],
        linewidth=2,
        color="purple",
    )
    ax.axhline(
        y=len(y_test) * 0.20, color="red", linestyle="--", label="20% of students"
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Number of Students Flagged", fontsize=12)
    ax.set_title("Intervention Volume vs Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Confusion Matrix Heatmap for Optimal Threshold
    ax = axes[1, 1]
    optimal_thresholds = find_optimal_thresholds(threshold_analysis, y_test)
    optimal_threshold = optimal_thresholds["max_f1"]["threshold"]

    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_optimal)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Success", "At-Risk"],
        yticklabels=["Success", "At-Risk"],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix (Threshold={optimal_threshold:.2f})",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(
        output_dir / "threshold_impact_analysis.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: {output_dir / 'threshold_impact_analysis.png'}")
    plt.close()


def generate_report(model_results, y_test, output_dir):
    """Generate comprehensive threshold optimization report"""

    output_dir = Path(output_dir)
    report_path = output_dir / "threshold_optimization_report.md"

    with open(report_path, "w") as f:
        f.write("# OULAD Threshold Optimization Report\n\n")
        f.write(
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            "This report analyzes optimal classification thresholds for at-risk student prediction.\n"
        )
        f.write(
            "Different thresholds are recommended for different deployment scenarios.\n\n"
        )

        # Dataset Statistics
        at_risk_count = y_test.sum()
        at_risk_rate = y_test.mean()
        f.write("### Dataset Statistics\n\n")
        f.write(f"- **Total Students**: {len(y_test):,}\n")
        f.write(
            f"- **At-Risk Students**: {at_risk_count:,} ({at_risk_rate*100:.1f}%)\n"
        )
        f.write(
            f"- **Success Students**: {len(y_test) - at_risk_count:,} ({(1-at_risk_rate)*100:.1f}%)\n\n"
        )

        # Optimal Thresholds by Scenario
        f.write("## Optimal Thresholds by Deployment Scenario\n\n")

        for model_name, results in model_results.items():
            f.write(f"### {model_name}\n\n")

            y_pred_proba = results["y_pred_proba"]
            threshold_analysis = analyze_threshold_impact(y_test, y_pred_proba)
            optimal_thresholds = find_optimal_thresholds(threshold_analysis, y_test)

            f.write(
                "| Scenario | Threshold | Precision | Recall | F1 | Description |\n"
            )
            f.write(
                "|----------|-----------|-----------|--------|----|--------------|\n"
            )

            for scenario_name, scenario_data in optimal_thresholds.items():
                f.write(f"| {scenario_name.replace('_', ' ').title()} | ")
                f.write(f"{scenario_data['threshold']:.3f} | ")
                f.write(f"{scenario_data['precision']:.3f} | ")
                f.write(f"{scenario_data['recall']:.3f} | ")
                f.write(f"{scenario_data['f1']:.3f} | ")
                f.write(f"{scenario_data['description']} |\n")

            f.write("\n")

        # Detailed Analysis for Best Model
        best_model_name = list(model_results.keys())[0]
        y_pred_proba = model_results[best_model_name]["y_pred_proba"]
        threshold_analysis = analyze_threshold_impact(y_test, y_pred_proba)
        optimal_thresholds = find_optimal_thresholds(threshold_analysis, y_test)

        f.write(f"## Detailed Analysis - {best_model_name}\n\n")

        for scenario_name, scenario_data in optimal_thresholds.items():
            f.write(f"### {scenario_name.replace('_', ' ').title()}\n\n")

            threshold = scenario_data["threshold"]
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            f.write(f"**Threshold**: {threshold:.3f}\n\n")
            f.write(f"**Performance Metrics**:\n")
            f.write(f"- Precision: {scenario_data['precision']:.3f}\n")
            f.write(f"- Recall: {scenario_data['recall']:.3f}\n")
            f.write(f"- F1 Score: {scenario_data['f1']:.3f}\n\n")

            f.write(f"**Confusion Matrix**:\n")
            f.write(f"- True Positives (Correctly identified at-risk): {tp}\n")
            f.write(f"- False Positives (False alarms): {fp}\n")
            f.write(f"- True Negatives (Correctly identified success): {tn}\n")
            f.write(f"- False Negatives (Missed at-risk students): {fn}\n\n")

            f.write(f"**Practical Implications**:\n")
            f.write(
                f"- Students flagged for intervention: {tp + fp} ({(tp + fp)/len(y_test)*100:.1f}%)\n"
            )
            f.write(
                f"- At-risk students identified: {tp} out of {tp + fn} ({tp/(tp+fn)*100:.1f}%)\n"
            )
            f.write(f"- False alarm rate: {fp/(fp+tn)*100:.1f}%\n\n")

            f.write(f"**Use Case**: {scenario_data['description']}\n\n")
            f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        f.write("### 1. For Early Warning Systems (Week 4-8)\n")
        f.write("- **Recommended**: High Recall threshold\n")
        f.write(
            "- **Rationale**: Early in semester, prioritize catching all at-risk students\n"
        )
        f.write("- **Accept**: Higher false positive rate (can be refined later)\n\n")

        f.write("### 2. For Resource-Constrained Institutions\n")
        f.write("- **Recommended**: Resource-Constrained threshold\n")
        f.write("- **Rationale**: Limit interventions to available capacity\n")
        f.write("- **Focus**: Highest-risk students within resource limits\n\n")

        f.write("### 3. For Targeted Interventions (Week 12-16)\n")
        f.write("- **Recommended**: High Precision threshold\n")
        f.write(
            "- **Rationale**: Later in semester, focus on students most likely to benefit\n"
        )
        f.write("- **Minimize**: Intervention fatigue from false alarms\n\n")

        f.write("### 4. For Balanced Approach\n")
        f.write("- **Recommended**: Max F1 threshold\n")
        f.write(
            "- **Rationale**: Balance between catching at-risk students and avoiding false alarms\n"
        )
        f.write("- **Best for**: General-purpose deployment\n\n")

        f.write("### 5. For Cost-Sensitive Scenarios\n")
        f.write("- **Recommended**: Cost-Sensitive threshold\n")
        f.write("- **Rationale**: Optimize based on relative costs of errors\n")
        f.write(
            "- **Customize**: Adjust cost ratio based on institutional priorities\n\n"
        )

        # Implementation Guide
        f.write("## Implementation Guide\n\n")
        f.write("```python\n")
        f.write("# Example: Using optimal threshold in production\n")
        f.write("import joblib\n\n")
        f.write("# Load trained model\n")
        f.write("model = joblib.load('best_model.pkl')\n\n")
        f.write("# Get probability predictions\n")
        f.write("y_pred_proba = model.predict_proba(X_new)[:, 1]\n\n")

        max_f1_threshold = optimal_thresholds["max_f1"]["threshold"]
        f.write(f"# Apply optimal threshold (Max F1 scenario)\n")
        f.write(f"optimal_threshold = {max_f1_threshold:.3f}\n")
        f.write("y_pred = (y_pred_proba >= optimal_threshold).astype(int)\n\n")
        f.write("# Flag students for intervention\n")
        f.write("at_risk_students = student_ids[y_pred == 1]\n")
        f.write("```\n\n")

        # Monitoring
        f.write("## Monitoring and Adjustment\n\n")
        f.write(
            "1. **Track Performance**: Monitor precision and recall in production\n"
        )
        f.write(
            "2. **Adjust Threshold**: Refine based on actual intervention outcomes\n"
        )
        f.write(
            "3. **Seasonal Variation**: Consider different thresholds for different semesters\n"
        )
        f.write(
            "4. **Course-Specific**: May need different thresholds for different courses\n"
        )
        f.write(
            "5. **Feedback Loop**: Incorporate intervention results to improve model\n\n"
        )

        f.write("---\n\n")
        f.write(
            "*This report was automatically generated by `src/threshold_optimization.py`*\n"
        )

    print(f"✓ Saved report: {report_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Load and prepare data
    df = load_and_prepare_data(prediction_week=8)

    # Prepare features and target
    exclude_cols = [
        "target",
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "date_registration",
        "date_unregistration",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["target"]

    # Split data
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    model_results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Create output directory
    output_dir = RESULTS_DIR / "threshold_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze and save threshold impact for each model
    print("\nAnalyzing threshold impact...")
    for model_name, results in model_results.items():
        y_pred_proba = results["y_pred_proba"]
        threshold_analysis = analyze_threshold_impact(y_test, y_pred_proba)

        filename = f"{model_name.lower().replace(' ', '_')}_threshold_analysis.csv"
        threshold_analysis.to_csv(output_dir / filename, index=False)

        # Find and save optimal thresholds
        optimal_thresholds = find_optimal_thresholds(threshold_analysis, y_test)
        optimal_df = pd.DataFrame(optimal_thresholds).T

        filename = f"{model_name.lower().replace(' ', '_')}_optimal_thresholds.csv"
        optimal_df.to_csv(output_dir / filename)

        print(f"  ✓ {model_name} analysis complete")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(model_results, y_test, output_dir)

    # Generate report
    print("\nGenerating comprehensive report...")
    generate_report(model_results, y_test, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Model-specific threshold analysis CSVs")
    print(f"  - Model-specific optimal thresholds CSVs")
    print(f"  - precision_recall_curves.png")
    print(f"  - threshold_impact_analysis.png")
    print(f"  - threshold_optimization_report.md")

    return model_results


if __name__ == "__main__":
    model_results = main()

# Made with Bob
