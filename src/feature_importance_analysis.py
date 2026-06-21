"""
OULAD Feature Importance Analysis
Comprehensive analysis of feature importance for best-performing models

This script analyzes which features are most predictive of at-risk students,
using multiple methods: tree-based importance, permutation importance, and SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Try to import SHAP (optional)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def load_and_prepare_data(prediction_week=8):
    """Load OULAD data and create features for specified week"""
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

    # Filter by prediction window (leakage-safe)
    window_days = prediction_week * 7
    vle_filtered = student_vle[student_vle["date"] <= window_days]

    # Merge assessments with dates
    assess_merged = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_filtered = assess_merged[assess_merged["date"] <= window_days]

    # Create VLE features
    vle_features = (
        vle_filtered.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            {
                "sum_click": ["sum", "mean", "std", "max"],
                "date": ["min", "max", "count"],
            }
        )
        .reset_index()
    )

    vle_features.columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "vle_total_clicks",
        "vle_mean_clicks",
        "vle_std_clicks",
        "vle_max_clicks",
        "vle_first_access",
        "vle_last_access",
        "vle_num_days",
    ]

    # Add activity type diversity
    vle_with_type = vle_filtered.merge(
        vle[["id_site", "activity_type"]], on="id_site", how="left"
    )
    activity_diversity = (
        vle_with_type.groupby(["code_module", "code_presentation", "id_student"])[
            "activity_type"
        ]
        .nunique()
        .reset_index()
    )
    activity_diversity.columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "vle_activity_diversity",
    ]

    vle_features = vle_features.merge(
        activity_diversity,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    # Create assessment features
    assess_features = (
        assess_filtered.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            {
                "score": ["mean", "std", "min", "max", "count"],
                "date_submitted": lambda x: (x.isna()).sum(),
            }
        )
        .reset_index()
    )

    assess_features.columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "assess_mean_score",
        "assess_std_score",
        "assess_min_score",
        "assess_max_score",
        "assess_count",
        "assess_missing_count",
    ]

    # Merge all features
    combined = student_info.merge(
        vle_features, on=["code_module", "code_presentation", "id_student"], how="left"
    ).merge(
        assess_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    # Fill NaN values
    combined = combined.fillna(0)

    # One-hot encode categorical variables
    categorical_cols = [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    # Clean feature names for XGBoost
    combined.columns = (
        combined.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
        .str.replace(",", "_", regex=False)
    )

    print(f"✓ Created {len(combined.columns)} features for {len(combined):,} students")

    return combined


def prepare_features_target(df):
    """Separate features and target"""
    # Columns to exclude
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

    return X, y, feature_cols


def get_tree_based_importance(X_train, y_train, X_test, y_test, feature_names):
    """Get feature importance from tree-based models"""
    print("\nCalculating tree-based feature importance...")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    importance_results = {}

    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)

        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(feature_names))

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        importance_results[model_name] = {
            "model": model,
            "importance_df": importance_df,
            "train_score": model.score(X_train, y_train),
            "test_score": model.score(X_test, y_test),
        }

        print(
            f"    Train accuracy: {importance_results[model_name]['train_score']:.3f}"
        )
        print(f"    Test accuracy: {importance_results[model_name]['test_score']:.3f}")

    return importance_results


def get_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """Calculate permutation importance"""
    print(f"\nCalculating permutation importance ({n_repeats} repeats)...")

    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print(f"✓ Permutation importance calculated")

    return perm_df


def get_shap_values(model, X_train, X_test, feature_names, model_name):
    """Calculate SHAP values if available"""
    if not SHAP_AVAILABLE:
        return None

    print(f"\nCalculating SHAP values for {model_name}...")

    try:
        # Use appropriate explainer based on model type
        if "XGBoost" in model_name or "LightGBM" in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            # Sample data for faster computation
            sample_size = min(100, len(X_train))
            X_train_sample = X_train.sample(n=sample_size, random_state=42)
            explainer = shap.Explainer(model.predict, X_train_sample)

        # Calculate SHAP values for test set (sample if too large)
        sample_size = min(500, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)

        shap_values = explainer(X_test_sample)

        # Get mean absolute SHAP values
        shap_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("shap_importance", ascending=False)

        print(f"✓ SHAP values calculated")

        return {
            "shap_values": shap_values,
            "shap_importance": shap_importance,
            "explainer": explainer,
        }

    except Exception as e:
        print(f"⚠️  SHAP calculation failed: {str(e)}")
        return None


def categorize_features(feature_names):
    """Categorize features into groups"""
    categories = {"Demographics": [], "VLE Activity": [], "Assessment": [], "Other": []}

    for feature in feature_names:
        if any(
            x in feature
            for x in ["gender_", "region_", "education_", "imd_", "age_", "disability_"]
        ):
            categories["Demographics"].append(feature)
        elif "vle_" in feature:
            categories["VLE Activity"].append(feature)
        elif "assess_" in feature:
            categories["Assessment"].append(feature)
        else:
            categories["Other"].append(feature)

    return categories


def create_visualizations(
    importance_results, perm_importance, shap_results, feature_categories, output_dir
):
    """Create comprehensive visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Top Features by Model
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Plot tree-based importance for each model
    for idx, (model_name, results) in enumerate(list(importance_results.items())[:3]):
        ax = axes[idx // 2, idx % 2]

        top_features = results["importance_df"].head(20)
        ax.barh(range(len(top_features)), top_features["importance"], color="steelblue")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"], fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Top 20 Features - {model_name}", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    # Plot permutation importance
    ax = axes[1, 1]
    top_perm = perm_importance.head(20)
    ax.barh(
        range(len(top_perm)),
        top_perm["importance_mean"],
        xerr=top_perm["importance_std"],
        color="coral",
        capsize=3,
    )
    ax.set_yticks(range(len(top_perm)))
    ax.set_yticklabels(top_perm["feature"], fontsize=9)
    ax.set_xlabel("Permutation Importance", fontsize=11)
    ax.set_title(
        "Top 20 Features - Permutation Importance", fontsize=13, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        output_dir / "feature_importance_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Saved: {output_dir / 'feature_importance_comparison.png'}")
    plt.close()

    # Figure 2: Feature Category Importance
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate average importance by category for best model
    best_model_name = list(importance_results.keys())[0]
    best_importance = importance_results[best_model_name]["importance_df"]

    category_importance = {}
    for category, features in feature_categories.items():
        if features:
            cat_importance = best_importance[best_importance["feature"].isin(features)][
                "importance"
            ].sum()
            category_importance[category] = cat_importance

    categories = list(category_importance.keys())
    importances = list(category_importance.values())

    ax.bar(categories, importances, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel("Total Importance", fontsize=12)
    ax.set_title(
        f"Feature Category Importance - {best_model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(importances):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "category_importance.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'category_importance.png'}")
    plt.close()

    # Figure 3: SHAP Summary Plot (if available)
    if shap_results and SHAP_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(
            shap_results["shap_values"].values,
            features=shap_results["shap_values"].data,
            feature_names=shap_results["shap_values"].feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        fig.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_dir / 'shap_summary.png'}")
        plt.close()


def generate_report(
    importance_results, perm_importance, shap_results, feature_categories, output_dir
):
    """Generate comprehensive feature importance report"""

    output_dir = Path(output_dir)
    report_path = output_dir / "feature_importance_report.md"

    with open(report_path, "w") as f:
        f.write("# OULAD Feature Importance Analysis Report\n\n")
        f.write(
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            "This report analyzes which features are most predictive of at-risk students in the OULAD dataset.\n\n"
        )

        # Top Features Across Models
        f.write("## Top 10 Features by Model\n\n")

        for model_name, results in importance_results.items():
            f.write(f"### {model_name}\n\n")
            f.write(f"- **Train Accuracy**: {results['train_score']:.3f}\n")
            f.write(f"- **Test Accuracy**: {results['test_score']:.3f}\n\n")

            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")

            top_10 = results["importance_df"].head(10)
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n")
            f.write("\n")

        # Permutation Importance
        f.write("## Top 10 Features - Permutation Importance\n\n")
        f.write("| Rank | Feature | Importance | Std Dev |\n")
        f.write("|------|---------|------------|----------|\n")

        top_10_perm = perm_importance.head(10)
        for idx, (_, row) in enumerate(top_10_perm.iterrows(), 1):
            f.write(
                f"| {idx} | {row['feature']} | {row['importance_mean']:.4f} | {row['importance_std']:.4f} |\n"
            )
        f.write("\n")

        # SHAP Importance (if available)
        if shap_results:
            f.write("## Top 10 Features - SHAP Values\n\n")
            f.write("| Rank | Feature | SHAP Importance |\n")
            f.write("|------|---------|------------------|\n")

            top_10_shap = shap_results["shap_importance"].head(10)
            for idx, (_, row) in enumerate(top_10_shap.iterrows(), 1):
                f.write(
                    f"| {idx} | {row['feature']} | {row['shap_importance']:.4f} |\n"
                )
            f.write("\n")

        # Feature Category Analysis
        f.write("## Feature Category Analysis\n\n")

        best_model_name = list(importance_results.keys())[0]
        best_importance = importance_results[best_model_name]["importance_df"]

        for category, features in feature_categories.items():
            if features:
                cat_importance = best_importance[
                    best_importance["feature"].isin(features)
                ]["importance"].sum()
                f.write(f"### {category}\n\n")
                f.write(f"- **Number of Features**: {len(features)}\n")
                f.write(f"- **Total Importance**: {cat_importance:.4f}\n")
                f.write(
                    f"- **Average Importance**: {cat_importance/len(features):.4f}\n\n"
                )

                # Top 5 features in category
                cat_top = best_importance[
                    best_importance["feature"].isin(features)
                ].head(5)
                if len(cat_top) > 0:
                    f.write("**Top Features**:\n")
                    for _, row in cat_top.iterrows():
                        f.write(f"- {row['feature']}: {row['importance']:.4f}\n")
                    f.write("\n")

        # Key Findings
        f.write("## Key Findings\n\n")

        # Most important feature
        best_model = list(importance_results.values())[0]
        top_feature = best_model["importance_df"].iloc[0]
        f.write(
            f"1. **Most Important Feature**: {top_feature['feature']} (importance: {top_feature['importance']:.4f})\n"
        )

        # Most important category
        best_importance = importance_results[best_model_name]["importance_df"]
        category_totals = {}
        for category, features in feature_categories.items():
            if features:
                category_totals[category] = best_importance[
                    best_importance["feature"].isin(features)
                ]["importance"].sum()

        top_category = max(category_totals, key=category_totals.get)
        f.write(
            f"2. **Most Important Category**: {top_category} (total importance: {category_totals[top_category]:.4f})\n"
        )

        # Consistency across methods
        f.write(
            "3. **Consistency**: Features appearing in top 10 across multiple methods are most reliable\n"
        )

        # Model agreement
        f.write(
            "4. **Model Agreement**: Features ranked highly by multiple models are more robust\n\n"
        )

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(
            "1. **Focus on Top Features**: Prioritize collecting and maintaining data for top 20 features\n"
        )
        f.write(
            "2. **Feature Engineering**: Create interaction terms between top features\n"
        )
        f.write(
            "3. **Data Quality**: Ensure high quality for most important features\n"
        )
        f.write(
            "4. **Interpretability**: Use SHAP values to explain predictions to stakeholders\n"
        )
        f.write(
            "5. **Feature Selection**: Consider removing low-importance features to reduce complexity\n\n"
        )

        f.write("---\n\n")
        f.write(
            "*This report was automatically generated by `src/feature_importance_analysis.py`*\n"
        )

    print(f"✓ Saved report: {report_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Load and prepare data
    df = load_and_prepare_data(prediction_week=8)
    X, y, feature_names = prepare_features_target(df)

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

    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Get tree-based importance
    importance_results = get_tree_based_importance(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_names
    )

    # Get permutation importance (using best model)
    best_model_name = list(importance_results.keys())[0]
    best_model = importance_results[best_model_name]["model"]
    perm_importance = get_permutation_importance(
        best_model, X_test_scaled, y_test, feature_names
    )

    # Get SHAP values (if available)
    shap_results = get_shap_values(
        best_model, X_train_scaled, X_test_scaled, feature_names, best_model_name
    )

    # Categorize features
    feature_categories = categorize_features(feature_names)

    # Create output directory
    output_dir = RESULTS_DIR / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    for model_name, results in importance_results.items():
        filename = f"{model_name.lower().replace(' ', '_')}_importance.csv"
        results["importance_df"].to_csv(output_dir / filename, index=False)

    perm_importance.to_csv(output_dir / "permutation_importance.csv", index=False)

    if shap_results:
        shap_results["shap_importance"].to_csv(
            output_dir / "shap_importance.csv", index=False
        )

    print(f"\n✓ Saved importance results to {output_dir}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        importance_results,
        perm_importance,
        shap_results,
        feature_categories,
        output_dir,
    )

    # Generate report
    print("\nGenerating comprehensive report...")
    generate_report(
        importance_results,
        perm_importance,
        shap_results,
        feature_categories,
        output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Model-specific importance CSVs")
    print(f"  - permutation_importance.csv")
    if shap_results:
        print(f"  - shap_importance.csv")
    print(f"  - feature_importance_comparison.png")
    print(f"  - category_importance.png")
    if shap_results:
        print(f"  - shap_summary.png")
    print(f"  - feature_importance_report.md")

    return importance_results, perm_importance, shap_results


if __name__ == "__main__":
    importance_results, perm_importance, shap_results = main()

# Made with Bob
