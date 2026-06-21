"""
OULAD Per-Course LCPO Analysis
Detailed analysis of Leave-Course-Presentation-Out performance variation

This script analyzes how model performance varies across different courses,
identifying which courses are easier/harder to predict and why.
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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


def load_lcpo_results():
    """Load LCPO evaluation results"""
    lcpo_path = RESULTS_DIR / "lcpo" / "lcpo_results_detailed.csv"

    if not lcpo_path.exists():
        print(f"⚠️  LCPO results not found at {lcpo_path}")
        print("Please run src/lcpo_evaluation.py first")
        return None

    df = pd.read_csv(lcpo_path)
    print(f"✓ Loaded LCPO results: {len(df)} rows")
    return df


def load_course_metadata():
    """Load course metadata for context"""
    student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
    courses = pd.read_csv(DATA_DIR / "courses.csv")

    # Calculate course statistics
    course_stats = (
        student_info.groupby(["code_module", "code_presentation"])
        .agg(
            {
                "id_student": "count",
                "final_result": lambda x: (x.isin(["Fail", "Withdrawn"])).mean(),
            }
        )
        .reset_index()
    )

    course_stats.columns = [
        "code_module",
        "code_presentation",
        "num_students",
        "at_risk_rate",
    ]

    # Merge with course info
    course_stats = course_stats.merge(
        courses, on=["code_module", "code_presentation"], how="left"
    )

    # Create course identifier
    course_stats["course_id"] = (
        course_stats["code_module"] + "-" + course_stats["code_presentation"]
    )

    return course_stats


def analyze_course_difficulty(lcpo_results, course_stats):
    """Analyze which courses are harder to predict"""

    # Average performance per course
    course_perf = (
        lcpo_results.groupby("test_course")
        .agg(
            {
                "AUROC": ["mean", "std", "min", "max"],
                "AUPRC": ["mean", "std"],
                "F1": ["mean", "std"],
            }
        )
        .reset_index()
    )

    course_perf.columns = [
        "course_id",
        "AUROC_mean",
        "AUROC_std",
        "AUROC_min",
        "AUROC_max",
        "AUPRC_mean",
        "AUPRC_std",
        "F1_mean",
        "F1_std",
    ]

    # Merge with course metadata
    course_analysis = course_perf.merge(
        course_stats[
            [
                "course_id",
                "code_module",
                "num_students",
                "at_risk_rate",
                "module_presentation_length",
            ]
        ],
        on="course_id",
        how="left",
    )

    # Categorize difficulty
    auroc_median = course_analysis["AUROC_mean"].median()
    course_analysis["difficulty"] = course_analysis["AUROC_mean"].apply(
        lambda x: (
            "Easy"
            if x > auroc_median + 0.05
            else ("Hard" if x < auroc_median - 0.05 else "Medium")
        )
    )

    return course_analysis


def analyze_model_consistency(lcpo_results):
    """Analyze which models are most consistent across courses"""

    model_consistency = (
        lcpo_results.groupby("model")
        .agg(
            {
                "AUROC": ["mean", "std", "min", "max"],
                "AUPRC": ["mean", "std"],
                "F1": ["mean", "std"],
            }
        )
        .reset_index()
    )

    model_consistency.columns = [
        "model",
        "AUROC_mean",
        "AUROC_std",
        "AUROC_min",
        "AUROC_max",
        "AUPRC_mean",
        "AUPRC_std",
        "F1_mean",
        "F1_std",
    ]

    # Calculate coefficient of variation (lower = more consistent)
    model_consistency["AUROC_cv"] = (
        model_consistency["AUROC_std"] / model_consistency["AUROC_mean"]
    )

    # Sort by consistency
    model_consistency = model_consistency.sort_values("AUROC_cv")

    return model_consistency


def identify_outlier_courses(course_analysis):
    """Identify courses with unusually high or low performance"""

    auroc_mean = course_analysis["AUROC_mean"].mean()
    auroc_std = course_analysis["AUROC_mean"].std()

    # Outliers are > 1.5 std from mean
    course_analysis["is_outlier"] = (
        np.abs(course_analysis["AUROC_mean"] - auroc_mean) > 1.5 * auroc_std
    )

    outliers = course_analysis[course_analysis["is_outlier"]].copy()
    outliers["outlier_type"] = outliers["AUROC_mean"].apply(
        lambda x: "High Performance" if x > auroc_mean else "Low Performance"
    )

    return outliers


def analyze_course_characteristics(course_analysis):
    """Analyze relationship between course characteristics and predictability"""

    # Correlation analysis
    numeric_cols = [
        "AUROC_mean",
        "num_students",
        "at_risk_rate",
        "module_presentation_length",
    ]
    correlations = course_analysis[numeric_cols].corr()["AUROC_mean"].drop("AUROC_mean")

    # Group by course module
    module_perf = (
        course_analysis.groupby("code_module")
        .agg(
            {
                "AUROC_mean": ["mean", "std", "count"],
                "at_risk_rate": "mean",
                "num_students": "mean",
            }
        )
        .reset_index()
    )

    module_perf.columns = [
        "code_module",
        "AUROC_mean",
        "AUROC_std",
        "num_presentations",
        "avg_at_risk_rate",
        "avg_num_students",
    ]

    return correlations, module_perf


def create_visualizations(lcpo_results, course_analysis, model_consistency, output_dir):
    """Create comprehensive visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Course Performance Distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: AUROC distribution across courses
    ax = axes[0, 0]
    course_analysis_sorted = course_analysis.sort_values("AUROC_mean")
    colors = course_analysis_sorted["difficulty"].map(
        {"Easy": "green", "Medium": "orange", "Hard": "red"}
    )
    ax.barh(
        range(len(course_analysis_sorted)),
        course_analysis_sorted["AUROC_mean"],
        color=colors,
    )
    ax.set_yticks(range(len(course_analysis_sorted)))
    ax.set_yticklabels(course_analysis_sorted["course_id"], fontsize=8)
    ax.set_xlabel("Mean AUROC", fontsize=12)
    ax.set_title("LCPO Performance by Course", fontsize=14, fontweight="bold")
    ax.axvline(
        course_analysis["AUROC_mean"].mean(),
        color="black",
        linestyle="--",
        label="Overall Mean",
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # 1b: Performance vs At-Risk Rate
    ax = axes[0, 1]
    scatter = ax.scatter(
        course_analysis["at_risk_rate"] * 100,
        course_analysis["AUROC_mean"],
        s=course_analysis["num_students"] / 10,
        c=course_analysis["AUROC_mean"],
        cmap="RdYlGn",
        alpha=0.6,
    )
    ax.set_xlabel("At-Risk Rate (%)", fontsize=12)
    ax.set_ylabel("Mean AUROC", fontsize=12)
    ax.set_title("Performance vs At-Risk Rate", fontsize=14, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="AUROC")

    # Add trend line
    z = np.polyfit(course_analysis["at_risk_rate"], course_analysis["AUROC_mean"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(
        course_analysis["at_risk_rate"].min(),
        course_analysis["at_risk_rate"].max(),
        100,
    )
    ax.plot(x_trend * 100, p(x_trend), "r--", alpha=0.8, linewidth=2, label="Trend")
    ax.legend()
    ax.grid(alpha=0.3)

    # 1c: Model Consistency
    ax = axes[1, 0]
    model_consistency_sorted = model_consistency.sort_values(
        "AUROC_mean", ascending=False
    )
    x = range(len(model_consistency_sorted))
    ax.bar(
        x,
        model_consistency_sorted["AUROC_mean"],
        yerr=model_consistency_sorted["AUROC_std"],
        capsize=5,
        alpha=0.7,
        color="steelblue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_consistency_sorted["model"], rotation=45, ha="right")
    ax.set_ylabel("Mean AUROC", fontsize=12)
    ax.set_title("Model Performance Across All Courses", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 1d: Performance Variance by Model
    ax = axes[1, 1]
    model_data = []
    for model in lcpo_results["model"].unique():
        model_aurocs = lcpo_results[lcpo_results["model"] == model]["AUROC"].values
        model_data.append(model_aurocs)

    bp = ax.boxplot(
        model_data, labels=lcpo_results["model"].unique(), patch_artist=True
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("AUROC Distribution by Model", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "lcpo_course_analysis.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved visualization: {output_dir / 'lcpo_course_analysis.png'}")
    plt.close()

    # Figure 2: Heatmap of Model Performance by Course
    fig, ax = plt.subplots(figsize=(14, 10))

    # Pivot data for heatmap
    heatmap_data = lcpo_results.pivot_table(
        index="test_course", columns="model", values="AUROC", aggfunc="mean"
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.70,
        vmin=0.55,
        vmax=0.85,
        ax=ax,
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title(
        "LCPO Performance Heatmap: Model × Course", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Test Course", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "lcpo_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved heatmap: {output_dir / 'lcpo_heatmap.png'}")
    plt.close()


def generate_report(
    course_analysis, model_consistency, outliers, correlations, module_perf, output_dir
):
    """Generate comprehensive text report"""

    output_dir = Path(output_dir)
    report_path = output_dir / "lcpo_course_analysis_report.md"

    with open(report_path, "w") as f:
        f.write("# OULAD LCPO Per-Course Analysis Report\n\n")
        f.write(
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Courses Analyzed**: {len(course_analysis)}\n")
        f.write(
            f"- **Overall Mean AUROC**: {course_analysis['AUROC_mean'].mean():.3f} ± {course_analysis['AUROC_mean'].std():.3f}\n"
        )
        f.write(
            f"- **AUROC Range**: {course_analysis['AUROC_mean'].min():.3f} - {course_analysis['AUROC_mean'].max():.3f}\n"
        )
        f.write(
            f"- **Performance Spread**: {course_analysis['AUROC_mean'].max() - course_analysis['AUROC_mean'].min():.3f}\n\n"
        )

        # Course Difficulty Distribution
        f.write("## Course Difficulty Distribution\n\n")
        difficulty_counts = course_analysis["difficulty"].value_counts()
        for difficulty, count in difficulty_counts.items():
            pct = count / len(course_analysis) * 100
            f.write(f"- **{difficulty}**: {count} courses ({pct:.1f}%)\n")
        f.write("\n")

        # Top Performing Courses
        f.write("## Top 5 Performing Courses (Easiest to Predict)\n\n")
        f.write("| Rank | Course | AUROC | At-Risk Rate | Students |\n")
        f.write("|------|--------|-------|--------------|----------|\n")
        top_courses = course_analysis.nlargest(5, "AUROC_mean")
        for idx, (_, row) in enumerate(top_courses.iterrows(), 1):
            f.write(
                f"| {idx} | {row['course_id']} | {row['AUROC_mean']:.3f} | {row['at_risk_rate']*100:.1f}% | {int(row['num_students'])} |\n"
            )
        f.write("\n")

        # Bottom Performing Courses
        f.write("## Bottom 5 Performing Courses (Hardest to Predict)\n\n")
        f.write("| Rank | Course | AUROC | At-Risk Rate | Students |\n")
        f.write("|------|--------|-------|--------------|----------|\n")
        bottom_courses = course_analysis.nsmallest(5, "AUROC_mean")
        for idx, (_, row) in enumerate(bottom_courses.iterrows(), 1):
            f.write(
                f"| {idx} | {row['course_id']} | {row['AUROC_mean']:.3f} | {row['at_risk_rate']*100:.1f}% | {int(row['num_students'])} |\n"
            )
        f.write("\n")

        # Model Consistency
        f.write("## Model Consistency Across Courses\n\n")
        f.write("| Model | Mean AUROC | Std Dev | CV | Min | Max |\n")
        f.write("|-------|------------|---------|----|----- |-----|\n")
        for _, row in model_consistency.iterrows():
            f.write(
                f"| {row['model']} | {row['AUROC_mean']:.3f} | {row['AUROC_std']:.3f} | {row['AUROC_cv']:.3f} | {row['AUROC_min']:.3f} | {row['AUROC_max']:.3f} |\n"
            )
        f.write("\n")
        f.write("*CV = Coefficient of Variation (lower = more consistent)*\n\n")

        # Outlier Courses
        if len(outliers) > 0:
            f.write("## Outlier Courses\n\n")
            f.write("| Course | AUROC | Type | At-Risk Rate | Students |\n")
            f.write("|--------|-------|------|--------------|----------|\n")
            for _, row in outliers.iterrows():
                f.write(
                    f"| {row['course_id']} | {row['AUROC_mean']:.3f} | {row['outlier_type']} | {row['at_risk_rate']*100:.1f}% | {int(row['num_students'])} |\n"
                )
            f.write("\n")

        # Course Characteristics Correlation
        f.write("## Course Characteristics vs Predictability\n\n")
        f.write("| Characteristic | Correlation with AUROC |\n")
        f.write("|----------------|------------------------|\n")
        for char, corr in correlations.items():
            f.write(f"| {char} | {corr:.3f} |\n")
        f.write("\n")

        # Module-Level Analysis
        f.write("## Performance by Course Module\n\n")
        f.write(
            "| Module | Mean AUROC | Std Dev | Presentations | Avg At-Risk Rate |\n"
        )
        f.write(
            "|--------|------------|---------|---------------|------------------|\n"
        )
        module_perf_sorted = module_perf.sort_values("AUROC_mean", ascending=False)
        for _, row in module_perf_sorted.iterrows():
            f.write(
                f"| {row['code_module']} | {row['AUROC_mean']:.3f} | {row['AUROC_std']:.3f} | {int(row['num_presentations'])} | {row['avg_at_risk_rate']*100:.1f}% |\n"
            )
        f.write("\n")

        # Key Findings
        f.write("## Key Findings\n\n")

        best_model = model_consistency.iloc[0]
        f.write(
            f"1. **Most Consistent Model**: {best_model['model']} (CV={best_model['AUROC_cv']:.3f})\n"
        )

        if correlations["at_risk_rate"] < 0:
            f.write(
                f"2. **At-Risk Rate Impact**: Negative correlation ({correlations['at_risk_rate']:.3f}) - courses with higher at-risk rates are slightly harder to predict\n"
            )
        else:
            f.write(
                f"2. **At-Risk Rate Impact**: Positive correlation ({correlations['at_risk_rate']:.3f}) - courses with higher at-risk rates are slightly easier to predict\n"
            )

        if correlations["num_students"] > 0:
            f.write(
                f"3. **Sample Size Effect**: Positive correlation ({correlations['num_students']:.3f}) - larger courses tend to be more predictable\n"
            )
        else:
            f.write(
                f"3. **Sample Size Effect**: Negative correlation ({correlations['num_students']:.3f}) - larger courses are not necessarily more predictable\n"
            )

        perf_spread = (
            course_analysis["AUROC_mean"].max() - course_analysis["AUROC_mean"].min()
        )
        f.write(
            f"4. **Performance Variability**: {perf_spread:.3f} AUROC spread indicates significant variation in cross-course generalization\n"
        )

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(
            "1. **Course-Specific Models**: Consider training separate models for difficult courses\n"
        )
        f.write(
            "2. **Feature Engineering**: Investigate why certain courses are harder to predict\n"
        )
        f.write(
            "3. **Ensemble Approaches**: Combine predictions from multiple models for robustness\n"
        )
        f.write(
            "4. **Data Augmentation**: For small courses, consider transfer learning or data augmentation\n"
        )
        f.write(
            "5. **Temporal Analysis**: Examine if course difficulty changes over time\n\n"
        )

        f.write("---\n\n")
        f.write(
            "*This report was automatically generated by `src/lcpo_course_analysis.py`*\n"
        )

    print(f"✓ Saved report: {report_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD LCPO PER-COURSE ANALYSIS")
    print("=" * 80)

    # Load data
    lcpo_results = load_lcpo_results()
    if lcpo_results is None:
        return

    course_stats = load_course_metadata()
    print(f"✓ Loaded metadata for {len(course_stats)} courses")

    # Analyze course difficulty
    print("\nAnalyzing course difficulty...")
    course_analysis = analyze_course_difficulty(lcpo_results, course_stats)

    # Analyze model consistency
    print("Analyzing model consistency...")
    model_consistency = analyze_model_consistency(lcpo_results)

    # Identify outliers
    print("Identifying outlier courses...")
    outliers = identify_outlier_courses(course_analysis)
    print(f"  Found {len(outliers)} outlier courses")

    # Analyze course characteristics
    print("Analyzing course characteristics...")
    correlations, module_perf = analyze_course_characteristics(course_analysis)

    # Create output directory
    output_dir = RESULTS_DIR / "lcpo_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    course_analysis.to_csv(output_dir / "course_difficulty_analysis.csv", index=False)
    model_consistency.to_csv(output_dir / "model_consistency.csv", index=False)
    outliers.to_csv(output_dir / "outlier_courses.csv", index=False)
    module_perf.to_csv(output_dir / "module_performance.csv", index=False)

    print(f"\n✓ Saved analysis results to {output_dir}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(lcpo_results, course_analysis, model_consistency, output_dir)

    # Generate report
    print("\nGenerating comprehensive report...")
    generate_report(
        course_analysis,
        model_consistency,
        outliers,
        correlations,
        module_perf,
        output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - course_difficulty_analysis.csv")
    print(f"  - model_consistency.csv")
    print(f"  - outlier_courses.csv")
    print(f"  - module_performance.csv")
    print(f"  - lcpo_course_analysis.png")
    print(f"  - lcpo_heatmap.png")
    print(f"  - lcpo_course_analysis_report.md")

    return course_analysis, model_consistency, outliers


if __name__ == "__main__":
    course_analysis, model_consistency, outliers = main()

# Made with Bob
