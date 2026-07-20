"""
OULAD Evaluation Runner — end-to-end pipeline orchestrator.

Runs all three evaluation strategies (random-student, LCPO, future-presentation)
using the shared ``evaluation_pipeline`` module, saves every result CSV, generates
the unified comparison table and course-difficulty chart, and updates
``results/overall_summary.csv``.

Usage
-----
    python src/run_evaluation.py

All outputs are written under ``results/`` as configured in ``src/config.py``.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a top-level script
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import (
    BASELINE_RESULTS_DIR,
    CROSS_COURSE_RESULTS_DIR,
    LCPO_RESULTS_DIR,
    PREDICTION_WINDOWS,
    RESULTS_DIR,
)
from check_data import check_data_files
from oulad_data import create_datasets, load_oulad_data
from evaluation_pipeline import (
    analyze_course_difficulty,
    build_unified_comparison_table,
    run_future_presentation_evaluation,
    run_lcpo_evaluation,
    run_random_student_evaluation,
)

# New result directory for the unified cross-split comparison
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"


def _save(df, path, label):
    """Save *df* to *path* and print a confirmation line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ✓ {label} → {path.relative_to(path.parent.parent.parent)}")


def _fmt(mean, std):
    """Format mean±std for the overall summary table."""
    return f"{mean:.3f}±{std:.3f}"


def _std0(x):
    """Population std (ddof=0) for use in .agg() calls."""
    return x.std(ddof=0)


_std0.__name__ = "std"


def _summarise_random(df):
    """Return the Week 8 / All_features / best-AUROC row for the summary."""
    w8 = df[(df["Week"] == 8) & (df["Features"] == "All_features")]
    grp = w8.groupby("Model")[["AUROC", "F1", "Precision", "Recall"]].mean()
    best = grp["AUROC"].idxmax()
    row = grp.loc[best]
    std_row = w8.groupby("Model")[["AUROC", "F1", "Precision", "Recall"]].std(ddof=0).loc[best]
    return best, row, std_row


def _summarise_lcpo(df):
    """Return the Week 8 / best-AUROC-mean row for the summary."""
    w8 = df[df["Week"] == 8]
    grp_mean = w8.groupby("Model")[["AUROC", "F1", "Precision", "Recall"]].mean()
    grp_std = w8.groupby("Model")[["AUROC", "F1", "Precision", "Recall"]].std(ddof=0)
    best = grp_mean["AUROC"].idxmax()
    return best, grp_mean.loc[best], grp_std.loc[best]


def _summarise_future(df):
    """Return the Week 8 / best-AUROC row for the summary."""
    w8 = df[df["Week"] == 8]
    best_idx = w8["AUROC"].idxmax()
    row = w8.loc[best_idx]
    return row["Model"], row, None


def main():
    print("=" * 80)
    print("OULAD EVALUATION PIPELINE — Full Run")
    print("=" * 80)

    # Preflight: verify all required data files exist before running
    check_data_files()

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    print("\n[1/7] Loading OULAD data …")
    student_info, student_vle, student_assess, assessments = load_oulad_data()
    weeks = [2, 4, 6, 8]
    datasets = create_datasets(student_info, student_vle, student_assess, assessments, weeks=weeks)
    print(f"  Loaded {len(student_info):,} enrollments across {student_info['id_student'].nunique():,} students")

    # ------------------------------------------------------------------ #
    # 2. Random-student evaluation                                         #
    # ------------------------------------------------------------------ #
    print("\n[2/7] Running random-student 5-fold GroupKFold CV …")
    random_df = run_random_student_evaluation(datasets, weeks=weeks)

    # Detailed results (one row per fold × week × model × feature subset)
    _save(random_df, BASELINE_RESULTS_DIR / "baseline_results_detailed.csv", "Baseline detailed")

    # Summary table: mean±std per (week, model, feature_subset)
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]
    summary = (
        random_df.groupby(["Week", "Model", "Features"])[metrics]
        .agg(["mean", _std0])
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    # Format as mean±std strings for readability
    for m in metrics:
        summary[m] = summary.apply(
            lambda r: _fmt(r[f"{m}_mean"], r[f"{m}_std"]), axis=1
        )
    table = summary[["Week", "Model", "Features"] + metrics]
    _save(table, BASELINE_RESULTS_DIR / "baseline_results_table.csv", "Baseline table")

    # ------------------------------------------------------------------ #
    # 3. LCPO evaluation                                                   #
    # ------------------------------------------------------------------ #
    print("\n[3/7] Running LCPO evaluation …")
    lcpo_df = run_lcpo_evaluation(datasets, weeks=weeks)
    _save(lcpo_df, LCPO_RESULTS_DIR / "lcpo_results_detailed.csv", "LCPO detailed")

    # Random vs LCPO comparison (Week 8, all models)
    rand_w8 = (
        random_df[(random_df["Week"] == 8) & (random_df["Features"] == "All_features")]
        .groupby("Model")[["AUROC", "F1", "Balanced_Acc"]]
        .agg(["mean", _std0])
    )
    rand_w8.columns = [f"{m}_{s}" for m, s in rand_w8.columns]
    rand_w8 = rand_w8.reset_index()
    rand_w8["Split"] = "Random"
    for m in ["AUROC", "F1", "Balanced_Acc"]:
        rand_w8[m] = rand_w8.apply(lambda r: _fmt(r[f"{m}_mean"], r[f"{m}_std"]), axis=1)
    rand_comp = rand_w8[["Model", "Split", "AUROC", "F1", "Balanced_Acc"]]

    lcpo_w8 = (
        lcpo_df[lcpo_df["Week"] == 8]
        .groupby("Model")[["AUROC", "F1", "Balanced_Acc"]]
        .agg(["mean", _std0])
    )
    lcpo_w8.columns = [f"{m}_{s}" for m, s in lcpo_w8.columns]
    lcpo_w8 = lcpo_w8.reset_index()
    lcpo_w8["Split"] = "LCPO"
    for m in ["AUROC", "F1", "Balanced_Acc"]:
        lcpo_w8[m] = lcpo_w8.apply(lambda r: _fmt(r[f"{m}_mean"], r[f"{m}_std"]), axis=1)
    lcpo_comp = lcpo_w8[["Model", "Split", "AUROC", "F1", "Balanced_Acc"]]

    comparison = pd.concat([rand_comp, lcpo_comp], ignore_index=True)
    _save(comparison, LCPO_RESULTS_DIR / "random_vs_lcpo_comparison.csv", "Random vs LCPO comparison")

    # ------------------------------------------------------------------ #
    # 4. Future-presentation evaluation                                    #
    # ------------------------------------------------------------------ #
    print("\n[4/7] Running future-presentation evaluation …")
    future_df = run_future_presentation_evaluation(datasets, weeks=weeks)
    _save(future_df, CROSS_COURSE_RESULTS_DIR / "future_presentation_results.csv", "Future-presentation")

    # ------------------------------------------------------------------ #
    # 5. Unified comparison table                                          #
    # ------------------------------------------------------------------ #
    print("\n[5/7] Building unified comparison table …")
    unified = build_unified_comparison_table(random_df, lcpo_df, future_df)
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save(unified, COMPARISON_RESULTS_DIR / "all_splits_comparison.csv", "Unified comparison")

    # ------------------------------------------------------------------ #
    # 6. Course difficulty analysis                                        #
    # ------------------------------------------------------------------ #
    print("\n[6/7] Analyzing course-level difficulty …")
    difficulty_df = analyze_course_difficulty(lcpo_df, output_dir=LCPO_RESULTS_DIR)

    # ------------------------------------------------------------------ #
    # 7. Overall summary                                                   #
    # ------------------------------------------------------------------ #
    print("\n[7/7] Regenerating overall_summary.csv …")

    best_rand, rand_means, rand_stds = _summarise_random(random_df)
    best_lcpo, lcpo_means, lcpo_stds = _summarise_lcpo(lcpo_df)
    best_fp, fp_row, _ = _summarise_future(future_df)

    # Build numeric summary for overall_summary.csv
    summary_rows = [
        {
            "Evaluation": "Baseline / Random-Student (Week 8)",
            "Best_Model": best_rand,
            "AUROC": _fmt(rand_means["AUROC"], rand_stds["AUROC"]),
            "F1": _fmt(rand_means["F1"], rand_stds["F1"]),
            "Precision": _fmt(rand_means["Precision"], rand_stds["Precision"]),
            "Recall": _fmt(rand_means["Recall"], rand_stds["Recall"]),
        },
        {
            "Evaluation": "LCPO (Week 8)",
            "Best_Model": best_lcpo,
            "AUROC": _fmt(lcpo_means["AUROC"], lcpo_stds["AUROC"]),
            "F1": _fmt(lcpo_means["F1"], lcpo_stds["F1"]),
            "Precision": _fmt(lcpo_means["Precision"], lcpo_stds["Precision"]),
            "Recall": _fmt(lcpo_means["Recall"], lcpo_stds["Recall"]),
        },
        {
            "Evaluation": "Future-Presentation (Week 8)",
            "Best_Model": best_fp,
            "AUROC": f"{fp_row['AUROC']:.3f}",
            "F1": f"{fp_row['F1']:.3f}",
            "Precision": f"{fp_row['Precision']:.3f}",
            "Recall": f"{fp_row['Recall']:.3f}",
        },
    ]
    overall = pd.DataFrame(summary_rows)
    _save(overall, RESULTS_DIR / "overall_summary.csv", "Overall summary")

    # ------------------------------------------------------------------ #
    # Completion summary                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE — Result Summary")
    print("=" * 80)
    print(f"\n{'Evaluation':<40} {'Best Model':<22} {'AUROC'}")
    print("-" * 80)
    for r in summary_rows:
        print(f"  {r['Evaluation']:<38} {r['Best_Model']:<22} {r['AUROC']}")

    print(f"\n  Hardest course-presentation: {difficulty_df.iloc[0]['Course_Presentation']} "
          f"(AUROC {difficulty_df.iloc[0]['AUROC_mean']:.3f})")
    print(f"  Easiest course-presentation: {difficulty_df.iloc[-1]['Course_Presentation']} "
          f"(AUROC {difficulty_df.iloc[-1]['AUROC_mean']:.3f})")

    print("\n  Output files:")
    output_files = [
        BASELINE_RESULTS_DIR / "baseline_results_detailed.csv",
        BASELINE_RESULTS_DIR / "baseline_results_table.csv",
        LCPO_RESULTS_DIR / "lcpo_results_detailed.csv",
        LCPO_RESULTS_DIR / "random_vs_lcpo_comparison.csv",
        LCPO_RESULTS_DIR / "course_presentation_difficulty.csv",
        LCPO_RESULTS_DIR / "course_difficulty_chart.png",
        CROSS_COURSE_RESULTS_DIR / "future_presentation_results.csv",
        COMPARISON_RESULTS_DIR / "all_splits_comparison.csv",
        RESULTS_DIR / "overall_summary.csv",
    ]
    for p in output_files:
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists} {p.relative_to(RESULTS_DIR.parent)}")

    return random_df, lcpo_df, future_df, unified, difficulty_df


if __name__ == "__main__":
    main()
