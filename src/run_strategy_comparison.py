"""
run_strategy_comparison.py — One-off script comparing Strategy A vs Strategy B filtering.

Strategy A: filter assessment submissions by due date only (assessments.date <= window).
Strategy B: dual guard — due date AND submission date (date_submitted <= window).

Runs 5-fold GroupKFold CV (All_features subset) for all 5 models × 4 prediction weeks
under both strategies.  Saves a 20-row CSV to:

    results/comparison/strategy_a_vs_b_comparison.csv

Columns:
    Week, Model,
    Strategy_A_AUROC_mean, Strategy_A_AUROC_std,
    Strategy_B_AUROC_mean, Strategy_B_AUROC_std,
    Delta_AUROC_mean,
    Rows_Dropped_A, Rows_Dropped_B, Rows_Dropped_Diff, Rows_Dropped_Pct

Note: std values are population std (ddof=0), consistent with all other
      comparison CSVs in this project.

Usage
-----
    source oulad_env/bin/activate
    python src/run_strategy_comparison.py

This script is NOT called by run_evaluation.py — it is intentionally separate
because it requires ~2× the computation time of a normal evaluation run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from check_data import check_data_files
from config import RESULTS_DIR
from evaluation_pipeline import run_strategy_comparison

COMPARISON_DIR = RESULTS_DIR / "comparison"


def main():
    print("=" * 70)
    print("Strategy A vs B Comparison — All models × 4 prediction weeks")
    print("  std convention: population std (ddof=0)")
    print("=" * 70)

    check_data_files()

    result = run_strategy_comparison()

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARISON_DIR / "strategy_a_vs_b_comparison.csv"
    result.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(result)}-row comparison → {out_path}")

    print("\nSummary (LightGBM, all weeks):")
    lgbm = result[result["Model"] == "LightGBM"][
        ["Week", "Strategy_A_AUROC_mean", "Strategy_B_AUROC_mean",
         "Delta_AUROC_mean", "Rows_Dropped_Pct"]
    ]
    print(lgbm.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
