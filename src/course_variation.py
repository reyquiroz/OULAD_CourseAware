"""Course-level LCPO variation analysis for GNN vs. LightGBM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "results" / "graph"
COMPARISON_PATH = GRAPH_DIR / "comparison_results.csv"
LCPO_PATH = GRAPH_DIR / "lcpo_results.csv"
OUTPUT_PATH = GRAPH_DIR / "course_variation.csv"


def build_course_variation(week: int = 8) -> pd.DataFrame:
    comparison_df = pd.read_csv(COMPARISON_PATH)
    lcpo_df = pd.read_csv(LCPO_PATH) if LCPO_PATH.exists() else pd.DataFrame()

    week_df = comparison_df[
        (comparison_df["split_type"] == "lcpo") & (comparison_df["week"] == week)
    ].copy()

    gnn_df = week_df[week_df["model"] == "GNN"][
        ["held_out_module", "held_out_presentation", "auroc", "f1"]
    ].rename(columns={"auroc": "gnn_auroc", "f1": "gnn_f1"})
    lgbm_df = week_df[week_df["model"] == "LightGBM"][
        ["held_out_module", "held_out_presentation", "auroc", "f1"]
    ].rename(columns={"auroc": "lgbm_auroc", "f1": "lgbm_f1"})

    merged = gnn_df.merge(
        lgbm_df,
        on=["held_out_module", "held_out_presentation"],
        how="outer",
    )

    if "n_test" in week_df.columns:
        n_test_df = week_df[
            ["held_out_module", "held_out_presentation", "n_test"]
        ].drop_duplicates()
    elif not lcpo_df.empty:
        n_test_df = lcpo_df[lcpo_df["week"] == week][
            ["held_out_module", "held_out_presentation", "n_test"]
        ].drop_duplicates()
    else:
        n_test_df = pd.DataFrame(
            columns=["held_out_module", "held_out_presentation", "n_test"]
        )

    merged = merged.merge(
        n_test_df,
        on=["held_out_module", "held_out_presentation"],
        how="left",
    )
    merged["auroc_delta"] = merged["gnn_auroc"] - merged["lgbm_auroc"]
    merged["f1_delta"] = merged["gnn_f1"] - merged["lgbm_f1"]

    merged = merged[
        [
            "held_out_module",
            "held_out_presentation",
            "gnn_auroc",
            "lgbm_auroc",
            "auroc_delta",
            "gnn_f1",
            "lgbm_f1",
            "f1_delta",
            "n_test",
        ]
    ].sort_values("auroc_delta", ascending=False, na_position="last")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-course LCPO GNN vs. LightGBM comparison table."
    )
    parser.add_argument("--week", type=int, default=8, help="Prediction week (default: 8)")
    args = parser.parse_args()

    course_df = build_course_variation(args.week)
    course_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved course variation results to {OUTPUT_PATH}")
    print()
    print(course_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
