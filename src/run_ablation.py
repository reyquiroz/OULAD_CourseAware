"""
run_ablation.py — Feature-group ablation study.

Runs all five ablation conditions plus the full model for a given week
and saves per-condition metrics to results/graph/ablation_results.csv.

Usage
-----
    python src/run_ablation.py --week 8 --seeds 42
    python src/run_ablation.py --week 8 --seeds 42 123 7
"""
import argparse
import os

import pandas as pd

# Ensure src/ is on the path when run from repo root
import sys
sys.path.insert(0, os.path.dirname(__file__))

from run_gnn_experiment import run_random_split_experiment, RESULTS_DIR

# "with_iw_attrs" uses the full feature set but enables interacted_with edge
# attribute projection in the model (topology-only is the default / "full").
CONDITIONS = [
    "full",
    "no_assessment",
    "no_vle",
    "no_temporal",
    "no_course_features",
    "no_edge_attrs",
    "with_iw_attrs",
]


def main():
    parser = argparse.ArgumentParser(description="Feature-group ablation study for OULAD GNN")
    parser.add_argument("--week", type=int, default=8, help="Prediction week (default: 8)")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds to use (default: 42)",
    )
    args = parser.parse_args()

    rows = []

    for condition in CONDITIONS:
        # "with_iw_attrs" runs the full feature set with iw edge attrs enabled.
        # All other named conditions zero out the relevant feature group.
        use_iw_attrs = condition == "with_iw_attrs"
        feature_mask = None if condition in ("full", "with_iw_attrs") else [condition]
        print(f"\n{'='*60}")
        print(f"  ABLATION CONDITION: {condition}")
        print(f"{'='*60}")

        for seed in args.seeds:
            row, metrics = run_random_split_experiment(
                week=args.week,
                weighted=True,
                seed=seed,
                feature_mask=feature_mask,
                use_interacted_with_attrs=use_iw_attrs,
            )
            ablation_row = {
                "condition": condition,
                "week": row["week"],
                "seed": row["seed"],
                "loss_weighting": row["loss_weighting"],
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "balanced_acc": row["balanced_acc"],
                "best_threshold": row["best_threshold"],
                "pipeline_version": "v2_corrected",
            }
            rows.append(ablation_row)
            print(
                f"  condition={condition}  seed={seed}  "
                f"auroc={row['auroc']:.4f}  f1={row['f1']:.4f}  "
                f"balanced_acc={row['balanced_acc']:.4f}"
            )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}  ({len(df)} rows)")

    # Summary table
    print("\n=== Ablation Summary (seed mean across seeds) ===")
    summary = (
        df.groupby("condition")[["auroc", "auprc", "f1", "balanced_acc"]]
        .mean()
        .round(4)
        .sort_values("auroc", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
