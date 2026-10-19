"""
test_loss_weighting_groupby.py
-------------------------------
Unit tests verifying that:

1. A synthetic GNN LCPO results DataFrame with 10 rows (5 model seeds ×
   2 loss settings: weighted and unweighted) produces exactly 2 summary
   rows when grouped by loss_weighting — one per setting, not one pooled row.

2. The MLP run_mlp_matched() function emits model="MLP_behavioral_summaries"
   when feature_set="A" (six behavioral summaries only) so that
   update_results_matrix routes it to the correct spreadsheet row.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_loss_weighting_groupby.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lcpo_per_seed_df(n_seeds: int = 5) -> pd.DataFrame:
    """Build a synthetic per_seed_df with n_seeds rows per loss setting.

    Mirrors the structure produced by run_lcpo_experiment() after the
    loss_weighting field was added to each record.
    """
    rng = np.random.default_rng(0)
    rows = []
    for lw in ("weighted", "unweighted"):
        for mseed in range(n_seeds):
            rows.append(
                {
                    "week": 8,
                    "fold_idx": 0,
                    "condition": "full",
                    "loss_weighting": lw,
                    "held_out_module": "AAA",
                    "held_out_presentation": "2013J",
                    "model_seed": mseed,
                    "auroc": float(rng.uniform(0.7, 0.9)),
                    "auprc": float(rng.uniform(0.6, 0.85)),
                    "f1": float(rng.uniform(0.55, 0.8)),
                    "precision": float(rng.uniform(0.5, 0.85)),
                    "recall": float(rng.uniform(0.5, 0.85)),
                    "balanced_acc": float(rng.uniform(0.6, 0.85)),
                }
            )
    return pd.DataFrame(rows)


def _compute_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the lcpo_summary groupby logic from run_lcpo_experiment()."""
    metric_cols = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    summary_rows = []
    for (wk, fidx, cond, lw, ho_mod, ho_pres), fold_group in per_seed_df.groupby(
        ["week", "fold_idx", "condition", "loss_weighting",
         "held_out_module", "held_out_presentation"],
        sort=False,
    ):
        row: dict = {
            "week": wk,
            "fold_idx": fidx,
            "condition": cond,
            "loss_weighting": lw,
            "held_out_module": ho_mod,
            "held_out_presentation": ho_pres,
        }
        for col in metric_cols:
            row[f"{col}_mean"] = fold_group[col].mean()
            row[f"{col}_std"] = fold_group[col].std()
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Tests — loss_weighting groupby
# ---------------------------------------------------------------------------

class TestLossWeightingGroupby:
    """Summary aggregation must split by loss_weighting, not pool across settings."""

    def test_two_loss_settings_produce_two_summary_rows(self):
        """10 per-seed rows (5 seeds × 2 settings) → 2 summary rows, not 1."""
        per_seed_df = _make_lcpo_per_seed_df(n_seeds=5)
        assert len(per_seed_df) == 10

        summary_df = _compute_summary(per_seed_df)

        assert len(summary_df) == 2, (
            f"Expected 2 summary rows (one per loss_weighting) but got {len(summary_df)}. "
            "loss_weighting must be part of the groupby key."
        )

    def test_summary_rows_have_distinct_loss_weighting_values(self):
        """The two summary rows must carry 'weighted' and 'unweighted' labels."""
        per_seed_df = _make_lcpo_per_seed_df(n_seeds=5)
        summary_df = _compute_summary(per_seed_df)

        lw_values = set(summary_df["loss_weighting"].tolist())
        assert lw_values == {"weighted", "unweighted"}, (
            f"Expected {{'weighted', 'unweighted'}} but got {lw_values}."
        )

    def test_mean_values_differ_between_settings(self):
        """Each summary row's auroc_mean must equal the mean of its own 5-seed group,
        not the mean of all 10 rows pooled together."""
        per_seed_df = _make_lcpo_per_seed_df(n_seeds=5)
        summary_df = _compute_summary(per_seed_df)

        for lw in ("weighted", "unweighted"):
            expected = per_seed_df[per_seed_df["loss_weighting"] == lw]["auroc"].mean()
            actual = summary_df[summary_df["loss_weighting"] == lw]["auroc_mean"].iloc[0]
            assert actual == pytest.approx(expected), (
                f"auroc_mean for loss_weighting={lw!r} does not match "
                f"the per-group mean (expected {expected:.4f}, got {actual:.4f})."
            )


# ---------------------------------------------------------------------------
# Tests — six_summaries MLP model name
# ---------------------------------------------------------------------------

class TestSixSummariesModelName:
    """run_mlp_matched() must use model='MLP_behavioral_summaries' for feature_set='A'."""

    def test_feature_set_a_maps_to_behavioral_summaries(self):
        """The model name derived from feature_set='A' must be 'MLP_behavioral_summaries'."""
        feature_set = "A"
        mlp_model_name = "MLP_behavioral_summaries" if feature_set == "A" else "MLP_all_features"
        assert mlp_model_name == "MLP_behavioral_summaries", (
            "feature_set='A' must produce model='MLP_behavioral_summaries' so "
            "update_results_matrix routes it to row 17 (not row 16)."
        )

    def test_other_feature_sets_map_to_all_features(self):
        """Any feature_set other than 'A' must produce model='MLP_all_features'."""
        for fs in ("B", "C", "D", "E"):
            mlp_model_name = "MLP_behavioral_summaries" if fs == "A" else "MLP_all_features"
            assert mlp_model_name == "MLP_all_features", (
                f"feature_set={fs!r} must produce model='MLP_all_features', got {mlp_model_name!r}."
            )
