"""Readers that report one GNN number must not average weighted and unweighted rows."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gnn_results import select_reference_gnn_rows  # noqa: E402


def _df():
    rows = []
    for cond in ("full", "with_enrollment_summaries"):
        for lw, a in (("weighted", 0.70), ("unweighted", 0.60)):
            rows.append({"fold_idx": 0, "condition": cond, "loss_weighting": lw, "auroc": a})
    return pd.DataFrame(rows)


def test_selects_full_weighted_only():
    out = select_reference_gnn_rows(_df())
    assert len(out) == 1
    assert out.iloc[0]["condition"] == "full" and out.iloc[0]["loss_weighting"] == "weighted"


def test_other_variant_can_be_requested():
    out = select_reference_gnn_rows(_df(), condition="full", loss_weighting="unweighted")
    assert list(out["auroc"]) == [0.60]


def test_files_without_the_columns_are_not_filtered():
    old = pd.DataFrame({"fold_idx": [0, 1], "auroc": [0.7, 0.8]})
    assert len(select_reference_gnn_rows(old)) == 2
