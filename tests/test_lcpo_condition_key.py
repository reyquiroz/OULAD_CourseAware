"""
Tests for _append_or_create_csv deduplication behaviour with `condition` key.

Verifies that:
  1. Two rows with the same (week, fold_idx, model_seed) but DIFFERENT condition values
     are both preserved — i.e., `condition` is part of the dedup key.
  2. Two rows with the same (week, fold_idx, model_seed, condition) result in the newer
     row replacing the older one (keep="last" semantics).

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_lcpo_condition_key.py -v
"""

import sys
import tempfile
import os
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_gnn_experiment import _append_or_create_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(week=8, fold_idx=0, model_seed=42, condition="base", auroc=0.75):
    return {
        "week": week,
        "fold_idx": fold_idx,
        "model_seed": model_seed,
        "condition": condition,
        "auroc": auroc,
    }


LCPO_DEDUP_KEYS = ["week", "fold_idx", "model_seed", "condition"]
SUMMARY_DEDUP_KEYS = ["week", "fold_idx", "condition"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_different_conditions_both_preserved():
    """Rows with same (week, fold_idx, model_seed) but different condition must NOT
    be deduplicated — both rows must survive in the CSV."""
    row_base = _make_row(condition="base", auroc=0.75)
    row_variant = _make_row(condition="with_enrollment_summaries", auroc=0.80)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "lcpo_results.csv")

        _append_or_create_csv(pd.DataFrame([row_base]), csv_path, dedup_keys=LCPO_DEDUP_KEYS)
        _append_or_create_csv(pd.DataFrame([row_variant]), csv_path, dedup_keys=LCPO_DEDUP_KEYS)

        result = pd.read_csv(csv_path)

    assert len(result) == 2, (
        f"Expected 2 rows (one per condition) but got {len(result)}. "
        "condition must be part of the dedup key."
    )
    conditions = set(result["condition"].tolist())
    assert conditions == {"base", "with_enrollment_summaries"}


def test_same_condition_newer_row_replaces_older():
    """Rows with identical (week, fold_idx, model_seed, condition) — the newer row
    must replace the older one (keep='last')."""
    row_old = _make_row(condition="base", auroc=0.75)
    row_new = _make_row(condition="base", auroc=0.82)  # same key, updated auroc

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "lcpo_results.csv")

        _append_or_create_csv(pd.DataFrame([row_old]), csv_path, dedup_keys=LCPO_DEDUP_KEYS)
        _append_or_create_csv(pd.DataFrame([row_new]), csv_path, dedup_keys=LCPO_DEDUP_KEYS)

        result = pd.read_csv(csv_path)

    assert len(result) == 1, (
        f"Expected 1 row after re-writing same key but got {len(result)}."
    )
    assert result.iloc[0]["auroc"] == pytest.approx(0.82), (
        "The newer row (auroc=0.82) should have replaced the older one (auroc=0.75)."
    )


def test_summary_different_conditions_both_preserved():
    """lcpo_summary.csv dedup key is (week, fold_idx, condition) — both conditions
    must survive when fold_idx matches but condition differs."""
    row_base = {"week": 8, "fold_idx": 0, "condition": "base", "auroc_mean": 0.75}
    row_variant = {"week": 8, "fold_idx": 0, "condition": "with_enrollment_summaries", "auroc_mean": 0.80}

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "lcpo_summary.csv")

        _append_or_create_csv(pd.DataFrame([row_base]), csv_path, dedup_keys=SUMMARY_DEDUP_KEYS)
        _append_or_create_csv(pd.DataFrame([row_variant]), csv_path, dedup_keys=SUMMARY_DEDUP_KEYS)

        result = pd.read_csv(csv_path)

    assert len(result) == 2, (
        f"Expected 2 summary rows (one per condition) but got {len(result)}."
    )


def test_summary_same_condition_newer_replaces():
    """lcpo_summary.csv: re-writing the same (week, fold_idx, condition) updates the row."""
    row_old = {"week": 8, "fold_idx": 0, "condition": "base", "auroc_mean": 0.75}
    row_new = {"week": 8, "fold_idx": 0, "condition": "base", "auroc_mean": 0.82}

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "lcpo_summary.csv")

        _append_or_create_csv(pd.DataFrame([row_old]), csv_path, dedup_keys=SUMMARY_DEDUP_KEYS)
        _append_or_create_csv(pd.DataFrame([row_new]), csv_path, dedup_keys=SUMMARY_DEDUP_KEYS)

        result = pd.read_csv(csv_path)

    assert len(result) == 1
    assert result.iloc[0]["auroc_mean"] == pytest.approx(0.82)
