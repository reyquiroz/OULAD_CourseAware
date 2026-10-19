"""
test_feature_set_matched_comparison.py
---------------------------------------
Unit tests for the --feature-set extension to build_tabular_features() in
src/run_matched_comparison.py.

All tests use a small synthetic DataFrame; no real OULAD data is needed.
The test directly calls build_tabular_features() with the data= kwarg so that
no file I/O occurs.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_matched_comparison import (
    _BASE_FEATURE_COLS,
    _DEMOGRAPHIC_COLS,
    _COURSE_IDENTITY_COLS,
    _FEATURE_SET_FLAGS,
    build_tabular_features,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(n: int = 20, seed: int = 0):
    """Return (student_info, student_vle, student_assess, assessments) minimal
    synthetic DataFrames sufficient for build_tabular_features().

    The student_info table includes all OULAD columns that build_features()
    expects, including the demographic and course-identity columns tested here.
    """
    rng = np.random.default_rng(seed)

    modules = ["AAA", "BBB"]
    presentations = ["2013J", "2014B"]

    student_info = pd.DataFrame(
        {
            "id_student": range(n),
            "code_module": rng.choice(modules, n),
            "code_presentation": rng.choice(presentations, n),
            "final_result": rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], n),
            "gender": rng.choice(["M", "F"], n),
            "region": rng.choice(["London", "South", "North"], n),
            "highest_education": rng.choice(["HE Qualification", "A Level or Equivalent"], n),
            "imd_band": rng.choice(["0-10%", "10-20%", "90-100%"], n),
            "disability": rng.choice(["Y", "N"], n),
            "age_band": rng.choice(["0-35", "35-55", "55<="], n),
            "num_of_prev_attempts": rng.integers(0, 3, n),
            "studied_credits": rng.integers(30, 120, n),
            # date_unregistration missing — all rows are never-unregistered (NaN → kept)
        }
    )
    student_info["target"] = student_info["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    # Minimal VLE: some rows per student so build_features() has something to aggregate.
    vle_rows = []
    for sid, mod, pres in zip(
        student_info["id_student"][:10],
        student_info["code_module"][:10],
        student_info["code_presentation"][:10],
    ):
        for day in range(1, 4):
            vle_rows.append(
                {
                    "id_student": sid,
                    "code_module": mod,
                    "code_presentation": pres,
                    "id_site": 1,
                    "date": day,
                    "sum_click": rng.integers(1, 20),
                }
            )
    student_vle = pd.DataFrame(vle_rows)

    # Minimal assessment metadata
    assessments = pd.DataFrame(
        {
            "id_assessment": [100, 101],
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2014B"],
            "assessment_type": ["TMA", "CMA"],
            "date": [10, 10],
            "weight": [50, 50],
        }
    )

    # Minimal student assessments (a few submissions)
    student_assess = pd.DataFrame(
        {
            "id_student": student_info["id_student"][:5].tolist(),
            "id_assessment": [100, 101, 100, 101, 100],
            "date_submitted": [5, 5, 5, 5, 5],
            "is_banked": [0, 0, 0, 0, 0],
            "score": [70, 80, 65, 55, 90],
        }
    )

    return student_info, student_vle, student_assess, assessments


# Patch out exclude_early_unregistered and _load_registration so no file I/O
# is needed — all enrollments are kept.
import unittest.mock as mock


def _patch_registration(monkeypatch):
    """Patch _load_registration and exclude_early_unregistered to be no-ops."""
    import run_matched_comparison as rmc

    # exclude_early_unregistered: return df unchanged, 0 removed
    monkeypatch.setattr(
        rmc,
        "exclude_early_unregistered",
        lambda enrollments_df, reg_df, cutoff: (enrollments_df, 0),
    )
    # _load_registration: return an empty DataFrame
    monkeypatch.setattr(
        rmc,
        "_load_registration",
        lambda: pd.DataFrame(columns=["id_student", "code_module", "code_presentation", "date_unregistration"]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureSetA:
    """Feature set A: only the original 9-column base set + age_band dummies."""

    def test_columns_are_base_set_only(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X, y, enrollment_df = build_tabular_features(8, data=data)

        # All base feature cols should be present (or absent if not in df — this
        # checks the set is limited to base + age_band, not demographics/course).
        col_set = set(X.columns)

        # Must NOT contain any demographic one-hots
        demo_prefixes = tuple(c + "_" for c in _DEMOGRAPHIC_COLS)
        demographic_cols_present = [c for c in col_set if c.startswith(demo_prefixes)]
        assert demographic_cols_present == [], (
            f"Feature set A should have no demographics cols; found: {demographic_cols_present}"
        )

        # Must NOT contain course identity one-hots
        ci_prefixes = tuple(c + "_" for c in _COURSE_IDENTITY_COLS)
        course_cols_present = [c for c in col_set if c.startswith(ci_prefixes)]
        assert course_cols_present == [], (
            f"Feature set A should have no course-identity cols; found: {course_cols_present}"
        )

        # age_band dummies MUST be present
        age_cols = [c for c in col_set if c.startswith("age_band_")]
        assert age_cols, "Feature set A must include age_band one-hot columns."


class TestFeatureSetB:
    """Feature set B: base + demographics."""

    def test_adds_demographic_columns(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X_a, _, _ = build_tabular_features(8, data=data, add_demographics=False)
        X_b, _, _ = build_tabular_features(8, data=data, add_demographics=True)

        assert X_b.shape[1] > X_a.shape[1], (
            "Feature set B must have more columns than A."
        )

        # At least one demographics-prefixed column should appear
        demo_prefixes = tuple(c + "_" for c in _DEMOGRAPHIC_COLS)
        demo_cols = [c for c in X_b.columns if c.startswith(demo_prefixes)]
        assert demo_cols, (
            "Feature set B must contain at least one demographics one-hot column."
        )

    def test_no_course_identity_columns(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X_b, _, _ = build_tabular_features(8, data=data, add_demographics=True, add_course_identity=False)

        ci_prefixes = tuple(c + "_" for c in _COURSE_IDENTITY_COLS)
        ci_cols = [c for c in X_b.columns if c.startswith(ci_prefixes)]
        assert ci_cols == [], (
            "Feature set B should not include course-identity columns."
        )


class TestFeatureSetC:
    """Feature set C: base + course identity."""

    def test_adds_course_identity_columns(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X_a, _, _ = build_tabular_features(8, data=data, add_course_identity=False)
        X_c, _, _ = build_tabular_features(8, data=data, add_course_identity=True)

        assert X_c.shape[1] > X_a.shape[1], (
            "Feature set C must have more columns than A."
        )

        ci_prefixes = tuple(c + "_" for c in _COURSE_IDENTITY_COLS)
        ci_cols = [c for c in X_c.columns if c.startswith(ci_prefixes)]
        assert ci_cols, (
            "Feature set C must contain at least one course-identity one-hot column."
        )

    def test_no_demographic_columns(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X_c, _, _ = build_tabular_features(8, data=data, add_demographics=False, add_course_identity=True)

        demo_prefixes = tuple(c + "_" for c in _DEMOGRAPHIC_COLS)
        demo_cols = [c for c in X_c.columns if c.startswith(demo_prefixes)]
        assert demo_cols == [], (
            "Feature set C should not include demographics columns."
        )


class TestFeatureSetFlags:
    """Verify the _FEATURE_SET_FLAGS mapping drives the correct kwargs."""

    def test_all_five_sets_present(self):
        assert set(_FEATURE_SET_FLAGS.keys()) == {"A", "B", "C", "D", "E"}

    def test_a_all_false(self):
        assert _FEATURE_SET_FLAGS["A"] == dict(
            add_demographics=False, add_course_identity=False, add_temporal=False
        )

    def test_b_demographics_only(self):
        f = _FEATURE_SET_FLAGS["B"]
        assert f["add_demographics"] is True
        assert f["add_course_identity"] is False
        assert f["add_temporal"] is False

    def test_c_course_identity_only(self):
        f = _FEATURE_SET_FLAGS["C"]
        assert f["add_demographics"] is False
        assert f["add_course_identity"] is True
        assert f["add_temporal"] is False

    def test_d_demographics_and_course(self):
        f = _FEATURE_SET_FLAGS["D"]
        assert f["add_demographics"] is True
        assert f["add_course_identity"] is True
        assert f["add_temporal"] is False

    def test_e_all_true(self):
        f = _FEATURE_SET_FLAGS["E"]
        assert f["add_demographics"] is True
        assert f["add_course_identity"] is True
        assert f["add_temporal"] is True


class TestFeatureSetColumnSanity:
    """No NaN values, all-numeric, correct row count."""

    def test_no_nans_set_a(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X, y, enrollment_df = build_tabular_features(8, data=data)
        assert not X.isnull().any().any(), "X for set A must not contain NaNs."

    def test_no_nans_set_b(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X, y, enrollment_df = build_tabular_features(8, data=data, add_demographics=True)
        assert not X.isnull().any().any(), "X for set B must not contain NaNs."

    def test_no_nans_set_c(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data()
        X, y, enrollment_df = build_tabular_features(8, data=data, add_course_identity=True)
        assert not X.isnull().any().any(), "X for set C must not contain NaNs."

    def test_aligned_lengths(self, monkeypatch):
        _patch_registration(monkeypatch)
        data = _make_synthetic_data(n=20)
        X, y, enrollment_df = build_tabular_features(8, data=data, add_demographics=True)
        assert len(X) == len(y) == len(enrollment_df), (
            "X, y, and enrollment_df must all have the same number of rows."
        )
