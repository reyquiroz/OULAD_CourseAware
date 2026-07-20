"""
Tests for filter_window() in src/oulad_data.py.

Covers boundary conditions for both temporal guards (Strategy B):
  Guard 1 — due-date guard:   assessments.date <= window
  Guard 2 — submission guard: date_submitted   <= window

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_filter_window.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from oulad_data import filter_window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WINDOW = 56  # Week 8 cutoff in days


def _make_vle(dates: list) -> pd.DataFrame:
    """Minimal studentVle rows with the given interaction dates."""
    return pd.DataFrame(
        {
            "id_student": 1,
            "id_site": 10,
            "code_module": "AAA",
            "code_presentation": "2013J",
            "date": dates,
            "sum_click": 5,
        }
    )


def _make_assess(date_submitted_values: list, id_assessments: list = None) -> pd.DataFrame:
    """Minimal studentAssessment rows."""
    n = len(date_submitted_values)
    if id_assessments is None:
        id_assessments = [100] * n
    return pd.DataFrame(
        {
            "id_student": 1,
            "id_assessment": id_assessments,
            "date_submitted": date_submitted_values,
            "score": 70.0,
            "code_module": "AAA",
            "code_presentation": "2013J",
            "is_banked": 0,
        }
    )


def _make_assessments_meta(due_date: int, id_assessment: int = 100) -> pd.DataFrame:
    """Minimal assessments metadata row."""
    return pd.DataFrame(
        {
            "id_assessment": [id_assessment],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "assessment_type": ["TMA"],
            "date": [due_date],
            "weight": [20.0],
        }
    )


# ---------------------------------------------------------------------------
# VLE boundary tests
# ---------------------------------------------------------------------------

class TestVleBoundary:
    def test_interaction_at_cutoff_included(self):
        vle = _make_vle([WINDOW])
        vle_w, _ = filter_window(vle, _make_assess([]), _make_assessments_meta(WINDOW), WINDOW)
        assert len(vle_w) == 1

    def test_interaction_before_cutoff_included(self):
        vle = _make_vle([WINDOW - 1])
        vle_w, _ = filter_window(vle, _make_assess([]), _make_assessments_meta(WINDOW), WINDOW)
        assert len(vle_w) == 1

    def test_interaction_after_cutoff_excluded(self):
        vle = _make_vle([WINDOW + 1])
        vle_w, _ = filter_window(vle, _make_assess([]), _make_assessments_meta(WINDOW), WINDOW)
        assert len(vle_w) == 0


# ---------------------------------------------------------------------------
# Assessment Guard 2 (submission-date) boundary tests
# ---------------------------------------------------------------------------

class TestSubmissionDateBoundary:
    """Guard 2: date_submitted <= window (due date is within window for all cases)."""

    def _run(self, date_submitted: int) -> int:
        """Return number of submissions retained by filter_window."""
        assess = _make_assess([date_submitted])
        meta = _make_assessments_meta(due_date=WINDOW - 5)  # due well before cutoff
        _, assess_w = filter_window(_make_vle([]), assess, meta, WINDOW)
        return len(assess_w)

    def test_submitted_at_cutoff_included(self):
        assert self._run(WINDOW) == 1

    def test_submitted_one_day_before_cutoff_included(self):
        assert self._run(WINDOW - 1) == 1

    def test_submitted_one_day_after_cutoff_excluded(self):
        assert self._run(WINDOW + 1) == 0


# ---------------------------------------------------------------------------
# Assessment Guard 1 (due-date) boundary tests
# ---------------------------------------------------------------------------

class TestDueDateBoundary:
    """Guard 1: assessments.date <= window (submission date is within window for all)."""

    def _run(self, due_date: int) -> int:
        assess = _make_assess([WINDOW - 1])  # submitted well before cutoff
        meta = _make_assessments_meta(due_date=due_date)
        _, assess_w = filter_window(_make_vle([]), assess, meta, WINDOW)
        return len(assess_w)

    def test_due_at_cutoff_included(self):
        assert self._run(WINDOW) == 1

    def test_due_one_day_before_cutoff_included(self):
        assert self._run(WINDOW - 1) == 1

    def test_due_one_day_after_cutoff_excluded(self):
        assert self._run(WINDOW + 1) == 0


# ---------------------------------------------------------------------------
# Combined guard test — due date in window but submitted after
# ---------------------------------------------------------------------------

class TestDualGuard:
    def test_due_in_window_submitted_after_excluded(self):
        """Assessment is due before cutoff but submitted after — must be excluded."""
        assess = _make_assess([WINDOW + 5])        # submitted late
        meta = _make_assessments_meta(due_date=WINDOW - 3)  # due before cutoff
        _, assess_w = filter_window(_make_vle([]), assess, meta, WINDOW)
        assert len(assess_w) == 0, (
            "Submission after cutoff should be excluded even when due date is within window"
        )

    def test_both_guards_satisfied_included(self):
        """Assessment due before cutoff AND submitted before cutoff — must be included."""
        assess = _make_assess([WINDOW - 1])
        meta = _make_assessments_meta(due_date=WINDOW - 3)
        _, assess_w = filter_window(_make_vle([]), assess, meta, WINDOW)
        assert len(assess_w) == 1
