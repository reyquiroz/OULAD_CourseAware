"""
Tests for exclude_early_unregistered in src/oulad_data.py.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_exclude_early_unregistered.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import PREDICTION_WINDOWS
from oulad_data import exclude_early_unregistered


def test_exclude_early_unregistered():
    cutoff = PREDICTION_WINDOWS["week_8"]

    enrollments = pd.DataFrame(
        {
            "id_student": [1, 2, 3, 4, 5],
            "code_module": ["AAA"] * 5,
            "code_presentation": ["2013J"] * 5,
            "target": [1, 1, 1, 0, 1],
        }
    )
    registration = pd.DataFrame(
        {
            "id_student": [1, 2, 3, 4, 5],
            "code_module": ["AAA"] * 5,
            "code_presentation": ["2013J"] * 5,
            # before cutoff, at cutoff, after cutoff, never, before cutoff
            "date_unregistration": [cutoff - 10, cutoff, cutoff + 1, np.nan, -5],
        }
    )

    filtered, n_removed = exclude_early_unregistered(enrollments, registration, cutoff)

    assert n_removed == 3
    assert sorted(filtered["id_student"]) == [3, 4]
    assert list(filtered.columns) == list(enrollments.columns)


def test_exclude_early_unregistered_same_student_other_course_kept():
    """Filtering is per enrollment triple, not per student."""
    enrollments = pd.DataFrame(
        {
            "id_student": [1, 1],
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2013J"],
        }
    )
    registration = pd.DataFrame(
        {
            "id_student": [1, 1],
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2013J"],
            "date_unregistration": [10, np.nan],
        }
    )

    filtered, n_removed = exclude_early_unregistered(enrollments, registration, 56)

    assert n_removed == 1
    assert filtered["code_module"].tolist() == ["BBB"]
