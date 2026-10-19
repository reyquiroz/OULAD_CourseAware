"""
tests/test_recency_sentinel.py
------------------------------
Verifies that enrollments with no VLE activity in the prediction window get
recency == window_days + 1 (never active) and active_in_window == False,
rather than recency == 0.0 (which would mean "active on the cutoff day").

Run from the project root:
    pytest tests/test_recency_sentinel.py -v
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_pipeline import build_edge_tables  # noqa: E402

WINDOW_DAYS = 28  # arbitrary; the test must hold for any window length


def _build_inputs():
    """Two enrollments in one course: student 1 has clicks, student 2 has none."""
    mod, pres = "AAA", "2013J"
    student_info = pd.DataFrame({
        "id_student": [1, 2],
        "code_module": [mod, mod],
        "code_presentation": [pres, pres],
        "age_band": ["0-35", "35-55"],
        "num_of_prev_attempts": [0, 0],
        "studied_credits": [60, 60],
    })
    student_vle = pd.DataFrame({
        "id_student": [1, 1],
        "code_module": [mod, mod],
        "code_presentation": [pres, pres],
        "id_site": [10, 10],
        "date": [3, 20],
        "sum_click": [5, 7],
    })
    student_assess = pd.DataFrame({
        "id_student": [1],
        "id_assessment": [100],
        "score": [80.0],
        "date_submitted": [10],
        "code_module": [mod],
        "code_presentation": [pres],
    })
    vle = pd.DataFrame({
        "id_site": [10],
        "code_module": [mod],
        "code_presentation": [pres],
    })
    filtered = {
        "student_info": student_info,
        "student_vle": student_vle,
        "student_assess": student_assess,
        "vle": vle,
    }
    nodes = {
        "student": pd.DataFrame({"id_student": [1, 2], "node_idx": [0, 1]}),
        "course_presentation": pd.DataFrame({
            "code_module": [mod], "code_presentation": [pres], "node_idx": [0],
        }),
        "assessment": pd.DataFrame({
            "id_assessment": [100],
            "code_module": [mod],
            "code_presentation": [pres],
            "node_idx": [0],
        }),
        "vle_resource": pd.DataFrame({"id_site": [10], "node_idx": [0]}),
    }
    return filtered, nodes


def test_inactive_enrollment_gets_recency_sentinel_and_active_flag_false():
    filtered, nodes = _build_inputs()
    edges, _ = build_edge_tables(filtered, nodes, window_days=WINDOW_DAYS)
    ei = edges["enrolled_in"].set_index("src")  # student 1 -> src 0, student 2 -> src 1

    inactive = ei.loc[1]
    assert inactive["recency"] == WINDOW_DAYS + 1
    assert not bool(inactive["active_in_window"])


def test_active_enrollment_has_real_recency_and_active_flag_true():
    filtered, nodes = _build_inputs()
    edges, _ = build_edge_tables(filtered, nodes, window_days=WINDOW_DAYS)
    ei = edges["enrolled_in"].set_index("src")

    active = ei.loc[0]
    assert active["recency"] < WINDOW_DAYS + 1
    assert active["recency"] == WINDOW_DAYS - 20  # last click on day 20
    assert bool(active["active_in_window"])


def test_recency_sentinel_when_no_vle_rows_at_all():
    """Empty VLE window: every enrollment is inactive."""
    filtered, nodes = _build_inputs()
    filtered["student_vle"] = filtered["student_vle"].iloc[0:0]
    edges, _ = build_edge_tables(filtered, nodes, window_days=WINDOW_DAYS)
    ei = edges["enrolled_in"]

    assert (ei["recency"] == WINDOW_DAYS + 1).all()
    assert not ei["active_in_window"].any()
