"""
Tests for build_train_subgraph() in gnn_model.py.

Uses the _build_toy_graph helper from test_gnn_data_flow.py (no parquet
artifacts required — all three tests run on a fully in-memory toy graph).

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_subgraph.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Re-use the shared toy-graph builder from the data-flow test module so we
# don't duplicate the fixture definition.
from test_gnn_data_flow import _build_toy_graph  # noqa: E402

from gnn_model import build_train_subgraph  # noqa: E402

EI_KEY = ("student", "enrolled_in", "course_presentation")


# ---------------------------------------------------------------------------
# Test 1 — held-out students absent from the subgraph's enrolled_in src
# ---------------------------------------------------------------------------

def test_held_out_students_absent_from_subgraph():
    """Student 0 (enrollment row 0) must not appear in the subgraph's
    enrolled_in src tensor when train_mask excludes that enrollment.

    Toy graph enrolled_in:
        row 0: student 0 → CP 0
        row 1: student 1 → CP 0
        row 2: student 1 → CP 1
        row 3: student 2 → CP 1
    train_mask = [False, True, True, True]  (exclude row 0)
    """
    data = _build_toy_graph()

    train_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
    sub = build_train_subgraph(data, train_mask)

    src_students = sub[EI_KEY].edge_index[0].tolist()
    assert 0 not in src_students, (
        f"Student 0 should be absent from subgraph enrolled_in src, got: {src_students}"
    )


# ---------------------------------------------------------------------------
# Test 2 — subgraph.y length equals number of True entries in train_mask
# ---------------------------------------------------------------------------

def test_subgraph_y_length_equals_train_count():
    """subgraph[ei_key].y.shape[0] must equal train_mask.sum()."""
    data = _build_toy_graph()

    train_mask = torch.tensor([True, True, False, True], dtype=torch.bool)
    sub = build_train_subgraph(data, train_mask)

    expected = int(train_mask.sum().item())
    actual = sub[EI_KEY].y.shape[0]
    assert actual == expected, (
        f"Expected subgraph y length {expected}, got {actual}"
    )


# ---------------------------------------------------------------------------
# Test 3 — original full graph's edge_index is unchanged after subgraph build
# ---------------------------------------------------------------------------

def test_full_graph_val_still_has_all_edges():
    """build_train_subgraph must not modify the original data in-place.
    The full graph's enrolled_in edge count must be the same before and after.
    """
    data = _build_toy_graph()
    original_edge_count = data[EI_KEY].edge_index.shape[1]

    train_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    _sub = build_train_subgraph(data, train_mask)

    after_edge_count = data[EI_KEY].edge_index.shape[1]
    assert after_edge_count == original_edge_count, (
        f"Full graph enrolled_in edge count changed: "
        f"{original_edge_count} → {after_edge_count}"
    )
