"""
Tests for train-mask-aware normalization in _normalize_numeric_features.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_normalization.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torch_geometric.data import HeteroData
from gnn_model import _normalize_numeric_features


# ---------------------------------------------------------------------------
# Toy-graph factory
# ---------------------------------------------------------------------------

def _make_toy_graph(n_ei: int = 6, n_students: int = 4) -> HeteroData:
    """Build a minimal HeteroData for normalization tests.

    Topology
    --------
    - n_students student nodes, 2 course_presentation nodes
    - n_ei enrolled_in edges  (student 0..n_students-1 → CP 0 or 1)
    - 4 submitted edges       (student 0 & 1 → assessments 0 & 1)
    - 3 interacted_with edges (student 0, 1, 2 → vle_resources 0, 1, 2)
    - 2 assessment nodes, 3 vle_resource nodes

    All numeric feature values are chosen to be non-trivially different so
    that global vs. train-subset statistics produce measurably different
    normalised values.
    """
    data = HeteroData()

    # ── Student nodes: one numeric feature column (e.g. "age proxy") ────────
    # Values: [1, 2, 3, 4, …] — clearly non-one-hot
    data["student"].x = torch.arange(1, n_students + 1, dtype=torch.float32).unsqueeze(1)

    # ── Course_presentation nodes ────────────────────────────────────────────
    data["course_presentation"].x = torch.tensor([[10.0], [20.0]])

    # ── Assessment nodes ─────────────────────────────────────────────────────
    data["assessment"].x = torch.tensor([[5.0], [15.0]])

    # ── vle_resource nodes ───────────────────────────────────────────────────
    data["vle_resource"].x = torch.tensor([[3.0], [6.0], [9.0]])

    # ── enrolled_in edges ────────────────────────────────────────────────────
    # Alternate students across 2 CPs; numeric attrs are [10, 20, 30, …]
    ei_src = torch.arange(n_ei, dtype=torch.long) % n_students
    ei_dst = torch.arange(n_ei, dtype=torch.long) % 2
    ei_key = ("student", "enrolled_in", "course_presentation")
    data[ei_key].edge_index = torch.stack([ei_src, ei_dst], dim=0)
    # Use a single clearly-numeric column (not one-hot)
    data[ei_key].edge_attr = (torch.arange(n_ei, dtype=torch.float32) * 10.0 + 10.0).unsqueeze(1)
    data[ei_key].y = torch.zeros(n_ei, dtype=torch.float32)
    data[ei_key].enrollment_idx = torch.arange(n_ei, dtype=torch.long)

    # reverse enrolled_in
    rev_ei_key = ("course_presentation", "rev_enrolled_in", "student")
    data[rev_ei_key].edge_index = torch.stack([ei_dst, ei_src], dim=0)

    # ── submitted edges ───────────────────────────────────────────────────────
    sub_key = ("student", "submitted", "assessment")
    data[sub_key].edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    data[sub_key].edge_attr = torch.tensor([[50.0], [60.0], [70.0], [80.0]])
    rev_sub_key = ("assessment", "rev_submitted", "student")
    data[rev_sub_key].edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)

    # ── interacted_with edges ─────────────────────────────────────────────────
    iw_key = ("student", "interacted_with", "vle_resource")
    data[iw_key].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    # Three numeric cols: [total_clicks, n_interactions, active_days]
    data[iw_key].edge_attr = torch.tensor(
        [[10.0, 2.0, 3.0],
         [20.0, 4.0, 5.0],
         [30.0, 6.0, 7.0]],
        dtype=torch.float32,
    )
    rev_iw_key = ("vle_resource", "rev_interacted_with", "student")
    data[rev_iw_key].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

    return data


# ---------------------------------------------------------------------------
# Test (a) — train-mask normalization: training slice is zero-mean
# ---------------------------------------------------------------------------

def test_train_mask_enrolled_in_zero_mean():
    """When a 50/50 train mask is supplied, enrolled_in training rows are zero-mean."""
    data = _make_toy_graph(n_ei=6)

    # First 3 edges = train, last 3 = test
    train_mask = torch.tensor([True, True, True, False, False, False])

    data_norm = _normalize_numeric_features(data, train_edge_mask=train_mask)

    ei_key = ("student", "enrolled_in", "course_presentation")
    train_vals = data_norm[ei_key].edge_attr[train_mask, 0]
    # Training subset must be zero-mean (within float tolerance)
    assert abs(train_vals.mean().item()) < 1e-5, (
        f"Expected zero-mean on training enrolled_in rows, got {train_vals.mean().item()}"
    )


def test_train_mask_differs_from_global():
    """Test subset normalizes differently from global (different mean/std used)."""
    data_global = _make_toy_graph(n_ei=6)
    data_train  = _make_toy_graph(n_ei=6)

    # Use only the first 2 edges as "training" — very asymmetric
    train_mask = torch.tensor([True, True, False, False, False, False])

    ei_key = ("student", "enrolled_in", "course_presentation")

    out_global = _normalize_numeric_features(data_global, train_edge_mask=None)
    out_train  = _normalize_numeric_features(data_train,  train_edge_mask=train_mask)

    # The last row (index 5) was outside training for train-mask run;
    # its normalised value must differ from the global run.
    val_global = out_global[ei_key].edge_attr[5, 0].item()
    val_train  = out_train [ei_key].edge_attr[5, 0].item()
    assert abs(val_global - val_train) > 1e-3, (
        f"Expected different normalised values for test edge (global={val_global:.4f} "
        f"train-masked={val_train:.4f}), but they are equal."
    )


# ---------------------------------------------------------------------------
# Test (b) — None mask produces global (backward-compatible) behaviour
# ---------------------------------------------------------------------------

def test_none_mask_equals_global():
    """train_edge_mask=None produces identical output to all-True mask."""
    data_none = _make_toy_graph(n_ei=6)
    data_all  = _make_toy_graph(n_ei=6)

    ei_key = ("student", "enrolled_in", "course_presentation")
    all_train = torch.ones(6, dtype=torch.bool)

    out_none = _normalize_numeric_features(data_none, train_edge_mask=None)
    out_all  = _normalize_numeric_features(data_all,  train_edge_mask=all_train)

    # enrolled_in attrs must be identical
    assert torch.allclose(
        out_none[ei_key].edge_attr,
        out_all [ei_key].edge_attr,
        atol=1e-5,
    ), "Global (None mask) and all-True mask should produce identical results."

    # Student node features must also be identical
    assert torch.allclose(
        out_none["student"].x,
        out_all ["student"].x,
        atol=1e-5,
    ), "Student node features differ between None mask and all-True mask."


# ---------------------------------------------------------------------------
# Test (c) — student node stats use training-student subset
# ---------------------------------------------------------------------------

def test_student_node_stats_from_train_subset():
    """Student node features use stats from training students only."""
    # 4 students: train = students 0 & 1, test = students 2 & 3
    # enrolled_in: student i → CP (i % 2), i in 0..3
    data_train  = _make_toy_graph(n_ei=4, n_students=4)
    data_global = _make_toy_graph(n_ei=4, n_students=4)

    # Students 0 & 1 appear in edges 0 & 1 → train mask = [True, True, False, False]
    train_mask = torch.tensor([True, True, False, False])

    out_train  = _normalize_numeric_features(data_train,  train_edge_mask=train_mask)
    out_global = _normalize_numeric_features(data_global, train_edge_mask=None)

    # Training students (0 & 1) should be zero-mean under the train-masked run
    train_student_vals = out_train["student"].x[[0, 1], 0]
    assert abs(train_student_vals.mean().item()) < 1e-5, (
        f"Expected training students to be zero-mean, got {train_student_vals.mean().item()}"
    )

    # Test students (2 & 3) should differ between runs
    test_student_global = out_global["student"].x[[2, 3], 0]
    test_student_train  = out_train ["student"].x[[2, 3], 0]
    assert not torch.allclose(test_student_global, test_student_train, atol=1e-3), (
        "Test-student normalised values should differ between global and train-masked runs."
    )
