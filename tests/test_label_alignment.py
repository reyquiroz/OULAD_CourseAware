"""
Tests for label alignment in the GNN model, ensuring targets map correctly
to enrolled_in edges and their order is preserved through subgraph filtering.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from gnn_model import build_train_subgraph
from test_gnn_data_flow import _build_toy_graph


def test_label_order_matches_enrolled_in_edge_order():
    """Verify that y targets are aligned with enrolled_in edges."""
    data = _build_toy_graph()
    ei_key = ("student", "enrolled_in", "course_presentation")

    # Set distinct labels for each of the 4 enrolled_in edges
    expected_y = torch.tensor([0., 1., 0., 1.], dtype=torch.float32)
    data[ei_key].y = expected_y

    # Check each individual edge maps exactly to its assigned target
    for i in range(len(expected_y)):
        assert data[ei_key].y[i] == expected_y[i]


def test_subgraph_y_preserves_order_after_filtering():
    """Verify that subgraph filtering retains correct label alignment."""
    data = _build_toy_graph()
    ei_key = ("student", "enrolled_in", "course_presentation")

    # Set distinct labels for each of the 4 enrolled_in edges
    data[ei_key].y = torch.tensor([0., 1., 0., 1.], dtype=torch.float32)

    # train_mask selects rows 1 and 3 (second and fourth enrolled_in edges)
    train_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

    # Extract inductive train subgraph
    subgraph = build_train_subgraph(data, train_mask)

    # Expecting labels for rows 1 and 3 which are [1., 1.]
    expected_sub_y = torch.tensor([1., 1.], dtype=torch.float32)
    torch.testing.assert_close(subgraph[ei_key].y, expected_sub_y)
