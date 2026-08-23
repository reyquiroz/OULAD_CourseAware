"""
Smoke test verifying that GNN training successfully runs and loss decreases.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_model import EnrollmentGNN, build_train_subgraph, run_training_loop


def test_training_loss_decreases_on_toy_graph():
    """Verify that training loop executes and loss decreases on a toy graph."""
    data = HeteroData()

    # 5 students, 2 course presentations, 3 assessments, 3 VLE resources
    data["student"].x = torch.randn((5, 8), dtype=torch.float32)
    data["course_presentation"].x = torch.randn((2, 8), dtype=torch.float32)
    data["assessment"].x = torch.randn((3, 8), dtype=torch.float32)
    data["vle_resource"].x = torch.randn((3, 8), dtype=torch.float32)

    # enrolled_in edges
    ei_key = ("student", "enrolled_in", "course_presentation")
    data[ei_key].edge_index = torch.tensor(
        [[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long
    )
    # 6 edges, 4 cols
    data[ei_key].edge_attr = torch.randn((6, 4), dtype=torch.float32)
    # Targets with both 0s and 1s
    data[ei_key].y = torch.tensor([0., 1., 0., 1., 0., 1.], dtype=torch.float32)
    data[ei_key].enrollment_idx = torch.arange(6, dtype=torch.long)

    # Reverse enrolled_in
    rev_ei_key = ("course_presentation", "rev_enrolled_in", "student")
    data[rev_ei_key].edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 1], [0, 1, 2, 2, 3, 4]], dtype=torch.long
    )

    # course structure: contains_assess
    ca_key = ("course_presentation", "contains_assess", "assessment")
    data[ca_key].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]], dtype=torch.long
    )
    rev_ca_key = ("assessment", "rev_contains_assess", "course_presentation")
    data[rev_ca_key].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]], dtype=torch.long
    )

    # course structure: has_resource
    hr_key = ("course_presentation", "has_resource", "vle_resource")
    data[hr_key].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]], dtype=torch.long
    )
    rev_hr_key = ("vle_resource", "rev_has_resource", "course_presentation")
    data[rev_hr_key].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]], dtype=torch.long
    )

    # behaviors: submitted
    sub_key = ("student", "submitted", "assessment")
    data[sub_key].edge_index = torch.tensor(
        [[1, 3], [0, 2]], dtype=torch.long
    )
    data[sub_key].edge_attr = torch.randn((2, 2), dtype=torch.float32)
    rev_sub_key = ("assessment", "rev_submitted", "student")
    data[rev_sub_key].edge_index = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.long
    )

    # behaviors: interacted_with
    iw_key = ("student", "interacted_with", "vle_resource")
    data[iw_key].edge_index = torch.tensor(
        [[1, 3], [0, 2]], dtype=torch.long
    )
    data[iw_key].edge_attr = torch.randn((2, 2), dtype=torch.float32)
    rev_iw_key = ("vle_resource", "rev_interacted_with", "student")
    data[rev_iw_key].edge_index = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.long
    )

    # Setup train & val masks (covering all edges so both classes exist in val)
    train_mask = torch.ones(6, dtype=torch.bool)
    val_mask = torch.ones(6, dtype=torch.bool)

    # Construct train subgraph
    train_subgraph = build_train_subgraph(data, train_mask)

    # Construct GNN model with hidden_dim=16
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1]

    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=16,
        n_enrolled_in_attr=n_enrolled_in_attr
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Run training for 5 epochs
    best_val_auroc, best_epoch, train_losses, val_aurocs = run_training_loop(
        model=model,
        train_subgraph=train_subgraph,
        full_data=data,
        val_mask=val_mask,
        optimizer=optimizer,
        max_epochs=5,
        patience=10,
        pos_weight=None
    )

    # Assert that training loss decreases or stays close to zero
    assert len(train_losses) == 5
    assert train_losses[-1] <= train_losses[0]

    # Verify that forward pass returns correct logit shape matching edge count
    model.eval()
    with torch.no_grad():
        subgraph_logits = model(train_subgraph)
    assert subgraph_logits.shape[0] == train_subgraph[ei_key].edge_index.shape[1]
    assert subgraph_logits.shape[0] == 6
