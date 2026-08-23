"""
tests/test_no_cross_course.py
------------------------------
Verifies that perturbing a held-out enrollment's edge attributes does not
affect the training logits for another enrollment of the same student.

This guards against the cross-course aggregation bug: if enrolled_in edge
attributes were scattered into the shared student node (as in the old
pyg_scatter implementation), changing a held-out enrollment's attributes
would pollute the training subgraph's student embedding and change the
training logit.  With per-enrollment projection in the edge head, held-out
edges are excluded from the training subgraph entirely, so their attributes
can never reach the training logits.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_no_cross_course.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Re-use the shared toy-graph builder from the data-flow tests.
def _build_toy_graph():
    from test_gnn_data_flow import _build_toy_graph as _btg
    return _btg()


def test_perturbing_held_out_enrollment_does_not_affect_training_logits():
    """Changing a held-out enrollment's edge_attr leaves training logits unchanged.

    Setup
    -----
    Toy graph has student 1 enrolled in CP 0 (edge rows 1 in enrolled_in) and
    CP 1 (edge row 2).  The train_mask selects only the enrollment to CP 1
    (row index 2 in the full enrolled_in tensor), so the enrollment to CP 0
    (row index 1) is held out.

    Perturbation
    ------------
    After recording the training logit, we change the held-out edge_attr row
    (student 1 → CP 0, index 1 in the full graph) to random values, rebuild
    the train subgraph from the same mask, and run a second forward pass.

    Assertion
    ---------
    Both forward passes must produce identical logits because the held-out
    edge is stripped from the training subgraph before the model sees it.
    """
    from gnn_model import EnrollmentGNN, build_train_subgraph

    torch.manual_seed(0)

    data = _build_toy_graph()

    ei_key = ("student", "enrolled_in", "course_presentation")
    # Full enrolled_in edges: rows 0–3.  Row 2 = student 1 → CP 1 (training).
    # train_mask keeps only row 2.
    n_edges = data[ei_key].edge_index.shape[1]  # 4
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[2] = True  # student 1 → CP 1

    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1]

    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=8,
        n_enrolled_in_attr=n_enrolled_in_attr,
    )
    model.eval()

    # --- First forward pass ---
    subgraph_before = build_train_subgraph(data, train_mask)
    with torch.no_grad():
        logits_before = model(subgraph_before)

    # --- Perturb the held-out enrollment's edge_attr (row 1: student 1 → CP 0) ---
    data[ei_key].edge_attr = data[ei_key].edge_attr.clone()
    data[ei_key].edge_attr[1] = torch.randn(n_enrolled_in_attr)

    # --- Second forward pass with same train_mask ---
    subgraph_after = build_train_subgraph(data, train_mask)
    with torch.no_grad():
        logits_after = model(subgraph_after)

    assert torch.allclose(logits_before, logits_after), (
        f"Perturbing a held-out enrollment changed training logits.\n"
        f"  before: {logits_before}\n"
        f"  after:  {logits_after}"
    )
