"""
Tests for GNN data flow: edge attributes, held-out masking, split alignment,
and prediction head alignment.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_gnn_data_flow.py -v

Tests that require artifact parquet files are skipped automatically on a fresh
clone (ARTIFACTS_PRESENT = False).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Artifact presence guard — skip tests that need real parquet files on CI
# ---------------------------------------------------------------------------

_ARTIFACT_DIR = Path("results/graph/artifacts")
_SPLIT_DIR = Path("results/graph/evaluation/week08/splits")
ARTIFACTS_PRESENT = (
    (_ARTIFACT_DIR / "week08_enrollments.parquet").exists()
    and (_SPLIT_DIR / "week08_random_split.parquet").exists()
)

# ---------------------------------------------------------------------------
# Toy-graph fixture shared by tests 3 (and optionally 1)
# ---------------------------------------------------------------------------

def _build_toy_graph():
    """Return a minimal HeteroData object with:

    Nodes
    -----
    - 3 students  (indices 0, 1, 2)
    - 2 course_presentations  (CP 0, CP 1)
    - 3 assessments  (0 → CP 0, 1 → CP 0, 2 → CP 1)
    - 3 vle_resources  (0 → CP 0, 1 → CP 0, 2 → CP 1)

    Edges
    -----
    enrolled_in:
        student 0 → CP 0
        student 1 → CP 0
        student 1 → CP 1
        student 2 → CP 1

    contains_assess:
        CP 0 → assessment 0
        CP 0 → assessment 1
        CP 1 → assessment 2

    has_resource:
        CP 0 → vle_resource 0
        CP 0 → vle_resource 1
        CP 1 → vle_resource 2

    submitted:
        student 1 → assessment 0  (belongs to CP 0 — should be removed when CP 0 held out)
        student 1 → assessment 2  (belongs to CP 1 — must be preserved)

    interacted_with:
        student 1 → vle_resource 0  (belongs to CP 0 — should be removed)
        student 1 → vle_resource 2  (belongs to CP 1 — must be preserved)
    """
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # ── node features (minimal 1-D placeholders) ──────────────────────────
    data["student"].x = torch.zeros((3, 1), dtype=torch.float32)
    data["course_presentation"].x = torch.zeros((2, 1), dtype=torch.float32)
    data["assessment"].x = torch.zeros((3, 1), dtype=torch.float32)
    data["vle_resource"].x = torch.zeros((3, 1), dtype=torch.float32)

    # ── enrolled_in ────────────────────────────────────────────────────────
    ei_key = ("student", "enrolled_in", "course_presentation")
    data[ei_key].edge_index = torch.tensor(
        [[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long
    )
    # age_band one-hot (2 classes) + 2 numeric = 4 cols
    data[ei_key].edge_attr = torch.zeros((4, 4), dtype=torch.float32)
    data[ei_key].y = torch.zeros(4, dtype=torch.float32)
    data[ei_key].enrollment_idx = torch.arange(4, dtype=torch.long)

    # reverse enrolled_in
    rev_ei_key = ("course_presentation", "rev_enrolled_in", "student")
    data[rev_ei_key].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long
    )

    # ── contains_assess ────────────────────────────────────────────────────
    ca_key = ("course_presentation", "contains_assess", "assessment")
    data[ca_key].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]], dtype=torch.long
    )
    rev_ca_key = ("assessment", "rev_contains_assess", "course_presentation")
    data[rev_ca_key].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]], dtype=torch.long
    )

    # ── has_resource ───────────────────────────────────────────────────────
    hr_key = ("course_presentation", "has_resource", "vle_resource")
    data[hr_key].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]], dtype=torch.long
    )
    rev_hr_key = ("vle_resource", "rev_has_resource", "course_presentation")
    data[rev_hr_key].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]], dtype=torch.long
    )

    # ── submitted ──────────────────────────────────────────────────────────
    #   student 1 → assessment 0  (CP 0's assessment — should be removed)
    #   student 1 → assessment 2  (CP 1's assessment — must be preserved)
    sub_key = ("student", "submitted", "assessment")
    data[sub_key].edge_index = torch.tensor(
        [[1, 1], [0, 2]], dtype=torch.long
    )
    data[sub_key].edge_attr = torch.zeros((2, 1), dtype=torch.float32)
    rev_sub_key = ("assessment", "rev_submitted", "student")
    data[rev_sub_key].edge_index = torch.tensor(
        [[0, 2], [1, 1]], dtype=torch.long
    )

    # ── interacted_with ────────────────────────────────────────────────────
    #   student 1 → vle_resource 0  (CP 0's resource — should be removed)
    #   student 1 → vle_resource 2  (CP 1's resource — must be preserved)
    iw_key = ("student", "interacted_with", "vle_resource")
    data[iw_key].edge_index = torch.tensor(
        [[1, 1], [0, 2]], dtype=torch.long
    )
    data[iw_key].edge_attr = torch.zeros((2, 1), dtype=torch.float32)
    rev_iw_key = ("vle_resource", "rev_interacted_with", "student")
    data[rev_iw_key].edge_index = torch.tensor(
        [[0, 2], [1, 1]], dtype=torch.long
    )

    return data


# ---------------------------------------------------------------------------
# Test 1 — enrolled_in edge_attr shape
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts not found")
def test_enrolled_in_edge_attr_shape():
    """data[enrolled_in].edge_attr is not None and has shape (N, D>=3)."""
    from gnn_model import GraphDataLoader

    data = GraphDataLoader(week=8).load()
    ei_key = ("student", "enrolled_in", "course_presentation")

    attr = data[ei_key].edge_attr
    assert attr is not None, "edge_attr should not be None"
    assert attr.ndim == 2, f"Expected 2-D tensor, got shape {attr.shape}"

    N = data[ei_key].edge_index.shape[1]
    D = attr.shape[1]
    assert attr.shape[0] == N, (
        f"edge_attr row count {attr.shape[0]} != edge count {N}"
    )
    # age_band one-hot (≥2 cols) + num_of_prev_attempts + studied_credits → D ≥ 3
    assert D >= 3, f"Expected D>=3 attribute columns, got {D}"


# ---------------------------------------------------------------------------
# Test 2 — edge attr reaches prediction head (logit count == #enrolled_in edges)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts not found")
def test_edge_attr_reaches_prediction_head():
    """Forward pass produces one logit per enrolled_in edge (32,593 for week 8)."""
    from gnn_model import GraphDataLoader, EnrollmentGNN

    data = GraphDataLoader(week=8).load()
    ei_key = ("student", "enrolled_in", "course_presentation")

    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1]
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}

    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=64,
        n_enrolled_in_attr=n_enrolled_in_attr,
    )
    model.eval()
    with torch.no_grad():
        logits = model(data)

    expected_n = data[ei_key].edge_index.shape[1]
    assert logits.shape[0] == expected_n, (
        f"logits length {logits.shape[0]} != enrolled_in edge count {expected_n}"
    )
    # Week-8 specific sanity check
    assert expected_n == 32593, (
        f"Expected 32593 enrollments for week 8, got {expected_n}"
    )


# ---------------------------------------------------------------------------
# Test 3 — LCPO masking does not strip cross-course edges (toy graph, no files)
# ---------------------------------------------------------------------------

def test_lcpo_mask_does_not_strip_cross_course_edges():
    """Holding out CP 0 removes only CP 0's submitted/interacted_with edges.

    Cross-course edges (student 1's links to CP 1's assessment and vle_resource)
    must survive masking.
    """
    from run_gnn_experiment import _mask_held_out_edges

    data = _build_toy_graph()

    # Build a minimal enroll_df that mirrors the toy enrolled_in edges:
    #   row 0: student 0 → CP 0  (module AAA, presentation 2013J)
    #   row 1: student 1 → CP 0
    #   row 2: student 1 → CP 1  (module BBB, presentation 2013J)
    #   row 3: student 2 → CP 1
    enroll_df = pd.DataFrame(
        {
            "id_student": [0, 1, 1, 2],
            "code_module": ["AAA", "AAA", "BBB", "BBB"],
            "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
            "target": [0, 0, 0, 0],
        }
    )

    # Hold out CP 0 (node index 0, module AAA / pres 2013J)
    masked = _mask_held_out_edges(
        data,
        cp_node_idx=0,
        enroll_df=enroll_df,
        held_out_module="AAA",
        held_out_pres="2013J",
    )

    sub_key = ("student", "submitted", "assessment")
    iw_key = ("student", "interacted_with", "vle_resource")

    # ── submitted checks ───────────────────────────────────────────────────
    sub_ei = masked[sub_key].edge_index  # (2, E_remaining)
    # (student 1 → assessment 0) must be removed  →  assessment 0 not in dst
    cp0_assess_nodes = {0, 1}  # assessments belonging to CP 0
    remaining_dst_sub = set(sub_ei[1].tolist())
    assert 0 not in remaining_dst_sub, (
        "assessment 0 (CP 0) should have been removed from submitted edges"
    )
    # (student 1 → assessment 2) must be preserved  →  assessment 2 in dst
    assert 2 in remaining_dst_sub, (
        "assessment 2 (CP 1) must be preserved in submitted edges"
    )

    # ── interacted_with checks ─────────────────────────────────────────────
    iw_ei = masked[iw_key].edge_index  # (2, E_remaining)
    remaining_dst_iw = set(iw_ei[1].tolist())
    assert 0 not in remaining_dst_iw, (
        "vle_resource 0 (CP 0) should have been removed from interacted_with edges"
    )
    assert 2 in remaining_dst_iw, (
        "vle_resource 2 (CP 1) must be preserved in interacted_with edges"
    )


# ---------------------------------------------------------------------------
# Test 4 — split mask length matches enrollment parquet row count
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts not found")
def test_split_mask_length_matches_enrollments():
    """train/val/test masks from load_split_masks(week=8, 'random') each have
    length equal to the number of rows in week08_enrollments.parquet."""
    from gnn_model import load_split_masks

    train_mask, val_mask, test_mask = load_split_masks(week=8, split_type="random")

    enroll_df = pd.read_parquet(
        _ARTIFACT_DIR / "week08_enrollments.parquet"
    )
    n_enrollments = len(enroll_df)

    assert train_mask.shape[0] == n_enrollments, (
        f"train_mask length {train_mask.shape[0]} != {n_enrollments}"
    )
    assert val_mask.shape[0] == n_enrollments, (
        f"val_mask length {val_mask.shape[0]} != {n_enrollments}"
    )
    assert test_mask.shape[0] == n_enrollments, (
        f"test_mask length {test_mask.shape[0]} != {n_enrollments}"
    )


# ---------------------------------------------------------------------------
# Test 5 — no student overlap in LCPO val draw (student-grouped sampling)
# ---------------------------------------------------------------------------

def test_no_student_overlap_in_lcpo_val_draw():
    """Student-grouped val sampling produces disjoint train and val student sets.

    This unit-tests only the sampling logic, not the full LCPO experiment.
    The same logic used in run_lcpo_experiment() is reproduced here directly
    to verify the guarantee.
    """
    rng_seed = 42

    # Synthetic enrollment table: 10 students, some have multiple enrollments
    rows = []
    for sid in range(10):
        rows.append({"id_student": sid, "code_module": "AAA", "code_presentation": "2013J"})
        if sid % 3 == 0:
            # Every 3rd student also has a second enrollment in a different CP
            rows.append({"id_student": sid, "code_module": "BBB", "code_presentation": "2013J"})
    enroll_df = pd.DataFrame(rows).reset_index(drop=True)

    # Identify non-test rows (all rows, since we haven't held anything out here)
    train_all_np = np.ones(len(enroll_df), dtype=bool)

    # Replicate the sampling logic from run_lcpo_experiment():
    #   get unique student IDs among non-test rows, sample 10%, expand to rows
    train_student_ids = enroll_df.loc[train_all_np, "id_student"].unique()
    rng = np.random.default_rng(rng_seed)
    val_size = max(1, int(0.10 * len(train_student_ids)))
    val_student_ids = set(rng.choice(train_student_ids, size=val_size, replace=False).tolist())

    val_mask_np = train_all_np & enroll_df["id_student"].isin(val_student_ids).to_numpy()
    train_mask_np = train_all_np & ~val_mask_np

    val_students = set(enroll_df.loc[val_mask_np, "id_student"].unique().tolist())
    train_students = set(enroll_df.loc[train_mask_np, "id_student"].unique().tolist())

    overlap = val_students & train_students
    assert len(overlap) == 0, (
        f"Students appear in both train and val masks: {overlap}"
    )
