"""
Tests that every model reads the same persisted splits.

Covers: split_loader alignment by enrollment triple, tabular == GNN partitions for the
same (week, seed / fold), and that no pipeline recomputes masks on its own.

Tests that need the regenerated week-8 artifacts skip themselves when absent.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import GRAPH_ARTIFACTS_DIR, GRAPH_EVALUATION_DIR  # noqa: E402
from split_loader import (  # noqa: E402
    KEY_COLS,
    load_lcpo_folds,
    load_lcpo_masks,
    load_random_split,
)

WEEK = 8
_SPLIT_DIR = GRAPH_EVALUATION_DIR / f"week{WEEK:02d}" / "splits"
_PRESENT = (
    (GRAPH_ARTIFACTS_DIR / f"week{WEEK:02d}_enrollments.parquet").exists()
    and (_SPLIT_DIR / f"week{WEEK:02d}_random_split_seed42.parquet").exists()
    and (_SPLIT_DIR / f"week{WEEK:02d}_lcpo_masks.parquet").exists()
)
needs_artifacts = pytest.mark.skipif(not _PRESENT, reason="week-8 artifacts/splits not generated")


def _keys(df, mask):
    return set(map(tuple, df.loc[mask, KEY_COLS].to_numpy()))


# ---------------------------------------------------------------------------
# Alignment (synthetic, no files needed)
# ---------------------------------------------------------------------------

def _write_toy_split(tmp_path):
    """Two-week-free toy: 6 enrollments, one student with two enrollments."""
    df = pd.DataFrame({
        "id_student": [1, 2, 3, 3, 4, 5],
        "code_module": ["AAA", "AAA", "BBB", "CCC", "BBB", "CCC"],
        "code_presentation": ["2013J"] * 6,
        "target": [0, 1, 0, 1, 1, 0],
    })
    sp = df.copy()
    sp["is_train"] = [True, True, False, False, True, False]
    sp["is_val"] = [False, False, True, True, False, False]
    sp["is_test"] = [False, False, False, False, False, True]
    d = tmp_path / "week08" / "splits"
    d.mkdir(parents=True)
    sp.to_parquet(d / "week08_random_split_seed7.parquet", index=False)
    return df


def test_masks_follow_the_triple_not_the_row_position(tmp_path):
    df = _write_toy_split(tmp_path)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    tr, va, te = load_random_split(8, 7, frame=shuffled, eval_dir=tmp_path)

    expected_train = {(1, "AAA", "2013J"), (2, "AAA", "2013J"), (4, "BBB", "2013J")}
    assert _keys(shuffled, tr.to_numpy()) == expected_train
    assert _keys(shuffled, te.to_numpy()) == {(5, "CCC", "2013J")}
    assert len(tr) == len(shuffled)
    assert (tr.to_numpy() ^ va.to_numpy() ^ te.to_numpy()).all()  # each row in exactly one


def test_returns_boolean_series_aligned_to_frame_index(tmp_path):
    df = _write_toy_split(tmp_path)
    df.index = [10, 11, 12, 13, 14, 15]
    tr, va, te = load_random_split(8, 7, frame=df, eval_dir=tmp_path)
    for m in (tr, va, te):
        assert isinstance(m, pd.Series) and m.dtype == bool
        assert list(m.index) == [10, 11, 12, 13, 14, 15]


def test_different_population_raises(tmp_path):
    df = _write_toy_split(tmp_path)
    with pytest.raises(ValueError, match="not the same population"):
        load_random_split(8, 7, frame=df.iloc[:-1], eval_dir=tmp_path)


def test_same_size_but_different_enrollments_raises(tmp_path):
    df = _write_toy_split(tmp_path)
    other = df.copy()
    other.loc[0, "id_student"] = 999
    with pytest.raises(ValueError, match="absent from the persisted split"):
        load_random_split(8, 7, frame=other, eval_dir=tmp_path)


def test_missing_split_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="save_graph_splits.py"):
        load_random_split(8, 123, eval_dir=tmp_path)


# ---------------------------------------------------------------------------
# Same partitions for tabular and GNN paths (real artifacts)
# ---------------------------------------------------------------------------

@needs_artifacts
@pytest.mark.parametrize("seed", [42, 123, 7, 17, 99])
def test_random_split_identical_for_tabular_and_gnn_order(seed):
    from gnn_model import load_split_masks

    enroll = pd.read_parquet(GRAPH_ARTIFACTS_DIR / f"week{WEEK:02d}_enrollments.parquet")
    # "Tabular" order: any row order, aligned by triple.
    tab = enroll.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    t_tr, t_va, t_te = load_random_split(WEEK, seed, frame=tab)
    g_tr, g_va, g_te = (m.numpy() for m in load_split_masks(WEEK, "random", seed=seed))

    for tab_mask, gnn_mask in ((t_tr, g_tr), (t_va, g_va), (t_te, g_te)):
        assert _keys(tab, tab_mask.to_numpy()) == _keys(enroll, gnn_mask)

    assert not (g_tr & g_va).any() and not (g_tr & g_te).any() and not (g_va & g_te).any()
    assert (g_tr | g_va | g_te).all()


@needs_artifacts
def test_random_split_keeps_students_in_one_partition():
    enroll = pd.read_parquet(GRAPH_ARTIFACTS_DIR / f"week{WEEK:02d}_enrollments.parquet")
    tr, va, te = load_random_split(WEEK, 42, frame=enroll)
    s = lambda m: set(enroll.loc[m, "id_student"])  # noqa: E731
    assert s(tr).isdisjoint(s(va)) and s(tr).isdisjoint(s(te)) and s(va).isdisjoint(s(te))


@needs_artifacts
def test_lcpo_identical_for_tabular_and_gnn_order_all_folds():
    from gnn_model import load_split_masks

    enroll = pd.read_parquet(GRAPH_ARTIFACTS_DIR / f"week{WEEK:02d}_enrollments.parquet")
    tab = enroll.sample(frac=1.0, random_state=1).reset_index(drop=True)
    folds = load_lcpo_folds(WEEK)
    assert len(folds) == 22

    for _, f in folds.iterrows():
        i = int(f["fold_idx"])
        t_tr, t_va, t_te = load_lcpo_masks(WEEK, i, frame=tab)
        g_tr, g_va, g_te = (
            m.numpy() for m in load_split_masks(WEEK, "lcpo", fold_idx=i)
        )
        for tab_mask, gnn_mask in ((t_tr, g_tr), (t_va, g_va), (t_te, g_te)):
            assert _keys(tab, tab_mask.to_numpy()) == _keys(enroll, gnn_mask)

        held = (enroll["code_module"] == f["held_out_module"]) & (
            enroll["code_presentation"] == f["held_out_presentation"]
        )
        assert (g_te == held.to_numpy()).all()
        assert not (g_tr & g_te).any() and not (g_va & g_te).any() and not (g_tr & g_va).any()
        assert (g_tr | g_va | g_te).all()
        assert int(g_va.sum()) == int(f["n_val"])
        assert int(g_tr.sum() + g_va.sum()) == int(f["n_train"])


@needs_artifacts
def test_tabular_feature_table_matches_graph_enrollment_population():
    """The tabular build and the graph artifact hold the same enrollments (week 8)."""
    from run_matched_comparison import build_tabular_features

    _, _, df = build_tabular_features(WEEK)
    enroll = pd.read_parquet(GRAPH_ARTIFACTS_DIR / f"week{WEEK:02d}_enrollments.parquet")
    assert _keys(df, np.ones(len(df), bool)) == _keys(enroll, np.ones(len(enroll), bool))
    merged = df[KEY_COLS + ["target"]].merge(
        enroll[KEY_COLS + ["target"]], on=KEY_COLS, suffixes=("_tab", "_gnn")
    )
    assert (merged["target_tab"] == merged["target_gnn"]).all()


# ---------------------------------------------------------------------------
# No pipeline recomputes masks on its own
# ---------------------------------------------------------------------------

_PIPELINES = ["run_matched_comparison.py", "run_gnn_experiment.py"]
_FORBIDDEN = [
    r"\brandom_student_split\s*\(",
    r"\blcpo_split\s*\(",
    r"\bsample_lcpo_val\s*\(",
    r"default_rng\s*\(\s*fold_idx",
]


@pytest.mark.parametrize("fname", _PIPELINES)
def test_pipeline_does_not_generate_its_own_masks(fname):
    code = (SRC / fname).read_text()
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    for pat in _FORBIDDEN:
        assert re.search(pat, code) is None, f"{fname} still generates masks: {pat}"
    assert "split_loader" in code, f"{fname} does not import split_loader"


def test_only_save_graph_splits_generates_the_graph_splits():
    code = (SRC / "save_graph_splits.py").read_text()
    for name in ("random_student_split", "lcpo_split", "sample_lcpo_val"):
        assert name in code
