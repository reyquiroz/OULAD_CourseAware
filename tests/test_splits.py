"""
Tests for random_student_split and lcpo_split in src/oulad_data.py.

Run from the project root:
    source oulad_env/bin/activate
    pytest tests/test_splits.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from oulad_data import lcpo_split, random_student_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def enrollments():
    """Synthetic enrollment table with 4 course-presentations and 100 students."""
    rng = np.random.default_rng(0)
    courses = [
        ("AAA", "2013J"),
        ("AAA", "2014J"),
        ("BBB", "2013J"),
        ("BBB", "2014J"),
    ]
    rows = []
    student_id = 1
    for module, pres in courses:
        for _ in range(25):  # 25 students per course-presentation
            rows.append(
                {
                    "id_student": student_id,
                    "code_module": module,
                    "code_presentation": pres,
                    "final_result": rng.choice(["Pass", "Fail"]),
                    "target": int(rng.integers(0, 2)),
                }
            )
            student_id += 1
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# random_student_split tests
# ---------------------------------------------------------------------------

class TestRandomStudentSplit:
    def test_no_overlap_train_test(self, enrollments):
        train, val, test = random_student_split(enrollments, seed=42)
        train_students = set(enrollments.loc[train, "id_student"].unique())
        test_students = set(enrollments.loc[test, "id_student"].unique())
        assert train_students.isdisjoint(test_students), (
            "Students found in both train and test"
        )

    def test_no_overlap_train_val(self, enrollments):
        train, val, test = random_student_split(enrollments, seed=42)
        train_students = set(enrollments.loc[train, "id_student"].unique())
        val_students = set(enrollments.loc[val, "id_student"].unique())
        assert train_students.isdisjoint(val_students)

    def test_no_overlap_val_test(self, enrollments):
        train, val, test = random_student_split(enrollments, seed=42)
        val_students = set(enrollments.loc[val, "id_student"].unique())
        test_students = set(enrollments.loc[test, "id_student"].unique())
        assert val_students.isdisjoint(test_students)

    def test_all_splits_nonempty(self, enrollments):
        train, val, test = random_student_split(enrollments, seed=42)
        assert train.sum() > 0
        assert val.sum() > 0
        assert test.sum() > 0

    def test_masks_cover_all_rows(self, enrollments):
        train, val, test = random_student_split(enrollments, seed=42)
        assert (train | val | test).all(), "Some rows are in no split"
        assert not (train & val).any(), "Overlap between train and val masks"
        assert not (train & test).any(), "Overlap between train and test masks"
        assert not (val & test).any(), "Overlap between val and test masks"

    def test_reproducibility(self, enrollments):
        t1, v1, te1 = random_student_split(enrollments, seed=42)
        t2, v2, te2 = random_student_split(enrollments, seed=42)
        assert (t1 == t2).all()
        assert (v1 == v2).all()
        assert (te1 == te2).all()

    def test_different_seed_different_split(self, enrollments):
        _, _, te1 = random_student_split(enrollments, seed=42)
        _, _, te2 = random_student_split(enrollments, seed=99)
        # Different seeds should (almost certainly) produce different test sets
        assert not (te1 == te2).all()

    def test_raises_when_too_few_students(self):
        tiny = pd.DataFrame({"id_student": [1, 2], "code_module": ["A", "A"],
                             "code_presentation": ["J", "J"]})
        with pytest.raises(ValueError, match="not enough unique students"):
            random_student_split(tiny, val_frac=0.5, test_frac=0.5)


# ---------------------------------------------------------------------------
# lcpo_split tests
# ---------------------------------------------------------------------------

class TestLcpoSplit:
    def test_test_set_is_held_out_presentation(self, enrollments):
        train, test = lcpo_split(enrollments, "AAA", "2013J")
        held = enrollments[test]
        assert (held["code_module"] == "AAA").all()
        assert (held["code_presentation"] == "2013J").all()

    def test_train_excludes_held_out(self, enrollments):
        train, test = lcpo_split(enrollments, "AAA", "2013J")
        train_rows = enrollments[train]
        in_train = (
            (train_rows["code_module"] == "AAA")
            & (train_rows["code_presentation"] == "2013J")
        )
        assert not in_train.any(), (
            "Held-out course-presentation found in training set"
        )

    def test_masks_are_complement(self, enrollments):
        train, test = lcpo_split(enrollments, "BBB", "2014J")
        assert (train | test).all()
        assert not (train & test).any()

    def test_all_presentations_give_nonempty_splits(self, enrollments):
        presentations = (
            enrollments[["code_module", "code_presentation"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        for row in presentations:
            train, test = lcpo_split(enrollments, row.code_module,
                                     row.code_presentation)
            assert train.sum() > 0, (
                f"Empty train for {row.code_module}/{row.code_presentation}"
            )
            assert test.sum() > 0, (
                f"Empty test for {row.code_module}/{row.code_presentation}"
            )

    def test_raises_for_unknown_presentation(self, enrollments):
        with pytest.raises(ValueError, match="no enrollments found"):
            lcpo_split(enrollments, "ZZZ", "9999X")
