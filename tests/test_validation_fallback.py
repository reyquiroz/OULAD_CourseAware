"""
Tests for LCPO validation fallback logic, ensuring robust validation sets
are sampled even when positive cases are extremely scarce.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_gnn_experiment import _sample_lcpo_val


def test_fallback_triggers_when_val_positives_below_threshold():
    """Verify fallback triggers and increases validation size to meet positive count threshold."""
    # Create synthetic enrollment DataFrame: 100 students, each with 1 enrollment.
    # We assign 80 positive targets (at-risk) out of 100.
    # The initial 10% draw of students will pick 10 students, yielding at most 10 positives,
    # which is strictly below the min_val_pos=20 threshold, forcing fallback to trigger.
    np.random.seed(42)
    n_rows = 100
    id_student = np.arange(1000, 1000 + n_rows)
    # 80 positive cases, 20 negative cases
    target = np.array([1] * 80 + [0] * 20)
    np.random.shuffle(target)

    enroll_df = pd.DataFrame({
        "id_student": id_student,
        "target": target
    })

    train_all_mask = np.ones(n_rows, dtype=bool)
    y = enroll_df["target"].to_numpy()

    # Call _sample_lcpo_val with fold_idx=0
    train_mask, val_mask = _sample_lcpo_val(
        enroll_df=enroll_df,
        train_all_mask=train_all_mask,
        y=y,
        fold_idx=0,
        min_val_pos=20
    )

    # Check that fallback successfully returned masks where val has >= 20 positive cases
    val_positives = int(y[val_mask].sum())
    assert val_positives >= 20
    assert len(train_mask) == n_rows
    assert len(val_mask) == n_rows
    # val_mask and train_mask are partition of train_all_mask
    assert np.all(train_mask | val_mask == train_all_mask)
    assert np.all(train_mask & val_mask == 0)


def test_val_and_train_students_are_disjoint_after_fallback():
    """Verify that student IDs in validation and training masks remain disjoint."""
    np.random.seed(123)
    n_rows = 100
    id_student = np.arange(1000, 1000 + n_rows)
    target = np.array([1] * 80 + [0] * 20)
    np.random.shuffle(target)

    enroll_df = pd.DataFrame({
        "id_student": id_student,
        "target": target
    })

    train_all_mask = np.ones(n_rows, dtype=bool)
    y = enroll_df["target"].to_numpy()

    # Trigger sampling with fold_idx=1
    train_mask, val_mask = _sample_lcpo_val(
        enroll_df=enroll_df,
        train_all_mask=train_all_mask,
        y=y,
        fold_idx=1,
        min_val_pos=20
    )

    # Get student IDs for training and validation subsets
    train_student_ids = enroll_df.loc[train_mask, "id_student"].unique()
    val_student_ids = enroll_df.loc[val_mask, "id_student"].unique()

    # Verify that the two sets are completely disjoint
    assert set(train_student_ids).isdisjoint(set(val_student_ids))
    assert len(train_student_ids) + len(val_student_ids) == n_rows
