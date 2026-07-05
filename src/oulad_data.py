"""
Shared OULAD data utilities for baseline and graph pipelines.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import DATA_DIR


def load_oulad_data(data_dir=None):
    """Load core OULAD tables and attach binary risk label to student info."""
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    student_info = pd.read_csv(data_dir / "studentInfo.csv")
    student_vle = pd.read_csv(data_dir / "studentVle.csv")
    student_assess = pd.read_csv(data_dir / "studentAssessment.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")

    student_info["target"] = student_info["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    return student_info, student_vle, student_assess, assessments


def load_supplementary_tables(data_dir=None):
    """Load tables used outside the baseline feature pipeline."""
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    vle = pd.read_csv(data_dir / "vle.csv")
    courses = pd.read_csv(data_dir / "courses.csv")
    student_registration = pd.read_csv(data_dir / "studentRegistration.csv")

    return vle, courses, student_registration


def filter_window(vle, assess, assessments, window):
    """Filter VLE and assessment submissions to records available by *window*.

    Assessments are included only if their due date (assessments.date) falls on
    or before *window*.  Using the submission date would leak future behaviour
    (a student could submit after the prediction cutoff).
    """
    vle_w = vle[vle["date"] <= window].copy()

    assess_with_dates = assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_w = assess_with_dates[assess_with_dates["date"] <= window].copy()

    return vle_w, assess_w


def build_features(vle_w, assess_w, student_info):
    """Build one row per student-course enrollment, retaining inactive students."""
    enrollments = student_info.copy()

    vle = vle_w.groupby(["id_student", "code_module", "code_presentation"]).agg(
        {"sum_click": ["sum", "mean", "std"]}
    )
    vle.columns = ["vle_total", "vle_mean", "vle_std"]
    vle = vle.reset_index()

    assess = assess_w.groupby(["id_student", "code_module", "code_presentation"]).agg(
        {"score": ["mean", "max"], "id_assessment": "count"}
    )
    assess.columns = ["assess_mean", "assess_max", "assess_count"]
    assess = assess.reset_index()

    df = enrollments.merge(
        vle, how="left", on=["id_student", "code_module", "code_presentation"]
    )
    df = df.merge(
        assess, how="left", on=["id_student", "code_module", "code_presentation"]
    )

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.columns.difference(num_cols)

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def sanitize_feature_names(df):
    """Sanitize column names for XGBoost compatibility."""
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )
    return df


def evaluate_metrics(y_true, y_pred, y_proba):
    """Compute the standard binary-classification metrics."""
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
    }


def create_datasets(student_info, student_vle, student_assess, assessments, weeks=(2, 4, 6, 8)):
    """Create per-week feature tables."""
    datasets = {}
    for week in weeks:
        vle_w, assess_w = filter_window(student_vle, student_assess, assessments, week * 7)
        datasets[week] = build_features(vle_w, assess_w, student_info)
    return datasets


# ---------------------------------------------------------------------------
# Evaluation split utilities
# ---------------------------------------------------------------------------

def random_student_split(enrollments_df, val_frac=0.1, test_frac=0.2, seed=42):
    """Return boolean train/val/test masks split on unique *students*.

    The same student will not appear in more than one partition.  Splits are
    applied at the student level then broadcast to all enrollment rows for
    that student.

    Args:
        enrollments_df: DataFrame with an ``id_student`` column (one row per
                        enrollment, as produced by build_enrollment_supervision).
        val_frac:        Fraction of unique students assigned to validation.
        test_frac:       Fraction of unique students assigned to test.
        seed:            Random seed for reproducibility.

    Returns:
        Tuple of three boolean Series (train_mask, val_mask, test_mask)
        aligned to enrollments_df.index.

    Raises:
        ValueError: if any resulting split would be empty.
    """
    rng = np.random.default_rng(seed)
    unique_students = np.array(enrollments_df["id_student"].unique())
    rng.shuffle(unique_students)

    n = len(unique_students)
    n_test = max(1, int(np.floor(n * test_frac)))
    n_val = max(1, int(np.floor(n * val_frac)))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"random_student_split: not enough unique students ({n}) to form "
            f"non-empty train/val/test with val_frac={val_frac}, "
            f"test_frac={test_frac}."
        )

    test_students = set(unique_students[:n_test])
    val_students = set(unique_students[n_test: n_test + n_val])
    # remaining → train (implicit; verified below by assertion)

    test_mask = enrollments_df["id_student"].isin(test_students)
    val_mask = enrollments_df["id_student"].isin(val_students)
    train_mask = ~test_mask & ~val_mask

    assert train_mask.sum() > 0, "random_student_split: train set is empty"
    assert val_mask.sum() > 0, "random_student_split: val set is empty"
    assert test_mask.sum() > 0, "random_student_split: test set is empty"
    # Verify no student overlap between train and test
    train_students = set(enrollments_df.loc[train_mask, "id_student"].unique())
    assert train_students.isdisjoint(test_students), (
        "random_student_split: student overlap detected between train and test"
    )

    return train_mask, val_mask, test_mask


def lcpo_split(enrollments_df, held_out_module, held_out_presentation):
    """Return boolean train/test masks for Leave-Course-Presentation-Out.

    The test set is all enrollments where ``code_module == held_out_module``
    **and** ``code_presentation == held_out_presentation``.  The train set is
    the complement.

    Args:
        enrollments_df:       DataFrame with ``id_student``, ``code_module``,
                              and ``code_presentation`` columns.
        held_out_module:      Module code to hold out (e.g. ``"BBB"``).
        held_out_presentation: Presentation code to hold out (e.g. ``"2013J"``).

    Returns:
        Tuple of two boolean Series (train_mask, test_mask) aligned to
        enrollments_df.index.

    Raises:
        ValueError: if either resulting split would be empty.
    """
    test_mask = (
        (enrollments_df["code_module"] == held_out_module)
        & (enrollments_df["code_presentation"] == held_out_presentation)
    )
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        raise ValueError(
            f"lcpo_split: no enrollments found for "
            f"{held_out_module}/{held_out_presentation}."
        )
    if train_mask.sum() == 0:
        raise ValueError(
            f"lcpo_split: train set is empty — only one course-presentation "
            f"exists in enrollments_df."
        )

    return train_mask, test_mask
