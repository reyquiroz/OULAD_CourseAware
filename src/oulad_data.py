"""
Shared OULAD data utilities for baseline and graph pipelines.

Public API
----------
load_oulad_data()         Load core OULAD tables; attach binary risk label.
load_supplementary_tables() Load vle, courses, studentRegistration tables.
filter_window()           Filter VLE interactions and assessment submissions
                          to records available by a prediction cutoff (dual guard:
                          due_date <= window AND date_submitted <= window).
build_features()          Build one feature row per enrollment.
sanitize_feature_names()  Sanitize column names for XGBoost / LightGBM.
evaluate_metrics()        Compute standard binary-classification metrics.
create_datasets()         Convenience wrapper: build per-week feature tables.
random_student_split()    Student-level train/val/test masks (no student overlap).
lcpo_split()              Leave-Course-Presentation-Out train/test masks.

Split utilities — graph pipeline usage
---------------------------------------
``random_student_split`` and ``lcpo_split`` operate on any DataFrame that has
an ``id_student`` column (and ``code_module`` / ``code_presentation`` for LCPO).
They are the canonical split utilities for **both** the tabular baseline pipeline
and the GNN training pipeline.

Typical graph usage::

    from graph_pipeline import (
        load_raw_tables, apply_window_cutoff,
        build_node_tables, build_edge_tables,
        build_enrollment_supervision,
    )
    from oulad_data import random_student_split, lcpo_split

    # Build the Week 8 graph
    raw      = load_raw_tables()
    filtered = apply_window_cutoff(raw, window_days=56)
    nodes    = build_node_tables(filtered)
    edges    = build_edge_tables(filtered, nodes)
    enrollments = build_enrollment_supervision(filtered)
    # enrollments has columns: id_student, code_module, code_presentation,
    #                          final_result, target

    # Random student split (70 / 10 / 20)
    train_mask, val_mask, test_mask = random_student_split(
        enrollments, val_frac=0.1, test_frac=0.2, seed=42
    )
    train_enroll = enrollments[train_mask]
    val_enroll   = enrollments[val_mask]
    test_enroll  = enrollments[test_mask]

    # LCPO split — hold out BBB/2013J
    train_mask, test_mask = lcpo_split(enrollments, "BBB", "2013J")

The masks index directly into the enrollment supervision table.  Pass them to
your GNN training loop to select the relevant node indices.
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


def filter_window(vle, assess, assessments, window, submission_date_guard: bool = True):
    """Filter VLE and assessment submissions to records available by *window*.

    Two independent guards are applied to assessment submissions (Strategy B —
    strictly leakage-free):

    1. Due-date guard  — ``assessments.date <= window``
       An assessment whose due date falls after the prediction cutoff is not yet
       "available" to a student, so its existence is unknown.

    2. Submission-date guard — ``studentAssessment.date_submitted <= window``
       Even if an assessment was due within the window, a score submitted *after*
       the cutoff would not be observable at prediction time.  In OULAD, 28.8% of
       all submissions carry a ``date_submitted`` greater than their due date
       (late extensions, grace periods).  Excluding them prevents a subtle form of
       future leakage.

    Note: ``date_submitted`` has no null values in OULAD (all 173,912 submission
    rows are populated), so the second guard never silently drops valid rows.

    Empirical impact (LightGBM, 5-fold GroupKFold CV):
        Week 2: AUROC delta  −0.0006  (100 rows dropped,  8.4%)
        Week 4: AUROC delta  −0.0009  (650 rows dropped,  2.9%)
        Week 6: AUROC delta  +0.0004  (538 rows dropped,  1.8%)
        Week 8: AUROC delta  +0.0024  (2332 rows dropped, 4.9%)
    All deltas are within ±1 std of either strategy — negligible performance impact.

    Args:
        vle:                   studentVle DataFrame with a ``date`` column (interaction day).
        assess:                studentAssessment DataFrame with ``id_assessment``,
                               ``date_submitted``, ``score`` columns.
        assessments:           assessments metadata DataFrame with ``id_assessment``,
                               ``date`` (due date) columns.
        window:                Prediction cutoff in days from course start (inclusive).
        submission_date_guard: When ``True`` (default), Guard 2 is applied.
                               Set to ``False`` to replicate the Strategy A
                               (due-date-only) behaviour for comparison purposes.

    Returns:
        Tuple (vle_w, assess_w) — filtered copies of the input DataFrames.
    """
    vle_w = vle[vle["date"] <= window].copy()

    # Attach due date from assessments metadata
    assess_with_dates = assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    # Guard 1: assessment must have been due by the prediction cutoff
    assess_w = assess_with_dates[assess_with_dates["date"] <= window].copy()
    # Guard 2: submission must have occurred by the prediction cutoff (Strategy B)
    if submission_date_guard:
        assess_w = assess_w[assess_w["date_submitted"] <= window].copy()

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


def create_datasets(
    student_info,
    student_vle,
    student_assess,
    assessments,
    weeks=(2, 4, 6, 8),
    submission_date_guard: bool = True,
):
    """Create per-week feature tables.

    Args:
        submission_date_guard: Forwarded to ``filter_window()``.  Set to
            ``False`` to use Strategy A (due-date-only) filtering.
    """
    datasets = {}
    for week in weeks:
        vle_w, assess_w = filter_window(
            student_vle, student_assess, assessments, week * 7,
            submission_date_guard=submission_date_guard,
        )
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

    This function is the canonical split utility for **both** the tabular
    baseline pipeline (used via 5-fold GroupKFold CV in evaluation_pipeline.py)
    and the GNN training pipeline (used directly on the enrollment supervision
    table from build_enrollment_supervision()).

    Args:
        enrollments_df: DataFrame with an ``id_student`` column (one row per
                        enrollment, as produced by build_enrollment_supervision
                        or build_features).
        val_frac:        Fraction of unique students assigned to validation.
        test_frac:       Fraction of unique students assigned to test.
        seed:            Random seed for reproducibility.

    Returns:
        Tuple of three boolean Series (train_mask, val_mask, test_mask)
        aligned to enrollments_df.index.

    Raises:
        ValueError: if any resulting split would be empty.

    Examples:
        Graph pipeline usage::

            from graph_pipeline import build_enrollment_supervision
            from oulad_data import random_student_split

            enrollments = build_enrollment_supervision(filtered)
            train_mask, val_mask, test_mask = random_student_split(
                enrollments, val_frac=0.1, test_frac=0.2, seed=42
            )
            # Index enrollment supervision table with masks
            train_enroll = enrollments[train_mask]
            # train_enroll contains only students assigned to train
            # — guaranteed no student overlap with val or test sets
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

    This function is the canonical LCPO split utility for **both** the tabular
    baseline pipeline and the GNN training pipeline.

    Args:
        enrollments_df:       DataFrame with ``id_student``, ``code_module``,
                              and ``code_presentation`` columns (one row per
                              enrollment, as produced by build_enrollment_supervision
                              or build_features).
        held_out_module:      Module code to hold out (e.g. ``"BBB"``).
        held_out_presentation: Presentation code to hold out (e.g. ``"2013J"``).

    Returns:
        Tuple of two boolean Series (train_mask, test_mask) aligned to
        enrollments_df.index.

    Raises:
        ValueError: if either resulting split would be empty.

    Examples:
        Graph pipeline usage (iterate all 22 course-presentations)::

            from graph_pipeline import build_enrollment_supervision
            from oulad_data import lcpo_split

            enrollments = build_enrollment_supervision(filtered)
            course_presentations = (
                enrollments[["code_module", "code_presentation"]]
                .drop_duplicates()
                .sort_values(["code_module", "code_presentation"])
            )
            for _, row in course_presentations.iterrows():
                train_mask, test_mask = lcpo_split(
                    enrollments, row["code_module"], row["code_presentation"]
                )
                train_enroll = enrollments[train_mask]
                test_enroll  = enrollments[test_mask]
                # train on train_enroll node indices, evaluate on test_enroll
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
