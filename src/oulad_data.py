"""
OULAD shared data utilities
Centralizes raw-data loading, leakage-safe temporal filtering, feature-name
sanitization, and metric calculation so that both the tabular baseline
(baseline_evaluation.py / lcpo_evaluation.py) and the graph pipeline
(graph_pipeline.py) can import from a single authoritative source instead
of duplicating these functions.

Label convention (consistent with existing baselines):
  1 = at-risk  (Fail / Withdrawn)   — positive class, intervention target
  0 = success  (Pass / Distinction) — negative class
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

from config import DATA_DIR


# ---------------------------------------------------------------------------
# Raw-table loading
# ---------------------------------------------------------------------------

def load_oulad_data(data_dir=None):
    """
    Load core OULAD tables and attach binary risk label to studentInfo.

    Args:
        data_dir: Path to the raw-data directory.  Defaults to config.DATA_DIR.

    Returns:
        Tuple: (student_info, student_vle, student_assess, assessments)
            student_info  – studentInfo.csv with 'target' column added
            student_vle   – studentVle.csv
            student_assess – studentAssessment.csv
            assessments   – assessments.csv
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    student_info = pd.read_csv(data_dir / "studentInfo.csv")
    student_vle = pd.read_csv(data_dir / "studentVle.csv")
    student_assess = pd.read_csv(data_dir / "studentAssessment.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")

    # Binary label: 1 = at-risk (Fail/Withdrawn), 0 = success (Pass/Distinction)
    student_info["target"] = student_info["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    return student_info, student_vle, student_assess, assessments


def load_supplementary_tables(data_dir=None):
    """
    Load VLE metadata, courses, and registration tables used by the graph
    pipeline but not required for the tabular baseline.

    Args:
        data_dir: Path to the raw-data directory.  Defaults to config.DATA_DIR.

    Returns:
        Tuple: (vle, courses, student_registration)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    vle = pd.read_csv(data_dir / "vle.csv")
    courses = pd.read_csv(data_dir / "courses.csv")
    student_registration = pd.read_csv(data_dir / "studentRegistration.csv")

    return vle, courses, student_registration


# ---------------------------------------------------------------------------
# Leakage-safe temporal filtering
# ---------------------------------------------------------------------------

def filter_window(vle, assess, assessments, window):
    """
    Filter VLE interactions and assessment submissions to those available by
    *window* days from the start of the course presentation.

    Assessment availability is determined by assessment **due date** (not
    submission date), matching the baseline rule documented in
    docs/LEAKAGE_PREVENTION.md.

    Args:
        vle:         studentVle DataFrame (must have 'date' column).
        assess:      studentAssessment DataFrame (must have 'id_assessment').
        assessments: assessments metadata DataFrame (must have 'id_assessment',
                     'code_module', 'code_presentation', 'date').
        window:      Maximum day (inclusive) to retain.

    Returns:
        Tuple: (vle_w, assess_w)  — filtered copies; assess_w has due-date
               columns merged in.
    """
    vle_w = vle[vle["date"] <= window].copy()

    assess_with_due = assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    assess_w = assess_with_due[assess_with_due["date"] <= window].copy()

    return vle_w, assess_w


# ---------------------------------------------------------------------------
# Feature-name sanitization
# ---------------------------------------------------------------------------

def sanitize_feature_names(df):
    """
    Replace characters in column names that are illegal for XGBoost / LightGBM
    (square brackets, angle brackets).

    Args:
        df: DataFrame whose column names need sanitizing (mutated in place and
            returned for convenience).

    Returns:
        The same DataFrame with sanitized column names.
    """
    df.columns = (
        df.columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )
    return df


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true, y_pred, y_proba):
    """
    Compute the six canonical evaluation metrics used by both the tabular
    baseline and the graph evaluation pipeline.

    Metrics: AUROC, AUPRC, F1, Precision, Recall, Balanced_Acc

    Args:
        y_true:  Ground-truth binary labels (0 / 1 array-like).
        y_pred:  Hard binary predictions (0 / 1 array-like).
        y_proba: Predicted probabilities for the positive class.

    Returns:
        dict mapping metric name → float value.
    """
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
    }
