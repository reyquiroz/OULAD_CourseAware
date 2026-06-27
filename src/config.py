"""
Configuration file for OULAD baseline analysis
Centralizes paths, hyperparameters, and settings for reproducibility
"""

import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "DATA/raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Result subdirectories
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
LCPO_RESULTS_DIR = RESULTS_DIR / "lcpo"
CROSS_COURSE_RESULTS_DIR = RESULTS_DIR / "cross_course"

# Graph pipeline subdirectories
GRAPH_RESULTS_DIR = RESULTS_DIR / "graph"
GRAPH_ARTIFACTS_DIR = GRAPH_RESULTS_DIR / "artifacts"
GRAPH_VALIDATION_DIR = GRAPH_RESULTS_DIR / "validation"
GRAPH_EVALUATION_DIR = GRAPH_RESULTS_DIR / "evaluation"

# Create directories if they don't exist
for directory in [
    RESULTS_DIR,
    BASELINE_RESULTS_DIR,
    LCPO_RESULTS_DIR,
    CROSS_COURSE_RESULTS_DIR,
    GRAPH_RESULTS_DIR,
    GRAPH_ARTIFACTS_DIR,
    GRAPH_VALIDATION_DIR,
    GRAPH_EVALUATION_DIR,
    MODELS_DIR,
    DOCS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data files
STUDENT_INFO_FILE = DATA_DIR / "studentInfo.csv"
STUDENT_VLE_FILE = DATA_DIR / "studentVle.csv"
STUDENT_ASSESSMENT_FILE = DATA_DIR / "studentAssessment.csv"
ASSESSMENTS_FILE = DATA_DIR / "assessments.csv"
COURSES_FILE = DATA_DIR / "courses.csv"
VLE_FILE = DATA_DIR / "vle.csv"
STUDENT_REGISTRATION_FILE = DATA_DIR / "studentRegistration.csv"

# Random state for reproducibility
RANDOM_STATE = 42

# Model hyperparameters
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "cv_folds": 5,
    "test_size": 0.2,
    # Model-specific parameters
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "random_forest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    "xgboost": {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss"},
    "lightgbm": {"n_estimators": 100, "random_state": 42, "verbose": -1},
}

# Prediction windows (in days)
PREDICTION_WINDOWS = {"week_2": 14, "week_4": 28, "week_6": 42, "week_8": 56}

# Feature groups for ablation studies
FEATURE_GROUPS = {
    "demographics": [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "disability",
    ],
    "vle": ["vle_total", "vle_mean", "vle_std"],
    "assessment": ["assess_mean", "assess_max", "assess_count"],
}

# Label convention (CORRECTED)
# 1 = at-risk (Fail/Withdrawn)
# 0 = success (Pass/Distinction)
LABEL_MAPPING = {"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1}

# Evaluation metrics
METRICS = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Acc"]

# Models to evaluate
MODELS = ["Majority", "LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]


def get_data_path(filename):
    """Get full path to a data file"""
    return DATA_DIR / filename


def get_results_path(filename, subdir="baseline"):
    """Get full path to a results file"""
    if subdir == "baseline":
        return BASELINE_RESULTS_DIR / filename
    elif subdir == "lcpo":
        return LCPO_RESULTS_DIR / filename
    elif subdir == "cross_course":
        return CROSS_COURSE_RESULTS_DIR / filename
    else:
        return RESULTS_DIR / filename


def get_model_path(filename):
    """Get full path to a saved model"""
    return MODELS_DIR / filename


# Made with Bob
