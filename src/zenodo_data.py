"""
Zenodo dataset utilities — KU Leuven Blackboard LMS data (Tiukhova et al., 2024).
Zenodo record: https://zenodo.org/records/17087849

Public API (mirrors oulad_data.py)
-----------------------------------
load_zenodo_tables()          Load all raw Zenodo tables from data/Zenodo/dataset.zip
                              and course_info.json.  Returns a dict of DataFrames with
                              OULAD-compatible column names where possible.
filter_window_zenodo()        Filter log_activity to records available by a day-offset
                              cutoff, computed from each course's start date.
build_features_zenodo()       Build one VLE-feature row per enrollment (vle_total,
                              vle_mean, vle_std only — assessment features unavailable).
random_student_split()        Imported directly from oulad_data — same function, same API.
lcpo_split()                  Imported directly from oulad_data — same function, same API.

Schema gaps vs. OULAD
----------------------
# GAP: no assessment submission data — submitted edges and contains_assess edges
#      cannot be built; assess_mean, assess_max, assess_count features unavailable.
# GAP: no student demographics — gender, age_band, region, imd_band, disability,
#      num_of_prev_attempts, studied_credits all absent; student nodes carry no features.
# GAP: no module_presentation_length — course duration computed from course_info.json
#      start/finish dates rather than a dedicated field.
# GAP: timestamps are calendar datetimes, not day-offsets — day offsets are computed
#      from course_info.json start dates using (TIMESTAMP.date() − course_start).days.

OULAD column name mapping
--------------------------
  OULAD               | Zenodo
  --------------------|-----------------------------
  id_student          | USER_ID
  code_module         | COURSE_ID  (e.g. "Accountancy")
  code_presentation   | year       (e.g. "1819")
  final_result        | derived from PASSED / SCORE_CATEGORY_FINAL
  target              | 1 if PASSED==0 else 0
  id_site             | CONTENT_ID
  activity_type       | CONTENT_TYPE
  sum_click           | 1 per log row (action count ≈ click count)
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_ZENODO_DIR = _PROJECT_ROOT / "data" / "Zenodo"
_ZIP_PATH = _ZENODO_DIR / "dataset.zip"
_COURSE_INFO_PATH = _ZENODO_DIR / "course_info.json"

# Academic years present in the dataset
_YEARS = ["1819", "1920", "2021"]

# ---------------------------------------------------------------------------
# Re-export split utilities from oulad_data (same API — no duplication)
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(_SRC_DIR))
from oulad_data import random_student_split, lcpo_split  # noqa: F401


# ---------------------------------------------------------------------------
# Stage 1 — Load raw tables
# ---------------------------------------------------------------------------

def load_zenodo_tables(zip_path: Path = None, course_info_path: Path = None) -> Dict[str, object]:
    """Load all raw Zenodo tables and return a dict of DataFrames + metadata.

    Returns a dict with keys:
        course_participation  — enrollment outcomes (all years combined)
        log_activity          — clickstream (all years combined; lazy-loaded)
        course_content        — VLE resource metadata (all years combined)
        df_consumption        — discussion post reads (all years combined)
        df_contribution       — discussion posts created (all years combined)
        course_info           — raw dict from course_info.json
        start_dates           — DataFrame: year, course_id, course_start (date),
                                course_finish (date), duration_days

    Notes
    -----
    log_activity is returned as a *list of (year, DataFrame)* tuples rather than
    a single concatenated DataFrame to avoid loading ~675 MB at once.  Callers
    that need the full log should use iter_log_activity().
    """
    if zip_path is None:
        zip_path = _ZIP_PATH
    if course_info_path is None:
        course_info_path = _COURSE_INFO_PATH

    with open(course_info_path) as f:
        course_info = json.load(f)

    # Build start-date lookup (year × course_id)
    start_rows = []
    for yr, courses in course_info.items():
        for course_id, info in courses.items():
            start = datetime(*info["course_start"]).date()
            finish = datetime(*info["course_finish"]).date()
            start_rows.append({
                "year": yr,
                "course_id": course_id,
                "course_start": start,
                "course_finish": finish,
                "duration_days": (finish - start).days,
            })
    start_dates = pd.DataFrame(start_rows)

    tables: Dict[str, object] = {"course_info": course_info, "start_dates": start_dates}

    with zipfile.ZipFile(zip_path) as zf:
        part_frames, content_frames, cons_frames, contrib_frames = [], [], [], []

        for yr in _YEARS:
            part = pd.read_excel(io.BytesIO(zf.read(f"dataset/{yr}_course_participation.xlsx")))
            part["year"] = yr
            part_frames.append(part)

            cc = pd.read_excel(io.BytesIO(zf.read(f"dataset/{yr}_course_content.xlsx")))
            cc["year"] = yr
            content_frames.append(cc)

            cons = pd.read_excel(io.BytesIO(zf.read(f"dataset/{yr}_df_consumption.xlsx")))
            cons["year"] = yr
            cons_frames.append(cons)

            contrib = pd.read_excel(io.BytesIO(zf.read(f"dataset/{yr}_df_contribution.xlsx")))
            contrib["year"] = yr
            contrib_frames.append(contrib)

    tables["course_participation"] = pd.concat(part_frames, ignore_index=True)
    tables["course_content"] = pd.concat(content_frames, ignore_index=True)
    tables["df_consumption"] = pd.concat(cons_frames, ignore_index=True)
    tables["df_contribution"] = pd.concat(contrib_frames, ignore_index=True)

    return tables


def iter_log_activity(zip_path: Path = None):
    """Yield (year, DataFrame) tuples for log_activity, one year at a time.

    Used to avoid loading all three CSVs (~675 MB uncompressed) simultaneously.
    Each yielded DataFrame has an additional 'year' column.
    """
    if zip_path is None:
        zip_path = _ZIP_PATH
    with zipfile.ZipFile(zip_path) as zf:
        for yr in _YEARS:
            with zf.open(f"dataset/{yr}_log_activity.csv") as f:
                df = pd.read_csv(f, parse_dates=["TIMESTAMP"])
            df["year"] = yr
            yield yr, df


# ---------------------------------------------------------------------------
# Stage 2 — Build OULAD-compatible tables
# ---------------------------------------------------------------------------

def build_oulad_compatible_tables(raw: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    """Convert raw Zenodo tables to DataFrames with OULAD-compatible column names.

    Returns a dict with keys:
        student_info     — one row per enrollment; columns: id_student,
                           code_module, code_presentation, final_result, target
        vle              — content metadata; columns: id_site, code_module,
                           code_presentation, activity_type, year
        courses          — course-presentation metadata; columns: code_module,
                           code_presentation, module_presentation_length
        start_dates      — passed through from load_zenodo_tables()

    # GAP: student_info has no demographics (gender, age_band, region, imd_band,
    #      disability, num_of_prev_attempts, studied_credits) — all absent from
    #      the Zenodo dataset.
    # GAP: no student_vle / studentAssessment / assessments tables — these
    #      are built separately from log_activity in build_log_features().
    """
    part = raw["course_participation"].copy()

    # --- student_info equivalent ---
    # Map PASSED==0 → at-risk (target=1), PASSED==1 → success (target=0)
    # PASSED==0 means "never passed any exam session" — closest to OULAD's
    # Fail+Withdrawn.  See docs/zenodo_dataset_feasibility.md §C2 for details.
    student_info = pd.DataFrame({
        "id_student": part["USER_ID"],
        # Normalise split-section names (e.g. "Global economics 1" → "Global economics")
        # so that enrolled_in edge keys align with vle / courses tables which also normalise.
        "code_module": part["COURSE_ID"].str.replace(r"\s+\d+$", "", regex=True),
        "code_presentation": part["year"],
        "final_result": part["PASSED"].map({1: "Pass", 0: "Fail"}),
        "target": (part["PASSED"] == 0).astype(int),
    })

    # --- vle equivalent (from course_content) ---
    cc = raw["course_content"].copy()
    # CONTENT_ID → id_site, CONTENT_TYPE → activity_type
    # Only include non-hidden content items that belong to a course
    vle = pd.DataFrame({
        "id_site": cc["CONTENT_ID"],
        "code_module": cc["COURSE_ID"],
        "code_presentation": cc["year"],
        "activity_type": cc["CONTENT_TYPE"].fillna("Unknown"),
    })
    # In 2021 Global economics was split into two LMS pages (Global economics 1/2);
    # normalise to the base course name for the course-presentation key.
    vle["code_module"] = vle["code_module"].str.replace(r"\s+\d+$", "", regex=True)

    # --- courses equivalent ---
    sd = raw["start_dates"].copy()
    courses = pd.DataFrame({
        "code_module": sd["course_id"].str.replace(r"\s+\d+$", "", regex=True),
        "code_presentation": sd["year"],
        "module_presentation_length": sd["duration_days"],
    }).drop_duplicates(["code_module", "code_presentation"])

    return {
        "student_info": student_info,
        "vle": vle,
        "courses": courses,
        "start_dates": raw["start_dates"],
    }


# ---------------------------------------------------------------------------
# Stage 3 — Filter log_activity to a prediction window
# ---------------------------------------------------------------------------

def filter_window_zenodo(
    log_df: pd.DataFrame,
    year: str,
    course_id: str,
    window_days: int,
    start_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Return log rows for (year, course_id) with day_offset <= window_days.

    Parameters
    ----------
    log_df       : DataFrame from iter_log_activity (single year).
    year         : Academic year string, e.g. "1819".
    course_id    : Base course name, e.g. "Accountancy".
    window_days  : Prediction cutoff in days from course start (inclusive).
    start_dates  : DataFrame from load_zenodo_tables()["start_dates"].

    Returns
    -------
    DataFrame with added columns:
        day_offset  — integer days from course start (0 = first day)
        id_student  — alias of USER_ID
        id_site     — alias of CONTENT_ID
        code_module — normalised course name (no trailing digit)
        code_presentation — year

    # GAP: no dual guard for submission-date (no assessment submissions exist).
    #      Only a single temporal guard is applied: day_offset <= window_days.
    """
    # Look up course start date; handle "Global economics 1/2" → "Global economics"
    base_id = course_id.rstrip(" 12").strip()
    match = start_dates[
        (start_dates["year"] == year) &
        (start_dates["course_id"].str.replace(r"\s+\d+$", "", regex=True) == base_id)
    ]
    if match.empty:
        raise ValueError(f"No start date found for year={year} course_id={course_id}")
    course_start = pd.Timestamp(match.iloc[0]["course_start"])

    # Filter to this (year, course) — handle split-page naming (1920 GE has two IDs)
    mask = log_df["COURSE_ID"].str.replace(r"\s+\d+$", "", regex=True) == base_id
    mask &= log_df["year"] == year
    df = log_df[mask].copy()

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["day_offset"] = (df["TIMESTAMP"] - course_start).dt.days

    # Apply window cutoff (inclusive)
    df = df[df["day_offset"] >= 0]
    df = df[df["day_offset"] <= window_days].copy()

    # Add OULAD-compatible column aliases
    df["id_student"] = df["USER_ID"]
    df["id_site"] = df["CONTENT_ID"]
    df["code_module"] = base_id
    df["code_presentation"] = year
    # sum_click equivalent: each log row = 1 action
    df["sum_click"] = 1

    return df


# ---------------------------------------------------------------------------
# Stage 4 — Build VLE features (OULAD-compatible subset)
# ---------------------------------------------------------------------------

def build_features_zenodo(
    log_filtered: pd.DataFrame,
    student_info: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per enrollment with VLE-only features.

    Features produced (3 of the 9 OULAD LightGBM features):
        vle_total  — total action count within the window
        vle_mean   — mean action count per content item
        vle_std    — std of action counts per content item

    # GAP: assess_mean, assess_max, assess_count, num_of_prev_attempts,
    #      studied_credits, age_band all absent (see docs/zenodo_dataset_feasibility.md).

    Parameters
    ----------
    log_filtered : DataFrame from filter_window_zenodo() for a single (year, course).
    student_info : OULAD-compatible student_info DataFrame from build_oulad_compatible_tables().

    Returns
    -------
    DataFrame with one row per enrollment; columns include id_student,
    code_module, code_presentation, target, vle_total, vle_mean, vle_std.
    """
    vle = (
        log_filtered
        .groupby(["id_student", "code_module", "code_presentation"])
        .agg(sum_click=("sum_click", "sum"))  # total actions in window
    )

    # Compute per-content-item stats for mean/std
    per_item = (
        log_filtered
        .groupby(["id_student", "id_site", "code_module", "code_presentation"])["sum_click"]
        .sum()
        .reset_index()
    )
    item_stats = (
        per_item
        .groupby(["id_student", "code_module", "code_presentation"])["sum_click"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "vle_mean", "std": "vle_std"})
    )

    vle = vle.join(item_stats).reset_index()
    vle = vle.rename(columns={"sum_click": "vle_total"})

    df = student_info.merge(
        vle, how="left", on=["id_student", "code_module", "code_presentation"]
    )
    df["vle_total"] = df["vle_total"].fillna(0)
    df["vle_mean"] = df["vle_mean"].fillna(0)
    df["vle_std"] = df["vle_std"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Stage 5 — Build enrollment supervision table
# ---------------------------------------------------------------------------

def build_enrollment_supervision_zenodo(student_info: pd.DataFrame) -> pd.DataFrame:
    """Return the enrollment supervision table (one row per enrollment).

    Columns: id_student, code_module, code_presentation, final_result, target
    """
    return (
        student_info[["id_student", "code_module", "code_presentation",
                       "final_result", "target"]]
        .drop_duplicates(["id_student", "code_module", "code_presentation"])
        .copy()
        .reset_index(drop=True)
    )
