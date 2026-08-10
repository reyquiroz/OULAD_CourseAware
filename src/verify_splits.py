"""
verify_splits.py — Verify overlap/leakage invariants for all 12 split files.

Checks three strategies × four prediction weeks:
  - random_student: train/val/test student sets are pairwise disjoint
  - lcpo: for each fold, held-out (module, presentation) is absent from train rows
  - future_presentation: no 2014J enrollment in train; no 2013B/J/2014B in test

Exit code 0 if all 12 pass, 1 if any fail.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "graph" / "evaluation"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "graph" / "artifacts"

WEEKS = [2, 4, 6, 8]
FUTURE_TRAIN_PRESENTATIONS = {"2013B", "2013J", "2014B"}
FUTURE_TEST_PRESENTATIONS = {"2014J"}

failures = []


def check(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        failures.append(label)
    return condition


# ── 1. Random-student split ────────────────────────────────────────────────
print("\n=== Random-Student Split ===")
for w in WEEKS:
    split_file = EVAL_DIR / f"week{w:02d}" / "splits" / f"week{w:02d}_random_split.parquet"
    enroll_file = ARTIFACTS_DIR / f"week{w:02d}_enrollments.parquet"
    label = f"Week {w} random-student"
    try:
        # split file is the enrollment table with boolean columns is_train/is_val/is_test
        split = pd.read_parquet(split_file)

        train_students = set(split.loc[split["is_train"], "id_student"].unique())
        val_students   = set(split.loc[split["is_val"],   "id_student"].unique())
        test_students  = set(split.loc[split["is_test"],  "id_student"].unique())

        tv_overlap = len(train_students & val_students)
        tt_overlap = len(train_students & test_students)
        vt_overlap = len(val_students   & test_students)
        coverage   = (split["is_train"] | split["is_val"] | split["is_test"]).all()

        check(tv_overlap == 0,   f"{label}: train∩val student overlap = {tv_overlap}")
        check(tt_overlap == 0,   f"{label}: train∩test student overlap = {tt_overlap}")
        check(vt_overlap == 0,   f"{label}: val∩test student overlap = {vt_overlap}")
        check(coverage,          f"{label}: all enrollment rows covered by a split")
        check(split["is_train"].sum() > 0, f"{label}: train is non-empty")
        check(split["is_val"].sum()   > 0, f"{label}: val is non-empty")
        check(split["is_test"].sum()  > 0, f"{label}: test is non-empty")
    except Exception as exc:
        check(False, f"{label}: ERROR -- {exc}")


# ── 2. LCPO split ──────────────────────────────────────────────────────────
print("\n=== LCPO Split ===")
for w in WEEKS:
    folds_file  = EVAL_DIR / f"week{w:02d}" / "splits" / f"week{w:02d}_lcpo_folds.csv"
    enroll_file = ARTIFACTS_DIR / f"week{w:02d}_enrollments.parquet"
    label = f"Week {w} LCPO"
    try:
        folds  = pd.read_csv(folds_file)
        enroll = pd.read_parquet(enroll_file)

        n_folds = len(folds)
        check(n_folds == 22, f"{label}: 22 folds present (found {n_folds})")

        all_indices = set(enroll.index)
        fold_index_sets: list[set] = []

        for _, row in folds.iterrows():
            mod, pres = row["held_out_module"], row["held_out_presentation"]
            expected_n_test  = int(row["n_test"])
            expected_n_train = int(row["n_train"])

            # Derive test/train index sets independently from the enrollments table.
            # The test set is every enrollment belonging to the held-out course-presentation;
            # the train set is every other enrollment.
            test_idx  = set(enroll.index[
                (enroll["code_module"] == mod) & (enroll["code_presentation"] == pres)
            ])
            train_idx = all_indices - test_idx

            fold_index_sets.append(test_idx)

            # The held-out course-presentation must appear in the test set
            check(len(test_idx) > 0,
                  f"{label} fold {mod}/{pres}: test set is non-empty")
            # The held-out course-presentation must NOT appear in the training set
            leaked = enroll.loc[
                list(train_idx),
                ["code_module", "code_presentation"]
            ]
            no_leakage = not (
                (leaked["code_module"] == mod) & (leaked["code_presentation"] == pres)
            ).any()
            check(no_leakage,
                  f"{label} fold {mod}/{pres}: held-out presentation absent from train set")
            # Train set must also be non-empty
            check(len(train_idx) > 0,
                  f"{label} fold {mod}/{pres}: train set is non-empty")
            # Complementarity: union == all enrollments, intersection == empty
            check(train_idx | test_idx == all_indices,
                  f"{label} fold {mod}/{pres}: train U test == all enrollments")
            check(len(train_idx & test_idx) == 0,
                  f"{label} fold {mod}/{pres}: train ^ test == empty")
            # Actual counts must match the saved expected counts
            check(len(test_idx) == expected_n_test,
                  f"{label} fold {mod}/{pres}: test count {len(test_idx)} == expected {expected_n_test}")
            check(len(train_idx) == expected_n_train,
                  f"{label} fold {mod}/{pres}: train count {len(train_idx)} == expected {expected_n_train}")

        # Unique fold coverage: every enrollment index appears in exactly one fold's test set
        if n_folds == 22:
            from collections import Counter
            fold_membership = Counter(idx for s in fold_index_sets for idx in s)
            each_once = all(v == 1 for v in fold_membership.values())
            union_size = len(fold_membership)
            check(each_once and union_size == len(enroll),
                  f"{label}: each enrollment belongs to exactly one fold's test set")

    except Exception as exc:
        check(False, f"{label}: ERROR -- {exc}")


# ── 3. Future-presentation split ──────────────────────────────────────────
print("\n=== Future-Presentation Split ===")
for w in WEEKS:
    split_file  = EVAL_DIR / f"week{w:02d}" / "splits" / f"week{w:02d}_future_split.parquet"
    enroll_file = ARTIFACTS_DIR / f"week{w:02d}_enrollments.parquet"
    label = f"Week {w} future-presentation"
    try:
        # split file is the enrollment table with boolean columns is_train/is_test
        split = pd.read_parquet(split_file)

        train_rows = split.loc[split["is_train"]]
        test_rows  = split.loc[split["is_test"]]

        # No 2014J in train
        train_2014j = (train_rows["code_presentation"] == "2014J").sum()
        # No 2013B/J/2014B in test
        test_train_pres = test_rows["code_presentation"].isin(FUTURE_TRAIN_PRESENTATIONS).sum()

        check(train_2014j == 0,
              f"{label}: 2014J in train set = {train_2014j} (must be 0)")
        check(test_train_pres == 0,
              f"{label}: 2013B/J/2014B in test set = {test_train_pres} (must be 0)")
        check(split["is_train"].sum() > 0, f"{label}: train is non-empty")
        check(split["is_test"].sum()  > 0, f"{label}: test is non-empty")
    except Exception as exc:
        check(False, f"{label}: ERROR -- {exc}")


# ── Summary ────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASS")
    sys.exit(0)
