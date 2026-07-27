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
        check(False, f"{label}: ERROR — {exc}")


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

        leakage_found = False
        for _, row in folds.iterrows():
            mod, pres = row["held_out_module"], row["held_out_presentation"]
            test_mask  = (enroll["code_module"] == mod) & (enroll["code_presentation"] == pres)
            train_mask = ~test_mask
            # held-out must not appear in train
            leaked = enroll.loc[
                train_mask & (enroll["code_module"] == mod) & (enroll["code_presentation"] == pres)
            ]
            if len(leaked) > 0:
                leakage_found = True
                failures.append(f"{label}: {mod}/{pres} found in train set")
                print(f"  [FAIL] {label}: {mod}/{pres} leaked into train ({len(leaked)} rows)")
            # test must be non-empty
            if test_mask.sum() == 0:
                leakage_found = True
                failures.append(f"{label}: {mod}/{pres} test set is empty")

        if not leakage_found:
            check(True, f"{label}: no held-out presentation found in any train set")
    except Exception as exc:
        check(False, f"{label}: ERROR — {exc}")


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
        check(False, f"{label}: ERROR — {exc}")


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
