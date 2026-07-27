# Evaluation Splits — Formal Definitions and Verification

This document provides formal definitions for the three evaluation strategies
used in this project, documents the leakage guarantees for each, and records
the computational verification results across all four prediction weeks.

---

## 1. Split Strategies

### 1.1 Random-Student Split

**Supervised unit**: enrollment `(id_student, code_module, code_presentation)`  
**Split key**: `id_student`  
**Boundary rule**: Unique students are randomly assigned to train / val / test
partitions (70% / 10% / 20% of unique students, seed=42). All enrollments for a
given student land in the same partition.  
**Leakage guarantee**: No student appears in more than one partition. A model
cannot observe test-set students' activity during training.  
**Files**: `results/graph/evaluation/week{N}/splits/week{N}_random_split.parquet`
(enrollment table with boolean columns `is_train`, `is_val`, `is_test`)

**Parameters** (from `week08_splits_config.json`):

| Parameter | Value |
|-----------|-------|
| `val_frac` | 0.10 |
| `test_frac` | 0.20 |
| `seed` | 42 |
| Train enrollments (Week 8) | 22,801 |
| Val enrollments (Week 8) | 3,280 |
| Test enrollments (Week 8) | 6,512 |

**Unit tests**: `tests/test_splits.py::TestRandomStudentSplit` (8 tests)
— covers no overlap (train∩val, train∩test, val∩test), all rows covered,
non-empty splits, reproducibility, seed sensitivity.

---

### 1.2 Leave-Course-Presentation-Out (LCPO)

**Supervised unit**: enrollment `(id_student, code_module, code_presentation)`  
**Split key**: `(code_module, code_presentation)` pair  
**Boundary rule**: In each fold, all enrollments in one (module, presentation)
pair form the test set; all remaining 21 pairs form the train set. 22 folds
total (one per course-presentation).  
**Leakage guarantee**: The held-out course-presentation never appears in the
train set for that fold. Tests cross-course generalization.  
**Files**: `results/graph/evaluation/week{N}/splits/week{N}_lcpo_folds.csv`
(fold index, held-out module, held-out presentation, n_train, n_test)

**Fold sizes** (all weeks identical — labels independent of time cutoff):

| Fold | Held-out | Train N | Test N |
|------|----------|---------|--------|
| 0 | AAA/2013J | 32,210 | 383 |
| 1 | AAA/2014J | 32,228 | 365 |
| 2 | BBB/2013B | 30,826 | 1,767 |
| 3 | BBB/2013J | 30,356 | 2,237 |
| 4 | BBB/2014B | 30,980 | 1,613 |
| 5 | BBB/2014J | 30,301 | 2,292 |
| 6 | CCC/2014B | 30,657 | 1,936 |
| 7 | CCC/2014J | 30,095 | 2,498 |
| 8 | DDD/2013B | 31,290 | 1,303 |
| 9 | DDD/2013J | 30,655 | 1,938 |
| 10 | DDD/2014B | 31,365 | 1,228 |
| 11 | DDD/2014J | 30,790 | 1,803 |
| 12 | EEE/2013J | 31,541 | 1,052 |
| 13 | EEE/2014B | 31,899 | 694 |
| 14 | EEE/2014J | 31,405 | 1,188 |
| 15 | FFF/2013B | 30,979 | 1,614 |
| 16 | FFF/2013J | 30,310 | 2,283 |
| 17 | FFF/2014B | 31,093 | 1,500 |
| 18 | FFF/2014J | 30,228 | 2,365 |
| 19 | GGG/2013J | 31,641 | 952 |
| 20 | GGG/2014B | 31,760 | 833 |
| 21 | GGG/2014J | 31,844 | 749 |

**Unit tests**: `tests/test_splits.py::TestLcpoSplit` (5 tests)
— covers held-out set isolation, train exclusion, complement masking, all
presentations non-empty, unknown presentation error.

---

### 1.3 Future-Presentation Split

**Supervised unit**: enrollment `(id_student, code_module, code_presentation)`  
**Split key**: `code_presentation` year  
**Boundary rule**: Train on all enrollments with presentation code in
{2013B, 2013J, 2014B}; test on all enrollments with presentation code 2014J.
This simulates deploying a model trained on earlier cohorts to predict for a
future cohort.  
**Leakage guarantee**: No 2014J enrollment appears in train; no 2013B/J/2014B
enrollment appears in test. At the enrollment level, train and test are fully
disjoint.

> **Note on student-level overlap**: The same student may appear in both the
> train set (via a 2013B/J or 2014B enrollment) and the test set (via a 2014J
> enrollment). This is intentional — the split tests temporal generalization
> across cohorts, not student-level generalization. Student-level overlap does
> not constitute leakage here because the model is evaluated on a different
> enrollment (different course, different time period) for any overlapping student.

**Files**: `results/graph/evaluation/week{N}/splits/week{N}_future_split.parquet`
(enrollment table with boolean columns `is_train`, `is_test`)

**Split sizes** (all weeks identical — splits derive from presentation codes,
independent of prediction window):

| Partition | Presentations | Enrollments |
|-----------|---------------|-------------|
| Train | 2013B, 2013J, 2014B | 21,333 |
| Test | 2014J | 11,260 |

---

## 2. Computational Verification

All 12 split files (3 strategies × 4 weeks) were verified by
`src/verify_splits.py`. The script checks:

- **Random-student**: train∩val, train∩test, val∩test student overlap = 0;
  all rows covered; all partitions non-empty.
- **LCPO**: 22 folds present per week; held-out (module, presentation) absent
  from the corresponding train rows for every fold.
- **Future-presentation**: no 2014J enrollment in train; no 2013B/J/2014B
  enrollment in test; both partitions non-empty.

### Verification result

```
=== Random-Student Split ===
  [PASS] Week 2 random-student: train∩val student overlap = 0
  [PASS] Week 2 random-student: train∩test student overlap = 0
  [PASS] Week 2 random-student: val∩test student overlap = 0
  [PASS] Week 2 random-student: all enrollment rows covered by a split
  [PASS] Week 2 random-student: train is non-empty
  [PASS] Week 2 random-student: val is non-empty
  [PASS] Week 2 random-student: test is non-empty
  ... (identical PASS for Weeks 4, 6, 8)

=== LCPO Split ===
  [PASS] Week 2 LCPO: 22 folds present (found 22)
  [PASS] Week 2 LCPO: no held-out presentation found in any train set
  ... (identical PASS for Weeks 4, 6, 8)

=== Future-Presentation Split ===
  [PASS] Week 2 future-presentation: 2014J in train set = 0 (must be 0)
  [PASS] Week 2 future-presentation: 2013B/J/2014B in test set = 0 (must be 0)
  [PASS] Week 2 future-presentation: train is non-empty
  [PASS] Week 2 future-presentation: test is non-empty
  ... (identical PASS for Weeks 4, 6, 8)

RESULT: ALL CHECKS PASS
```

**56 / 56 checks passed** across all strategies and weeks.

To re-run verification:

```bash
source oulad_env/bin/activate
python src/verify_splits.py
```

---

## 3. Cross-Reference: Implementation and Tests

| Split strategy | Implementation | Unit tests | Verification script |
|----------------|---------------|-----------|---------------------|
| Random-student | `src/oulad_data.py::random_student_split()` | `tests/test_splits.py::TestRandomStudentSplit` | `src/verify_splits.py` |
| LCPO | `src/oulad_data.py::lcpo_split()` | `tests/test_splits.py::TestLcpoSplit` | `src/verify_splits.py` |
| Future-presentation | `src/save_graph_splits.py` | — | `src/verify_splits.py` |

For the full split parameter configuration see
`results/graph/evaluation/week{N}/splits/week{N}_splits_config.json`.

For a high-level description of each strategy see `docs/EVALUATION_SPLITS.md`.
