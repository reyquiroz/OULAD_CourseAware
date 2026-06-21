# Evaluation Split Strategies for OULAD

## Overview

This document describes the three evaluation split strategies used to assess model performance and generalization in the OULAD student success prediction task. Each strategy tests different aspects of model robustness and real-world applicability.

## Label Convention

**Important**: All evaluations use the corrected label convention:
- **1 = at-risk** (Fail/Withdrawn) - positive class, students needing intervention
- **0 = success** (Pass/Distinction) - negative class, students on track

Metrics (precision, recall, F1, AUPRC) refer to identifying at-risk students.

## Split Strategy 1: Random Student-Course Split

### Description

Standard stratified k-fold cross-validation where student-course pairs are randomly split into train and test sets.

### Implementation

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

### Characteristics

- **Split Unit**: Individual student-course enrollments
- **Stratification**: Maintains class balance (at-risk vs success) in each fold
- **Randomization**: Shuffled with fixed random seed (42) for reproducibility
- **Folds**: 5-fold cross-validation

### What It Tests

- **Within-distribution performance**: How well models perform when train and test data come from the same distribution
- **Statistical reliability**: Cross-validation provides confidence intervals (mean ± std)
- **Baseline performance**: Establishes upper bound on expected performance

### Advantages

✓ Standard evaluation approach, easy to compare with literature  
✓ Maximizes training data usage  
✓ Provides robust performance estimates with confidence intervals  
✓ Tests model's ability to generalize to unseen students in same courses

### Limitations

✗ May overestimate performance in deployment (train and test from same courses)  
✗ Doesn't test cross-course generalization  
✗ Doesn't test temporal generalization

### Results Location

- Detailed results: `results/baseline/detailed_results_all_features.csv`
- Summary table: `results/baseline/baseline_results_table.csv`

### Expected Performance

Based on Week 8, All Features:
- AUROC: 0.83-0.84
- F1: 0.78-0.79
- Recall: 0.84-0.85

## Split Strategy 2: Leave-Course-Presentation-Out (LCPO)

### Description

Cross-validation where each unique course-presentation combination is held out as a test set, and the model is trained on all other course-presentations.

### Implementation

```python
# Get unique course presentations
course_presentations = df[["code_module", "code_presentation"]].drop_duplicates()

for _, cp_row in course_presentations.iterrows():
    module = cp_row["code_module"]
    presentation = cp_row["code_presentation"]
    
    # Split data
    test_mask = (df["code_module"] == module) & (df["code_presentation"] == presentation)
    train_mask = ~test_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    # Train and evaluate
```

### Characteristics

- **Split Unit**: Course-presentation combinations (e.g., AAA-2013B, BBB-2014J)
- **Number of Folds**: 22 (one for each course presentation in OULAD)
- **Train Size**: ~21/22 of data per fold
- **Test Size**: ~1/22 of data per fold

### What It Tests

- **Cross-course generalization**: Can models trained on some courses predict outcomes in completely unseen courses?
- **Course-specific effects**: How much does performance vary across different courses?
- **Robustness**: Are models overfitting to specific course characteristics?

### Advantages

✓ Tests realistic deployment scenario (new course offerings)  
✓ Identifies course-specific challenges  
✓ Reveals generalization gaps  
✓ Provides course-level performance analysis

### Limitations

✗ Some test sets may be small (< 100 students)  
✗ High variance in performance across courses  
✗ Computationally expensive (22 train-test iterations)

### Course-Presentation Combinations

OULAD contains 22 unique course-presentation pairs:

| Module | Presentations | Total |
|--------|---------------|-------|
| AAA | 2013B, 2013J, 2014B, 2014J | 4 |
| BBB | 2013B, 2013J, 2014B, 2014J | 4 |
| CCC | 2014B, 2014J | 2 |
| DDD | 2013B, 2013J, 2014B, 2014J | 4 |
| EEE | 2013J, 2014B, 2014J | 3 |
| FFF | 2013B, 2013J, 2014B, 2014J | 4 |
| GGG | 2013J, 2014J | 2 |

### Results Location

- Detailed results: `results/lcpo/lcpo_results_detailed.csv`
- Comparison with random split: `results/lcpo/random_vs_lcpo_comparison.csv`
- Course-level analysis: `results/lcpo/course_level_performance.csv`

### Expected Performance

Based on Week 8, All Features:
- AUROC: 0.80-0.81 (3-4% drop from random split)
- F1: 0.75-0.76
- High variance: ±0.08-0.09 (indicates course-specific effects)

### Course Difficulty Analysis

**Challenging Courses** (AUROC < 0.70):
- GGG courses: Lower performance, may need course-specific models

**High-Performing Courses** (AUROC > 0.85):
- DDD, FFF, EEE courses: Excellent generalization

## Split Strategy 3: Future-Presentation Split

### Description

Temporal split where models are trained on earlier course presentations and tested on later presentations, simulating deployment to future course offerings.

### Implementation

```python
# Define temporal split
train_presentations = ["2013B", "2013J", "2014B"]
test_presentations = ["2014J"]

# Split by presentation year/semester
train_mask = df["code_presentation"].isin(train_presentations)
test_mask = df["code_presentation"].isin(test_presentations)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
```

### Characteristics

- **Split Unit**: Presentation time period
- **Train**: Earlier presentations (2013B, 2013J, 2014B)
- **Test**: Later presentations (2014J)
- **Temporal Order**: Strictly enforced (no future data in training)

### What It Tests

- **Temporal generalization**: Can models trained on past data predict future outcomes?
- **Concept drift**: Do student behaviors or course characteristics change over time?
- **Deployment realism**: Most realistic evaluation for production deployment

### Advantages

✓ Most realistic deployment scenario  
✓ Tests temporal stability  
✓ Identifies concept drift  
✓ Single train-test split (fast evaluation)

### Limitations

✗ Limited test data (only 2014J presentations)  
✗ May not have enough temporal separation to detect drift  
✗ Imbalanced train/test sizes

### Presentation Timeline

```
2013B (Feb 2013) ─┐
2013J (Oct 2013) ─┤ TRAIN
2014B (Feb 2014) ─┘
                   │
2014J (Oct 2014) ──┘ TEST
```

### Results Location

- Results: `results/cross_course/future_presentation_results.csv`
- Temporal analysis: `results/cross_course/temporal_generalization_analysis.csv`

### Expected Performance

Hypothesis: Similar to LCPO performance (0.80-0.81 AUROC) if no significant concept drift.

## Comparison of Split Strategies

| Aspect | Random Split | LCPO | Future-Presentation |
|--------|-------------|------|---------------------|
| **Realism** | Low | Medium | High |
| **Generalization Test** | Within-distribution | Cross-course | Temporal |
| **Performance** | Highest | Medium | Medium-Low |
| **Variance** | Low | High | Medium |
| **Computation** | Fast | Slow | Fast |
| **Use Case** | Baseline | New courses | Future offerings |

## Preprocessing Steps (Common to All Splits)

### 1. Data Loading

```python
from config import DATA_DIR

student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
student_assess = pd.read_csv(DATA_DIR / "studentAssessment.csv")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
```

### 2. Target Creation

```python
# CORRECTED label convention
student_info["target"] = student_info["final_result"].apply(
    lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
)
```

### 3. Temporal Filtering

```python
# For each prediction window (weeks 2, 4, 6, 8)
window_days = week * 7
vle_filtered = student_vle[student_vle["date"] <= window_days]
assess_filtered = student_assess[student_assess["date"] <= window_days]
```

### 4. Feature Engineering

```python
# VLE features
vle_features = vle_filtered.groupby(["id_student", "code_module", "code_presentation"]).agg({
    "sum_click": ["sum", "mean", "std"]
})

# Assessment features
assess_features = assess_filtered.groupby(["id_student", "code_module", "code_presentation"]).agg({
    "score": ["mean", "max"],
    "date": "count"
})

# Merge with demographics
features = vle_features.merge(assess_features).merge(student_info)
```

### 5. Feature Preparation

```python
# Drop non-feature columns
X = features.drop(columns=[
    "target", "final_result", "id_student", 
    "code_module", "code_presentation"
])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Target
y = features["target"]
```

### 6. Missing Value Handling

```python
# Numerical features: fill with 0 (no activity)
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = X[num_cols].fillna(0)

# Categorical features: fill with "Unknown"
cat_cols = X.select_dtypes(include=["object"]).columns
X[cat_cols] = X[cat_cols].fillna("Unknown")
```

## Evaluation Metrics

All splits use the same metrics:

1. **AUROC** (Area Under ROC Curve): Overall discrimination ability
2. **AUPRC** (Area Under Precision-Recall Curve): Performance on imbalanced data
3. **F1 Score**: Harmonic mean of precision and recall
4. **Precision**: Accuracy of at-risk predictions
5. **Recall**: Coverage of actual at-risk students
6. **Balanced Accuracy**: Average of sensitivity and specificity

## Reproducibility

All splits use fixed random seeds:
- Random split: `random_state=42`
- LCPO: Deterministic (no randomness)
- Future-presentation: Deterministic (temporal order)

## Best Practices

1. **Always use stratification** for random splits to maintain class balance
2. **Check test set size** in LCPO (skip if < 50 samples or single class)
3. **Verify temporal order** in future-presentation split
4. **Report both mean and std** for random split and LCPO
5. **Document split definitions** for reproducibility

## Conclusion

Using multiple evaluation strategies provides a comprehensive assessment of model performance:

- **Random split**: Establishes baseline performance
- **LCPO**: Tests cross-course generalization
- **Future-presentation**: Tests temporal generalization

Together, these splits reveal both the potential and limitations of our models for real-world deployment.