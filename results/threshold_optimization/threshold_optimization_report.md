# OULAD Threshold Optimization Report

**Generated**: 2026-06-21 00:39:20

---

## Executive Summary

This report analyzes optimal classification thresholds for at-risk student prediction.
Different thresholds are recommended for different deployment scenarios.

### Dataset Statistics

- **Total Students**: 6,519
- **At-Risk Students**: 3,442 (52.8%)
- **Success Students**: 3,077 (47.2%)

## Optimal Thresholds by Deployment Scenario

### Random Forest

| Scenario | Threshold | Precision | Recall | F1 | Description |
|----------|-----------|-----------|--------|----|--------------|
| Max F1 | 0.450 | 0.803 | 0.791 | 0.797 | Balanced precision and recall |
| High Precision | 0.350 | 0.726 | 0.860 | 0.787 | Minimize false alarms (precision ≥ 0.7) |
| High Recall | 0.400 | 0.766 | 0.830 | 0.797 | Catch most at-risk students (recall ≥ 0.8) |
| Cost Sensitive | 0.250 | 0.650 | 0.932 | 0.766 | Minimize cost (FN cost = 3x FP cost) |

### XGBoost

| Scenario | Threshold | Precision | Recall | F1 | Description |
|----------|-----------|-----------|--------|----|--------------|
| Max F1 | 0.400 | 0.771 | 0.816 | 0.793 | Balanced precision and recall |
| High Precision | 0.300 | 0.711 | 0.869 | 0.782 | Minimize false alarms (precision ≥ 0.7) |
| High Recall | 0.400 | 0.771 | 0.816 | 0.793 | Catch most at-risk students (recall ≥ 0.8) |
| Cost Sensitive | 0.250 | 0.678 | 0.899 | 0.773 | Minimize cost (FN cost = 3x FP cost) |

### LightGBM

| Scenario | Threshold | Precision | Recall | F1 | Description |
|----------|-----------|-----------|--------|----|--------------|
| Max F1 | 0.400 | 0.781 | 0.829 | 0.804 | Balanced precision and recall |
| High Precision | 0.300 | 0.710 | 0.893 | 0.791 | Minimize false alarms (precision ≥ 0.7) |
| High Recall | 0.400 | 0.781 | 0.829 | 0.804 | Catch most at-risk students (recall ≥ 0.8) |
| Cost Sensitive | 0.300 | 0.710 | 0.893 | 0.791 | Minimize cost (FN cost = 3x FP cost) |

## Detailed Analysis - Random Forest

### Max F1

**Threshold**: 0.450

**Performance Metrics**:
- Precision: 0.803
- Recall: 0.791
- F1 Score: 0.797

**Confusion Matrix**:
- True Positives (Correctly identified at-risk): 2723
- False Positives (False alarms): 668
- True Negatives (Correctly identified success): 2409
- False Negatives (Missed at-risk students): 719

**Practical Implications**:
- Students flagged for intervention: 3391 (52.0%)
- At-risk students identified: 2723 out of 3442 (79.1%)
- False alarm rate: 21.7%

**Use Case**: Balanced precision and recall

---

### High Precision

**Threshold**: 0.350

**Performance Metrics**:
- Precision: 0.726
- Recall: 0.860
- F1 Score: 0.787

**Confusion Matrix**:
- True Positives (Correctly identified at-risk): 2961
- False Positives (False alarms): 1120
- True Negatives (Correctly identified success): 1957
- False Negatives (Missed at-risk students): 481

**Practical Implications**:
- Students flagged for intervention: 4081 (62.6%)
- At-risk students identified: 2961 out of 3442 (86.0%)
- False alarm rate: 36.4%

**Use Case**: Minimize false alarms (precision ≥ 0.7)

---

### High Recall

**Threshold**: 0.400

**Performance Metrics**:
- Precision: 0.766
- Recall: 0.830
- F1 Score: 0.797

**Confusion Matrix**:
- True Positives (Correctly identified at-risk): 2856
- False Positives (False alarms): 871
- True Negatives (Correctly identified success): 2206
- False Negatives (Missed at-risk students): 586

**Practical Implications**:
- Students flagged for intervention: 3727 (57.2%)
- At-risk students identified: 2856 out of 3442 (83.0%)
- False alarm rate: 28.3%

**Use Case**: Catch most at-risk students (recall ≥ 0.8)

---

### Cost Sensitive

**Threshold**: 0.250

**Performance Metrics**:
- Precision: 0.650
- Recall: 0.932
- F1 Score: 0.766

**Confusion Matrix**:
- True Positives (Correctly identified at-risk): 3207
- False Positives (False alarms): 1726
- True Negatives (Correctly identified success): 1351
- False Negatives (Missed at-risk students): 235

**Practical Implications**:
- Students flagged for intervention: 4933 (75.7%)
- At-risk students identified: 3207 out of 3442 (93.2%)
- False alarm rate: 56.1%

**Use Case**: Minimize cost (FN cost = 3x FP cost)

---

## Recommendations

### 1. For Early Warning Systems (Week 4-8)
- **Recommended**: High Recall threshold
- **Rationale**: Early in semester, prioritize catching all at-risk students
- **Accept**: Higher false positive rate (can be refined later)

### 2. For Resource-Constrained Institutions
- **Recommended**: Resource-Constrained threshold
- **Rationale**: Limit interventions to available capacity
- **Focus**: Highest-risk students within resource limits

### 3. For Targeted Interventions (Week 12-16)
- **Recommended**: High Precision threshold
- **Rationale**: Later in semester, focus on students most likely to benefit
- **Minimize**: Intervention fatigue from false alarms

### 4. For Balanced Approach
- **Recommended**: Max F1 threshold
- **Rationale**: Balance between catching at-risk students and avoiding false alarms
- **Best for**: General-purpose deployment

### 5. For Cost-Sensitive Scenarios
- **Recommended**: Cost-Sensitive threshold
- **Rationale**: Optimize based on relative costs of errors
- **Customize**: Adjust cost ratio based on institutional priorities

## Implementation Guide

```python
# Example: Using optimal threshold in production
import joblib

# Load trained model
model = joblib.load('best_model.pkl')

# Get probability predictions
y_pred_proba = model.predict_proba(X_new)[:, 1]

# Apply optimal threshold (Max F1 scenario)
optimal_threshold = 0.450
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Flag students for intervention
at_risk_students = student_ids[y_pred == 1]
```

## Monitoring and Adjustment

1. **Track Performance**: Monitor precision and recall in production
2. **Adjust Threshold**: Refine based on actual intervention outcomes
3. **Seasonal Variation**: Consider different thresholds for different semesters
4. **Course-Specific**: May need different thresholds for different courses
5. **Feedback Loop**: Incorporate intervention results to improve model

---

*This report was automatically generated by `src/threshold_optimization.py`*
