# OULAD Student Success Prediction - Baseline Analysis

## Overview
This repository contains a comprehensive baseline analysis for predicting student success using the Open University Learning Analytics Dataset (OULAD). The analysis implements multiple machine learning models with rigorous evaluation strategies to establish strong baselines for early student risk prediction.

## Dataset: OULAD
The Open University Learning Analytics Dataset contains data from 32,593 students across 22 course presentations. 

### Risk Label Definition
- **Success (1)**: Students who Pass or achieve Distinction
- **At-Risk (0)**: Students who Fail or Withdraw
- **Class Distribution**: 15,385 success (47.2%) vs 17,208 at-risk (52.8%)

## Features

### Feature Categories
1. **VLE Activity Features** (Virtual Learning Environment)
   - `vle_total`: Total clicks across all activities
   - `vle_mean`: Average clicks per activity
   - `vle_std`: Standard deviation of clicks

2. **Assessment Features**
   - `assess_mean`: Average assessment score
   - `assess_max`: Maximum assessment score
   - `assess_count`: Number of assessments completed

3. **Demographic Features**
   - Gender, age band, education level
   - Region, IMD band (deprivation index)
   - Number of previous attempts
   - Disability status

### Temporal Windows
Predictions are made at four time points to enable early intervention:
- **Week 2**: Very early prediction (limited data)
- **Week 4**: Early prediction
- **Week 6**: Mid-term prediction
- **Week 8**: Later prediction (more reliable)

### Leakage Prevention
All features are temporally filtered to only include data available up to the prediction window, preventing data leakage from future information.

## Models Evaluated

### Baseline Models
1. **Majority Classifier**: Always predicts the majority class (baseline reference)
2. **Logistic Regression**: Linear model with L2 regularization
3. **Random Forest**: Ensemble of 100 decision trees
4. **XGBoost**: Gradient boosting with 100 estimators
5. **LightGBM**: Fast gradient boosting with 100 estimators

### Feature Subsets
To understand feature importance, we evaluate models on:
- **VLE-only**: Only VLE activity features
- **Assessment-only**: Only assessment performance features
- **VLE+Assessment**: Combined activity and assessment features
- **All features**: Complete feature set including demographics

## Evaluation Metrics

All models are evaluated using 6 comprehensive metrics:
1. **AUROC** (Area Under ROC Curve): Overall discrimination ability
2. **AUPRC** (Area Under Precision-Recall Curve): Performance on imbalanced data
3. **F1 Score**: Harmonic mean of precision and recall
4. **Precision**: Accuracy of positive predictions
5. **Recall**: Coverage of actual positive cases
6. **Balanced Accuracy**: Average of sensitivity and specificity

## Results

### Baseline Results (5-Fold Cross-Validation)

#### Week 8 Performance (All Features)
| Model | AUROC | AUPRC | F1 | Precision | Recall | Balanced Acc |
|-------|-------|-------|-------|-----------|--------|--------------|
| Majority | 0.500±0.000 | 0.527±0.000 | 0.690±0.000 | 0.527±0.000 | 1.000±0.000 | 0.500±0.000 |
| LogisticRegression | 0.772±0.005 | 0.769±0.007 | 0.730±0.005 | 0.694±0.005 | 0.770±0.007 | 0.696±0.006 |
| RandomForest | 0.825±0.004 | 0.806±0.006 | 0.777±0.004 | 0.741±0.006 | 0.816±0.003 | 0.749±0.006 |
| XGBoost | 0.824±0.004 | 0.809±0.005 | 0.775±0.003 | 0.737±0.003 | 0.818±0.004 | 0.746±0.004 |
| **LightGBM** | **0.835±0.005** | **0.823±0.006** | **0.788±0.004** | **0.740±0.005** | **0.842±0.003** | **0.757±0.005** |

### Key Findings

1. **Best Model**: LightGBM achieves the highest performance across all metrics at Week 8
   - AUROC: 0.835 (excellent discrimination)
   - F1: 0.788 (strong balance of precision/recall)
   - Recall: 0.842 (identifies 84% of at-risk students)

2. **Temporal Progression**: Performance improves significantly with more data
   - Week 2 AUROC: 0.714 (LightGBM)
   - Week 8 AUROC: 0.835 (LightGBM)
   - +17% improvement with additional 6 weeks of data

3. **Feature Importance**:
   - Assessment features alone: AUROC ~0.78 (Week 8)
   - VLE features alone: AUROC ~0.71 (Week 8)
   - Combined features: AUROC ~0.80 (Week 8)
   - All features (including demographics): AUROC ~0.84 (Week 8)
   - **Demographics add ~4% AUROC improvement**

4. **Model Comparison**:
   - Tree-based models (RF, XGBoost, LightGBM) outperform Logistic Regression
   - LightGBM shows best performance with lowest variance
   - All models significantly outperform majority baseline

### LCPO Evaluation (Leave-Course-Presentation-Out)

To test generalization across different courses, we performed LCPO cross-validation where each course presentation is held out as a test set.

#### Random Split vs LCPO Comparison (Week 8)
| Model | Split | AUROC | F1 | Balanced Acc |
|-------|-------|-------|-------|--------------|
| LogisticRegression | Random | 0.772±0.005 | 0.730±0.005 | 0.696±0.006 |
| LogisticRegression | LCPO | 0.768±0.079 | 0.656±0.269 | 0.682±0.084 |
| RandomForest | Random | 0.825±0.004 | 0.777±0.004 | 0.749±0.006 |
| RandomForest | LCPO | 0.792±0.094 | 0.750±0.065 | 0.720±0.076 |
| XGBoost | Random | 0.824±0.004 | 0.775±0.003 | 0.746±0.004 |
| XGBoost | LCPO | 0.792±0.087 | 0.745±0.081 | 0.716±0.076 |
| **LightGBM** | Random | **0.835±0.005** | **0.788±0.004** | **0.757±0.005** |
| **LightGBM** | LCPO | **0.804±0.087** | **0.758±0.066** | **0.726±0.074** |

#### LCPO Insights
1. **Generalization Gap**: 3-4% AUROC drop when testing on unseen courses
2. **Course Variability**: High standard deviation in LCPO (±0.087) indicates significant performance variation across courses
3. **Challenging Courses**: GGG courses show lower performance (AUROC ~0.60-0.63), suggesting course-specific characteristics
4. **Best Courses**: DDD, FFF, and EEE courses show excellent generalization (AUROC >0.85)
5. **Model Robustness**: LightGBM maintains best performance even in LCPO setting

## Repository Structure

```
OULAD/
├── DATA/                                    # OULAD dataset files
│   ├── studentInfo.csv
│   ├── studentVle.csv
│   ├── studentAssessment.csv
│   └── assessments.csv
├── OULAD_baseline_analysis_v5.py           # Main baseline evaluation script
├── OULAD_LCPO_evaluation.py                # LCPO cross-validation script
├── baseline_results_detailed.csv           # Detailed results (all models/features/weeks)
├── baseline_results_table.csv              # Summary results table
├── baseline_results_plot.png               # Visualization of results
├── lcpo_results_detailed.csv               # LCPO detailed results
├── random_vs_lcpo_comparison.csv           # Comparison of evaluation strategies
├── requirements.txt                        # Python dependencies
├── OULAD_IMPLEMENTATION_GUIDE.md          # Implementation guide
└── README.md                               # This file
```

## Setup and Installation

### 1. Create Virtual Environment
```bash
python3 -m venv oulad_env
source oulad_env/bin/activate  # On Windows: oulad_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Baseline Analysis

This will:
- Load and preprocess OULAD data
- Build features for weeks 2, 4, 6, 8
- Evaluate 5 models on 4 feature subsets
- Generate results files and visualization

### 4. Run LCPO Evaluation

This will:
- Perform leave-course-presentation-out cross-validation
- Compare with random split results
- Generate LCPO results and comparison

## Key Observations

### 1. Early Prediction is Challenging
- Week 2 performance (AUROC ~0.71) is significantly lower than Week 8 (AUROC ~0.84)
- Assessment data is sparse early in the course
- VLE activity patterns take time to establish

### 2. Assessment Performance is Highly Predictive
- Assessment-only features achieve AUROC ~0.78 at Week 8
- Stronger than VLE-only features (AUROC ~0.71)
- Suggests academic performance is the strongest indicator

### 3. Demographics Add Value
- Including demographics improves AUROC by ~4%
- Particularly important for early prediction (Week 2-4)
- Helps when behavioral data is limited

### 4. Course-Specific Challenges
- GGG courses show consistently lower performance
- May require course-specific models or features
- Suggests different student populations or course structures

### 5. Model Selection
- LightGBM provides best overall performance
- Random Forest is a close second
- Both significantly outperform linear models
- Tree-based models better capture non-linear patterns

## Future Work

### 1. Hyperparameter Tuning
- Grid search or Bayesian optimization for each model
- May improve performance by 2-5%
- Focus on LightGBM and Random Forest

### 2. Feature Engineering
- Temporal patterns (trend, acceleration)
- Interaction features (VLE × Assessment)
- Course-specific features
- Social network features (forum participation)

### 3. Advanced Models
- Deep learning (LSTM for temporal sequences)
- Attention mechanisms for activity patterns
- Multi-task learning (predict final grade + dropout)

### 4. Course-Specific Models
- Train separate models for challenging courses (GGG)
- Transfer learning from high-performing courses
- Meta-learning across course presentations

### 5. Interpretability
- SHAP values for feature importance
- Individual prediction explanations
- Identify intervention points

### 6. Deployment
- Real-time prediction API
- Dashboard for instructors
- Automated alert system for at-risk students

## Citation

If you use this code or analysis, please cite:

```bibtex
@misc{oulad_baseline_2026,
  title={OULAD Student Success Prediction: Comprehensive Baseline Analysis},
  author={BioAI Systems Lab},
  year={2026},
  url={https://github.com/BioAI-Systems-Lab/CourseAware}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Open University for providing the OULAD dataset
- Kuzilek J., Hlosta M., Zdrahal Z. (2017) Open University Learning Analytics dataset. Scientific Data 4:170171

## Contact

For questions or collaboration:
- GitHub: https://github.com/BioAI-Systems-Lab/CourseAware
- Issues: https://github.com/BioAI-Systems-Lab/CourseAware/issues