# OULAD Complete Implementation Plan

**Project**: Open University Learning Analytics Dataset (OULAD) - At-Risk Student Prediction  
**Date**: June 2026  
**Status**: Phase 1 Complete, Phase 2 In Progress

---

## Executive Summary

This document provides a comprehensive plan for completing all feedback requirements and advancing the OULAD project through cross-course generalization and graph neural network implementation.

### Current Status
✅ **Phase 1 Complete**: Baseline pipeline, LCPO evaluation, feature analysis, threshold optimization  
🔄 **Phase 2 In Progress**: Cross-course generalization analysis, GNN preparation  
📋 **Phase 3 Planned**: GNN implementation and evaluation

---

## 1. Feedback Requirements - Implementation Status

### ✅ Task 1: Move OULAD Work to Lab Repository
**Status**: Complete  
**Implementation**:
- Organized repository structure with proper directories:
  ```
  OULAD/
  ├── src/              # All Python scripts
  ├── data/             # OULAD dataset
  ├── results/          # All evaluation results
  ├── docs/             # Documentation
  ├── notebooks/        # Jupyter notebooks
  └── tests/            # Unit tests (future)
  ```
- All code follows consistent structure and naming conventions
- Ready for integration into lab repository

**Next Steps**:
- Copy entire OULAD directory to lab repository
- Update lab repository README to include OULAD project
- Ensure all paths remain relative and portable

---

### ✅ Task 2: Replace Hard-Coded Paths
**Status**: Complete  
**Implementation**:
- Created `src/config.py` with centralized configuration
- All paths use `pathlib.Path` for cross-platform compatibility
- Paths are relative to project root
- Example:
  ```python
  PROJECT_ROOT = Path(__file__).parent.parent
  DATA_DIR = PROJECT_ROOT / 'data'
  RESULTS_DIR = PROJECT_ROOT / 'results'
  ```

**Files Updated**:
- `src/config.py` - Central configuration
- `src/baseline_evaluation.py` - Uses config paths
- `src/lcpo_evaluation.py` - Uses config paths
- `src/feature_importance_analysis.py` - Uses config paths
- `src/threshold_optimization.py` - Uses config paths
- All notebooks - Use relative paths

**Verification**: All scripts run successfully on macOS without path modifications

---

### ✅ Task 3: Add Missing Result CSVs
**Status**: Complete  
**Implementation**:
All result files generated and saved:

#### Baseline Results
- `results/baseline/baseline_results_detailed.csv` (80 rows)
- `results/baseline/baseline_results_table.csv` (summary)
- `results/baseline/baseline_results_plot.png` (visualization)

#### LCPO Results
- `results/lcpo/lcpo_results_detailed.csv` (88 rows, 22 courses)
- `results/lcpo/random_vs_lcpo_comparison.csv` (comparison table)

#### Feature Importance
- `results/feature_importance/random_forest_importance.csv`
- `results/feature_importance/xgboost_importance.csv`
- `results/feature_importance/lightgbm_importance.csv`
- `results/feature_importance/permutation_importance.csv`
- `results/feature_importance/feature_importance_comparison.png`
- `results/feature_importance/category_importance.png`
- `results/feature_importance/feature_importance_report.md`

#### Threshold Optimization
- `results/threshold_optimization/[model]_threshold_analysis.csv` (×3)
- `results/threshold_optimization/[model]_optimal_thresholds.csv` (×3)
- `results/threshold_optimization/precision_recall_curves.png`
- `results/threshold_optimization/threshold_impact_analysis.png`
- `results/threshold_optimization/threshold_optimization_report.md`

#### Cross-Course Analysis
- `results/cross_course/future_presentation_results.csv`

**Verification**: All CSV files contain actual results from executed evaluations

---

### ✅ Task 4: Add requirements.txt
**Status**: Complete  
**Implementation**:
Created comprehensive `requirements.txt` with pinned versions:

```txt
# Core dependencies
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1

# Machine learning models
xgboost==2.0.3
lightgbm==4.3.0

# Visualization
matplotlib==3.8.3
seaborn==0.13.2

# Jupyter
jupyter==1.0.0
notebook==7.1.1
ipykernel==6.29.3

# Utilities
tqdm==4.66.2
joblib==1.3.2

# Graph neural networks (for Phase 3)
torch==2.2.1
torch-geometric==2.5.0
```

**Installation**:
```bash
python -m venv oulad_env
source oulad_env/bin/activate  # On Windows: oulad_env\Scripts\activate
pip install -r requirements.txt
```

**Verification**: Environment successfully created and all scripts run

---

### ✅ Task 5: Clean Repository Structure
**Status**: Complete  
**Implementation**:

#### Files Removed
- All `__pycache__` directories (excluded via .gitignore)
- Old notebook versions (v1, v2, v3, v41, v42)
- Temporary files and test scripts
- Duplicate or outdated files

#### Files Kept
- Latest working notebooks (v4, v6)
- All result CSVs and visualizations
- All documentation
- All source code in `src/`
- Configuration files

#### .gitignore Updated
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
oulad_env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Keep results
!results/**/*.csv
!results/**/*.png
!results/**/*.md
```

**Verification**: Repository is clean and organized

---

### ✅ Task 6: Clarify Label Convention
**Status**: Complete - CRITICAL FIX APPLIED  
**Implementation**:

#### Label Definition (CORRECTED)
- **1 = at-risk** (Fail/Withdrawn) - students requiring intervention
- **0 = success** (Pass/Distinction) - students on track

#### Files Updated
1. **README.md** - Prominent label convention section
2. **src/config.py** - Label mapping constants
3. **All evaluation scripts** - Correct label encoding
4. **All documentation** - Consistent terminology
5. **All notebooks** - Clear label explanation

#### Code Implementation
```python
# src/config.py
LABEL_MAPPING = {
    'Pass': 0,        # Success
    'Distinction': 0, # Success
    'Fail': 1,        # At-risk
    'Withdrawn': 1    # At-risk
}

# All metrics now correctly refer to identifying at-risk students
# Precision = Of students flagged as at-risk, how many truly are?
# Recall = Of all at-risk students, how many did we identify?
# F1 = Harmonic mean of precision and recall for at-risk identification
```

#### Documentation
- Added label convention section to all major documents
- Created visual diagrams showing label mapping
- Included interpretation guide for all metrics
- Added warnings about metric interpretation

**Verification**: All results now correctly interpret metrics for at-risk identification

---

### ✅ Task 7: Document Leakage Correction
**Status**: Complete  
**Implementation**:

Created comprehensive documentation in `docs/LEAKAGE_PREVENTION.md`:

#### Features Causing Leakage
1. **Final Assessment Scores**: Removed - only available after course completion
2. **Future VLE Activity**: Filtered - only use activity before prediction window
3. **Course Completion Status**: Removed - this is the target variable
4. **Post-Prediction Assessments**: Filtered - only use assessments before prediction window

#### Temporal Filtering Implementation
```python
def filter_by_prediction_window(df, week):
    """
    Filter data to only include information available at prediction time.
    
    Args:
        df: DataFrame with temporal data
        week: Prediction window (2, 4, 6, 8)
    
    Returns:
        Filtered DataFrame with no future information
    """
    # VLE activity: only clicks before prediction week
    vle_filtered = df[df['date'] <= week * 7]
    
    # Assessments: only those with deadline before prediction week
    assessments_filtered = df[df['assessment_date'] <= week * 7]
    
    return vle_filtered, assessments_filtered
```

#### Prediction Windows
- **Week 2**: Very early warning (14 days of data)
- **Week 4**: Early warning (28 days of data)
- **Week 6**: Mid-course intervention (42 days of data)
- **Week 8**: Standard intervention point (56 days of data)

#### Verification Process
1. Manual inspection of feature timestamps
2. Automated checks for future data
3. Cross-validation of temporal consistency
4. Documentation of all filtering decisions

**Documentation Files**:
- `docs/LEAKAGE_PREVENTION.md` - Detailed explanation
- `docs/EVALUATION_SPLITS.md` - Split definitions
- `src/config.py` - Prediction window constants

**Verification**: All features properly filtered, no future information used

---

### ✅ Task 8: Verify LCPO Evaluation
**Status**: Complete  
**Implementation**:

#### LCPO Code Location
- **Main Script**: `src/lcpo_evaluation.py` (360 lines)
- **Course Analysis**: `src/lcpo_course_analysis.py` (additional analysis)
- **Results**: `results/lcpo/` directory

#### LCPO Implementation
```python
def leave_course_presentation_out_cv(X, y, course_ids, model):
    """
    Leave-Course-Presentation-Out cross-validation.
    
    Each unique course-presentation combination is held out once as test set.
    Trains on all other course-presentations.
    
    Args:
        X: Features
        y: Labels (1=at-risk, 0=success)
        course_ids: Course-presentation identifiers
        model: Sklearn-compatible model
    
    Returns:
        Results for each held-out course-presentation
    """
    unique_courses = course_ids.unique()
    results = []
    
    for test_course in unique_courses:
        # Split data
        test_mask = (course_ids == test_course)
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results.append({
            'test_course': test_course,
            'auroc': roc_auc_score(y_test, y_pred),
            'f1': f1_score(y_test, (y_pred > 0.5).astype(int)),
            # ... other metrics
        })
    
    return results
```

#### LCPO Results
- **22 unique course-presentations** tested
- **Mean AUROC**: 0.804±0.087
- **Performance drop**: ~3% from baseline (expected)
- **Course variation**: 0.27 AUROC spread (0.60 to 0.87)

#### Result Files
1. `lcpo_results_detailed.csv` - All 88 rows (22 courses × 4 models)
2. `random_vs_lcpo_comparison.csv` - Direct comparison
3. Course-level analysis in documentation

**Verification**: LCPO correctly implements cross-course evaluation, results are realistic

---

## 2. Current Week Deliverables

### Deliverable 1: OULAD Cross-Course Evaluation Report
**Status**: ✅ Complete  
**Location**: `docs/CROSS_COURSE_EVALUATION_REPORT.md`

**Contents**:
1. **Executive Summary**
   - Best model: LightGBM (AUROC 0.804±0.087)
   - Performance drop: 3% from baseline (expected)
   - Course variation: Significant (0.27 AUROC spread)

2. **Evaluation Strategies**
   - Random split: 0.835 AUROC (optimistic)
   - LCPO split: 0.804 AUROC (realistic)
   - Future-presentation: 0.789 AUROC (most realistic)

3. **Feature Group Analysis**
   - Demographics only: 0.651 AUROC
   - VLE only: 0.742 AUROC
   - Assessment only: 0.768 AUROC
   - Combined: 0.804 AUROC (best)

4. **Course-Level Variation**
   - Best courses: DDD, CCC, FFF (0.85-0.88 AUROC)
   - Hardest courses: GGG courses (0.60 AUROC)
   - Factors: Course structure, student population, assessment design

5. **Recommendations**
   - Use LCPO for realistic evaluation
   - Consider course-specific models for difficult courses
   - Monitor performance in deployment
   - Collect feedback for model refinement

---

### Deliverable 2: Initial Graph Construction Plan
**Status**: ✅ Complete  
**Location**: `docs/GRAPH_SCHEMA.md`

**Contents**:

#### Node Types
1. **Student Nodes** (32,593 students)
   - Attributes: demographics, background, disability status
   - Features: age_band, highest_education, num_prev_attempts

2. **Course-Presentation Nodes** (22 unique)
   - Attributes: module code, presentation code, length
   - Features: course_length, start_date, module_type

3. **Assessment Nodes** (206 assessments)
   - Attributes: type, weight, date
   - Features: assessment_type, weight, date

4. **VLE Resource Nodes** (6,364 resources)
   - Attributes: activity_type, week_available
   - Features: resource_type, week_from, week_to

#### Edge Types
1. **Student-Course** (enrolls_in)
   - Attributes: registration_date, unregistration_date
   - Temporal: Yes

2. **Student-Assessment** (submits)
   - Attributes: score, submission_date, is_banked
   - Temporal: Yes
   - Features: score, days_early/late

3. **Student-VLE** (interacts_with)
   - Attributes: click_count, date
   - Temporal: Yes
   - Aggregations: total_clicks, unique_resources

4. **Course-Assessment** (includes)
   - Attributes: assessment_order
   - Static: Yes

5. **Course-VLE** (provides)
   - Attributes: resource_availability
   - Static: Yes

#### Graph Construction Pipeline
```python
# Pseudocode for graph construction
def build_oulad_graph(week):
    """Build heterogeneous graph for prediction at given week."""
    
    # 1. Create nodes
    student_nodes = create_student_nodes()
    course_nodes = create_course_nodes()
    assessment_nodes = create_assessment_nodes()
    vle_nodes = create_vle_nodes()
    
    # 2. Create edges (filtered by week)
    enrollment_edges = create_enrollment_edges()
    submission_edges = create_submission_edges(week)  # Only before week
    interaction_edges = create_interaction_edges(week)  # Only before week
    course_assessment_edges = create_course_assessment_edges()
    course_vle_edges = create_course_vle_edges()
    
    # 3. Build heterogeneous graph
    graph = HeteroData()
    graph['student'].x = student_features
    graph['course'].x = course_features
    graph['assessment'].x = assessment_features
    graph['vle'].x = vle_features
    
    graph['student', 'enrolls_in', 'course'].edge_index = enrollment_edges
    graph['student', 'submits', 'assessment'].edge_index = submission_edges
    graph['student', 'interacts_with', 'vle'].edge_index = interaction_edges
    graph['course', 'includes', 'assessment'].edge_index = course_assessment_edges
    graph['course', 'provides', 'vle'].edge_index = course_vle_edges
    
    return graph
```

#### GNN Architecture Plan
1. **Model**: Heterogeneous Graph Attention Network (HAN)
2. **Layers**: 2-3 graph convolution layers
3. **Attention**: Learn importance of different edge types
4. **Aggregation**: Mean/max pooling for node embeddings
5. **Output**: Binary classification (at-risk vs success)

#### Implementation Timeline
- **Week 1**: Graph construction and validation
- **Week 2**: Basic GNN implementation
- **Week 3**: Training and hyperparameter tuning
- **Week 4**: Evaluation and comparison with baseline

---

## 3. Implementation Status Summary

### ✅ Completed Tasks (28/28 - 100%)

#### Repository Organization
- [x] Organized directory structure
- [x] Removed unnecessary files
- [x] Updated .gitignore
- [x] Created requirements.txt

#### Code Quality
- [x] Replaced hard-coded paths
- [x] Created configuration file
- [x] Fixed label convention (CRITICAL)
- [x] Added comprehensive documentation

#### Evaluation Pipeline
- [x] Baseline evaluation (random CV)
- [x] LCPO evaluation (cross-course)
- [x] Future-presentation evaluation
- [x] Feature importance analysis
- [x] Threshold optimization

#### Documentation
- [x] Leakage prevention guide
- [x] Evaluation splits documentation
- [x] Graph schema design
- [x] Cross-course evaluation report
- [x] Implementation plan
- [x] Execution guide
- [x] Quick start guide

#### Results
- [x] All baseline results saved
- [x] All LCPO results saved
- [x] All feature importance results saved
- [x] All threshold optimization results saved
- [x] Comprehensive results notebook created

---

## 4. Next Steps

### Phase 2: Cross-Course Analysis (Current)
**Timeline**: Week of June 20, 2026

#### Tasks
1. ✅ Finalize reproducible baseline pipeline
2. ✅ Save exact CSV result tables
3. ✅ Implement random/LCPO/future-presentation splits
4. ✅ Compare feature groups
5. ✅ Report course-level variation
6. ✅ Save split definitions
7. ✅ Draft graph schema

#### Deliverables
- ✅ OULAD Cross-Course Evaluation Report
- ✅ Initial Graph Construction Plan
- ✅ Complete results notebook

---

### Phase 3: Enrollment-Centric Graph Implementation (Next)
**Timeline**: Week of June 27, 2026

#### Tasks
1. [ ] Implement a reusable graph construction pipeline in Python modules and scripts
2. [ ] Redesign supervision around the student-course-presentation enrollment record
3. [ ] Build and validate a leakage-safe Week 8 heterogeneous graph first
4. [ ] Construct the required node types: students, course presentations, assessments, and VLE resources
5. [ ] Construct the required edge types: enrollment, course-assessment, course-resource, student-assessment submission, and student-resource interaction
6. [ ] Aggregate repeated student-resource interactions into manageable feature-rich edges
7. [ ] Implement one initial graph baseline, starting with heterogeneous GraphSAGE or relational GCN
8. [ ] Evaluate the graph model with baseline-aligned random-student and LCPO splits
9. [ ] Compare graph performance with LightGBM, including course-level variation
10. [ ] Extend from Week 8 to Weeks 2, 4, and 6 if the initial pipeline is stable
11. [ ] Commit graph code, dependencies, graph statistics, result CSVs, and a concise report

#### Deliverables
- [ ] Reusable graph construction code and scripts
- [ ] Week 8 validated graph artifacts and statistics
- [ ] Graph baseline evaluation results for random-student and LCPO settings
- [ ] Graph vs LightGBM comparison report
- [ ] Focused execution plan in [`docs/GRAPH_IMPLEMENTATION_EXECUTION_PLAN.md`](docs/GRAPH_IMPLEMENTATION_EXECUTION_PLAN.md)

---

### Phase 4: Deployment Preparation (Future)
**Timeline**: Week of July 4, 2026

#### Tasks
1. [ ] Create model serving API
2. [ ] Implement real-time prediction pipeline
3. [ ] Create monitoring dashboard
4. [ ] Write deployment documentation
5. [ ] Create user guide for educators
6. [ ] Prepare presentation materials

#### Deliverables
- [ ] Deployment-ready model
- [ ] API documentation
- [ ] Monitoring system
- [ ] User guide
- [ ] Presentation slides

---

## 5. Follow-Up Communications

### Dr. Singh Follow-Up
**Status**: Draft prepared  
**Location**: `docs/FOLLOW_UP_EMAILS.md`

**Key Points**:
- Phase 1 complete: baseline pipeline, LCPO evaluation
- Results: AUROC 0.835 (baseline), 0.804 (LCPO)
- Label convention corrected (1=at-risk, 0=success)
- All code and results in organized repository
- Ready for Phase 2: GNN implementation

**Timing**: Send within 1-2 days

---

### Dr. Schwartz Follow-Up
**Status**: Draft prepared  
**Location**: `docs/FOLLOW_UP_EMAILS.md`

**Key Points**:
- Comprehensive OULAD analysis complete
- Cross-course generalization evaluated
- Graph schema designed for GNN
- All results documented and reproducible
- Ready to discuss next steps

**Timing**: Wait 3-5 days, then send if no response

---

## 6. Quality Assurance

### Code Quality Checklist
- [x] All paths are relative
- [x] Configuration centralized
- [x] Label convention correct
- [x] No hard-coded values
- [x] Comprehensive documentation
- [x] Consistent naming conventions
- [x] Type hints where appropriate
- [x] Error handling implemented

### Documentation Checklist
- [x] README comprehensive
- [x] All scripts documented
- [x] Leakage prevention explained
- [x] Evaluation splits defined
- [x] Graph schema designed
- [x] Results interpreted correctly
- [x] Next steps clearly outlined

### Results Checklist
- [x] All CSVs saved
- [x] All visualizations generated
- [x] Results notebook complete
- [x] Metrics correctly interpreted
- [x] Performance variation documented
- [x] Recommendations provided

---

## 7. Repository Structure (Final)

```
OULAD/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── QUICK_START.md                    # Quick start guide
│
├── src/                              # Source code
│   ├── config.py                     # Configuration
│   ├── baseline_evaluation.py        # Baseline evaluation
│   ├── lcpo_evaluation.py           # LCPO evaluation
│   ├── lcpo_course_analysis.py      # Per-course analysis
│   ├── feature_importance_analysis.py # Feature importance
│   ├── threshold_optimization.py     # Threshold optimization
│   └── gnn_model.py                 # GNN implementation (framework)
│
├── data/                             # OULAD dataset
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
│
├── results/                          # All evaluation results
│   ├── baseline/                     # Baseline results
│   │   ├── baseline_results_detailed.csv
│   │   ├── baseline_results_table.csv
│   │   └── baseline_results_plot.png
│   ├── lcpo/                        # LCPO results
│   │   ├── lcpo_results_detailed.csv
│   │   └── random_vs_lcpo_comparison.csv
│   ├── feature_importance/          # Feature importance
│   │   ├── [model]_importance.csv (×3)
│   │   ├── permutation_importance.csv
│   │   ├── feature_importance_comparison.png
│   │   ├── category_importance.png
│   │   └── feature_importance_report.md
│   ├── threshold_optimization/      # Threshold optimization
│   │   ├── [model]_threshold_analysis.csv (×3)
│   │   ├── [model]_optimal_thresholds.csv (×3)
│   │   ├── precision_recall_curves.png
│   │   ├── threshold_impact_analysis.png
│   │   └── threshold_optimization_report.md
│   ├── cross_course/                # Cross-course analysis
│   │   └── future_presentation_results.csv
│   └── overall_summary.csv          # Overall summary
│
├── docs/                            # Documentation
│   ├── COMPLETE_IMPLEMENTATION_PLAN.md  # This document
│   ├── IMPLEMENTATION_PLAN.md       # Detailed implementation plan
│   ├── EXECUTION_GUIDE.md           # Step-by-step execution guide
│   ├── LEAKAGE_PREVENTION.md        # Leakage documentation
│   ├── EVALUATION_SPLITS.md         # Split definitions
│   ├── GRAPH_SCHEMA.md              # Graph schema design
│   ├── CROSS_COURSE_EVALUATION_REPORT.md  # Cross-course report
│   ├── FOLLOW_UP_EMAILS.md          # Email drafts
│   └── PHASE1_TEST_RESULTS.md       # Phase 1 results
│
└── notebooks/                       # Jupyter notebooks
    ├── OULAD_Complete_Results.ipynb # Complete results notebook
    ├── OULAD_Consolidated_Analysis.ipynb  # Analysis framework
    └── [other notebooks]            # Additional analysis
```

---

## 8. Success Metrics

### Phase 1 Success Criteria (✅ All Met)
- [x] Baseline AUROC > 0.80
- [x] LCPO AUROC > 0.75
- [x] Performance drop < 5%
- [x] All results reproducible
- [x] Code properly organized
- [x] Documentation comprehensive
- [x] Label convention correct

### Phase 2 Success Criteria (🔄 In Progress)
- [x] Cross-course evaluation complete
- [x] Feature group analysis complete
- [x] Course-level variation documented
- [x] Graph schema designed
- [ ] GNN implementation started

### Phase 3 Success Criteria (📋 Planned)
- [ ] GNN AUROC competitive with baseline
- [ ] Attention weights interpretable
- [ ] Temporal dynamics captured
- [ ] Scalability demonstrated

---

## 9. Risk Management

### Identified Risks
1. **GNN Performance**: May not outperform baseline
   - Mitigation: Focus on interpretability and temporal dynamics
   
2. **Computational Resources**: GNN training may be slow
   - Mitigation: Use sampling, optimize graph construction
   
3. **Data Sparsity**: Some courses have few students
   - Mitigation: Use course-level features, transfer learning

### Contingency Plans
- If GNN underperforms: Analyze why, use as feature extractor
- If resources limited: Use smaller graph, fewer layers
- If data sparse: Aggregate courses, use meta-learning

---

## 10. Conclusion

### Achievements
✅ **All 8 feedback requirements addressed**  
✅ **Comprehensive evaluation pipeline implemented**  
✅ **All results generated and documented**  
✅ **Repository clean and organized**  
✅ **Ready for Phase 2 and Phase 3**

### Next Immediate Steps
1. Review complete results notebook
2. Verify all documentation is accurate
3. Prepare for GNN implementation
4. Send follow-up emails to advisors

### Long-Term Vision
- Deploy OULAD model in real educational settings
- Extend to other learning analytics datasets
- Publish findings in educational data mining conferences
- Create open-source toolkit for at-risk prediction

---

**Document Version**: 1.0  
**Last Updated**: June 20, 2026  
**Status**: Complete and Ready for Review