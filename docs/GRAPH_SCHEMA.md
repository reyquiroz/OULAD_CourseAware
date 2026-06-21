# OULAD Heterogeneous Graph Schema

## Overview

This document defines the heterogeneous graph structure for representing the Open University Learning Analytics Dataset (OULAD) as a graph neural network (GNN) input. The graph captures the complex relationships between students, courses, assessments, and learning resources.

## Graph Type

**Heterogeneous Graph**: Multiple node types and edge types with different feature spaces.

## Node Types

### 1. Student Nodes

**Type**: `student`  
**Count**: 32,593 unique students  
**ID**: `id_student`

**Node Features**:

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `gender` | Categorical | Student gender | M, F |
| `region` | Categorical | Geographic region | 13 regions |
| `highest_education` | Categorical | Education level at enrollment | No Formal quals, Lower Than A Level, A Level or Equivalent, HE Qualification, Post Graduate Qualification |
| `imd_band` | Categorical | Index of Multiple Deprivation | 0-10%, 10-20%, ..., 90-100% |
| `age_band` | Categorical | Age group | 0-35, 35-55, 55<= |
| `num_of_prev_attempts` | Numerical | Previous course attempts | 0, 1, 2, 3+ |
| `disability` | Categorical | Disability status | Y, N |
| `studied_credits` | Numerical | Credits studied | Integer |

**Feature Encoding**:
- Categorical features: One-hot encoding or embedding
- Numerical features: Standardization or normalization

**Node Label** (for supervised learning):
- `target`: Binary (1 = at-risk, 0 = success)

### 2. Course-Presentation Nodes

**Type**: `course_presentation`  
**Count**: 22 unique course-presentation combinations  
**ID**: `code_module + "_" + code_presentation`

**Node Features**:

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `code_module` | Categorical | Course code | AAA, BBB, CCC, DDD, EEE, FFF, GGG |
| `code_presentation` | Categorical | Presentation semester | 2013B, 2013J, 2014B, 2014J |
| `module_presentation_length` | Numerical | Course duration in days | Integer |
| `num_students` | Numerical | Number of enrolled students | Integer |
| `pass_rate` | Numerical | Historical pass rate | 0.0-1.0 |
| `avg_vle_clicks` | Numerical | Average VLE activity | Float |
| `num_assessments` | Numerical | Number of assessments | Integer |

**Derived Features**:
- `presentation_year`: Extracted from code_presentation
- `presentation_semester`: B (February) or J (October)

### 3. Assessment Nodes

**Type**: `assessment`  
**Count**: 206 unique assessments  
**ID**: `id_assessment`

**Node Features**:

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `assessment_type` | Categorical | Type of assessment | TMA (Tutor Marked Assessment), CMA (Computer Marked Assessment), Exam |
| `date` | Numerical | Due date (days from start) | Integer |
| `weight` | Numerical | Weight in final grade | 0.0-100.0 |
| `date_normalized` | Numerical | Normalized due date | 0.0-1.0 |

**Temporal Features**:
- `week_due`: Week number when due
- `is_early`: Boolean (due in first 4 weeks)
- `is_final`: Boolean (final exam)

### 4. VLE Resource Nodes

**Type**: `vle_resource`  
**Count**: ~20,000 unique VLE resources  
**ID**: `id_site`

**Node Features**:

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `activity_type` | Categorical | Type of resource | oucontent, resource, homepage, subpage, glossary, forumng, oucollaborate, dataplus, quiz, ouelluminate, sharedsubpage, questionnaire, page, externalquiz, ouwiki, dualpane, repeatactivity, folder, url, htmlactivity |
| `week_from` | Numerical | Week available from | Integer |
| `week_to` | Numerical | Week available to | Integer |
| `avg_clicks` | Numerical | Average clicks per student | Float |
| `total_clicks` | Numerical | Total clicks across all students | Integer |

**Derived Features**:
- `availability_duration`: week_to - week_from
- `is_always_available`: Boolean (week_from == 0)

## Edge Types

### 1. Student → Course-Presentation (ENROLLED_IN)

**Type**: `enrolled_in`  
**Direction**: Student → Course-Presentation  
**Count**: 32,593 enrollments

**Edge Features**:

| Feature | Type | Description |
|---------|------|-------------|
| `date_registration` | Numerical | Registration date (days from start) |
| `date_unregistration` | Numerical | Unregistration date (if withdrawn) |
| `is_early_registration` | Boolean | Registered before course start |
| `enrollment_duration` | Numerical | Days enrolled |

**Note**: `date_unregistration` should be used carefully to avoid leakage (it reveals withdrawal).

### 2. Student → Assessment (SUBMITTED)

**Type**: `submitted`  
**Direction**: Student → Assessment  
**Count**: ~173,000 submissions

**Edge Features**:

| Feature | Type | Description |
|---------|------|-------------|
| `score` | Numerical | Assessment score (0-100) |
| `date_submitted` | Numerical | Submission date (days from start) |
| `is_banked` | Boolean | From previous attempt |
| `days_late` | Numerical | Days after due date (0 if on time) |
| `score_normalized` | Numerical | Z-score within assessment |

**Temporal Filtering**: For prediction at week W, only include submissions where `date_submitted <= W * 7`.

### 3. Student → VLE Resource (INTERACTED_WITH)

**Type**: `interacted_with`  
**Direction**: Student → VLE Resource  
**Count**: ~10 million interactions

**Edge Features**:

| Feature | Type | Description |
|---------|------|-------------|
| `sum_click` | Numerical | Total clicks on this resource |
| `date_first` | Numerical | First interaction date |
| `date_last` | Numerical | Last interaction date |
| `num_days_active` | Numerical | Number of days with activity |
| `click_frequency` | Numerical | Clicks per day active |

**Aggregation**: Multiple interactions on same day are aggregated.

**Temporal Filtering**: For prediction at week W, only include interactions where `date <= W * 7`.

### 4. Course-Presentation → Assessment (HAS_ASSESSMENT)

**Type**: `has_assessment`  
**Direction**: Course-Presentation → Assessment  
**Count**: 206 (one per assessment)

**Edge Features**:

| Feature | Type | Description |
|---------|------|-------------|
| `weight` | Numerical | Assessment weight in course |
| `date` | Numerical | Due date |
| `is_required` | Boolean | Required for completion |

### 5. Course-Presentation → VLE Resource (CONTAINS_RESOURCE)

**Type**: `contains_resource`  
**Direction**: Course-Presentation → VLE Resource  
**Count**: ~20,000

**Edge Features**:

| Feature | Type | Description |
|---------|------|-------------|
| `week_from` | Numerical | Week available from |
| `week_to` | Numerical | Week available to |

## Graph Construction Pipeline

### Step 1: Load Data

```python
import pandas as pd
from config import DATA_DIR

# Load all OULAD tables
student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
student_assess = pd.read_csv(DATA_DIR / "studentAssessment.csv")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")
vle = pd.read_csv(DATA_DIR / "vle.csv")
student_registration = pd.read_csv(DATA_DIR / "studentRegistration.csv")
```

### Step 2: Create Node Feature Matrices

```python
import torch
from torch_geometric.data import HeteroData

data = HeteroData()

# Student nodes
student_features = prepare_student_features(student_info)
data['student'].x = torch.tensor(student_features, dtype=torch.float)
data['student'].y = torch.tensor(student_info['target'].values, dtype=torch.long)

# Course-presentation nodes
course_features = prepare_course_features(courses, student_info)
data['course_presentation'].x = torch.tensor(course_features, dtype=torch.float)

# Assessment nodes
assessment_features = prepare_assessment_features(assessments)
data['assessment'].x = torch.tensor(assessment_features, dtype=torch.float)

# VLE resource nodes
vle_features = prepare_vle_features(vle, student_vle)
data['vle_resource'].x = torch.tensor(vle_features, dtype=torch.float)
```

### Step 3: Create Edge Indices and Features

```python
# Student → Course-Presentation edges
enrollment_edges, enrollment_features = create_enrollment_edges(student_info)
data['student', 'enrolled_in', 'course_presentation'].edge_index = enrollment_edges
data['student', 'enrolled_in', 'course_presentation'].edge_attr = enrollment_features

# Student → Assessment edges
submission_edges, submission_features = create_submission_edges(student_assess)
data['student', 'submitted', 'assessment'].edge_index = submission_edges
data['student', 'submitted', 'assessment'].edge_attr = submission_features

# Student → VLE Resource edges
interaction_edges, interaction_features = create_interaction_edges(student_vle)
data['student', 'interacted_with', 'vle_resource'].edge_index = interaction_edges
data['student', 'interacted_with', 'vle_resource'].edge_attr = interaction_features

# Course-Presentation → Assessment edges
course_assess_edges = create_course_assessment_edges(assessments)
data['course_presentation', 'has_assessment', 'assessment'].edge_index = course_assess_edges

# Course-Presentation → VLE Resource edges
course_vle_edges = create_course_vle_edges(vle)
data['course_presentation', 'contains_resource', 'vle_resource'].edge_index = course_vle_edges
```

### Step 4: Temporal Filtering (for prediction windows)

```python
def filter_graph_by_window(data, window_days):
    """
    Filter graph to only include information available up to window_days
    """
    # Filter submission edges
    submission_mask = data['student', 'submitted', 'assessment'].edge_attr[:, 1] <= window_days
    data['student', 'submitted', 'assessment'].edge_index = \
        data['student', 'submitted', 'assessment'].edge_index[:, submission_mask]
    data['student', 'submitted', 'assessment'].edge_attr = \
        data['student', 'submitted', 'assessment'].edge_attr[submission_mask]
    
    # Filter interaction edges
    interaction_mask = data['student', 'interacted_with', 'vle_resource'].edge_attr[:, 2] <= window_days
    data['student', 'interacted_with', 'vle_resource'].edge_index = \
        data['student', 'interacted_with', 'vle_resource'].edge_index[:, interaction_mask]
    data['student', 'interacted_with', 'vle_resource'].edge_attr = \
        data['student', 'interacted_with', 'vle_resource'].edge_attr[interaction_mask]
    
    return data
```

## Graph Statistics

### Node Counts

| Node Type | Count | Avg Degree |
|-----------|-------|------------|
| Student | 32,593 | ~320 |
| Course-Presentation | 22 | ~1,500 |
| Assessment | 206 | ~840 |
| VLE Resource | ~20,000 | ~500 |

### Edge Counts

| Edge Type | Count | Sparsity |
|-----------|-------|----------|
| Student → Course-Presentation | 32,593 | Dense |
| Student → Assessment | ~173,000 | Sparse |
| Student → VLE Resource | ~10M | Very Sparse |
| Course-Presentation → Assessment | 206 | Dense |
| Course-Presentation → VLE Resource | ~20,000 | Dense |

### Graph Properties

- **Heterogeneous**: 4 node types, 5 edge types
- **Temporal**: Edge features include timestamps
- **Attributed**: Rich node and edge features
- **Large-scale**: ~10M edges
- **Sparse**: Most student-resource pairs have no interaction

## GNN Model Architectures

### 1. Heterogeneous Graph Attention Network (HAN)

```python
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class OULAD_HAN(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        
        self.conv1 = HeteroConv({
            ('student', 'enrolled_in', 'course_presentation'): GATConv(-1, hidden_dim, heads=num_heads),
            ('student', 'submitted', 'assessment'): GATConv(-1, hidden_dim, heads=num_heads),
            ('student', 'interacted_with', 'vle_resource'): GATConv(-1, hidden_dim, heads=num_heads),
        })
        
        self.conv2 = HeteroConv({
            ('student', 'enrolled_in', 'course_presentation'): GATConv(hidden_dim * num_heads, hidden_dim),
            ('student', 'submitted', 'assessment'): GATConv(hidden_dim * num_heads, hidden_dim),
            ('student', 'interacted_with', 'vle_resource'): GATConv(hidden_dim * num_heads, hidden_dim),
        })
        
        self.classifier = nn.Linear(hidden_dim, 2)  # Binary classification
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # Classify student nodes
        return self.classifier(x_dict['student'])
```

### 2. Relational Graph Convolutional Network (R-GCN)

```python
from torch_geometric.nn import RGCNConv

class OULAD_RGCN(nn.Module):
    def __init__(self, num_relations=5, hidden_dim=64):
        super().__init__()
        
        self.conv1 = RGCNConv(-1, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return self.classifier(x)
```

## Advantages of Graph Representation

1. **Relational Learning**: Captures complex relationships between entities
2. **Information Propagation**: Students can benefit from similar students' patterns
3. **Multi-hop Reasoning**: Can learn from indirect connections
4. **Heterogeneous Information**: Integrates multiple data types naturally
5. **Temporal Dynamics**: Edge features capture temporal patterns

## Challenges and Considerations

1. **Scalability**: 10M edges require efficient sampling strategies
2. **Temporal Leakage**: Must carefully filter edges by prediction window
3. **Class Imbalance**: 52.8% at-risk vs 47.2% success
4. **Cold Start**: New students have no historical interactions
5. **Computational Cost**: GNN training is more expensive than traditional ML

## Next Steps

1. Implement graph construction pipeline
2. Benchmark simple GNN models (GCN, GAT)
3. Compare with traditional ML baselines
4. Explore temporal graph networks (TGN)
5. Investigate graph sampling strategies for scalability

## References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Heterogeneous Graph Learning: https://arxiv.org/abs/1903.07293
- Temporal Graph Networks: https://arxiv.org/abs/2006.10637