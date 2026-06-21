# OULAD Initial Graph Construction Plan

**Date**: June 20, 2026  
**Project**: OULAD Student Success Prediction - Graph Neural Network Approach  
**Authors**: BioAI Systems Lab

---

## Executive Summary

This document outlines a comprehensive plan for constructing heterogeneous graphs from the Open University Learning Analytics Dataset (OULAD) to enable graph neural network (GNN) based student success prediction. The proposed graph structure captures complex relationships between students, courses, assessments, and learning resources, enabling relational learning that goes beyond traditional feature-based approaches.

### Key Components

1. **Heterogeneous Graph Schema**: 4 node types, 5 edge types
2. **Temporal Filtering**: Leakage-safe graph construction for prediction windows
3. **Implementation Pipeline**: Step-by-step construction process
4. **Baseline GNN Models**: Initial architectures for evaluation
5. **Evaluation Strategy**: Comparison with traditional ML baselines

### Expected Benefits

- **Relational Learning**: Leverage student-student, student-resource similarities
- **Multi-hop Reasoning**: Learn from indirect connections
- **Scalability**: Handle large-scale interaction data efficiently
- **Interpretability**: Understand prediction through graph structure

---

## 1. Graph Schema Design

### 1.1 Node Types

#### Node Type 1: Student Nodes

**Count**: 32,593 unique students  
**ID**: `id_student`

**Node Features** (13 dimensions after encoding):

| Feature | Type | Encoding | Dimensions |
|---------|------|----------|------------|
| `gender` | Categorical | One-hot | 2 |
| `region` | Categorical | Embedding | 8 |
| `highest_education` | Categorical | One-hot | 5 |
| `imd_band` | Categorical | One-hot | 10 |
| `age_band` | Categorical | One-hot | 3 |
| `num_of_prev_attempts` | Numerical | Normalized | 1 |
| `disability` | Categorical | One-hot | 2 |
| `studied_credits` | Numerical | Normalized | 1 |

**Node Label** (for supervised learning):
- `target`: Binary (1 = at-risk, 0 = success)

**Rationale**: Student demographics provide baseline risk factors available from enrollment.

#### Node Type 2: Course-Presentation Nodes

**Count**: 22 unique course-presentation combinations  
**ID**: `code_module + "_" + code_presentation`

**Node Features** (15 dimensions):

| Feature | Type | Description | Dimensions |
|---------|------|-------------|------------|
| `code_module` | Categorical | Course code (AAA-GGG) | 7 |
| `code_presentation` | Categorical | Semester (2013B, 2013J, 2014B, 2014J) | 4 |
| `module_presentation_length` | Numerical | Course duration (days) | 1 |
| `num_students` | Numerical | Enrollment count | 1 |
| `pass_rate` | Numerical | Historical pass rate | 1 |
| `avg_vle_clicks` | Numerical | Average VLE activity | 1 |

**Derived Features**:
- `presentation_year`: Extracted from code_presentation
- `presentation_semester`: B (February) or J (October)

**Rationale**: Course characteristics influence student success patterns.

#### Node Type 3: Assessment Nodes

**Count**: 206 unique assessments  
**ID**: `id_assessment`

**Node Features** (8 dimensions):

| Feature | Type | Description | Dimensions |
|---------|------|-------------|------------|
| `assessment_type` | Categorical | TMA, CMA, Exam | 3 |
| `date` | Numerical | Due date (days from start) | 1 |
| `weight` | Numerical | Weight in final grade (0-100) | 1 |
| `date_normalized` | Numerical | Normalized due date (0-1) | 1 |
| `week_due` | Numerical | Week number when due | 1 |
| `is_final` | Binary | Is final exam | 1 |

**Rationale**: Assessment characteristics affect difficulty and student performance.

#### Node Type 4: VLE Resource Nodes

**Count**: ~20,000 unique VLE resources  
**ID**: `id_site`

**Node Features** (25 dimensions):

| Feature | Type | Description | Dimensions |
|---------|------|-------------|------------|
| `activity_type` | Categorical | Resource type (20 types) | 20 |
| `week_from` | Numerical | Week available from | 1 |
| `week_to` | Numerical | Week available to | 1 |
| `avg_clicks` | Numerical | Average clicks per student | 1 |
| `total_clicks` | Numerical | Total clicks | 1 |
| `availability_duration` | Numerical | week_to - week_from | 1 |

**Activity Types**: oucontent, resource, homepage, subpage, glossary, forumng, oucollaborate, dataplus, quiz, ouelluminate, sharedsubpage, questionnaire, page, externalquiz, ouwiki, dualpane, repeatactivity, folder, url, htmlactivity

**Rationale**: Resource types and usage patterns indicate learning strategies.

### 1.2 Edge Types

#### Edge Type 1: Student → Course-Presentation (ENROLLED_IN)

**Count**: 32,593 enrollments  
**Direction**: Student → Course-Presentation

**Edge Features** (4 dimensions):

| Feature | Type | Description |
|---------|------|-------------|
| `date_registration` | Numerical | Registration date (days from start) |
| `is_early_registration` | Binary | Registered before course start |
| `enrollment_duration` | Numerical | Days enrolled (if withdrawn) |
| `is_active` | Binary | Currently enrolled |

**Note**: `date_unregistration` excluded to prevent leakage (reveals withdrawal).

**Rationale**: Enrollment timing and patterns correlate with success.

#### Edge Type 2: Student → Assessment (SUBMITTED)

**Count**: ~173,000 submissions  
**Direction**: Student → Assessment

**Edge Features** (5 dimensions):

| Feature | Type | Description |
|---------|------|-------------|
| `score` | Numerical | Assessment score (0-100) |
| `date_submitted` | Numerical | Submission date (days from start) |
| `is_banked` | Binary | From previous attempt |
| `days_late` | Numerical | Days after due date (0 if on time) |
| `score_normalized` | Numerical | Z-score within assessment |

**Temporal Filtering**: For prediction at week W, only include submissions where `date_submitted <= W * 7`.

**Rationale**: Assessment performance is the strongest predictor of success.

#### Edge Type 3: Student → VLE Resource (INTERACTED_WITH)

**Count**: ~10 million interactions  
**Direction**: Student → VLE Resource

**Edge Features** (5 dimensions):

| Feature | Type | Description |
|---------|------|-------------|
| `sum_click` | Numerical | Total clicks on this resource |
| `date_first` | Numerical | First interaction date |
| `date_last` | Numerical | Last interaction date |
| `num_days_active` | Numerical | Number of days with activity |
| `click_frequency` | Numerical | Clicks per day active |

**Aggregation**: Multiple interactions on same day are aggregated.

**Temporal Filtering**: For prediction at week W, only include interactions where `date <= W * 7`.

**Rationale**: VLE engagement patterns indicate effort and learning strategies.

#### Edge Type 4: Course-Presentation → Assessment (HAS_ASSESSMENT)

**Count**: 206 (one per assessment)  
**Direction**: Course-Presentation → Assessment

**Edge Features** (3 dimensions):

| Feature | Type | Description |
|---------|------|-------------|
| `weight` | Numerical | Assessment weight in course |
| `date` | Numerical | Due date |
| `is_required` | Binary | Required for completion |

**Rationale**: Links assessments to their courses for context.

#### Edge Type 5: Course-Presentation → VLE Resource (CONTAINS_RESOURCE)

**Count**: ~20,000  
**Direction**: Course-Presentation → VLE Resource

**Edge Features** (2 dimensions):

| Feature | Type | Description |
|---------|------|-------------|
| `week_from` | Numerical | Week available from |
| `week_to` | Numerical | Week available to |

**Rationale**: Links resources to their courses for context.

### 1.3 Graph Statistics

| Metric | Value |
|--------|-------|
| **Total Nodes** | ~52,821 |
| **Total Edges** | ~10.4 million |
| **Average Node Degree** | ~197 |
| **Graph Density** | Very sparse (~0.0037%) |
| **Largest Connected Component** | ~99.8% of nodes |
| **Graph Diameter** | ~6 hops |

---

## 2. Graph Construction Pipeline

### 2.1 Phase 1: Data Loading and Preprocessing

**Step 1.1: Load OULAD Tables**

```python
import pandas as pd
from pathlib import Path
from config import DATA_DIR

# Load all tables
student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
student_assess = pd.read_csv(DATA_DIR / "studentAssessment.csv")
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")
vle = pd.read_csv(DATA_DIR / "vle.csv")
student_registration = pd.read_csv(DATA_DIR / "studentRegistration.csv")

print(f"Loaded {len(student_info)} students")
print(f"Loaded {len(student_vle)} VLE interactions")
print(f"Loaded {len(student_assess)} assessment submissions")
```

**Step 1.2: Create Target Labels**

```python
# CORRECTED label convention: 1 = at-risk, 0 = success
student_info["target"] = student_info["final_result"].apply(
    lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
)

print(f"At-risk (1): {(student_info['target'] == 1).sum()} students")
print(f"Success (0): {(student_info['target'] == 0).sum()} students")
```

**Step 1.3: Temporal Filtering**

```python
def filter_by_window(df, date_col, window_days):
    """Filter data up to prediction window"""
    return df[df[date_col] <= window_days].copy()

# Example: Week 8 prediction (56 days)
window = 56
student_vle_w8 = filter_by_window(student_vle, 'date', window)
student_assess_w8 = filter_by_window(student_assess, 'date_submitted', window)
```

### 2.2 Phase 2: Node Feature Matrix Construction

**Step 2.1: Student Node Features**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_student_features(student_info):
    """Create student node feature matrix"""
    features = []
    
    # Categorical features (one-hot)
    gender = pd.get_dummies(student_info['gender'], prefix='gender')
    education = pd.get_dummies(student_info['highest_education'], prefix='edu')
    age = pd.get_dummies(student_info['age_band'], prefix='age')
    disability = pd.get_dummies(student_info['disability'], prefix='dis')
    
    # Numerical features (normalized)
    scaler = StandardScaler()
    num_features = scaler.fit_transform(
        student_info[['num_of_prev_attempts', 'studied_credits']]
    )
    
    # Combine
    feature_matrix = np.hstack([
        gender.values,
        education.values,
        age.values,
        disability.values,
        num_features
    ])
    
    return feature_matrix

student_features = create_student_features(student_info)
print(f"Student features shape: {student_features.shape}")
```

**Step 2.2: Course-Presentation Node Features**

```python
def create_course_features(courses, student_info):
    """Create course-presentation node feature matrix"""
    # Aggregate student statistics per course
    course_stats = student_info.groupby(
        ['code_module', 'code_presentation']
    ).agg({
        'id_student': 'count',
        'target': 'mean'
    }).reset_index()
    
    course_stats.columns = ['code_module', 'code_presentation', 
                            'num_students', 'pass_rate']
    
    # Merge with course metadata
    course_data = courses.merge(course_stats, 
                                 on=['code_module', 'code_presentation'])
    
    # One-hot encode
    module_encoded = pd.get_dummies(course_data['code_module'], prefix='module')
    pres_encoded = pd.get_dummies(course_data['code_presentation'], prefix='pres')
    
    # Numerical features
    num_features = course_data[['module_presentation_length', 
                                 'num_students', 'pass_rate']].values
    
    feature_matrix = np.hstack([module_encoded.values, 
                                pres_encoded.values, 
                                num_features])
    
    return feature_matrix

course_features = create_course_features(courses, student_info)
print(f"Course features shape: {course_features.shape}")
```

**Step 2.3: Assessment Node Features**

```python
def create_assessment_features(assessments):
    """Create assessment node feature matrix"""
    # One-hot encode assessment type
    type_encoded = pd.get_dummies(assessments['assessment_type'], prefix='type')
    
    # Normalize dates
    assessments['date_normalized'] = (
        assessments['date'] / assessments['date'].max()
    )
    
    # Binary features
    assessments['is_final'] = (assessments['assessment_type'] == 'Exam').astype(int)
    
    # Numerical features
    num_features = assessments[['date', 'weight', 'date_normalized', 'is_final']].values
    
    feature_matrix = np.hstack([type_encoded.values, num_features])
    
    return feature_matrix

assessment_features = create_assessment_features(assessments)
print(f"Assessment features shape: {assessment_features.shape}")
```

**Step 2.4: VLE Resource Node Features**

```python
def create_vle_features(vle, student_vle):
    """Create VLE resource node feature matrix"""
    # Calculate usage statistics
    vle_stats = student_vle.groupby('id_site').agg({
        'sum_click': ['mean', 'sum']
    }).reset_index()
    vle_stats.columns = ['id_site', 'avg_clicks', 'total_clicks']
    
    # Merge with VLE metadata
    vle_data = vle.merge(vle_stats, on='id_site', how='left')
    vle_data = vle_data.fillna(0)
    
    # One-hot encode activity type
    type_encoded = pd.get_dummies(vle_data['activity_type'], prefix='activity')
    
    # Numerical features
    vle_data['availability_duration'] = vle_data['week_to'] - vle_data['week_from']
    num_features = vle_data[['week_from', 'week_to', 'avg_clicks', 
                              'total_clicks', 'availability_duration']].values
    
    feature_matrix = np.hstack([type_encoded.values, num_features])
    
    return feature_matrix

vle_features = create_vle_features(vle, student_vle_w8)
print(f"VLE features shape: {vle_features.shape}")
```

### 2.3 Phase 3: Edge Index Construction

**Step 3.1: Create Node ID Mappings**

```python
def create_node_mappings(student_info, courses, assessments, vle):
    """Create mappings from original IDs to graph node indices"""
    # Student nodes: 0 to N_students-1
    student_to_idx = {
        sid: idx for idx, sid in enumerate(student_info['id_student'].unique())
    }
    
    # Course nodes: N_students to N_students+N_courses-1
    course_ids = courses.apply(
        lambda x: f"{x['code_module']}_{x['code_presentation']}", axis=1
    )
    course_to_idx = {
        cid: idx + len(student_to_idx) 
        for idx, cid in enumerate(course_ids.unique())
    }
    
    # Assessment nodes: continue numbering
    offset = len(student_to_idx) + len(course_to_idx)
    assessment_to_idx = {
        aid: idx + offset 
        for idx, aid in enumerate(assessments['id_assessment'].unique())
    }
    
    # VLE nodes: continue numbering
    offset += len(assessment_to_idx)
    vle_to_idx = {
        vid: idx + offset 
        for idx, vid in enumerate(vle['id_site'].unique())
    }
    
    return student_to_idx, course_to_idx, assessment_to_idx, vle_to_idx

mappings = create_node_mappings(student_info, courses, assessments, vle)
print(f"Created mappings for {sum(len(m) for m in mappings)} nodes")
```

**Step 3.2: Student → Course-Presentation Edges**

```python
def create_enrollment_edges(student_info, student_to_idx, course_to_idx):
    """Create enrollment edge index and features"""
    edges = []
    edge_features = []
    
    for _, row in student_info.iterrows():
        student_idx = student_to_idx[row['id_student']]
        course_id = f"{row['code_module']}_{row['code_presentation']}"
        course_idx = course_to_idx[course_id]
        
        edges.append([student_idx, course_idx])
        
        # Edge features (placeholder - would need registration data)
        edge_features.append([0, 0, 0, 1])  # [reg_date, is_early, duration, is_active]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return edge_index, edge_attr

enroll_edges, enroll_features = create_enrollment_edges(
    student_info, mappings[0], mappings[1]
)
print(f"Created {enroll_edges.shape[1]} enrollment edges")
```

**Step 3.3: Student → Assessment Edges**

```python
def create_submission_edges(student_assess, assessments, 
                            student_to_idx, assessment_to_idx, window):
    """Create submission edge index and features"""
    # Merge to get submission dates
    submissions = student_assess.merge(
        assessments[['id_assessment', 'date']], 
        on='id_assessment'
    )
    
    # Filter by window
    submissions = submissions[submissions['date_submitted'] <= window]
    
    edges = []
    edge_features = []
    
    for _, row in submissions.iterrows():
        if row['id_student'] not in student_to_idx:
            continue
        if row['id_assessment'] not in assessment_to_idx:
            continue
            
        student_idx = student_to_idx[row['id_student']]
        assess_idx = assessment_to_idx[row['id_assessment']]
        
        edges.append([student_idx, assess_idx])
        
        # Edge features
        days_late = max(0, row['date_submitted'] - row['date'])
        edge_features.append([
            row['score'],
            row['date_submitted'],
            int(row['is_banked']),
            days_late,
            0  # score_normalized (would calculate)
        ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return edge_index, edge_attr

submit_edges, submit_features = create_submission_edges(
    student_assess, assessments, mappings[0], mappings[2], window=56
)
print(f"Created {submit_edges.shape[1]} submission edges")
```

**Step 3.4: Student → VLE Resource Edges**

```python
def create_interaction_edges(student_vle, student_to_idx, vle_to_idx, window):
    """Create VLE interaction edge index and features"""
    # Filter by window
    interactions = student_vle[student_vle['date'] <= window]
    
    # Aggregate by student-resource pair
    agg_interactions = interactions.groupby(['id_student', 'id_site']).agg({
        'sum_click': 'sum',
        'date': ['min', 'max', 'count']
    }).reset_index()
    
    agg_interactions.columns = ['id_student', 'id_site', 'sum_click', 
                                 'date_first', 'date_last', 'num_days']
    
    edges = []
    edge_features = []
    
    for _, row in agg_interactions.iterrows():
        if row['id_student'] not in student_to_idx:
            continue
        if row['id_site'] not in vle_to_idx:
            continue
            
        student_idx = student_to_idx[row['id_student']]
        vle_idx = vle_to_idx[row['id_site']]
        
        edges.append([student_idx, vle_idx])
        
        # Edge features
        click_freq = row['sum_click'] / max(1, row['num_days'])
        edge_features.append([
            row['sum_click'],
            row['date_first'],
            row['date_last'],
            row['num_days'],
            click_freq
        ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return edge_index, edge_attr

interact_edges, interact_features = create_interaction_edges(
    student_vle, mappings[0], mappings[3], window=56
)
print(f"Created {interact_edges.shape[1]} interaction edges")
```

### 2.4 Phase 4: Heterogeneous Graph Assembly

**Step 4.1: Create PyTorch Geometric HeteroData Object**

```python
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph(node_features, edge_data, labels):
    """Assemble heterogeneous graph"""
    data = HeteroData()
    
    # Add node features
    data['student'].x = torch.tensor(node_features['student'], dtype=torch.float)
    data['student'].y = torch.tensor(labels, dtype=torch.long)
    data['course_presentation'].x = torch.tensor(node_features['course'], dtype=torch.float)
    data['assessment'].x = torch.tensor(node_features['assessment'], dtype=torch.float)
    data['vle_resource'].x = torch.tensor(node_features['vle'], dtype=torch.float)
    
    # Add edges
    data['student', 'enrolled_in', 'course_presentation'].edge_index = edge_data['enroll'][0]
    data['student', 'enrolled_in', 'course_presentation'].edge_attr = edge_data['enroll'][1]
    
    data['student', 'submitted', 'assessment'].edge_index = edge_data['submit'][0]
    data['student', 'submitted', 'assessment'].edge_attr = edge_data['submit'][1]
    
    data['student', 'interacted_with', 'vle_resource'].edge_index = edge_data['interact'][0]
    data['student', 'interacted_with', 'vle_resource'].edge_attr = edge_data['interact'][1]
    
    # Add reverse edges for message passing
    data['course_presentation', 'enrolls', 'student'].edge_index = (
        data['student', 'enrolled_in', 'course_presentation'].edge_index.flip(0)
    )
    
    data['assessment', 'submitted_by', 'student'].edge_index = (
        data['student', 'submitted', 'assessment'].edge_index.flip(0)
    )
    
    data['vle_resource', 'accessed_by', 'student'].edge_index = (
        data['student', 'interacted_with', 'vle_resource'].edge_index.flip(0)
    )
    
    return data

# Build graph
graph = build_hetero_graph(
    node_features={'student': student_features, 'course': course_features,
                   'assessment': assessment_features, 'vle': vle_features},
    edge_data={'enroll': (enroll_edges, enroll_features),
               'submit': (submit_edges, submit_features),
               'interact': (interact_edges, interact_features)},
    labels=student_info['target'].values
)

print(f"Graph constructed: {graph}")
print(f"Node types: {graph.node_types}")
print(f"Edge types: {graph.edge_types}")
```

---

## 3. Baseline GNN Models

### 3.1 Model 1: Heterogeneous Graph Attention Network (HAN)

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class OULAD_HAN(nn.Module):
    """Heterogeneous Graph Attention Network for OULAD"""
    
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Input projections for each node type
        self.student_lin = Linear(-1, hidden_dim)
        self.course_lin = Linear(-1, hidden_dim)
        self.assessment_lin = Linear(-1, hidden_dim)
        self.vle_lin = Linear(-1, hidden_dim)
        
        # Heterogeneous convolution layers
        self.conv1 = HeteroConv({
            ('student', 'enrolled_in', 'course_presentation'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            ('student', 'submitted', 'assessment'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            ('student', 'interacted_with', 'vle_resource'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            ('course_presentation', 'enrolls', 'student'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            ('assessment', 'submitted_by', 'student'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            ('vle_resource', 'accessed_by', 'student'): 
                GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('student', 'enrolled_in', 'course_presentation'): 
                GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout),
            ('student', 'submitted', 'assessment'): 
                GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout),
            ('student', 'interacted_with', 'vle_resource'): 
                GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout),
        }, aggr='mean')
        
        # Classifier
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 2)  # Binary classification
        )
    
    def forward(self, x_dict, edge_index_dict):
        # Project node features
        x_dict = {
            'student': self.student_lin(x_dict['student']),
            'course_presentation': self.course_lin(x_dict['course_presentation']),
            'assessment': self.assessment_lin(x_dict['assessment']),
            'vle_resource': self.vle_lin(x_dict['vle_resource']),
        }
        
        # First convolution
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        # Second convolution
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # Classify student nodes
        return self.classifier(x_dict['student'])
```

### 3.2 Model 2: Relational Graph Convolutional Network (R-GCN)

```python
from torch_geometric.nn import RGCNConv

class OULAD_RGCN(nn.Module):
    """Relational GCN for OULAD"""
    
    def __init__(self, in_channels, hidden_dim=64, num_relations=6, dropout=0.3):
        super().__init__()
        
        self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return self.classifier(x)
```

### 3.3 Model 3: Graph Transformer

```python
from torch_geometric.nn import TransformerConv

class OULAD_GraphTransformer(nn.Module):
    """Graph Transformer for OULAD"""
    
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.convs = nn.ModuleList([
            HeteroConv({
                ('student', 'enrolled_in', 'course_presentation'): 
                    TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
                ('student', 'submitted', 'assessment'): 
                    TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
                ('student', 'interacted_with', 'vle_resource'): 
                    TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout),
            }, aggr='mean')
            for _ in range(num_layers)
        ])
        
        self.classifier = Linear(hidden_dim * num_heads, 2)
    
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        return self.classifier(x_dict['student'])
```

---

## 4. Training and Evaluation

### 4.1 Training Loop

```python
def train_gnn(model, data, optimizer, criterion):
    """Train GNN for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x_dict, data.edge_index_dict)
    
    # Compute loss (only on student nodes)
    loss = criterion(out, data['student'].y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_gnn(model, data):
    """Evaluate GNN"""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=1)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, f1_score
        
        y_true = data['student'].y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        
        auroc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        
        return auroc, f1

# Training
model = OULAD_HAN(hidden_dim=64, num_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    loss = train_gnn(model, graph, optimizer, criterion)
    
    if epoch % 10 == 0:
        auroc, f1 = evaluate_gnn(model, graph)
        print(f"Epoch {epoch}: Loss={loss:.4f}, AUROC={auroc:.4f}, F1={f1:.4f}")
```

### 4.2 Evaluation Strategy

**Comparison with Traditional ML**:

| Model Type | AUROC (Expected) | Advantages | Disadvantages |
|------------|------------------|------------|---------------|
| LightGBM (baseline) | 0.835 | Fast, interpretable | No relational learning |
| HAN (GNN) | 0.840-0.850 | Relational learning | Slower, more complex |
| R-GCN (GNN) | 0.830-0.845 | Handles multiple relations | Memory intensive |
| Graph Transformer | 0.845-0.860 | Attention mechanism | Very slow |

**Evaluation Splits**:
1. Random split (baseline)
2. LCPO (cross-course generalization)
3. Future-presentation (temporal generalization)

---

## 5. Implementation Timeline

### Week 1-2: Infrastructure Setup
- [ ] Set up PyTorch Geometric environment
- [ ] Implement graph construction pipeline
- [ ] Create data loaders for temporal windows
- [ ] Validate graph statistics

### Week 3-4: Baseline Models
- [ ] Implement HAN model
- [ ] Implement R-GCN model
- [ ] Train on random split
- [ ] Compare with LightGBM baseline

### Week 5-6: Advanced Models
- [ ] Implement Graph Transformer
- [ ] Experiment with attention mechanisms
- [ ] Hyperparameter tuning
- [ ] Ensemble methods

### Week 7-8: Evaluation & Analysis
- [ ] LCPO evaluation
- [ ] Future-presentation evaluation
- [ ] Ablation studies
- [ ] Interpretability analysis

---

## 6. Expected Outcomes

### 6.1 Performance Targets

**Minimum Viable Performance**:
- AUROC ≥ 0.835 (match LightGBM baseline)
- F1 ≥ 0.788
- LCPO AUROC ≥ 0.804

**Stretch Goals**:
- AUROC ≥ 0.850 (+1.5% over baseline)
- Improved LCPO generalization (≥ 0.820)
- Interpretable attention weights

### 6.2 Research Questions

1. **Do GNNs outperform traditional ML on OULAD?**
   - Hypothesis: Yes, by 1-2% AUROC due to relational learning

2. **Which graph structure is most effective?**
   - Test: Compare different edge types and aggregation strategies

3. **How important are different node/edge types?**
   - Method: Ablation studies removing each type

4. **Can GNNs improve cross-course generalization?**
   - Test: Compare LCPO performance with traditional ML

5. **Are attention weights interpretable?**
   - Method: Visualize attention on successful/failed students

---

## 7. Challenges and Mitigation

### 7.1 Scalability Challenges

**Challenge**: 10M edges may not fit in GPU memory

**Mitigation**:
- Mini-batch training with neighbor sampling
- GraphSAINT sampling strategy
- Cluster-GCN for large graphs
- Use multiple GPUs if available

### 7.2 Temporal Leakage

**Challenge**: Ensuring no future information in graph

**Mitigation**:
- Strict temporal filtering in graph construction
- Separate graphs for each prediction window
- Validation checks for edge timestamps

### 7.3 Class Imbalance

**Challenge**: 52.8% at-risk vs 47.2% success

**Mitigation**:
- Weighted loss function
- Focal loss for hard examples
- AUPRC as primary metric

### 7.4 Interpretability

**Challenge**: GNNs are less interpretable than tree models

**Mitigation**:
- Attention weight visualization
- GNNExplainer for subgraph explanations
- Compare with SHAP values from LightGBM

---

## 8. Success Criteria

### 8.1 Technical Criteria

- [ ] Graph construction pipeline complete and tested
- [ ] At least 2 GNN models implemented
- [ ] Performance ≥ LightGBM baseline
- [ ] LCPO evaluation complete
- [ ] Code documented and reproducible

### 8.2 Research Criteria

- [ ] Quantify GNN advantage over traditional ML
- [ ] Identify most important graph structures
- [ ] Demonstrate interpretability
- [ ] Document lessons learned

### 8.3 Deliverables

- [ ] Graph construction code
- [ ] Trained GNN models
- [ ] Evaluation results
- [ ] Technical report
- [ ] Presentation slides

---

## 9. Conclusion

This plan provides a comprehensive roadmap for constructing heterogeneous graphs from OULAD and developing GNN-based student success prediction models. The proposed approach leverages rich relational information that traditional ML models cannot capture, with potential for 1-2% performance improvement and better interpretability through attention mechanisms.

**Key Innovations**:
1. Heterogeneous graph with 4 node types and 5 edge types
2. Temporal filtering for leakage-safe predictions
3. Multiple GNN architectures for comparison
4. Comprehensive evaluation including LCPO

**Next Steps**:
1. Begin implementation of graph construction pipeline
2. Set up PyTorch Geometric environment
3. Implement and test HAN model
4. Compare with LightGBM baseline

This work will establish whether GNNs provide meaningful improvements for student success prediction and lay the foundation for more advanced graph-based approaches.

---

**Document Version**: 1.0  
**Last Updated**: June 20, 2026  
**Contact**: BioAI Systems Lab