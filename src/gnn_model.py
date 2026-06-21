"""
OULAD Graph Neural Network Implementation
Heterogeneous graph-based model for at-risk student prediction

This module implements a GNN architecture for OULAD using PyTorch Geometric.
Based on the graph schema defined in docs/INITIAL_GRAPH_CONSTRUCTION_PLAN.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for OULAD

    Node types:
    - student: Student nodes with demographic features
    - course: Course-presentation nodes
    - assessment: Assessment nodes
    - vle_resource: VLE resource nodes

    Edge types:
    - (student, enrolled_in, course)
    - (student, interacts_with, vle_resource)
    - (student, submits, assessment)
    - (course, contains, assessment)
    - (course, has_resource, vle_resource)
    - And reverse edges
    """

    def __init__(
        self,
        metadata: Tuple,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = False,
    ):
        """
        Initialize Heterogeneous GNN

        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Use GAT instead of GraphSAGE
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Choose convolution type
        if use_attention:
            conv_class = lambda in_ch, out_ch: GATConv(
                in_ch, out_ch, heads=4, concat=False
            )
        else:
            conv_class = lambda in_ch, out_ch: SAGEConv(in_ch, out_ch)

        # Create heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type

                if i == 0:
                    # First layer: use actual input dimensions (will be set dynamically)
                    conv_dict[edge_type] = conv_class(-1, hidden_channels)
                else:
                    conv_dict[edge_type] = conv_class(hidden_channels, hidden_channels)

            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Batch normalization for each node type
        self.batch_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.BatchNorm1d(hidden_channels)
                        for node_type in metadata[0]
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Final classifier for student nodes
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2),  # Binary classification
        )

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}

        Returns:
            Logits for student nodes
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            # Heterogeneous convolution
            x_dict = conv(x_dict, edge_index_dict)

            # Apply batch norm and activation
            x_dict = {
                key: F.relu(self.batch_norms[i][key](x)) for key, x in x_dict.items()
            }

            # Dropout
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Classification on student nodes only
        student_embeddings = x_dict["student"]
        logits = self.classifier(student_embeddings)

        return logits

    def get_embeddings(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Get node embeddings after GNN layers"""
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.relu(self.batch_norms[i][key](x)) for key, x in x_dict.items()
            }
        return x_dict


class GraphConstructor:
    """Construct heterogeneous graph from OULAD data"""

    def __init__(self, prediction_week: int = 8):
        """
        Initialize graph constructor

        Args:
            prediction_week: Week for temporal filtering (leakage prevention)
        """
        self.prediction_week = prediction_week
        self.window_days = prediction_week * 7

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load OULAD datasets"""
        print(f"Loading OULAD data for week {self.prediction_week}...")

        data = {
            "student_info": pd.read_csv(DATA_DIR / "studentInfo.csv"),
            "student_vle": pd.read_csv(DATA_DIR / "studentVle.csv"),
            "student_assessment": pd.read_csv(DATA_DIR / "studentAssessment.csv"),
            "assessments": pd.read_csv(DATA_DIR / "assessments.csv"),
            "vle": pd.read_csv(DATA_DIR / "vle.csv"),
            "courses": pd.read_csv(DATA_DIR / "courses.csv"),
        }

        # Apply label mapping
        data["student_info"]["target"] = data["student_info"]["final_result"].apply(
            lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
        )

        # Temporal filtering (leakage prevention)
        data["student_vle"] = data["student_vle"][
            data["student_vle"]["date"] <= self.window_days
        ]

        assess_with_dates = data["student_assessment"].merge(
            data["assessments"][["id_assessment", "date"]],
            on="id_assessment",
            how="left",
        )
        data["student_assessment"] = assess_with_dates[
            assess_with_dates["date"] <= self.window_days
        ]

        print(f"✓ Data loaded and filtered to week {self.prediction_week}")
        return data

    def create_node_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, torch.Tensor]:
        """Create node features for all node types"""
        print("Creating node features...")

        node_features = {}

        # 1. Student node features
        student_df = data["student_info"].copy()

        # Demographic features (one-hot encoded)
        categorical_cols = [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "disability",
        ]
        student_encoded = pd.get_dummies(student_df[categorical_cols], drop_first=True)

        # Numeric features
        numeric_features = student_df[
            ["num_of_prev_attempts", "studied_credits"]
        ].fillna(0)

        # Combine
        student_features = pd.concat([student_encoded, numeric_features], axis=1)
        node_features["student"] = torch.FloatTensor(student_features.values)

        # 2. Course node features
        course_df = data["courses"].copy()
        course_features = pd.get_dummies(course_df[["code_module"]], drop_first=True)
        course_features["length"] = course_df["module_presentation_length"].fillna(0)
        node_features["course"] = torch.FloatTensor(course_features.values)

        # 3. Assessment node features
        assessment_df = data["assessments"].copy()
        assessment_encoded = pd.get_dummies(
            assessment_df[["assessment_type"]], drop_first=True
        )
        assessment_features = pd.concat(
            [assessment_encoded, assessment_df[["weight", "date"]].fillna(0)], axis=1
        )
        node_features["assessment"] = torch.FloatTensor(assessment_features.values)

        # 4. VLE resource node features
        vle_df = data["vle"].copy()
        vle_encoded = pd.get_dummies(vle_df[["activity_type"]], drop_first=True)
        vle_features = pd.concat(
            [vle_encoded, vle_df[["week_from", "week_to"]].fillna(0)], axis=1
        )
        node_features["vle_resource"] = torch.FloatTensor(vle_features.values)

        print(f"✓ Created features for {len(node_features)} node types")
        for node_type, features in node_features.items():
            print(f"  {node_type}: {features.shape}")

        return node_features

    def create_node_mappings(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Create mappings from original IDs to graph node indices"""
        print("Creating node ID mappings...")

        mappings = {}

        # Student mapping
        student_ids = data["student_info"]["id_student"].unique()
        mappings["student"] = {sid: idx for idx, sid in enumerate(student_ids)}

        # Course mapping (code_module + code_presentation)
        courses = data["courses"][
            ["code_module", "code_presentation"]
        ].drop_duplicates()
        course_ids = [
            f"{row['code_module']}_{row['code_presentation']}"
            for _, row in courses.iterrows()
        ]
        mappings["course"] = {cid: idx for idx, cid in enumerate(course_ids)}

        # Assessment mapping
        assessment_ids = data["assessments"]["id_assessment"].unique()
        mappings["assessment"] = {aid: idx for idx, aid in enumerate(assessment_ids)}

        # VLE resource mapping
        vle_ids = data["vle"]["id_site"].unique()
        mappings["vle_resource"] = {vid: idx for idx, vid in enumerate(vle_ids)}

        print(f"✓ Created mappings:")
        for node_type, mapping in mappings.items():
            print(f"  {node_type}: {len(mapping)} nodes")

        return mappings

    def create_edges(
        self, data: Dict[str, pd.DataFrame], mappings: Dict[str, Dict]
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Create edge indices for all edge types"""
        print("Creating edges...")

        edge_index_dict = {}

        # 1. Student -> Course (enrolled_in)
        student_course = data["student_info"][
            ["id_student", "code_module", "code_presentation"]
        ].copy()
        student_course["course_id"] = (
            student_course["code_module"] + "_" + student_course["code_presentation"]
        )

        src = [
            mappings["student"][sid]
            for sid in student_course["id_student"]
            if sid in mappings["student"]
        ]
        dst = [
            mappings["course"][cid]
            for cid in student_course["course_id"]
            if cid in mappings["course"]
        ]

        edge_index_dict[("student", "enrolled_in", "course")] = torch.tensor(
            [src, dst], dtype=torch.long
        )
        edge_index_dict[("course", "rev_enrolled_in", "student")] = torch.tensor(
            [dst, src], dtype=torch.long
        )

        # 2. Student -> VLE Resource (interacts_with)
        student_vle = data["student_vle"][["id_student", "id_site"]].copy()

        src = [
            mappings["student"][sid]
            for sid in student_vle["id_student"]
            if sid in mappings["student"]
        ]
        dst = [
            mappings["vle_resource"][vid]
            for vid in student_vle["id_site"]
            if vid in mappings["vle_resource"]
        ]

        edge_index_dict[("student", "interacts_with", "vle_resource")] = torch.tensor(
            [src, dst], dtype=torch.long
        )
        edge_index_dict[("vle_resource", "rev_interacts_with", "student")] = (
            torch.tensor([dst, src], dtype=torch.long)
        )

        # 3. Student -> Assessment (submits)
        student_assess = data["student_assessment"][
            ["id_student", "id_assessment"]
        ].copy()

        src = [
            mappings["student"][sid]
            for sid in student_assess["id_student"]
            if sid in mappings["student"]
        ]
        dst = [
            mappings["assessment"][aid]
            for aid in student_assess["id_assessment"]
            if aid in mappings["assessment"]
        ]

        edge_index_dict[("student", "submits", "assessment")] = torch.tensor(
            [src, dst], dtype=torch.long
        )
        edge_index_dict[("assessment", "rev_submits", "student")] = torch.tensor(
            [dst, src], dtype=torch.long
        )

        # 4. Course -> Assessment (contains)
        course_assess = data["assessments"][
            ["code_module", "code_presentation", "id_assessment"]
        ].copy()
        course_assess["course_id"] = (
            course_assess["code_module"] + "_" + course_assess["code_presentation"]
        )

        src = [
            mappings["course"][cid]
            for cid in course_assess["course_id"]
            if cid in mappings["course"]
        ]
        dst = [
            mappings["assessment"][aid]
            for aid in course_assess["id_assessment"]
            if aid in mappings["assessment"]
        ]

        edge_index_dict[("course", "contains", "assessment")] = torch.tensor(
            [src, dst], dtype=torch.long
        )
        edge_index_dict[("assessment", "rev_contains", "course")] = torch.tensor(
            [dst, src], dtype=torch.long
        )

        # 5. Course -> VLE Resource (has_resource)
        course_vle = data["vle"][["code_module", "code_presentation", "id_site"]].copy()
        course_vle["course_id"] = (
            course_vle["code_module"] + "_" + course_vle["code_presentation"]
        )

        src = [
            mappings["course"][cid]
            for cid in course_vle["course_id"]
            if cid in mappings["course"]
        ]
        dst = [
            mappings["vle_resource"][vid]
            for vid in course_vle["id_site"]
            if vid in mappings["vle_resource"]
        ]

        edge_index_dict[("course", "has_resource", "vle_resource")] = torch.tensor(
            [src, dst], dtype=torch.long
        )
        edge_index_dict[("vle_resource", "rev_has_resource", "course")] = torch.tensor(
            [dst, src], dtype=torch.long
        )

        print(f"✓ Created {len(edge_index_dict)} edge types:")
        for edge_type, edge_index in edge_index_dict.items():
            print(f"  {edge_type}: {edge_index.shape[1]} edges")

        return edge_index_dict

    def create_labels(
        self, data: Dict[str, pd.DataFrame], mappings: Dict[str, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create labels and masks for student nodes"""
        num_students = len(mappings["student"])

        # Initialize labels
        labels = torch.zeros(num_students, dtype=torch.long)

        # Fill in labels
        student_info = data["student_info"]
        for _, row in student_info.iterrows():
            if row["id_student"] in mappings["student"]:
                idx = mappings["student"][row["id_student"]]
                labels[idx] = row["target"]

        # Create train/val/test masks (80/10/10 split)
        num_train = int(0.8 * num_students)
        num_val = int(0.1 * num_students)

        indices = torch.randperm(num_students)

        train_mask = torch.zeros(num_students, dtype=torch.bool)
        val_mask = torch.zeros(num_students, dtype=torch.bool)
        test_mask = torch.zeros(num_students, dtype=torch.bool)

        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train : num_train + num_val]] = True
        test_mask[indices[num_train + num_val :]] = True

        return labels, train_mask, val_mask, test_mask

    def construct_graph(self) -> HeteroData:
        """Construct complete heterogeneous graph"""
        print("=" * 80)
        print("CONSTRUCTING HETEROGENEOUS GRAPH")
        print("=" * 80)

        # Load data
        data = self.load_data()

        # Create node features
        node_features = self.create_node_features(data)

        # Create node mappings
        mappings = self.create_node_mappings(data)

        # Create edges
        edge_index_dict = self.create_edges(data, mappings)

        # Create labels and masks
        labels, train_mask, val_mask, test_mask = self.create_labels(data, mappings)

        # Create HeteroData object
        graph = HeteroData()

        # Add node features
        for node_type, features in node_features.items():
            graph[node_type].x = features

        # Add edges
        for edge_type, edge_index in edge_index_dict.items():
            graph[edge_type].edge_index = edge_index

        # Add labels and masks (only for student nodes)
        graph["student"].y = labels
        graph["student"].train_mask = train_mask
        graph["student"].val_mask = val_mask
        graph["student"].test_mask = test_mask

        print("\n" + "=" * 80)
        print("GRAPH CONSTRUCTION COMPLETE")
        print("=" * 80)
        print(f"\nGraph Statistics:")
        print(f"  Node types: {len(graph.node_types)}")
        print(f"  Edge types: {len(graph.edge_types)}")
        print(f"  Total nodes: {sum(graph[nt].num_nodes for nt in graph.node_types)}")
        print(f"  Total edges: {sum(graph[et].num_edges for et in graph.edge_types)}")
        print(f"\nStudent nodes:")
        print(f"  Total: {graph['student'].num_nodes}")
        print(f"  Train: {train_mask.sum().item()}")
        print(f"  Val: {val_mask.sum().item()}")
        print(f"  Test: {test_mask.sum().item()}")
        print(f"  At-risk rate: {labels.float().mean():.3f}")

        return graph, mappings


def train_epoch(model, graph, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(graph.x_dict, graph.edge_index_dict)

    # Calculate loss on training nodes only
    loss = criterion(
        out[graph["student"].train_mask],
        graph["student"].y[graph["student"].train_mask],
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, graph, mask):
    """Evaluate model"""
    model.eval()

    out = model(graph.x_dict, graph.edge_index_dict)
    pred = out.argmax(dim=1)

    correct = (pred[mask] == graph["student"].y[mask]).sum()
    acc = correct / mask.sum()

    return acc.item()


def main():
    """Main execution function"""
    print("=" * 80)
    print("OULAD GRAPH NEURAL NETWORK")
    print("=" * 80)

    # Construct graph
    constructor = GraphConstructor(prediction_week=8)
    graph, mappings = constructor.construct_graph()

    # Initialize model
    print("\nInitializing GNN model...")
    model = HeteroGNN(
        metadata=(graph.node_types, graph.edge_types),
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        use_attention=False,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining GNN...")
    num_epochs = 100
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, graph, optimizer, criterion)

        if epoch % 10 == 0:
            train_acc = evaluate(model, graph, graph["student"].train_mask)
            val_acc = evaluate(model, graph, graph["student"].val_mask)
            test_acc = evaluate(model, graph, graph["student"].test_mask)

            print(
                f"Epoch {epoch:03d}: Loss={loss:.4f}, "
                f"Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model
                torch.save(model.state_dict(), RESULTS_DIR / "gnn" / "best_model.pt")

    # Final evaluation
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(RESULTS_DIR / "gnn" / "best_model.pt"))
    test_acc = evaluate(model, graph, graph["student"].test_mask)
    print(f"Final test accuracy: {test_acc:.4f}")

    return model, graph, mappings


if __name__ == "__main__":
    # Create output directory
    (RESULTS_DIR / "gnn").mkdir(parents=True, exist_ok=True)

    model, graph, mappings = main()

# Made with Bob
