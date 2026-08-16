# NOTE: Prediction unit is the enrolled_in edge (one per enrollment, 32,593 total)
# Each enrolled_in edge connects a student node to a course_presentation node and
# carries one binary label (target: 1 = at-risk, 0 = success). The GNN predicts
# at-risk probability per edge rather than per student, so students with multiple
# enrollments each get an independent prediction without label ambiguity.

import os
import glob as _glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter as pyg_scatter
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
import copy

ARTIFACT_DIR = "results/graph/artifacts"
EVAL_DIR = "results/graph/evaluation"

SEED = 42


# ---------------------------------------------------------------------------
# Helper — one-hot encode a categorical Series into a float tensor
# ---------------------------------------------------------------------------

def _onehot(series: pd.Series) -> torch.Tensor:
    le = LabelEncoder()
    codes = le.fit_transform(series.fillna("Unknown").astype(str))
    n_classes = len(le.classes_)
    t = torch.zeros(len(codes), n_classes, dtype=torch.float32)
    t[torch.arange(len(codes)), torch.tensor(codes)] = 1.0
    return t


def _numeric(series: pd.Series) -> torch.Tensor:
    return torch.tensor(series.fillna(0).values, dtype=torch.float32).unsqueeze(1)


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def _normalize_numeric_features(data: HeteroData) -> HeteroData:
    """Standardize numeric columns in node and edge feature tensors in-place.

    One-hot columns (all values in {0, 1} with at most 2 unique values per
    column) are left untouched.  For total_clicks and n_interactions a log1p
    transform is applied before standardizing.

    Means and stds are computed over the full graph (not per split).
    Columns with zero variance are left as-is (already constant).
    """
    # log1p column name → index mapping is built per tensor, so we identify
    # them by matching the order they were concatenated in load().
    # For interacted_with edge attrs the columns are:
    #   [total_clicks, n_interactions, first_day, last_day, active_days]
    #   indices         0               1
    IW_LOG1P_COLS = {0, 1}  # total_clicks, n_interactions

    def _is_onehot_col(col: torch.Tensor) -> bool:
        """True if every value is 0 or 1 and there are at most 2 unique values."""
        unique_vals = col.unique()
        if unique_vals.numel() > 2:
            return False
        return bool(((unique_vals == 0) | (unique_vals == 1)).all())

    def _standardize(tensor: torch.Tensor, log1p_col_indices: set = None) -> torch.Tensor:
        """Return a copy of tensor with numeric columns standardized."""
        tensor = tensor.clone()
        n_cols = tensor.shape[1]
        for c in range(n_cols):
            col = tensor[:, c]
            if _is_onehot_col(col):
                continue
            if log1p_col_indices and c in log1p_col_indices:
                col = torch.log1p(col.clamp(min=0.0))
            mean = col.mean()
            std = col.std(unbiased=False)
            if std.item() == 0.0:
                tensor[:, c] = 0.0
            else:
                tensor[:, c] = (col - mean) / std
        return tensor

    # Node features
    for ntype in data.node_types:
        store = data[ntype]
        if hasattr(store, "x") and store.x is not None and store.x.numel() > 0 and store.x.shape[1] > 0:
            store.x = _standardize(store.x)

    # Edge features
    for etype in data.edge_types:
        store = data[etype]
        if hasattr(store, "edge_attr") and store.edge_attr is not None and store.edge_attr.numel() > 0:
            log1p_cols = IW_LOG1P_COLS if etype == ("student", "interacted_with", "vle_resource") else None
            store.edge_attr = _standardize(store.edge_attr, log1p_col_indices=log1p_cols)

    return data


# ---------------------------------------------------------------------------
# GraphDataLoader
# ---------------------------------------------------------------------------

class GraphDataLoader:
    """Loads all parquet artifacts for a given week and returns a HeteroData object.

    Node features are derived from the parquet columns:
    - Categorical columns → one-hot encoded
    - Numeric columns → raw float, shape (N, 1)
    Features for each node type are concatenated into a single x tensor.

    Edge features are stored in `edge_attr` tensors.

    The enrolled_in edge additionally receives:
    - `y`: binary target labels (from week{N}_enrollments.parquet)
    - `edge_index_in_enrollments`: integer row positions (0..32592) used by
      load_split_masks to align the split file to edges.
    """

    def __init__(self, week: int, artifact_dir: str = ARTIFACT_DIR):
        self.week = week
        self.prefix = os.path.join(artifact_dir, f"week{week:02d}")

    def _path(self, suffix: str) -> str:
        return f"{self.prefix}_{suffix}.parquet"

    def load(self) -> HeteroData:
        data = HeteroData()

        # ---- Node tables ----

        # student
        st = pd.read_parquet(self._path("nodes_student"))
        st_cat = ["gender", "region", "highest_education", "imd_band", "disability"]
        data["student"].x = torch.cat(
            [_onehot(st[c]) for c in st_cat], dim=1
        )  # shape (N_student, sum_of_onehot_dims)
        data["student"].node_id = torch.tensor(st["id_student"].values, dtype=torch.long)

        # course_presentation
        cp = pd.read_parquet(self._path("nodes_course_presentation"))
        cp_cat = ["code_module", "code_presentation"]
        cp_num = _numeric(cp["module_presentation_length"])
        data["course_presentation"].x = torch.cat(
            [_onehot(cp[c]) for c in cp_cat] + [cp_num], dim=1
        )
        data["course_presentation"].node_id = torch.arange(len(cp), dtype=torch.long)

        # assessment  — may have 0 rows at Week 2
        ass = pd.read_parquet(self._path("nodes_assessment"))
        if len(ass) > 0:
            ass_cat = ["assessment_type"]
            ass_num = [_numeric(ass["weight"]), _numeric(ass["date"])]
            data["assessment"].x = torch.cat(
                [_onehot(ass[c]) for c in ass_cat] + ass_num, dim=1
            )
        else:
            # Placeholder feature tensor with 1 dummy feature so downstream
            # linear layers don't fail on empty node sets.
            data["assessment"].x = torch.zeros((0, 1), dtype=torch.float32)
        data["assessment"].node_id = torch.arange(len(ass), dtype=torch.long)

        # vle_resource
        vle = pd.read_parquet(self._path("nodes_vle_resource"))
        vle_cat = ["activity_type", "code_module", "code_presentation"]
        vle_num = [_numeric(vle["week_from"]), _numeric(vle["week_to"])]
        data["vle_resource"].x = torch.cat(
            [_onehot(vle[c]) for c in vle_cat] + vle_num, dim=1
        )
        data["vle_resource"].node_id = torch.arange(len(vle), dtype=torch.long)

        # ---- Enrollment labels ----
        enroll = pd.read_parquet(self._path("enrollments"))
        labels = torch.tensor(enroll["target"].values, dtype=torch.float32)

        # ---- Edge tables ----

        # enrolled_in  (student → course_presentation)
        ei = pd.read_parquet(self._path("edges_enrolled_in"))
        ei_src = torch.tensor(ei["src"].values, dtype=torch.long)
        ei_dst = torch.tensor(ei["dst"].values, dtype=torch.long)
        # enrolled_in edge attributes: num_of_prev_attempts, studied_credits (numeric)
        # age_band (categorical → one-hot)
        ei_age = _onehot(ei["age_band"])
        ei_num = torch.cat(
            [_numeric(ei["num_of_prev_attempts"]), _numeric(ei["studied_credits"])], dim=1
        )
        ei_attr = torch.cat([ei_age, ei_num], dim=1)

        ei_key = ("student", "enrolled_in", "course_presentation")
        data[ei_key].edge_index = torch.stack([ei_src, ei_dst], dim=0)
        data[ei_key].edge_attr = ei_attr
        data[ei_key].y = labels
        # Row order of enrolled_in == row order of enrollments.parquet (both from studentInfo)
        data[ei_key].enrollment_idx = torch.arange(len(ei), dtype=torch.long)

        # Reverse edge: course_presentation → student (lets student nodes receive messages)
        rev_ei_key = ("course_presentation", "rev_enrolled_in", "student")
        data[rev_ei_key].edge_index = torch.stack([ei_dst, ei_src], dim=0)

        # contains_assess  (course_presentation → assessment)
        ca = pd.read_parquet(self._path("edges_contains_assess"))
        if len(ca) > 0:
            ca_key = ("course_presentation", "contains_assess", "assessment")
            data[ca_key].edge_index = torch.tensor(
                np.stack([ca["src"].values, ca["dst"].values], axis=0), dtype=torch.long
            )
            # Reverse: assessment → course_presentation
            rev_ca_key = ("assessment", "rev_contains_assess", "course_presentation")
            data[rev_ca_key].edge_index = torch.tensor(
                np.stack([ca["dst"].values, ca["src"].values], axis=0), dtype=torch.long
            )

        # has_resource  (course_presentation → vle_resource)
        hr = pd.read_parquet(self._path("edges_has_resource"))
        hr_key = ("course_presentation", "has_resource", "vle_resource")
        data[hr_key].edge_index = torch.tensor(
            np.stack([hr["src"].values, hr["dst"].values], axis=0), dtype=torch.long
        )
        # Reverse: vle_resource → course_presentation
        rev_hr_key = ("vle_resource", "rev_has_resource", "course_presentation")
        data[rev_hr_key].edge_index = torch.tensor(
            np.stack([hr["dst"].values, hr["src"].values], axis=0), dtype=torch.long
        )

        # submitted  (student → assessment)
        sub = pd.read_parquet(self._path("edges_submitted"))
        if len(sub) > 0:
            sub_key = ("student", "submitted", "assessment")
            data[sub_key].edge_index = torch.tensor(
                np.stack([sub["src"].values, sub["dst"].values], axis=0), dtype=torch.long
            )
            data[sub_key].edge_attr = _numeric(sub["score"])
            # Reverse: assessment → student
            rev_sub_key = ("assessment", "rev_submitted", "student")
            data[rev_sub_key].edge_index = torch.tensor(
                np.stack([sub["dst"].values, sub["src"].values], axis=0), dtype=torch.long
            )

        # interacted_with  (student → vle_resource)
        iw = pd.read_parquet(self._path("edges_interacted_with"))
        iw_num_cols = ["total_clicks", "n_interactions", "first_day", "last_day", "active_days"]
        iw_key = ("student", "interacted_with", "vle_resource")
        data[iw_key].edge_index = torch.tensor(
            np.stack([iw["src"].values, iw["dst"].values], axis=0), dtype=torch.long
        )
        data[iw_key].edge_attr = torch.cat(
            [_numeric(iw[c]) for c in iw_num_cols], dim=1
        )
        # Reverse: vle_resource → student
        rev_iw_key = ("vle_resource", "rev_interacted_with", "student")
        data[rev_iw_key].edge_index = torch.tensor(
            np.stack([iw["dst"].values, iw["src"].values], axis=0), dtype=torch.long
        )

        data = _normalize_numeric_features(data)
        return data


# ---------------------------------------------------------------------------
# load_split_masks
# ---------------------------------------------------------------------------

def load_split_masks(
    week: int,
    split_type: str = "random",
    eval_dir: str = EVAL_DIR,
):
    """Return (train_mask, val_mask, test_mask) boolean tensors over 32,593 enrollments.

    Parameters
    ----------
    week:       int — prediction week (2, 4, 6, 8)
    split_type: "random" or "lcpo"
    eval_dir:   base directory for evaluation split files

    The random split parquet has columns is_train / is_val / is_test (bool).
    Row order matches week{N}_enrollments.parquet (both derived from studentInfo.csv).
    """
    w = f"week{week:02d}"
    split_dir = os.path.join(eval_dir, w, "splits")

    if split_type == "random":
        path = os.path.join(split_dir, f"{w}_random_split.parquet")
        sp = pd.read_parquet(path)
        train_mask = torch.tensor(sp["is_train"].values, dtype=torch.bool)
        val_mask = torch.tensor(sp["is_val"].values, dtype=torch.bool)
        test_mask = torch.tensor(sp["is_test"].values, dtype=torch.bool)
    elif split_type == "lcpo":
        path = os.path.join(split_dir, f"{w}_lcpo_folds.csv")
        sp = pd.read_csv(path)
        # LCPO CSV has a 'fold' column; use fold 0 as test, rest as train, no val
        if "fold" not in sp.columns:
            raise ValueError(f"LCPO file {path} has no 'fold' column")
        test_fold = sp["fold"].min()
        test_mask = torch.tensor((sp["fold"] == test_fold).values, dtype=torch.bool)
        val_mask = torch.zeros(len(sp), dtype=torch.bool)
        train_mask = ~test_mask
    else:
        raise ValueError(f"Unknown split_type '{split_type}'. Use 'random' or 'lcpo'.")

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# EnrollmentGNN
# ---------------------------------------------------------------------------

class EnrollmentGNN(nn.Module):
    """Heterogeneous GNN with an edge-level prediction head on enrolled_in edges.

    Architecture:
    - Two rounds of HeteroConv (wrapping SAGEConv per edge type)
    - Edge representation = concat(src_embedding, dst_embedding) for enrolled_in
    - Linear output head → scalar logit → sigmoid → at-risk probability
    """

    def __init__(self, in_channels_dict: dict, hidden_dim: int = 64, out_dim: int = 1, n_enrolled_in_attr: int = 0):
        super().__init__()
        torch.manual_seed(SEED)

        # Build two HeteroConv layers.  SAGEConv expects (in_channels, out_channels).
        # Layer 1: heterogeneous in → hidden
        conv1_dict = {}
        conv2_dict = {}

        # Forward edge types present in every week
        base_edge_types = [
            ("student", "enrolled_in", "course_presentation"),
            ("course_presentation", "has_resource", "vle_resource"),
            ("student", "interacted_with", "vle_resource"),
            # Reverse edges to ensure all node types receive messages
            ("course_presentation", "rev_enrolled_in", "student"),
            ("vle_resource", "rev_has_resource", "course_presentation"),
            ("vle_resource", "rev_interacted_with", "student"),
        ]
        # Edge types present only when assessments exist
        assess_edge_types = [
            ("course_presentation", "contains_assess", "assessment"),
            ("student", "submitted", "assessment"),
            ("assessment", "rev_contains_assess", "course_presentation"),
            ("assessment", "rev_submitted", "student"),
        ]

        all_edge_types = base_edge_types + assess_edge_types

        for et in all_edge_types:
            src_type, _, dst_type = et
            in_src = in_channels_dict.get(src_type, hidden_dim)
            in_dst = in_channels_dict.get(dst_type, hidden_dim)
            conv1_dict[et] = SAGEConv((in_src, in_dst), hidden_dim)
            conv2_dict[et] = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

        self.conv1 = HeteroConv(conv1_dict, aggr="sum")
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")

        self.act = nn.ReLU()

        # Projects enrolled_in edge attributes into hidden_dim space so they can
        # be added to student representations between layer 1 and layer 2.
        self.enrollment_attr_proj = nn.Linear(n_enrolled_in_attr, hidden_dim) if n_enrolled_in_attr > 0 else None

        # Edge-level prediction head: concat student + course_presentation embeddings
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _fill_missing(self, h_dict: dict, x_dict: dict, hidden_dim: int) -> dict:
        """Ensure every node type has a hidden representation (zeros if unreached)."""
        device = next(iter(h_dict.values())).device
        for ntype, x in x_dict.items():
            if ntype not in h_dict:
                h_dict[ntype] = torch.zeros(x.size(0), hidden_dim, device=device)
        return h_dict

    def forward(self, data: HeteroData):
        hidden_dim = next(iter(self.conv1.convs.values())).out_channels
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}

        # Build edge_index_dict — only include edge types actually present in data
        # and registered in conv layers (handles conditional assessment edges)
        conv1_types = set(self.conv1.convs.keys())
        conv2_types = set(self.conv2.convs.keys())

        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
            if hasattr(data[et], "edge_index") and data[et].edge_index.numel() > 0
        }

        ei_dict_1 = {et: ei for et, ei in edge_index_dict.items() if et in conv1_types}
        h_dict = self.conv1(x_dict, ei_dict_1)
        h_dict = {k: self.act(v) for k, v in h_dict.items()}
        h_dict = self._fill_missing(h_dict, x_dict, hidden_dim)

        # --- Inject enrollment edge attributes between layer 1 and layer 2 ---
        if self.enrollment_attr_proj is not None:
            ei_key = ("student", "enrolled_in", "course_presentation")
            ei_attr = data[ei_key].edge_attr          # (E, n_enrolled_in_attr)
            src_idx = data[ei_key].edge_index[0]      # (E,) student node indices
            proj = self.enrollment_attr_proj(ei_attr)  # (E, hidden_dim)
            n_students = h_dict["student"].size(0)
            enrollment_proj = pyg_scatter(
                proj, src_idx, dim=0, dim_size=n_students, reduce="mean"
            )  # (N_student, hidden_dim)
            h_dict["student"] = h_dict["student"] + enrollment_proj

        ei_dict_2 = {et: ei for et, ei in edge_index_dict.items() if et in conv2_types}
        h_dict = self.conv2(h_dict, ei_dict_2)
        h_dict = {k: self.act(v) for k, v in h_dict.items()}
        h_dict = self._fill_missing(h_dict, x_dict, hidden_dim)

        # Edge-level prediction on enrolled_in
        ei_key = ("student", "enrolled_in", "course_presentation")
        src_idx, dst_idx = data[ei_key].edge_index
        h_src = h_dict["student"][src_idx]              # (E, hidden)
        h_dst = h_dict["course_presentation"][dst_idx]  # (E, hidden)
        edge_repr = torch.cat([h_src, h_dst], dim=1)    # (E, 2*hidden)
        logits = self.edge_head(edge_repr).squeeze(-1)   # (E,)
        return logits


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train(
    model: EnrollmentGNN,
    data: HeteroData,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One training epoch. Returns mean BCE loss."""
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    y = data[("student", "enrolled_in", "course_presentation")].y
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[train_mask], y[train_mask]
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_pos_weight(train_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss (n_neg / n_pos)."""
    train_labels = labels[train_mask]
    n_pos = (train_labels == 1).sum().item()
    n_neg = (train_labels == 0).sum().item()
    if n_pos == 0:
        raise ValueError("No positive examples in training set")
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    return pos_weight


def run_training_loop(
    model: EnrollmentGNN,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int = 200,
    patience: int = 20,
    pos_weight: torch.Tensor = None,
) -> tuple:
    """Multi-epoch training loop with early stopping and class weighting.

    Parameters
    ----------
    pos_weight : torch.Tensor or None
        If None, computes pos_weight from the training labels (weighted loss).
        Pass ``torch.tensor([1.0])`` or use the sentinel ``False`` via the
        caller to disable weighting entirely — but the cleaner API is to pass
        ``pos_weight=None`` and let the function compute it, or to pass a
        pre-computed tensor.  To run *unweighted*, pass the string sentinel by
        setting pos_weight to the special value ``"none"`` (handled below).

    Returns
    -------
    tuple: (best_val_auroc, best_epoch, train_losses, val_aurocs)
        train_losses : list[float] — train BCE loss per epoch
        val_aurocs   : list[float] — val AUROC per epoch (float('nan') when
                       validation set has only one class)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y = data[("student", "enrolled_in", "course_presentation")].y

    if pos_weight is None:
        # Weighted loss: compute class-imbalance weight from training labels
        _pw = compute_pos_weight(train_mask, y)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pw)
    else:
        # Unweighted loss: caller explicitly passed pos_weight=False/tensor([1.])
        # Use plain BCE with no pos_weight argument when caller passes False.
        if pos_weight is False:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = -1
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0

    train_losses: list = []
    val_aurocs: list = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Evaluate on validation set (no print to avoid clutter)
        model.eval()
        with torch.no_grad():
            logits_val = model(data)
        probs_val = torch.sigmoid(logits_val[val_mask]).cpu().numpy()
        labels_val = y[val_mask].cpu().numpy()

        if labels_val.sum() > 0 and (1 - labels_val).sum() > 0:
            val_auroc = roc_auc_score(labels_val, probs_val)
            val_aurocs.append(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        else:
            val_aurocs.append(float("nan"))

    # Restore best state dict
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_val_auroc, best_epoch, train_losses, val_aurocs


def select_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Select classification threshold maximizing F1 on provided probs/labels.

    Sweeps thresholds from 0.05 to 0.95 in steps of 0.05.
    Returns the threshold with the highest F1 score.
    Falls back to 0.5 if all candidates produce degenerate predictions.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    candidates = np.arange(0.05, 1.00, 0.05)
    for t in candidates:
        preds = (probs >= t).astype(int)
        # Skip degenerate case (all one class predicted)
        if preds.sum() == 0 or (1 - preds).sum() == 0:
            continue
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute full metrics dict from probabilities and labels.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities (shape (n,))
    labels : np.ndarray
        True labels (shape (n,))
    threshold : float, optional
        Classification threshold (default 0.5).  Pass the output of
        ``select_threshold()`` for data-driven threshold selection.

    Returns
    -------
    dict with keys: auroc, auprc, f1, precision, recall, balanced_acc
    """
    predictions = (probs >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "balanced_acc": balanced_accuracy_score(labels, predictions),
    }
    return metrics


def evaluate(
    model: EnrollmentGNN,
    data: HeteroData,
    mask: torch.Tensor,
    label: str = "val",
) -> dict:
    """Compute full metrics dict on the given mask subset."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
    y = data[("student", "enrolled_in", "course_presentation")].y
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = y[mask].cpu().numpy()
    
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        print(f"[{label}] Metrics: N/A (only one class present)")
        return {}
    
    metrics = compute_metrics(probs, labels)
    print(f"[{label}] Metrics: {metrics}")
    return metrics


def run_overfit_check(
    data: HeteroData,
    train_mask: torch.Tensor,
    n_samples: int = 128,
    max_epochs: int = 200,
) -> float:
    """Verify the model can overfit a small subset of training data.

    Selects the first ``n_samples`` True positions from ``train_mask``, trains
    a fresh EnrollmentGNN to convergence on just those examples, and returns
    the final train loss.  Prints a warning if the final loss exceeds 0.1.

    Parameters
    ----------
    data        : full HeteroData graph
    train_mask  : boolean mask over enrolled_in edges (full length)
    n_samples   : how many training examples to use (default 128)
    max_epochs  : maximum training epochs (default 200)

    Returns
    -------
    float : final (last-epoch) training loss on the subset
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build a tiny mask using the first n_samples True positions
    true_positions = train_mask.nonzero(as_tuple=True)[0][:n_samples]
    tiny_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    tiny_mask[true_positions] = True

    # Use an empty val mask (we only care about train loss here)
    empty_val = torch.zeros_like(train_mask, dtype=torch.bool)

    # Build a fresh model — same architecture as the main experiment
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    ei_key = ("student", "enrolled_in", "course_presentation")
    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1]
    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=64,
        n_enrolled_in_attr=n_enrolled_in_attr,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    y = data[ei_key].y
    pos_weight = compute_pos_weight(tiny_mask, y)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    final_loss = float("nan")
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits[tiny_mask], y[tiny_mask])
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    if final_loss > 0.1:
        print(
            f"[overfit_check] WARNING: final train loss {final_loss:.4f} > 0.1 — "
            "model may not be fitting the training data."
        )
    else:
        print(f"[overfit_check] OK — final train loss {final_loss:.4f} (≤ 0.1)")

    return final_loss


# ---------------------------------------------------------------------------
# Smoke test / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading Week 2 artifacts...")
    loader = GraphDataLoader(week=2)
    data = loader.load()

    print("Graph summary:")
    for ntype in data.node_types:
        print(f"  {ntype}: {data[ntype].x.shape}")
    for etype in data.edge_types:
        print(f"  {etype}: {data[etype].edge_index.shape}")

    print("\nLoading random split masks...")
    train_mask, val_mask, test_mask = load_split_masks(week=2, split_type="random")
    print(f"  train: {train_mask.sum().item()}  val: {val_mask.sum().item()}  test: {test_mask.sum().item()}")

    # Build in_channels_dict from actual feature dimensions
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    print(f"\nNode feature dims: {in_channels_dict}")

    print("\nBuilding EnrollmentGNN...")
    model = EnrollmentGNN(in_channels_dict=in_channels_dict, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nRunning one forward pass (smoke test)...")
    model.eval()
    with torch.no_grad():
        logits = model(data)
    probs = torch.sigmoid(logits)
    print(f"  logits shape: {logits.shape}  (expected: torch.Size([32593]))")
    print(f"  probs range:  [{probs.min():.4f}, {probs.max():.4f}]")

    print("\nRunning one training step...")
    loss = train(model, data, train_mask, optimizer)
    print(f"  Train loss: {loss:.4f}")

    print("\nRunning multi-epoch training loop (5 epochs, patience=3)...")
    y = data[("student", "enrolled_in", "course_presentation")].y
    pos_weight = compute_pos_weight(train_mask, y)
    best_val_auroc, best_epoch = run_training_loop(
        model, data, train_mask, val_mask, optimizer,
        max_epochs=5, patience=3, pos_weight=pos_weight
    )
    print(f"  Best val-AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    print("\nEvaluating on val and test sets...")
    val_metrics = evaluate(model, data, val_mask, label="val")
    test_metrics = evaluate(model, data, test_mask, label="test")

    print("\nSmoke test complete.")
    assert logits.shape == torch.Size([32593]), f"Unexpected logits shape: {logits.shape}"
    print("[OK] logits shape is correct (32,593 enrollments)")
