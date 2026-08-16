"""
run_gnn_experiment.py
---------------------
End-to-end GNN evaluation for a given prediction week.

Usage
-----
  python src/run_gnn_experiment.py                      # week 8, both experiments
  python src/run_gnn_experiment.py --quick              # 5 epochs, first 2 LCPO folds
  python src/run_gnn_experiment.py --random-only        # skip LCPO
  python src/run_gnn_experiment.py --week 6 --quick
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gnn_model import (
    ARTIFACT_DIR,
    EVAL_DIR,
    SEED,
    EnrollmentGNN,
    GraphDataLoader,
    compute_metrics,
    compute_pos_weight,
    load_split_masks,
    run_overfit_check,
    run_training_loop,
    select_threshold,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/graph"
DEFAULT_WEEK = 8
MAX_EPOCHS = 200
PATIENCE = 20
HIDDEN_DIM = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_and_optimizer(data):
    """Construct a fresh EnrollmentGNN and Adam optimizer."""
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    ei_key = ("student", "enrolled_in", "course_presentation")
    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1]
    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=HIDDEN_DIM,
        n_enrolled_in_attr=n_enrolled_in_attr,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def _infer_probs(model, data, mask):
    """Return predicted probabilities for the given boolean mask."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
    y = data[("student", "enrolled_in", "course_presentation")].y
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = y[mask].cpu().numpy()
    return probs, labels


def _mask_held_out_edges(data, cp_node_idx, enroll_df, held_out_module, held_out_pres):
    """Return a deep copy of *data* with held-out course edges removed.

    Removes from the training graph all edges whose source or destination belongs
    to the held-out course_presentation node, its assessments, or its VLE resources.

    The course_presentation *node* itself is kept (so the model can use its
    features at inference time), but every edge that would leak held-out course
    relationships is filtered out.
    """
    import copy as _copy

    # --- Deep-copy first so we never mutate the caller's graph ---
    masked = _copy.deepcopy(data)

    # Helper: mutate an edge store in-place (deepcopy already owns it)
    def _filter_edge(store, keep_mask, has_attr=False, has_y=False, has_enroll_idx=False):
        store.edge_index = store.edge_index[:, keep_mask]
        if has_attr and store.get("edge_attr") is not None:
            store.edge_attr = store.edge_attr[keep_mask]
        if has_y and store.get("y") is not None:
            # y (labels) stays full length — it's indexed by enrollment_idx not edge position
            pass
        if has_enroll_idx and store.get("enrollment_idx") is not None:
            store.enrollment_idx = store.enrollment_idx[keep_mask]

    # NOTE: enrolled_in / rev_enrolled_in edges are intentionally *not* filtered.
    # The logit tensor produced by model(data) is indexed by enrolled_in position,
    # so its length must equal the full enrollment count (32,593) to keep train/test
    # mask alignment.  Information leakage from held-out enrollment features is
    # blocked by the train_mask (held-out rows are False in train_mask, so no
    # gradient flows through their loss).
    #
    # submitted / interacted_with are filtered by *destination* node (assessment or
    # vle_resource belonging to the held-out CP), NOT by source student.  This
    # preserves cross-course behavioral signal for students enrolled in multiple
    # courses — only their activity that specifically belongs to the held-out CP is
    # removed from message-passing.

    # 2. contains_assess / rev_contains_assess — filter by CP src/dst
    ca_key = ("course_presentation", "contains_assess", "assessment")
    if ca_key in masked.edge_types:
        ca_store = masked[ca_key]
        keep_ca = (ca_store.edge_index[0] != cp_node_idx)
        _filter_edge(ca_store, keep_ca)

        rev_ca_key = ("assessment", "rev_contains_assess", "course_presentation")
        if rev_ca_key in masked.edge_types:
            rev_ca_store = masked[rev_ca_key]
            keep_rev_ca = (rev_ca_store.edge_index[1] != cp_node_idx)
            _filter_edge(rev_ca_store, keep_rev_ca)

    # 3. has_resource / rev_has_resource — filter by CP src/dst
    hr_key = ("course_presentation", "has_resource", "vle_resource")
    if hr_key in masked.edge_types:
        hr_store = masked[hr_key]
        keep_hr = (hr_store.edge_index[0] != cp_node_idx)
        _filter_edge(hr_store, keep_hr)

        rev_hr_key = ("vle_resource", "rev_has_resource", "course_presentation")
        if rev_hr_key in masked.edge_types:
            rev_hr_store = masked[rev_hr_key]
            keep_rev_hr = (rev_hr_store.edge_index[1] != cp_node_idx)
            _filter_edge(rev_hr_store, keep_rev_hr)

    # 4. submitted / rev_submitted — filter by held-out CP's assessment nodes
    #    Build the set of assessment node indices that belong to the held-out CP by
    #    reading the original (pre-copy) contains_assess edges.  Only edges whose
    #    destination assessment is in that set are removed; cross-course submissions
    #    by the same student are preserved.
    sub_key = ("student", "submitted", "assessment")
    ca_key_check = ("course_presentation", "contains_assess", "assessment")
    if sub_key in masked.edge_types:
        if ca_key_check in data.edge_types:
            ca_ei = data[ca_key_check].edge_index
            ho_assess_node_indices = set(
                ca_ei[1][ca_ei[0] == cp_node_idx].tolist()
            )
        else:
            ho_assess_node_indices = set()

        if ho_assess_node_indices:
            sub_store = masked[sub_key]
            keep_sub = torch.tensor(
                [d.item() not in ho_assess_node_indices for d in sub_store.edge_index[1]],
                dtype=torch.bool,
            )
            _filter_edge(sub_store, keep_sub, has_attr=True)

            rev_sub_key = ("assessment", "rev_submitted", "student")
            if rev_sub_key in masked.edge_types:
                rev_sub_store = masked[rev_sub_key]
                keep_rev_sub = torch.tensor(
                    [s.item() not in ho_assess_node_indices for s in rev_sub_store.edge_index[0]],
                    dtype=torch.bool,
                )
                _filter_edge(rev_sub_store, keep_rev_sub)

    # 5. interacted_with / rev_interacted_with — filter by held-out CP's vle_resource nodes
    #    Build the set of vle_resource node indices that belong to the held-out CP via
    #    has_resource edges.  Cross-course VLE interactions by the same student are kept.
    iw_key = ("student", "interacted_with", "vle_resource")
    hr_key_check = ("course_presentation", "has_resource", "vle_resource")
    if iw_key in masked.edge_types:
        if hr_key_check in data.edge_types:
            hr_ei = data[hr_key_check].edge_index
            ho_vle_node_indices = set(
                hr_ei[1][hr_ei[0] == cp_node_idx].tolist()
            )
        else:
            ho_vle_node_indices = set()

        if ho_vle_node_indices:
            iw_store = masked[iw_key]
            keep_iw = torch.tensor(
                [d.item() not in ho_vle_node_indices for d in iw_store.edge_index[1]],
                dtype=torch.bool,
            )
            _filter_edge(iw_store, keep_iw, has_attr=True)

            rev_iw_key = ("vle_resource", "rev_interacted_with", "student")
            if rev_iw_key in masked.edge_types:
                rev_iw_store = masked[rev_iw_key]
                keep_rev_iw = torch.tensor(
                    [s.item() not in ho_vle_node_indices for s in rev_iw_store.edge_index[0]],
                    dtype=torch.bool,
                )
                _filter_edge(rev_iw_store, keep_rev_iw)

    return masked


# ---------------------------------------------------------------------------
# Experiment 1: Random-student split
# ---------------------------------------------------------------------------

def run_random_split_experiment(
    week: int = DEFAULT_WEEK,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    weighted: bool = True,
    seed: int = SEED,
):
    """Train and evaluate EnrollmentGNN on a 70/10/20 random-student split.

    Parameters
    ----------
    weighted : bool
        If True (default), use class-weighted BCE loss (pos_weight computed
        from training labels).  If False, use unweighted BCE loss.
    seed : int
        Random seed used to draw the student split.  Passed to
        ``random_student_split()`` so different seeds produce independent
        partitions.  Defaults to the global SEED constant.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_weighting = "weighted" if weighted else "unweighted"
    print(f"\n=== Random-split experiment  (week {week:02d}, {loss_weighting}, seed {seed}) ===")

    # Load graph
    data = GraphDataLoader(week).load()

    # Build split masks for this seed using random_student_split()
    from oulad_data import random_student_split as _random_student_split
    _enroll_df = pd.read_parquet(
        os.path.join(ARTIFACT_DIR, f"week{week:02d}_enrollments.parquet")
    )
    _train_s, _val_s, _test_s = _random_student_split(_enroll_df, seed=seed)
    train_mask = torch.tensor(_train_s.values, dtype=torch.bool)
    val_mask   = torch.tensor(_val_s.values,   dtype=torch.bool)
    test_mask  = torch.tensor(_test_s.values,  dtype=torch.bool)

    # Class-weighting (or None for unweighted run)
    y = data[("student", "enrolled_in", "course_presentation")].y
    if weighted:
        pos_weight = compute_pos_weight(train_mask, y)
    else:
        pos_weight = False  # sentinel: triggers plain BCEWithLogitsLoss()

    # Model + optimizer
    model, optimizer = _build_model_and_optimizer(data)

    # Training — captures per-epoch curves
    best_val_auroc, best_epoch, train_losses, val_aurocs = run_training_loop(
        model, data, train_mask, val_mask, optimizer,
        max_epochs=max_epochs,
        patience=patience,
        pos_weight=pos_weight,
    )
    print(f"  best_val_auroc={best_val_auroc:.4f}  best_epoch={best_epoch}")

    # Save per-epoch loss curves
    os.makedirs(RESULTS_DIR, exist_ok=True)
    curves_path = os.path.join(RESULTS_DIR, f"training_curves_random_student_{loss_weighting}.npz")
    np.savez(curves_path, train_losses=np.array(train_losses), val_aurocs=np.array(val_aurocs))
    print(f"  Loss curves → {curves_path}")

    # Threshold tuning on validation set
    val_probs, val_labels = _infer_probs(model, data, val_mask)
    best_threshold = 0.5
    if val_labels.sum() > 0 and (1 - val_labels).sum() > 0:
        best_threshold = select_threshold(val_probs, val_labels)
    print(f"  best_threshold={best_threshold:.2f}")

    # Test evaluation using tuned threshold
    probs, labels = _infer_probs(model, data, test_mask)
    metrics = compute_metrics(probs, labels, threshold=best_threshold)

    row = {
        "week": week,
        "seed": seed,
        "model": "EnrollmentGNN",
        "split": "random_student",
        "loss_weighting": loss_weighting,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "balanced_acc": metrics["balanced_acc"],
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
    }

    return row, metrics


# ---------------------------------------------------------------------------
# Experiment 2: LCPO (Leave-Course-Presentation-Out) evaluation
# ---------------------------------------------------------------------------

def run_lcpo_experiment(week: int = DEFAULT_WEEK, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, max_folds: int = None):
    """Train and evaluate one model per LCPO fold.

    Parameters
    ----------
    week       : prediction week
    max_epochs : max training epochs per fold
    patience   : early-stopping patience
    max_folds  : if set, run only the first N folds (for --quick mode)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n=== LCPO experiment  (week {week:02d}) ===")

    w = f"week{week:02d}"
    folds_path = os.path.join(EVAL_DIR, w, "splits", f"{w}_lcpo_folds.csv")
    folds_df = pd.read_csv(folds_path)

    enroll_df = pd.read_parquet(os.path.join(ARTIFACT_DIR, f"{w}_enrollments.parquet"))
    cp_df = pd.read_parquet(os.path.join(ARTIFACT_DIR, f"{w}_nodes_course_presentation.parquet"))
    n_enrollments = len(enroll_df)

    if max_folds is not None:
        folds_df = folds_df.iloc[:max_folds].copy()
        print(f"  (quick mode: running only first {max_folds} folds)")

    records = []

    for _, fold_row in folds_df.iterrows():
        fold_idx = int(fold_row["fold_idx"])
        ho_module = fold_row["held_out_module"]
        ho_pres = fold_row["held_out_presentation"]

        print(f"  Fold {fold_idx:02d}: held-out {ho_module} / {ho_pres}", end="  ", flush=True)

        # --- Find held-out CP node index ---
        cp_match = cp_df[
            (cp_df["code_module"] == ho_module)
            & (cp_df["code_presentation"] == ho_pres)
        ]
        if len(cp_match) == 0:
            print(f"WARNING: held-out course-presentation not found — skipping")
            continue
        cp_node_idx = int(cp_match.iloc[0]["node_idx"])

        # --- Build train / test masks ---
        ho_rows = (
            (enroll_df["code_module"] == ho_module)
            & (enroll_df["code_presentation"] == ho_pres)
        )
        test_mask_np = ho_rows.values
        train_all_np = ~test_mask_np

        # 10% of non-test students → validation set for early stopping
        train_student_ids = enroll_df.loc[train_all_np, "id_student"].unique()
        rng = np.random.default_rng(SEED + fold_idx)
        val_size = max(1, int(0.10 * len(train_student_ids)))
        val_student_ids = rng.choice(train_student_ids, size=val_size, replace=False)

        val_mask_np = train_all_np & enroll_df["id_student"].isin(val_student_ids).to_numpy()
        train_mask_np = train_all_np & ~val_mask_np

        train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
        test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

        # --- Load graph and mask held-out edges ---
        data = GraphDataLoader(week).load()
        # Attach week to data for use inside helper
        data._held_out_week = week
        data_masked = _mask_held_out_edges(data, cp_node_idx, enroll_df, ho_module, ho_pres)

        # --- Train fresh model ---
        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)
        model, optimizer = _build_model_and_optimizer(data_masked)

        y = data_masked[("student", "enrolled_in", "course_presentation")].y
        pos_weight = compute_pos_weight(train_mask, y)

        best_val_auroc, best_epoch, _train_losses, _val_aurocs = run_training_loop(
            model, data_masked, train_mask, val_mask, optimizer,
            max_epochs=max_epochs,
            patience=patience,
            pos_weight=pos_weight,
        )

        # --- Evaluate on full (unmasked) graph ---
        probs, labels = _infer_probs(model, data, test_mask)
        if labels.sum() == 0 or (1 - labels).sum() == 0:
            print("SKIP (single class in test)")
            continue

        metrics = compute_metrics(probs, labels)

        record = {
            "fold_idx": fold_idx,
            "held_out_module": ho_module,
            "held_out_presentation": ho_pres,
            "n_train": int(train_mask_np.sum()),
            "n_test": int(test_mask_np.sum()),
            "best_val_auroc": best_val_auroc,
            "best_epoch": best_epoch,
            **metrics,
        }
        records.append(record)
        print(f"auroc={metrics['auroc']:.4f}  auprc={metrics['auprc']:.4f}")

    if not records:
        print("  No folds completed.")
        return pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_fold_df = pd.DataFrame(records)

    per_fold_path = os.path.join(RESULTS_DIR, "lcpo_results.csv")
    per_fold_df.to_csv(per_fold_path, index=False)
    print(f"  Per-fold results → {per_fold_path}")

    # Summary: mean ± std across folds
    metric_cols = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    summary_rows = []
    for col in metric_cols:
        summary_rows.append({
            "metric": col,
            "mean": per_fold_df[col].mean(),
            "std": per_fold_df[col].std(),
            "min": per_fold_df[col].min(),
            "max": per_fold_df[col].max(),
        })
    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(RESULTS_DIR, "lcpo_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary → {summary_path}")

    return per_fold_df


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def _print_summary(random_metrics, lcpo_df):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if random_metrics:
        print("\nRandom-student split (single run, Week 8):")
        for k, v in random_metrics.items():
            print(f"  {k:<15} {v:.4f}")

    if lcpo_df is not None and len(lcpo_df) > 0:
        metric_cols = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
        print(f"\nLCPO ({len(lcpo_df)} folds) — mean ± std:")
        for col in metric_cols:
            m, s = lcpo_df[col].mean(), lcpo_df[col].std()
            print(f"  {col:<15} {m:.4f} ± {s:.4f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN experiment runner for OULAD")
    parser.add_argument("--week", type=int, default=DEFAULT_WEEK,
                        help=f"Prediction week (default: {DEFAULT_WEEK})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: MAX_EPOCHS=5, PATIENCE=3, first 2 LCPO folds only")
    parser.add_argument("--random-only", action="store_true",
                        help="Skip LCPO; run only the random-student experiment")
    parser.add_argument("--overfit-check", action="store_true",
                        help="Run overfit sanity check before the main experiment and print result")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="One or more random seeds for the random-student split experiment "
                             "(default: 42).  Each seed produces an independent train/val/test "
                             "partition; rows are accumulated and written together.")
    args = parser.parse_args()

    epochs = 5 if args.quick else MAX_EPOCHS
    pat = 3 if args.quick else PATIENCE
    # --quick limits LCPO to 2 folds; a production run uses all folds (max_folds=None)
    max_folds = 2 if args.quick else None

    random_metrics = None
    lcpo_df = None

    # --- Optional overfit check ---
    if args.overfit_check:
        print("\n=== Overfit check ===")
        data_for_check = GraphDataLoader(args.week).load()
        train_mask_check, _, _ = load_split_masks(args.week, split_type="random")
        check_loss = run_overfit_check(data_for_check, train_mask_check)
        print(f"  Overfit check final loss: {check_loss:.4f}")

    # --- Random-student experiment: loop over (seed × weighted/unweighted) ---
    rows = []
    for seed_val in args.seeds:
        for weighted_flag in (True, False):
            row, metrics = run_random_split_experiment(
                week=args.week, max_epochs=epochs, patience=pat,
                weighted=weighted_flag, seed=seed_val,
            )
            rows.append(row)
            if weighted_flag and seed_val == args.seeds[0]:
                random_metrics = metrics  # used for _print_summary (first seed, weighted)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "random_student_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Saved {len(rows)} rows ({len(args.seeds)} seed(s) × 2 weightings) → {out_path}")

    if not args.random_only:
        lcpo_df = run_lcpo_experiment(week=args.week, max_epochs=epochs, patience=pat, max_folds=max_folds)

    _print_summary(random_metrics, lcpo_df)
