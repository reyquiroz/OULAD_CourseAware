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

from config import GRAPH_RESULTS_DIR, PIPELINE_VERSION
from gnn_model import (
    ARTIFACT_DIR,
    EVAL_DIR,
    SEED,
    EnrollmentGNN,
    GraphDataLoader,
    _normalize_numeric_features,
    build_train_subgraph,
    compute_metrics,
    compute_pos_weight,
    load_split_masks,
    run_overfit_check,
    run_training_loop,
    select_threshold,
)
from split_loader import load_lcpo_masks, load_random_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = str(GRAPH_RESULTS_DIR)
DEFAULT_WEEK = 8
MAX_EPOCHS = 200
PATIENCE = 20
HIDDEN_DIM = 64

# Distinguishing keys for the LCPO CSVs. loss_weighting must be part of both: weighted and
# unweighted runs of the same fold/seed/condition would otherwise overwrite each other.
LCPO_RESULT_KEYS = ["week", "fold_idx", "model_seed", "condition", "loss_weighting"]
LCPO_SUMMARY_KEYS = ["week", "fold_idx", "condition", "loss_weighting"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_or_create_csv(new_df: pd.DataFrame, path: str, dedup_keys: list) -> None:
    """Append new_df to an existing CSV, deduplicating on dedup_keys. Creates if absent."""
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _build_model_and_optimizer(
    data,
    use_interacted_with_attrs: bool = False,
    use_enrollment_nodes: bool = False,
    use_edge_weights: bool = False,
):
    """Construct a fresh EnrollmentGNN and Adam optimizer.

    Parameters
    ----------
    data : HeteroData
        Full heterogeneous graph; used to infer input channel dimensions.
    use_interacted_with_attrs : bool
        Passed through to ``EnrollmentGNN``.  When True, the model projects
        ``interacted_with`` edge attributes into the student embeddings.
        Default False (topology-only; preserves current behaviour).
    use_enrollment_nodes : bool
        If True, use enrollment node representations for predictions.
    """
    in_channels_dict = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    ei_key = ("student", "enrolled_in", "course_presentation")
    n_enrolled_in_attr = data[ei_key].edge_attr.shape[1] if hasattr(data[ei_key], "edge_attr") and data[ei_key].edge_attr is not None else 0

    model = EnrollmentGNN(
        in_channels_dict=in_channels_dict,
        hidden_dim=HIDDEN_DIM,
        n_enrolled_in_attr=n_enrolled_in_attr,
        use_interacted_with_attrs=use_interacted_with_attrs,
        use_enrollment_nodes=use_enrollment_nodes,
        use_edge_weights=use_edge_weights,
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
    # so its length must equal the full enrollment count for the week to keep train/test
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
    feature_mask=None,
    use_interacted_with_attrs: bool = False,
    use_enrollment_summaries: bool = False,
    use_enrollment_nodes: bool = False,
    use_edge_weights: bool = False,
    use_temporal_features: bool = False,
):
    """Train and evaluate EnrollmentGNN on a 70/10/20 random-student split.

    Parameters
    ----------
    weighted : bool
        If True (default), use class-weighted BCE loss (pos_weight computed
        from training labels).  If False, use unweighted BCE loss.
    seed : int
        Split seed: selects the persisted random-student split
        (``week*_random_split_seed{seed}.parquet`` from ``save_graph_splits.py``)
        and also seeds model initialisation. Defaults to the global SEED constant.
    feature_mask : list[str] | None
        Ablation conditions to apply (e.g. ["no_vle"]).  Passed through to
        GraphDataLoader so the correct features are zeroed before training.
    use_interacted_with_attrs : bool
        If True, pass the ``interacted_with`` edge attributes into the model
        for the ablation condition ``"with_iw_attrs"``.  Default False.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_weighting = "weighted" if weighted else "unweighted"
    print(f"\n=== Random-split experiment  (week {week:02d}, {loss_weighting}, seed {seed}) ===")

    # Load raw graph (without normalization) so we can normalise with train mask
    data = GraphDataLoader(
        week,
        feature_mask=feature_mask,
        skip_normalize=True,
        use_enrollment_summaries=use_enrollment_summaries,
        use_temporal_features=use_temporal_features,
        use_enrollment_nodes=use_enrollment_nodes,
    ).load()

    # Persisted split for this seed (written by save_graph_splits.py, shared with the
    # tabular models), aligned to the enrollment table by triple.
    _enroll_df = pd.read_parquet(
        os.path.join(ARTIFACT_DIR, f"week{week:02d}_enrollments.parquet")
    )
    _train_s, _val_s, _test_s = load_random_split(week, seed, frame=_enroll_df)
    train_mask = torch.tensor(_train_s.values, dtype=torch.bool)
    val_mask   = torch.tensor(_val_s.values,   dtype=torch.bool)
    test_mask  = torch.tensor(_test_s.values,  dtype=torch.bool)

    # Train-only normalization: stats computed from train mask only to prevent leakage
    # into test/val data. See _normalize_numeric_features() in gnn_model.py.
    # Note: _apply_feature_mask was already applied by load(); only normalise here.
    data = _normalize_numeric_features(data, train_edge_mask=train_mask)

    # Inductive evaluation: only training enrollment edges are visible during message
    # passing. Test node features are present but test edges are never in the training graph.
    train_subgraph = build_train_subgraph(data, train_mask)

    # Class-weighting (or None for unweighted run)
    train_y = train_subgraph[("student", "enrolled_in", "course_presentation")].y
    if weighted:
        pos_weight = compute_pos_weight(None, train_y)
    else:
        pos_weight = False  # sentinel: triggers plain BCEWithLogitsLoss()

    # Model + optimizer — built from full data so inference channel dims are correct
    model, optimizer = _build_model_and_optimizer(
        data,
        use_interacted_with_attrs=use_interacted_with_attrs,
        use_enrollment_nodes=use_enrollment_nodes,
        use_edge_weights=use_edge_weights,
    )

    # Training — captures per-epoch curves
    best_val_auroc, best_epoch, train_losses, val_aurocs = run_training_loop(
        model, train_subgraph, data, val_mask, optimizer,
        max_epochs=max_epochs,
        patience=patience,
        pos_weight=pos_weight,
    )
    print(f"  best_val_auroc={best_val_auroc:.4f}  best_epoch={best_epoch}")

    # Save per-epoch loss curves
    os.makedirs(RESULTS_DIR, exist_ok=True)
    curves_path = os.path.join(RESULTS_DIR, f"training_curves_random_student_week{week:02d}_seed{seed}_{loss_weighting}.npz")
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
        "pipeline_version": PIPELINE_VERSION,
    }

    return row, metrics


# ---------------------------------------------------------------------------
# Experiment 2: LCPO (Leave-Course-Presentation-Out) evaluation
# ---------------------------------------------------------------------------

def run_lcpo_experiment(
    week: int = DEFAULT_WEEK,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    max_folds: int = None,
    lcpo_patience: int = 50,
    model_seeds: list = None,
    start_fold: int = 0,
    use_enrollment_summaries: bool = False,
    use_enrollment_nodes: bool = False,
    use_edge_weights: bool = False,
    use_temporal_features: bool = False,
    condition: str = "base",
    weighted: bool = True,
):
    """Train and evaluate models per LCPO fold, running one model per seed.

    Parameters
    ----------
    week          : prediction week
    max_epochs    : max training epochs per fold
    patience      : early-stopping patience (unused; kept for API compat)
    max_folds     : if set, run only the first N folds (for --quick mode)
    lcpo_patience : early-stopping patience used in the LCPO training loop
                    (default 50, separate from the random-split PATIENCE=20)
    model_seeds   : list of model initialisation seeds (default: [42, 123, 7, 17, 99]).
                    One independent training run is performed per seed per fold;
                    the fold result is the mean ± std across seeds.
    start_fold    : skip all folds with fold_idx < start_fold (resumption after
                    a partial run).  Default 0 (run all folds).
    condition     : experiment condition label written into every row so that
                    variants can coexist in the same CSV without overwriting each
                    other's rows during deduplication.
    weighted      : if True (default) use class-weighted BCE (pos_weight from the
                    training labels); if False use plain BCE. The only difference
                    between the two runs is the loss: both read the same persisted
                    train/val/test masks. Written to ``loss_weighting`` in every row.
    """
    if model_seeds is None:
        model_seeds = [42, 123, 7, 17, 99]

    loss_weighting = "weighted" if weighted else "unweighted"
    print(f"\n=== LCPO experiment  (week {week:02d}, {loss_weighting}) ===")

    w = f"week{week:02d}"
    folds_path = os.path.join(EVAL_DIR, w, "splits", f"{w}_lcpo_folds.csv")
    folds_df = pd.read_csv(folds_path)

    enroll_df = pd.read_parquet(os.path.join(ARTIFACT_DIR, f"{w}_enrollments.parquet"))
    cp_df = pd.read_parquet(os.path.join(ARTIFACT_DIR, f"{w}_nodes_course_presentation.parquet"))
    n_enrollments = len(enroll_df)

    if start_fold > 0:
        folds_df = folds_df[folds_df["fold_idx"] >= start_fold].copy()
        print(f"  (resuming from fold {start_fold}; skipping {start_fold} earlier folds)")

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

        # --- Persisted train / val / test masks (save_graph_splits.py, shared with tabular) ---
        tr_s, va_s, te_s = load_lcpo_masks(week, fold_idx, frame=enroll_df)
        train_mask_np = tr_s.to_numpy()
        val_mask_np = va_s.to_numpy()
        test_mask_np = te_s.to_numpy()
        assert (enroll_df.loc[test_mask_np, "code_module"] == ho_module).all()
        assert (enroll_df.loc[test_mask_np, "code_presentation"] == ho_pres).all()

        train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
        val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
        test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

        # --- Load raw graph, normalise with train mask, then mask held-out edges ---
        # Load without normalization so stats can be computed from training rows only.
        data = GraphDataLoader(
            week,
            skip_normalize=True,
            use_enrollment_summaries=use_enrollment_summaries,
            use_temporal_features=use_temporal_features,
            use_enrollment_nodes=use_enrollment_nodes,
        ).load()
        # Train-only normalization: stats computed from train mask only to prevent leakage
        # into test/val data. See _normalize_numeric_features() in gnn_model.py.
        data = _normalize_numeric_features(data, train_edge_mask=train_mask)
        # Attach week to data for use inside helper
        data._held_out_week = week
        data_masked = _mask_held_out_edges(data, cp_node_idx, enroll_df, ho_module, ho_pres)

        # Inductive evaluation: only training enrollment edges are visible during message
        # passing. Test node features are present but test edges are never in the training graph.
        train_subgraph = build_train_subgraph(data_masked, train_mask)

        train_y = train_subgraph[("student", "enrolled_in", "course_presentation")].y
        if weighted:
            pos_weight = compute_pos_weight(None, train_y)
        else:
            pos_weight = False  # sentinel: plain BCEWithLogitsLoss (see run_training_loop)

        # --- Inner loop over model seeds: one independent run per seed ---
        fold_skipped = False
        for mseed in model_seeds:
            torch.manual_seed(mseed)
            np.random.seed(mseed)
            model, optimizer = _build_model_and_optimizer(
                data_masked,
                use_enrollment_nodes=use_enrollment_nodes,
                use_edge_weights=use_edge_weights,
            )

            best_val_auroc, best_epoch, seed_train_losses, seed_val_aurocs = run_training_loop(
                model, train_subgraph, data_masked, val_mask, optimizer,
                max_epochs=max_epochs,
                patience=lcpo_patience,
                pos_weight=pos_weight,
            )

            # Save per-fold-per-seed training curves
            os.makedirs(RESULTS_DIR, exist_ok=True)
            curves_path = os.path.join(
                RESULTS_DIR,
                f"training_curves_lcpo_week{week:02d}_{condition}_{loss_weighting}"
                f"_fold{fold_idx:02d}_seed{mseed}.npz",
            )
            np.savez(
                curves_path,
                train_losses=np.array(seed_train_losses),
                val_aurocs=np.array(seed_val_aurocs),
            )

            # --- Threshold tuning on val set (F1-max, matching random-split path) ---
            val_probs, val_labels = _infer_probs(model, data, val_mask)
            best_threshold = 0.5
            if val_labels.sum() > 0 and (1 - val_labels).sum() > 0:
                best_threshold = select_threshold(val_probs, val_labels)

            # --- Evaluate on full (unmasked) graph ---
            probs, labels = _infer_probs(model, data, test_mask)
            if labels.sum() == 0 or (1 - labels).sum() == 0:
                print("SKIP (single class in test)")
                fold_skipped = True
                break

            metrics = compute_metrics(probs, labels, threshold=best_threshold)

            record = {
                "week": week,
                "fold_idx": fold_idx,
                "held_out_module": ho_module,
                "held_out_presentation": ho_pres,
                "model_seed": mseed,
                "condition": condition,
                "loss_weighting": loss_weighting,
                "n_train": int(train_mask_np.sum()),
                "n_test": int(test_mask_np.sum()),
                "best_val_auroc": best_val_auroc,
                "best_epoch": best_epoch,
                **metrics,
                "pipeline_version": PIPELINE_VERSION,
            }
            records.append(record)

        if fold_skipped:
            continue

        # Report fold-level mean across seeds
        seed_aurocs = [r["auroc"] for r in records if r["fold_idx"] == fold_idx]
        seed_auprcs = [r["auprc"] for r in records if r["fold_idx"] == fold_idx]
        print(
            f"auroc={np.mean(seed_aurocs):.4f}±{np.std(seed_aurocs):.4f}  "
            f"auprc={np.mean(seed_auprcs):.4f}±{np.std(seed_auprcs):.4f}"
        )

    if not records:
        print("  No folds completed.")
        return pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed_df = pd.DataFrame(records)

    # --- lcpo_results.csv: one row per (fold, seed) ---
    per_fold_path = os.path.join(RESULTS_DIR, "lcpo_results.csv")
    _append_or_create_csv(per_seed_df, per_fold_path, dedup_keys=LCPO_RESULT_KEYS)
    print(f"  Per-fold-per-seed results → {per_fold_path}")

    # --- lcpo_summary.csv: one row per fold with mean ± std across seeds ---
    # Group by loss_weighting so that weighted and unweighted settings produce
    # separate summary rows and are never conflated in mean ± std computation.
    metric_cols = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
    summary_rows = []
    for (wk, fidx, cond, lw, ho_mod, ho_pres), fold_group in per_seed_df.groupby(
        ["week", "fold_idx", "condition", "loss_weighting",
         "held_out_module", "held_out_presentation"], sort=False
    ):
        row: dict = {
            "week": wk,
            "fold_idx": fidx,
            "condition": cond,
            "loss_weighting": lw,
            "held_out_module": ho_mod,
            "held_out_presentation": ho_pres,
        }
        for col in metric_cols:
            row[f"{col}_mean"] = fold_group[col].mean()
            row[f"{col}_std"] = fold_group[col].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(RESULTS_DIR, "lcpo_summary.csv")
    _append_or_create_csv(summary_df, summary_path, dedup_keys=LCPO_SUMMARY_KEYS)
    print(f"  Fold summary (mean±std across seeds) → {summary_path}")

    return per_seed_df


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
    parser.add_argument("--week", type=int, default=None,
                        help=f"Single prediction week (default: {DEFAULT_WEEK}). "
                             "Superseded by --weeks when both are given.")
    parser.add_argument("--weeks", nargs="+", type=int, default=None,
                        help="One or more prediction weeks to run in sequence "
                             "(e.g. --weeks 2 4 6 8). Overrides --week.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: MAX_EPOCHS=5, PATIENCE=3, first 2 LCPO folds only")
    parser.add_argument("--random-only", action="store_true",
                        help="Skip LCPO; run only the random-student experiment")
    parser.add_argument("--lcpo-only", action="store_true",
                        help="Skip random-student experiment; run only LCPO")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Skip LCPO folds with fold_idx < N (for resuming a partial run). Default: 0")
    parser.add_argument("--overfit-check", action="store_true",
                        help="Run overfit sanity check before the main experiment and print result")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="One or more random seeds for the random-student split experiment "
                             "(default: 42).  Each seed produces an independent train/val/test "
                             "partition; rows are accumulated and written together.")
    parser.add_argument("--lcpo-patience", type=int, default=50,
                        help="Early-stopping patience for the LCPO training loop (default: 50).")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[42, 123, 7, 17, 99],
                        help="Model initialisation seeds for LCPO (one run per seed per fold). "
                             "Default: 42 123 7 17 99")
    # NOTE: base GraphSAGE (condition='full') is missing seed 42 at all weeks in the
    # current results (lcpo_results.csv only has seeds 123, 7, 17, 99 for that condition).
    # Before computing final summary statistics, re-run with:
    #   python src/run_gnn_experiment.py --lcpo-only --condition full --model-seeds 42
    # to fill the gap. Do NOT change the default seeds list above.
    parser.add_argument("--lcpo-loss-weighting", choices=["weighted", "unweighted", "both"],
                        default="weighted", dest="lcpo_loss_weighting",
                        help="Loss for LCPO runs: class-weighted BCE (default), plain BCE, or "
                             "both (two separate runs per fold/seed, stored as separate rows).")
    parser.add_argument("--condition", type=str, default="full",
                        help="GNN condition to run: 'full', 'with_enrollment_summaries', 'with_iw_attrs', 'enrollment_node', etc. Default: full")
    parser.add_argument("--use-enrollment-nodes", action="store_true", default=False,
                        help="Gate enrollment node loading in GraphDataLoader (default: False). "
                             "When set, also implied by --condition enrollment_node.")
    parser.add_argument("--splits", nargs="+", choices=["random", "lcpo"], default=None,
                        help="Which splits to run: random, lcpo, or both. Overrides --random-only/--lcpo-only.")
    args = parser.parse_args()

    # Resolve week list: --weeks wins over --week; fall back to DEFAULT_WEEK
    if args.weeks is not None:
        weeks_to_run = args.weeks
    elif args.week is not None:
        weeks_to_run = [args.week]
    else:
        weeks_to_run = [DEFAULT_WEEK]

    epochs = 5 if args.quick else MAX_EPOCHS
    pat = 3 if args.quick else PATIENCE
    # --quick limits LCPO to 2 folds; a production run uses all folds (max_folds=None)
    max_folds = 2 if args.quick else None

    all_random_rows: list[dict] = []
    all_lcpo_dfs: list = []
    random_metrics = None
    lcpo_df = None

    for week in weeks_to_run:
        print(f"\n{'='*60}")
        print(f"WEEK {week}")
        print(f"{'='*60}")

        # --- Optional overfit check ---
        if args.overfit_check:
            print("\n=== Overfit check ===")
            data_for_check = GraphDataLoader(week).load()
            train_mask_check, _, _ = load_split_masks(week, split_type="random")
            check_loss = run_overfit_check(data_for_check, train_mask_check)
            print(f"  Overfit check final loss: {check_loss:.4f}")

        use_enroll_summaries = (args.condition == "with_enrollment_summaries")
        use_iw = (args.condition == "with_iw_attrs")
        use_enroll_nodes = args.use_enrollment_nodes or (args.condition == "enrollment_node")
        use_edge_w = (args.condition == "edge_aware_mp")
        use_temporal = (args.condition in ("temporal_features", "temporal"))

        # Determine which splits to run
        if args.splits is not None:
            run_random = "random" in args.splits
            run_lcpo_flag = "lcpo" in args.splits
        else:
            run_random = not args.lcpo_only
            run_lcpo_flag = not args.random_only

        # --- Random-student experiment: loop over (seed × weighted/unweighted) ---
        if run_random:
            for seed_val in args.seeds:
                for weighted_flag in (True, False):
                    row, metrics = run_random_split_experiment(
                        week=week, max_epochs=epochs, patience=pat,
                        weighted=weighted_flag, seed=seed_val,
                        use_interacted_with_attrs=use_iw,
                        use_enrollment_summaries=use_enroll_summaries,
                        use_enrollment_nodes=use_enroll_nodes,
                        use_edge_weights=use_edge_w,
                        use_temporal_features=use_temporal,
                    )
                    row["condition"] = args.condition
                    all_random_rows.append(row)
                    if random_metrics is None and weighted_flag:
                        random_metrics = metrics  # first seed, weighted — for _print_summary

        if run_lcpo_flag:
            lcpo_patience_val = 3 if args.quick else args.lcpo_patience
            lcpo_weightings = {
                "weighted": [True], "unweighted": [False], "both": [True, False],
            }[args.lcpo_loss_weighting]
            for weighted_flag in lcpo_weightings:
                week_lcpo_df = run_lcpo_experiment(
                    week=week, max_epochs=epochs, patience=pat,
                    max_folds=max_folds, lcpo_patience=lcpo_patience_val,
                    model_seeds=args.model_seeds,
                    start_fold=args.start_fold,
                    use_enrollment_summaries=use_enroll_summaries,
                    use_enrollment_nodes=use_enroll_nodes,
                    use_edge_weights=use_edge_w,
                    use_temporal_features=use_temporal,
                    condition=args.condition,
                    weighted=weighted_flag,
                )
                if not week_lcpo_df.empty:
                    all_lcpo_dfs.append(week_lcpo_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Save random results (all weeks stacked) ---
    if all_random_rows:
        out_path = os.path.join(RESULTS_DIR, "random_student_results.csv")
        _append_or_create_csv(
            pd.DataFrame(all_random_rows), out_path,
            dedup_keys=["week", "seed", "loss_weighting", "condition"],
        )
        print(f"\n  Saved {len(all_random_rows)} rows "
              f"({len(weeks_to_run)} week(s) × {len(args.seeds)} seed(s) × 2 weightings) → {out_path}")

    # --- Save LCPO results (all weeks stacked) ---
    if all_lcpo_dfs:
        lcpo_df = pd.concat(all_lcpo_dfs, ignore_index=True)
        lcpo_path = os.path.join(RESULTS_DIR, "lcpo_results.csv")
        _append_or_create_csv(lcpo_df, lcpo_path, dedup_keys=LCPO_RESULT_KEYS)
        print(f"  Saved {len(lcpo_df)} LCPO rows (all weeks) → {lcpo_path}")

    _print_summary(random_metrics, lcpo_df)
