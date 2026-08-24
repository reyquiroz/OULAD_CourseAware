"""reporting_schema.py — Canonical column schemas for cross-dataset reporting.

All result CSVs that will eventually support multiple datasets must include
a `dataset` column as the primary grouping key. Results MUST NEVER be pooled
across datasets.
"""

DATASET_COL = "dataset"

METRIC_COLS = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]

RANDOM_RESULT_COLS = [
    "dataset",
    "week",
    "model",
    "split",
    "loss_weighting",
    "seed",
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "balanced_acc",
    "best_val_auroc",
    "best_epoch",
    "best_threshold",
]

LCPO_RESULT_COLS = [
    "dataset",
    "week",
    "fold_idx",
    "held_out_module",
    "held_out_presentation",
    "n_train",
    "n_test",
    "model_seed",
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "balanced_acc",
    "best_val_auroc",
    "best_epoch",
]

COMPARISON_RESULT_COLS = [
    "dataset",
    "week",
    "model",
    "split_type",
    "fold_or_seed",
    "held_out_module",
    "held_out_presentation",
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "balanced_acc",
]

GROUPING_RULE = "Always group by dataset before any aggregation."
