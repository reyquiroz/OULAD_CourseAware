"""
summarize_lcpo_modules.py — Aggregate LCPO results to the module level.

Reads:
  results/graph/lcpo_results.csv     — one row per (fold, seed), GNN
  results/graph/comparison_results.csv — combined GNN + LightGBM per-fold rows

Writes:
  results/graph/lcpo_module_summary.csv

Groups by (held_out_module, model) and computes mean ± std for AUROC, AUPRC,
F1, and Balanced_Acc across all folds × seeds within that module.
Also records n_presentations and n_enrollments_total (sum of n_test).

Usage
-----
    python src/summarize_lcpo_modules.py          # default: week 8
    python src/summarize_lcpo_modules.py --week 8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to project root — script is expected to be run from root)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_GRAPH_DIR = _PROJECT_ROOT / "results" / "graph"

_LCPO_RESULTS_PATH = _GRAPH_DIR / "lcpo_results.csv"
_COMPARISON_RESULTS_PATH = _GRAPH_DIR / "comparison_results.csv"
_FOLDS_TEMPLATE = _GRAPH_DIR / "evaluation" / "week{week:02d}" / "splits" / "week{week:02d}_lcpo_folds.csv"
_OUTPUT_PATH = _GRAPH_DIR / "lcpo_module_summary.csv"

_METRIC_COLS = ["auroc", "auprc", "f1", "balanced_acc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_folds(week: int) -> pd.DataFrame:
    """Load the LCPO fold definitions for *week* (provides n_test per fold)."""
    path = Path(str(_FOLDS_TEMPLATE).format(week=week, week02=f"{week:02d}"))
    # Use the template string approach to handle the double-substitution
    path = _GRAPH_DIR / "evaluation" / f"week{week:02d}" / "splits" / f"week{week:02d}_lcpo_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Folds CSV not found: {path}")
    return pd.read_csv(path)


def _agg_module(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Group *df* by held_out_module and compute mean ± std per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: held_out_module, held_out_presentation, + _METRIC_COLS.
        For GNN this has one row per (fold, seed); for LightGBM one row per fold.
    model_label : str
        String to assign in the 'model' column of the output.

    Returns
    -------
    pd.DataFrame with columns:
        held_out_module, model,
        {metric}_mean, {metric}_std  for each metric in _METRIC_COLS,
        n_presentations, n_enrollments_total (if n_test present in df)
    """
    records = []
    for module, grp in df.groupby("held_out_module", sort=True):
        row: dict = {"held_out_module": module, "model": model_label}
        for m in _METRIC_COLS:
            if m in grp.columns:
                row[f"{m}_mean"] = float(grp[m].mean())
                row[f"{m}_std"] = float(grp[m].std(ddof=1)) if len(grp) > 1 else float("nan")
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
        row["n_presentations"] = int(grp["held_out_presentation"].nunique())
        if "n_test" in grp.columns:
            # n_test is per fold; for multi-seed GNN rows sum at the fold level first
            fold_n_test = grp.drop_duplicates(subset=["fold_idx"])["n_test"] if "fold_idx" in grp.columns else grp["n_test"]
            row["n_enrollments_total"] = int(fold_n_test.sum())
        else:
            row["n_enrollments_total"] = int("nan") if False else -1
        records.append(row)
    return pd.DataFrame(records)


def _fmt_table(df: pd.DataFrame) -> str:
    """Return a simple formatted string table for stdout."""
    col_width = 14
    metric_pairs = [(m, f"{m}_mean", f"{m}_std") for m in _METRIC_COLS]
    header_parts = [f"{'Module':<8}", f"{'Model':<12}"]
    for m, _, _ in metric_pairs:
        header_parts.append(f"{m.upper():>{col_width}}")
    header_parts += [f"{'N_pres':>7}", f"{'N_enroll':>9}"]
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for _, r in df.iterrows():
        parts = [f"{r['held_out_module']:<8}", f"{r['model']:<12}"]
        for _, mean_col, std_col in metric_pairs:
            mean_v = r.get(mean_col, float("nan"))
            std_v = r.get(std_col, float("nan"))
            if pd.isna(mean_v):
                parts.append(f"{'—':>{col_width}}")
            elif pd.isna(std_v):
                parts.append(f"{mean_v:>{col_width}.4f}")
            else:
                cell = f"{mean_v:.4f}±{std_v:.4f}"
                parts.append(f"{cell:>{col_width}}")
        parts += [f"{int(r.get('n_presentations', -1)):>7}",
                  f"{int(r.get('n_enrollments_total', -1)):>9}"]
        lines.append("  ".join(parts))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_module_summary(week: int = 8) -> pd.DataFrame:
    """Compute module-level summaries for GNN and LightGBM, merge, and save.

    Returns the merged module summary DataFrame.
    """
    # ---- Load GNN per-seed-per-fold results ----
    if not _LCPO_RESULTS_PATH.exists():
        print(f"ERROR: {_LCPO_RESULTS_PATH} not found. Run run_gnn_experiment.py first.",
              file=sys.stderr)
        sys.exit(1)

    gnn_df = pd.read_csv(_LCPO_RESULTS_PATH)
    gnn_week = gnn_df[gnn_df["week"] == week].copy() if "week" in gnn_df.columns else gnn_df.copy()

    if gnn_week.empty:
        print(f"WARNING: No GNN LCPO rows found for week {week}.", file=sys.stderr)

    # ---- Load fold n_test (for n_enrollments_total) and merge into GNN df ----
    folds_df = _load_folds(week)
    # folds_df has: fold_idx, held_out_module, held_out_presentation, n_train, n_test
    if "n_test" not in gnn_week.columns and "fold_idx" in gnn_week.columns:
        gnn_week = gnn_week.merge(
            folds_df[["fold_idx", "n_test"]],
            on="fold_idx", how="left"
        )

    # ---- Load LightGBM LCPO rows from comparison_results.csv ----
    lgbm_df = pd.DataFrame()
    if _COMPARISON_RESULTS_PATH.exists():
        comp_df = pd.read_csv(_COMPARISON_RESULTS_PATH)
        lgbm_week = comp_df[
            (comp_df["split_type"] == "lcpo") &
            (comp_df["model"] == "LightGBM")
        ].copy()
        if "week" in lgbm_week.columns:
            lgbm_week = lgbm_week[lgbm_week["week"] == week]
        if not lgbm_week.empty:
            # Merge n_test from folds
            if "n_test" not in lgbm_week.columns and "fold_or_seed" in lgbm_week.columns:
                lgbm_week = lgbm_week.merge(
                    folds_df[["fold_idx", "n_test"]].rename(columns={"fold_idx": "fold_or_seed"}),
                    on="fold_or_seed", how="left"
                )
            lgbm_df = lgbm_week
    else:
        print(f"WARNING: {_COMPARISON_RESULTS_PATH} not found — LightGBM rows will be absent.",
              file=sys.stderr)

    # ---- Aggregate to module level ----
    gnn_module = _agg_module(gnn_week, "GNN") if not gnn_week.empty else pd.DataFrame()
    lgbm_module = _agg_module(lgbm_df, "LightGBM") if not lgbm_df.empty else pd.DataFrame()

    # ---- Stack both models into one summary frame ----
    frames = [f for f in [gnn_module, lgbm_module] if not f.empty]
    if not frames:
        print("ERROR: No data to summarise.", file=sys.stderr)
        sys.exit(1)

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(["held_out_module", "model"]).reset_index(drop=True)

    # ---- Write output ----
    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(_OUTPUT_PATH, index=False)
    print(f"[summarize_lcpo_modules] Wrote {len(summary)} rows → {_OUTPUT_PATH}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate LCPO results to module level (GNN and LightGBM)."
    )
    parser.add_argument("--week", type=int, default=8,
                        help="Prediction week to summarise (default: 8).")
    args = parser.parse_args()

    summary = build_module_summary(week=args.week)

    print(f"\n{'='*60}")
    print(f"Module-level LCPO summary  (week {args.week})")
    print(f"{'='*60}")
    print(_fmt_table(summary))
    print(f"\nModules: {sorted(summary['held_out_module'].unique())}")
    gnn_rows = summary[summary["model"] == "GNN"]
    lgbm_rows = summary[summary["model"] == "LightGBM"]
    if not gnn_rows.empty:
        print(f"GNN  — across-module mean AUROC: {gnn_rows['auroc_mean'].mean():.4f}")
    if not lgbm_rows.empty:
        print(f"LGBM — across-module mean AUROC: {lgbm_rows['auroc_mean'].mean():.4f}")


if __name__ == "__main__":
    main()
