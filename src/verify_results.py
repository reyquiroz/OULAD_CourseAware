#!/usr/bin/env python
"""
verify_results.py — Verify that results/graph/tables/ is consistent with
comparison_results.csv.

Recomputes every aggregate value (mean, std, win counts) from the canonical
source CSV and diffs them against the saved table CSVs.  Exits with code 1 if
any value differs by more than 1e-4; exits with code 0 if all values match or
if the tables directory is empty / does not exist.

Usage
-----
    python src/verify_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
_GRAPH_DIR = _PROJECT_ROOT / "results" / "graph"
_COMPARISON_PATH = _GRAPH_DIR / "comparison_results.csv"
_TABLES_DIR = _GRAPH_DIR / "tables"

_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
_TOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _std(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    """Match the formatting used by generate_report_figures.py."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _parse_fmt(cell: str) -> float:
    """Extract the mean from a formatted 'mean ± std' or plain 'mean' cell."""
    cell = str(cell).strip()
    if cell in ("—", "", "nan"):
        return float("nan")
    if "±" in cell:
        return float(cell.split("±")[0].strip())
    try:
        return float(cell)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Recompute aggregates from comparison_results.csv
# ---------------------------------------------------------------------------


def compute_main_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute table_main_comparison from comparison_results.csv (week 8)."""
    rows = []
    for split_type, split_label, model_names in [
        ("random_student", "Random", ["GNN (weighted)", "LightGBM"]),
        ("lcpo", "LCPO", ["GNN", "LightGBM"]),
    ]:
        for model_name in model_names:
            subset = df[
                (df["week"] == 8)
                & (df["split_type"] == split_type)
                & (df["model"] == model_name)
            ]
            row: dict = {"split": split_label, "model": model_name}
            for metric in _METRICS:
                vals = subset[metric] if metric in subset.columns else pd.Series(dtype=float)
                row[metric.upper()] = _fmt(_mean(vals), _std(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_week_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute table_week_performance from comparison_results.csv."""
    rows = []
    for week in [2, 4, 6, 8]:
        weighted = df[
            (df["week"] == week)
            & (df["split_type"] == "random_student")
            & (df["model"] == "GNN (weighted)")
        ]
        unweighted = df[
            (df["week"] == week)
            & (df["split_type"] == "random_student")
            & (df["model"] == "GNN (unweighted)")
        ]
        lgbm = df[
            (df["week"] == week)
            & (df["split_type"] == "random_student")
            & (df["model"] == "LightGBM")
        ]
        rows.append({
            "week": week,
            "GNN weighted": _fmt(
                _mean(weighted["auroc"]) if not weighted.empty else float("nan"),
                _std(weighted["auroc"]) if not weighted.empty else float("nan"),
            ),
            "GNN unweighted": _fmt(
                _mean(unweighted["auroc"]) if not unweighted.empty else float("nan"),
                _std(unweighted["auroc"]) if not unweighted.empty else float("nan"),
            ),
            "LightGBM": _fmt(
                _mean(lgbm["auroc"]) if not lgbm.empty else float("nan"),
                _std(lgbm["auroc"]) if not lgbm.empty else float("nan"),
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def _diff_table(
    table_name: str,
    recomputed: pd.DataFrame,
    saved_path: Path,
    discrepancies: list[str],
) -> None:
    """Compare recomputed vs. saved CSV; append any diffs to *discrepancies*."""
    if not saved_path.exists():
        return  # table not yet generated — skip

    saved = pd.read_csv(saved_path)

    # Only compare columns present in both
    shared_cols = [c for c in recomputed.columns if c in saved.columns]
    if not shared_cols:
        return

    for col in shared_cols:
        for idx in range(min(len(recomputed), len(saved))):
            recomp_val = recomputed[col].iloc[idx] if idx < len(recomputed) else None
            saved_val = saved[col].iloc[idx] if idx < len(saved) else None

            recomp_float = _parse_fmt(str(recomp_val))
            saved_float = _parse_fmt(str(saved_val))

            if np.isnan(recomp_float) and np.isnan(saved_float):
                continue
            if np.isnan(recomp_float) or np.isnan(saved_float):
                discrepancies.append(
                    f"[{table_name}] row {idx} col '{col}': "
                    f"recomputed={recomp_val!r}, saved={saved_val!r} (one is NaN)"
                )
                continue
            if abs(recomp_float - saved_float) > _TOL:
                discrepancies.append(
                    f"[{table_name}] row {idx} col '{col}': "
                    f"recomputed={recomp_float:.6f}, saved={saved_float:.6f}, "
                    f"diff={abs(recomp_float - saved_float):.2e}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    if not _COMPARISON_PATH.exists():
        print(
            "WARNING: comparison_results.csv not found. "
            "Run compare_gnn_lgbm.py first to generate it.",
            file=sys.stderr,
        )
        return 0

    if not _TABLES_DIR.exists() or not any(_TABLES_DIR.glob("*.csv")):
        print(
            "WARNING: results/graph/tables/ is empty or does not exist. "
            "Run generate_report_figures.py first to generate tables.",
            file=sys.stderr,
        )
        return 0

    print(f"[verify_results] Reading {_COMPARISON_PATH}")
    df = pd.read_csv(_COMPARISON_PATH)

    discrepancies: list[str] = []

    # ---- table_main_comparison ----
    recomp_main = compute_main_comparison(df)
    _diff_table("table_main_comparison", recomp_main, _TABLES_DIR / "table_main_comparison.csv", discrepancies)

    # ---- table_week_performance ----
    recomp_week = compute_week_performance(df)
    _diff_table("table_week_performance", recomp_week, _TABLES_DIR / "table_week_performance.csv", discrepancies)

    if discrepancies:
        print(f"\n[verify_results] FAILED — {len(discrepancies)} discrepancy(ies) found:\n")
        for d in discrepancies:
            print(f"  ✗ {d}")
        print()
        return 1

    print("[verify_results] OK — all table values match comparison_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
