"""
update_results_matrix.py
------------------------
Reads verified per-run CSV files from results/ and writes mean ± std summaries
into AUROC_Results_Matrix_by_Dr_Guo.xlsx.

Usage
-----
    # Preview what would be written (no file changes):
    PYTHONPATH=src python src/update_results_matrix.py --dry-run

    # Write all verified results to the Excel file:
    PYTHONPATH=src python src/update_results_matrix.py

    # Write from a specific CSV only:
    PYTHONPATH=src python src/update_results_matrix.py \
        --csv results/matched/lgbm_matched_results.csv

Design rules
------------
- Only rows >= 12 (matched-protocol rows) may be written. Rows 7–11 are
  historical baselines that must never be overwritten by this script.
- A cell is written as "mean ± std" (3 d.p.) when verified data exists.
- A cell is left unchanged if no data is available for that model+week+split.
- The per-seed / per-fold CSVs in results/ are the source of truth; this
  script only summarises them. It never modifies those CSVs.
- All paths via config.py constants.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from config import RESULTS_DIR  # noqa: E402

PROJECT_ROOT = _SRC_DIR.parent
EXCEL_PATH = PROJECT_ROOT / "AUROC_Results_Matrix_by_Dr_Guo.xlsx"
MATCHED_DIR = RESULTS_DIR / "matched"
GRAPH_DIR = RESULTS_DIR / "graph"
ZENODO_DIR = RESULTS_DIR / "zenodo"

# ── Excel layout constants ───────────────────────────────────────────────────
# First row that this script is allowed to write (rows 7-11 are protected).
FIRST_WRITABLE_ROW = 12

# Sheet name → metric column in results CSVs
SHEET_TO_METRIC = {
    "Results":           "auroc",
    "AUPRC":             "auprc",
    "F1":                "f1",
    "Balanced Accuracy": "balanced_acc",
    "Precision":         "precision",
    "Recall":            "recall",
}

# Column letters in the Excel sheet
# Random-student splits: weeks 2,4,6,8 → cols B,C,D,E
# LCPO splits:           weeks 2,4,6,8 → cols F,G,H,I
# Future presentation:   weeks 2,4,6,8 → cols J,K,L,M  (not populated here)
# Zenodo random W8 → col N;  Zenodo LCPO W8 → col O
_RANDOM_WEEK_TO_COL = {2: "B", 4: "C", 6: "D", 8: "E"}
_LCPO_WEEK_TO_COL   = {2: "F", 4: "G", 6: "H", 8: "I"}
_ZENODO_COL         = {"random": "N", "lcpo": "O"}

# Model name (as it appears in results CSVs) → Excel row number
# Only rows >= FIRST_WRITABLE_ROW are listed.
MODEL_TO_ROW = {
    # Matched comparison rows
    "LightGBM_matched":                        12,
    "MLP_all_features":                        16,
    "MLP_behavioral_summaries":                17,
    # GraphSAGE variants
    "GraphSAGE_base_weighted":                 18,
    "GraphSAGE_base_unweighted":               19,
    "GraphSAGE_interaction_attrs":             20,
    "GraphSAGE_enrollment_summaries":          21,
    "GraphSAGE_enrollment_node":               22,
    "GraphSAGE_edge_aware_mp":                 23,
    "GraphSAGE_course_design":                 24,
    "GraphSAGE_temporal":                      25,
    "GraphSAGE_course_conditioned":            26,
    # Architecture comparison
    "GCN":                                     27,
    "RGCN":                                    28,
    "HGT":                                     29,
    # Ablation rows (existing verified results — script will not overwrite
    # these unless a CSV entry explicitly uses one of these model names)
    "GraphSAGE_no_edge_attrs":                 30,
    "GraphSAGE_no_temporal":                   31,
    "GraphSAGE_no_course":                     32,
    "GraphSAGE_no_vle":                        33,
    "GraphSAGE_no_assessment":                 34,
}

# Legacy condition names from ablation_results.csv → model name key above
CONDITION_TO_MODEL = {
    "full":                    None,   # base condition; handled by random_student_results
    "no_assessment":           "GraphSAGE_no_assessment",
    "no_vle":                  "GraphSAGE_no_vle",
    "no_course_features":      "GraphSAGE_no_course",
    "no_temporal":             "GraphSAGE_no_temporal",
    "no_edge_attrs":           "GraphSAGE_no_edge_attrs",
    "with_enrollment_summaries": "GraphSAGE_enrollment_summaries",
    "with_interaction_attrs":  "GraphSAGE_interaction_attrs",
    "enrollment_node":         "GraphSAGE_enrollment_node",
    "edge_aware_mp":           "GraphSAGE_edge_aware_mp",
    "course_design":           "GraphSAGE_course_design",
    "temporal_features":       "GraphSAGE_temporal",
    "course_conditioned":      "GraphSAGE_course_conditioned",
    "gcn":                     "GCN",
    "rgcn":                    "RGCN",
    "hgt":                     "HGT",
}

# loss_weighting value that corresponds to the "base" published result
BASE_LOSS_WEIGHTING = "weighted"


# ── Helper functions ─────────────────────────────────────────────────────────

def _fmt(mean: float, std: float) -> str:
    """Format a mean ± std string to 3 decimal places."""
    return f"{mean:.3f} ± {std:.3f}"


def _summarise(df: pd.DataFrame, metric: str) -> str | None:
    """Return 'mean ± std' string for the given metric over all rows in df.
    Returns None if df is empty or the metric column is missing."""
    if df is None or df.empty or metric not in df.columns:
        return None
    values = df[metric].dropna()
    if len(values) == 0:
        return None
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return _fmt(mean, std)


def _col_letter_to_index(letter: str) -> int:
    """Convert column letter (A=1, B=2, …) to 1-based openpyxl index."""
    letter = letter.upper()
    result = 0
    for ch in letter:
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result


# ── CSV loaders ──────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_all_sources() -> dict[str, pd.DataFrame | None]:
    return {
        "lgbm_matched":    _load(MATCHED_DIR / "lgbm_matched_results.csv"),
        "mlp_matched":     _load(MATCHED_DIR / "mlp_matched_results.csv"),
        "gnn_random":      _load(GRAPH_DIR / "random_student_results.csv"),
        "gnn_lcpo":        _load(GRAPH_DIR / "lcpo_results.csv"),
        "ablation":        _load(GRAPH_DIR / "ablation_results.csv"),
        "zenodo_random":   _load(ZENODO_DIR / "random_student_results.csv"),
        "zenodo_lcpo":     _load(ZENODO_DIR / "lcpo_results.csv"),
    }


# ── Build the update plan ────────────────────────────────────────────────────

def build_update_plan(sources: dict) -> list[dict]:
    """Return a list of update records, each with:
       sheet, row, col_letter, value (formatted string)
    """
    updates: list[dict] = []

    def _add(sheet: str, row: int, col: str, value: str | None) -> None:
        if value is None:
            return
        if row < FIRST_WRITABLE_ROW:
            return  # never touch protected rows
        updates.append({"sheet": sheet, "row": row, "col": col, "value": value})

    # ── Matched LightGBM (row 12) ──────────────────────────────────────────
    lgbm = sources.get("lgbm_matched")
    if lgbm is not None:
        for sheet, metric in SHEET_TO_METRIC.items():
            for week, col in _RANDOM_WEEK_TO_COL.items():
                sub = lgbm[(lgbm["week"] == week) & (lgbm["split_type"] == "random")]
                _add(sheet, 12, col, _summarise(sub, metric))
            for week, col in _LCPO_WEEK_TO_COL.items():
                sub = lgbm[(lgbm["week"] == week) & (lgbm["split_type"] == "lcpo")]
                _add(sheet, 12, col, _summarise(sub, metric))

    # ── Matched MLP — all enrollment features (row 16) ────────────────────
    # ── MLP — six behavioral summaries only (row 17) ──────────────────────
    mlp = sources.get("mlp_matched")
    if mlp is not None:
        for sheet, metric in SHEET_TO_METRIC.items():
            for week, col in _RANDOM_WEEK_TO_COL.items():
                sub_all = mlp[
                    (mlp["week"] == week)
                    & (mlp["split_type"] == "random")
                    & (mlp["model"] == "MLP_all_features")
                ]
                _add(sheet, 16, col, _summarise(sub_all, metric))
                sub_beh = mlp[
                    (mlp["week"] == week)
                    & (mlp["split_type"] == "random")
                    & (mlp["model"] == "MLP_behavioral_summaries")
                ]
                _add(sheet, 17, col, _summarise(sub_beh, metric))
            for week, col in _LCPO_WEEK_TO_COL.items():
                sub_all = mlp[
                    (mlp["week"] == week)
                    & (mlp["split_type"] == "lcpo")
                    & (mlp["model"] == "MLP_all_features")
                ]
                _add(sheet, 16, col, _summarise(sub_all, metric))
                sub_beh = mlp[
                    (mlp["week"] == week)
                    & (mlp["split_type"] == "lcpo")
                    & (mlp["model"] == "MLP_behavioral_summaries")
                ]
                _add(sheet, 17, col, _summarise(sub_beh, metric))

    # ── GNN random-student results ─────────────────────────────────────────
    # Maps rows 18 (weighted base), 19 (unweighted base), 20 (interaction attrs)
    gnn_rand = sources.get("gnn_random")
    if gnn_rand is not None:
        # Base weighted → row 18
        for sheet, metric in SHEET_TO_METRIC.items():
            for week, col in _RANDOM_WEEK_TO_COL.items():
                sub = gnn_rand[
                    (gnn_rand["week"] == week)
                    & (gnn_rand.get("loss_weighting", pd.Series(["weighted"] * len(gnn_rand))) == "weighted")
                    & (gnn_rand.get("condition", pd.Series([""] * len(gnn_rand))).fillna("") == "")
                ]
                _add(sheet, 18, col, _summarise(sub, metric))
        # Base unweighted → row 19
        for sheet, metric in SHEET_TO_METRIC.items():
            for week, col in _RANDOM_WEEK_TO_COL.items():
                sub = gnn_rand[
                    (gnn_rand["week"] == week)
                    & (gnn_rand.get("loss_weighting", pd.Series(["weighted"] * len(gnn_rand))) == "unweighted")
                    & (gnn_rand.get("condition", pd.Series([""] * len(gnn_rand))).fillna("") == "")
                ]
                _add(sheet, 19, col, _summarise(sub, metric))

    # ── Ablation results (conditions → model name → Excel row) ────────────
    ablation = sources.get("ablation")
    if ablation is not None and "condition" in ablation.columns:
        for cond, model_name in CONDITION_TO_MODEL.items():
            if model_name is None:
                continue
            row = MODEL_TO_ROW.get(model_name)
            if row is None:
                continue
            cond_df = ablation[ablation["condition"] == cond]
            if cond_df.empty:
                continue
            for sheet, metric in SHEET_TO_METRIC.items():
                for week, col in _RANDOM_WEEK_TO_COL.items():
                    sub = cond_df[cond_df["week"] == week]
                    _add(sheet, row, col, _summarise(sub, metric))

    # ── GNN LCPO results → row 18 (base weighted only for now) ────────────
    gnn_lcpo = sources.get("gnn_lcpo")
    if gnn_lcpo is not None:
        for sheet, metric in SHEET_TO_METRIC.items():
            for week, col in _LCPO_WEEK_TO_COL.items():
                sub = gnn_lcpo[gnn_lcpo["week"] == week]
                if sub.empty:
                    continue
                # Use per-fold mean across model seeds as one observation per fold,
                # then mean ± std across folds
                fold_col = next(
                    (c for c in ["fold_idx", "fold"] if c in sub.columns), None
                )
                if fold_col and metric in sub.columns:
                    per_fold = sub.groupby(fold_col)[metric].mean()
                    if len(per_fold) > 0:
                        m = per_fold.mean()
                        s = per_fold.std(ddof=1) if len(per_fold) > 1 else 0.0
                        _add(sheet, 18, col, _fmt(m, s))

    # ── Zenodo results → cols N (random W8) and O (LCPO W8) ───────────────
    zen_rand = sources.get("zenodo_random")
    if zen_rand is not None and "model" in zen_rand.columns:
        for sheet, metric in SHEET_TO_METRIC.items():
            # LightGBM matched on Zenodo → row 12, col N
            sub_lgbm = zen_rand[
                zen_rand["model"].str.contains("LightGBM", case=False, na=False)
            ]
            _add(sheet, 12, "N", _summarise(sub_lgbm, metric))
            # GraphSAGE base on Zenodo → row 18, col N
            sub_gnn = zen_rand[
                zen_rand["model"].str.contains("GraphSAGE|GNN|Enrollment", case=False, na=False)
            ]
            _add(sheet, 18, "N", _summarise(sub_gnn, metric))

    zen_lcpo = sources.get("zenodo_lcpo")
    if zen_lcpo is not None:
        for sheet, metric in SHEET_TO_METRIC.items():
            if metric not in zen_lcpo.columns:
                continue
            fold_col = next(
                (c for c in ["fold_idx", "fold"] if c in zen_lcpo.columns), None
            )
            model_col = "model" if "model" in zen_lcpo.columns else None
            if fold_col and model_col:
                for model_name, row in [("LightGBM", 12), ("GraphSAGE", 18)]:
                    sub = zen_lcpo[
                        zen_lcpo[model_col].str.contains(model_name, case=False, na=False)
                    ]
                    if sub.empty:
                        continue
                    per_fold = sub.groupby(fold_col)[metric].mean()
                    if len(per_fold) > 0:
                        m = per_fold.mean()
                        s = per_fold.std(ddof=1) if len(per_fold) > 1 else 0.0
                        _add(sheet, row, "O", _fmt(m, s))

    return updates


# ── Apply the plan to Excel ───────────────────────────────────────────────────

def update_excel(
    results_csv_path: Path | None = None,
    excel_path: Path = EXCEL_PATH,
    dry_run: bool = False,
) -> list[dict]:
    """Apply all updates to the Excel file.

    Parameters
    ----------
    results_csv_path : optional path to a single CSV to load (used by tests).
        When None, all standard CSVs are loaded.
    excel_path : path to the Excel workbook to update.
    dry_run : if True, print the plan but do not write anything.

    Returns
    -------
    List of update records that were (or would be) applied.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "openpyxl is required to write to Excel files. "
            "Install it with: pip install openpyxl"
        )

    if results_csv_path is not None:
        # Single-CSV mode: used by tests to inject synthetic data.
        csv_df = _load(Path(results_csv_path))
        # Try to determine which source key this CSV belongs to.
        sources: dict = {k: None for k in [
            "lgbm_matched", "mlp_matched", "gnn_random", "gnn_lcpo",
            "ablation", "zenodo_random", "zenodo_lcpo",
        ]}
        if csv_df is not None and "model" in csv_df.columns:
            model_vals = csv_df["model"].unique()
            if any("LightGBM" in str(m) for m in model_vals):
                sources["lgbm_matched"] = csv_df
            elif any("MLP" in str(m) for m in model_vals):
                sources["mlp_matched"] = csv_df
        if csv_df is not None and "condition" in csv_df.columns:
            sources["ablation"] = csv_df
    else:
        sources = _load_all_sources()

    updates = build_update_plan(sources)

    if dry_run:
        print(f"\n{'DRY RUN':=^60}")
        print(f"Excel target: {excel_path}")
        print(f"Total cells to update: {len(updates)}\n")
        if updates:
            print(f"{'Sheet':<22} {'Cell':<6} {'Value'}")
            print("-" * 50)
            for u in sorted(updates, key=lambda x: (x["sheet"], x["row"], x["col"])):
                cell = f"{u['col']}{u['row']}"
                print(f"{u['sheet']:<22} {cell:<6} {u['value']}")
        else:
            print("No updates to apply (no verified results found).")
        return updates

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if not updates:
        print("No updates to apply — all matched-protocol CSVs are empty or missing.")
        return updates

    wb = openpyxl.load_workbook(excel_path)
    applied = 0

    for u in updates:
        sheet_name = u["sheet"]
        row = u["row"]
        col_idx = _col_letter_to_index(u["col"])
        value = u["value"]

        if sheet_name not in wb.sheetnames:
            print(f"  WARNING: Sheet '{sheet_name}' not found — skipping")
            continue
        if row < FIRST_WRITABLE_ROW:
            print(f"  BLOCKED: Row {row} is a protected historical row — skipping")
            continue

        ws = wb[sheet_name]
        ws.cell(row=row, column=col_idx).value = value
        applied += 1

    wb.save(excel_path)
    print(f"Updated {applied} cells in {excel_path.name}")
    return updates


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate AUROC_Results_Matrix_by_Dr_Guo.xlsx from verified CSVs."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying the Excel file.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: load a single CSV instead of all standard result files.",
    )
    p.add_argument(
        "--excel",
        type=Path,
        default=EXCEL_PATH,
        metavar="PATH",
        help=f"Path to the Excel workbook (default: {EXCEL_PATH.name})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    update_excel(
        results_csv_path=args.csv,
        excel_path=args.excel,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
