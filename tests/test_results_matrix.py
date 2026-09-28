"""
Tests that verify results destined for AUROC_Results_Matrix_by_Dr_Guo.xlsx are
correct before they are written.

Three test groups:

1. Writer correctness — given a synthetic results CSV, does update_results_matrix.py
   map values to the right Excel cells and compute mean ± std correctly?

2. Protocol invariants — are existing results CSVs internally consistent with the
   matched protocol (enrollment counts, threshold values, AUROC plausibility)?

3. Historical row protection — does the writer leave already-verified rows untouched?

Run from the project root:
    pytest tests/test_results_matrix.py -v
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
MATCHED_DIR = RESULTS_DIR / "matched"
GRAPH_DIR = RESULTS_DIR / "graph"
EXCEL_PATH = PROJECT_ROOT / "AUROC_Results_Matrix_by_Dr_Guo.xlsx"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

WEEKS = [2, 4, 6, 8]
SEEDS = [42, 123, 7, 17, 99]
PLAUSIBLE_AUROC_MIN = 0.55   # below this on OULAD is a failure signal
PLAUSIBLE_AUROC_MAX = 0.99   # above this is suspicious (data leakage)


def _load_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Group 1 — Writer correctness
# ---------------------------------------------------------------------------

class TestWriterCorrectness:
    """Tests for update_results_matrix.py cell-mapping and aggregation logic."""

    @pytest.fixture()
    def synthetic_lgbm_csv(self, tmp_path):
        """Five-seed week-8 random-split LightGBM result with known mean/std."""
        auroc_values = [0.85, 0.86, 0.84, 0.87, 0.83]
        rows = []
        for i, seed in enumerate(SEEDS):
            rows.append({
                "model": "LightGBM_matched",
                "week": 8,
                "split_type": "random",
                "seed": seed,
                "fold": -1,
                "auroc": auroc_values[i],
                "auprc": auroc_values[i] - 0.01,
                "f1": auroc_values[i] - 0.08,
                "precision": auroc_values[i] - 0.05,
                "recall": auroc_values[i] - 0.06,
                "balanced_acc": auroc_values[i] - 0.09,
                "threshold": 0.40,
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "lgbm_matched_results.csv"
        df.to_csv(csv_path, index=False)
        return csv_path, df

    @pytest.fixture()
    def excel_copy(self, tmp_path):
        """Working copy of the Excel file so we never touch the original."""
        if not EXCEL_PATH.exists():
            pytest.skip("Excel file not found — skipping writer tests")
        dest = tmp_path / EXCEL_PATH.name
        shutil.copy2(EXCEL_PATH, dest)
        return dest

    def test_mean_std_computation(self, synthetic_lgbm_csv):
        """Mean and std of the synthetic data must match pandas computation."""
        _, df = synthetic_lgbm_csv
        random_rows = df[df["split_type"] == "random"]
        expected_mean = round(random_rows["auroc"].mean(), 4)
        expected_std = round(random_rows["auroc"].std(ddof=1), 4)
        assert 0.83 <= expected_mean <= 0.87, "Mean AUROC out of expected range"
        assert expected_std >= 0.0, "Std dev must be non-negative"
        # Confirm the formatted string is the expected format
        formatted = f"{expected_mean:.3f} ± {expected_std:.3f}"
        assert "±" in formatted

    def test_writer_maps_to_correct_row_column(self, synthetic_lgbm_csv, excel_copy):
        """update_results_matrix.py must write LightGBM_matched week-8 random
        result to Results sheet row 12, column E (AUROC)."""
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        try:
            import update_results_matrix as urm  # noqa: F401
        except ImportError:
            pytest.skip("update_results_matrix.py not on sys.path")

        csv_path, _ = synthetic_lgbm_csv
        urm.update_excel(
            results_csv_path=csv_path,
            excel_path=excel_copy,
            dry_run=False,
        )

        wb = openpyxl.load_workbook(excel_copy)
        ws = wb["Results"]
        cell_value = ws["E12"].value
        assert cell_value is not None, "Cell E12 should have been written"
        assert "±" in str(cell_value), (
            f"Expected 'mean ± std' format in E12, got: {cell_value}"
        )
        wb.close()

    def test_writer_does_not_overwrite_historical_rows(self, synthetic_lgbm_csv, excel_copy):
        """Writer must not change rows 7–11 (historical baselines)."""
        try:
            import openpyxl
            import update_results_matrix as urm
        except ImportError:
            pytest.skip("openpyxl or update_results_matrix not available")

        # Snapshot rows 7–11 before writing
        wb_before = openpyxl.load_workbook(excel_copy)
        ws_before = wb_before["Results"]
        historical_before = {
            (r, c): ws_before.cell(row=r, column=c).value
            for r in range(7, 12)
            for c in range(1, 16)
        }
        wb_before.close()

        csv_path, _ = synthetic_lgbm_csv
        urm.update_excel(
            results_csv_path=csv_path,
            excel_path=excel_copy,
            dry_run=False,
        )

        wb_after = openpyxl.load_workbook(excel_copy)
        ws_after = wb_after["Results"]
        historical_after = {
            (r, c): ws_after.cell(row=r, column=c).value
            for r in range(7, 12)
            for c in range(1, 16)
        }
        wb_after.close()

        assert historical_before == historical_after, (
            "Historical rows 7–11 were modified by the writer. "
            "Only matched-protocol rows (12+) should be updated."
        )

    def test_dry_run_does_not_modify_excel(self, synthetic_lgbm_csv, excel_copy):
        """--dry-run must not change the Excel file at all."""
        try:
            import openpyxl
            import update_results_matrix as urm
        except ImportError:
            pytest.skip("openpyxl or update_results_matrix not available")

        before_bytes = excel_copy.read_bytes()
        csv_path, _ = synthetic_lgbm_csv
        urm.update_excel(
            results_csv_path=csv_path,
            excel_path=excel_copy,
            dry_run=True,
        )
        after_bytes = excel_copy.read_bytes()
        assert before_bytes == after_bytes, "--dry-run must not write to the Excel file"

    def test_missing_model_writes_dash(self, excel_copy):
        """If results CSV has no rows for a model+week, cell must stay '—'."""
        try:
            import openpyxl
            import update_results_matrix as urm
        except ImportError:
            pytest.skip("openpyxl or update_results_matrix not available")

        # Empty CSV — no rows for any model
        empty_csv = Path(tempfile.mktemp(suffix=".csv"))
        pd.DataFrame(columns=[
            "model", "week", "split_type", "seed", "fold",
            "auroc", "auprc", "f1", "precision", "recall", "balanced_acc", "threshold",
        ]).to_csv(empty_csv, index=False)

        try:
            urm.update_excel(
                results_csv_path=empty_csv,
                excel_path=excel_copy,
                dry_run=False,
            )
        finally:
            empty_csv.unlink(missing_ok=True)

        wb = openpyxl.load_workbook(excel_copy)
        ws = wb["Results"]
        cell = ws["E12"].value  # LightGBM matched, week 8 random
        wb.close()
        assert cell in ("—", None, ""), (
            f"Expected '—' or empty for missing data in E12, got '{cell}'"
        )

    def test_all_six_sheets_updated(self, synthetic_lgbm_csv, excel_copy):
        """Writer must update all six metric sheets (AUROC, AUPRC, F1, etc.)."""
        try:
            import openpyxl
            import update_results_matrix as urm
        except ImportError:
            pytest.skip("openpyxl or update_results_matrix not available")

        csv_path, _ = synthetic_lgbm_csv
        urm.update_excel(
            results_csv_path=csv_path,
            excel_path=excel_copy,
            dry_run=False,
        )

        wb = openpyxl.load_workbook(excel_copy)
        sheets_with_data = []
        for sheet_name in ["Results", "AUPRC", "F1", "Balanced Accuracy", "Precision", "Recall"]:
            ws = wb[sheet_name]
            val = ws["E12"].value
            if val is not None and val != "—":
                sheets_with_data.append(sheet_name)
        wb.close()
        assert len(sheets_with_data) == 6, (
            f"Expected all 6 metric sheets updated, only found data in: {sheets_with_data}"
        )


# ---------------------------------------------------------------------------
# Group 2 — Protocol invariants on existing CSVs
# ---------------------------------------------------------------------------

class TestProtocolInvariants:
    """Validate that existing results CSVs satisfy the matched protocol."""

    def test_lgbm_matched_has_required_columns(self):
        """lgbm_matched_results.csv must have all metric columns."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        required = {
            "model", "week", "split_type", "seed", "fold",
            "auroc", "auprc", "f1", "precision", "recall", "balanced_acc", "threshold",
        }
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in lgbm_matched_results.csv: {missing}"

    def test_lgbm_matched_auroc_plausible(self):
        """All AUROC values in lgbm_matched_results.csv must be in plausible range."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        bad = df[(df["auroc"] < PLAUSIBLE_AUROC_MIN) | (df["auroc"] > PLAUSIBLE_AUROC_MAX)]
        assert len(bad) == 0, (
            f"Implausible AUROC in lgbm_matched_results.csv:\n"
            f"{bad[['model', 'week', 'seed', 'auroc']]}"
        )

    def test_mlp_matched_has_required_columns(self):
        """mlp_matched_results.csv must have all metric columns."""
        df = _load_csv_if_exists(MATCHED_DIR / "mlp_matched_results.csv")
        if df is None:
            pytest.skip("mlp_matched_results.csv not yet generated")
        required = {
            "model", "week", "split_type", "seed", "fold",
            "auroc", "auprc", "f1", "precision", "recall", "balanced_acc", "threshold",
        }
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in mlp_matched_results.csv: {missing}"

    def test_mlp_matched_auroc_plausible(self):
        """All AUROC values in mlp_matched_results.csv must be in plausible range."""
        df = _load_csv_if_exists(MATCHED_DIR / "mlp_matched_results.csv")
        if df is None:
            pytest.skip("mlp_matched_results.csv not yet generated")
        bad = df[(df["auroc"] < PLAUSIBLE_AUROC_MIN) | (df["auroc"] > PLAUSIBLE_AUROC_MAX)]
        assert len(bad) == 0, (
            f"Implausible AUROC in mlp_matched_results.csv:\n"
            f"{bad[['model', 'week', 'seed', 'auroc']]}"
        )

    def test_lgbm_and_mlp_same_run_count_per_week_split(self):
        """LightGBM and MLP matched must have identical run counts per week+split_type."""
        lgbm = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        mlp = _load_csv_if_exists(MATCHED_DIR / "mlp_matched_results.csv")
        if lgbm is None or mlp is None:
            pytest.skip("One or both matched result CSVs not yet generated")
        lgbm_counts = lgbm.groupby(["week", "split_type"]).size().sort_index()
        mlp_counts = mlp.groupby(["week", "split_type"]).size().sort_index()
        pd.testing.assert_series_equal(
            lgbm_counts, mlp_counts, check_names=False,
            obj="Run counts per week/split_type differ between LightGBM and MLP matched",
        )

    def test_no_fixed_half_threshold_in_lgbm_lcpo(self):
        """No LCPO fold in lgbm_matched_results.csv should have ALL seeds at threshold=0.5.
        That pattern signals the threshold bug was not fixed."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        lcpo = df[df["split_type"] == "lcpo"]
        if lcpo.empty:
            pytest.skip("No LCPO rows in lgbm_matched_results.csv yet")
        all_half = lcpo.groupby("fold")["threshold"].apply(lambda x: (x == 0.5).all())
        bad_folds = all_half[all_half].index.tolist()
        assert not bad_folds, (
            f"Folds {bad_folds} have ALL seeds at threshold=0.5 in LightGBM LCPO — "
            "check that F1-max threshold selection is applied correctly."
        )

    def test_no_fixed_half_threshold_in_gnn_lcpo(self):
        """No LCPO fold in lcpo_results.csv should have ALL model seeds at threshold=0.5.
        This was the bug fixed in Sub-Task 1."""
        df = _load_csv_if_exists(GRAPH_DIR / "lcpo_results.csv")
        if df is None:
            pytest.skip("lcpo_results.csv not yet generated")
        threshold_col = next(
            (c for c in ["best_threshold", "threshold"] if c in df.columns), None
        )
        if threshold_col is None:
            pytest.skip("No threshold column found in lcpo_results.csv")
        fold_col = next(
            (c for c in ["fold_idx", "fold"] if c in df.columns), None
        )
        if fold_col is None:
            pytest.skip("No fold column found in lcpo_results.csv")
        all_half = df.groupby(fold_col)[threshold_col].apply(
            lambda x: (x == 0.5).all()
        )
        bad_folds = all_half[all_half].index.tolist()
        assert not bad_folds, (
            f"GNN LCPO folds {bad_folds} have all model seeds at threshold=0.5 — "
            "the LCPO threshold bug may still be present in run_gnn_experiment.py."
        )

    def test_gnn_random_auroc_plausible(self):
        """All GNN random-student AUROC values must be in plausible range."""
        df = _load_csv_if_exists(GRAPH_DIR / "random_student_results.csv")
        if df is None:
            pytest.skip("random_student_results.csv not yet generated")
        bad = df[(df["auroc"] < PLAUSIBLE_AUROC_MIN) | (df["auroc"] > PLAUSIBLE_AUROC_MAX)]
        assert len(bad) == 0, (
            f"Implausible GNN random AUROC:\n{bad[['week', 'seed', 'model', 'auroc']]}"
        )

    def test_gnn_lcpo_auroc_plausible(self):
        """All GNN LCPO AUROC values must be in plausible range."""
        df = _load_csv_if_exists(GRAPH_DIR / "lcpo_results.csv")
        if df is None:
            pytest.skip("lcpo_results.csv not yet generated")
        bad = df[(df["auroc"] < PLAUSIBLE_AUROC_MIN) | (df["auroc"] > PLAUSIBLE_AUROC_MAX)]
        assert len(bad) == 0, (
            f"Implausible GNN LCPO AUROC:\n"
            f"{bad[['week', 'fold_idx', 'model_seed', 'auroc']]}"
        )

    def test_ablation_full_condition_present(self):
        """ablation_results.csv must contain the 'full' baseline condition."""
        df = _load_csv_if_exists(GRAPH_DIR / "ablation_results.csv")
        if df is None:
            pytest.skip("ablation_results.csv not yet generated")
        assert "condition" in df.columns, (
            "ablation_results.csv must have a 'condition' column"
        )
        assert "full" in df["condition"].values, (
            "'full' condition missing from ablation_results.csv"
        )

    @pytest.mark.parametrize("week", WEEKS)
    def test_lgbm_matched_covers_week(self, week):
        """lgbm_matched_results.csv must have rows for every required week."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        assert len(df[df["week"] == week]) > 0, (
            f"No rows for week {week} in lgbm_matched_results.csv"
        )

    @pytest.mark.parametrize("split_type", ["random", "lcpo"])
    def test_lgbm_matched_covers_split(self, split_type):
        """lgbm_matched_results.csv must cover both random and LCPO splits."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        assert len(df[df["split_type"] == split_type]) > 0, (
            f"No rows for split_type='{split_type}' in lgbm_matched_results.csv"
        )

    def test_lgbm_matched_label_is_binary(self):
        """Spot-check: all AUROC values must be > 0.5 (better than random).
        Values at or below 0.5 suggest the label convention was inverted."""
        df = _load_csv_if_exists(MATCHED_DIR / "lgbm_matched_results.csv")
        if df is None:
            pytest.skip("lgbm_matched_results.csv not yet generated")
        inverted = df[df["auroc"] <= 0.5]
        assert len(inverted) == 0, (
            f"AUROC ≤ 0.5 found — labels may be inverted (check LABEL_MAPPING in config.py):\n"
            f"{inverted[['model', 'week', 'seed', 'auroc']]}"
        )


# ---------------------------------------------------------------------------
# Group 3 — Historical row protection (read-only Excel checks)
# ---------------------------------------------------------------------------

class TestHistoricalRowIntegrity:
    """Verify that existing verified values in the Excel matrix are intact."""

    @pytest.fixture()
    def workbook(self):
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        if not EXCEL_PATH.exists():
            pytest.skip("Excel file not found")
        wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
        yield wb
        wb.close()

    def test_excel_has_all_six_sheets(self, workbook):
        """Excel must contain all six metric sheets."""
        expected = {
            "Results", "AUPRC", "F1", "Balanced Accuracy", "Precision", "Recall"
        }
        missing = expected - set(workbook.sheetnames)
        assert not missing, f"Missing Excel sheets: {missing}"

    def test_lgbm_historical_week8_random_results_sheet(self, workbook):
        """Row 11, col E (LightGBM historical, week 8 random) must have a value."""
        ws = workbook["Results"]
        cell_value = ws["E11"].value
        assert cell_value not in (None, "—", ""), (
            "Results!E11 (LightGBM historical week-8 random AUROC) is missing. "
            "Historical rows must not be cleared."
        )

    def test_graphsage_base_week8_random_results_sheet(self, workbook):
        """Row 18, col E (GraphSAGE base weighted, week 8 random) must have a value."""
        ws = workbook["Results"]
        cell_value = ws["E18"].value
        assert cell_value not in (None, "—", ""), (
            "Results!E18 (GraphSAGE base week-8 random AUROC) is missing. "
            "Historical rows must not be cleared."
        )

    def test_ablation_rows_week8_present(self, workbook):
        """Rows 30–34 (GraphSAGE ablation conditions) must have week-8 AUROC values."""
        ws = workbook["Results"]
        for row in range(30, 35):
            cell_value = ws.cell(row=row, column=5).value  # col E = week 8 random
            assert cell_value not in (None, "—", ""), (
                f"Results!E{row} (ablation row {row}) is missing a value. "
                "These rows were verified and must not be cleared."
            )

    def test_matched_protocol_rows_not_raw_numbers(self, workbook):
        """Rows 12, 16, 21 (matched protocol) must be '—' or 'mean ± std'.
        A bare float means someone manually typed a value — use update_results_matrix.py."""
        ws = workbook["Results"]
        for row_num, label in [
            (12, "LightGBM matched"),
            (16, "MLP all enrollment features"),
            (21, "GraphSAGE + six enrollment summaries"),
        ]:
            cell_value = ws.cell(row=row_num, column=5).value  # col E = week 8 random
            if cell_value in (None, "—", ""):
                continue  # empty is fine — not yet run
            val_str = str(cell_value)
            assert "±" in val_str, (
                f"Row {row_num} ({label}) col E contains '{cell_value}' — "
                "looks like a raw number typed manually. "
                "Only write to matched-protocol rows from update_results_matrix.py "
                "using verified CSV outputs."
            )

    @pytest.mark.parametrize("sheet", ["Results", "AUPRC", "F1", "Balanced Accuracy", "Precision", "Recall"])
    def test_all_sheets_have_header_row(self, workbook, sheet):
        """Every sheet must have 'Method or variant' in cell A5 (the header row)."""
        ws = workbook[sheet]
        header = ws["A5"].value
        assert header is not None, f"Sheet '{sheet}' A5 header is empty"
        assert "method" in str(header).lower() or "variant" in str(header).lower(), (
            f"Sheet '{sheet}' A5 does not look like the expected header row: '{header}'"
        )
