"""
gnn_results.py — select the reference GNN rows from the result CSVs.

``lcpo_results.csv`` and ``lcpo_summary.csv`` hold several variants side by side
(``condition`` and ``loss_weighting``). A reader that groups by fold or module without
filtering would silently average them together. Readers that report a single GNN number
call ``select_reference_gnn_rows`` so the choice is explicit and printed.

Reference = condition "full" (base GraphSAGE) with class-weighted loss, which matches the
earlier reports. Weighted and unweighted results are never combined.
"""

import pandas as pd

REFERENCE_CONDITION = "full"
REFERENCE_WEIGHTING = "weighted"


def select_reference_gnn_rows(
    df: pd.DataFrame,
    condition: str = REFERENCE_CONDITION,
    loss_weighting: str = REFERENCE_WEIGHTING,
) -> pd.DataFrame:
    """Return the rows of *df* for one condition and one loss weighting.

    Columns that are absent are not filtered (files written before the column existed).
    A message is printed so the report states which variant it describes.
    """
    out = df
    if "condition" in out.columns:
        out = out[out["condition"] == condition]
    if "loss_weighting" in out.columns:
        out = out[out["loss_weighting"] == loss_weighting]
    print(
        f"[gnn_results] GNN rows used: condition={condition!r}, "
        f"loss_weighting={loss_weighting!r} ({len(out)} of {len(df)} rows)"
    )
    return out.copy()
