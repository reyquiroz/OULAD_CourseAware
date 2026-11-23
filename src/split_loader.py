"""
split_loader.py — the single read path for persisted evaluation splits.

``save_graph_splits.py`` is the only place that generates splits (via
``oulad_data.random_student_split``, ``lcpo_split`` and ``sample_lcpo_val``). Every
model (LightGBM, MLP, GNN random-student, GNN LCPO) must obtain its train/val/test
masks from this module so all models see identical partitions.

Masks are boolean ``pd.Series``. They are aligned to the enrollment triple
``(id_student, code_module, code_presentation)``, never to row position alone:
pass ``frame`` (any DataFrame holding those three columns, e.g. the tabular
feature table) and the masks are re-ordered to match it. The call raises if the
frame and the persisted split do not cover exactly the same enrollments.

Files read, all under ``GRAPH_EVALUATION_DIR/weekNN/splits/``:
    weekNN_random_split_seed{S}.parquet   is_train / is_val / is_test
    weekNN_lcpo_folds.csv                 fold definitions
    weekNN_lcpo_masks.parquet             foldNN columns: "train" / "val" / "test"
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import GRAPH_EVALUATION_DIR

KEY_COLS = ["id_student", "code_module", "code_presentation"]
_REGEN_HINT = "Regenerate with: python src/save_graph_splits.py --seeds 42 123 7 17 99"


def _split_dir(week: int, eval_dir: Optional[Path] = None) -> Path:
    base = Path(eval_dir) if eval_dir is not None else GRAPH_EVALUATION_DIR
    return base / f"week{week:02d}" / "splits"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}\n{_REGEN_HINT}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _position_map(split_keys: pd.DataFrame, frame: pd.DataFrame) -> Optional[pd.Index]:
    """Return positions in *split_keys* for each row of *frame*, or None if identical order.

    Raises ValueError when the two tables do not hold the same set of unique triples.
    """
    for name, table in (("split", split_keys), ("frame", frame)):
        missing = [c for c in KEY_COLS if c not in table.columns]
        if missing:
            raise ValueError(f"{name} table lacks key columns {missing}")
    split_idx = pd.MultiIndex.from_frame(split_keys[KEY_COLS])
    frame_idx = pd.MultiIndex.from_frame(frame[KEY_COLS])
    if not split_idx.is_unique or not frame_idx.is_unique:
        raise ValueError("Enrollment triple is not unique; prediction unit must be the enrollment")
    if len(split_idx) != len(frame_idx):
        raise ValueError(
            f"Frame has {len(frame_idx)} enrollments but the persisted split has "
            f"{len(split_idx)}; the frame is not the same population. {_REGEN_HINT}"
        )
    if split_idx.equals(frame_idx):
        return None
    pos = split_idx.get_indexer(frame_idx)
    if (pos < 0).any():
        raise ValueError(
            f"{int((pos < 0).sum())} frame enrollments are absent from the persisted split"
        )
    return pos


def _to_series(masks, split_keys: pd.DataFrame, frame: Optional[pd.DataFrame]):
    """Align boolean arrays (ordered like *split_keys*) to *frame* and wrap as Series."""
    if frame is None:
        return tuple(pd.Series(m, index=pd.RangeIndex(len(m))) for m in masks)
    pos = _position_map(split_keys, frame)
    out = []
    for m in masks:
        arr = m if pos is None else m[pos]
        out.append(pd.Series(arr, index=frame.index))
    return tuple(out)


def load_random_split(
    week: int,
    seed: int,
    frame: Optional[pd.DataFrame] = None,
    eval_dir: Optional[Path] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (train, val, test) boolean Series for the random-student split of *seed*."""
    sp = _read(_split_dir(week, eval_dir) / f"week{week:02d}_random_split_seed{seed}.parquet")
    masks = [sp[c].to_numpy(dtype=bool) for c in ("is_train", "is_val", "is_test")]
    return _to_series(masks, sp, frame)


def load_lcpo_folds(week: int, eval_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return the LCPO fold table (fold_idx, held_out_module, held_out_presentation, ...)."""
    return _read(_split_dir(week, eval_dir) / f"week{week:02d}_lcpo_folds.csv")


def load_lcpo_masks(
    week: int,
    fold_idx: int,
    frame: Optional[pd.DataFrame] = None,
    eval_dir: Optional[Path] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (train, val, test) boolean Series for LCPO fold *fold_idx*.

    Train and val together are every enrollment outside the held-out
    course-presentation; test is the held-out course-presentation.
    """
    sp = _read(_split_dir(week, eval_dir) / f"week{week:02d}_lcpo_masks.parquet")
    col = f"fold{fold_idx:02d}"
    if col not in sp.columns:
        raise KeyError(f"LCPO fold column '{col}' not found for week {week}")
    status = sp[col].to_numpy()
    masks = [status == "train", status == "val", status == "test"]
    return _to_series(masks, sp, frame)
