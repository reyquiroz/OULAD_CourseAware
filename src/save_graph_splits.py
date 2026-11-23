"""
save_graph_splits.py — Save per-week split definitions for GNN training.

For each prediction week, loads the enrollment supervision table and writes
these files into GRAPH_EVALUATION_DIR/week{N}/splits/ (results/results/graph/evaluation/):

  week{N}_random_split_seed{S}.parquet   (one per split seed in --seeds)
      The enrollment table with three boolean columns added:
      is_train, is_val, is_test  (val_frac=0.1, test_frac=0.2)

  week{N}_random_split.parquet
      Legacy name; identical to the seed-42 file (read by verify_splits.py).

  week{N}_lcpo_folds.csv
      One row per LCPO fold with columns:
      fold_idx, held_out_module, held_out_presentation, n_train, n_test,
      n_val, n_val_pos.  n_train is the full non-test set (validation included);
      n_val counts the validation enrollments sampled from it.

  week{N}_lcpo_masks.parquet
      The enrollment table plus one column per fold (fold00 .. fold21) holding
      "train", "val" or "test". Validation uses oulad_data.sample_lcpo_val, the
      single sampler shared with the GNN runner, so every model sees identical
      train/val/test enrollments per fold. Model seeds do not change the masks.

  week{N}_splits_config.json
      Documents the split parameters for full reproducibility.

  week{N}_future_split.parquet
      The enrollment table with is_train / is_test boolean columns for the
      future-presentation split: train on 2013B/2013J/2014B, test on 2014J.

Usage
-----
    source oulad_env/bin/activate
    python src/save_graph_splits.py           # all four weeks
    python src/save_graph_splits.py --week 8  # single week
    python src/save_graph_splits.py --seeds 42 123 7 17 99

Prerequisite: enrollment artifacts must exist:
    python src/run_graph_pipeline.py --week {2,4,6,8}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import GRAPH_ARTIFACTS_DIR, GRAPH_EVALUATION_DIR
from oulad_data import lcpo_split, random_student_split, sample_lcpo_val

WEEKS = [2, 4, 6, 8]
DEFAULT_SEEDS = [42, 123, 7, 17, 99]
VAL_FRAC = 0.1
TEST_FRAC = 0.2
LEGACY_SEED = 42
LCPO_MIN_VAL_POS = 20

# Future-presentation split — consistent with evaluation_pipeline.py
_TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
_TEST_PRESENTATIONS  = ["2014J"]


def save_splits_for_week(week: int, seeds: list) -> None:
    prefix    = f"week{week:02d}"
    enroll_path = GRAPH_ARTIFACTS_DIR / f"{prefix}_enrollments.parquet"

    if not enroll_path.exists():
        raise FileNotFoundError(
            f"Enrollment artifact not found: {enroll_path}\n"
            f"Run: python src/run_graph_pipeline.py --week {week}"
        )

    enrollments = pd.read_parquet(enroll_path)

    split_dir = GRAPH_EVALUATION_DIR / f"week{week:02d}" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Random student splits (70 / 10 / 20), one file per split seed ───
    random_info = {}
    for seed in seeds:
        train_mask, val_mask, test_mask = random_student_split(
            enrollments, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=seed
        )
        rs = enrollments.copy()
        rs["is_train"] = train_mask
        rs["is_val"]   = val_mask
        rs["is_test"]  = test_mask

        # Integrity checks
        assert (rs["is_train"] | rs["is_val"] | rs["is_test"]).all(), \
            "Some rows missing from all splits"
        assert not (rs["is_train"] & rs["is_val"]).any(), "Overlap train/val"
        assert not (rs["is_train"] & rs["is_test"]).any(), "Overlap train/test"
        assert not (rs["is_val"]   & rs["is_test"]).any(), "Overlap val/test"
        train_stu = set(enrollments.loc[train_mask, "id_student"].unique())
        val_stu   = set(enrollments.loc[val_mask,   "id_student"].unique())
        test_stu  = set(enrollments.loc[test_mask,  "id_student"].unique())
        assert train_stu.isdisjoint(val_stu),  "Student overlap train/val"
        assert train_stu.isdisjoint(test_stu), "Student overlap train/test"
        assert val_stu.isdisjoint(test_stu),   "Student overlap val/test"

        rs.to_parquet(split_dir / f"{prefix}_random_split_seed{seed}.parquet", index=False)
        if seed == LEGACY_SEED:
            rs.to_parquet(split_dir / f"{prefix}_random_split.parquet", index=False)
        random_info[str(seed)] = {
            "n_train": int(train_mask.sum()),
            "n_val":   int(val_mask.sum()),
            "n_test":  int(test_mask.sum()),
        }

    # ── 2. LCPO folds and persisted train/val/test masks ──────────────────
    presentations = (
        enrollments[["code_module", "code_presentation"]]
        .drop_duplicates()
        .sort_values(["code_module", "code_presentation"])
        .itertuples(index=False)
    )
    y = enrollments["target"].to_numpy().astype(np.int32)
    lcpo_rows = []
    lcpo_masks = enrollments.copy()
    for fold_idx, row in enumerate(presentations):
        tr_mask, te_mask = lcpo_split(enrollments, row.code_module, row.code_presentation)
        train_all = tr_mask.to_numpy()
        # sample_lcpo_val prints a progress tag; keep the output readable.
        train_np, val_np = sample_lcpo_val(
            enroll_df=enrollments,
            train_all_mask=train_all,
            y=y,
            fold_idx=fold_idx,
            min_val_pos=LCPO_MIN_VAL_POS,
        )
        status = np.full(len(enrollments), "train", dtype=object)
        status[val_np] = "val"
        status[te_mask.to_numpy()] = "test"
        assert (status == "val").sum() == int(val_np.sum())
        assert not (val_np & te_mask.to_numpy()).any(), "LCPO val overlaps test"
        lcpo_masks[f"fold{fold_idx:02d}"] = status
        lcpo_rows.append({
            "fold_idx":               fold_idx,
            "held_out_module":        row.code_module,
            "held_out_presentation":  row.code_presentation,
            "n_train":                int(tr_mask.sum()),
            "n_test":                 int(te_mask.sum()),
            "n_val":                  int(val_np.sum()),
            "n_val_pos":              int(y[val_np].sum()),
        })
    print()
    lcpo_df = pd.DataFrame(lcpo_rows)
    lcpo_path = split_dir / f"{prefix}_lcpo_folds.csv"
    lcpo_df.to_csv(lcpo_path, index=False)
    lcpo_masks_path = split_dir / f"{prefix}_lcpo_masks.parquet"
    lcpo_masks.to_parquet(lcpo_masks_path, index=False)

    # ── 3. Future-presentation split ──────────────────────────────────────
    fp = enrollments.copy()
    fp["is_train"] = fp["code_presentation"].isin(_TRAIN_PRESENTATIONS)
    fp["is_test"]  = fp["code_presentation"].isin(_TEST_PRESENTATIONS)
    fp_path = split_dir / f"{prefix}_future_split.parquet"
    fp.to_parquet(fp_path, index=False)

    n_fp_train = int(fp["is_train"].sum())
    n_fp_test  = int(fp["is_test"].sum())

    # ── 4. Splits config JSON ──────────────────────────────────────────────
    config = {
        "week":                     week,
        "enrollment_count":         len(enrollments),
        "label_at_risk_count":      int(y.sum()),
        "random_student_split": {
            "seeds":     list(seeds),
            "val_frac":  VAL_FRAC,
            "test_frac": TEST_FRAC,
            "files":     {str(sd): f"{prefix}_random_split_seed{sd}.parquet" for sd in seeds},
            "legacy_file": f"{prefix}_random_split.parquet (= seed {LEGACY_SEED})",
            "per_seed":  random_info,
        },
        "lcpo": {
            "n_folds":    len(lcpo_rows),
            "folds_file": f"{prefix}_lcpo_folds.csv",
            "masks_file": f"{prefix}_lcpo_masks.parquet",
            "val_sampler": "oulad_data.sample_lcpo_val",
            "min_val_pos": LCPO_MIN_VAL_POS,
        },
        "future_presentation": {
            "train_presentations": _TRAIN_PRESENTATIONS,
            "test_presentations":  _TEST_PRESENTATIONS,
            "n_train":             n_fp_train,
            "n_test":              n_fp_test,
        },
        "note": (
            "Boolean masks reference rows in the enrollment supervision table "
            f"({prefix}_enrollments.parquet in GRAPH_ARTIFACTS_DIR). "
            "That table excludes enrollments unregistered on or before the cutoff "
            "day, so the population differs by week. Rows are aligned by position "
            "and by the (id_student, code_module, code_presentation) key."
        ),
    }
    cfg_path = split_dir / f"{prefix}_splits_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    first = random_info[str(seeds[0])]
    print(
        f"  Week {week}: {len(enrollments)} enrollments "
        f"| random seed {seeds[0]} {first['n_train']}/{first['n_val']}/{first['n_test']} "
        f"({len(seeds)} seeds) "
        f"| LCPO {len(lcpo_rows)} folds "
        f"| future {n_fp_train}/{n_fp_test}"
    )
    for p in [lcpo_path, lcpo_masks_path, fp_path, cfg_path]:
        print(f"    ✓ {p.relative_to(GRAPH_EVALUATION_DIR.parent.parent)}")


def main():
    p = argparse.ArgumentParser(
        description="Save per-week GNN split definitions."
    )
    p.add_argument(
        "--week", type=int, default=None, choices=WEEKS,
        help="Single week to process (default: all four weeks)."
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random-student split seeds (default: 42 123 7 17 99)."
    )
    args = p.parse_args()

    weeks = [args.week] if args.week else WEEKS

    print("=" * 60)
    print("Saving graph split definitions")
    print("=" * 60)
    for week in weeks:
        save_splits_for_week(week, args.seeds)

    print("\nAll split definitions saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
