"""
run_graph_pipeline.py — CLI entry point for the OULAD graph pipeline

Usage:
    python src/run_graph_pipeline.py [--week 8] [--data-dir DATA/raw]
                                     [--save-dir results/graph/artifacts]

Runs the full staged graph pipeline (load → filter → nodes → edges →
enrollments → validate → materialize) for the specified prediction week and
persists artifacts to disk under results/graph/.

Extend to Weeks 2, 4, 6 only after the Week 8 path is validated.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from config import GRAPH_ARTIFACTS_DIR, GRAPH_VALIDATION_DIR
from graph_pipeline import run_pipeline
from graph_validation import run_validation


def parse_args():
    p = argparse.ArgumentParser(
        description="Build and persist the OULAD enrollment-centric graph."
    )
    p.add_argument(
        "--week",
        type=int,
        default=8,
        choices=[2, 4, 6, 8],
        help="Prediction week (default: 8).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw OULAD data directory (default: DATA/raw from config).",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Directory to write graph artifacts "
            "(default: results/graph/artifacts from config)."
        ),
    )
    return p.parse_args()


def _build_metadata(week: int, window_days: int, result: dict) -> dict:
    """
    Build a self-describing metadata record for the persisted graph artifacts.

    Captures the temporal cutoff, node and edge schema (column names), artifact
    file names, and a concise construction summary so that downstream training
    scripts can inspect the artifact manifest without re-running the pipeline.
    """
    nodes = result["nodes"]
    edges = result["edges"]
    enrollments = result["enrollments"]
    artifacts = result["artifacts"]

    node_schema = {ntype: list(df.columns) for ntype, df in nodes.items()}
    edge_schema = {etype: list(df.columns) for etype, df in edges.items()}

    return {
        "week": week,
        "window_days": window_days,
        "temporal_cutoff_rule": (
            "VLE interactions: date <= window_days; "
            "Assessments: due_date <= window_days (not submission date). "
            "See docs/LEAKAGE_PREVENTION.md."
        ),
        "supervised_unit": "enrollment (id_student, code_module, code_presentation)",
        "node_schema": node_schema,
        "edge_schema": {
            **edge_schema,
            "_notes": {
                "interacted_with": (
                    "Aggregated from raw interaction records grouped by "
                    "(id_student, id_site, code_module, code_presentation) "
                    "before edge construction; the saved edge table stores "
                    "student-node to resource-node pairs with aggregate "
                    "features total_clicks / n_interactions / first_day / "
                    "last_day / active_days."
                ),
                "submitted": (
                    "Enrollment-scoped: matched on "
                    "(id_student, code_module, code_presentation). "
                    "score=0 where raw score is null."
                ),
            },
        },
        "node_counts": {ntype: int(len(df)) for ntype, df in nodes.items()},
        "edge_counts": {etype: int(len(df)) for etype, df in edges.items()},
        "enrollment_count": int(len(enrollments)),
        "label_at_risk_count": int((enrollments["target"] == 1).sum()),
        "label_success_count": int((enrollments["target"] == 0).sum()),
        "label_at_risk_rate": round(float((enrollments["target"] == 1).mean()), 4),
        "artifacts": {name: str(path.name) for name, path in artifacts.items()},
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_memory_mb": result["peak_memory_mb"],
    }


def main():
    args = parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else GRAPH_ARTIFACTS_DIR

    print(f"Running graph pipeline for Week {args.week}...")
    result = run_pipeline(
        week=args.week,
        data_dir=args.data_dir,
        save_dir=save_dir,
    )

    GRAPH_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── integrity report ──────────────────────────────────────────────────
    integrity_path = (
        GRAPH_VALIDATION_DIR / f"week{args.week:02d}_integrity.json"
    )
    with open(integrity_path, "w") as f:
        json.dump(
            {
                "week": args.week,
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_memory_mb": result["peak_memory_mb"],
                "integrity": result["integrity"],
                "node_counts": {
                    ntype: int(len(df))
                    for ntype, df in result["nodes"].items()
                },
                "edge_counts": {
                    etype: int(len(df))
                    for etype, df in result["edges"].items()
                },
                "enrollment_count": int(len(result["enrollments"])),
            },
            f,
            indent=2,
        )
    print(f"Integrity report saved to {integrity_path}")

    # ── artifact metadata ─────────────────────────────────────────────────
    from config import PREDICTION_WINDOWS
    window_days = PREDICTION_WINDOWS.get(f"week_{args.week}", args.week * 7)
    metadata = _build_metadata(args.week, window_days, result)
    metadata_path = save_dir / f"week{args.week:02d}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Artifact metadata saved to  {metadata_path}")

    # ── subtask-4 validation report ───────────────────────────────────────
    print("\nRunning graph validation and statistics report...")
    run_validation(
        week=args.week,
        artifacts_dir=save_dir,
        validation_dir=GRAPH_VALIDATION_DIR,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
