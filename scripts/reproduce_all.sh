#!/usr/bin/env bash
set -euo pipefail
# Reproduce all OULAD GraphSAGE results from raw data.
# Usage: bash scripts/reproduce_all.sh
# Assumes: raw OULAD CSVs in data/raw/, Python env with requirements.txt installed.

cd "$(dirname "$0")/.."

echo "=== 1. Graph construction (all weeks) ==="
for week in 2 4 6 8; do
    python src/run_graph_pipeline.py --week $week
done

echo "=== 2. Split generation ==="
python src/save_graph_splits.py --weeks 2 4 6 8

echo "=== 3. GNN experiments ==="
python src/run_gnn_experiment.py --weeks 8 --seeds 42 123 7
python src/run_gnn_experiment.py --weeks 2 4 6 --seeds 42 123 7 --random-only

echo "=== 4. LightGBM comparison ==="
python src/compare_gnn_lgbm.py --weeks 2 4 6 8 --seeds 42 123 7

echo "=== 5. Ablation study ==="
python src/run_ablation.py --week 8 --seeds 42

echo "=== 6. Course variation ==="
python src/course_variation.py --week 8

echo "=== 7. Figures and tables ==="
python src/generate_report_figures.py

echo "Done."
