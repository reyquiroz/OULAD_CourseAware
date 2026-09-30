#!/bin/bash
#SBATCH -J oulad_gnn_${VARIANT}
#SBATCH -o logs/gnn_${VARIANT}_%j.out
#SBATCH -e logs/gnn_${VARIANT}_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A Lonestar6

# Submit one variant at a time via --export:
#   sbatch --export=VARIANT=enrollment_node scripts/ls6_gnn_architectural.sh
#
# Or loop over all Sub-Task 6–11 variants:
#   for VARIANT in enrollment_node edge_aware_mp course_design temporal \
#                  course_conditioned gcn rgcn hgt; do
#       sbatch --export=VARIANT=$VARIANT scripts/ls6_gnn_architectural.sh
#   done

if [ -z "$VARIANT" ]; then
    echo "ERROR: VARIANT is not set. Pass it via --export=VARIANT=<name>"
    exit 1
fi

mkdir -p logs results/graph

module load python3/3.11.2
source $WORK/OULAD/oulad_env/bin/activate
cd $WORK/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# --seeds  = random-student split seeds (one run per seed)
# --model-seeds = model init seeds for LCPO (per-fold multi-seed averaging)
python src/run_gnn_experiment.py \
    --condition $VARIANT \
    --weeks 8 \
    --seeds 42 123 7 17 99 \
    --model-seeds 42 123 7 17 99 \
    --splits random lcpo
