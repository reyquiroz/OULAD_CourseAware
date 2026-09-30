#!/bin/bash
#SBATCH -J oulad_gnn_enroll_summaries
#SBATCH -o logs/gnn_enroll_%j.out
#SBATCH -e logs/gnn_enroll_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH -A Lonestar6

mkdir -p logs results/graph

module load python3/3.11.2
source $WORK/OULAD/oulad_env/bin/activate
cd $WORK/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# --seeds  = random-student split seeds (one independent run per seed)
# --model-seeds = model init seeds for LCPO (per-fold multi-seed averaging)
python src/run_gnn_experiment.py \
    --condition with_enrollment_summaries \
    --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 \
    --model-seeds 42 123 7 17 99 \
    --splits random lcpo
