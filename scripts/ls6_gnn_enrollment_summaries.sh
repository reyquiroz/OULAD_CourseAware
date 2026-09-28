#!/bin/bash
#SBATCH -J oulad_gnn_enroll_summaries
#SBATCH -o logs/gnn_enroll_%j.out
#SBATCH -e logs/gnn_enroll_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A <YOUR_ALLOCATION>  # TODO: replace with your TACC allocation account

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_gnn_experiment.py \
    --condition with_enrollment_summaries \
    --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 \
    --splits random lcpo
