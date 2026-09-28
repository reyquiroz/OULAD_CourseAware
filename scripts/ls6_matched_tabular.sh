#!/bin/bash
#SBATCH -J oulad_matched_tabular
#SBATCH -o logs/matched_tabular_%j.out
#SBATCH -e logs/matched_tabular_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -A <YOUR_ALLOCATION>  # TODO: replace with your TACC allocation account

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_matched_comparison.py \
    --models lgbm mlp \
    --weeks 2 4 6 8 \
    --seeds 42 123 7 17 99 \
    --splits random lcpo
