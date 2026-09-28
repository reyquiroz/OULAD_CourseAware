#!/bin/bash
#SBATCH -J oulad_graph_artifacts
#SBATCH -o logs/artifacts_%j.out
#SBATCH -e logs/artifacts_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A <YOUR_ALLOCATION>  # TODO: replace with your TACC allocation account

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

for week in 2 4 6 8; do
    python src/run_graph_pipeline.py --week $week
done
python src/save_graph_splits.py --weeks 2 4 6 8
