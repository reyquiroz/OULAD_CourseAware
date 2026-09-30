#!/bin/bash
#SBATCH -J oulad_zenodo
#SBATCH -o logs/zenodo_%j.out
#SBATCH -e logs/zenodo_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A Lonestar6

mkdir -p logs results/zenodo

module load python3/3.11.2
source $WORK/OULAD/oulad_env/bin/activate
cd $WORK/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# --seeds  = random-student split seeds
# --model-seeds = model init seeds for GNN LCPO (must match --seeds count)
python src/run_zenodo_pipeline.py \
    --mode all \
    --seeds 42 123 7 17 99 \
    --model-seeds 42 123 7 17 99
