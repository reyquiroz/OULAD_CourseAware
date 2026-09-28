#!/bin/bash
#SBATCH -J oulad_zenodo
#SBATCH -o logs/zenodo_%j.out
#SBATCH -e logs/zenodo_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -A # TODO: replace with your TACC allocation account

module load python3/3.11.2
source $SCRATCH/OULAD/oulad_env/bin/activate
cd $SCRATCH/OULAD
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/run_zenodo_pipeline.py --mode all --seeds 42 123 7 17 99
