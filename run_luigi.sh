#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --mem=16000              # Maximum memory required (in megabytes)
#SBATCH --job-name=lpyroclast
#SBATCH --partition=scott,gpu
#SBATCH --gres=gpu:1
#SBATCH --licenses=common

module load anaconda
conda activate tf2
python -m luigi --module $@ # <module> <task> <parameters>
