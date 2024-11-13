#!/bin/bash

#SBATCH --job-name=math_test   # create a short name for your job
#SBATCH --partition=mig_class     # Partition name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=12:00:00            # Job time limit
#SBATCH --mem=128bGB               # Allocate 128GB RAM
#SBATCH --output="test.out"       # Output file
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks-per-node=6       # Request 6 tasks per node
#SBATCH --output=slurm_logs/math.out   # output file name
#SBATCH --error=slurm_logs/math.out    # error file name

# Run the Python script
module load anaconda
conda --version
conda activate math-freq

python /home/cs601-zmou1/Math-Reasoning-on-Frequency/test.py