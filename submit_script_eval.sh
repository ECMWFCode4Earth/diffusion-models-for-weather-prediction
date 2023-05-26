#!/bin/bash

#SBATCH --job-name=PD_eval
#SBATCH --time=0-03:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o pd_eval.out
#SBATCH -e pd_eval.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s3_generate_output.py