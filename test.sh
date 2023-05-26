#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=0-00:09:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o test.out
#SBATCH -e test.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

python test_environment.py