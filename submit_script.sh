#!/bin/bash

#SBATCH --job-name=PD_2
#SBATCH --time=0-03:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o pd_3.out
#SBATCH -e pd_3.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s2_train_PixelDiffusion.py