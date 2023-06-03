#!/bin/bash

#SBATCH --job-name=CondDiff
#SBATCH --time=1-23:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o pd_c.out
#SBATCH -e pd_c.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s4_train_conditional_PixelDiffusion.py  -did 278771

