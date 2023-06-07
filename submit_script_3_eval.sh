#!/bin/bash

#SBATCH --job-name=PD_eval
#SBATCH --time=0-03:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o c_eval.out
#SBATCH -e c_eval.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s5_write_predictions_CPD.py -did 278771 -mid 011A3B
