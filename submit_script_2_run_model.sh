#!/bin/bash

#SBATCH --job-name=RunCond
#SBATCH --time=1-20:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

# read in command line arguments:
helpFunction()
{
   echo ""
   echo "Usage: $0 -d DatasetID"
   echo -e "\t-d The ID of the dataset to train the model on. The ID is created when creating the dataset."
   exit 1 # Exit script after printing help
}

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

srun python s2_train_conditional_pixel_diffusion.py

