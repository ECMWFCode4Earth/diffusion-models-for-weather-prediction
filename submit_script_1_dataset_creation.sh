#!/bin/bash

#SBATCH --job-name=CreateData
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=16G

# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_data

srun python s1_write_dataset.py