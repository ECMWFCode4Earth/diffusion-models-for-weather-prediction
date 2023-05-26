#!/bin/bash

#SBATCH --job-name=PD_data_creation
#SBATCH --time=0-03:45:00
#SBATCH --mem-per-cpu=16G

# output files
#SBATCH -o pd_ds_creation.out
#SBATCH -e pd_ds_creation.err


module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s1_write_dataset.py  -cf "/data/compoundx/WeatherDiff/config_file/template_geopotential_500.yml"
