#!/bin/bash

#SBATCH --job-name=RunCond
#SBATCH --time=1-2:45:00
#SBATCH -G 1 #  nvidia-a100:1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

# read in command line arguments:
helpFunction()
{
   echo ""
   echo "Usage: $0 -e experiment_name"
   echo -e "\t-e The name of the experiment template to be used."
   exit 1 # Exit script after printing help
}

while getopts "e:" opt
do
   case "$opt" in
      e ) experiment_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if  [ -z "$experiment_name" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

srun python s2_train_conditional_pixel_diffusion.py +experiment=$experiment_name
