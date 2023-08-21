#!/bin/bash

#SBATCH --job-name=Diffusion
#SBATCH --time=0-3:35:00
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

conda activate TORCH311

srun python s10_train_LFD.py +experiment=$experiment_name