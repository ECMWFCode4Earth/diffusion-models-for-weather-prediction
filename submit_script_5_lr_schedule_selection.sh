#!/bin/bash

#SBATCH --job-name=LRSchedule
#SBATCH --time=2-23:45:00
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
   echo -e "\t-l The LR-Scheduler to use. Name must be in list in ConditionalPixelDiffusion class."
   exit 1 # Exit script after printing help
}

while getopts "d:l:" opt
do
   case "$opt" in
      d ) DatasetID="$OPTARG" ;;
      l ) LRScheduleName="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$DatasetID" ] || [ -z "$LRScheduleName" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# end reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s5_lr_schedule_selection.py  -did $DatasetID -lrs $LRScheduleName

