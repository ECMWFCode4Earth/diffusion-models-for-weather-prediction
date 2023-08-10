#!/bin/bash

#SBATCH --job-name=EvalCond
#SBATCH --time=0-03:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

# begin reading command line arguments
helpFunction()
{
   echo ""
   echo "Usage: $0 -m modelName -e NEnsembleMembers"
   echo -e "\t-m The name of the model the predictions should be created with."
   echo -e "\t-e The number of ensemble members to be created."
   exit 1 # Exit script after printing help
}

while getopts "m:e:" opt
do
   case "$opt" in
      m ) ModelID="$OPTARG" ;;
      e ) EnsembleMembers="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$ModelID" ] || [ -z "$EnsembleMembers" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

srun python s3_write_predictions_CPD.py +model_name=$ModelID +n_ensemble_members=$EnsembleMembers
