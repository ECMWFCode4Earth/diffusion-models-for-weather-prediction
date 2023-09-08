#!/bin/bash

#SBATCH --job-name=EvalCondIter
#SBATCH --time=0-07:45:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=20G
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err


# begin reading command line arguments
helpFunction()
{
   echo ""
   echo "Usage: $0 -t DatasetTemplateName -e ExperimentName -m modelName -n NEnsembleMembers -s NSteps"
   echo -t "\t-m The name of the dataset template that should be used."
   echo -e "\t-e The name of the experiment conducted on the dataset."
   echo -e "\t-m The name of the model the predictions should be created with."
   echo -e "\t-n The number of ensemble members to be created."
   echo -e "\t-s The number of steps in the trajectories created."
   exit 1 # Exit script after printing help
}

while getopts "t:e:m:n:s:" opt
do
   case "$opt" in
      t ) TemplateName="$OPTARG" ;;
      e ) ExperimentName="$OPTARG" ;;
      m ) ModelID="$OPTARG" ;;
      n ) EnsembleMembers="$OPTARG" ;;
      s ) Steps="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$TemplateName" ] || [ -z "$ExperimentName" ] || [ -z "$ModelID" ] || [ -z "$EnsembleMembers" ] || [ -z "$Steps" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

srun python s13_write_predictions_iterative.py +data.template=$TemplateName +experiment=$ExperimentName +model_name=$ModelID +n_steps=$Steps
