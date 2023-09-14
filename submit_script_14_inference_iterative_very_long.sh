#!/bin/bash

#SBATCH --job-name=EvalCondIterLong
#SBATCH --time=0-20:00:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=45G
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

# begin reading command line arguments
helpFunction()
{
   echo ""
   echo "Usage: $0 -t DatasetTemplateName -e ExperimentName -m modelName -n NEnsembleMembers"
   echo -t "\t-t The name of the dataset template that should be used."
   echo -e "\t-e The name of the experiment conducted on the dataset."
   echo -e "\t-m The name of the model the predictions should be created with."
   echo -e "\t-n The number of ensemble members to be created."
   exit 1 # Exit script after printing help
}

while getopts "t:e:m:n:" opt
do
   case "$opt" in
      t ) TemplateName="$OPTARG" ;;
      e ) ExperimentName="$OPTARG" ;;
      m ) ModelID="$OPTARG" ;;
      n ) EnsembleMembers="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$TemplateName" ] || [ -z "$ExperimentName" ] || [ -z "$ModelID" ] || [ -z "$EnsembleMembers" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

python s14_very_long_iterative_run.py +data.template=$TemplateName +experiment=$ExperimentName +model_name=$ModelID +n_ensemble_members=$EnsembleMembers
