#!/bin/bash

#SBATCH --job-name=EvalUNet
#SBATCH --time=0-03:45:00
#SBATCH -G 1 # nvidia-a100:1
#SBATCH --mem-per-cpu=16G
# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_log/%x-%u-%j.err

# begin reading command line arguments
helpFunction()
{
   echo ""
   echo "Usage: $0 -t DatasetTemplateName -e ExperimentName -m modelName"
   echo -t "\t-m The name of the dataset template that should be used."
   echo -e "\t-e The name of the experiment conducted on the dataset."
   echo -e "\t-m The name of the model the predictions should be created with."
   exit 1 # Exit script after printing help
}

while getopts "t:e:m:" opt
do
   case "$opt" in
      t ) TemplateName="$OPTARG" ;;
      e ) ExperimentName="$OPTARG" ;;
      m ) ModelID="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$TemplateName" ] || [ -z "$ExperimentName" ] || [ -z "$ModelID" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate WD_model

python s8_write_predictions_unet.py +data.template=$TemplateName +experiment=$ExperimentName +model_name=$ModelID

