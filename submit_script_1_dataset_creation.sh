#!/bin/bash

#SBATCH --job-name=CreateData
#SBATCH --time=0-03:45:00
#SBATCH --mem-per-cpu=32G

# output files
#SBATCH -o /data/compoundx/WeatherDiff/job_logs/%x-%u-%j.out
#SBATCH -e /data/compoundx/WeatherDiff/job_logs/%x-%u-%j.err

# begin reading command line arguments
helpFunction()
{
   echo ""
   echo "Usage: $0 -p Path"
   echo -e "\t-p The path to a template configuration file to use in the dataset creation."
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   echo "$opt"
   case "$opt" in
      p ) ConfigPath="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$ConfigPath" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi
# stop reading command line arguments

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

srun python s1_write_dataset.py  -cf $ConfigPath