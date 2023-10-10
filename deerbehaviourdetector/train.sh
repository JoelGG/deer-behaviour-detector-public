#!/bin/sh
#SBATCH --job-name=deer-behaviour-detector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=cosc028044
#SBATCH --output=./logs/%j.out

FILE=/projects/Animal_Biometrics/Video_All
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
fi

#purge existing modules
echo "Cleaning modules"
module purge
echo "Cleaned"

#load the environment
echo "Loading environment"
module load languages/anaconda3/2022.12-3.9.13-torch-cuda-11.7
echo "Loaded"

echo "Activating environment"
source activate base
echo "Activated"


echo "Start"
python train.py \
  --learning-rate 5e-5 \
  --batch-size 128 \
  --worker-count 1
echo "Done"
