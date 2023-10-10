#!/bin/sh
#SBATCH --job-name=train-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --time=4:45:00
#SBATCH --mem=80000M
#SBATCH --account=cosc028044
#SBATCH --output=./logs/%j.out

#purge existing modules
#purge existing modules
echo "cleaning modules"
module purge
echo "cleaned"

#load the environment
echo "starting conda"
source /user/work/ki19061/initconda.sh
echo "done"

echo "activating environment"
conda activate deep-learn
echo "activated"


echo "Start"
python -u /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/train.py \
  --learning-rate 5e-5 \
  --l2-alpha 1e-3 \
  --annotation-root "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv" \
  --batch-size 8 \
  --worker-count 0
echo "Done"
