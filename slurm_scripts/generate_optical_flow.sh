#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_veryshort
#SBATCH --job-name=generate-optical-flow
#SBATCH --time=4:20:00
#SBATCH --mem=10000M
#SBATCH --account=cosc028044
#SBATCH --output=./logs/%j.out

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
python /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/generate_raft_flow.py "/user/work/ki19061/dataset" \
    --annotation-path "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv"
echo "Done"
