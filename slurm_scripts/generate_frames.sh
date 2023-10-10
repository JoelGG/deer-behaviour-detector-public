#!/bin/sh
#SBATCH --job-name=frame-generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition cpu
#SBATCH --time=4:40:00
#SBATCH --mem=10000M
#SBATCH --account=cosc016282
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
conda activate cameratraps-detector-2
echo "activated"

echo "Start"
# python /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/generate_frames.py \
#   "/user/work/ki19061/dataset" \
#   --text-list-path "/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/video_ids.txt"

python /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/generate_frames.py \
  "/user/work/ki19061/dataset" \
  --annotation-path "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv"

echo "Done"
