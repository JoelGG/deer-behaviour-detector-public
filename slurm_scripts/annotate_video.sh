#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition cpu
#SBATCH --job-name=annotate-videos
#SBATCH --time=2:40:00
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
python -u /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/annotation_generation/annotate_frames.py \
  "/user/work/ki19061/dataset" \
  "/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/video_ids.txt"
# python -u /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/annotation_generation/annotation_generation.py \
#   "/user/work/ki19061/dataset" \
#   --annotation-path "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv"
echo "Done"
