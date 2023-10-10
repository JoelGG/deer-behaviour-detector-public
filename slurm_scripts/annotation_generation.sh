#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=annotation-generation-2
#SBATCH --time=12:40:00
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
conda activate cameratraps-detector-2
echo "activated"

export PYTHONPATH="$PYTHONPATH:/user/work/ki19061/git/cameratraps:/user/work/ki19061/git/ai4eutils:/user/work/ki19061/git/yolov5"
echo $PYTHONPATH

echo "Start"
python -u /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/annotation_generation/annotation_generation.py \
  "/user/work/ki19061/dataset" \
  --text-list-path "/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/frame_generation/video_ids.txt"
# python -u /user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/utils/annotation_generation/annotation_generation.py \
#   "/user/work/ki19061/dataset" \
#   --annotation-path "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv"
echo "Done"
