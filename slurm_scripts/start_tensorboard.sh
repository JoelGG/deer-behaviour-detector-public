PORT=$((($UID-6025) % 65274))
hostname -s

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"
tensorboard --logdir "/user/work/ki19061/deer-behaviour-detector/slurm_scripts/tensorboard_logs" --port "29615" --bind_all