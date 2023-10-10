import torch
import torchvision

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

vid = torchvision.io.read_video(
    "/projects/Animal_Biometrics/Video_All/HT_T01/HT_cam11_432958_5887023_20190611/06110001.MP4"
)
video, audio, meta = vid

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]


video_r = torch.permute(video, (3, 0, 1, 2))
print(video_r.size())
video_r = Lambda(lambda x: x / 255.0)(video_r)
video_r = NormalizeVideo(mean, std)(video_r)
print(video_r)
print(video_r.size())
