from data.frame_dataset_rgb import BrandenburgFrameDatasetRGB

from torch.utils.data import DataLoader
from torchvision import transforms as t
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import transforms
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


def slow_R50_transform():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate) / frames_per_second
    return transform


frame_transform = t.Compose([transforms.ToTensor(),])
video_transform = slow_R50_transform()

ds = BrandenburgFrameDatasetRGB(
    dataset_root="/user/work/ki19061/dataset",
    annotation_root="/user/work/ki19061/deerbehaviourdetector/behaviour_annotations.csv",
    frame_transform=frame_transform,
    video_transform=video_transform,
)


loader = DataLoader(ds, batch_size=1)
data = {"video": [], "start": [], "end": [], "tensorsize": []}
for batch in loader:
    for i in range(len(batch[0])):
        data["tensorsize"].append(batch[0][i].size())
        print(data)
