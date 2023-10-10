import itertools
import os
import math
import random
import pathlib
import csv
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import torch
import torchvision
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset
from torchvision.io import read_image
from PIL import Image
from utils.data.get_deer import get_deer

BRANDENBURG_ROOT = "/projects/Animal_Biometrics/Video_All"


class BrandenburgFrameDatasetFlow(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        annotation_root,
        frame_transform,
        flow_transform,
        clip_size=16,
    ):
        super(BrandenburgFrameDatasetFlow).__init__()

        self.root = dataset_root
        self.annotations = pd.read_csv(annotation_root)

        self.samples = self.annotations.index
        (
            self.kinetic_classes,
            self.kinetic_class_to_idx,
            self.classes,
            self.class_to_idx,
        ) = self._find_classes()

        self.frame_transform = frame_transform
        self.flow_transform = flow_transform
        self.clip_size = clip_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        details = self.annotations.iloc[idx]
        video_id = details["video_id"]
        index = details["index"]
        target = details["target"]
        start = details["start"]
        end = details["end"]

        folder, bboxs = get_deer(video_id, index, self.root)
        bbox_locations = torch.Tensor(
            [i for i in range(0, len(bboxs)) if bboxs[i] != []]
        )
        bbox_locations = bbox_locations[start - 1 : end - 1]

        indices = torch.linspace(0, len(bbox_locations) - 1, self.clip_size)
        indices = torch.clamp(indices, 0, len(bbox_locations) - 1).long()
        indices = bbox_locations[indices].int().tolist()

        rgb_frames = []
        frame_folder = f"{folder}/frames"
        for i in indices:
            frame = Image.open(f"{frame_folder}/{i + 1}.jpg")
            frame = frame.crop(bboxs[i])
            rgb_frames.append(self.frame_transform(frame))

        flow_frames = []
        for i in indices:
            frame_h = Image.open(f"{folder}/horizontal_flow/{i + 1}.jpg")
            frame_v = Image.open(f"{folder}/vertical_flow/{i + 1}.jpg")

            width, height = frame.size
            bbox = bboxs[i]
            bounding_box = (
                int((bbox[0]) * width), # x
                int((bbox[1]) * height), # y
                int((bbox[0] + bbox[2]) * width), # x + w
                int((bbox[1] + bbox[3]) * height), # y + h
            )

            frame_h = frame_h.crop(bounding_box)
            frame_v = frame_v.crop(bounding_box)

            frame_h = self.flow_transform(frame_h).squeeze()
            frame_v = self.flow_transform(frame_v).squeeze()

            composite = torch.stack([frame_h, frame_v, frame_h], 0)
            flow_frames.append(composite)

        # stack frames to (T, C, H, W)
        try:
            video = torch.stack(rgb_frames, 0)
        except:
            print("No frames found for {} {} {} {}".format(video_id, index, start, end))
            return None

        # reshape to (C, T, H, W)
        video = torch.permute(video, (1, 0, 2, 3))

        output = (video, self.kinetic_class_to_idx[target], self.class_to_idx[target])
        return output

    def _find_classes(self):
        # 0 = static, 1 = kinetic
        kinetic_classes = [0, 1]
        kinetic_class_to_idx = {
            "walk_left": 1,
            "walk_right": 1,
            "walk_towards": 1,
            "walk_away": 1,
            "standing": 0,
            "grooming": 0,
            "browsing": 0,
            "ear_scratching": 0,
        }
        classes = self.annotations["target"].fillna("none").unique()
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return kinetic_classes, kinetic_class_to_idx, classes, class_to_idx
