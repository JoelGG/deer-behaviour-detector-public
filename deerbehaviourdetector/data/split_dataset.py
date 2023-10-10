import os
import csv

import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.data.get_deer import get_deer

BRANDENBURG_ROOT = "/projects/Animal_Biometrics/Video_All"


class SplitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        annotations,
        frame_transform,
        flow_transform,
        clip_size=16,
        frame_skip=4,
        cache_name="data",
    ):
        super(SplitDataset).__init__()

        self.root = root
        self.annotations = annotations
        (
            self.kinetic_classes,
            self.kinetic_class_to_idx,
            self.classes,
            self.class_to_idx,
        ) = self.get_classes()
        self.frame_transform = frame_transform
        self.flow_transform = flow_transform
        self.clip_size = clip_size
        self.frame_skip = frame_skip

        self.data = []

        cache_path = f"/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/data/cache/{cache_name}.pt"
        if os.path.exists(cache_path):
            print(f"Loading cached dataset at {cache_path}")
            self.data = torch.load(cache_path)
        else:
            print("No cache found, constructing dataset")
            self.data = self.init_dataset()
            torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def init_dataset(self):
        data = []

        with open(self.annotations, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder, bboxs = get_deer(row["video_id"], int(row["index"]), self.root)
                print(f'loading {row["video_id"]}, {row["start"]}, {row["end"]}')
                indices = self.get_indices(
                    bboxs, self.clip_size, int(row["start"]), int(row["end"])
                )
                indices = indices[:: self.frame_skip]

                block_count = (len(indices) - 1) // self.clip_size
                for block in range(0, block_count):
                    print(indices)
                    print(len(indices))
                    print(indices[block * self.clip_size])
                    print(
                        f'loading block {row["video_id"]}, {indices[block * self.clip_size]}, {indices[(block + 1) * self.clip_size]}'
                    )
                    block_indices = indices[
                        block * self.clip_size : (block + 1) * self.clip_size
                    ]
                    frames = {
                        "rgb": [],
                        "rgb_cropped": [],
                        "flow": [],
                        "flow_cropped": [],
                    }

                    frame_folder = f"{folder}/frames"
                    flow_folder_h = f"{folder}/horizontal_flow"
                    flow_folder_v = f"{folder}/vertical_flow"
                    for i in block_indices:
                        frame = self.process_frame(
                            i, bboxs[i], frame_folder, flow_folder_h, flow_folder_v
                        )
                        frames["rgb"].append(frame["rgb"])
                        frames["rgb_cropped"].append(frame["rgb_cropped"])
                        frames["flow"].append(frame["flow"])
                        frames["flow_cropped"].append(frame["flow_cropped"])

                    # stack frames to (T, C, H, W)
                    try:
                        print(frames["rgb"][0].size())
                        print(frames["flow"][0].size())
                        video = {
                            "rgb": torch.stack(frames["rgb"], 0),
                            "rgb_cropped": torch.stack(frames["rgb_cropped"], 0),
                            "flow": torch.stack(frames["flow"], 0),
                            "flow_cropped": torch.stack(frames["flow_cropped"], 0),
                        }
                    except:
                        print(
                            "No frames found for {} {} {} {}".format(
                                row["video_id"],
                                int(row["index"]),
                                int(block_count * self.clip_size),
                                int((block_count + 1) * self.clip_size),
                            )
                        )
                        return None

                    # reshape to (C, T, H, W)
                    video = {
                        "rgb": torch.permute(video["rgb"], (1, 0, 2, 3)),
                        "rgb_cropped": torch.permute(
                            video["rgb_cropped"], (1, 0, 2, 3)
                        ),
                        "flow": torch.permute(video["flow"], (1, 0, 2, 3)),
                        "flow_cropped": torch.permute(
                            video["flow_cropped"], (1, 0, 2, 3)
                        ),
                    }

                    output = {
                        "video": video,
                        "kinetic_target": self.kinetic_class_to_idx[row["target"]],
                        "target": self.class_to_idx[row["target"]],
                        "video_id": row["video_id"],
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                    }
                    data.append(output)
        return data

    def process_frame(
        self,
        idx,
        bbox,
        frame_folder,
        flow_folder_h,
        flow_folder_v,
    ):
        frame = Image.open(f"{frame_folder}/{idx + 1}.jpg")

        flow_h = Image.open(f"{flow_folder_h}/{idx + 1}.jpg")
        flow_v = Image.open(f"{flow_folder_v}/{idx + 1}.jpg")
        f_width, f_height = flow_v.size
        flow_t = transforms.ToPILImage()(
            torch.stack(
                [
                    transforms.ToTensor()(flow_h),
                    transforms.ToTensor()(flow_v),
                    torch.zeros((1, f_height, f_width)),
                ]
            ).squeeze()
        )

        return {
            "rgb": self.frame_transform(frame),
            "rgb_cropped": self.frame_transform(self.crop(frame, bbox)),
            "flow": self.flow_transform(flow_t),
            "flow_cropped": self.flow_transform(self.crop(flow_t, bbox)),
        }

    def get_indices(self, bboxs, clip_size, start, end):
        bbox_locations = [i for i in range(start, end - 1) if bboxs[i] != []]
        return bbox_locations

    def crop(self, frame, bbox):
        width, height = frame.size
        bounding_box = (
            int((bbox[0]) * width),  # x
            int((bbox[1]) * height),  # y
            int((bbox[0] + bbox[2]) * width),  # x + w
            int((bbox[1] + bbox[3]) * height),  # y + h
        )
        # print(bounding_box)
        return frame.crop(bounding_box)

    def get_classes(self):
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
        classes = [
            "walk_left",
            "walk_right",
            "walk_towards",
            "walk_away",
            "standing",
            "grooming",
            "browsing",
            "ear_scratching",
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return kinetic_classes, kinetic_class_to_idx, classes, class_to_idx
