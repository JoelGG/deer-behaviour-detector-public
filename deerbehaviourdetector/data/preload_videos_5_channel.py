import csv
import os
import torch

from utils.data.get_deer import get_deer
from torchvision import transforms
from PIL import Image


class PreloaderFiveChannel:
    def __init__(self, root, annotations):
        self.root = root
        self.annotations = annotations
        (
            self.kinetic_classes,
            self.kinetic_class_to_idx,
            self.classes,
            self.class_to_idx,
        ) = self.__find_classes()

    def preload(
        self,
        frame_transform,
        flow_transform,
        crop=True,
        clip_size=16,
        cache_name="data",
    ):
        print(
            f"/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/data/cache/{cache_name}.pt"
        )
        if os.path.exists(
            f"/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/data/cache/{cache_name}.pt"
        ):
            return torch.load(
                f"/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/data/cache/{cache_name}.pt"
            )
        data = []

        with open(self.annotations, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]
                index = int(row["index"])
                target = row["target"]
                start = int(row["start"])
                end = int(row["end"])

                folder, bboxs = get_deer(video_id, index, self.root)
                indices = self.__get_indices(bboxs, clip_size, start, end)

                frames = {"rgb": [], "rgb_cropped": [], "flow": [], "flow_cropped": []}

                frame_folder = f"{folder}/frames"
                flow_folder_h = f"{folder}/horizontal_flow"
                flow_folder_v = f"{folder}/vertical_flow"
                print(frame_folder)
                for i in indices:
                    frame = Image.open(f"{frame_folder}/{i + 1}.jpg")

                    flow_h = Image.open(f"{flow_folder_h}/{i+1}.jpg")
                    flow_v = Image.open(f"{flow_folder_v}/{i+1}.jpg")
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

                    all_frames = torch.cat(
                        (frame_transform(frame), flow_transform(flow_t))
                    )[
                        :5,
                    ]
                    all_frames_cropped = torch.cat(
                        (
                            frame_transform(self.__crop(frame, bboxs[i])),
                            flow_transform(self.__crop(flow_t, bboxs[i])),
                        ),
                    )[
                        :5,
                    ]

                    frames["rgb"].append(all_frames)
                    frames["rgb_cropped"].append(all_frames_cropped)

                    frames["flow"].append(all_frames)
                    frames["flow_cropped"].append(all_frames_cropped)

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
                            video_id, index, start, end
                        )
                    )
                    return None

                # reshape to (C, T, H, W)
                video = {
                    "rgb": torch.permute(video["rgb"], (1, 0, 2, 3)),
                    "rgb_cropped": torch.permute(video["rgb_cropped"], (1, 0, 2, 3)),
                    "flow": torch.permute(video["flow"], (1, 0, 2, 3)),
                    "flow_cropped": torch.permute(video["flow_cropped"], (1, 0, 2, 3)),
                }

                output = {
                    "video": video,
                    "kinetic_target": self.kinetic_class_to_idx[target],
                    "target": self.class_to_idx[target],
                }
                data.append(output)
        torch.save(
            data,
            f"/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/data/cache/{cache_name}.pt",
        )
        return data

    def __get_indices(self, bboxs, clip_size, start, end):
        bbox_locations = torch.Tensor(
            [i for i in range(0, len(bboxs) - 1) if bboxs[i] != []]
        )
        bbox_locations = bbox_locations[start - 1 : end - 1]
        indices = torch.linspace(0, len(bbox_locations) - 1, clip_size)
        indices = torch.clamp(indices, 0, len(bbox_locations) - 1).long()
        indices = bbox_locations[indices].int().tolist()
        return indices

    def __crop(self, frame, bbox):
        width, height = frame.size
        bounding_box = (
            int((bbox[0]) * width),  # x
            int((bbox[1]) * height),  # y
            int((bbox[0] + bbox[2]) * width),  # x + w
            int((bbox[1] + bbox[3]) * height),  # y + h
        )
        # print(bounding_box)
        return frame.crop(bounding_box)

    def __find_classes(self):
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
