from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

import os
import gc
import argparse
import random
import csv


def main(args):
    if args.annotation_path:
        video_ids = get_video_ids(args.annotation_path)
        print(video_ids)
    else:
        video_ids = os.listdir(args.video_path)
    random.shuffle(video_ids)

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        report_gpu()

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    weights = Raft_Large_Weights.DEFAULT
    optical_transforms = weights.transforms()

    for video in tqdm(video_ids, position=0):
        video_files = os.listdir(f"{args.video_path}/{video}")
        print(video_files)
        if "frames" not in video_files:
            print(f"No frames for {args.video_path}/{video}")
            continue

        path_to_frames = f"{args.video_path}/{video}/frames"
        frames = sorted(os.listdir(path_to_frames))

        for frame in tqdm(range(1, len(frames)), position=1):
            frame_1_path = f"{args.video_path}/{video}/frames/{frame}.jpg"
            frame_2_path = f"{args.video_path}/{video}/frames/{frame + 1}.jpg"

            flow_path = f"{args.video_path}/{video}/flow/{frame}.jpg"
            h_flow_path = f"{args.video_path}/{video}/horizontal_flow/{frame}.jpg"
            v_flow_path = f"{args.video_path}/{video}/vertical_flow/{frame}.jpg"

            if (
                os.path.exists(flow_path)
                and os.path.exists(h_flow_path)
                and os.path.exists(v_flow_path)
            ):
                print(f"Optical flow for {frame} already exists")
                continue

            base_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((520, 960), antialias=False)]
            )

            frame_1, frame_2 = optical_transforms(
                base_transform(Image.open(frame_1_path)).unsqueeze(0),
                base_transform(Image.open(frame_2_path)).unsqueeze(0),
            )

            list_of_flows = model(frame_1.to(device), frame_2.to(device))
            predicted_flows = list_of_flows[-1]
            flow_imgs = flow_to_image(predicted_flows).squeeze(0)

            if not os.path.exists(f"{args.video_path}/{video}/flow"):
                os.mkdir(f"{args.video_path}/{video}/flow")
            if not os.path.exists(f"{args.video_path}/{video}/horizontal_flow"):
                os.mkdir(f"{args.video_path}/{video}/horizontal_flow")
            if not os.path.exists(f"{args.video_path}/{video}/vertical_flow"):
                os.mkdir(f"{args.video_path}/{video}/vertical_flow")

            transforms.ToPILImage()(flow_imgs).save(flow_path)
            transforms.ToPILImage()(predicted_flows[0, 0]).save(h_flow_path)
            transforms.ToPILImage()(predicted_flows[0, 1]).save(v_flow_path)


def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


def get_video_ids(annotation_file):
    video_ids = []
    with open(annotation_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_ids.append(row["video_id"])
    return list(set(video_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--annotation-path", type=str, help="Path to video file")
    args = parser.parse_args()
    main(args)
