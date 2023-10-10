from tqdm import tqdm
import os
import argparse
import cv2
import random


def main(args):
    video_ids = os.listdir(args.video_path)
    video_ids.sort()
    open("video_ids.txt", "w").close()
    with open("video_ids.txt", "w") as f:
        for video in tqdm(video_ids, position=0):
            video_files = os.listdir(f"{args.video_path}/{video}")
            if "detections.json" in video_files and "frames" in video_files:
                f.write(f"{video}\n")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video file")
    args = parser.parse_args()
    main(args)
