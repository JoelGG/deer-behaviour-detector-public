from tqdm import tqdm
import os
import argparse
import random
import cv2
import detection.process_video


def main(args):
    video_ids = os.listdir(args.video_path)
    random.shuffle(video_ids)

    options = detection.process_video.ProcessVideoOptions()
    options.model_file = "/user/work/ki19061/megadetector/md_v5a.0.0.pt"
    options.json_confidence_threshold = 0.8

    for video in tqdm(video_ids, position=0):
        video_files = os.listdir(f"{args.video_path}/{video}")
        print(video_files)
        if "detections.json" not in video_files:
            print(f"No detections.json file found for {args.video_path}/{video}")
            continue

        video_files_video = [
            x for x in video_files if (x.endswith(".MP4") or x.endswith(".AVI"))
        ]
        if len(video_files_video) == 0:
            print(f"No video files found for {args.video_path}/{video}")
            continue
        elif len(video_files_video) > 1:
            print(f"More than one video file found for {args.video_path}/{video}")
            continue
        else:
            video_file = video_files_video[0]

        options.input_video_file = f"{args.video_path}/{video}/{video_file}"
        options.output_json_file = f"{args.video_path}/{video}/detections.json"
        detection.process_video.process_video(options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video file")
    args = parser.parse_args()
    main(args)
