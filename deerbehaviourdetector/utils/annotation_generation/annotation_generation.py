from tqdm import tqdm
import os
import argparse
import random
import cv2
import detection.process_video
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(args):
    if args.annotation_path:
        video_ids = get_video_ids(args.annotation_path)
        print(video_ids)
    elif args.text_list_path:
        video_ids = get_ids_from_txt(args.text_list_path, args.video_path)
        print(video_ids)
    else:
        video_ids = os.listdir(args.video_path)
    random.shuffle(video_ids)

    options = detection.process_video.ProcessVideoOptions()
    options.model_file = "/user/work/ki19061/megadetector/md_v5a.0.0.pt"
    options.json_confidence_threshold = 0.7
    # options.render_output_video = True
    # options.keep_extracted_frames = True

    for video in tqdm(video_ids, position=0):
        video_files = os.listdir(f"{args.video_path}/{video}")
        print(video_files)

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
        # options.frame_folder = f"{args.video_path}/{video}/extracted_frames"
        # # options.output_video_file = f"{args.video_path}/{video}/solution_video.MP4"
        print("start processing")
        detection.process_video.process_video(options)
        print("finished processing")


def get_video_ids(annotation_file):
    video_ids = []
    with open(annotation_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_ids.append(row["video_id"])
    return list(set(video_ids))


def get_ids_from_txt(text_file, video_path):
    camera_ids = []
    with open(text_file) as txtfile:
        for line in txtfile:
            camera_ids.append(line.strip())

    videos = os.listdir(args.video_path)
    video_ids = []
    for video in videos:
        if video.startswith(tuple(camera_ids)):
            video_ids.append(video)
    return list(set(video_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--annotation-path", type=str, help="Path to video file")
    parser.add_argument("--text-list-path", type=str)
    args = parser.parse_args()
    main(args)
