from tqdm import tqdm
import os
import argparse
import cv2
import random
import csv


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

    for video in tqdm(video_ids, position=0):
        video_files = os.listdir(f"{args.video_path}/{video}")
        # if "detections.json" not in video_files:
        #     print(f"No bounding boxes for {args.video_path}/{video}")
        #     continue

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

        cap = cv2.VideoCapture(f"{args.video_path}/{video}/{video_file}")

        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_count in tqdm(range(1, frame_total + 1), position=1):
            frame_location = f"{args.video_path}/{video}/frames/{frame_count}.jpg"
            if os.path.exists(frame_location):
                print(f"Frame {frame_count} already exists for {video}")
                continue

            success, frame = cap.read()
            if not success:
                print(f"Error reading frame {frame_count} from {video}")
                continue

            try:
                if not os.path.exists(f"{args.video_path}/{video}/frames"):
                    os.mkdir(f"{args.video_path}/{video}/frames")
            except OSError:
                print(f"Error creating directory for {video}")

            cv2.imwrite(frame_location, frame)

        cap.release()
        cv2.destroyAllWindows()


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
