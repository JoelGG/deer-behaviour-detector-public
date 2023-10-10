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
    else:
        video_ids = os.listdir(args.video_path)
    random.shuffle(video_ids)

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

            h_flow_path = f"{args.video_path}/{video}/horizontal_flow/{frame}.jpg"
            v_flow_path = f"{args.video_path}/{video}/vertical_flow/{frame}.jpg"

            if os.path.exists(h_flow_path) and os.path.exists(v_flow_path):
                continue

            frame_1 = cv2.imread(frame_1_path)
            frame_2 = cv2.imread(frame_2_path)
            frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
            frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

            dtvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = dtvl1.calc(frame_1_gray, frame_2_gray, None)
            horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            horz = horz.astype("uint8")
            vert = vert.astype("uint8")

            if not os.path.exists(f"{args.video_path}/{video}/horizontal_flow"):
                os.mkdir(f"{args.video_path}/{video}/horizontal_flow")
            if not os.path.exists(f"{args.video_path}/{video}/vertical_flow"):
                os.mkdir(f"{args.video_path}/{video}/vertical_flow")

            cv2.imwrite(h_flow_path, horz)
            cv2.imwrite(v_flow_path, vert)


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
