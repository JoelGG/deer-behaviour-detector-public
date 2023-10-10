import argparse
import os
import cv2
import json
from tqdm import tqdm


def main(args):
    # video_ids = get_ids_from_txt(args.text_list_path, args.video_path)
    # video_ids = sorted(video_ids)
    # print(video_ids)
    # for video_id in video_ids:
    annotate_video("T09L00009")


def annotate_video(video_id):
    frames_path = f"/user/work/ki19061/dataset/{video_id}/frames"
    output_path = f"/user/work/ki19061/dataset/{video_id}/frame-annotations"
    detections_file = f"/user/work/ki19061/dataset/{video_id}/detections.json"

    frames = os.listdir(frames_path)
    print(frames)
    if not os.path.exists(detections_file):
        print(f"no detections for {detections_file}")
        return None
    with open(detections_file) as f:
        images = json.load(f)["images"]
        for i, image in enumerate(images):
            print(i)
            frame_detections = image["detections"]
            if len(frame_detections) == 0:
                continue
            frame_detections = sorted(frame_detections, key=lambda x: x["bbox"][0])
            frame = cv2.imread(f"{frames_path}/{i + 1}.jpg")
            height, width, channels = frame.shape

            for j, detection in enumerate(frame_detections):
                bbox = detection["bbox"]
                start = (int((bbox[0]) * width), int((bbox[1]) * height))  # (x, y)
                end = (
                    int((bbox[0] + bbox[2]) * width),
                    int((bbox[1] + bbox[3]) * height),
                )  # ((x + w), (y + h))
                print(start)
                print(end)
                cv2.rectangle(frame, start, end, (36, 255, 12), 1)
                frame = cv2.putText(
                    frame,
                    f"index: {j}",
                    (int((bbox[0]) * width), int((bbox[1]) * height) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (36, 255, 12),
                    2,
                )
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            cv2.imwrite(f"{output_path}/{i + 1}.jpg", frame)


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
    parser.add_argument("video_path", type=str)
    parser.add_argument("text_list_path", type=str)
    args = parser.parse_args()
    main(args)
