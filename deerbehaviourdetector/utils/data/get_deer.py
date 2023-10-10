import os
import json


def get_deer(video_id: str, index: int, dataset_root: str):
    video_folders = os.listdir(dataset_root)
    frames_folder, detection_file = get_paths(video_folders, video_id, dataset_root)
    bounding_boxes = detections_for_idx(detection_file, index)
    return frames_folder, bounding_boxes


def detections_for_idx(detection_file: str, index: int):
    detections = []
    with open(detection_file, "r") as f:
        data = json.load(f)
        images = data["images"]
        for image in images:
            frame_detections = image["detections"]
            frame_detections = sorted(frame_detections, key=lambda x: x["bbox"][0])
            if 0 <= index < len(frame_detections):
                detections.append(frame_detections[index]["bbox"])
            else:
                detections.append([])
    return detections


def get_paths(path, video_id, dataset_root):
    if video_id not in path:
        print(f"Folder {video_id} not found")
        exit(1)
    path_to_video = f"{dataset_root}/{video_id}"

    path_contents = os.listdir(path_to_video)
    if "frames" not in path_contents:
        print(f"Frames folder of {video_id} not found")
        exit(1)
    frames_folder = f"{path_to_video}"

    if "detections.json" not in os.listdir(path_to_video):
        print(f"Frames folder of {video_id} not found")
        exit(1)
    detections = f"{path_to_video}/detections.json"
    return frames_folder, detections
