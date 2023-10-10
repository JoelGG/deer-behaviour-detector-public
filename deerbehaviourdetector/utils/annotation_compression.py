import csv
import json
import logging

from path import Path

HEADERS = [
    "video_id",
    "frame",
    "detection_idx",
    "path",
    "activity",
    "start",
    "end",
]


def main():
    behaviour_annotations_path = (
        "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations.csv"
    )
    compressed_annotations_path = "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations_compressed.csv"

    with open(compressed_annotations_path, "w") as compressed_annotations, open(
        behaviour_annotations_path
    ) as behaviour_annotations:
        reader = csv.DictReader(behaviour_annotations, delimiter=",")
        writer = csv.DictWriter(
            compressed_annotations, fieldnames=HEADERS, delimiter=","
        )
        writer.writeheader()

        current = next(reader)
        frame = 0
        for index, row in enumerate(reader):
            if index > 1332:
                break
            if row["activity"] == "":
                continue
            if (current["activity"] != row["activity"]) or (
                current["video_id"] != row["video_id"]
            ):
                new_detection = {
                    "video_id": current["video_id"],
                    "frame": current["frame"],
                    "detection_idx": current["detection_idx"],
                    "path": current["path"],
                    "activity": current["activity"],
                    "start": current["frame"],
                    "end": frame,
                }
                writer.writerow(new_detection)

                current = row
            frame = row["frame"]


if __name__ == "__main__":
    main()
