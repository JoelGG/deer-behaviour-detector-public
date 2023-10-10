base = "/projects/Animal_Biometrics/Video_All"
destination = "/user/work/ki19061/deerbehaviourdetector/dataset"
path = "/mnt/storage/scratch/gi17153/Video_All\HT_T01\HT_cam11_432958_5887023_20190611\06110001"

import os
import csv


def main():
    with open(
        "../../Annotations_All.csv", mode="r", encoding="utf-8-sig"
    ) as annotations:
        reader = csv.DictReader(annotations, delimiter=",")
        for index, row in enumerate(reader):
            if row["species"] == "capreolus":
                path = f'{row["subfolder"]}/{row["upload_folder_name"]}/0{row["upload_file_name"]}'
                path = row["Path"][38:].replace("\\", "/")

                location = f"{base}{path}"

                video_format = ""
                if os.path.isfile(f"{location}.MP4"):
                    video_format = ".MP4"
                elif os.path.isfile(f"{location}.AVI"):
                    video_format = ".AVI"
                else:
                    print(f"File {location} not found")
                    break

                location = location + video_format
                dest = f'{destination}/{row["ID"]}/{row["ID"]}{video_format}'
                print(location)
                print(dest)

                os.system(f"mkdir {destination}/{row['ID']}")
                os.system(f"rsync {location} {dest}")
                print(f"File {index} uploaded")


if __name__ == "__main__":
    main()
