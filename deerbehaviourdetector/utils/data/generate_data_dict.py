import csv
import argparse

def main(args):
    new_annotations = generate_data_dict(args.annotations, args.clip_len)
    if args.output:
        with open(args.output, "w+") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "index", "target", "start", "end"])
            writer.writeheader()
            for row in new_annotations:
                writer.writerow(row)
    else:
        for row in new_annotations:
            print(row)
    
                
def generate_data_dict(annotations, clip_len):
    new_annotations = []
    with open(args.annotations, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            index = row["index"]
            target = row["target"]
            start = int(row["start"])
            end = int(row["end"])

            while start + clip_len < end:
                new_annotations.append({
                    "video_id": video_id, 
                    "index": index, 
                    "target": target, 
                    "start": start, 
                    "end": start + clip_len,
                })
                start += clip_len
            if start < end:
                new_annotations.append({
                    "video_id": video_id, 
                    "index": index, 
                    "target": target, 
                    "start": start, 
                    "end": end,
                })
    return new_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=str, help="Path to annotations")
    parser.add_argument("--clip-len", default=300, type=int, help="Path to annotations")
    parser.add_argument("--output", type=str, help="Path to annotations")
    args = parser.parse_args()
    main(args)
