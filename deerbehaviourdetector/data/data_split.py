import csv
import argparse
from sklearn.model_selection import train_test_split
import numpy as np


def train_test_indices(annotations, test_size=0.2):
    targets = []
    with open(annotations, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append(row["target"])
    train_indices, test_indices = train_test_split(
        range(len(targets)), test_size=test_size, stratify=targets
    )
    return train_indices, test_indices


def train_test_indices_ds(dataset, test_size=0.2):
    targets = []
    for value in dataset:
        targets.append(value["target"])
    train_indices, test_indices = train_test_split(
        range(len(targets)), test_size=test_size, stratify=targets
    )
    return train_indices, test_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=str, help="Path to annotations")
    args = parser.parse_args()
    train_indices, test_indices = train_test_indices(args.annotations)
    print(train_indices)
    print(test_indices)
