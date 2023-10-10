import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


def balanced_wrsampler(dataset):
    labels = np.array([sample["target"] for sample in dataset])
    class_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    print(class_count)
    weight = 1.0 / class_count
    weight = weight**2
    samples_weight = np.array([weight[t] for t in labels])
    print(samples_weight)
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    print(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler
