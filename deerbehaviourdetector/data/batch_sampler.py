import random
from typing import Optional, Sized
import torch
import torchvision
import torch.utils.data


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, class_count):
        self.dataset = [[] for _ in range(class_count)]
        self.max_samples_per_class = 0

        for sample_index, sample in enumerate(data_source):
            target = sample["target"]
            if target not in self.dataset:
                self.dataset[target].append(sample_index)

        for target in range(class_count):
            if len(self.dataset[target]) > self.max_samples_per_class:
                self.max_samples_per_class = len(self.dataset[target])

        for target in range(class_count):
            while len(self.dataset[target]) < self.max_samples_per_class:
                self.dataset[target].append(random.choice(self.dataset[target]))
