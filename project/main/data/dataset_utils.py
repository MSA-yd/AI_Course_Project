import torch
from torch.utils.data import Subset

def get_class_sample_counts(dataset):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        original_dataset = dataset
        subset_indices = None

    counts = [0] * len(original_dataset.classes)

    if subset_indices is None:
        for _, label in original_dataset:
            counts[label] += 1
    else:
        for idx in subset_indices:
            label = original_dataset.targets[idx]
            counts[label] += 1

    return torch.tensor(counts)