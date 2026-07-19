"""Task splits over cached features. Split CIFAR-10: 5 tasks x 2 classes."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_splits(n_classes: int = 10, n_tasks: int = 5):
    per = n_classes // n_tasks
    assert per * n_tasks == n_classes
    return [list(range(t * per, (t + 1) * per)) for t in range(n_tasks)]


def task_tensors(features: torch.Tensor, labels: torch.Tensor, class_group):
    mask = torch.isin(labels, torch.tensor(class_group))
    x = features[mask]
    y_orig = labels[mask]
    remap = {c: i for i, c in enumerate(class_group)}
    y = torch.tensor([remap[int(c)] for c in y_orig], dtype=torch.long)
    return x, y


def task_loader(features, labels, class_group, batch_size=128, shuffle=True):
    x, y = task_tensors(features, labels, class_group)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def subsample(x: torch.Tensor, y: torch.Tensor, n: int, seed: int = 0):
    if len(x) <= n:
        return x, y
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n]
    return x[idx], y[idx]
