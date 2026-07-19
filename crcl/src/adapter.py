"""Adapter: 2-layer ReLU MLP on frozen 512-d features + one linear head per task.

No BatchNorm anywhere (SynFlow's all-ones trick requires plain ReLU network).
"""
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, d_in: int = 512, d_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.heads = nn.ModuleDict()

    def add_head(self, task_id: int, n_classes: int):
        self.heads[str(task_id)] = nn.Linear(self.fc2.out_features, n_classes)

    def freeze_head(self, task_id: int):
        for p in self.heads[str(task_id)].parameters():
            p.requires_grad_(False)

    def hidden(self, x, zero_h1=None, zero_h2=None):
        h1 = torch.relu(self.fc1(x))
        if zero_h1 is not None:
            h1 = h1 * (~zero_h1).float()
        h2 = torch.relu(self.fc2(h1))
        if zero_h2 is not None:
            h2 = h2 * (~zero_h2).float()
        return h1, h2

    def forward(self, x, task_id: int, zero_h1=None, zero_h2=None):
        h1, h2 = self.hidden(x, zero_h1, zero_h2)
        logits = self.heads[str(task_id)](h2)
        return h1, h2, logits
