"""Run the frozen backbone ONCE over the dataset and cache 512-d features to disk.

After this, all training reads cache/*.pt and never touches images again.
Backbone is pluggable: resnet18 now, CLIP (open_clip) can be added in get_backbone.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import CACHE_DIR, DATA_DIR, get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_backbone(name: str):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()  # output = 512-d global-average-pool features
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return m, tf, 512
    # Phase 3: add e.g. "clip_vitb32" via open_clip here — same caching flow.
    raise ValueError(f"unknown backbone {name}")


def main(dataset: str, backbone: str, batch_size: int, workers: int):
    device = get_device()
    model, tf, dim = get_backbone(backbone)
    model.eval().to(device)
    ds_cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[dataset]
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split, train in [("train", True), ("test", False)]:
        out_path = os.path.join(CACHE_DIR, f"{dataset}_{backbone}_{split}.pt")
        if os.path.exists(out_path):
            print(f"skip {out_path} (exists)", flush=True)
            continue
        ds = ds_cls(DATA_DIR, train=train, download=True, transform=tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        feats, labels = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(dl):
                feats.append(model(x.to(device)).cpu())
                labels.append(y)
                if i % 25 == 0:
                    print(f"{split}: batch {i}/{len(dl)}", flush=True)
        blob = {"features": torch.cat(feats).float(), "labels": torch.cat(labels).long()}
        torch.save(blob, out_path)
        print(f"saved {out_path} features={tuple(blob['features'].shape)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    main(a.dataset, a.backbone, a.batch_size, a.workers)
