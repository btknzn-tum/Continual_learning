"""Run the frozen encoder ONCE over the dataset and cache features to disk.

Supports multiple encoders (resnet18/50, clip_vitb32) and MULTIPLE INSERTION
DEPTHS in a single forward pass — "adapters used piece-by-piece, not only at
the end". Each depth is saved as its own cache file so downstream CL runs just
pick a (dataset, backbone, depth) triple.

Cache file: cache/{dataset}_{backbone}_{depth}_{split}.pt
            = {"features": Float[N,D], "labels": Long[N]}
Backwards-compatible alias: depth "final" also written as the old
            {dataset}_{backbone}_{split}.pt name for resnet18.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import CACHE_DIR, DATA_DIR, get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TF = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def resnet_depth_features(model, x):
    """Return dict depth_name -> pooled feature [B, C] for each ResNet stage."""
    out = {}
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.layer1(z); out["layer1"] = F.adaptive_avg_pool2d(z, 1).flatten(1)
    z = model.layer2(z); out["layer2"] = F.adaptive_avg_pool2d(z, 1).flatten(1)
    z = model.layer3(z); out["layer3"] = F.adaptive_avg_pool2d(z, 1).flatten(1)
    z = model.layer4(z); out["layer4"] = F.adaptive_avg_pool2d(z, 1).flatten(1)
    out["final"] = out["layer4"]  # final pooled feature == layer4 pooled
    return out


def get_encoder(name: str):
    """Return (model, transform, extract_fn). extract_fn(model, x) -> {depth: feat}."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        return m, IMAGENET_TF, resnet_depth_features
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        return m, IMAGENET_TF, resnet_depth_features
    if name == "clip_vitb32":
        import open_clip
        m, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")

        def clip_depths(model, x):
            # Version-robust extraction: forward via encode_image (always correct)
            # and capture intermediate blocks with hooks. Never re-implement the
            # ViT forward — open_clip switched LND->batch-first between versions,
            # and a wrong layout silently yields garbage (chance-level) features.
            v = model.visual
            taps = {3: "block3", 6: "block6", 9: "block9"}
            captured, hooks = {}, []

            def mk_hook(name):
                def h(_mod, _inp, out):
                    captured[name] = out[0] if isinstance(out, tuple) else out
                return h

            for i, blk in enumerate(v.transformer.resblocks):
                if (i + 1) in taps:
                    hooks.append(blk.register_forward_hook(mk_hook(taps[i + 1])))
            try:
                final = model.encode_image(x)
            finally:
                for h in hooks:
                    h.remove()
            B = x.shape[0]
            out = {"final": final.float()}
            for name, t in captured.items():
                # CLS = sequence index 0; detect whether batch dim is 0 (NLD)
                # or 1 (LND) by matching B (batch != seq_len for our batches).
                if t.shape[0] == B and t.shape[1] != B:
                    cls = t[:, 0, :]
                elif t.shape[1] == B:
                    cls = t[0]
                else:
                    cls = t[:, 0, :]
                out[name] = cls.float()
            return out

        return m, preprocess, clip_depths
    raise ValueError(f"unknown encoder {name}")


class RGBWrapper(torch.utils.data.Dataset):
    """Wrap a dataset so every image is 3-channel RGB before the encoder transform
    (MNIST/Fashion/KMNIST are grayscale; ResNet/CLIP expect RGB)."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        return self.transform(img.convert("RGB")), y


# 5-Datasets benchmark members. notMNIST -> KMNIST (torchvision-native, fully
# reproducible; documented substitution). Each is one task, 10 classes.
FIVE_DATASETS = ["cifar10", "mnist", "svhn", "fashion", "kmnist"]


def build_dataset(name: str, train: bool, raw_transform):
    """Return a dataset yielding (transformed_rgb_tensor, label)."""
    kw = dict(root=DATA_DIR, download=True)
    if name == "cifar10":
        base = datasets.CIFAR10(train=train, **kw)
    elif name == "cifar100":
        base = datasets.CIFAR100(train=train, **kw)
    elif name == "mnist":
        base = datasets.MNIST(train=train, **kw)
    elif name == "fashion":
        base = datasets.FashionMNIST(train=train, **kw)
    elif name == "kmnist":
        base = datasets.KMNIST(train=train, **kw)
    elif name == "svhn":
        base = datasets.SVHN(split="train" if train else "test", **kw)
    else:
        raise ValueError(f"unknown sub-dataset {name}")
    return RGBWrapper(base, raw_transform)


def cache_one(name, backbone, model, tf, extract, depths, batch_size, workers, device):
    for split, train in [("train", True), ("test", False)]:
        targets = {d: os.path.join(CACHE_DIR, f"{name}_{backbone}_{d}_{split}.pt")
                   for d in depths}
        if all(os.path.exists(p) for p in targets.values()):
            print(f"skip {name} {split} (cached)", flush=True)
            continue
        ds = build_dataset(name, train, tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        buffers = {d: [] for d in depths}
        labels = []
        with torch.no_grad():
            for i, (x, y) in enumerate(dl):
                feats = extract(model, x.to(device))
                for d in depths:
                    buffers[d].append(feats[d].cpu())
                labels.append(y)
                if i % 25 == 0:
                    print(f"{name}/{split}: batch {i}/{len(dl)}", flush=True)
        y_all = torch.cat(labels).long()
        for d in depths:
            feat = torch.cat(buffers[d]).float()
            torch.save({"features": feat, "labels": y_all}, targets[d])
            print(f"saved {targets[d]} {tuple(feat.shape)}", flush=True)


def main(dataset: str, backbone: str, batch_size: int, workers: int, depths):
    device = get_device()
    model, tf, extract = get_encoder(backbone)
    model.eval().to(device)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if dataset == "fivedata":
        for name in FIVE_DATASETS:
            cache_one(name, backbone, model, tf, extract, depths,
                      batch_size, workers, device)
        return

    ds_cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[dataset]

    for split, train in [("train", True), ("test", False)]:
        # skip if all requested depths already cached
        targets = {d: os.path.join(CACHE_DIR, f"{dataset}_{backbone}_{d}_{split}.pt")
                   for d in depths}
        if all(os.path.exists(p) for p in targets.values()):
            print(f"skip {split} (all depths cached)", flush=True)
            continue
        ds = ds_cls(DATA_DIR, train=train, download=True, transform=tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        buffers = {d: [] for d in depths}
        labels = []
        with torch.no_grad():
            for i, (x, y) in enumerate(dl):
                feats = extract(model, x.to(device))
                for d in depths:
                    buffers[d].append(feats[d].cpu())
                labels.append(y)
                if i % 25 == 0:
                    print(f"{split}: batch {i}/{len(dl)}", flush=True)
        y_all = torch.cat(labels).long()
        for d in depths:
            feat = torch.cat(buffers[d]).float()
            torch.save({"features": feat, "labels": y_all}, targets[d])
            print(f"saved {targets[d]} {tuple(feat.shape)}", flush=True)
            if d == "final":  # legacy alias for backwards compat
                alias = os.path.join(CACHE_DIR, f"{dataset}_{backbone}_{split}.pt")
                if not os.path.exists(alias):
                    torch.save({"features": feat, "labels": y_all}, alias)


RESNET_DEPTHS = ["layer1", "layer2", "layer3", "layer4", "final"]
CLIP_DEPTHS = ["block3", "block6", "block9", "final"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--depths", nargs="*", default=None,
                   help="subset of depths; default = all for the backbone")
    a = p.parse_args()
    default_depths = CLIP_DEPTHS if a.backbone.startswith("clip") else RESNET_DEPTHS
    main(a.dataset, a.backbone, a.batch_size, a.workers, a.depths or default_depths)
