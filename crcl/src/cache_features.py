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
            v = model.visual
            # replicate open_clip ViT forward, tapping intermediate blocks
            z = v.conv1(x)                                   # [B, width, gh, gw]
            z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(z.dtype) + torch.zeros(
                z.shape[0], 1, z.shape[-1], dtype=z.dtype, device=z.device)
            z = torch.cat([cls, z], dim=1)
            z = z + v.positional_embedding.to(z.dtype)
            z = v.ln_pre(z)
            z = z.permute(1, 0, 2)  # LND
            taps = {3: "block3", 6: "block6", 9: "block9"}
            out = {}
            blocks = v.transformer.resblocks
            for i, blk in enumerate(blocks):
                z = blk(z)
                if (i + 1) in taps:
                    out[taps[i + 1]] = z[0].float()  # CLS token, [B, width]
            z = z.permute(1, 0, 2)
            pooled = v.ln_post(z[:, 0, :])
            if v.proj is not None:
                pooled = pooled @ v.proj
            out["final"] = pooled.float()
            return out

        return m, preprocess, clip_depths
    raise ValueError(f"unknown encoder {name}")


def main(dataset: str, backbone: str, batch_size: int, workers: int, depths):
    device = get_device()
    model, tf, extract = get_encoder(backbone)
    model.eval().to(device)
    ds_cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[dataset]
    os.makedirs(CACHE_DIR, exist_ok=True)

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
