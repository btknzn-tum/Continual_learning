"""Cache MNIST as flattened normalized pixel vectors (784-d) — no backbone.

Split-MNIST is the classic quick CL benchmark; an MLP on raw pixels is the
standard setup, so the adapter consumes pixels directly (d_in=784).
Saved in the exact same cache format: mnist_pixels_{train,test}.pt
"""
import os

import torch
from torchvision import datasets

from common import CACHE_DIR, DATA_DIR


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split, train in [("train", True), ("test", False)]:
        out_path = os.path.join(CACHE_DIR, f"mnist_pixels_{split}.pt")
        if os.path.exists(out_path):
            print(f"skip {out_path} (exists)")
            continue
        ds = datasets.MNIST(DATA_DIR, train=train, download=True)
        x = ds.data.float().div_(255.0)
        x = (x - 0.1307) / 0.3081
        feats = x.reshape(len(ds), -1)
        blob = {"features": feats, "labels": ds.targets.long()}
        torch.save(blob, out_path)
        print(f"saved {out_path} features={tuple(feats.shape)}")


if __name__ == "__main__":
    main()
