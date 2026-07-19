"""End-to-end smoke test on SYNTHETIC features (no cache needed).

Builds learnable fake 512-d features (class-dependent means), writes them to the
cache in the exact cache format, runs ours + naive + joint with tiny epochs, and
checks shapes/metrics come out sane. Catches wiring bugs before the real run.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import common  # noqa: E402
from metrics import compute_metrics  # noqa: E402

# point the cache at a temp location with synthetic data
common.CACHE_DIR = os.path.join(os.path.dirname(__file__), "_tmp_cache")
import train_cl  # noqa: E402
train_cl.CACHE_DIR = common.CACHE_DIR


def make_synthetic(n_per_class_train=300, n_per_class_test=80, d=512, n_classes=10):
    g = torch.Generator().manual_seed(0)
    mus = torch.randn(n_classes, d, generator=g) * 2.0
    def build(n_per):
        xs, ys = [], []
        for c in range(n_classes):
            xs.append(mus[c] + torch.randn(n_per, d, generator=g))
            ys.append(torch.full((n_per,), c, dtype=torch.long))
        return torch.cat(xs).abs(), torch.cat(ys)  # abs: mimic post-ReLU resnet feats
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    xtr, ytr = build(n_per_class_train)
    xte, yte = build(n_per_class_test)
    torch.save({"features": xtr, "labels": ytr},
               os.path.join(common.CACHE_DIR, "cifar10_resnet18_train.pt"))
    torch.save({"features": xte, "labels": yte},
               os.path.join(common.CACHE_DIR, "cifar10_resnet18_test.pt"))


def main():
    make_synthetic()
    cfg = dict(common.DEFAULT_CONFIG)
    cfg.update({"epochs": 3, "seed": 42})

    for method in ["naive", "ours", "mas_adapter"]:
        cfg["method"] = method
        payload, A, stats = train_cl.run_sequence(cfg)
        assert A.shape == (5, 5)
        m = compute_metrics(A)
        assert 0 <= m["avg_acc"] <= 1 and m["plasticity"] > 0.8, m
        print(f"{method}: {m}")
        if method == "ours":
            assert stats["dormancy"] is not None
            print("  dormancy:", stats["dormancy"])
            print("  claimed per task:", stats["claimed_per_task"])
            print("  delta_rel:", [round(d, 4) for d in stats["delta_rel"]])
        else:
            print("  accidental dormancy:", stats["accidental_dormancy"])

    cfg["method"] = "joint"
    per_task = train_cl.run_joint(cfg)
    assert len(per_task) == 5
    print("joint per-task:", [round(p, 3) for p in per_task])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
