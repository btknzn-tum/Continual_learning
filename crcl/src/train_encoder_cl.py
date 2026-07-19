"""Selective-plasticity CL with a partially trainable encoder (our method).

Per task t>0 the loss is
    CE(current task)
  + alpha     * sum S_ad  (theta_ad - theta_ad_old)^2     [adapter MAS penalty]
  + alpha_enc * sum S_enc (theta_e  - theta_e_old)^2      [penalty on the freed
                                                           encoder fraction]
  + sim_lambda * MSE(f_t(x), f_{t-1}(x))                  [feature-similarity
                                                           anchor to the frozen
                                                           previous encoder]
with the encoder HARD-masked so only the bottom enc_q fraction of parameters
(by accumulated MAS importance through encoder+adapter) receives gradients.
Once a task claims a parameter its importance rises, so it (a) gets penalized
and (b) rotates OUT of the trainable set at the next task — capacity rotation
without stored per-task masks (unlike PackNet).

Task 0 trains the adapter only (encoder frozen; importance undefined yet) —
features are extracted once, so task 0 matches the cached reg:mas pipeline.
BatchNorm stays in eval() mode for the entire stream (running stats frozen);
BN affine params may still enter the trainable subset. NOT full fine-tuning.

Ablations: --sim-lambda 0 (no anchor), --alpha-enc 0 (no encoder penalty),
--enc-q 0 (frozen encoder == e2e replica of cached reg:mas).
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import models

from cache_features import (FIVE_DATASETS, IMAGENET_TF, build_dataset,
                            resnet_depth_features)
from common import DEFAULT_CONFIG, RESULTS_DIR, set_seed
from metrics import compute_metrics, save_result
from tasks import make_splits
from train_cl import PARAM_NAMES, _params, _snapshot, git_hash
from adapter import Adapter

TAP_DIMS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}


def bwt(A):
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, t] - A[t, t] for t in range(T - 1)]))


class TaskView(torch.utils.data.Dataset):
    """Subset of a base dataset restricted to `indices`, labels remapped to
    0..C_t-1 via the sorted class list."""

    def __init__(self, base, indices, classes):
        self.base = base
        self.indices = indices
        self.remap = {c: i for i, c in enumerate(sorted(classes))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.remap[int(y)]


def get_labels(wrapped):
    base = getattr(wrapped, "base", wrapped)
    t = base.targets if hasattr(base, "targets") else base.labels
    if torch.is_tensor(t):
        return t.long().tolist()
    return [int(v) for v in t]


def build_stream(dataset, n_tasks):
    """Return (train_views, test_views, class_counts) of image datasets."""
    if dataset == "fivedata":
        train_views, test_views, counts = [], [], []
        for name in FIVE_DATASETS:
            tr = build_dataset(name, True, IMAGENET_TF)
            te = build_dataset(name, False, IMAGENET_TF)
            classes = sorted(set(get_labels(tr)))
            train_views.append(TaskView(tr, list(range(len(tr))), classes))
            test_views.append(TaskView(te, list(range(len(te))), classes))
            counts.append(len(classes))
        return train_views, test_views, counts
    tr = build_dataset(dataset, True, IMAGENET_TF)
    te = build_dataset(dataset, False, IMAGENET_TF)
    ytr = np.array(get_labels(tr))
    yte = np.array(get_labels(te))
    splits = make_splits(int(ytr.max()) + 1, n_tasks)
    train_views = [TaskView(tr, np.where(np.isin(ytr, g))[0].tolist(), g)
                   for g in splits]
    test_views = [TaskView(te, np.where(np.isin(yte, g))[0].tolist(), g)
                  for g in splits]
    return train_views, test_views, [len(g) for g in splits]


def sub_view(view, n, seed):
    if n >= len(view):
        return view
    g = torch.Generator().manual_seed(seed)
    return Subset(view, torch.randperm(len(view), generator=g)[:n].tolist())


def enc_named_params(encoder):
    """Encoder params eligible for selective training (classifier fc excluded)."""
    return {n: p for n, p in encoder.named_parameters() if not n.startswith("fc")}


def tap_feats(encoder, x, taps, stats):
    d = resnet_depth_features(encoder, x)
    parts = []
    for t in taps:
        f = d[t]
        if stats is not None:
            mu, sd = stats[t]
            f = (f - mu) / sd
        parts.append(f)
    return torch.cat(parts, 1)


@torch.no_grad()
def extract_view(encoder, view, taps, stats, device, batch, workers):
    dl = DataLoader(view, batch_size=batch, shuffle=False, num_workers=workers)
    fs, ys = [], []
    for xb, yb in dl:
        fs.append(tap_feats(encoder, xb.to(device), taps, stats).cpu())
        ys.append(yb)
    return torch.cat(fs), torch.cat(ys).long()


def joint_mas_importance(encoder, adapter, train_views, tasks, taps, stats,
                         device, samples, batch, workers, seed):
    """MAS importance |d||logits||^2/dw| accumulated over prev-task samples,
    jointly for encoder AND adapter (one backward pass computes both)."""
    ep = enc_named_params(encoder)
    imp_e = {n: torch.zeros_like(p) for n, p in ep.items()}
    imp_a = {k: torch.zeros_like(v) for k, v in _params(adapter).items()}
    for p in ep.values():
        p.requires_grad_(True)
    nb = 0
    for k in tasks:
        dl = DataLoader(sub_view(train_views[k], samples, seed + k),
                        batch_size=batch, shuffle=False, num_workers=workers)
        for xb, _ in dl:
            feats = tap_feats(encoder, xb.to(device), taps, stats)
            _, _, logits = adapter(feats, k)
            loss = logits.pow(2).sum(1).mean()
            encoder.zero_grad(set_to_none=False)
            adapter.zero_grad(set_to_none=False)
            loss.backward()
            for n, p in ep.items():
                if p.grad is not None:
                    imp_e[n] += p.grad.abs()
            for kk, v in _params(adapter).items():
                if v.grad is not None:
                    imp_a[kk] += v.grad.abs()
            nb += 1
    encoder.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    imp_e = {n: v / nb for n, v in imp_e.items()}
    imp_a = {k: v / nb for k, v in imp_a.items()}
    # global single-max normalization (preserves cross-tensor ratios), applied
    # separately per component — mask needs encoder RANKS, penalty needs scale
    gmax_a = max(float(v.max()) for v in imp_a.values()) + 1e-12
    imp_a = {k: v / gmax_a for k, v in imp_a.items()}
    gmax_e = max(float(v.max()) for v in imp_e.values()) + 1e-12
    imp_e = {n: v / gmax_e for n, v in imp_e.items()}
    return imp_e, imp_a


def bottom_q_masks(imp_e, q):
    """1.0 where the param is in the LEAST-important q fraction (trainable)."""
    if q <= 0:
        return {n: torch.zeros_like(v) for n, v in imp_e.items()}, 0.0
    flat = torch.cat([v.flatten() for v in imp_e.values()])
    kth = max(1, int(q * flat.numel()))
    thresh = flat.kthvalue(kth).values
    masks = {n: (v <= thresh).float() for n, v in imp_e.items()}
    frac = float(sum(m.sum() for m in masks.values()) / flat.numel())
    return masks, frac


@torch.no_grad()
def eval_head(adapter, feats, ys, task_id, batch=2048):
    correct = 0
    for i in range(0, len(feats), batch):
        _, _, logits = adapter(feats[i:i + batch], task_id)
        correct += int((logits.argmax(1) == ys[i:i + batch]).sum())
    return correct / len(feats)


def run_stream(cfg, device):
    set_seed(cfg["seed"])
    train_views, test_views, counts = build_stream(cfg["dataset"], cfg["n_tasks"])
    T = len(train_views)
    taps = cfg["depth"].split("+")

    encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    encoder.eval().to(device)  # eval() for the WHOLE stream: BN stats frozen
    for p in encoder.parameters():
        p.requires_grad_(False)

    # fixed per-tap z-score stats from task-0 train features (initial encoder)
    raw_stats = {}
    with torch.no_grad():
        dl = DataLoader(train_views[0], batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=cfg["workers"])
        buf = {t: [] for t in taps}
        for xb, _ in dl:
            d = resnet_depth_features(encoder, xb.to(device))
            for t in taps:
                buf[t].append(d[t].cpu())
        for t in taps:
            f = torch.cat(buf[t])
            raw_stats[t] = (f.mean(0).to(device), (f.std(0) + 1e-6).to(device))
    stats = raw_stats

    d_in = sum(TAP_DIMS[t] for t in taps)
    adapter = Adapter(d_in=d_in, d_hidden=cfg["d_hidden"]).to(device)

    A = np.zeros((T, T))
    run_stats = {"trainable_frac": [], "enc_delta": [], "task_sec": []}
    test_cache = {}  # task -> (feats, ys) under the CURRENT encoder

    for t in range(T):
        t0 = time.time()
        adapter.add_head(t, counts[t])
        adapter.heads[str(t)].to(device)

        if t == 0:
            # encoder frozen -> features constant: extract once, train fast
            feats, ys = extract_view(encoder, train_views[0], taps, stats,
                                     device, cfg["batch_size"], cfg["workers"])
            dl = DataLoader(TensorDataset(feats, ys),
                            batch_size=cfg["batch_size"], shuffle=True)
            opt = torch.optim.AdamW(
                [p for p in adapter.parameters() if p.requires_grad],
                lr=cfg["lr"], weight_decay=0.0)
            for _ in range(cfg["epochs"]):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    _, _, logits = adapter(xb, 0)
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            S_enc, S_ad = joint_mas_importance(
                encoder, adapter, train_views, list(range(t)), taps, stats,
                device, cfg["phi_samples"], cfg["batch_size"], cfg["workers"],
                cfg["seed"])
            enc_masks, frac = bottom_q_masks(S_enc, cfg["enc_q"])
            run_stats["trainable_frac"].append(round(frac, 4))
            theta_a = _snapshot(adapter)
            ep = enc_named_params(encoder)
            theta_e = {n: p.detach().clone() for n, p in ep.items()}
            teacher = copy.deepcopy(encoder).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            for p in ep.values():
                p.requires_grad_(cfg["enc_q"] > 0)

            groups = [{"params": [p for p in adapter.parameters()
                                  if p.requires_grad], "lr": cfg["lr"]}]
            if cfg["enc_q"] > 0:
                groups.append({"params": list(ep.values()), "lr": cfg["enc_lr"]})
            opt = torch.optim.AdamW(groups, weight_decay=0.0)

            dl = DataLoader(train_views[t], batch_size=cfg["batch_size"],
                            shuffle=True, num_workers=cfg["workers"])
            for _ in range(cfg["epochs"]):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    feats = tap_feats(encoder, xb, taps, stats)
                    _, _, logits = adapter(feats, t)
                    loss = F.cross_entropy(logits, yb)
                    if cfg["alpha"] > 0:
                        loss = loss + cfg["alpha"] * sum(
                            (S_ad[k] * (v - theta_a[k]) ** 2).sum()
                            for k, v in _params(adapter).items())
                    if cfg["enc_q"] > 0 and cfg["alpha_enc"] > 0:
                        loss = loss + cfg["alpha_enc"] * sum(
                            (S_enc[n] * enc_masks[n]
                             * (p - theta_e[n]) ** 2).sum()
                            for n, p in ep.items())
                    if cfg["enc_q"] > 0 and cfg["sim_lambda"] > 0:
                        with torch.no_grad():
                            tfeat = tap_feats(teacher, xb, taps, stats)
                        loss = loss + cfg["sim_lambda"] * F.mse_loss(feats, tfeat)
                    opt.zero_grad()
                    loss.backward()
                    if cfg["enc_q"] > 0:
                        for n, p in ep.items():
                            if p.grad is not None:
                                p.grad *= enc_masks[n]
                    opt.step()
            with torch.no_grad():
                dsq = sum(((p - theta_e[n]) ** 2).sum() for n, p in ep.items())
                den = torch.sqrt(sum((v ** 2).sum() for v in theta_e.values()))
                run_stats["enc_delta"].append(float(torch.sqrt(dsq) / den))
            for p in ep.values():
                p.requires_grad_(False)
            del teacher

        adapter.freeze_head(t)
        # encoder changed -> re-extract ALL seen test sets under current encoder
        for k in range(t + 1):
            test_cache[k] = extract_view(encoder, test_views[k], taps, stats,
                                         device, cfg["batch_size"],
                                         cfg["workers"])
        for k in range(t + 1):
            fk, yk = test_cache[k]
            A[t, k] = eval_head(adapter, fk.to(device), yk.to(device), k)
        run_stats["task_sec"].append(round(time.time() - t0, 1))
        print(f"  task {t}: acc_so_far={A[t, :t + 1].mean() * 100:.2f} "
              f"({run_stats['task_sec'][-1]}s)", flush=True)

    return A, run_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar10", "cifar100", "fivedata"])
    p.add_argument("--depths", nargs="*", default=["layer3+layer4"])
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--enc-q", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="adapter MAS penalty (tuned value for reg:mas)")
    p.add_argument("--alpha-enc", type=float, default=None,
                   help="encoder-side MAS penalty; default = --alpha")
    p.add_argument("--sim-lambda", type=float, default=1.0)
    p.add_argument("--enc-lr", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--phi-samples", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tag-suffix", default="")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "fivedata") else 10)

    for depth in a.depths:
        for seed in a.seeds:
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            cfg.update({
                "dataset": a.dataset, "depth": depth, "n_tasks": n_tasks,
                "seed": seed, "method": "enc:spmas", "enc_q": a.enc_q,
                "alpha": a.alpha,
                "alpha_enc": a.alpha if a.alpha_enc is None else a.alpha_enc,
                "sim_lambda": a.sim_lambda, "enc_lr": a.enc_lr, "lr": a.lr,
                "epochs": a.epochs, "batch_size": a.batch_size,
                "phi_samples": a.phi_samples, "workers": a.workers,
                "backbone": f"r50enc_{depth}",
            })
            tag = (f"{a.dataset}_r50enc_{depth}_t{n_tasks}_spmas"
                   f"_q{int(round(a.enc_q * 100))}{a.tag_suffix}")
            print(f"== {tag} seed {seed} ==", flush=True)
            t0 = time.time()
            A, run_stats = run_stream(cfg, device)
            m = compute_metrics(A)
            m["bwt"] = bwt(A)
            payload = {"config": cfg, "acc_matrix": A.tolist(), "metrics": m,
                       "stats": run_stats, "git": git_hash(),
                       "runtime_sec": round(time.time() - t0, 1)}
            save_result(RESULTS_DIR, tag, seed, payload)
            print(f"== {tag} seed {seed}: AvgAcc={m['avg_acc'] * 100:.2f} "
                  f"Forget={m['forgetting'] * 100:.2f} "
                  f"({payload['runtime_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
