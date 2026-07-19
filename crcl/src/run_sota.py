"""SOTA frozen-feature CL baselines on the cached features, same protocol.

ranpac — RanPAC (McDonnell et al., NeurIPS 2023), phase-2 variant on cached
         features: random projection W (d x M, M=10000), h = ReLU(xW), streaming
         Gram G += h^T h and class matrix C += h^T Y, ridge classifier
         beta = (G + lam I)^-1 C. lam picked per task on a 5% TRAIN holdout
         (test never touched). Task-IL eval: logits restricted to task classes.
gpm_ncm / "ncm" — SimpleCIL-style nearest-class-mean prototype classifier
         (cosine similarity), the weakest honest prototype baseline.

Runs on GPU when available (the Gram/solve steps love it), CPU otherwise.
Writes the same JSON layout as run_benchmark (results/{tag}/seed{k}.json).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DEFAULT_CONFIG, RESULTS_DIR, set_seed
from metrics import compute_metrics, save_result
from train_cl import build_task_stream, git_hash

SEEDS_FULL = [42, 123, 456, 789, 1337]
LAM_GRID = [1e-3, 1e-1, 1e1, 1e3, 1e5]
RP_DIM = 10000


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def task_offsets(counts):
    off, out = 0, []
    for c in counts:
        out.append(off)
        off += c
    return out, off


def eval_matrix_entry(logits_all, off, cnt, y_local):
    logits = logits_all[:, off:off + cnt]
    return float((logits.argmax(1).cpu() == y_local).float().mean())


def run_ranpac(cfg, seed):
    set_seed(seed)
    cfg = {**cfg, "seed": seed}
    train_sets, test_sets, counts = build_task_stream(cfg)
    T = len(train_sets)
    offs, n_cls = task_offsets(counts)
    dev = device()
    d = train_sets[0][0].shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(d, RP_DIM, generator=g) / (d ** 0.5)).to(dev)
    G = torch.zeros(RP_DIM, RP_DIM, device=dev)
    C = torch.zeros(RP_DIM, n_cls, device=dev)
    ho_h, ho_y, ho_t = [], [], []   # 5% train holdout for lambda selection
    A = np.zeros((T, T))

    def project(x):
        return F.relu(x.to(dev) @ W)

    for t in range(T):
        x, y = train_sets[t]
        gsp = torch.Generator().manual_seed(seed * 1000 + t)
        perm = torch.randperm(len(x), generator=gsp)
        n_ho = max(1, int(0.05 * len(x)))
        ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]
        for i in range(0, len(tr_idx), 2048):
            idx = tr_idx[i:i + 2048]
            h = project(x[idx])
            G += h.T @ h
            Y = F.one_hot(y[idx] + offs[t], n_cls).float().to(dev)
            C += h.T @ Y
        ho_h.append(project(x[ho_idx]))
        ho_y.append(y[ho_idx])
        ho_t.append(t)

        # pick lambda on the accumulated train-holdout, then evaluate
        best_lam, best_acc, best_beta = None, -1.0, None
        I = torch.eye(RP_DIM, device=dev)
        for lam in LAM_GRID:
            try:
                beta = torch.linalg.solve(G + lam * I, C)
            except RuntimeError:
                continue
            correct = tot = 0
            for hh, yy, tt in zip(ho_h, ho_y, ho_t):
                lo = (hh @ beta)[:, offs[tt]:offs[tt] + counts[tt]]
                correct += int((lo.argmax(1).cpu() == yy).sum())
                tot += len(yy)
            acc = correct / max(1, tot)
            if acc > best_acc:
                best_lam, best_acc, best_beta = lam, acc, beta
        for k in range(t + 1):
            xk, yk = test_sets[k]
            logits = torch.cat([project(xk[i:i + 4096]) @ best_beta
                                for i in range(0, len(xk), 4096)])
            A[t, k] = eval_matrix_entry(logits, offs[k], counts[k], yk)
    return A


def run_ncm(cfg, seed):
    set_seed(seed)
    cfg = {**cfg, "seed": seed}
    train_sets, test_sets, counts = build_task_stream(cfg)
    T = len(train_sets)
    offs, n_cls = task_offsets(counts)
    d = train_sets[0][0].shape[1]
    protos = torch.zeros(n_cls, d)
    A = np.zeros((T, T))
    for t in range(T):
        x, y = train_sets[t]
        for c in range(counts[t]):
            protos[offs[t] + c] = x[y == c].mean(0)
        P = F.normalize(protos, dim=1)
        for k in range(t + 1):
            xk, yk = test_sets[k]
            sims = F.normalize(xk, dim=1) @ P.T
            A[t, k] = eval_matrix_entry(sims, offs[k], counts[k], yk)
    return A


def bwt(A):
    T = A.shape[0]
    return 0.0 if T < 2 else float(np.mean([A[T-1, t] - A[t, t] for t in range(T-1)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="full spec, e.g. resnet50_layer4")
    p.add_argument("--methods", nargs="*", default=["ranpac", "ncm"])
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS_FULL)
    a = p.parse_args()
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "mnist", "fivedata") else 10)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update({"dataset": a.dataset, "backbone": a.backbone, "n_tasks": n_tasks})

    for method in a.methods:
        fn = {"ranpac": run_ranpac, "ncm": run_ncm}[method]
        rows = []
        for seed in a.seeds:
            t0 = time.time()
            A = fn(cfg, seed)
            m = compute_metrics(A)
            m["bwt"] = bwt(A)
            payload = {"config": {**cfg, "seed": seed, "method": method},
                       "acc_matrix": A.tolist(), "metrics": m,
                       "git": git_hash(),
                       "runtime_sec": round(time.time() - t0, 1)}
            tag = f"{a.dataset}_{a.backbone}_t{n_tasks}_{method}"
            save_result(RESULTS_DIR, tag, seed, payload)
            rows.append(m)
            print(f"{method} seed{seed}: AvgAcc={m['avg_acc']*100:.2f} "
                  f"Forget={m['forgetting']*100:.2f} ({payload['runtime_sec']}s)",
                  flush=True)
        aa = [m["avg_acc"] for m in rows]; fg = [m["forgetting"] for m in rows]
        print(f"== {a.backbone} {method}: AvgAcc {np.mean(aa)*100:.2f}±{np.std(aa)*100:.2f} "
              f"Forget {np.mean(fg)*100:.2f}±{np.std(fg)*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
