"""Full benchmark grid: {encoder x depth} x {dataset} x {method} x {seed} -> CSV.

Reads cached (dataset, backbone, depth) feature files; each CL run is a small MLP,
seconds-to-minutes on CPU. Writes one JSON per run + an aggregate CSV.
"""
import argparse
import copy
import csv
import os
import time

import numpy as np

from common import DEFAULT_CONFIG, RESULTS_DIR, CACHE_DIR
from metrics import compute_metrics, save_result
from train_cl import run_joint, run_sequence

SEEDS_FAST = [42, 123, 456]
SEEDS_FULL = [42, 123, 456, 789, 1337]

# method label -> config overrides. Pure regularizers via "reg:<signal>".
METHODS = {
    "naive": {"method": "naive"},
    "l2sp": {"method": "reg:l2"},
    "ewc": {"method": "reg:ewc"},
    "mas": {"method": "reg:mas"},
    "sfxphi": {"method": "reg:sfxphi"},
    "sf": {"method": "reg:sf"},
    "phi": {"method": "reg:phi"},
    "wanda": {"method": "reg:wanda"},
    "taylor": {"method": "reg:taylor"},
    "ours_framework": {"method": "ours"},        # reserve + SF×phi + norm cap
    "mas_framework": {"method": "reg:mas", "gamma": 100.0, "beta_res": 0.0},  # cap ablation
    "joint": {"method": "joint"},
}


def cache_exists(dataset, backbone, depth):
    return os.path.exists(os.path.join(
        CACHE_DIR, f"{dataset}_{backbone}_{depth}_{split_probe(dataset)}"))


def split_probe(dataset):
    return "train.pt"  # helper for existence check filename tail


def run_cell(dataset, backbone, depth, method_label, seed, alpha, n_tasks):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(METHODS[method_label])
    cfg.update({"dataset": dataset, "backbone": f"{backbone}_{depth}",
                "seed": seed, "alpha": alpha, "n_tasks": n_tasks})
    tag = f"{dataset}_{backbone}_{depth}_{method_label}"
    t0 = time.time()
    if cfg["method"] == "joint":
        per_task = run_joint(cfg)
        m = {"avg_acc": float(np.mean(per_task)), "forgetting": 0.0,
             "plasticity": float(np.mean(per_task)), "bwt": 0.0}
        payload = {"config": cfg, "per_task_acc": per_task, "metrics": m}
    else:
        payload, A, stats = run_sequence(cfg)
        m = compute_metrics(A)
        m["bwt"] = bwt(A)
        payload["metrics"] = m
        payload["stats"] = stats
    payload["runtime_sec"] = round(time.time() - t0, 1)
    save_result(RESULTS_DIR, tag, seed, payload)
    return m


def bwt(A):
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, t] - A[t, t] for t in range(T - 1)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--depths", nargs="*", required=True)
    p.add_argument("--methods", nargs="*", default=list(METHODS.keys()))
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--full", action="store_true", help="5 seeds instead of 3")
    a = p.parse_args()
    seeds = SEEDS_FULL if a.full else SEEDS_FAST
    n_tasks = a.n_tasks or (5 if a.dataset == "cifar10" else 10)

    rows = []
    for depth in a.depths:
        train_cache = os.path.join(
            CACHE_DIR, f"{a.dataset}_{a.backbone}_{depth}_train.pt")
        if not os.path.exists(train_cache):
            print(f"MISSING cache {train_cache} — skip depth {depth}", flush=True)
            continue
        for method in a.methods:
            metrics = []
            for seed in seeds:
                m = run_cell(a.dataset, a.backbone, depth, method, seed,
                             a.alpha, n_tasks)
                metrics.append(m)
            agg = {k: (float(np.mean([mm[k] for mm in metrics])),
                       float(np.std([mm[k] for mm in metrics])))
                   for k in metrics[0]}
            row = {"dataset": a.dataset, "encoder": a.backbone, "depth": depth,
                   "method": method, "seeds": len(seeds)}
            for k, (mu, sd) in agg.items():
                row[f"{k}_mean"] = round(mu * 100, 3)
                row[f"{k}_std"] = round(sd * 100, 3)
            rows.append(row)
            print(f"{a.backbone}/{depth} {method:16s} "
                  f"AvgAcc={row['avg_acc_mean']:.2f}±{row['avg_acc_std']:.2f} "
                  f"Forget={row['forgetting_mean']:.2f}±{row['forgetting_std']:.2f}",
                  flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"bench_{a.dataset}_{a.backbone}.csv")
    if rows:
        write_header = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
