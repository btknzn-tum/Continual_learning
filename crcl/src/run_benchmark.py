"""Full benchmark grid: {encoder x depth} x {dataset} x {method} x {seed} -> CSV.

Reads cached (dataset, backbone, depth) feature files; each CL run is a small MLP,
seconds-to-minutes on CPU. Writes one JSON per run + appends an aggregate CSV row
per (depth, method).

Depth spec supports multi-tap concatenation: "layer3+layer4".
Per-method tuned hyperparameters can be supplied via --tuned-file (JSON produced
by run_sweep.py: {method: {param: value}}).
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np

from common import DEFAULT_CONFIG, RESULTS_DIR, CACHE_DIR
from metrics import compute_metrics, save_result
from train_cl import run_joint, run_sequence

SEEDS_FAST = [42, 123, 456]
SEEDS_FULL = [42, 123, 456, 789, 1337]

METHODS = {
    "naive": {"method": "naive"},
    "l2sp": {"method": "reg:l2"},
    "ewc": {"method": "reg:ewc"},
    "si": {"method": "si"},
    "lwf": {"method": "lwf"},
    "er": {"method": "er"},
    "mas": {"method": "reg:mas"},
    "maskmas": {"method": "mask:mas"},
    "maskewc": {"method": "mask:ewc"},
    "sfxphi": {"method": "reg:sfxphi"},
    "ours_framework": {"method": "ours"},
    "joint": {"method": "joint"},
}
# Ordered by reporting priority: our method first, then the most informative
# contrasts (lower bound, classic rival, upper bound), then remaining baselines.
DEFAULT_METHODS = ["mas", "naive", "ewc", "joint", "si", "lwf", "er"]


def bwt(A):
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, t] - A[t, t] for t in range(T - 1)]))


def run_cell(dataset, backbone_full, method_label, seed, n_tasks, tuned,
             d_hidden=None, arch="mlp", mask_q=None, eval_classil=False):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(METHODS[method_label])
    cfg.update(tuned.get(method_label, {}))
    cfg.update({"dataset": dataset, "backbone": backbone_full,
                "seed": seed, "n_tasks": n_tasks, "arch": arch})
    if d_hidden:
        cfg["d_hidden"] = d_hidden
    if mask_q is not None:
        cfg["mask_q"] = mask_q
    if eval_classil:
        cfg["eval_classil"] = True
    # n_tasks in the tag: 10- and 20-task CIFAR-100 runs must not share a dir;
    # non-default d_hidden / lora arch get their own suffix (capacity controls)
    tag = f"{dataset}_{backbone_full}_t{n_tasks}_{method_label}"
    if d_hidden and d_hidden != DEFAULT_CONFIG["d_hidden"]:
        tag += f"_h{d_hidden}"
    if arch == "lora":
        tag += f"_lora{cfg['lora_rank']}"
    if method_label.startswith("mask") and mask_q is not None:
        tag += f"_q{int(round(mask_q * 100))}"
    if eval_classil:
        tag += "_cil"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="encoder name, e.g. resnet50")
    p.add_argument("--depths", nargs="*", required=True,
                   help='depth specs, e.g. layer4 "layer3+layer4"; use "-" for legacy/pixels')
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--d-hidden", type=int, default=None,
                   help="override adapter width (capacity-control runs)")
    p.add_argument("--arch", default="mlp", choices=["mlp", "lora"])
    p.add_argument("--mask-q", type=float, default=None,
                   help="trainable fraction for mask:* methods (e.g. 0.05)")
    p.add_argument("--eval-classil", action="store_true",
                   help="task-agnostic merged-head evaluation (_cil tag)")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="explicit seed list (overrides --full/fast)")
    p.add_argument("--tuned-file", default=None)
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--full", action="store_true", help="5 seeds instead of 3")
    a = p.parse_args()
    seeds = a.seeds if a.seeds else (SEEDS_FULL if a.full else SEEDS_FAST)
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "mnist", "fivedata") else 10)
    tuned = {}
    if a.tuned_file and os.path.exists(a.tuned_file):
        tuned = json.load(open(a.tuned_file))
        print(f"tuned params: {tuned}", flush=True)
    for m in tuned.values():
        m.setdefault("alpha", a.alpha)
    base_tuned = {m: tuned.get(m, {"alpha": a.alpha}) for m in a.methods}

    rows = []
    for depth in a.depths:
        backbone_full = a.backbone if depth == "-" else f"{a.backbone}_{depth}"
        # existence check on the first tap of the spec
        if depth != "-":
            # probe EVERY tap in the spec (a missing later tap would crash mid-grid)
            probe_ds = "cifar10" if a.dataset == "fivedata" else a.dataset
            missing = [t for t in depth.split("+") if not os.path.exists(
                os.path.join(CACHE_DIR, f"{probe_ds}_{a.backbone}_{t}_train.pt"))]
            if missing:
                print(f"MISSING cache taps {missing} for {depth} — skip", flush=True)
                continue
        for method in a.methods:
            metrics = []
            for seed in seeds:
                m = run_cell(a.dataset, backbone_full, method, seed, n_tasks,
                             base_tuned, d_hidden=a.d_hidden, arch=a.arch,
                             mask_q=a.mask_q, eval_classil=a.eval_classil)
                metrics.append(m)
            agg = {k: (float(np.mean([mm[k] for mm in metrics])),
                       float(np.std([mm[k] for mm in metrics])))
                   for k in metrics[0]}
            eff = base_tuned.get(method, {})
            row = {"dataset": a.dataset, "encoder": a.backbone, "depth": depth,
                   "method": method, "seeds": len(seeds), "n_tasks": n_tasks,
                   "alpha": eff.get("alpha", ""),
                   "lwf_lambda": eff.get("lwf_lambda", ""),
                   "er_per_class": DEFAULT_CONFIG["er_per_class"],
                   "git": __import__("train_cl").git_hash()}
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
        import fcntl
        fields = list(rows[0].keys())
        write_header = not os.path.exists(out)
        if not write_header:
            with open(out) as f:
                existing = f.readline().strip().split(",")
            if existing != fields:  # schema changed → don't misalign; version the file
                out = os.path.join(RESULTS_DIR,
                                   f"bench_{a.dataset}_{a.backbone}_{fields.__len__()}c.csv")
                write_header = not os.path.exists(out)
        # exclusive lock: multiple parallel benchmark processes append to the
        # same CSV — without it rows can interleave mid-line
        with open(out, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerows(rows)
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"\nappended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
