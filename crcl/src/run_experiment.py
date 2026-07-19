"""Run ours + naive + joint across seeds on cached features; write JSON + summary."""
import argparse
import copy
import json
import os
import time

import numpy as np

from common import DEFAULT_CONFIG, RESULTS_DIR
from metrics import compute_metrics, save_result
from train_cl import run_joint, run_sequence

SEEDS = [42, 123, 456]


def run_one(method: str, seed: int, overrides: dict):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    cfg["method"] = method
    cfg["seed"] = seed
    t0 = time.time()
    if method == "joint":
        per_task = run_joint(cfg)
        m = {"avg_acc": float(np.mean(per_task)), "forgetting": 0.0,
             "plasticity": float(np.mean(per_task))}
        payload = {"config": cfg, "per_task_acc": per_task, "metrics": m}
    else:
        payload, A, stats = run_sequence(cfg)
        m = compute_metrics(A)
        payload["metrics"] = m
        payload["stats"] = stats
    payload["runtime_sec"] = round(time.time() - t0, 1)
    save_result(RESULTS_DIR, method if not overrides.get("tag") else overrides["tag"],
                seed, payload)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-alpha", action="store_true",
                   help="quick alpha sweep for ours, seed 42 only")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--backbone", default=None)
    a = p.parse_args()

    base = {}
    if a.dataset is not None:
        base["dataset"] = a.dataset
    if a.backbone is not None:
        base["backbone"] = a.backbone
    prefix = f"{a.dataset}_" if a.dataset not in (None, "cifar10") else ""

    if a.sweep_alpha:
        print(f"{'alpha':>8} {'AvgAcc':>8} {'Forget':>8} {'Plast':>8}")
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            r = run_one("ours", 42, {**base, "alpha": alpha,
                                     "tag": f"{prefix}sweep_alpha{alpha}"})
            m = r["metrics"]
            print(f"{alpha:>8} {m['avg_acc']*100:8.2f} {m['forgetting']*100:8.2f} "
                  f"{m['plasticity']*100:8.2f}", flush=True)
        return

    overrides = dict(base)
    if a.alpha is not None:
        overrides["alpha"] = a.alpha

    summary = {}
    for method in ["naive", "ours", "mas_adapter", "joint"]:
        rows = []
        for seed in SEEDS:
            r = run_one(method, seed, {**overrides, "tag": f"{prefix}{method}"})
            rows.append(r["metrics"])
            print(f"{method} seed{seed}: "
                  f"AvgAcc={r['metrics']['avg_acc']*100:.2f} "
                  f"Forget={r['metrics']['forgetting']*100:.2f} "
                  f"Plast={r['metrics']['plasticity']*100:.2f} "
                  f"({r['runtime_sec']}s)", flush=True)
        summary[method] = {
            k: {"mean": float(np.mean([m[k] for m in rows])),
                "std": float(np.std([m[k] for m in rows]))}
            for k in rows[0]
        }

    print("\n=== SUMMARY (mean±std over seeds, %) ===")
    print(f"{'method':>8} {'AvgAcc':>16} {'Forgetting':>16} {'Plasticity':>16}")
    for method, m in summary.items():
        print(f"{method:>8} "
              f"{m['avg_acc']['mean']*100:8.2f}±{m['avg_acc']['std']*100:<7.2f} "
              f"{m['forgetting']['mean']*100:8.2f}±{m['forgetting']['std']*100:<7.2f} "
              f"{m['plasticity']['mean']*100:8.2f}±{m['plasticity']['std']*100:<7.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{prefix}summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
