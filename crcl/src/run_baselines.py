"""Classic-baseline head-to-head on the SAME adapter/trainer.

ewc / mas = pure classic regularization: importance-weighted quadratic penalty
only (no reserve loss, no delta-norm constraint), same alpha as ours.
"""
import argparse
import json
import os

import numpy as np

from common import RESULTS_DIR
from run_experiment import SEEDS, run_one

BASELINES = {
    "ewc": {"importance": "ewc", "beta_res": 0.0, "gamma": 0.0},
    "mas": {"importance": "mas", "beta_res": 0.0, "gamma": 0.0},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--backbone", default="pixels")
    p.add_argument("--alpha", type=float, default=10.0)
    a = p.parse_args()
    prefix = f"{a.dataset}_" if a.dataset != "cifar10" else ""

    summary = {}
    for name, ov in BASELINES.items():
        rows = []
        for seed in SEEDS:
            r = run_one("ours", seed, {
                **ov, "dataset": a.dataset, "backbone": a.backbone,
                "alpha": a.alpha, "tag": f"{prefix}{name}",
            })
            rows.append(r["metrics"])
            print(f"{name} seed{seed}: "
                  f"AvgAcc={r['metrics']['avg_acc']*100:.2f} "
                  f"Forget={r['metrics']['forgetting']*100:.2f} "
                  f"Plast={r['metrics']['plasticity']*100:.2f} "
                  f"({r['runtime_sec']}s)", flush=True)
        summary[name] = {
            k: {"mean": float(np.mean([m[k] for m in rows])),
                "std": float(np.std([m[k] for m in rows]))}
            for k in rows[0]
        }

    print("\n=== BASELINES (mean±std over seeds, %) ===")
    for name, m in summary.items():
        print(f"{name:>6} AvgAcc {m['avg_acc']['mean']*100:.2f}±{m['avg_acc']['std']*100:.2f}  "
              f"Forget {m['forgetting']['mean']*100:.2f}±{m['forgetting']['std']*100:.2f}  "
              f"Plast {m['plasticity']['mean']*100:.2f}±{m['plasticity']['std']*100:.2f}")

    out = os.path.join(RESULTS_DIR, f"{prefix}baselines_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
