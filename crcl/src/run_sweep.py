"""Hyperparameter sweep -> results/tuned_{dataset}_{backbone}.json.

Selection is done on a VALIDATION holdout carved from train (val_frac=0.1) and on
a SWEEP SEED (7) disjoint from the reporting seeds {42,123,456,789,1337} — the test
set is never used for tuning. Sweeps: alpha for mas/ewc/si; lambda for lwf. ER stays
at its fixed default (buffer size is a protocol choice, not a tunable). Selection:
lowest forgetting among configs within 0.5pp of the best AvgAcc for that method.
"""
import argparse
import copy
import json
import os

import numpy as np

from common import DEFAULT_CONFIG, RESULTS_DIR
from metrics import compute_metrics
from train_cl import run_sequence

GRIDS = {
    "mas": ("alpha", [1.0, 10.0, 100.0, 1000.0], {"method": "reg:mas"}),
    "ewc": ("alpha", [1.0, 10.0, 100.0, 1000.0], {"method": "reg:ewc"}),
    "si": ("alpha", [0.1, 1.0, 10.0, 100.0], {"method": "si"}),
    "lwf": ("lwf_lambda", [0.5, 1.0, 2.0], {"method": "lwf"}),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="full spec, e.g. resnet50_layer4")
    p.add_argument("--n-tasks", type=int, default=None)
    a = p.parse_args()
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "mnist") else 10)

    tuned = {}
    for method, (param, values, overrides) in GRIDS.items():
        results = []
        for v in values:
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            cfg.update(overrides)
            cfg.update({"dataset": a.dataset, "backbone": a.backbone,
                        "seed": 7, "n_tasks": n_tasks, "val_frac": 0.1, param: v})
            _, A, _ = run_sequence(cfg)
            m = compute_metrics(A)
            results.append((v, m))
            print(f"{method} {param}={v}: AvgAcc={m['avg_acc']*100:.2f} "
                  f"Forget={m['forgetting']*100:.2f}", flush=True)
        best_acc = max(m["avg_acc"] for _, m in results)
        eligible = [(v, m) for v, m in results if m["avg_acc"] >= best_acc - 0.005]
        best_v = min(eligible, key=lambda vm: vm[1]["forgetting"])[0]
        tuned[method] = {param: best_v}
        print(f"-> {method}: {param}={best_v}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"tuned_{a.dataset}_{a.backbone}.json")
    with open(out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
