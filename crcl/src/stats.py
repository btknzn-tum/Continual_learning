"""Significance tables: Welch t-tests (MAS vs each competitor) + Bonferroni.

Reads per-run JSONs written by run_benchmark (results/{tag}/seed{k}.json).
Usage: python stats.py --dataset cifar100 --backbone resnet50_layer4
"""
import argparse
import glob
import json
import os

import numpy as np

from common import RESULTS_DIR

COMPETITORS = ["ewc", "si", "lwf", "er", "naive"]
REPORT_SEEDS = [42, 123, 456, 789, 1337]
# hyperparameters that define a run's "config regime"; all seeds of a tag must agree
HP_KEYS = ["alpha", "lwf_lambda", "lwf_T", "si_xi", "er_per_class", "er_weight",
           "epochs", "lr", "n_tasks", "method"]


def seed_values(tag, metric, seeds=None):
    """Collect metric over the explicit REPORT_SEEDS only, asserting every loaded
    run shares the same hyperparameter regime (guards against stale/mixed JSONs)."""
    seeds = seeds or REPORT_SEEDS
    vals, regimes = [], set()
    for s in seeds:
        path = os.path.join(RESULTS_DIR, tag, f"seed{s}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        cfg = d.get("config", {})
        regimes.add(tuple(cfg.get(k) for k in HP_KEYS))
        vals.append(d["metrics"][metric])
    if len(regimes) > 1:
        raise SystemExit(f"[stats] {tag}: mixed hyperparameter regimes across seeds "
                         f"{regimes} — rerun the tag cleanly before testing.")
    return np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="full spec, e.g. resnet50_layer4")
    p.add_argument("--metric", default="forgetting")
    p.add_argument("--n-tasks", type=int, default=None)
    a = p.parse_args()

    from scipy import stats as sps
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "mnist", "fivedata") else 10)
    ref_tag = f"{a.dataset}_{a.backbone}_t{n_tasks}_mas"
    ref = seed_values(ref_tag, a.metric)
    if len(ref) == 0:
        raise SystemExit(f"no results for {ref_tag}")
    n_tests = len(COMPETITORS)
    print(f"MAS {a.metric}: {ref.mean()*100:.2f}±{ref.std()*100:.2f} (n={len(ref)})")
    print(f"{'vs':>8} {'mean±std':>16} {'t':>8} {'p':>10} {'p*Bonf':>10} sig")
    for comp in COMPETITORS:
        vals = seed_values(f"{a.dataset}_{a.backbone}_t{n_tasks}_{comp}", a.metric)
        if len(vals) == 0:
            print(f"{comp:>8} (missing)")
            continue
        t, pval = sps.ttest_ind(ref, vals, equal_var=False)
        p_corr = min(1.0, pval * n_tests)
        print(f"{comp:>8} {vals.mean()*100:8.2f}±{vals.std()*100:<7.2f} "
              f"{t:8.3f} {pval:10.4f} {p_corr:10.4f} "
              f"{'YES' if p_corr < 0.05 else 'no'}")


if __name__ == "__main__":
    main()
