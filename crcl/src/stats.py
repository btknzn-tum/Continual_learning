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


def seed_values(tag, metric):
    vals = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, tag, "seed*.json"))):
        d = json.load(open(path))
        vals.append(d["metrics"][metric])
    return np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="full spec, e.g. resnet50_layer4")
    p.add_argument("--metric", default="forgetting")
    a = p.parse_args()

    from scipy import stats as sps
    ref_tag = f"{a.dataset}_{a.backbone}_mas"
    ref = seed_values(ref_tag, a.metric)
    if len(ref) == 0:
        raise SystemExit(f"no results for {ref_tag}")
    n_tests = len(COMPETITORS)
    print(f"MAS {a.metric}: {ref.mean()*100:.2f}±{ref.std()*100:.2f} (n={len(ref)})")
    print(f"{'vs':>8} {'mean±std':>16} {'t':>8} {'p':>10} {'p*Bonf':>10} sig")
    for comp in COMPETITORS:
        vals = seed_values(f"{a.dataset}_{a.backbone}_{comp}", a.metric)
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
