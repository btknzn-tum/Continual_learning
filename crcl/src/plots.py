"""Figures from results/: method comparison bars + ours accuracy-matrix heatmap."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_DIR

SEEDS = [42, 123, 456]


def load_summary(prefix):
    path = os.path.join(RESULTS_DIR, f"{prefix}summary.json")
    base_path = os.path.join(RESULTS_DIR, f"{prefix}baselines_summary.json")
    s = json.load(open(path)) if os.path.exists(path) else {}
    if os.path.exists(base_path):
        s.update(json.load(open(base_path)))
    return s


def bars(ax, summary, metric, title, order):
    names = [n for n in order if n in summary]
    means = [summary[n][metric]["mean"] * 100 for n in names]
    stds = [summary[n][metric]["std"] * 100 for n in names]
    colors = ["#999999" if n == "naive" else
              "#d62728" if n == "ours" else
              "#1f77b4" if n in ("ewc", "mas") else "#2ca02c" for n in names]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)


def heatmap(ax, prefix, tag, seed, title):
    path = os.path.join(RESULTS_DIR, f"{prefix}{tag}", f"seed{seed}.json")
    if not os.path.exists(path):
        ax.axis("off")
        return
    A = np.array(json.load(open(path))["acc_matrix"]) * 100
    T = A.shape[0]
    masked = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
    im = ax.imshow(masked, vmin=80, vmax=100, cmap="viridis")
    for i in range(T):
        for j in range(i + 1):
            ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color="white")
    ax.set_xlabel("evaluated task")
    ax.set_ylabel("after training task")
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    order = ["naive", "mas", "ewc", "ours", "joint"]
    for prefix, label in [("", "Split CIFAR-10 (ResNet-18 features)"),
                          ("mnist_", "Split MNIST (pixels)")]:
        s = load_summary(prefix)
        if not s:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        bars(axes[0, 0], s, "avg_acc", "Average Accuracy (%) ↑", order)
        bars(axes[0, 1], s, "forgetting", "Forgetting (pp) ↓", order)
        heatmap(axes[1, 0], prefix, "naive" if prefix == "" else "mnist_naive",
                42, "naive: accuracy matrix (seed 42)")
        heatmap(axes[1, 1], prefix, "ours" if prefix == "" else "mnist_ours",
                42, "ours: accuracy matrix (seed 42)")
        fig.suptitle(label, fontsize=13)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"{prefix or 'cifar10_'}figures.png")
        fig.savefig(out, dpi=150)
        print("saved", out)


if __name__ == "__main__":
    main()
