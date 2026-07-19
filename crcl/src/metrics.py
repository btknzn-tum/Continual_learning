"""Metrics from the accuracy matrix A[T, T] (A[k, t] = acc on task t after task k)."""
import json
import os

import numpy as np


def compute_metrics(A: np.ndarray):
    T = A.shape[0]
    final = A[T - 1, :]
    avg_acc = float(final.mean())
    plasticity = float(np.mean([A[t, t] for t in range(T)]))
    if T > 1:
        forgetting = float(np.mean([A[:, t].max() - A[T - 1, t] for t in range(T - 1)]))
    else:
        forgetting = 0.0
    return {"avg_acc": avg_acc, "forgetting": forgetting, "plasticity": plasticity}


def save_result(results_dir: str, name: str, seed: int, payload: dict):
    d = os.path.join(results_dir, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
