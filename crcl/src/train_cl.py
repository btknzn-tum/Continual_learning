"""Sequential continual-learning trainer over cached features.

Methods:
  ours  — reserve loss on task 1 + SF×phi-protected training on tasks 2..T
  naive — plain sequential fine-tuning (lower bound)
  joint — all tasks pooled, multi-head (upper bound)
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adapter import Adapter
from common import CACHE_DIR, set_seed
from importance import importance, phi_activations
from reserve import (claim_units, dormancy_fraction, make_reserved_masks,
                     mean_activations, reserve_loss, verify_dormant)
from tasks import make_splits, subsample, task_tensors


def load_cached(dataset: str, backbone: str):
    tr = torch.load(os.path.join(CACHE_DIR, f"{dataset}_{backbone}_train.pt"))
    te = torch.load(os.path.join(CACHE_DIR, f"{dataset}_{backbone}_test.pt"))
    return tr["features"], tr["labels"], te["features"], te["labels"]


@torch.no_grad()
def evaluate(model, x, y, task_id, batch_size: int = 2048, zero_h1=None, zero_h2=None):
    correct = 0
    for i in range(0, len(x), batch_size):
        _, _, logits = model(x[i:i + batch_size], task_id, zero_h1, zero_h2)
        correct += int((logits.argmax(1) == y[i:i + batch_size]).sum())
    return correct / len(x)


def _snapshot(model):
    return {
        "w": {n: getattr(model, n).weight.detach().clone() for n in ("fc1", "fc2")},
        "b": {n: getattr(model, n).bias.detach().clone() for n in ("fc1", "fc2")},
    }


def train_task(model, x, y, task_id, cfg, res_masks=None, S=None, theta_old=None):
    dl = DataLoader(TensorDataset(x, y), batch_size=cfg["batch_size"], shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    denom = None
    if theta_old is not None:
        denom = torch.sqrt(sum((w ** 2).sum() for w in theta_old["w"].values()))
    for _ in range(cfg["epochs"]):
        for xb, yb in dl:
            h1, h2, logits = model(xb, task_id)
            loss = F.cross_entropy(logits, yb)
            if res_masks is not None and cfg["beta_res"] > 0:
                loss = loss + cfg["beta_res"] * reserve_loss(h1, h2, res_masks)
            if S is not None and cfg["alpha"] > 0:
                prot = 0.0
                for n in ("fc1", "fc2"):
                    mod = getattr(model, n)
                    prot = prot + (S[n] * (mod.weight - theta_old["w"][n]) ** 2).sum()
                    prot = prot + (S["b_" + n] * (mod.bias - theta_old["b"][n]) ** 2).sum()
                loss = loss + cfg["alpha"] * prot
            if theta_old is not None and cfg["gamma"] > 0:
                dsq = sum(((getattr(model, n).weight - theta_old["w"][n]) ** 2).sum()
                          for n in ("fc1", "fc2"))
                rel = torch.sqrt(dsq + 1e-12) / denom
                loss = loss + cfg["gamma"] * torch.relu(rel - cfg["beta_crit"]) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()


def _trim_and_freeze_head(model, task_id, task_x, tau, enabled):
    """Zero head columns reading from units dormant on this task's data, then freeze.

    A dormant unit contributes ~0 to this head now; zeroing its column makes the
    head provably immune to that unit being woken by a FUTURE task.
    """
    if enabled:
        _, a2 = mean_activations(model, task_x)
        with torch.no_grad():
            model.heads[str(task_id)].weight[:, a2 < tau] = 0.0
    model.freeze_head(task_id)


def run_sequence(cfg):
    set_seed(cfg["seed"])
    xtr, ytr, xte, yte = load_cached(cfg["dataset"], cfg["backbone"])
    n_classes = int(ytr.max()) + 1
    splits = make_splits(n_classes, cfg["n_tasks"])
    T = cfg["n_tasks"]
    model = Adapter(d_in=xtr.shape[1], d_hidden=cfg["d_hidden"])

    train_sets = [task_tensors(xtr, ytr, g) for g in splits]
    test_sets = [task_tensors(xte, yte, g) for g in splits]

    use_reserve = cfg["method"] == "ours"
    use_mas_adapter = cfg["method"] == "mas_adapter"
    R = make_reserved_masks(cfg["d_hidden"], cfg["q"], seed=cfg["seed"]) if use_reserve else None
    R_unclaimed = copy.deepcopy(R) if use_reserve else None

    A = np.zeros((T, T))
    stats = {"dormancy": None, "claimed_per_task": [], "delta_rel": []}

    for t in range(T):
        x, y = train_sets[t]
        model.add_head(t, len(splits[t]))
        if t == 0 or (not use_reserve and not use_mas_adapter):
            train_task(model, x, y, t, cfg,
                       res_masks=R if use_reserve else None)
        else:
            prev_data = [(k,) + subsample(*train_sets[k], cfg["phi_samples"], seed=cfg["seed"] + k)
                         for k in range(t)]
            importance_method = "mas" if use_mas_adapter else cfg["importance"]
            S = importance(model, prev_data, list(range(t)), method=importance_method)
            theta_old = _snapshot(model)
            train_task(model, x, y, t, cfg,
                       res_masks=R_unclaimed if use_reserve else None,
                       S=S, theta_old=theta_old)
            with torch.no_grad():
                dsq = sum(((getattr(model, n).weight - theta_old["w"][n]) ** 2).sum()
                          for n in ("fc1", "fc2"))
                den = torch.sqrt(sum((w ** 2).sum() for w in theta_old["w"].values()))
                stats["delta_rel"].append(float(torch.sqrt(dsq) / den))

        if not use_reserve and not use_mas_adapter and t == 0:
            # control: accidental dormancy right after task 1, same timepoint
            # as the ours-side measurement below (comparing after-1-task vs
            # after-T-tasks states would invalidate the comparison)
            rand_masks = make_reserved_masks(cfg["d_hidden"], cfg["q"],
                                             seed=cfg["seed"] + 999)
            stats["accidental_dormancy"] = dormancy_fraction(
                model, x, rand_masks, tau=cfg["tau_dormant"])

        if use_reserve and t == 0:
            # dormancy verification on task-1 data
            dormant, frac = verify_dormant(model, x, R, tau=cfg["tau_dormant"])
            xt0, yt0 = test_sets[0]
            acc_plain = evaluate(model, xt0, yt0, 0)
            acc_zeroed = evaluate(model, xt0, yt0, 0,
                                  zero_h1=dormant["h1"], zero_h2=dormant["h2"])
            stats["dormancy"] = {
                "frac_verified": frac,
                "zero_out_acc_delta": acc_plain - acc_zeroed,
            }

        if use_reserve:
            phi1, phi2 = phi_activations(model, x)
            before = int(R_unclaimed["h1"].sum() + R_unclaimed["h2"].sum())
            R_unclaimed = claim_units(R_unclaimed, phi1, phi2, cfg["tau_claim"])
            after = int(R_unclaimed["h1"].sum() + R_unclaimed["h2"].sum())
            stats["claimed_per_task"].append(before - after)

        _trim_and_freeze_head(model, t, x, cfg["tau_dormant"],
                              enabled=use_reserve and cfg["head_trim"])

        for k in range(t + 1):
            A[t, k] = evaluate(model, *test_sets[k], k)

    return {"config": cfg, "acc_matrix": A.tolist()}, A, stats


def run_joint(cfg):
    set_seed(cfg["seed"])
    xtr, ytr, xte, yte = load_cached(cfg["dataset"], cfg["backbone"])
    n_classes = int(ytr.max()) + 1
    splits = make_splits(n_classes, cfg["n_tasks"])
    T = cfg["n_tasks"]
    model = Adapter(d_in=xtr.shape[1], d_hidden=cfg["d_hidden"])
    xs, ys, ts = [], [], []
    for t, g in enumerate(splits):
        model.add_head(t, len(g))
        x, y = task_tensors(xtr, ytr, g)
        xs.append(x); ys.append(y); ts.append(torch.full((len(y),), t, dtype=torch.long))
    x_all, y_all, t_all = torch.cat(xs), torch.cat(ys), torch.cat(ts)
    dl = DataLoader(TensorDataset(x_all, y_all, t_all),
                    batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    for _ in range(cfg["epochs"]):
        for xb, yb, tb in dl:
            loss = 0.0
            for t in tb.unique():
                sel = tb == t
                _, _, logits = model(xb[sel], int(t))
                loss = loss + F.cross_entropy(logits, yb[sel], reduction="sum")
            loss = loss / len(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    per_task = [evaluate(model, *task_tensors(xte, yte, g), t)
                for t, g in enumerate(splits)]
    return per_task
