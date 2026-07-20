"""Sequential continual-learning trainer over cached features.

Methods (cfg["method"]):
  naive          — plain sequential fine-tuning (lower bound)
  reg:<signal>   — pure importance-weighted quadratic penalty; signal in
                   {mas, ewc, l2, sfxphi, sf, phi, wanda, taylor}
                   ("mas_adapter" is an alias for reg:mas)
  mask:<signal>  — selective plasticity: rank adapter params by accumulated
                   importance, HARD-freeze the top (1-mask_q) fraction and
                   train only the least-important mask_q fraction (no soft
                   penalty). Task 0 trains everything.
  si             — Synaptic Intelligence (online path-integral importance)
  lwf            — Learning without Forgetting (distill previous heads on current data)
  er             — Experience Replay (small class-balanced feature buffer)
  ours           — Phase-1 framework (reserve + SF×phi + norm cap); kept as ablation
  joint          — all tasks pooled, multi-head (upper bound)

Backbone naming for cached features:
  "pixels" / "resnet18"                  → legacy single file  {ds}_{bb}_{split}.pt (raw)
  "<enc>_<depth>"                        → {ds}_<enc>_<depth>_{split}.pt (z-scored)
  "<enc>_<d1>+<d2>+..." (multi-tap)      → per-tap z-score (train stats), concat
"""
import copy
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adapter import Adapter, LoRAAdapter
from common import CACHE_DIR, set_seed


def _build_model(cfg, d_in):
    if cfg.get("arch", "mlp") == "lora":
        return LoRAAdapter(d_in=d_in, rank=cfg.get("lora_rank", 144))
    return Adapter(d_in=d_in, d_hidden=cfg["d_hidden"])
from importance import importance, phi_activations
from reserve import (claim_units, dormancy_fraction, make_reserved_masks,
                     mean_activations, reserve_loss, verify_dormant)
from tasks import make_splits, subsample, task_tensors

PARAM_NAMES = ("fc1", "fc2")
LEGACY_BACKBONES = ("pixels", "resnet18")  # raw features, single cache file


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _load_split(dataset, backbone, split):
    if backbone in LEGACY_BACKBONES:
        blob = torch.load(os.path.join(CACHE_DIR, f"{dataset}_{backbone}_{split}.pt"))
        return blob["features"], blob["labels"]
    enc, depth_spec = backbone.rsplit("_", 1)
    feats, labels = [], None
    for d in depth_spec.split("+"):
        blob = torch.load(os.path.join(CACHE_DIR, f"{dataset}_{enc}_{d}_{split}.pt"))
        feats.append(blob["features"])
        labels = blob["labels"]
    return feats, labels


def _zscore_concat(tr_parts, te_parts):
    xtr_parts, xte_parts = [], []
    for ftr, fte in zip(tr_parts, te_parts):
        mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
        xtr_parts.append((ftr - mu) / sd)
        xte_parts.append((fte - mu) / sd)
    return torch.cat(xtr_parts, 1), torch.cat(xte_parts, 1)


def load_cached(dataset: str, backbone: str):
    """Returns (xtr, ytr, xte, yte). Multi-tap: per-tap z-score from TRAIN stats."""
    tr, ytr = _load_split(dataset, backbone, "train")
    te, yte = _load_split(dataset, backbone, "test")
    if backbone in LEGACY_BACKBONES:
        return tr, ytr, te, yte
    xtr, xte = _zscore_concat(tr, te)
    return xtr, ytr, xte, yte


# 5-Datasets benchmark: each member dataset is one task (notMNIST->KMNIST).
FIVE_DATASETS = ["cifar10", "mnist", "svhn", "fashion", "kmnist"]


def _holdout(train_sets, val_frac, seed):
    """Carve a per-task validation split from train. Returns (new_train, val)."""
    new_train, val = [], []
    for i, (x, y) in enumerate(train_sets):
        g = torch.Generator().manual_seed(seed + i)
        perm = torch.randperm(len(x), generator=g)
        n_val = int(round(val_frac * len(x)))
        vi, ti = perm[:n_val], perm[n_val:]
        new_train.append((x[ti], y[ti]))
        val.append((x[vi], y[vi]))
    return new_train, val


def build_task_stream(cfg):
    """Return (train_sets, eval_sets, class_counts) as per-task (x, y) tensors,
    labels remapped to 0..C_t-1.

    dataset == "fivedata": 5 heterogeneous datasets, one per task.
    otherwise: one dataset split into cfg["n_tasks"] class groups.

    cfg["val_frac"] > 0: eval_sets are a validation holdout carved from TRAIN
    (test never touched) — used by run_sweep for honest hyperparameter selection.
    """
    if cfg["dataset"] == "fivedata":
        train_sets, test_sets, counts = [], [], []
        for name in FIVE_DATASETS:
            xtr, ytr, xte, yte = load_cached(name, cfg["backbone"])
            classes = sorted(int(c) for c in ytr.unique())
            train_sets.append(task_tensors(xtr, ytr, classes))
            test_sets.append(task_tensors(xte, yte, classes))
            counts.append(len(classes))
        counts_out = counts
    else:
        xtr, ytr, xte, yte = load_cached(cfg["dataset"], cfg["backbone"])
        n_classes = int(ytr.max()) + 1
        splits = make_splits(n_classes, cfg["n_tasks"])
        train_sets = [task_tensors(xtr, ytr, g) for g in splits]
        test_sets = [task_tensors(xte, yte, g) for g in splits]
        counts_out = [len(g) for g in splits]

    if cfg.get("val_frac", 0.0) > 0.0:
        train_sets, test_sets = _holdout(train_sets, cfg["val_frac"], cfg["seed"])
    return train_sets, test_sets, counts_out


@torch.no_grad()
def evaluate(model, x, y, task_id, batch_size: int = 2048, zero_h1=None, zero_h2=None):
    correct = 0
    for i in range(0, len(x), batch_size):
        _, _, logits = model(x[i:i + batch_size], task_id, zero_h1, zero_h2)
        correct += int((logits.argmax(1) == y[i:i + batch_size]).sum())
    return correct / len(x)


def _params(model):
    out = {}
    for n in PARAM_NAMES:
        mod = getattr(model, n)
        out[n] = mod.weight
        out["b_" + n] = mod.bias
    return out


def _snapshot(model):
    return {k: v.detach().clone() for k, v in _params(model).items()}


class SIState:
    """Synaptic Intelligence accumulators over the adapter parameters."""

    def __init__(self, model, xi=1e-3):
        self.xi = xi
        self.omega = {k: torch.zeros_like(v) for k, v in _params(model).items()}
        self._step_acc = None
        self._w_task_start = None

    def start_task(self, model):
        self._step_acc = {k: torch.zeros_like(v) for k, v in _params(model).items()}
        self._w_task_start = _snapshot(model)

    def pre_step(self, model):
        self._g = {k: (v.grad.detach().clone() if v.grad is not None else None)
                   for k, v in _params(model).items()}
        self._w_pre = _snapshot(model)

    def post_step(self, model):
        for k, v in _params(model).items():
            if self._g[k] is not None:
                delta = v.detach() - self._w_pre[k]
                self._step_acc[k] += -self._g[k] * delta

    def end_task(self, model):
        w_end = _snapshot(model)
        for k in self.omega:
            total_delta = w_end[k] - self._w_task_start[k]
            self.omega[k] += self._step_acc[k].clamp(min=0) / (total_delta ** 2 + self.xi)

    def normalized(self):
        # Global (single-max) scaling — preserves cross-tensor ratios AND the
        # task-count growth of the cumulative omega (per-tensor max would erase
        # both); matches the normalization applied to reg:* signals.
        gmax = max(float(v.max()) for v in self.omega.values()) + 1e-12
        return {k: v / gmax for k, v in self.omega.items()}


class ReplayBuffer:
    """Class-balanced feature buffer over previous tasks."""

    def __init__(self, per_class, seed):
        self.per_class = per_class
        self.g = torch.Generator().manual_seed(seed)
        self.x, self.y, self.t = [], [], []

    def add_task(self, task_id, x, y):
        for c in y.unique():
            idx = (y == c).nonzero(as_tuple=True)[0]
            pick = idx[torch.randperm(len(idx), generator=self.g)[:self.per_class]]
            self.x.append(x[pick])
            self.y.append(y[pick])
            self.t.append(torch.full((len(pick),), task_id, dtype=torch.long))

    def __len__(self):
        return sum(len(x) for x in self.x)

    def sample(self, n):
        x = torch.cat(self.x); y = torch.cat(self.y); t = torch.cat(self.t)
        idx = torch.randperm(len(x), generator=self.g)[:n]
        return x[idx], y[idx], t[idx]


def _replay_loss(model, buffer, n):
    xb, yb, tb = buffer.sample(n)
    loss = 0.0
    for k in tb.unique():
        sel = tb == k
        _, _, logits = model(xb[sel], int(k))
        loss = loss + F.cross_entropy(logits, yb[sel], reduction="sum")
    return loss / len(xb)


def _lwf_loss(model, teacher, xb, prev_tasks, T):
    loss = 0.0
    for k in prev_tasks:
        with torch.no_grad():
            _, _, t_logits = teacher(xb, k)
        _, _, s_logits = model(xb, k)
        loss = loss + F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean") * (T * T)
    # Sum over previous heads (constant per-head KD weight, per Li & Hoiem);
    # averaging would decay each old head's distillation as 1/t on long streams.
    return loss


def train_task(model, x, y, task_id, cfg, res_masks=None, S=None, theta_old=None,
               si_state=None, teacher=None, prev_tasks=None, buffer=None,
               grad_masks=None):
    dl = DataLoader(TensorDataset(x, y), batch_size=cfg["batch_size"], shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    denom = None
    if theta_old is not None:
        denom = torch.sqrt(sum((theta_old[n] ** 2).sum() for n in PARAM_NAMES))
    for _ in range(cfg["epochs"]):
        for xb, yb in dl:
            h1, h2, logits = model(xb, task_id)
            loss = F.cross_entropy(logits, yb)
            if res_masks is not None and cfg["beta_res"] > 0:
                loss = loss + cfg["beta_res"] * reserve_loss(h1, h2, res_masks)
            if S is not None and cfg["alpha"] > 0:
                prot = sum((S[k] * (v - theta_old[k]) ** 2).sum()
                           for k, v in _params(model).items())
                loss = loss + cfg["alpha"] * prot
            if theta_old is not None and cfg["gamma"] > 0:
                dsq = sum(((getattr(model, n).weight - theta_old[n]) ** 2).sum()
                          for n in PARAM_NAMES)
                rel = torch.sqrt(dsq + 1e-12) / denom
                loss = loss + cfg["gamma"] * torch.relu(rel - cfg["beta_crit"]) ** 2
            if teacher is not None and prev_tasks:
                loss = loss + cfg["lwf_lambda"] * _lwf_loss(
                    model, teacher, xb, prev_tasks, cfg["lwf_T"])
            if buffer is not None and len(buffer) > 0:
                loss = loss + cfg["er_weight"] * _replay_loss(
                    model, buffer, min(cfg["batch_size"], len(buffer)))
            opt.zero_grad()
            loss.backward()
            if grad_masks is not None:
                # selective plasticity: hard-freeze protected params by zeroing
                # their gradients (only the least-important fraction may move)
                for k, v in _params(model).items():
                    if v.grad is not None:
                        v.grad *= grad_masks[k]
            if si_state is not None:
                si_state.pre_step(model)
                opt.step()
                si_state.post_step(model)
            else:
                opt.step()


def _trim_and_freeze_head(model, task_id, task_x, tau, enabled):
    """Zero head columns reading from units dormant on this task's data, then freeze."""
    if enabled:
        _, a2 = mean_activations(model, task_x)
        with torch.no_grad():
            model.heads[str(task_id)].weight[:, a2 < tau] = 0.0
    model.freeze_head(task_id)


def run_sequence(cfg):
    set_seed(cfg["seed"])
    train_sets, test_sets, class_counts = build_task_stream(cfg)
    T = len(train_sets)
    d_in = train_sets[0][0].shape[1]
    model = _build_model(cfg, d_in)

    method = cfg["method"]
    use_reserve = method == "ours"
    if method == "mas_adapter":
        pure_signal = "mas"
    elif method.startswith("reg:"):
        pure_signal = method.split(":", 1)[1]
    else:
        pure_signal = None
    use_pure = pure_signal is not None
    use_mask = method.startswith("mask:")
    mask_signal = method.split(":", 1)[1] if use_mask else None
    use_si = method == "si"
    use_lwf = method == "lwf"
    use_er = method == "er"
    # Pure/SI paths must not inherit the harmful framework knobs (gamma norm cap,
    # reserve loss) from cfg defaults.
    cfg_clean = {**cfg, "gamma": 0.0, "beta_res": 0.0}

    R = make_reserved_masks(cfg["d_hidden"], cfg["q"], seed=cfg["seed"]) if use_reserve else None
    R_unclaimed = copy.deepcopy(R) if use_reserve else None
    si_state = SIState(model, cfg["si_xi"]) if use_si else None
    buffer = ReplayBuffer(cfg["er_per_class"], cfg["seed"] + 7) if use_er else None

    A = np.zeros((T, T))
    stats = {"dormancy": None, "claimed_per_task": [], "delta_rel": []}

    for t in range(T):
        x, y = train_sets[t]
        model.add_head(t, class_counts[t])
        if use_si:
            si_state.start_task(model)

        if t == 0:
            train_task(model, x, y, t, cfg if use_reserve else cfg_clean,
                       res_masks=R if use_reserve else None, si_state=si_state)
        elif use_reserve or use_pure:
            prev_data = [(k,) + subsample(*train_sets[k], cfg["phi_samples"], seed=cfg["seed"] + k)
                         for k in range(t)]
            sig = pure_signal if use_pure else cfg["importance"]
            S = importance(model, prev_data, list(range(t)), method=sig)
            theta_old = _snapshot(model)
            train_task(model, x, y, t, cfg if use_reserve else cfg_clean,
                       res_masks=R_unclaimed if use_reserve else None,
                       S=S, theta_old=theta_old)
            with torch.no_grad():
                dsq = sum(((getattr(model, n).weight - theta_old[n]) ** 2).sum()
                          for n in PARAM_NAMES)
                den = torch.sqrt(sum((theta_old[n] ** 2).sum() for n in PARAM_NAMES))
                stats["delta_rel"].append(float(torch.sqrt(dsq) / den))
        elif use_mask:
            prev_data = [(k,) + subsample(*train_sets[k], cfg["phi_samples"], seed=cfg["seed"] + k)
                         for k in range(t)]
            S = importance(model, prev_data, list(range(t)), method=mask_signal)
            # bottom mask_q fraction (least important) stays trainable; ties at
            # the threshold (e.g. exact zeros) may admit slightly more than q
            flat = torch.cat([v.flatten() for v in S.values()])
            kth = max(1, int(cfg["mask_q"] * flat.numel()))
            thresh = flat.kthvalue(kth).values
            grad_masks = {k: (v <= thresh).float() for k, v in S.items()}
            train_task(model, x, y, t, cfg_clean, grad_masks=grad_masks)
        elif use_si:
            theta_old = _snapshot(model)
            train_task(model, x, y, t, cfg_clean,
                       S=si_state.normalized(), theta_old=theta_old,
                       si_state=si_state)
        elif use_lwf:
            teacher = copy.deepcopy(model)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            train_task(model, x, y, t, cfg_clean,
                       teacher=teacher, prev_tasks=list(range(t)))
        elif use_er:
            train_task(model, x, y, t, cfg_clean, buffer=buffer)
        else:  # naive
            train_task(model, x, y, t, cfg_clean)

        if use_si:
            si_state.end_task(model)
        if use_er:
            buffer.add_task(t, x, y)

        # diagnostic assumes the MLP adapter's hidden sizes — skip for LoRA
        if method == "naive" and t == 0 and cfg.get("arch", "mlp") == "mlp":
            rand_masks = make_reserved_masks(cfg["d_hidden"], cfg["q"],
                                             seed=cfg["seed"] + 999)
            stats["accidental_dormancy"] = dormancy_fraction(
                model, x, rand_masks, tau=cfg["tau_dormant"])

        if use_reserve and t == 0:
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

    return {"config": cfg, "acc_matrix": A.tolist(), "git": git_hash()}, A, stats


def run_joint(cfg):
    set_seed(cfg["seed"])
    train_sets, test_sets, class_counts = build_task_stream(cfg)
    d_in = train_sets[0][0].shape[1]
    model = _build_model(cfg, d_in)
    xs, ys, ts = [], [], []
    for t, (x, y) in enumerate(train_sets):
        model.add_head(t, class_counts[t])
        xs.append(x); ys.append(y)
        ts.append(torch.full((len(y),), t, dtype=torch.long))
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
    return [evaluate(model, x, y, t) for t, (x, y) in enumerate(test_sets)]
