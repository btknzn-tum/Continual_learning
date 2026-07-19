"""Selective-plasticity CL with a partially trainable encoder (our method).

Per task t>0 the loss is
    CE(current task)
  + alpha     * sum S_ad  (theta_ad - theta_ad_old)^2     [adapter MAS penalty]
  + alpha_enc * sum S_enc (theta_e  - theta_e_old)^2      [penalty on the freed
                                                           encoder fraction]
  + sim_lambda * MSE(f_t(x), f_{t-1}(x))                  [feature-similarity
                                                           anchor to the frozen
                                                           previous encoder]
with the encoder HARD-masked so only the bottom enc_q fraction of parameters
(by accumulated MAS importance through encoder+adapter) receives gradients.
Once a task claims a parameter its importance rises, so it (a) gets penalized
and (b) rotates OUT of the trainable set at the next task — capacity rotation
without stored per-task masks (unlike PackNet). Mask overlap across tasks is
logged (stats["mask_overlap"]) as direct evidence for/against rotation.

Only encoder stages UPSTREAM of the deepest requested tap are eligible for
selective training (params downstream of the taps get zero gradient and would
otherwise absorb the whole bottom-q budget as a silent no-op).

Task 0 trains the adapter only (encoder frozen; importance undefined yet).
Z-score stats are computed PER TASK at task arrival (current encoder, that
task's train data only — causal, mirrors the cached pipeline's per-member
stats on fivedata). Defaults (epochs=20, phi_samples=2000) match the cached
benchmark for honest comparison. BatchNorm stays in eval() mode for the whole
stream (running stats frozen). NOT full fine-tuning.

DATA PATH: raw uint8 images live in RAM; resize-to-224 (bicubic) + ImageNet
normalization run ON THE GPU per batch. No DataLoader workers — throughput is
GPU-bound and immune to CPU contention from parallel benchmark jobs.

Ablations (get their own result tags automatically): --sim-lambda 0,
--alpha-enc 0, --enc-q 0 (frozen encoder).
Training/importance/extraction run under bf16 autocast; penalties in fp32.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models

from cache_features import FIVE_DATASETS, IMAGENET_MEAN, IMAGENET_STD
from common import DATA_DIR, DEFAULT_CONFIG, RESULTS_DIR, set_seed
from metrics import compute_metrics, save_result
from tasks import make_splits
from train_cl import _params, _snapshot, git_hash
from adapter import Adapter

TAP_DIMS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
# encoder stages in forward order; eligibility stops at the deepest tap
STAGE_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def bwt(A):
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, t] - A[t, t] for t in range(T - 1)]))


def _qstr(q):
    return f"{q:g}".replace(".", "p")


# ---------------------------------------------------------------- raw tensors
def load_raw(name, train):
    """(uint8 [N,3,H,W] CPU tensor, LongTensor labels) straight from disk."""
    kw = dict(root=DATA_DIR, download=True)
    if name == "cifar10":
        ds = datasets.CIFAR10(train=train, **kw)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets)
    elif name == "cifar100":
        ds = datasets.CIFAR100(train=train, **kw)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets)
    elif name == "svhn":
        ds = datasets.SVHN(split="train" if train else "test", **kw)
        x = torch.from_numpy(ds.data)  # already [N,3,32,32]
        y = torch.tensor(ds.labels)
    elif name in ("mnist", "fashion", "kmnist"):
        cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST,
               "kmnist": datasets.KMNIST}[name]
        ds = cls(train=train, **kw)
        x = ds.data.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        y = ds.targets.clone()
    else:
        raise ValueError(name)
    return x, y.long()


class GPUTransform:
    """uint8 batch -> resized (224, bicubic) ImageNet-normalized float on GPU."""

    def __init__(self, device):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, xb_u8):
        x = xb_u8.float().div_(255.0)
        x = F.interpolate(x, size=224, mode="bicubic",
                          align_corners=False).clamp_(0.0, 1.0)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)


def iter_batches(x, y, batch, device, shuffle=False, generator=None):
    idx = (torch.randperm(len(x), generator=generator) if shuffle
           else torch.arange(len(x)))
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        yield x[sel].to(device, non_blocking=True), y[sel].to(device)


def build_stream(dataset, n_tasks, val_frac=0.0, seed=42):
    """Return (train_sets, eval_sets, class_counts) as raw-uint8 (x, y) pairs,
    labels remapped to 0..C_t-1. val_frac>0 carves a per-task validation split
    from TRAIN (test never touched)."""
    def remap(y, classes):
        lut = torch.full((int(max(classes)) + 1,), -1, dtype=torch.long)
        for i, c in enumerate(sorted(classes)):
            lut[c] = i
        return lut[y]

    train_sets, eval_sets, counts = [], [], []
    if dataset == "fivedata":
        for name in FIVE_DATASETS:
            tload = time.time()
            xtr, ytr = load_raw(name, True)
            xte, yte = load_raw(name, False)
            print(f"  [data] {name}: {len(xtr)} train ({time.time() - tload:.0f}s)",
                  flush=True)
            classes = sorted(ytr.unique().tolist())
            train_sets.append((xtr, remap(ytr, classes)))
            eval_sets.append((xte, remap(yte, classes)))
            counts.append(len(classes))
    else:
        xtr, ytr = load_raw(dataset, True)
        xte, yte = load_raw(dataset, False)
        for g in make_splits(int(ytr.max()) + 1, n_tasks):
            mtr, mte = torch.isin(ytr, torch.tensor(g)), torch.isin(yte, torch.tensor(g))
            train_sets.append((xtr[mtr], remap(ytr[mtr], g)))
            eval_sets.append((xte[mte], remap(yte[mte], g)))
            counts.append(len(g))
    if val_frac > 0:
        new_tr, new_ev = [], []
        for i, (x, y) in enumerate(train_sets):
            g = torch.Generator().manual_seed(seed + i)
            perm = torch.randperm(len(x), generator=g)
            n_val = int(round(val_frac * len(x)))
            vi, ti = perm[:n_val], perm[n_val:]
            new_tr.append((x[ti], y[ti]))
            new_ev.append((x[vi], y[vi]))
        train_sets, eval_sets = new_tr, new_ev
    return train_sets, eval_sets, counts


def subsample_xy(x, y, n, seed):
    if n >= len(x):
        return x, y
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n]
    return x[idx], y[idx]


# ------------------------------------------------------------------- encoder
def enc_named_params(encoder, taps):
    """Encoder params eligible for selective training: only stages UPSTREAM of
    (and including) the deepest requested tap; fc always excluded."""
    deepest = max(STAGE_ORDER.index(t) for t in taps)
    keep = set(STAGE_ORDER[:deepest + 1])
    return {n: p for n, p in encoder.named_parameters()
            if n.split(".")[0] in keep}


def resnet_taps(model, x, taps):
    """Forward only as deep as the deepest requested tap."""
    out = {}
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    deepest = max(STAGE_ORDER.index(t) for t in taps)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        z = getattr(model, name)(z)
        if name in taps:
            out[name] = F.adaptive_avg_pool2d(z, 1).flatten(1)
        if STAGE_ORDER.index(name) >= deepest:
            break
    return out


def tap_feats(encoder, x, taps, stats):
    d = resnet_taps(encoder, x, taps)
    parts = []
    for t in taps:
        f = d[t]
        if stats is not None:
            mu, sd = stats[t]
            f = (f - mu) / sd
        parts.append(f)
    return torch.cat(parts, 1)


@torch.no_grad()
def compute_tap_stats(encoder, x, y, taps, device, batch, tf):
    """Per-tap (mu, sd) over the task's train data, CURRENT encoder."""
    buf = {t: [] for t in taps}
    for xb, _ in iter_batches(x, y, batch, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = resnet_taps(encoder, tf(xb), taps)
        for t in taps:
            buf[t].append(d[t].float().cpu())
    out = {}
    for t in taps:
        f = torch.cat(buf[t])
        out[t] = (f.mean(0).to(device), (f.std(0) + 1e-6).to(device))
    return out


@torch.no_grad()
def extract_feats(encoder, x, y, taps, stats, device, batch, tf):
    fs = []
    for xb, _ in iter_batches(x, y, batch, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = tap_feats(encoder, tf(xb), taps, stats)
        fs.append(f.float())
    return torch.cat(fs), y.to(device)


def joint_mas_importance(encoder, adapter, train_sets, tasks, taps,
                         stats_per_task, device, samples, batch, tf, seed):
    """MAS importance |d||logits||^2/dw| accumulated over prev-task samples,
    jointly for encoder AND adapter (one backward computes both)."""
    ep = enc_named_params(encoder, taps)
    imp_e = {n: torch.zeros_like(p) for n, p in ep.items()}
    imp_a = {k: torch.zeros_like(v) for k, v in _params(adapter).items()}
    for p in ep.values():
        p.requires_grad_(True)
    nb = 0
    for k in tasks:
        xs, ys = subsample_xy(*train_sets[k], samples, seed + k)
        for xb, _ in iter_batches(xs, ys, batch, device):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = tap_feats(encoder, tf(xb), taps, stats_per_task[k])
                _, _, logits = adapter(feats, k)
                loss = logits.float().pow(2).sum(1).mean()
            encoder.zero_grad(set_to_none=False)
            adapter.zero_grad(set_to_none=False)
            loss.backward()
            for n, p in ep.items():
                if p.grad is not None:
                    imp_e[n] += p.grad.abs()
            for kk, v in _params(adapter).items():
                if v.grad is not None:
                    imp_a[kk] += v.grad.abs()
            nb += 1
    encoder.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    imp_e = {n: v / nb for n, v in imp_e.items()}
    imp_a = {k: v / nb for k, v in imp_a.items()}
    # global single-max normalization (preserves cross-tensor ratios)
    gmax_a = max(float(v.max()) for v in imp_a.values()) + 1e-12
    imp_a = {k: v / gmax_a for k, v in imp_a.items()}
    gmax_e = max(float(v.max()) for v in imp_e.values()) + 1e-12
    imp_e = {n: v / gmax_e for n, v in imp_e.items()}
    return imp_e, imp_a


def bottom_q_masks(imp_e, q):
    """1.0 where the param is in the LEAST-important q fraction (trainable)."""
    if q <= 0:
        return {n: torch.zeros_like(v) for n, v in imp_e.items()}, 0.0
    flat = torch.cat([v.flatten() for v in imp_e.values()])
    kth = max(1, int(q * flat.numel()))
    thresh = flat.kthvalue(kth).values
    masks = {n: (v <= thresh).float() for n, v in imp_e.items()}
    frac = float(sum(m.sum() for m in masks.values()) / flat.numel())
    return masks, frac


@torch.no_grad()
def eval_head(adapter, feats, ys, task_id, batch=2048):
    correct = 0
    for i in range(0, len(feats), batch):
        _, _, logits = adapter(feats[i:i + batch], task_id)
        correct += int((logits.argmax(1) == ys[i:i + batch]).sum())
    return correct / len(feats)


def run_stream(cfg, device):
    set_seed(cfg["seed"])
    train_sets, eval_sets, counts = build_stream(
        cfg["dataset"], cfg["n_tasks"], cfg.get("val_frac", 0.0), cfg["seed"])
    T = len(train_sets)
    taps = cfg["depth"].split("+")
    tf = GPUTransform(device)
    B = cfg["batch_size"]

    print(f"  [data] stream ready ({T} tasks)", flush=True)
    encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    encoder.eval().to(device)  # eval() for the WHOLE stream: BN stats frozen
    encoder.to(memory_format=torch.channels_last)
    for p in encoder.parameters():
        p.requires_grad_(False)
    print("  [enc] resnet50 on gpu", flush=True)

    d_in = sum(TAP_DIMS[t] for t in taps)
    adapter = Adapter(d_in=d_in, d_hidden=cfg["d_hidden"]).to(device)

    A = np.zeros((T, T))
    run_stats = {"trainable_frac": [], "enc_delta": [], "task_sec": [],
                 "mask_overlap": []}
    stats_per_task = {}
    prev_mask_flat = None
    gen = torch.Generator().manual_seed(cfg["seed"])

    for t in range(T):
        t0 = time.time()
        adapter.add_head(t, counts[t])
        adapter.heads[str(t)].to(device)
        x_t, y_t = train_sets[t]
        # per-task z-stats at ARRIVAL, current encoder, this task's data only
        stats_per_task[t] = compute_tap_stats(encoder, x_t, y_t, taps, device,
                                              B, tf)
        print(f"  [t{t}] z-stats done ({time.time() - t0:.0f}s)", flush=True)

        if t == 0:
            # encoder frozen -> features constant: extract once, train fast
            feats, ys = extract_feats(encoder, x_t, y_t, taps,
                                      stats_per_task[0], device, B, tf)
            opt = torch.optim.AdamW(
                [p for p in adapter.parameters() if p.requires_grad],
                lr=cfg["lr"], weight_decay=0.0)
            for _ in range(cfg["epochs"]):
                perm = torch.randperm(len(feats), generator=gen)
                for i in range(0, len(perm), B):
                    sel = perm[i:i + B]
                    _, _, logits = adapter(feats[sel], 0)
                    loss = F.cross_entropy(logits, ys[sel])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            S_enc, S_ad = joint_mas_importance(
                encoder, adapter, train_sets, list(range(t)), taps,
                stats_per_task, device, cfg["phi_samples"], B, tf, cfg["seed"])
            enc_masks, frac = bottom_q_masks(S_enc, cfg["enc_q"])
            run_stats["trainable_frac"].append(round(frac, 4))
            cur_flat = (torch.cat([m.flatten() for m in enc_masks.values()])
                        .bool() if cfg["enc_q"] > 0 else None)
            if prev_mask_flat is not None and cur_flat is not None:
                inter = (prev_mask_flat & cur_flat).sum()
                union = (prev_mask_flat | cur_flat).sum()
                run_stats["mask_overlap"].append(
                    round(float(inter) / max(float(union), 1.0), 4))
            prev_mask_flat = cur_flat
            # penalty scale: renormalize importance WITHIN the trainable subset
            # (bottom-q values are tiny under the global max; without this the
            # encoder penalty is structurally a no-op and cannot be ablated)
            if cfg["enc_q"] > 0:
                pmax = max(float((v * enc_masks[n]).max())
                           for n, v in S_enc.items()) + 1e-12
                S_pen = {n: (v * enc_masks[n]) / pmax
                         for n, v in S_enc.items()}
            theta_a = _snapshot(adapter)
            ep = enc_named_params(encoder, taps)
            theta_e = {n: p.detach().clone() for n, p in ep.items()}
            use_sim = cfg["enc_q"] > 0 and cfg["sim_lambda"] > 0
            teacher = None
            if use_sim:
                teacher = copy.deepcopy(encoder).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)
            for p in ep.values():
                p.requires_grad_(cfg["enc_q"] > 0)

            groups = [{"params": [p for p in adapter.parameters()
                                  if p.requires_grad], "lr": cfg["lr"]}]
            if cfg["enc_q"] > 0:
                groups.append({"params": list(ep.values()), "lr": cfg["enc_lr"]})
            opt = torch.optim.AdamW(groups, weight_decay=0.0)

            for _ in range(cfg["epochs"]):
                for xb, yb in iter_batches(x_t, y_t, B, device, shuffle=True,
                                           generator=gen):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        xg = tf(xb)
                        feats = tap_feats(encoder, xg, taps, stats_per_task[t])
                        _, _, logits = adapter(feats, t)
                        loss = F.cross_entropy(logits, yb).float()
                        if use_sim:
                            with torch.no_grad():
                                tfeat = tap_feats(teacher, xg, taps,
                                                  stats_per_task[t])
                            loss = loss + cfg["sim_lambda"] * F.mse_loss(
                                feats.float(), tfeat.float())
                    # penalties in fp32 outside autocast (param-space sums)
                    if cfg["alpha"] > 0:
                        loss = loss + cfg["alpha"] * sum(
                            (S_ad[k] * (v - theta_a[k]) ** 2).sum()
                            for k, v in _params(adapter).items())
                    if cfg["enc_q"] > 0 and cfg["alpha_enc"] > 0:
                        loss = loss + cfg["alpha_enc"] * sum(
                            (S_pen[n] * (p - theta_e[n]) ** 2).sum()
                            for n, p in ep.items())
                    opt.zero_grad()
                    loss.backward()
                    if cfg["enc_q"] > 0:
                        for n, p in ep.items():
                            if p.grad is not None:
                                p.grad *= enc_masks[n]
                    opt.step()
            with torch.no_grad():
                dsq = sum(((p - theta_e[n]) ** 2).sum() for n, p in ep.items())
                den = torch.sqrt(sum((v ** 2).sum() for v in theta_e.values()))
                run_stats["enc_delta"].append(float(torch.sqrt(dsq) / den))
            for p in ep.values():
                p.requires_grad_(False)
            del teacher

        adapter.freeze_head(t)
        # encoder changed -> re-extract ALL seen eval sets under current
        # encoder, each with ITS OWN task's frozen z-stats
        for k in range(t + 1):
            fk, yk = extract_feats(encoder, *eval_sets[k], taps,
                                   stats_per_task[k], device, B, tf)
            A[t, k] = eval_head(adapter, fk, yk, k)
        run_stats["task_sec"].append(round(time.time() - t0, 1))
        print(f"  task {t}: acc_so_far={A[t, :t + 1].mean() * 100:.2f} "
              f"({run_stats['task_sec'][-1]}s)", flush=True)

    return A, run_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar10", "cifar100", "fivedata"])
    p.add_argument("--depths", nargs="*", default=["layer3+layer4"])
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--enc-q", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="adapter MAS penalty (tuned value for reg:mas)")
    p.add_argument("--alpha-enc", type=float, default=None,
                   help="encoder-side MAS penalty; default = --alpha")
    p.add_argument("--sim-lambda", type=float, default=1.0)
    p.add_argument("--enc-lr", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--phi-samples", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.0,
                   help=">0: tune on a train holdout (sweeps only)")
    p.add_argument("--workers", type=int, default=0, help="unused (GPU pipeline)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tag-suffix", default="")
    a = p.parse_args()
    assert torch.cuda.is_available(), "CUDA required (refusing silent CPU run)"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__}", flush=True)
    n_tasks = a.n_tasks or (5 if a.dataset in ("cifar10", "fivedata") else 10)
    alpha_enc = a.alpha if a.alpha_enc is None else a.alpha_enc

    for depth in a.depths:
        for seed in a.seeds:
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            cfg.update({
                "dataset": a.dataset, "depth": depth, "n_tasks": n_tasks,
                "seed": seed, "method": "enc:spmas", "enc_q": a.enc_q,
                "alpha": a.alpha, "alpha_enc": alpha_enc,
                "sim_lambda": a.sim_lambda, "enc_lr": a.enc_lr, "lr": a.lr,
                "epochs": a.epochs, "batch_size": a.batch_size,
                "phi_samples": a.phi_samples, "val_frac": a.val_frac,
                "backbone": f"r50enc_{depth}",
            })
            # ablations and HP variants get distinct tags automatically
            tag = (f"{a.dataset}_r50enc_{depth}_t{n_tasks}_spmas"
                   f"_q{_qstr(a.enc_q)}")
            if a.sim_lambda != 1.0:
                tag += f"_sl{_qstr(a.sim_lambda)}"
            if alpha_enc != a.alpha:
                tag += f"_ae{_qstr(alpha_enc)}"
            tag += a.tag_suffix
            out = os.path.join(RESULTS_DIR, tag, f"seed{seed}.json")
            if os.path.exists(out) and not a.overwrite:
                print(f"skip {tag} seed {seed} (exists)", flush=True)
                continue
            print(f"== {tag} seed {seed} ==", flush=True)
            t0 = time.time()
            A, run_stats = run_stream(cfg, device)
            m = compute_metrics(A)
            m["bwt"] = bwt(A)
            payload = {"config": cfg, "acc_matrix": A.tolist(), "metrics": m,
                       "stats": run_stats, "git": git_hash(),
                       "runtime_sec": round(time.time() - t0, 1)}
            save_result(RESULTS_DIR, tag, seed, payload)
            print(f"== {tag} seed {seed}: AvgAcc={m['avg_acc'] * 100:.2f} "
                  f"Forget={m['forgetting'] * 100:.2f} "
                  f"({payload['runtime_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
