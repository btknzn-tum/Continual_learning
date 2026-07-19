"""Per-weight importance scores for protected training.

Core method "sfxphi": S(w) = SF(w) * phi(source unit)
  SF  = SynFlow path importance, data-free, one backward pass on an abs-weight
        clone with all-ones input (= backward DP over |w| path products).
  phi = mean post-ReLU activation of the weight's source unit on PREVIOUS
        tasks' data.
Baselines behind the same interface: sf, phi, wanda, taylor, mas.

Returned dict: {"fc1": [256,512], "fc2": [256,256], "b_fc1": [256], "b_fc2": [256]}
each normalized to [0, 1] per tensor.
"""
import copy

import torch
import torch.nn.functional as F

from reserve import mean_activations


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return t / (t.max() + 1e-12)


def synflow_scores(model, prev_task_ids):
    """SF for fc1/fc2 weights and biases. Operates on a clone, never the real model."""
    clone = copy.deepcopy(model)
    for p in clone.parameters():
        p.requires_grad_(True)
        with torch.no_grad():
            p.abs_()
    x = torch.ones(1, clone.fc1.in_features)
    h1 = torch.relu(clone.fc1(x))
    h2 = torch.relu(clone.fc2(h1))
    if prev_task_ids:
        out = sum(clone.heads[str(t)](h2).sum() for t in prev_task_ids)
    else:
        out = h2.sum()
    out.backward()
    sf = {}
    for name, mod in (("fc1", clone.fc1), ("fc2", clone.fc2)):
        sf[name] = (mod.weight.detach().abs() * mod.weight.grad.abs())
        sf["b_" + name] = (mod.bias.detach().abs() * mod.bias.grad.abs())
    return sf


def phi_activations(model, prev_x):
    """Normalized mean activation per hidden unit over previous tasks' features."""
    a1, a2 = mean_activations(model, prev_x)
    return _normalize(a1), _normalize(a2)


def importance(model, prev_data, prev_task_ids, method: str = "sfxphi"):
    """prev_data: list of (task_id, x, y) tensors from previous tasks."""
    prev_x = torch.cat([x for _, x, _ in prev_data])
    d1 = model.fc1.out_features
    d2 = model.fc2.out_features

    if method == "l2":
        # uniform importance = plain L2-SP (cheapest control)
        S = {
            "fc1": torch.ones_like(model.fc1.weight),
            "fc2": torch.ones_like(model.fc2.weight),
            "b_fc1": torch.ones(d1),
            "b_fc2": torch.ones(d2),
        }
    elif method in ("sf", "sfxphi", "phi"):
        phi1, phi2 = phi_activations(model, prev_x)
        if method == "phi":
            S = {
                "fc1": torch.ones_like(model.fc1.weight),          # source = raw input
                "fc2": phi1.unsqueeze(0).expand(d2, d1).clone(),   # source unit of W2 is h1 unit
                "b_fc1": torch.ones(d1),
                "b_fc2": torch.ones(d2),
            }
        else:
            sf = synflow_scores(model, prev_task_ids)
            S = {k: v.clone() for k, v in sf.items()}
            if method == "sfxphi":
                S["fc2"] = S["fc2"] * phi1.unsqueeze(0)  # gate by source-unit activation
    elif method == "wanda":
        # |w| * L2 norm of source activation per column
        n = len(prev_x)
        x_sq = (prev_x ** 2).sum(0)
        h1_sq = None
        with torch.no_grad():
            for i in range(0, n, 1024):
                h1, _ = model.hidden(prev_x[i:i + 1024])
                h1_sq = (h1 ** 2).sum(0) if h1_sq is None else h1_sq + (h1 ** 2).sum(0)
        S = {
            "fc1": model.fc1.weight.detach().abs() * torch.sqrt(x_sq).unsqueeze(0),
            "fc2": model.fc2.weight.detach().abs() * torch.sqrt(h1_sq).unsqueeze(0),
            "b_fc1": model.fc1.bias.detach().abs(),
            "b_fc2": model.fc2.bias.detach().abs(),
        }
    elif method in ("taylor", "mas", "ewc"):
        acc = {"fc1": 0, "fc2": 0, "b_fc1": 0, "b_fc2": 0}
        n_batches = 0
        for t, x, y in prev_data:
            for i in range(0, len(x), 256):
                xb, yb = x[i:i + 256], y[i:i + 256]
                model.zero_grad()
                _, _, logits = model(xb, t)
                if method == "mas":
                    loss = logits.pow(2).sum(dim=1).mean()
                else:
                    loss = F.cross_entropy(logits, yb)
                loss.backward()
                for name, mod in (("fc1", model.fc1), ("fc2", model.fc2)):
                    g_w, g_b = mod.weight.grad, mod.bias.grad
                    if method == "taylor":
                        acc[name] = acc[name] + (mod.weight.detach() * g_w).abs()
                        acc["b_" + name] = acc["b_" + name] + (mod.bias.detach() * g_b).abs()
                    elif method == "ewc":
                        acc[name] = acc[name] + g_w ** 2
                        acc["b_" + name] = acc["b_" + name] + g_b ** 2
                    else:  # mas
                        acc[name] = acc[name] + g_w.abs()
                        acc["b_" + name] = acc["b_" + name] + g_b.abs()
                n_batches += 1
        model.zero_grad()
        S = {k: v / n_batches for k, v in acc.items()}
    else:
        raise ValueError(f"unknown importance method {method}")

    return {k: _normalize(v) for k, v in S.items()}
