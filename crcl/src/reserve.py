"""Reserve loss: manufacture dormant spare-capacity units, verify, claim."""
import torch


def make_reserved_masks(d_hidden: int, q: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    masks = {}
    k = int(round(q * d_hidden))
    for layer in ("h1", "h2"):
        perm = torch.randperm(d_hidden, generator=g)
        m = torch.zeros(d_hidden, dtype=torch.bool)
        m[perm[:k]] = True
        masks[layer] = m
    return masks


def reserve_loss(h1, h2, masks):
    """(1/|R|) * sum over reserved units of mean-over-batch post-ReLU activation."""
    total = h1.new_zeros(())
    count = 0
    for h, m in ((h1, masks["h1"]), (h2, masks["h2"])):
        if m.any():
            total = total + h[:, m].mean(dim=0).sum()
            count += int(m.sum())
    return total / count if count > 0 else total


@torch.no_grad()
def mean_activations(model, x, batch_size: int = 1024):
    """Mean post-ReLU activation per hidden unit over tensor x. Returns (a1, a2)."""
    s1 = s2 = None
    n = 0
    for i in range(0, len(x), batch_size):
        h1, h2 = model.hidden(x[i:i + batch_size])
        s1 = h1.sum(0) if s1 is None else s1 + h1.sum(0)
        s2 = h2.sum(0) if s2 is None else s2 + h2.sum(0)
        n += h1.shape[0]
    return s1 / n, s2 / n


@torch.no_grad()
def verify_dormant(model, x, masks, tau: float = 1e-3):
    """Which reserved units are verifiably dormant (mean act < tau) on data x."""
    a1, a2 = mean_activations(model, x)
    dormant = {"h1": masks["h1"] & (a1 < tau), "h2": masks["h2"] & (a2 < tau)}
    n_res = int(masks["h1"].sum() + masks["h2"].sum())
    n_dorm = int(dormant["h1"].sum() + dormant["h2"].sum())
    frac = n_dorm / max(1, n_res)
    return dormant, frac


@torch.no_grad()
def dormancy_fraction(model, x, masks, tau: float = 1e-3):
    """Fraction of an arbitrary unit set with mean act < tau (control check)."""
    _, frac = verify_dormant(model, x, masks, tau)
    return frac


def claim_units(unclaimed, phi1_norm, phi2_norm, tau_claim: float):
    """Remove from the unclaimed pool the units the current task woke up."""
    return {
        "h1": unclaimed["h1"] & ~(phi1_norm > tau_claim),
        "h2": unclaimed["h2"] & ~(phi2_norm > tau_claim),
    }
