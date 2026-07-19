import os
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # crcl/
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


DEFAULT_CONFIG = {
    "dataset": "cifar10",
    "backbone": "resnet18",
    "n_tasks": 5,
    "d_hidden": 256,
    # reserve
    "q": 0.30,
    "beta_res": 0.01,
    "tau_dormant": 1e-3,
    "tau_claim": 0.05,
    # protection
    "alpha": 10.0,
    "gamma": 100.0,
    "beta_crit": 0.10,
    "importance": "sfxphi",
    "phi_samples": 2000,  # per previous task
    # optimization
    "lr": 1e-3,
    "epochs": 20,
    "batch_size": 128,
    # method: naive | reg:<signal> | si | lwf | er | ours | joint
    "method": "reg:mas",
    "head_trim": True,
    # SI / LwF / ER hyperparameters
    "si_xi": 1e-3,
    "lwf_lambda": 1.0,
    "lwf_T": 2.0,
    "er_per_class": 20,
    "er_weight": 1.0,
    "seed": 42,
}
