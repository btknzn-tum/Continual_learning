# Continual Learning — MAS-Protected Adapters Benchmark

Systematic benchmark of **importance-regularized adapters** for continual learning on
frozen vision encoders: which protection method (MAS vs EWC / SI / LwF / ER) and which
**adapter insertion depth** (single & multi-tap, ResNet-50 stages / CLIP ViT blocks).
Target: Q1 journal paper. Full research plan: [plan.md](plan.md).

**Headline Phase-1 result** (Split CIFAR-10, frozen ResNet-18 features, 3 seeds):
MAS-protected adapter reaches **96.92 ± 0.24 AvgAcc / 0.49 ± 0.25 forgetting** vs naive
89.40 / 10.12 and joint upper bound 97.25 — rehearsal-free, minutes of training on cached
features.

## Repo layout

```
plan.md                  # the research plan (read this first)
crcl/
  src/
    cache_features.py    # frozen encoder → disk, multi-depth taps (resnet50, clip_vitb32)
    cache_pixels.py      # MNIST raw-pixel cache
    adapter.py           # 2-layer MLP adapter + per-task heads
    importance.py        # MAS / EWC / SI / SF×φ / wanda / taylor / l2 signals
    train_cl.py          # sequential trainer (reg:<signal>, naive, ours-framework, joint)
    run_experiment.py    # naive/ours/mas/joint × seeds quick driver
    run_benchmark.py     # full grid driver {encoder×depth×dataset×method×seed} → CSV
    run_baselines.py     # classic-baseline driver
    metrics.py, plots.py # metrics, figures
  tests/smoke_test.py    # end-to-end smoke on synthetic features (no downloads needed)
  cache/ data/ results/  # gitignored artifacts (features, datasets, run JSONs/CSVs)
```

## Local setup (dev machine)

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision "numpy<2" matplotlib open_clip_torch certifi
cd crcl/tests && ../../.venv/bin/python smoke_test.py   # must print SMOKE TEST PASSED
```

## Server workflow (all real training runs on the GPU)

Code flows **only through git** — never edit on the server without committing back.

```bash
# on the server (first time)
git clone https://github.com/btknzn-tum/Continual_learning.git && cd Continual_learning
pip install torch torchvision "numpy<2" matplotlib open_clip_torch
python crcl/tests/smoke_test.py

# every session: match commits, run inside tmux, one chain at a time
git pull
tmux new -s bench
python crcl/src/cache_features.py --dataset cifar100 --backbone resnet50
python crcl/src/run_benchmark.py --dataset cifar100 --backbone resnet50 \
    --depths layer1 layer2 layer3 layer4 --full

# pull results back to the laptop (scp; results/ is gitignored)
scp -i ~/.ssh/id_ed25519_new -P <PORT> -r root@<HOST>:~/Continual_learning/crcl/results ./crcl/
```

## GPU Connection Information

Vast.ai rented instance — **NVIDIA RTX 3090**.

### SSH key

- Private key: `~/.ssh/id_ed25519_new` (keep secret, never share)
- Public key: `~/.ssh/id_ed25519_new.pub` (paste into Vast.ai → Account → SSH Keys)

Public key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICH26FEBkCtTBCX8R50WillmVTHtvfIvI4zgLmlec7wM ozenbatukaan@gmail.com
```

### Connect

Use either command (same machine). Always pass the key with `-i`:

```bash
ssh -i ~/.ssh/id_ed25519_new -p 25372 root@154.64.230.67 -L 8080:localhost:8080
```

or via the Vast.ai proxy host:

```bash
ssh -i ~/.ssh/id_ed25519_new -p 20417 root@ssh4.vast.ai -L 8080:localhost:8080
```

> Note: port/host change every time you rent a new instance. Grab the current
> values from the Vast.ai **Connect** button and keep the `-i ~/.ssh/id_ed25519_new` part.
