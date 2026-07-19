# PLAN — MAS-Protected Adapters for Continual Learning:
## A Systematic Benchmark across Insertion Depths and Vision Encoders (Q1-journal target)

**Working title:** *"Where to Adapt and How to Protect: A Systematic Study of
Importance-Regularized Adapters for Continual Learning across Encoder Depths."*

**Target venues (Q1):** Neural Networks → Pattern Recognition → TNNLS (in that order;
all value rigorous empirical studies with reproducible code).

---

## 0. WHAT WE KNOW (Phase-1, laptop, completed — the pivot record)

Phase-1 falsified our original novel framework (reserve loss + SF×φ + norm cap) and
identified the winner. Split CIFAR-10, frozen ResNet-18 final features, adapter, 3 seeds:

| Config | AvgAcc | Forgetting |
|--------|--------|-----------|
| naive | 89.40 ± 1.54 | 10.12 ± 1.87 |
| EWC (pure penalty) | 94.33 ± 1.89 | 3.75 ± 2.21 |
| ours-framework (reserve+SF×φ+cap) | 95.31 ± 0.78 | 1.40 ± 1.14 |
| SF×φ pure | 96.09 ± 0.05 | 0.56 ± 0.07 |
| **MAS pure** | **96.92 ± 0.24** | **0.49 ± 0.25** | 
| joint (upper bound) | 97.25 ± 0.16 | 0.00 |

Findings: (1) the extra "framework" hurts every signal — pure importance-weighted
quadratic penalty is the right mechanism; (2) **MAS is the best signal** (direct output
sensitivity, unlabeled, does not vanish at task minima like EWC's Fisher → best mean AND
lowest variance). MNIST agreed. Split-MNIST full results also archived in `crcl/results/`.

**Paper thesis:** for frozen-encoder + adapter continual learning, MAS-protected adapters
are a strong, stable, rehearsal-free default — and *where* the adapter attaches (encoder
depth, single or multi-tap) matters as much as *which* method protects it. Nobody has
mapped this (method × placement × encoder) space systematically. That map is the paper.

---

## 1. SYSTEM UNDER STUDY

Frozen encoder → cached features at chosen **insertion point(s)** → trainable **adapter**
(2-layer ReLU MLP, d_hidden=256, no BatchNorm) → per-task linear heads (task-IL).
Multi-tap placement = concatenate the cached features of the selected depths (d_in = Σ dims).

Primary method — **MAS-adapter** (`reg:mas`): for task t ≥ 2,
`L = CE + α · Σ_w S_w (w − w_old)²` with `S = mean |∂‖logits_prev‖²/∂w|` computed on
≤2000 samples/previous task, per-tensor normalized; previous heads frozen. No reserve
loss, no norm cap (Phase-1 showed they hurt).

## 2. BASELINES (4 realistic competitors + 2 bounds, all in the SAME adapter/trainer)

| Method | Type | Why it must be in a Q1 benchmark |
|--------|------|--------------------------------|
| **EWC** | quadratic penalty, Fisher | the canonical regularization baseline |
| **SI** | quadratic penalty, online path-integral importance | importance accumulated DURING training — philosophical opposite of MAS's post-hoc measurement |
| **LwF** | functional regularization (distill previous heads' logits on current-task data) | the standard data-free distillation representative |
| **ER** | experience replay, small buffer (20 samples/class) | reviewers always demand a rehearsal reference; small-buffer ER is the honest one |
| naive | α=0 lower bound | quantifies the problem |
| joint | pooled upper bound | quantifies the gap |

Phase-1 signals (SF×φ, wanda, taylor, L2-SP) move to an appendix ablation.
To implement: **SI, LwF, ER** (~150 lines total in the existing trainer); EWC exists.

## 3. BENCHMARK GRID

### 3.1 Datasets (added incrementally, easy → hard)
1. **Split-MNIST** (5×2, pixels) — sanity; done on laptop, re-verify on server.
2. **Split CIFAR-10** (5×2) — done for ResNet-18-final; extend to all placements/encoders.
3. **Split CIFAR-100 (10×10)** — the core field-standard table.
4. **Split CIFAR-100 (20×5)** — long-stream stress test.
5. **TinyImageNet (10×20)** — scale + harder features (stretch goal, same caching flow).

### 3.2 Encoders (frozen, cached once per dataset on the GPU server)
- **ResNet-50** (ImageNet-V2): taps = GAP(layer1 256-d, layer2 512-d, layer3 1024-d, layer4 2048-d).
- **CLIP ViT-B/32** (open_clip, LAION-2B): taps = CLS(block3, block6, block9) 768-d + final proj 512-d.
- ResNet-18 stays as the laptop dev/debug backbone only.

### 3.3 Adapter placement — "every combination"
Per encoder: 4 taps → **15 non-empty subsets** (4 single + 11 multi-tap concats).
- **Placement study (MAS only):** all 15 subsets × both encoders × CIFAR-100-10 × 5 seeds
  → the paper's placement heatmap/curve figure. (~150 GPU-cheap runs.)
- **Method benchmark:** all 6 methods × {4 single taps + best multi-tap combo} × both
  encoders × {CIFAR-10, CIFAR-100-10, CIFAR-100-20} × 5 seeds → the main tables.
- MNIST runs only at "pixels" (no encoder).

### 3.4 Protocol rigor (Q1 hygiene)
- Seeds {42,123,456,789,1337}; report mean±std; Welch t-tests + Bonferroni for headline claims.
- α (and ER buffer, LwF temperature, SI c) tuned ONCE per (encoder, dataset) on seed 42
  with a 5-point sweep; frozen for the 5-seed runs; sweeps reported in appendix.
- Metrics from A[k,t]: AvgAcc, Forgetting, Plasticity (learning acc), BWT.
- One JSON per run (config embedded); aggregate CSVs; every table/figure generated by
  script from CSVs (`stats.py`, `plots.py`) — no hand-copied numbers.

## 4. INFRASTRUCTURE — GPU server + GitHub sync (single source of truth)

- **Server:** Vast.ai RTX 3090 (connection details in README.md; host/port change per
  rental). PyTorch + CUDA + open_clip on the server.
- **Sync rule: code flows ONLY through git.** Local (this laptop) = development + smoke
  tests; server = all real training. Never edit code directly on the server; if a hotfix
  happens there, commit+push from the server immediately. Every results file records
  `git rev-parse HEAD`.
- **Repo:** https://github.com/btknzn-tum/Continual_learning.git (credentials supplied
  out-of-band; NEVER committed to files).
- **Run discipline:** one experiment chain at a time inside `tmux` on the server
  (`scripts/run_*.sh`), stdout → `logs/`, results → `results/` (gitignored), pulled back
  to the laptop with `scripts/sync_results.sh` (scp). Caching on the 3090 is minutes per
  (dataset × encoder), vs ~1 h on the laptop.

## 5. EXECUTION ORDER (server milestones, each = one tmux run, verified before the next)

- **S0** — Git bootstrap: repo pushed from laptop; server clones; `nvidia-smi` + smoke test
  (synthetic features) passes on GPU. ✅ gate: same commit hash both sides.
- **S1** — Implement SI, LwF, ER in the trainer (+ unit smoke on synthetic). Laptop, push.
- **S2** — Server caches: CIFAR-10 + CIFAR-100 × {ResNet-50 all taps, CLIP all taps}.
  ✅ gate: probe accuracy on task-0 ≥ 80% per (encoder, final tap).
- **S3** — Hyperparameter sweeps (seed 42) per (encoder, dataset): α for MAS/EWC/SI,
  LwF T, ER buffer. Produces frozen config table.
- **S4** — Method benchmark tier 1: CIFAR-100-10, both encoders, 5 placements × 6 methods
  × 5 seeds. → main table + first figures.
- **S5** — Placement study: MAS × all 15 subsets × both encoders × CIFAR-100-10 × 5 seeds.
  → placement figure.
- **S6** — Tier 2: CIFAR-10 (all placements) + CIFAR-100-20 (best placement) + MNIST re-run.
- **S7** — (stretch) TinyImageNet tier.
- **S8** — `stats.py` significance tables, `plots.py` final figures, results frozen.
- **S9** — **Paper writing** (LaTeX, venue template): Intro / Related work (adapter CL,
  regularization CL, MAS/EWC/SI/LwF/ER, depth studies) / Method / Benchmark setup /
  Results (main table, placement map, stability analysis EWC-vs-MAS variance, honest
  negative: our framework ablation) / Discussion / Reproducibility statement (repo+seeds).

## 6. PITFALLS CHECKLIST (unchanged core + server additions)

- Cache with `eval()`, `no_grad()`, NO augmentation; deterministic.
- Previous heads frozen at task end; importance data = previous tasks only.
- ER buffer sampled class-balanced from previous tasks ONLY; buffer size reported.
- α/hyperparams never tuned per-seed. One JSON per run or it didn't happen.
- Server runs record commit hash; never run uncommitted code.
- CLIP uses open_clip's own preprocess; ResNet uses ImageNet transform.
- Multi-tap concat: per-tap feature standardization (z-score from train stats) before
  concat so no tap dominates by scale — store the stats with the cache.
