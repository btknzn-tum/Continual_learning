# PLAN — Regularization-Based Continual Learning in Adapters: A Benchmark Study
## Method: importance-weighted adapters (MAS as the strong default), across backbone depths and vision encoders

---

## 0. WHAT CHANGED AND WHY (pivot record — read first)

The original idea was a novel three-part method (reserve loss to manufacture dormant
capacity + SF×φ importance + a soft protection framework with a delta-norm cap).
Phase-1 experiments on this laptop **falsified the framework and the SF×φ signal as the
winner**, but produced a clean, defensible empirical result. The decisive numbers
(Split CIFAR-10, frozen ResNet-18 features, 3 seeds):

| Config | AvgAcc | Forgetting | Note |
|--------|--------|-----------|------|
| naive (no protection) | 89.40 ± 1.54 | 10.12 ± 1.87 | forgetting is real & large |
| ours (reserve + SF×φ + norm-cap) | 95.31 ± 0.78 | 1.40 ± 1.14 | the full fancy method |
| SF×φ **signal alone** (no reserve/cap) | 96.09 ± 0.05 | 0.56 ± 0.07 | framework was HURTING us |
| MAS + our framework | 96.53 ± 0.68 | 0.86 ± 0.80 | framework hurts MAS too |
| **MAS signal alone** | **96.92 ± 0.24** | **0.49 ± 0.25** | winner |
| EWC signal alone | 94.33 ± 1.89 | 3.75 ± 2.21 | unstable (Fisher vanishes at min) |
| joint (upper bound) | 97.25 ± 0.16 | 0.00 | — |

**Two findings drive the new plan:**
1. The reserve-loss + delta-norm "framework" is a **net negative** — it costs plasticity
   for both signals without buying enough stability. Drop it. Pure importance-weighted
   quadratic protection is what works.
2. Among pure importance signals, **MAS wins**: it measures output sensitivity directly
   (unlabeled, on previous-task data) and — unlike EWC's CE-gradient Fisher, which
   collapses at a task minimum → seed-unstable — MAS's `‖output‖²` gradient does not
   vanish, giving the best forgetting AND the tightest variance.

**The paper is therefore NOT "a new method."** "MAS on adapters" is a known combination
and would be rejected for novelty alone. The contribution is the **systematic benchmark**:
*where* trainable adapter capacity should attach in a frozen encoder, *which* importance
signal to protect it with, and *how this interacts with the encoder* (CNN vs. CLIP-ViT) —
delivered as a reproducible, multi-seed, multi-backbone, multi-depth study with MAS
established as the strong, stable default. SF×φ and the reserve loss survive as ablations
(honest negative results: "structural importance needs depth; manufactured dormancy does
not help soft regularization").

Working title: **"Where and How to Protect Adapters: A Benchmark of Regularization-Based
Continual Learning across Encoder Depths and Vision Backbones."**

---

## 1. METHOD (what the primary system is now)

Frozen vision encoder → cached features at a chosen **insertion depth** → trainable
**adapter** (2-layer ReLU MLP bottleneck, no BatchNorm) + one linear head per task
(task-incremental). Tasks arrive sequentially.

Protection for task t ≥ 2 (pure, no framework):
```
theta_old = adapter weights snapshot after task t-1
S = importance(model, previous-task data, method)    # MAS by default
L = CE(task t) + alpha * sum_w S_w * (w - w_old)^2
```
Previous-task heads frozen (`requires_grad=False`) the moment their task ends.

`importance(method=...)` supports the full comparison set:
- **mas** (default): `mean_batches |∂‖logits‖²/∂w|` on previous-task data, unlabeled.
- **ewc**: diagonal Fisher `mean (∂CE/∂w)²`.
- **sf**, **phi**, **sfxphi**: the original SynFlow / activation / product signals (ablation).
- **wanda**, **taylor**: pruning-derived signals (ablation).
- **l2**: uniform `S=1` (plain L2-SP — cheapest control).

Dropped from the primary path (kept only as ablation switches): reserve loss,
delta-norm regime cap, head-trim, unit claiming.

---

## 2. THE BENCHMARK AXES (this is where "many experiments" come from)

Full grid = {encoder} × {insertion depth} × {dataset} × {method} × {5 seeds}.

### 2.1 Encoders (backbone, frozen, cached once)
- **resnet18** (ImageNet) — CNN, the CVPR-era standard.
- **resnet50** (ImageNet) — depth/width scaling check (2048-d).
- **clip_vitb32** (open_clip, LAION) — vision-language encoder; tests whether the ranking
  holds for modern ViT features. 512-d image embedding, cached the same way.

### 2.2 Insertion depth — "adapters used piece-by-piece, not only at the end"
The core structural axis. Instead of only attaching the adapter to the FINAL pooled
feature, cache **intermediate** representations and attach the adapter there:
- **resnet**: global-average-pool the output of `layer1, layer2, layer3, layer4` →
  64-d / 128-d / 256-d / 512-d fixed vectors. Four depths per image, cached in one pass.
- **clip_vitb32**: the CLS token after transformer blocks `{3, 6, 9, 12}` (final) →
  four depths.
This directly answers "where should continual-learning capacity live?" and multiplies the
experiment count cheaply (one forward pass caches all depths; each downstream CL run is
seconds–minutes on CPU).

### 2.3 Datasets / task streams (proper benchmarking, not just a toy)
- **Split CIFAR-10**: 5 tasks × 2 classes (warm-up, already done for resnet18-final).
- **Split CIFAR-100**: **10 tasks × 10 classes** — the field-standard task-IL benchmark.
- **Split CIFAR-100 / 20 tasks × 5 classes** — longer stream, stresses capacity.
- (Phase 3, if time) TinyImageNet / ImageNet-R via the same caching flow.

### 2.4 Methods compared (all in the same adapter + trainer)
Primary: **mas**. Regularization baselines: ewc, l2-SP. Signal ablations: sfxphi, sf, phi,
wanda, taylor. Framework ablation: mas / sfxphi WITH the old reserve+cap (to show it hurts).
Bounds: naive (α=0), joint (pooled). Optional structural baselines: PackNet-in-adapter
(hard mask), LoRA-seq — added only if the regularization story needs a capacity-based
contrast.

---

## 3. METRICS & PROTOCOL (rigorous)

From accuracy matrix `A[k,t]` (acc on task t after training task k), per run:
- **AvgAcc** = mean_t A[T,t]
- **Forgetting** = mean_{t<T} (max_k A[k,t] − A[T,t])
- **Plasticity (learning acc)** = mean_t A[t,t]
- **BWT** = mean_{t<T} (A[T,t] − A[t,t])   (backward transfer, standard in the CL lit)

Report **mean ± std over 5 seeds {42,123,456,789,1337}**. For the headline claims
(MAS vs EWC, MAS vs SF×φ) run **Welch t-tests with Bonferroni correction** and report
corrected p-values. One JSON log per run (config embedded). CSV aggegate per grid slice.
α tuned once per (encoder, dataset) on seed 42 via the existing sweep, then frozen for the
5-seed runs (report the sweep in an appendix — no per-seed tuning).

**Decision rules (baked into the analysis script):**
- MAS is the recommended default iff it is within noise of the best method on AvgAcc AND
  best-or-tied on Forgetting AND lowest variance, across ≥ 2 encoders.
- Depth finding is reportable iff the best insertion depth is consistent (or consistently
  varies with encoder) across seeds with non-overlapping error bars.

---

## 4. HARDWARE STRATEGY (Intel i5, 4 cores, 16 GB, no GPU) — unchanged & central

- Every encoder runs **once**; all depths cached in that single pass to `cache/`.
  ResNet-18/50 and CLIP-ViT-B/32 all fit; caching CIFAR-100 ≈ ResNet-18 CIFAR-10 time.
- After caching, each CL run is a tiny MLP over cached vectors: **seconds to ~2 min on CPU**.
  The whole 5-seed × multi-method grid for one (encoder, depth, dataset) is minutes.
- torch 2.2.2 CPU wheels; `open_clip_torch` for CLIP (CPU inference is fine for caching).
- CLIP later already handled: `get_backbone("clip_vitb32")` returns encoder + preprocess;
  same cache format `{"features","labels"}`, adapter code unchanged (just d_in differs).

---

## 5. REPO CHANGES

```
crcl/src/
  cache_features.py   # + resnet50, + clip_vitb32, + multi-depth (layerN pooled) caching
  adapter.py          # unchanged (d_in parameterized)
  importance.py       # + "l2" (uniform); mas/ewc primary, others ablation
  train_cl.py         # primary path = pure importance penalty (method="mas_adapter" etc.)
  run_benchmark.py    # NEW: drives {encoder}×{depth}×{dataset}×{method}×{seed}, writes CSV
  stats.py            # NEW: Welch t-test + Bonferroni, aggregate CSV → results table
  plots.py            # + depth-vs-accuracy curves, per-encoder method bars, heatmaps
  reserve.py          # kept for the framework-ablation only
results/              # per-run JSON + aggregate CSVs + figures
```

---

## 6. EXECUTION ORDER (milestones, laptop-friendly)

- **B0** ✅ done: Split CIFAR-10 / resnet18-final — naive/ours/mas/ewc/joint + ablations, 3 seeds.
- **B1**: multi-depth caching for resnet18 on CIFAR-10 + CIFAR-100 (one pass each).
- **B2**: install open_clip; cache clip_vitb32 (all depths) on CIFAR-10 + CIFAR-100.
- **B3**: `run_benchmark.py` — full grid at 3 seeds first (fast triage), then promote the
  live slices to 5 seeds + stats.
- **B4**: resnet50 caching + grid (scaling check).
- **B5**: `stats.py` significance tables + `plots.py` figures (depth curves, method bars,
  encoder comparison, accuracy heatmaps). This + the CSVs = the paper's result package.

---

## 7. PITFALLS CHECKLIST

- Cache WITHOUT augmentation, `model.eval()`, `no_grad()`; deterministic features.
- Previous-task heads frozen the moment their task ends (a leaking head fakes low forgetting).
- `phi`/MAS/EWC importance measured on PREVIOUS tasks' data only.
- α tuned ONCE per (encoder, dataset) on seed 42, then frozen — no per-seed tuning.
- Report mean ± std over the 5 fixed seeds + corrected p-values; never a single seed.
- Depth features have different dims — set adapter `d_in` per depth; keep d_hidden fixed
  for comparability (note this in the writeup).
- CLIP uses its OWN preprocessing (open_clip transform), not ImageNet transform.
- One JSON per run; no result exists unless it is in a file.
