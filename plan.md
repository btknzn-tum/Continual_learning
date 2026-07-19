# PLAN — Capacity-Reserving Continual Learning in Adapters (CRCL)
## Phase 1 demonstration: Split CIFAR-10, frozen ResNet-18 features, laptop-only

**Goal of this phase:** show the full method working end-to-end on CIFAR-10 (5 tasks × 2 classes) with strong numbers: near-zero forgetting, plasticity close to the joint upper bound, clearly beating the naive sequential baseline. CIFAR-100/10-task and the full ablation grid (old M7) move to Phase 2; a CLIP backbone is Phase 3.

---

## 1. METHOD (exact spec — what the code implements)

Setup: a **frozen pretrained ResNet-18** (ImageNet weights) maps images to 512-d features. Features are computed **once** and cached to disk (`cache/*.pt`) — after that, training never touches the backbone or raw images. On top: a trainable **adapter** (2-layer MLP, 256 hidden units per layer, ReLU, no BatchNorm) + one small linear head per task (task-incremental: task id known at test time).

Three ingredients:

### 1.1 Reserve loss (manufactured dormancy)
At init, randomly designate a fraction `q = 0.30` of hidden units in each layer as the reserved set **R** (boolean mask per layer). During task-1 training add
`L_res = (1/|R|) * Σ_{u∈R} mean_batch(relu_out_u)` — an L1 penalty on reserved units' post-ReLU activations. Objective: `L = CE + beta_res * L_res`. Result: R becomes verifiably dormant (mean activation < tau=1e-3) → spare capacity deliberately set aside for future tasks.

### 1.2 Importance score — the DP path value × mean activation (core novelty)
For every adapter weight, `S(w) = SF(w) * phi(u_source)`:

- **`SF(w)` — data-free path importance (SynFlow, "the DP from output back to input"):** clone the adapter + previous-task heads, replace all weights by |w|, feed all-ones input, sum the outputs into scalar `R`, backprop once; `SF(w) = |w| * |∂R/∂w|`. With all-positive weights and all-ones input, ReLUs are identity, so this equals the sum over all input→output paths through w of products of |w| — exactly the backward DP over connections. Computed on a clone; per-layer normalized `SF ← SF/SF.max()`.
- **`phi(u)` — mean activation on PREVIOUS tasks' data:** for each hidden unit, mean post-ReLU activation over a loader of all previous tasks' cached features (≤2000 samples/task). Per-layer normalized. This is the "which neurons are actually active for the old classes" factor.
- **Combination:** for W2 rows×cols `S[v,u] = SF(W2)[v,u] * phi_h1[u]` (source-unit activation gates the weight's use); for W1, `S = SF(W1)` (source is raw input). Baseline scores (`sf`, `phi`, `wanda`, `taylor`, `mas`) behind the same `importance(model, loader, method=...)` interface for later ablations.

### 1.3 Protected training of new tasks (small + sparse delta)
For task t ≥ 2, with `theta_old` = frozen snapshot after task t−1:

```
L = CE(task t)
  + alpha    * Σ_w S_w (w − w_old)²                          # don't move important (old-task) weights
  + gamma    * relu(‖delta‖_F / ‖theta_old‖_F − beta_crit)²  # keep total change small
  + beta_res * L_res(R_unclaimed)                            # re-reserve remaining dormant units
```

Because `S` is near-zero exactly on the dormant units' weights, gradient flows freely into spare capacity while old-task circuitry is pinned — "change the inactive neurons, with small activation change elsewhere." After task t: units in R whose phi on task-t data > `tau_claim = 0.05` are marked claimed by task t; `R_unclaimed` shrinks. Previous-task heads: `requires_grad=False` from the moment their task ends.

Defaults: `q=0.30, beta_res=0.01, alpha=1.0 (sweep {0.1,1,10,100} once), gamma=100, beta_crit=0.10, tau_claim=0.05`, AdamW lr=1e-3, 20 epochs/task, batch 128, seeds {42, 123, 456}.

---

## 2. HARDWARE STRATEGY (this laptop: Intel i5, 4 cores, 16 GB, no GPU)

- **Backbone runs exactly once.** `cache_features.py` pushes all of CIFAR-10 (train+test) through ResNet-18 in `eval()` + `no_grad()`, saves `{"features": Float[N,512], "labels": Long[N]}` to `cache/`. ~30–60 min once, on CPU. All experiments afterwards read the cache; a full 5-task CL run is then **~1–2 min on CPU** (tiny MLP on 512-d vectors).
- No augmentation in the cache pass (features must be deterministic), transform = resize 224 → center crop 224 → ImageNet normalize.
- torch 2.2.2 CPU wheels (last PyTorch supporting Intel macOS), `numpy<2`.
- **CLIP later? Yes, same trick, even better.** The backbone is behind a `get_backbone(name)` switch in `cache_features.py`; adding `open_clip` ViT-B/32 (512-d output — adapter unchanged) is a ~10-line addition, one more caching pass, everything downstream identical. Phase 3.

---

## 3. EXPERIMENT — Split CIFAR-10, 5 tasks × 2 classes

Task splits: classes {0,1}, {2,3}, {4,5}, {6,7}, {8,9} (fixed order; seed only affects init/batching). Labels remapped to {0,1} within each task. After each task t, evaluate on all tasks 1..t → accuracy matrix `A[t, 1..t]`.

Runs (all share the same trainer, config-switched):

| Run | What | Purpose |
|-----|------|---------|
| **ours** | reserve + SF×phi protection (full method) | the result |
| **naive** | alpha=gamma=beta_res=0 | lower bound, shows forgetting exists |
| **joint** | all tasks pooled, one run | upper bound |

Metrics (from `A[T,T]`, mean±std over 3 seeds):
- `AvgAcc = mean_t A[T,t]` — final average accuracy
- `Forgetting = mean_{t<T} (max_k A[k,t] − A[T,t])`
- `Plasticity = mean_t A[t,t]`

**Success criteria for this phase:** ours Forgetting < 1pp; ours AvgAcc within ~1pp of joint; naive shows clearly worse Forgetting; task-1 accuracy with reserve loss within 1pp of without (reserve is free); ≥90% of R verified dormant after task 1 (zero-out test changes task-1 acc ≤ 0.1pp).

---

## 4. REPO LAYOUT

```
crcl/
  cache/    data/    results/          # gitignored artifacts
  src/
    common.py            # set_seed, device, config
    cache_features.py    # backbone → disk (resnet18 now, clip later)
    tasks.py             # class splits + loaders over cached tensors
    adapter.py           # Adapter (2×256 MLP) + per-task heads
    reserve.py           # reserved set R, L_res, verify_dormant, claiming
    importance.py        # SynFlow SF, phi, S=SF×phi (+ sf/phi/wanda/taylor/mas)
    train_cl.py          # run_sequence(config): ours / naive / joint
    metrics.py           # AvgAcc, Forgetting, Plasticity, JSON logs
    run_experiment.py    # runs ours+naive+joint × seeds, writes summary
```

Every run writes `results/<name>/seed<k>.json` with config + full accuracy matrix. No result exists unless it is in a JSON file.

---

## 5. PITFALLS CHECKLIST

- Cache WITHOUT augmentation, `model.eval()`, `no_grad()` — else every downstream number is noise.
- Previous-task heads frozen (`requires_grad=False`) the moment their task ends — a leaking head silently destroys the forgetting measurement.
- SynFlow on a CLONE with abs weights; never touch the real model.
- `phi` measured on PREVIOUS tasks' data only — the score protects the past, not the present.
- Per-layer normalize SF and phi before multiplying (raw scales differ by orders of magnitude).
- All-ones SynFlow input is exact only because the adapter is ReLU-only, no BatchNorm — do not add BatchNorm.
- Report mean±std over the 3 fixed seeds, never a single seed.

---

## 6. LATER PHASES (not now)

- **Phase 2:** CIFAR-100/10-task, full ablation grid A1–A10 (importance-method comparison sfxphi vs sf/phi/wanda/taylor/mas, hard-mask variants, EWC/MAS/PackNet/LoRA-seq baselines), 5 seeds, Welch t-tests — the H1/H2 go/no-go gate from the original plan.
- **Phase 3:** CLIP ViT-B/32 backbone (cached the same way), TinyImageNet / ImageNet-R.
