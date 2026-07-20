# SSAR-LoRA: Activation-Gated Shared LoRA with Weight Regularization

## Goal
Beat the frozen-feature ceiling / EWC-LoRA by combining **activation-based
capacity allocation (SSAR)** with **EWC-LoRA weight regularization**, on the
EWC-LoRA paper's exact pipeline (ViT-B/16-21k, LoRA on attention K/V, class-IL).

## Core method
A single SHARED LoRA per attention layer (K and V), as in EWC-LoRA. Each task
writes its update into a disjoint slice of channels chosen by activation, and
an accumulated-Fisher penalty protects earlier slices against low-rank leakage.

**Why both parts are needed (the key argument):** LoRA is low-rank
(ΔW = B·A, shared factors), so gradient-masking a task to its own output
channels does NOT perfectly isolate it — updates leak through the rank
bottleneck into earlier channels. The EWC penalty on ΔW closes that leak.
Structural gate (activation) + functional protection (EWC) are complementary.

### Per task t (know total T in advance)
1. **Activation profile** (on task t's train/val data, current model): mean
   |output activation| per K/V channel, per layer.
2. **Slice selection from the FREE pool:** channels not yet claimed by tasks
   1..t-1. Pick the top-active `budget = round(1/T * dim)` channels among the
   free pool, **equally per layer** (each layer contributes `budget` channels
   for K and V). Collision-free by construction.
3. **Gated training:** train `lora_new_B/A` with EWC-LoRA loss, but mask
   `lora_new_B` row-gradients so only the selected channels update
   (`grad *= channel_mask[:, None]`). EWC penalty term unchanged.
4. **Claim:** add the selected channels to the used set; `accumulate_and_reset`
   folds new LoRA into accumulated (as in EWC-LoRA).

### Test time
Shared accumulated LoRA, no task ID needed -> class-IL, directly comparable to
EWC-LoRA.

## Integration points (repo: low-rank-cl, forked)
- `models/vit_ewclora.py::Attention_LoRA` -> `vit_ssarlora.py`: add per-layer
  `chan_mask_k/v` buffers (dim,), apply row-mask to `lora_new_B_*` gradients
  via hook or in-training; expose activation hooks to read K/V magnitudes.
- `methods/ewclora.py` -> `methods/ssarlora.py`: add `profile_activations()`,
  `select_slice()` (free-pool top-active), `claim()`, set masks before
  `_train_function`, apply grad mask in the loop. Keep the EWC Fisher path.
- `models/net_ewclora.py` -> `net_ssarlora.py`: pass masks through.
- `utils/factory.py`: register `ssarlora`.
- `configs/ssarlora/cifar10.json`: init_cls=2, increment=2, sessions=5,
  rank=10, epochs=20, gamma=0.9, lambda tuned; add `budget_frac`.

## Experiments (start CIFAR-10, then scale)
- **E1 Split CIFAR-10** (5 tasks x 2 cls, ViT-B/16-21k, class-IL, seed 0 first):
  SSAR-LoRA vs EWC-LoRA vs Vanilla LoRA (their code, same config).
- **E2 CIFAR-100** (10 tasks): compare to their reported EWC-LoRA 87.91.
- Ablations: gating on/off (=EWC-LoRA), EWC on/off (=pure PackNet-LoRA),
  budget_frac in {1/T, 0.5/T, 2/T}, activation- vs random-slice.
- Metrics: last-task acc (A_T), average acc, forgetting. Seeds after CIFAR-10
  sanity: {0,1,2}.

## Honesty protocol
- Slice/HP selection on validation (or fixed 1/T budget — no test tuning).
- Report EWC-LoRA baseline reproduced in our environment (not just their
  paper number) for a fair same-machine comparison.
- If it does NOT beat EWC-LoRA, report honestly + ablate why.

## Success criterion
SSAR-LoRA > EWC-LoRA (same config, same machine) on CIFAR-10, then CIFAR-100.
Stretch: the gating lets a SMALLER rank match EWC-LoRA (capacity efficiency).

## Status of prior work
The insertion-depth placement study (paper/) is complete and set aside; its
result tables are kept. The failed selective-plasticity code
(train_encoder_cl.py + enc* scripts) is removed for a clean start.
