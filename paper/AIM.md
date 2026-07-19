# Paper Aim — determined FROM the data (2026-07-20, snapshot 119 tags)

## Working title
**Where to Protect: Insertion Depth Determines the Fate of Importance-Protected
Adapters in Continual Learning**

## Target venue
TMLR (primary; not novelty-gated, values rigor + claims-match-evidence).
Fallbacks: Neural Networks (Q1), CoLLAs 2027. Template: tmlr.sty (fetched).

## The story the data actually tells (evidence-mapped)

**Aim 1 — Placement is a first-order, method-agnostic factor.**
- ResNet-50 depth map (5-seed): naive/MAS/EWC at layer1..4, layer3+4 best
  (C100 t10: MAS 90.17 vs 89.61 layer4-only).
- IT TRANSFERS TO SOTA: applying our multi-tap placement to RanPAC improves it
  on every stream: 5-Datasets 82.69 -> 87.28 (+4.6!), C100 t20 95.88 -> 96.15,
  C10 98.30 -> 98.42. => We IMPROVE published-SOTA-family numbers with a
  placement change only. Headline result.

**Aim 2 — The optimal placement is encoder-dependent.**
- ResNet (supervised CNN): late-pair multi-tap wins (layer3+4 > layer4).
- CLIP ViT-B/32: final block dominates; adding early blocks HURTS
  (final 94.05 vs block3+final 92.78 on C100). Interaction = a finding.

**Aim 3 — Protection is most critical at early insertion depths.**
- Naive collapse grows as taps move earlier, on BOTH encoders:
  ResNet layer1 F=31.8 vs layer4 F=19.4; CLIP block3 F=35.5 vs final F=5.8.
- MAS/EWC restore near-upper-bound performance at every depth
  (e.g., CLIP block3: 41.97 -> 67.81, +26).

**Aim 4 — Honest positioning of protected adapters vs SOTA.**
- Within adapter family: MAS/EWC-adapter reaches ~99% of its own joint upper
  bound (t20: 94.47 vs UB 95.14) with F ~1; naive loses 15-22 points.
- vs RanPAC: tied at ceiling (C10, MNIST), behind on C100-t20/5-Datasets —
  and the gap is a CLASSIFIER-CAPACITY gap, not a forgetting gap (RanPAC 96.15
  exceeds our joint UB 95.14). State this plainly; it motivates Aim 5.
- Capacity-matched controls (h354/h1024/LoRA): gains are placement+protection,
  not parameter count.

**Aim 5 (conditional; tonight's runs) — Beyond frozen features: selective
encoder plasticity.**
- MAS-adapter + bottom-q% encoder params trainable + similarity anchor.
- Bar to clear (set by our own Aim-1 result): 5-Datasets 87.28 (multi-tap
  RanPAC), t20 96.15. Cached-MAS baseline to beat first: 79.54 (fivedata l3+4).
- If it clears/approaches the bar -> method section + headline shift.
  If not -> ablation-style section "how far can frozen features go"; paper
  stands on Aims 1-4.

## Claims we must NOT make
- No unqualified SOTA claims (RanPAC ties/beats us on raw acc off-ceiling).
- No "we invented importance-protected adapters" (EWC-LoRA ICLR'26, EKPC,
  Online-LoRA exist; we cite and differentiate: placement axis + controls).
- 5-Datasets = Ebrahimi et al. stream with KMNIST swap — say so.

## Table plan
- T1 main: {ResNet, CLIP} x {best single tap, best multi-tap} x
  {naive, EWC, MAS, NCM, RanPAC(+our placement), joint} x {C10, C100-t10}.
- T2 long/heterogeneous: C100-t20 + 5-Datasets, same columns.
- T3 depth map: per-depth naive/MAS/EWC (ResNet 5 depths, CLIP 4 blocks).
- T4 placement transfer: RanPAC single-tap vs multi-tap on all streams.
- T5 capacity: h256/h354/h1024/LoRA at fixed placement.
- T6 (cond.): selective encoder plasticity + ablations (sim, alpha_enc, q).
- Fig 1: depth-vs-forgetting curves (both encoders). Fig 2: placement heatmap
  (MAS over tap combos). Fig 3: stability-plasticity scatter.

## Stats protocol for every claimed delta
5 seeds {42,123,456,789,1337}; Welch t-test + Bonferroni; mark n.s. deltas.
Single-seed cells (SOTA CLIP, capacity) -> promote to 5 seeds before claiming.

## Pending experiment queue feeding the paper
clip 5-seed grid (running) · CLIP SOTA (launched) · compare/phase2 (capacity,
LoRA, C100 SOTA 5-seed) · encmethod seed-42 -> 5-seed if good ·
[decision-gated] class-IL + per-task baseline.
