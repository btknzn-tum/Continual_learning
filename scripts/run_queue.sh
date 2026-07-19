#!/usr/bin/env bash
# Full experiment queue for the GPU server. Run inside tmux:
#   tmux new -s bench
#   bash scripts/run_queue.sh 2>&1 | tee -a logs/queue.log
# Steps run sequentially; a failed step is logged and the queue continues.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY="python"
SRC="crcl/src"

step() {
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo "[QUEUE] $(date '+%F %T') START $name  (commit $(git rev-parse --short HEAD))"
  echo "=================================================================="
  if "$@" >> "logs/${name}.log" 2>&1; then
    echo "[QUEUE] $(date '+%F %T') DONE  $name"
  else
    echo "[QUEUE] $(date '+%F %T') FAIL  $name (see logs/${name}.log)"
  fi
}

# ---- S0: smoke ---------------------------------------------------------------
step smoke $PY crcl/tests/smoke_test.py

# ---- S2: caching (GPU) -------------------------------------------------------
step cache_mnist        $PY $SRC/cache_pixels.py
step cache_c10_r50      $PY $SRC/cache_features.py --dataset cifar10  --backbone resnet50    --batch-size 256
step cache_c100_r50     $PY $SRC/cache_features.py --dataset cifar100 --backbone resnet50    --batch-size 256
step cache_c10_clip     $PY $SRC/cache_features.py --dataset cifar10  --backbone clip_vitb32 --batch-size 256
step cache_c100_clip    $PY $SRC/cache_features.py --dataset cifar100 --backbone clip_vitb32 --batch-size 256
# 5-Datasets (CIFAR10/MNIST/SVHN/Fashion/KMNIST — heterogeneous stream)
step cache_5d_r50       $PY $SRC/cache_features.py --dataset fivedata --backbone resnet50    --batch-size 256
step cache_5d_clip      $PY $SRC/cache_features.py --dataset fivedata --backbone clip_vitb32 --batch-size 256

# ---- S3: hyperparameter sweeps (seed 42) on the CORE dataset -----------------
step sweep_c100_r50     $PY $SRC/run_sweep.py --dataset cifar100 --backbone resnet50_layer4
step sweep_c100_clip    $PY $SRC/run_sweep.py --dataset cifar100 --backbone clip_vitb32_final

# ---- S4: method benchmark tier 1 — CIFAR-100/10 tasks, 5 seeds ---------------
R50_DEPTHS=(layer1 layer2 layer3 layer4 "layer3+layer4")
CLIP_DEPTHS=(block3 block6 block9 final "block9+final")

step bench_c100_r50  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 \
  --depths "${R50_DEPTHS[@]}" --tuned-file crcl/results/tuned_cifar100_resnet50_layer4.json --full
step bench_c100_clip $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 \
  --depths "${CLIP_DEPTHS[@]}" --tuned-file crcl/results/tuned_cifar100_clip_vitb32_final.json --full

# ---- S5: placement study — MAS only, ALL 15 depth subsets --------------------
R50_ALL=(layer1 layer2 layer3 layer4 "layer1+layer2" "layer1+layer3" "layer1+layer4" \
  "layer2+layer3" "layer2+layer4" "layer3+layer4" "layer1+layer2+layer3" \
  "layer1+layer2+layer4" "layer1+layer3+layer4" "layer2+layer3+layer4" \
  "layer1+layer2+layer3+layer4")
CLIP_ALL=(block3 block6 block9 final "block3+block6" "block3+block9" "block3+final" \
  "block6+block9" "block6+final" "block9+final" "block3+block6+block9" \
  "block3+block6+final" "block3+block9+final" "block6+block9+final" \
  "block3+block6+block9+final")

step place_c100_r50  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 \
  --depths "${R50_ALL[@]}" --methods mas \
  --tuned-file crcl/results/tuned_cifar100_resnet50_layer4.json --full
step place_c100_clip $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 \
  --depths "${CLIP_ALL[@]}" --methods mas \
  --tuned-file crcl/results/tuned_cifar100_clip_vitb32_final.json --full

# ---- S6: tier 2 — CIFAR-10 all placements, CIFAR-100/20 tasks, MNIST ---------
step bench_c10_r50   $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 \
  --depths "${R50_DEPTHS[@]}" --tuned-file crcl/results/tuned_cifar100_resnet50_layer4.json --full
step bench_c10_clip  $PY $SRC/run_benchmark.py --dataset cifar10 --backbone clip_vitb32 \
  --depths "${CLIP_DEPTHS[@]}" --tuned-file crcl/results/tuned_cifar100_clip_vitb32_final.json --full
step bench_c100t20_r50  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 \
  --depths layer4 --n-tasks 20 --tuned-file crcl/results/tuned_cifar100_resnet50_layer4.json --full
step bench_c100t20_clip $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 \
  --depths final --n-tasks 20 --tuned-file crcl/results/tuned_cifar100_clip_vitb32_final.json --full
step bench_mnist     $PY $SRC/run_benchmark.py --dataset mnist --backbone pixels --depths - --full

# ---- S6b: 5-Datasets — heterogeneous-stream benchmark (max distribution shift) --
step bench_5d_r50    $PY $SRC/run_benchmark.py --dataset fivedata --backbone resnet50 \
  --depths layer4 "layer3+layer4" \
  --tuned-file crcl/results/tuned_cifar100_resnet50_layer4.json --full
step bench_5d_clip   $PY $SRC/run_benchmark.py --dataset fivedata --backbone clip_vitb32 \
  --depths final "block9+final" \
  --tuned-file crcl/results/tuned_cifar100_clip_vitb32_final.json --full

echo ""
echo "[QUEUE] $(date '+%F %T') ALL STEPS FINISHED"
