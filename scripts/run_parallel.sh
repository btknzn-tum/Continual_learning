#!/usr/bin/env bash
# Massively parallel benchmark runner for a many-core server (128 cores).
# 16 concurrent jobs x 8 threads each. Run inside tmux:
#   tmux new -s par "source /venv/main/bin/activate && bash scripts/run_parallel.sh"
# CSV appends are flock-protected in run_benchmark.py; result tags include
# n_tasks so no two jobs write the same path.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="python"
SRC="crcl/src"
TUNED_R50="crcl/results/tuned_cifar100_resnet50_layer4.json"
TUNED_CLIP="crcl/results/tuned_cifar100_clip_vitb32_final.json"

job() {  # job <name> <cmd...>
  local name="$1"; shift
  (
    echo "[PAR] $(date '+%F %T') START $name (commit $(git rev-parse --short HEAD))"
    if "$@" > "logs/par_${name}.log" 2>&1; then
      echo "[PAR] $(date '+%F %T') DONE  $name"
    else
      echo "[PAR] $(date '+%F %T') FAIL  $name (logs/par_${name}.log)"
    fi
  ) &
}

wait_for_clip() {
  until grep -q "saved .*tuned_cifar100_clip_vitb32_final.json" logs/clipfix.log 2>/dev/null; do
    sleep 60
  done
}

# ---------------- ResNet-50 side (caches ready) ----------------
# Tier-1 CIFAR-100/10: one job per depth (7 methods x 5 seeds each)
job c100r50_l1   $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer1 --tuned-file $TUNED_R50 --full
job c100r50_l2   $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer2 --tuned-file $TUNED_R50 --full
job c100r50_l3   $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer3 --tuned-file $TUNED_R50 --full
job c100r50_l4   $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer4 --tuned-file $TUNED_R50 --full
job c100r50_c34  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths "layer3+layer4" --tuned-file $TUNED_R50 --full

# CIFAR-10 tier — one job per depth for max parallelism
job c10r50_l1    $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 --depths layer1 --tuned-file $TUNED_R50 --full
job c10r50_l2    $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 --depths layer2 --tuned-file $TUNED_R50 --full
job c10r50_l3    $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 --depths layer3 --tuned-file $TUNED_R50 --full
job c10r50_l4    $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 --depths layer4 --tuned-file $TUNED_R50 --full
job c10r50_c34   $PY $SRC/run_benchmark.py --dataset cifar10 --backbone resnet50 --depths "layer3+layer4" --tuned-file $TUNED_R50 --full

# long stream + MNIST + 5-Datasets (r50, split by depth — heaviest runs)
job t20_r50      $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer4 --n-tasks 20 --tuned-file $TUNED_R50 --full
job mnist        $PY $SRC/run_benchmark.py --dataset mnist --backbone pixels --depths - --full
job fd_r50_l4    $PY $SRC/run_benchmark.py --dataset fivedata --backbone resnet50 --depths layer4 --tuned-file $TUNED_R50 --full
job fd_r50_c34   $PY $SRC/run_benchmark.py --dataset fivedata --backbone resnet50 --depths "layer3+layer4" --tuned-file $TUNED_R50 --full

# Placement study (MAS only) — the 11 multi-tap combos (singles come from tier-1)
job place_r50_a  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --methods mas \
  --depths "layer1+layer2" "layer1+layer3" "layer1+layer4" "layer2+layer3" "layer2+layer4" --tuned-file $TUNED_R50 --full
job place_r50_b  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone resnet50 --methods mas \
  --depths "layer1+layer2+layer3" "layer1+layer2+layer4" "layer1+layer3+layer4" "layer2+layer3+layer4" "layer1+layer2+layer3+layer4" --tuned-file $TUNED_R50 --full

# ---------------- CLIP side (waits for clipfix re-cache + sweep) ----------------
( wait_for_clip
  job c100clip_a  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 --depths block3 block6 --tuned-file $TUNED_CLIP --full
  job c100clip_b  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 --depths block9 final "block9+final" --tuned-file $TUNED_CLIP --full
  job c10clip     $PY $SRC/run_benchmark.py --dataset cifar10 --backbone clip_vitb32 --depths block3 block6 block9 final "block9+final" --tuned-file $TUNED_CLIP --full
  job fd_clip     $PY $SRC/run_benchmark.py --dataset fivedata --backbone clip_vitb32 --depths final "block9+final" --tuned-file $TUNED_CLIP --full
  job place_clip  $PY $SRC/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 --methods mas \
    --depths "block3+block6" "block3+block9" "block3+final" "block6+block9" "block6+final" \
    "block3+block6+block9" "block3+block6+final" "block3+block9+final" "block6+block9+final" \
    "block3+block6+block9+final" --tuned-file $TUNED_CLIP --full
  wait
) &

wait
echo "[PAR] $(date '+%F %T') ALL PARALLEL JOBS FINISHED"
