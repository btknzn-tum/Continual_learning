#!/bin/bash
# OUR METHOD: selective-plasticity encoder + MAS adapter + similarity anchor.
# bf16 autocast, batch 64, epochs 20 (matches cached baselines).
# Priority: 5-Datasets (beat RanPAC 82.69) -> C100 t10 -> C100 t20.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python crcl/src/train_encoder_cl.py"
$P --dataset fivedata --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
$P --dataset cifar100 --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
$P --dataset cifar100 --depths layer4 --n-tasks 20 --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
echo "[ENC] SEED-42 PRIORITY RUNS DONE"
