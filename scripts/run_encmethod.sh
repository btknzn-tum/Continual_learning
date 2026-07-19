#!/bin/bash
# OUR METHOD: selective-plasticity encoder + MAS adapter + similarity anchor.
# GPU preprocessing, bf16, channels_last, batch 128.
# Fast first signal: fivedata at 10 epochs (tagged _e10), then full 20-epoch runs.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python crcl/src/train_encoder_cl.py"
$P --dataset fivedata --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0 --epochs 10 --tag-suffix _e10
$P --dataset cifar100 --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
$P --dataset cifar100 --depths layer4 --n-tasks 20 --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
$P --dataset fivedata --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0
echo "[ENC] SEED-42 PRIORITY RUNS DONE"
