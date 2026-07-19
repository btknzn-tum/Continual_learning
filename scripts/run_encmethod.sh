#!/bin/bash
# OUR METHOD: selective-plasticity encoder + MAS adapter + similarity anchor.
# Priority order: 5-Datasets (beat RanPAC 82.69) -> C100 t10 -> C100 t20.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P="python crcl/src/train_encoder_cl.py"
$P --dataset fivedata --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0 --epochs 10
$P --dataset cifar100 --depths "layer3+layer4" --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0 --epochs 10
$P --dataset cifar100 --depths layer4 --n-tasks 20 --seeds 42 --enc-q 0.05 --sim-lambda 1.0 --alpha 1.0 --epochs 10
echo "[ENC] SEED-42 PRIORITY RUNS DONE"
