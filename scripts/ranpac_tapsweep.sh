#!/bin/bash
# Can lower-level taps beat layer3+4 (87.28) for RanPAC on the heterogeneous
# 5-dataset stream? MNIST/SVHN/KMNIST are far from ImageNet -> may need
# low-level features. Seed 42 quick look; promote winners to 5-seed if >87.28.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES="0"
for TAPS in "layer2+layer3+layer4" "layer1+layer2+layer3+layer4" "layer2+layer4" "layer1+layer4" "layer2+layer3" "layer1+layer2"; do
  python crcl/src/run_sota.py --dataset fivedata --backbone "resnet50_${TAPS}" --methods ranpac --seeds 42
done
echo "[TAPSWEEP] DONE"
