#!/bin/bash
# Confirm the fivedata mid-level-tap win at 5 seeds + test on MAS adapter.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES="0"
T="crcl/results/tuned_cifar100_resnet50_layer4.json"
# 5-seed RanPAC on the top-2 fivedata configs
for TAPS in "layer2+layer3" "layer2+layer3+layer4"; do
  python crcl/src/run_sota.py --dataset fivedata --backbone "resnet50_${TAPS}" --methods ranpac --seeds 123 456 789 1337
done
# does the placement shift help the MAS adapter too? (seed 42 first)
python crcl/src/run_benchmark.py --dataset fivedata --backbone resnet50 --depths "layer2+layer3" "layer2+layer3+layer4" --methods mas ewc naive --seeds 42 --tuned-file $T
echo "[TAPCONFIRM] DONE"
