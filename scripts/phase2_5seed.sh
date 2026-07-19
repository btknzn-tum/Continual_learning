#!/bin/bash
# Waits for the single-seed compare chain to finish, then promotes
# SOTA/capacity/LoRA comparisons to the remaining 4 report seeds.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
T="crcl/results/tuned_cifar100_resnet50_layer4.json"
S="123 456 789 1337"
until grep -q "COMPARE ALL DONE" logs/compare.log 2>/dev/null; do sleep 60; done
echo "[P2] $(date +%T) compare done -> 5-seed promotion"
python crcl/src/run_sota.py --dataset cifar10  --backbone resnet50_layer4          --seeds $S
python crcl/src/run_sota.py --dataset cifar10  --backbone "resnet50_layer3+layer4" --seeds $S
python crcl/src/run_sota.py --dataset cifar100 --backbone resnet50_layer4          --seeds $S
python crcl/src/run_sota.py --dataset cifar100 --backbone "resnet50_layer3+layer4" --seeds $S
python crcl/src/run_benchmark.py --dataset cifar10  --backbone resnet50 --depths layer4 --methods mas ewc       --d-hidden 354  --seeds $S --tuned-file $T
python crcl/src/run_benchmark.py --dataset cifar10  --backbone resnet50 --depths layer4 --methods mas ewc naive --d-hidden 1024 --seeds $S --tuned-file $T
python crcl/src/run_benchmark.py --dataset cifar10  --backbone resnet50 --depths layer4 --methods mas ewc naive --arch lora     --seeds $S --tuned-file $T
python crcl/src/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer4 --methods mas ewc       --d-hidden 354  --seeds $S --tuned-file $T
python crcl/src/run_benchmark.py --dataset cifar100 --backbone resnet50 --depths layer4 --methods mas ewc naive --arch lora     --seeds $S --tuned-file $T
echo "[P2] $(date +%T) PHASE2 5-SEED ALL DONE"
