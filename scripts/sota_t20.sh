#!/bin/bash
# SOTA baselines for the 20-task CIFAR-100 stream: seed 42 first, then 4 more.
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python crcl/src/run_sota.py --dataset cifar100 --backbone resnet50_layer4          --n-tasks 20 --seeds 42
python crcl/src/run_sota.py --dataset cifar100 --backbone "resnet50_layer3+layer4" --n-tasks 20 --seeds 42
echo "[T20] single-seed done -> 5-seed"
python crcl/src/run_sota.py --dataset cifar100 --backbone resnet50_layer4          --n-tasks 20 --seeds 123 456 789 1337
python crcl/src/run_sota.py --dataset cifar100 --backbone "resnet50_layer3+layer4" --n-tasks 20 --seeds 123 456 789 1337
echo "[T20] SOTA T20 ALL DONE"
