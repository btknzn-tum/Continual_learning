#!/bin/bash
# Waits for CLIP sweep to save tuned params, then single-seed quick pass over
# CLIP depths (5-seed --full versions auto-launch from run_parallel.sh gate).
cd /root/Continual_learning
source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
T="crcl/results/tuned_cifar100_clip_vitb32_final.json"
until grep -q "saved .*tuned_cifar100_clip_vitb32_final.json" logs/clipfix.log 2>/dev/null; do sleep 60; done
echo "[CQ] $(date +%T) clip sweep done -> single-seed quick pass"
python crcl/src/run_benchmark.py --dataset cifar100 --backbone clip_vitb32 --depths final "block9+final" --methods mas naive ewc --seeds 42 --tuned-file $T
python crcl/src/run_benchmark.py --dataset cifar10  --backbone clip_vitb32 --depths final "block9+final" --methods mas naive ewc --seeds 42 --tuned-file $T
echo "[CQ] $(date +%T) CLIP QUICK DONE"
