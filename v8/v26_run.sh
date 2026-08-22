#!/bin/bash
# V26 LACE-IQA AIGIQA-20K 单 Overall MOS 消融 (20ep, no-stop)
# B0/B1/B2 并行，然后 B3
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

wait_gpu_idle() {
  echo "[$(date '+%m-%d %H:%M')] waiting GPU idle..."
  while :; do
    np=$(pgrep -f "train.py --cfg-path" | wc -l)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$np" -le 1 ] && [ "${mem:-9999}" -lt 500 ]; then echo "GPU idle"; return 0; fi
    sleep 15
  done
}

echo "[$(date '+%m-%d %H:%M')] batch1: B0 B1 B2"
$PY train.py --cfg-path configs/v26_aigiqa20k/b0.yaml --seed 42 --num_cv 1 > run/B0_baseline.log 2>&1 &
$PY train.py --cfg-path configs/v26_aigiqa20k/b1.yaml --seed 42 --num_cv 1 > run/B1_lqea.log 2>&1 &
$PY train.py --cfg-path configs/v26_aigiqa20k/b2.yaml --seed 42 --num_cv 1 > run/B2_csae.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: B3"
$PY train.py --cfg-path configs/v26_aigiqa20k/b3.yaml --seed 42 --num_cv 1 > run/B3_full.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V26 done"
