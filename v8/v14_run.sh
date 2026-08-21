#!/bin/bash
# V14: split4, R0/B1/B2-MSRC/Full-MSRC (matched init, stopgrad), 50ep
# 单视觉前向 ~6GB -> 3+1: R0 B1 B2 并行, 然后 Full
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

wait_gpu_idle() {
  echo "[$(date '+%m-%d %H:%M')] waiting for GPU idle ..."
  while :; do
    np=$(pgrep -f "train.py --cfg-path" | wc -l)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$np" -le 1 ] && [ "${mem:-9999}" -lt 500 ]; then echo "GPU idle"; return 0; fi
    sleep 15
  done
}

echo "[$(date '+%m-%d %H:%M')] batch1: R0 B1 B2-MSRC"
$PY train.py --cfg-path configs/r0_v14_s4.yaml     --seed 42 --num_cv 1 > run/R0_V14_s4.log 2>&1 &
$PY train.py --cfg-path configs/b1_v14_s4.yaml     --seed 42 --num_cv 1 > run/B1_V14_s4.log 2>&1 &
$PY train.py --cfg-path configs/b2msrc_v14_s4.yaml --seed 42 --num_cv 1 > run/B2_MSRC_V14_s4.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: Full-MSRC"
$PY train.py --cfg-path configs/fullmsrc_v14_s4.yaml --seed 42 --num_cv 1 > run/Full_MSRC_V14_s4.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V14 split4 done"
