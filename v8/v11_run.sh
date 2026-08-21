#!/bin/bash
# V11 第一阶段: split4, R0/B1/B2/Full 四变体 (全 v11 框架, 50ep)
# 显存: Full-V11 双视觉前向 ~9.4GB -> 2+2 并行 (R0 B1 -> B2 Full)
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

echo "[$(date '+%m-%d %H:%M')] batch1: R0 B1"
$PY train.py --cfg-path configs/r0_v11_s4.yaml --seed 42 --num_cv 1 > run/R0_V11_s4.log 2>&1 &
$PY train.py --cfg-path configs/b1_v11_s4.yaml --seed 42 --num_cv 1 > run/B1_V11_s4.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: B2 Full"
$PY train.py --cfg-path configs/b2_v11_s4.yaml --seed 42 --num_cv 1 > run/B2_V11_s4.log 2>&1 &
$PY train.py --cfg-path configs/full_v11_s4.yaml --seed 42 --num_cv 1 > run/Full_V11_s4.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V11 split4 done"
