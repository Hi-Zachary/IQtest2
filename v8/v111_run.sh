#!/bin/bash
# V11.1 matched-initialization 控制: split4, B1/B2/Full (dropout=0, 同 init ckpt)
# 显存 ~9GB/变体(双视觉前向) -> B1+B2 并行, 然后 Full
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

echo "[$(date '+%m-%d %H:%M')] batch1: B1 B2 (matched init)"
$PY train.py --cfg-path configs/b1_v111_s4.yaml --seed 42 --num_cv 1 > run/B1_V111_s4.log 2>&1 &
$PY train.py --cfg-path configs/b2_v111_s4.yaml --seed 42 --num_cv 1 > run/B2_V111_s4.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: Full"
$PY train.py --cfg-path configs/full_v111_s4.yaml --seed 42 --num_cv 1 > run/Full_V111_s4.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V11.1 split4 done"
