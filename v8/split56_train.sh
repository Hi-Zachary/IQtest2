#!/bin/bash
# split5/split6 泛化验证: B1/B2/Full 各 50ep
# 先跑 split5 (B1 B2 Full 并行) -> 等 GPU 空闲 -> 再跑 split6 (B1 B2 Full 并行)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

run_split() {
  local s="$1"
  echo "[$(date '+%m-%d %H:%M')] split$s: start B1 B2 Full"
  $PY train.py --cfg-path "configs/b1_dgmpq_s${s}.yaml" --seed 42 --num_cv 1 > "run/B1_s${s}.log" 2>&1 &
  $PY train.py --cfg-path "configs/b2_hcmi_s${s}.yaml" --seed 42 --num_cv 1 > "run/B2_s${s}.log" 2>&1 &
  $PY train.py --cfg-path "configs/full_s${s}.yaml"   --seed 42 --num_cv 1 > "run/Full_s${s}.log" 2>&1 &
  wait
  echo "[$(date '+%m-%d %H:%M')] split$s done"
  while :; do
    np=$(pgrep -f "train.py --cfg-path" | wc -l)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$np" -le 1 ] && [ "${mem:-9999}" -lt 500 ]; then break; fi
    sleep 15
  done
}

run_split 5
run_split 6
echo "[$(date '+%m-%d %H:%M')] ALL split5/split6 done"
