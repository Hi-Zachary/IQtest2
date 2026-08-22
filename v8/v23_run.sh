#!/bin/bash
# V23 MLPQ+MGSC 双分支 (AGIQA-3K split5, 50ep, doublescore)
# E1/E2/E3 并行，然后 E0 (LoRA baseline)
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

echo "[$(date '+%m-%d %H:%M')] batch1: E1 E2 E3"
$PY train.py --cfg-path configs/v23/e1.yaml --seed 42 --num_cv 1 > run/E1_mlpq.log 2>&1 &
$PY train.py --cfg-path configs/v23/e2.yaml --seed 42 --num_cv 1 > run/E2_mgsc.log 2>&1 &
$PY train.py --cfg-path configs/v23/e3.yaml --seed 42 --num_cv 1 > run/E3_full.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: E0"
$PY train.py --cfg-path configs/v23/e0.yaml --seed 42 --num_cv 1 > run/E0_lora_base.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V23 done"
