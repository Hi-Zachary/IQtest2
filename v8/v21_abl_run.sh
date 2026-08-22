#!/bin/bash
# V21 DG-MPQ 2x2 因子消融 (AGIQA-3K split5, 50ep, doublescore)
# A/B/C 并行，然后 D (Full DG-MPQ)
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

echo "[$(date '+%m-%d %H:%M')] batch1: A B C"
$PY train.py --cfg-path configs/v21_abl/a.yaml --seed 42 --num_cv 1 > run/A_single_nodev.log 2>&1 &
$PY train.py --cfg-path configs/v21_abl/b.yaml --seed 42 --num_cv 1 > run/B_single_dev.log 2>&1 &
$PY train.py --cfg-path configs/v21_abl/c.yaml --seed 42 --num_cv 1 > run/C_multi_nodev.log 2>&1 &
wait
wait_gpu_idle

echo "[$(date '+%m-%d %H:%M')] batch2: D"
$PY train.py --cfg-path configs/v21_abl/d.yaml --seed 42 --num_cv 1 > run/D_full_dgmpq.log 2>&1 &
wait

echo "[$(date '+%m-%d %H:%M')] ALL V21 ablation done"
