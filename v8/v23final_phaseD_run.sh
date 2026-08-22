#!/bin/bash
# V23-Final Phase D: MGSC granularity 消融 (AGIQA-3K split5, 50ep)
# global / local / global_local 三并行 (Full, MLPQ on)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] Phase D: global local global_local"
$PY train.py --cfg-path configs/v23_final/global.yaml       --seed 42 --num_cv 1 > run/D_mgsc_global.log 2>&1 &
$PY train.py --cfg-path configs/v23_final/local.yaml        --seed 42 --num_cv 1 > run/D_mgsc_local.log 2>&1 &
$PY train.py --cfg-path configs/v23_final/global_local.yaml --seed 42 --num_cv 1 > run/D_mgsc_global_local.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V23 Final Phase D done"
