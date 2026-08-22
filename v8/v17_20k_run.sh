#!/bin/bash
# V17 on AIGIQA-20K: R0 / B1(DG-MPQ) / Full(DG-MPQ+QARD) 三并行 (single MOS, 50ep)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigiqa20k: start R0 B1 Full"
$PY train.py --cfg-path configs/aigiqa20k/r0_v17.yaml            --seed 42 --num_cv 1 > run/R0_V17_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/b1_dgmpq_v17.yaml     --seed 42 --num_cv 1 > run/B1_V17_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/full_dgmpq_qard_v17.yaml --seed 42 --num_cv 1 > run/Full_V17_20k.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL AIGIQA-20K done"
