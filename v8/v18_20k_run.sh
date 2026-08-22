#!/bin/bash
# V18 on AIGIQA-20K: B1 / B2-CADR / Full 三并行 (single MOS, 30ep)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigiqa20k v18: start B1 B2-CADR Full"
$PY train.py --cfg-path configs/aigiqa20k/v18/b1_dgmpq.yaml        --seed 42 --num_cv 1 > run/B1_V18_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v18/b2_cadr.yaml         --seed 42 --num_cv 1 > run/B2_CADR_V18_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v18/full_dgmpq_cadr.yaml --seed 42 --num_cv 1 > run/Full_V18_20k.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL AIGIQA-20K V18 done"
