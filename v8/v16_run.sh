#!/bin/bash
# V16 第一轮: split4, B1 / B2-DAPS / Full 三并行 (matched init, stopgrad), 50ep
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split4: start B1 B2-DAPS Full"
$PY train.py --cfg-path configs/v16/b1_s4.yaml   --seed 42 --num_cv 1 > run/B1_V16_s4.log 2>&1 &
$PY train.py --cfg-path configs/v16/b2_s4.yaml   --seed 42 --num_cv 1 > run/B2_DAPS_V16_s4.log 2>&1 &
$PY train.py --cfg-path configs/v16/full_s4.yaml --seed 42 --num_cv 1 > run/Full_V16_s4.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V16 split4 done"
