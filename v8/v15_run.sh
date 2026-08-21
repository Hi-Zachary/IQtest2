#!/bin/bash
# V15 第一轮: split4, R0 + B2-PTLC (matched init, stopgrad), 50ep
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split4: start R0 B2-PTLC"
$PY train.py --cfg-path configs/r0_v15_s4.yaml    --seed 42 --num_cv 1 > run/R0_V15_s4.log 2>&1 &
$PY train.py --cfg-path configs/b2ptlc_v15_s4.yaml --seed 42 --num_cv 1 > run/B2_PTLC_V15_s4.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V15 split4 done"
