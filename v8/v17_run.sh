#!/bin/bash
# V17 第一轮: split5, B1 / B2-QARD / Full 三并行 (matched init, stopgrad), 50ep
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split5: start B1 B2-QARD Full"
$PY train.py --cfg-path configs/v17/b1_s5.yaml   --seed 42 --num_cv 1 > run/B1_V17_s5.log 2>&1 &
$PY train.py --cfg-path configs/v17/b2_s5.yaml   --seed 42 --num_cv 1 > run/B2_QARD_V17_s5.log 2>&1 &
$PY train.py --cfg-path configs/v17/full_s5.yaml --seed 42 --num_cv 1 > run/Full_V17_s5.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V17 split5 done"
