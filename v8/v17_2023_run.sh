#!/bin/bash
# V17 在 AIGCIQA-2023 上的 B1/B2-QARD/Full 三并行 (matched init, stopgrad), 50ep
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigciqa2023: start B1 B2-QARD Full"
$PY train.py --cfg-path configs/aigciqa2023/b1_v17.yaml   --seed 42 --num_cv 1 > run/B1_V17_2023.log 2>&1 &
$PY train.py --cfg-path configs/aigciqa2023/b2_v17.yaml   --seed 42 --num_cv 1 > run/B2_QARD_V17_2023.log 2>&1 &
$PY train.py --cfg-path configs/aigciqa2023/full_v17.yaml --seed 42 --num_cv 1 > run/Full_V17_2023.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL AIGCIQA2023 done"
