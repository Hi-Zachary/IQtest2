#!/bin/bash
# V20 on AIGIQA-20K: E1(HMQE) / E2(DCGA) / Full 三并行 (20 epochs)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigiqa20k v20: start E1 E2 Full"
$PY train.py --cfg-path configs/aigiqa20k/v20/e1.yaml    --seed 42 --num_cv 1 > run/E1_HMQE_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v20/e2.yaml    --seed 42 --num_cv 1 > run/E2_DCGA_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v20/full.yaml  --seed 42 --num_cv 1 > run/Full_V20_20k.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL AIGIQA-20K V20 done"
