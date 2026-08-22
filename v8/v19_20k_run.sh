#!/bin/bash
# V19 on AIGIQA-20K: E1(DG-MPQ+) / E2(DG-MPQ+DCAR) / Full(DG-MPQ++DCAR) 三并行
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigiqa20k v19: start E1 E2 Full"
$PY train.py --cfg-path configs/aigiqa20k/v19/e1.yaml    --seed 42 --num_cv 1 > run/E1_DGMPQplus_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v19/e2.yaml    --seed 42 --num_cv 1 > run/E2_DGDCAR_20k.log 2>&1 &
$PY train.py --cfg-path configs/aigiqa20k/v19/full.yaml  --seed 42 --num_cv 1 > run/Full_V19_20k.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL AIGIQA-20K V19 done"
