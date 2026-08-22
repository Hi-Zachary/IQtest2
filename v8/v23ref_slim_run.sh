#!/bin/bash
# V23-Refined Phase A: MGSC Slim 验证 (AGIQA-3K split5/6, 50ep)
# split5 E2/E3 + split6 E2/E3 四并行
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] start slim E2/E3 split5/6"
$PY train.py --cfg-path configs/v23_refined/e2.yaml    --seed 42 --num_cv 1 > run/E2_mgsc_slim.log 2>&1 &
$PY train.py --cfg-path configs/v23_refined/e3.yaml    --seed 42 --num_cv 1 > run/E3_full_slim.log 2>&1 &
$PY train.py --cfg-path configs/v23_refined_s6/e2.yaml --seed 42 --num_cv 1 > run/E2_mgsc_slim_s6.log 2>&1 &
$PY train.py --cfg-path configs/v23_refined_s6/e3.yaml --seed 42 --num_cv 1 > run/E3_full_slim_s6.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V23Refined Slim done"
