#!/bin/bash
# V10 第一阶段: split4, B1 / B2-MSCM / Full-MSCM 三并行 (全部 v10 框架, 50ep)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split4: start B1 B2-MSCM Full-MSCM"
$PY train.py --cfg-path "configs/b1_mscm_s4.yaml"    --seed 42 --num_cv 1 > "run/B1_MSCM_s4.log" 2>&1 &
$PY train.py --cfg-path "configs/b2_mscm_s4.yaml"    --seed 42 --num_cv 1 > "run/B2_MSCM_s4.log" 2>&1 &
$PY train.py --cfg-path "configs/full_mscm_s4.yaml"  --seed 42 --num_cv 1 > "run/Full_MSCM_s4.log" 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V10 split4 done"
