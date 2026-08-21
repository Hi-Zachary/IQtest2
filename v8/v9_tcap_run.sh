#!/bin/bash
# V9 第一阶段: split4, B1 / B2-TCAP / Full-TCAP 三并行 (全部 v9 框架, 50ep)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split4: start B1 B2-TCAP Full-TCAP"
$PY train.py --cfg-path "configs/b1_tcap_s4.yaml"    --seed 42 --num_cv 1 > "run/B1_TCAP_s4.log" 2>&1 &
$PY train.py --cfg-path "configs/b2_tcap_s4.yaml"    --seed 42 --num_cv 1 > "run/B2_TCAP_s4.log" 2>&1 &
$PY train.py --cfg-path "configs/full_tcap_s4.yaml"  --seed 42 --num_cv 1 > "run/Full_TCAP_s4.log" 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V9 split4 done"
