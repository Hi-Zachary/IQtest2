#!/bin/bash
# V22 de-LoRA 验证: Frozen CLIP + DG-MPQ + CADR (AGIQA-3K split5, 50ep)
# F0/F1/F2 三并行
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] aigiqa3k frozen: start F0 F1 F2"
$PY train.py --cfg-path configs/v22_frozen/f0.yaml --seed 42 --num_cv 1 > run/F0_frozen.log 2>&1 &
$PY train.py --cfg-path configs/v22_frozen/f1.yaml --seed 42 --num_cv 1 > run/F1_frozendg.log 2>&1 &
$PY train.py --cfg-path configs/v22_frozen/f2.yaml --seed 42 --num_cv 1 > run/F2_frozendgcadr.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL V22 frozen done"
