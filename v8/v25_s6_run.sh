#!/bin/bash
# V25 Joint-LoRA split6 复核: Reference (stopgrad=True) vs Joint-LoRA (stopgrad=False)
PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
mkdir -p run

echo "[$(date '+%m-%d %H:%M')] split6: Ref vs Joint-LoRA"
$PY train.py --cfg-path configs/v23_final/global_local_s6.yaml --seed 42 --num_cv 1 > run/Ref_GL_s6.log 2>&1 &
$PY train.py --cfg-path configs/v25_joint_lora/e3_s6.yaml    --seed 42 --num_cv 1 > run/Joint_LoRA_s6.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] ALL split6 done"
