#!/bin/bash
# 一键串行跑 v8 主消融 (AGIQA-3K, split3 seed42, 30ep)
# 调整/改进1.md 第 10/17/11 节
# 用法: bash serial_train.sh [B0|R0|B1|B2|Full ...]   默认全部

PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
RUNS=${@:-B0 R0 B1 B2 Full}

for v in $RUNS; do
  case $v in
    B0)  cfg=configs/b0_frozen.yaml ;;
    R0)  cfg=configs/r0_lora.yaml ;;
    B1)  cfg=configs/b1_dgmpq.yaml ;;
    B2)  cfg=configs/b2_dphcmi.yaml ;;
    Full) cfg=configs/full.yaml ;;
    *) echo "unknown variant: $v"; exit 1 ;;
  esac
  echo "=============================== $v ($cfg) ==============================="
  $PY train.py --cfg-path "$cfg" --seed 42 --num_cv 1
done
