#!/bin/bash
# 一键串行跑 AGIQA-3K B0-B4（统一协议：seed42 固定划分 / 512 / batch32 / 100 epoch / AMP）
set -e
cd "$(dirname "$0")"

PY=${PY:-/root/autodl-tmp/CondaEnv/ipiqa/bin/python}

run_task() {
    echo "=================================================="
    echo "[$(date '+%m-%d %H:%M')] $1"
    echo "=================================================="
    $PY train.py --cfg-path "$2" --seed 42 --num_cv 1 || exit 1
}

run_task "B0 baseline"              "projects/agiqa3k/b0_baseline.yaml"
run_task "B1 +MSQR"                 "projects/agiqa3k/b1_msqr.yaml"
run_task "B2 +SHCMI"                "projects/agiqa3k/b2_shcmi.yaml"
run_task "B3 +MSQR+SHCMI"           "projects/agiqa3k/b3_msqr_shcmi.yaml"
run_task "B4 Full (all)"            "projects/agiqa3k/b4_full.yaml"

echo "[$(date '+%m-%d %H:%M')] All AGIQA-3K ablations done."
